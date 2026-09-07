"""Train and evaluate B4: static edge-level GraphSAGE baseline (no energy).

Edge labels are DERIVED, not observed - see edge_labels.py and the Step 8
report for why (no dataset in this repository ships a real per-edge
anomaly label). Because a derived label is a deterministic function of its
two endpoint node labels, this script also runs the required leakage
diagnostic: a stratified breakdown by endpoint-label configuration, and a
"trivial reference" score built purely from an already-trained B0 node
model's own predictions, with no tuning of the learned edge classifier
against it.

Reuses, unmodified: b0_train.load_static_dataset / split_names (same
dataset loaders, same splits, same preprocessing as Steps 3-7),
E_train.compute_comprehensive_metrics / get_best_f1 (same metric
implementation), and the same optimizer/scheduler/seeding/loss-weighting
pattern as b0_train.py / b1_train.py / b2_train.py / b3_train.py.
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from b0_train import load_static_dataset, split_names
from b4_model import B4EdgeAnomalySAGE
from E_train import compute_comprehensive_metrics, get_best_f1
from edge_labels import (
    STRATUM_NAMES,
    assign_edge_splits,
    build_supervised_edge_index,
    derive_edge_labels,
    endpoint_stratum,
)


# An edge subset (especially a single endpoint-configuration stratum) can
# legitimately contain no positive predictions or no positive ground truth.
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def load_node_scores(path, num_nodes):
    """Load a previously saved B0 per-node anomaly_score column, ordered by node_id."""
    if not path.exists():
        raise FileNotFoundError(
            f"Trivial-reference node scores not found at {path}. "
            "Run b0_train.py for this dataset/seed first "
            "(same --dataset/--seed as this B4 run), or pass --node_scores_path."
        )
    frame = pd.read_csv(path).sort_values("node_id")
    if len(frame) != num_nodes or frame["node_id"].max() != num_nodes - 1:
        raise ValueError(
            f"{path} does not have exactly one row per node (expected {num_nodes})."
        )
    return torch.tensor(frame["anomaly_score"].to_numpy(), dtype=torch.float32)


def stratum_metrics(labels_np, scores_np, preds_np):
    """Metrics for one endpoint-label stratum. AUROC/AUPRC are None when undefined.

    precision/recall/f1 here are BINARY (positive class = anomalous edge),
    not the macro-F1 used for the aggregate Step-7-style result: macro-F1
    is not meaningful on a stratum that is single-class by construction
    (both_normal is always negative, both_anomalous is always positive).
    The "overall_test" row is computed separately, directly from
    best_test_metrics, so it stays comparable to the aggregate CSV and to
    the trivial reference row (both macro-F1-based).
    """
    n = int(len(labels_np))
    result = {"n_edges": n}
    if n == 0:
        result.update(
            accuracy=None, precision=None, recall=None, f1=None,
            auroc=None, auprc=None,
            note="empty stratum",
        )
        return result

    result["accuracy"] = float((preds_np == labels_np).mean())
    result["precision"] = float(precision_score(labels_np, preds_np, pos_label=1, zero_division=0))
    result["recall"] = float(recall_score(labels_np, preds_np, pos_label=1, zero_division=0))
    result["f1"] = float(f1_score(labels_np, preds_np, pos_label=1, zero_division=0))

    if len(np.unique(labels_np)) < 2:
        result["auroc"] = None
        result["auprc"] = None
        result["note"] = (
            "AUROC/AUPRC undefined: by construction this stratum has a single "
            "ground-truth class (both_normal is always negative, both_anomalous "
            "is always positive)."
        )
    else:
        result["auroc"] = float(roc_auc_score(labels_np, scores_np))
        result["auprc"] = float(average_precision_score(labels_np, scores_np))
        result["note"] = None
    return result


def compute_trivial_reference(node_scores, edge_pairs, hard_label, edge_val_mask, edge_test_mask):
    """Score each edge as avg(node_score_u, node_score_v) from an already-trained node model.

    Threshold is tuned on validation edges (mirroring how the learned
    model's own threshold is chosen) and applied to test edges. Nothing
    here is fit or tuned against the learned B4 edge classifier's results.
    """
    u, v = edge_pairs[0].cpu(), edge_pairs[1].cpu()
    edge_score = (node_scores[u] + node_scores[v]) / 2.0
    edge_score_np = edge_score.numpy()
    hard_label_np = hard_label.cpu().numpy()

    val_idx = edge_val_mask.cpu().numpy()
    test_idx = edge_test_mask.cpu().numpy()

    val_probs_2col = np.stack(
        [1.0 - edge_score_np[val_idx], edge_score_np[val_idx]], axis=1
    )
    _, val_threshold = get_best_f1(hard_label_np[val_idx], val_probs_2col)

    test_scores = edge_score_np[test_idx]
    test_labels = hard_label_np[test_idx]
    test_preds = (test_scores > val_threshold).astype(np.int64)

    metrics = compute_comprehensive_metrics(test_labels, test_scores, test_preds)
    metrics["threshold"] = val_threshold
    return metrics, edge_score_np


def save_edge_outputs(output_dir, dataset, seed, edge_pairs, split_names_arr, stratum, soft_label, hard_label, anomaly_scores, threshold):
    """Save per-edge scores, mirroring the node-level save_node_scores schema."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"b4_{dataset}_seed{seed}"
    scores_np = anomaly_scores.cpu().numpy()
    stratum_np = stratum.cpu().numpy()

    frame = pd.DataFrame(
        {
            "edge_id": np.arange(edge_pairs.shape[1]),
            "u": edge_pairs[0].cpu().numpy(),
            "v": edge_pairs[1].cpu().numpy(),
            "split": split_names_arr,
            "endpoint_stratum": [STRATUM_NAMES[s] for s in stratum_np],
            "soft_label": soft_label.cpu().numpy(),
            "hard_label": hard_label.cpu().numpy(),
            "anomaly_score": scores_np,
            "predicted_label": (scores_np > threshold).astype(np.int64),
        }
    )
    path = output_dir / f"{stem}_edge_scores.csv"
    frame.to_csv(path, index=False)
    print(f"Saved B4 edge anomaly scores: {path}")
    return path


def train_b4(args):
    """Train B4 with the same data, split logic, loss, and evaluation family as B0-B3."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("B4 - EDGE-LEVEL GRAPHSAGE, DERIVED (UNIGAD-STYLE) LABELS")
    print("=" * 70)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed}")
    print(f"Edge label: P_anom(u,v) = avg(label_u, label_v); hard label = (P_anom > 0)")
    print("Input: raw node features only (no spectral energy)")

    data = load_static_dataset(args)
    graph = data.graph.to(device)
    features = data.features.float().to(device)
    labels = data.labels.long().to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)

    # --- Build the canonical supervised edge set -------------------------------
    raw_src, raw_dst = graph.edges()
    num_raw_edges = raw_src.numel()
    num_self_loops = int((raw_src == raw_dst).sum())

    edge_pairs = build_supervised_edge_index(graph)
    num_supervised_edges = edge_pairs.shape[1]
    print("\nEdge set construction:")
    print(f"  Raw directed edges (incl. self-loops): {num_raw_edges:,}")
    print(f"  Self-loops removed:                    {num_self_loops:,}")
    print(f"  Unique undirected supervised edges:    {num_supervised_edges:,}")

    soft_label, hard_label = derive_edge_labels(labels, edge_pairs)
    stratum = endpoint_stratum(labels, edge_pairs)
    splits = assign_edge_splits(edge_pairs, train_mask, val_mask, test_mask)

    n_train = int(splits.train.sum())
    n_val = int(splits.val.sum())
    n_test = int(splits.test.sum())
    n_excluded = int(splits.excluded.sum())
    print("\nEdge splits (by strict endpoint-mask membership; mixed-membership edges excluded):")
    print(f"  Train: {n_train:,}")
    print(f"  Val:   {n_val:,}")
    print(f"  Test:  {n_test:,}")
    print(f"  Excluded (endpoints straddle splits, or touch an unlabeled node): {n_excluded:,}")
    print(
        "  NOTE: this rules out any test edge whose both endpoints are train "
        "nodes. It does NOT remove transductive leakage through message "
        "passing: the encoder still runs SAGEConv over the full graph on "
        "every forward pass, exactly as B0-B3 do for node classification, "
        "so test-node embeddings are computed using aggregated FEATURES "
        "(never labels) from train-node neighbors."
    )

    anomaly_count = int(hard_label[splits.train].sum())
    if anomaly_count == 0:
        raise RuntimeError("The training edge split contains no anomalous edges.")
    class_weight_value = (
        (hard_label[splits.train] == 0).sum().item() / anomaly_count
    )
    class_weight = torch.tensor([1.0, class_weight_value], device=device)
    print(f"\nClass weight (anomaly edge), recomputed for edges: {class_weight_value:.2f}")

    model = B4EdgeAnomalySAGE(
        in_feats=features.shape[1],
        hidden_dim=args.hidden_dim,
        num_classes=data.num_classes,
        dropout=args.dropout,
        aggregator_type="mean",
    ).to(device)
    print(f"B4 trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=20, min_lr=1e-6
    )

    best_val_f1 = 0.0
    best_model_state = None
    best_test_metrics = None
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        edge_logits = model(graph, features, edge_pairs)
        loss = F.cross_entropy(
            edge_logits[splits.train], hard_label[splits.train], weight=class_weight
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            edge_logits = model(graph, features, edge_pairs)
            probabilities = edge_logits.softmax(dim=1)
            val_f1, val_threshold = get_best_f1(
                hard_label[splits.val].cpu().numpy(),
                probabilities[splits.val].cpu().numpy(),
            )
            predictions = (
                probabilities[:, 1].cpu().numpy() > val_threshold
            ).astype(np.int64)
            test_mask_np = splits.test.cpu().numpy()
            test_labels_np = hard_label.cpu().numpy()[test_mask_np]
            test_preds_np = predictions[test_mask_np]
            test_scores_np = probabilities[:, 1].cpu().numpy()[test_mask_np]
            test_metrics = compute_comprehensive_metrics(
                test_labels_np, test_scores_np, test_preds_np
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                best_test_metrics = {
                    **test_metrics,
                    "threshold": val_threshold,
                    "epoch": epoch + 1,
                }
            scheduler.step(val_f1)

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print(
                f"Epoch {epoch+1:3d}/{args.epochs} | "
                f"Loss: {loss.item():.4f} | "
                f"Val F1: {val_f1:.4f} (best: {best_val_f1:.4f}) | "
                f"Test F1: {test_metrics['macro_f1']:.4f} | "
                f"AUROC: {test_metrics['auroc']:.4f} | "
                f"AUPRC: {test_metrics['auprc']:.4f}"
            )

    if best_model_state is None or best_test_metrics is None:
        raise RuntimeError("No best B4 model was selected.")

    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        best_logits = model(graph, features, edge_pairs)
        best_probabilities = best_logits.softmax(dim=1)
    best_threshold = best_test_metrics["threshold"]

    split_names_arr = split_names(splits.train, splits.val, splits.test)
    score_path = save_edge_outputs(
        Path(args.output_dir), args.dataset, args.seed, edge_pairs,
        split_names_arr, stratum, soft_label, hard_label,
        best_probabilities[:, 1], best_threshold,
    )

    # --- Leakage diagnostic: stratified breakdown on TEST edges -----------------
    test_idx_np = splits.test.cpu().numpy()
    test_stratum_np = stratum.cpu().numpy()[test_idx_np]
    test_hard_np = hard_label.cpu().numpy()[test_idx_np]
    test_score_np = best_probabilities[:, 1].cpu().numpy()[test_idx_np]
    test_pred_np = (test_score_np > best_threshold).astype(np.int64)

    diagnostic_rows = []
    # "overall_test" reuses best_test_metrics (macro-F1, same definition as
    # the aggregate Step-7-style row) rather than stratum_metrics' binary
    # framing, so it stays directly comparable to the trivial-reference row
    # below and to the returned aggregate result.
    overall = {
        "group": "overall_test",
        "n_edges": n_test,
        "accuracy": float((test_pred_np == test_hard_np).mean()),
        "precision": best_test_metrics["precision"],
        "recall": best_test_metrics["recall"],
        "f1": best_test_metrics["macro_f1"],
        "auroc": best_test_metrics["auroc"],
        "auprc": best_test_metrics["auprc"],
        "note": "f1 is macro-F1, not the binary positive-class F1 used in the "
        "endpoint-stratum rows below.",
    }
    diagnostic_rows.append(overall)
    for stratum_id, stratum_name in STRATUM_NAMES.items():
        sub = test_stratum_np == stratum_id
        row = stratum_metrics(test_hard_np[sub], test_score_np[sub], test_pred_np[sub])
        row["group"] = stratum_name
        diagnostic_rows.append(row)

    # --- Trivial reference: avg of B0's own node predictions ---------------------
    node_scores_path = (
        Path(args.node_scores_path)
        if args.node_scores_path
        else Path(args.node_scores_dir) / f"b0_{args.dataset}_seed{args.seed}_node_scores.csv"
    )
    node_scores = load_node_scores(node_scores_path, labels.numel())
    reference_metrics, _ = compute_trivial_reference(
        node_scores, edge_pairs, hard_label, splits.val, splits.test
    )
    reference_row = {
        "group": "trivial_reference_avg_b0_node_score",
        "n_edges": n_test,
        "accuracy": None,
        "precision": reference_metrics["precision"],
        "recall": reference_metrics["recall"],
        "f1": reference_metrics["macro_f1"],
        "auroc": reference_metrics["auroc"],
        "auprc": reference_metrics["auprc"],
        "note": f"threshold tuned on val edges = {reference_metrics['threshold']:.4f}; source: {node_scores_path}",
    }
    diagnostic_rows.append(reference_row)

    diagnostic_frame = pd.DataFrame(diagnostic_rows)
    diagnostic_path = Path(args.output_dir) / f"b4_{args.dataset}_seed{args.seed}_leakage_diagnostic.csv"
    diagnostic_frame.to_csv(diagnostic_path, index=False)
    print(f"Saved B4 leakage diagnostic: {diagnostic_path}")

    print("\nLearned edge classifier vs. trivial (avg node-score) reference on test edges:")
    print("(Same metric implementation on both sides: E_train.compute_comprehensive_metrics.")
    print(" No tuning of the learned classifier against this reference was performed.)")
    for metric_name in ("f1", "auroc", "auprc"):
        learned_value = overall[metric_name]
        reference_value = reference_row[metric_name]
        if learned_value is None or reference_value is None:
            print(f"  {metric_name.upper():6s}: learned={learned_value}  reference={reference_value}  (n/a)")
            continue
        beats = learned_value > reference_value
        print(
            f"  {metric_name.upper():6s}: learned={learned_value:.4f}  "
            f"reference={reference_value:.4f}  learned beats reference: {beats}"
        )

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("B4 FINAL RESULTS (overall test edges)")
    print("=" * 70)
    print(f"Training time: {elapsed:.2f}s")
    print(f"Best epoch: {best_test_metrics['epoch']}")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Validation threshold: {best_threshold:.4f}")
    print(f"Test Recall:    {best_test_metrics['recall']*100:.2f}%")
    print(f"Test Precision: {best_test_metrics['precision']*100:.2f}%")
    print(f"Test Macro F1:  {best_test_metrics['macro_f1']*100:.2f}%")
    print(f"Test AUROC:     {best_test_metrics['auroc']*100:.2f}%")
    print(f"Test AUPRC:     {best_test_metrics['auprc']*100:.2f}%")
    print(f"Test RecK:      {best_test_metrics['reck']*100:.2f}%")
    print(f"Edge scores: {score_path}")
    print(f"Leakage diagnostic: {diagnostic_path}")

    return {
        "model": "B4",
        "dataset": args.dataset,
        "seed": args.seed,
        "level": "edge",
        "AUROC": best_test_metrics["auroc"],
        "AUPRC": best_test_metrics["auprc"],
        "F1": best_test_metrics["macro_f1"],
        "training_time": elapsed,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="B4: edge-level GraphSAGE with UniGAD-style derived labels"
    )
    parser.add_argument(
        "--dataset",
        default="amazon",
        choices=["amazon", "yelp", "tfinance", "tsocial"],
    )
    parser.add_argument("--train_ratio", type=float, default=0.4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--undirected", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--output_dir", default="results/b4")
    parser.add_argument(
        "--node_scores_dir",
        default="results/b0",
        help="Directory containing a previously saved b0_{dataset}_seed{seed}_node_scores.csv "
        "for the trivial-reference diagnostic.",
    )
    parser.add_argument(
        "--node_scores_path",
        default=None,
        help="Explicit path to the B0 node-score CSV, overriding --node_scores_dir.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train_b4(parse_args())
