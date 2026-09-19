"""Train and evaluate B5: edge-level GraphSAGE with gated, edge-centered
bidirectional spectral energy (Step 9).

Reuses, unmodified: b0_train.load_static_dataset / split_names (same
dataset loaders, splits, preprocessing as Steps 3-8), E_train.
compute_comprehensive_metrics / get_best_f1 (same metric implementation),
edge_labels.* (same derived-label formula, edge-set construction, and
split logic as Step 8), and b4_train.stratum_metrics /
compute_trivial_reference / load_node_scores (same leakage diagnostic and
trivial reference as Step 8 - not reimplemented here).

The only new pieces are: edge_energy.py (D1/D2/D3 edge-centered energy)
and b5_model.py (the gated fusion model).
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

from b0_train import load_static_dataset, split_names
from b4_train import compute_trivial_reference, load_node_scores, stratum_metrics
from b5_model import B5EdgeBidirectionalEnergySAGE
from E_train import compute_comprehensive_metrics, get_best_f1
from edge_energy import compute_edge_energy
from edge_labels import (
    EdgeSplit,
    STRATUM_NAMES,
    assign_edge_splits,
    build_supervised_edge_index,
    derive_edge_labels,
    endpoint_stratum,
)

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


@torch.no_grad()
def prepare_edge_energy(mode, features, edge_pairs, graph, edge_train_mask):
    """Compute E_R(e)/E_L(e) for the chosen mode, then normalize with
    TRAIN-EDGE statistics, mirroring b1_train.py/b3_train.py's node-level
    convention: both branches normalized using E_R's own train stats.
    """
    node_energy_right = None
    if mode == "d1":
        from fast_e import local_1hop_energy_lnorm

        src, dst = graph.edges()
        node_edge_index = torch.stack([dst, src], dim=0)
        node_energy_right = local_1hop_energy_lnorm(
            X=features, edge_index=node_edge_index, edge_weight=None,
            eps=1e-8, deg_eps=1e-12,
        )

    energy_right_raw, energy_left_raw = compute_edge_energy(
        mode, features, edge_pairs, graph, node_energy_right=node_energy_right,
    )

    right_mean = energy_right_raw[edge_train_mask].mean(dim=0, keepdim=True)
    right_std = energy_right_raw[edge_train_mask].std(dim=0, keepdim=True).clamp_min(1e-8)
    energy_right = (energy_right_raw - right_mean) / right_std
    energy_left = (energy_left_raw - right_mean) / right_std
    return energy_right, energy_left


def train_b5(args):
    """Train B5 with the same data, split logic, loss, and evaluation family as Step 8."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("B5 - EDGE-LEVEL GRAPHSAGE, GATED BIDIRECTIONAL EDGE ENERGY")
    print("=" * 70)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed}")
    print(f"Energy mode: {args.energy_mode}")
    print(f"Fusion: {args.fusion}")

    data = load_static_dataset(args)
    graph = data.graph.to(device)
    features = data.features.float().to(device)
    labels = data.labels.long().to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)

    edge_pairs = build_supervised_edge_index(graph)
    num_supervised_edges = edge_pairs.shape[1]
    print(f"Unique undirected supervised edges: {num_supervised_edges:,}")

    soft_label, hard_label = derive_edge_labels(labels, edge_pairs)
    stratum = endpoint_stratum(labels, edge_pairs)
    splits = assign_edge_splits(edge_pairs, train_mask, val_mask, test_mask)

    n_train = int(splits.train.sum())
    n_val = int(splits.val.sum())
    n_test = int(splits.test.sum())
    n_excluded = int(splits.excluded.sum())
    print(
        f"Edge splits: train={n_train:,} val={n_val:,} test={n_test:,} "
        f"excluded={n_excluded:,}"
    )

    # Drop mixed-split-membership edges up front. They are never read in the
    # loss, evaluation, or leakage diagnostic below, but were still being
    # pushed through every forward pass (energy computation, gate,
    # projections, edge_head) for all `num_supervised_edges`, not just the
    # train+val+test subset actually used - on Amazon this exclusion rate is
    # ~73%, and computing the full set every epoch is what exhausts a 14.56
    # GiB T4 at the edge_head's hidden-layer activation. Filtering here cuts
    # every downstream tensor (energy, gate, projections, concat, edge_head,
    # and D3's per-edge Python loop) by the same fraction, with no change to
    # which edges are used for train/val/test.
    keep = ~splits.excluded
    edge_pairs = edge_pairs[:, keep]
    soft_label = soft_label[keep]
    hard_label = hard_label[keep]
    stratum = stratum[keep]
    splits = EdgeSplit(
        train=splits.train[keep],
        val=splits.val[keep],
        test=splits.test[keep],
        excluded=torch.zeros(edge_pairs.shape[1], dtype=torch.bool, device=edge_pairs.device),
    )
    print(f"Edges kept for training/eval (train+val+test only): {edge_pairs.shape[1]:,}")

    if args.energy_mode == "d3" and args.d3_edge_limit >= 0 and edge_pairs.shape[1] > args.d3_edge_limit:
        raise RuntimeError(
            f"D3 requested on {edge_pairs.shape[1]:,} kept edges (train+val+test, "
            "after dropping mixed-split-membership edges), exceeding "
            f"--d3_edge_limit={args.d3_edge_limit:,}. D3 loops in Python over "
            "every kept edge calling dgl.khop_in_subgraph, and has not "
            "been benchmarked at this scale. Raise --d3_edge_limit explicitly "
            "(or pass -1 to disable this guard) once you have confirmed the "
            "runtime is acceptable, or test on a smaller dataset/subset first."
        )

    anomaly_count = int(hard_label[splits.train].sum())
    if anomaly_count == 0:
        raise RuntimeError("The training edge split contains no anomalous edges.")
    class_weight_value = (
        (hard_label[splits.train] == 0).sum().item() / anomaly_count
    )
    class_weight = torch.tensor([1.0, class_weight_value], device=device)
    print(f"Class weight (anomaly edge): {class_weight_value:.2f}")

    print("Computing edge-centered spectral energy (frozen, no-grad)...")
    energy_right, energy_left = prepare_edge_energy(
        args.energy_mode, features, edge_pairs, graph, splits.train
    )
    if not torch.isfinite(energy_right).all() or not torch.isfinite(energy_left).all():
        raise RuntimeError("B5 produced non-finite normalized edge energy.")
    print(f"E_R(e) shape: {tuple(energy_right.shape)}")
    print(f"E_L(e) shape: {tuple(energy_left.shape)}")

    model = B5EdgeBidirectionalEnergySAGE(
        in_feats=features.shape[1],
        energy_dim=features.shape[1],
        hidden_dim=args.hidden_dim,
        num_classes=data.num_classes,
        dropout=args.dropout,
        aggregator_type="mean",
        gate_hidden_dim=args.gate_hidden_dim,
        fusion=args.fusion,
    ).to(device)
    print(f"B5 trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

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
        edge_logits, _, _ = model(graph, features, edge_pairs, energy_right, energy_left)
        loss = F.cross_entropy(
            edge_logits[splits.train], hard_label[splits.train], weight=class_weight
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            edge_logits, gate, _ = model(graph, features, edge_pairs, energy_right, energy_left)
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
            gate_str = (
                f" | Gate mean: {gate.mean().item():.4f}" if gate is not None else ""
            )
            print(
                f"Epoch {epoch+1:3d}/{args.epochs} | "
                f"Loss: {loss.item():.4f} | "
                f"Val F1: {val_f1:.4f} (best: {best_val_f1:.4f}) | "
                f"Test F1: {test_metrics['macro_f1']:.4f} | "
                f"AUROC: {test_metrics['auroc']:.4f} | "
                f"AUPRC: {test_metrics['auprc']:.4f}{gate_str}"
            )

    if best_model_state is None or best_test_metrics is None:
        raise RuntimeError("No best B5 model was selected.")

    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        best_logits, best_gate, _ = model(graph, features, edge_pairs, energy_right, energy_left)
        best_probabilities = best_logits.softmax(dim=1)
    best_threshold = best_test_metrics["threshold"]

    split_names_arr = split_names(splits.train, splits.val, splits.test)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"b5_{args.dataset}_seed{args.seed}_{args.energy_mode}_{args.fusion}"

    scores_np = best_probabilities[:, 1].cpu().numpy()
    stratum_np = stratum.cpu().numpy()
    frame_data = {
        "edge_id": np.arange(edge_pairs.shape[1]),
        "u": edge_pairs[0].cpu().numpy(),
        "v": edge_pairs[1].cpu().numpy(),
        "split": split_names_arr,
        "endpoint_stratum": [STRATUM_NAMES[s] for s in stratum_np],
        "soft_label": soft_label.cpu().numpy(),
        "hard_label": hard_label.cpu().numpy(),
        "anomaly_score": scores_np,
        "predicted_label": (scores_np > best_threshold).astype(np.int64),
    }
    if best_gate is not None:
        frame_data["gate_mean"] = best_gate.mean(dim=1).cpu().numpy()
    score_path = output_dir / f"{stem}_edge_scores.csv"
    pd.DataFrame(frame_data).to_csv(score_path, index=False)
    print(f"Saved B5 edge anomaly scores: {score_path}")

    # --- Leakage diagnostic: reused, unmodified, from b4_train.py ---------------
    test_idx_np = splits.test.cpu().numpy()
    test_stratum_np = stratum_np[test_idx_np]
    test_hard_np = hard_label.cpu().numpy()[test_idx_np]
    test_score_np = scores_np[test_idx_np]
    test_pred_np = (test_score_np > best_threshold).astype(np.int64)

    diagnostic_rows = []
    overall = {
        "group": "overall_test",
        "n_edges": n_test,
        "accuracy": float((test_pred_np == test_hard_np).mean()),
        "precision": best_test_metrics["precision"],
        "recall": best_test_metrics["recall"],
        "f1": best_test_metrics["macro_f1"],
        "auroc": best_test_metrics["auroc"],
        "auprc": best_test_metrics["auprc"],
        "note": "f1 is macro-F1, matching the aggregate row and the reference row below.",
    }
    diagnostic_rows.append(overall)
    for stratum_id, stratum_name in STRATUM_NAMES.items():
        sub = test_stratum_np == stratum_id
        row = stratum_metrics(test_hard_np[sub], test_score_np[sub], test_pred_np[sub])
        row["group"] = stratum_name
        diagnostic_rows.append(row)

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

    diagnostic_path = output_dir / f"{stem}_leakage_diagnostic.csv"
    pd.DataFrame(diagnostic_rows).to_csv(diagnostic_path, index=False)
    print(f"Saved B5 leakage diagnostic: {diagnostic_path}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("B5 FINAL RESULTS (overall test edges)")
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
        "model": "B5",
        "dataset": args.dataset,
        "seed": args.seed,
        "level": "edge",
        "energy_mode": args.energy_mode,
        "fusion": args.fusion,
        "AUROC": best_test_metrics["auroc"],
        "AUPRC": best_test_metrics["auprc"],
        "F1": best_test_metrics["macro_f1"],
        "training_time": elapsed,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="B5: edge-level GraphSAGE with gated bidirectional edge energy"
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
    parser.add_argument("--gate_hidden_dim", type=int, default=16)
    parser.add_argument("--energy_mode", default="d3", choices=["d1", "d2", "d3"])
    parser.add_argument("--fusion", default="gated", choices=["gated", "naive_concat"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--output_dir", default="results/b5")
    parser.add_argument("--node_scores_dir", default="results/b0")
    parser.add_argument("--node_scores_path", default=None)
    parser.add_argument(
        "--d3_edge_limit",
        type=int,
        default=200_000,
        help=(
            "Safety guard: refuse to run D3 above this many supervised edges "
            "unless raised explicitly (pass -1 to disable). D3 loops in "
            "Python calling dgl.khop_in_subgraph once per edge and has not "
            "been benchmarked at full Amazon scale (~4.4M edges)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    train_b5(parse_args())
