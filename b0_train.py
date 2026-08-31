"""Train and evaluate B0, the non-spectral static GraphSAGE baseline."""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.exceptions import UndefinedMetricWarning

from b0_model import B0GraphSAGE
from E_train import compute_comprehensive_metrics, get_best_f1
from get_amazon import amazon_data
from get_tfinance import tfinance_data
from get_tsocial import tsocial_data
from get_yelp import yelp_data


# A model can legitimately predict no anomalous nodes during an ablation run.
# sklearn assigns precision=0.0 in that case; suppress only the repeated warning.
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def load_static_dataset(args):
    """Use the same static loaders and preprocessing as the original EGNN."""
    common = {
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "random_state": args.seed,
        "undirected": args.undirected,
        "verbose": True,
    }
    if args.dataset == "amazon":
        return amazon_data(homo=True, **common)
    if args.dataset == "yelp":
        return yelp_data(homo=True, **common)
    if args.dataset == "tfinance":
        return tfinance_data(**common)
    if args.dataset == "tsocial":
        return tsocial_data(**common)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def split_names(train_mask, val_mask, test_mask):
    """Create a split label for each node without changing any masks."""
    names = np.full(train_mask.numel(), "unlabeled", dtype=object)
    names[train_mask.cpu().numpy()] = "train"
    names[val_mask.cpu().numpy()] = "validation"
    names[test_mask.cpu().numpy()] = "test"
    return names


def save_node_scores(
    path,
    labels,
    train_mask,
    val_mask,
    test_mask,
    anomaly_scores,
    threshold,
):
    """Save anomaly probabilities for all nodes in the static graph."""
    path.parent.mkdir(parents=True, exist_ok=True)
    scores_np = anomaly_scores.cpu().numpy()
    frame = pd.DataFrame(
        {
            "node_id": np.arange(labels.numel()),
            "split": split_names(train_mask, val_mask, test_mask),
            "label": labels.cpu().numpy(),
            "anomaly_score": scores_np,
            "predicted_label": (scores_np > threshold).astype(np.int64),
        }
    )
    frame.to_csv(path, index=False)
    print(f"Saved B0 node anomaly scores: {path}")


def train_b0(args):
    """Train B0 with the original EGNN split, loss, and evaluation protocol."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("B0 - NON-SPECTRAL GRAPHSAGE NODE ANOMALY BASELINE")
    print("=" * 70)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed}")
    print(f"Epochs: {args.epochs}")
    print(f"Train/Val/Test: {args.train_ratio}/{args.val_ratio}/{1-args.train_ratio-args.val_ratio}")
    print(f"Undirected: {args.undirected}")
    print("Input: original node features (no spectral energy)")

    data = load_static_dataset(args)
    graph = data.graph.to(device)
    features = data.features.float().to(device)
    labels = data.labels.long().to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)

    print("\nDataset splits:")
    print(f"  Train: {int(train_mask.sum()):,}")
    print(f"  Val:   {int(val_mask.sum()):,}")
    print(f"  Test:  {int(test_mask.sum()):,}")

    anomaly_count = int(labels[train_mask].sum())
    if anomaly_count == 0:
        raise RuntimeError("The training split contains no anomalous nodes.")
    class_weight_value = (
        (labels[train_mask] == 0).sum().item() / anomaly_count
    )
    class_weight = torch.tensor([1.0, class_weight_value], device=device)
    print(f"Class weight (anomaly): {class_weight_value:.2f}")

    model = B0GraphSAGE(
        in_feats=features.shape[1],
        hidden_dim=args.hidden_dim,
        num_classes=data.num_classes,
        dropout=args.dropout,
        aggregator_type="mean",
    ).to(device)
    print(f"B0 trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=20,
        min_lr=1e-6,
    )

    best_val_f1 = 0.0
    best_model_state = None
    best_test_metrics = None
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        logits = model(graph, features)
        loss = F.cross_entropy(
            logits[train_mask], labels[train_mask], weight=class_weight
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(graph, features)
            probabilities = logits.softmax(dim=1)
            val_f1, val_threshold = get_best_f1(
                labels[val_mask].cpu().numpy(),
                probabilities[val_mask].cpu().numpy(),
            )

            predictions = (
                probabilities[:, 1].cpu().numpy() > val_threshold
            ).astype(np.int64)
            test_labels = labels[test_mask].cpu().numpy()
            test_predictions = predictions[test_mask.cpu().numpy()]
            test_scores = probabilities[test_mask, 1].cpu().numpy()
            test_metrics = compute_comprehensive_metrics(
                test_labels, test_scores, test_predictions
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
        raise RuntimeError("No best B0 model was selected.")

    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        best_probabilities = model(graph, features).softmax(dim=1)

    score_path = (
        Path(args.output_dir)
        / f"b0_{args.dataset}_seed{args.seed}_node_scores.csv"
    )
    save_node_scores(
        score_path,
        labels,
        train_mask,
        val_mask,
        test_mask,
        best_probabilities[:, 1],
        best_test_metrics["threshold"],
    )

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("B0 FINAL RESULTS")
    print("=" * 70)
    print(f"Training time: {elapsed:.2f}s")
    print(f"Best epoch: {best_test_metrics['epoch']}")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Validation threshold: {best_test_metrics['threshold']:.4f}")
    print(f"Test Recall:    {best_test_metrics['recall']*100:.2f}%")
    print(f"Test Precision: {best_test_metrics['precision']*100:.2f}%")
    print(f"Test Macro F1:  {best_test_metrics['macro_f1']*100:.2f}%")
    print(f"Test AUROC:     {best_test_metrics['auroc']*100:.2f}%")
    print(f"Test AUPRC:     {best_test_metrics['auprc']*100:.2f}%")
    print(f"Test RecK:      {best_test_metrics['reck']*100:.2f}%")
    print(f"Node scores: {score_path}")

    return {
        "model": "B0",
        "dataset": args.dataset,
        "seed": args.seed,
        "AUROC": best_test_metrics["auroc"],
        "AUPRC": best_test_metrics["auprc"],
        "F1": best_test_metrics["macro_f1"],
        "training_time": elapsed,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="B0: non-spectral GraphSAGE node anomaly baseline"
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
    parser.add_argument("--output_dir", default="results/b0")
    return parser.parse_args()


if __name__ == "__main__":
    train_b0(parse_args())
