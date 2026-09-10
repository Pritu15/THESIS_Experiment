"""Train B1: static GraphSAGE with right spectral energy only."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from b0_train import load_static_dataset, split_names
from b1_model import B1RightEnergySAGE
from E_train import compute_comprehensive_metrics, get_best_f1
from fast_e import local_1hop_energy_lnorm


def save_node_scores(
    path,
    labels,
    train_mask,
    val_mask,
    test_mask,
    anomaly_scores,
    threshold,
    energy_right_node,
):
    """Save B1 anomaly scores and the node-level mean right energy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    scores_np = anomaly_scores.cpu().numpy()
    frame = pd.DataFrame(
        {
            "node_id": np.arange(labels.numel()),
            "split": split_names(train_mask, val_mask, test_mask),
            "label": labels.cpu().numpy(),
            "energy_right_mean": energy_right_node.cpu().numpy(),
            "anomaly_score": scores_np,
            "predicted_label": (scores_np > threshold).astype(np.int64),
        }
    )
    frame.to_csv(path, index=False)
    print(f"Saved B1 node anomaly scores: {path}")


@torch.no_grad()
def compute_right_energy(features, graph, train_mask):
    """Compute existing E_R and normalize it using training nodes only."""
    src, dst = graph.edges()
    edge_index = torch.stack([dst, src], dim=0)
    energy_right_raw = local_1hop_energy_lnorm(
        X=features,
        edge_index=edge_index,
        edge_weight=None,
        eps=1e-8,
        deg_eps=1e-12,
    )
    mean = energy_right_raw[train_mask].mean(dim=0, keepdim=True)
    std = energy_right_raw[train_mask].std(dim=0, keepdim=True).clamp_min(1e-8)
    energy_right_normalized = (energy_right_raw - mean) / std
    return energy_right_raw, energy_right_normalized


def train_b1(args):
    """Train B1 with B0's data, optimizer, loss, and evaluation protocol."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("B1 - GRAPHSAGE + RIGHT SPECTRAL ENERGY")
    print("=" * 70)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed}")
    print(f"Epochs: {args.epochs}")
    print(f"Train/Val/Test: {args.train_ratio}/{args.val_ratio}/{1-args.train_ratio-args.val_ratio}")
    print(f"Undirected: {args.undirected}")
    print("Energy: existing local_1hop_energy_lnorm (right branch only)")
    print("Fusion: concatenate X with training-normalized E_R")

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

    energy_right_raw, energy_right = compute_right_energy(
        features, graph, train_mask
    )
    if not torch.isfinite(energy_right).all():
        raise RuntimeError("B1 produced non-finite right-energy values.")
    print(f"Raw E_R shape: {tuple(energy_right_raw.shape)}")
    print(
        "Normalized E_R (train nodes): "
        f"mean={energy_right[train_mask].mean().item():.6f}, "
        f"std={energy_right[train_mask].std().item():.6f}"
    )

    model = B1RightEnergySAGE(
        in_feats=features.shape[1],
        hidden_dim=args.hidden_dim,
        num_classes=data.num_classes,
        dropout=args.dropout,
        aggregator_type="mean",
    ).to(device)
    print(f"B1 trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

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
        logits = model(graph, features, energy_right)
        loss = F.cross_entropy(
            logits[train_mask], labels[train_mask], weight=class_weight
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(graph, features, energy_right)
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
        raise RuntimeError("No best B1 model was selected.")

    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        best_probabilities = model(graph, features, energy_right).softmax(dim=1)

    score_path = (
        Path(args.output_dir)
        / f"b1_{args.dataset}_seed{args.seed}_node_scores.csv"
    )
    save_node_scores(
        score_path,
        labels,
        train_mask,
        val_mask,
        test_mask,
        best_probabilities[:, 1],
        best_test_metrics["threshold"],
        energy_right_raw.mean(dim=1),
    )

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("B1 FINAL RESULTS")
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
        "model": "B1",
        "dataset": args.dataset,
        "seed": args.seed,
        "AUROC": best_test_metrics["auroc"],
        "AUPRC": best_test_metrics["auprc"],
        "F1": best_test_metrics["macro_f1"],
        "training_time": elapsed,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="B1: GraphSAGE with right spectral energy only"
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
    parser.add_argument("--output_dir", default="results/b1")
    return parser.parse_args()


if __name__ == "__main__":
    train_b1(parse_args())
