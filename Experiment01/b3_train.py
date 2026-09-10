"""Train B3: static GraphSAGE with adaptively gated E_R and E_L."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from b0_train import load_static_dataset, split_names
from b3_model import B3BidirectionalEnergySAGE
from E_train import compute_comprehensive_metrics, get_best_f1
from fast_e import local_1hop_energy_lnorm


def save_outputs(
    output_dir,
    dataset,
    seed,
    labels,
    train_mask,
    val_mask,
    test_mask,
    anomaly_scores,
    threshold,
    gates,
    mixed_energy,
):
    """Save node scores and every learned gate value from the best model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"b3_{dataset}_seed{seed}"
    split = split_names(train_mask, val_mask, test_mask)
    node_ids = np.arange(labels.numel())
    scores_np = anomaly_scores.cpu().numpy()

    score_path = output_dir / f"{stem}_node_scores.csv"
    pd.DataFrame(
        {
            "node_id": node_ids,
            "split": split,
            "label": labels.cpu().numpy(),
            "mixed_energy_mean": mixed_energy.mean(dim=1).cpu().numpy(),
            "anomaly_score": scores_np,
            "predicted_label": (scores_np > threshold).astype(np.int64),
        }
    ).to_csv(score_path, index=False)

    gates_np = gates.cpu().numpy()
    gate_data = {
        "node_id": node_ids,
        "split": split,
        "label": labels.cpu().numpy(),
        "gate_mean": gates_np.mean(axis=1),
    }
    for gate_idx in range(gates_np.shape[1]):
        gate_data[f"gate_f{gate_idx}"] = gates_np[:, gate_idx]

    gate_path = output_dir / f"{stem}_gate_values.csv"
    pd.DataFrame(gate_data).to_csv(gate_path, index=False)
    print(f"Saved B3 node anomaly scores: {score_path}")
    print(f"Saved B3 learned gate values: {gate_path}")
    return score_path, gate_path


@torch.no_grad()
def prepare_inputs(features, graph, train_mask):
    """Prepare X, E_R, and E_L exactly with EGNN's training statistics."""
    src, dst = graph.edges()
    edge_index = torch.stack([dst, src], dim=0)
    energy_right_raw = local_1hop_energy_lnorm(
        X=features,
        edge_index=edge_index,
        edge_weight=None,
        eps=1e-8,
        deg_eps=1e-12,
    )
    energy_left_raw = 2.0 - energy_right_raw

    x_mean = features[train_mask].mean(dim=0, keepdim=True)
    x_std = features[train_mask].std(dim=0, keepdim=True).clamp_min(1e-8)
    features_for_gate = (features - x_mean) / x_std

    right_mean = energy_right_raw[train_mask].mean(dim=0, keepdim=True)
    right_std = (
        energy_right_raw[train_mask]
        .std(dim=0, keepdim=True)
        .clamp_min(1e-8)
    )
    energy_right = (energy_right_raw - right_mean) / right_std
    energy_left = (energy_left_raw - right_mean) / right_std
    return features_for_gate, energy_right, energy_left


def train_b3(args):
    """Train B3 with the same data, optimizer, loss, and evaluation as B0-B2."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("B3 - GRAPHSAGE + ADAPTIVELY GATED E_R AND E_L")
    print("=" * 70)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed}")
    print(f"Epochs: {args.epochs}")
    print(f"Train/Val/Test: {args.train_ratio}/{args.val_ratio}/{1-args.train_ratio-args.val_ratio}")
    print(f"Undirected: {args.undirected}")
    print(f"Gate type: {args.gate_type}")
    print(f"Gate layers: {args.gate_num_layers}")
    print("Existing EGNN mixture: Z = G * E_R + (1 - G) * E_L")
    print("Fusion: concatenate X with gated bidirectional energy Z")

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

    features_for_gate, energy_right, energy_left = prepare_inputs(
        features, graph, train_mask
    )
    if not all(
        torch.isfinite(tensor).all()
        for tensor in (features_for_gate, energy_right, energy_left)
    ):
        raise RuntimeError("B3 produced non-finite normalized inputs.")
    print(f"E_R shape: {tuple(energy_right.shape)}")
    print(f"E_L shape: {tuple(energy_left.shape)}")

    model = B3BidirectionalEnergySAGE(
        in_feats=features.shape[1],
        hidden_dim=args.hidden_dim,
        num_classes=data.num_classes,
        dropout=args.dropout,
        aggregator_type="mean",
        gate_hidden_dim=args.gate_hidden_dim,
        gate_type=args.gate_type,
        gate_num_layers=args.gate_num_layers,
    ).to(device)
    print(f"B3 trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

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
        logits, _, _ = model(
            graph, features, features_for_gate, energy_right, energy_left
        )
        loss = F.cross_entropy(
            logits[train_mask], labels[train_mask], weight=class_weight
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits, gates, mixed_energy = model(
                graph, features, features_for_gate, energy_right, energy_left
            )
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
                f"AUPRC: {test_metrics['auprc']:.4f} | "
                f"Gate mean: {gates.mean().item():.4f}"
            )

    if best_model_state is None or best_test_metrics is None:
        raise RuntimeError("No best B3 model was selected.")

    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        best_logits, best_gates, best_mixed_energy = model(
            graph, features, features_for_gate, energy_right, energy_left
        )
        best_probabilities = best_logits.softmax(dim=1)

    score_path, gate_path = save_outputs(
        Path(args.output_dir),
        args.dataset,
        args.seed,
        labels,
        train_mask,
        val_mask,
        test_mask,
        best_probabilities[:, 1],
        best_test_metrics["threshold"],
        best_gates,
        best_mixed_energy,
    )

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("B3 FINAL RESULTS")
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
    print("\nLearned gate statistics (1=E_R, 0=E_L):")
    print(f"Gate shape: {tuple(best_gates.shape)}")
    print(f"Gate mean: {best_gates.mean().item():.6f}")
    print(f"Gate std:  {best_gates.std().item():.6f}")
    print(f"Gate min:  {best_gates.min().item():.6f}")
    print(f"Gate max:  {best_gates.max().item():.6f}")
    print(f"Node scores: {score_path}")
    print(f"Gate values: {gate_path}")

    return {
        "model": "B3",
        "dataset": args.dataset,
        "seed": args.seed,
        "AUROC": best_test_metrics["auroc"],
        "AUPRC": best_test_metrics["auprc"],
        "F1": best_test_metrics["macro_f1"],
        "training_time": elapsed,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="B3: GraphSAGE with adaptively gated E_R and E_L"
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
    parser.add_argument(
        "--gate_type",
        default="per_feature",
        choices=["per_node", "per_feature"],
    )
    parser.add_argument("--gate_hidden_dim", type=int, default=16)
    parser.add_argument(
        "--gate_num_layers", type=int, default=2, choices=[2, 3, 4]
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--output_dir", default="results/b3")
    return parser.parse_args()


if __name__ == "__main__":
    train_b3(parse_args())
