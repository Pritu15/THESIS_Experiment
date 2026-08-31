"""Empirically inspect the spectral energies used by the static EGNN.

This diagnostic intentionally reuses the repository's dataset loaders and
spectral-energy functions. It does not train or modify the EGNN model.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from fast_e import local_1hop_energy_lnorm
from get_amazon import amazon_data
from get_tfinance import tfinance_data
from get_tsocial import tsocial_data
from get_yelp import yelp_data


def load_static_dataset(args):
    """Load a dataset with the same loaders and preprocessing as E_train.py."""
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


def summarize(values):
    """Return scalar descriptive statistics for a one-dimensional tensor."""
    if values.numel() == 0:
        raise ValueError("Cannot summarize an empty node group.")
    return {
        "count": int(values.numel()),
        "mean": values.mean().item(),
        "std": values.std(unbiased=False).item(),
        "min": values.min().item(),
        "max": values.max().item(),
    }


def print_summary(name, stats):
    print(
        f"{name:24s} | n={stats['count']:6d} | "
        f"mean={stats['mean']:.6f} | std={stats['std']:.6f} | "
        f"min={stats['min']:.6f} | max={stats['max']:.6f}"
    )


def save_energy_csv(path, node_ids, labels, energy_right, energy_left):
    """Save labeled-node scalar and feature-wise energies for later analysis."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required to save CSV output.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    right_np = energy_right.cpu().numpy()
    left_np = energy_left.cpu().numpy()

    data = {
        "node_id": node_ids.cpu().numpy(),
        "label": labels.cpu().numpy(),
        "energy_right_mean": right_np.mean(axis=1),
        "energy_left_mean": left_np.mean(axis=1),
    }
    for feature_idx in range(right_np.shape[1]):
        data[f"energy_right_f{feature_idx}"] = right_np[:, feature_idx]
        data[f"energy_left_f{feature_idx}"] = left_np[:, feature_idx]

    pd.DataFrame(data).to_csv(path, index=False)
    print(f"Saved energy values: {path}")


def save_energy_plot(path, labels, energy_right_node, energy_left_node):
    """Save distributions and an E_R-versus-E_L scatter plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to generate the plot.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    labels_np = labels.cpu().numpy()
    right_np = energy_right_node.cpu().numpy()
    left_np = energy_left_node.cpu().numpy()
    normal = labels_np == 0
    anomaly = labels_np == 1

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].hist(right_np[normal], bins=50, alpha=0.65, density=True, label="Normal")
    axes[0].hist(right_np[anomaly], bins=50, alpha=0.65, density=True, label="Anomaly")
    axes[0].set_title("Right energy distribution")
    axes[0].set_xlabel("Per-node mean E_R")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    axes[1].hist(left_np[normal], bins=50, alpha=0.65, density=True, label="Normal")
    axes[1].hist(left_np[anomaly], bins=50, alpha=0.65, density=True, label="Anomaly")
    axes[1].set_title("Left/flip energy distribution")
    axes[1].set_xlabel("Per-node mean E_L")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    axes[2].scatter(right_np[normal], left_np[normal], s=8, alpha=0.25, label="Normal")
    axes[2].scatter(right_np[anomaly], left_np[anomaly], s=10, alpha=0.5, label="Anomaly")
    axes[2].set_title("E_R versus E_L")
    axes[2].set_xlabel("Per-node mean E_R")
    axes[2].set_ylabel("Per-node mean E_L")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization: {path}")


@torch.no_grad()
def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data = load_static_dataset(args)
    graph = data.graph.to(device)
    features = data.features.float().to(device)
    labels = data.labels.long().to(device)

    # Match E_train.py's expected edge orientation: [destination, source].
    src, dst = graph.edges()
    edge_index = torch.stack([dst, src], dim=0)

    energy_right = local_1hop_energy_lnorm(
        X=features,
        edge_index=edge_index,
        edge_weight=None,
        eps=1e-8,
        deg_eps=1e-12,
    )
    # Match the exact flipped-energy formulation used in GatedEnergySAGE.
    energy_left = 2.0 - energy_right

    # Use only nodes included in the loader's labeled evaluation population.
    labeled_mask = (
        data.train_mask.to(device)
        | data.val_mask.to(device)
        | data.test_mask.to(device)
    )
    labeled_node_ids = torch.nonzero(labeled_mask, as_tuple=False).squeeze(1)
    labeled_labels = labels[labeled_mask]
    labeled_right = energy_right[labeled_mask]
    labeled_left = energy_left[labeled_mask]

    if not torch.isfinite(labeled_right).all() or not torch.isfinite(labeled_left).all():
        raise RuntimeError("Non-finite spectral-energy values were detected.")

    # E_R and E_L are [N, F]. Average features only for node-level summaries.
    right_node = labeled_right.mean(dim=1)
    left_node = labeled_left.mean(dim=1)
    normal_mask = labeled_labels == 0
    anomaly_mask = labeled_labels == 1

    print("\nEnergy tensor information")
    print(f"  Full E_R shape: {tuple(energy_right.shape)}")
    print(f"  Full E_L shape: {tuple(energy_left.shape)}")
    print(f"  Labeled nodes: {int(labeled_mask.sum())}")
    print(f"  Normal labeled nodes: {int(normal_mask.sum())}")
    print(f"  Anomalous labeled nodes: {int(anomaly_mask.sum())}")
    print(f"  max|E_L - (2 - E_R)|: {(energy_left - (2.0 - energy_right)).abs().max().item():.10f}")

    print("\nPer-node energy statistics (feature mean per node)")
    print_summary("E_R normal", summarize(right_node[normal_mask]))
    print_summary("E_R anomalous", summarize(right_node[anomaly_mask]))
    print_summary("E_L normal", summarize(left_node[normal_mask]))
    print_summary("E_L anomalous", summarize(left_node[anomaly_mask]))

    output_dir = Path(args.output_dir)
    stem = f"{args.dataset}_seed{args.seed}"
    if args.save_csv:
        save_energy_csv(
            output_dir / f"{stem}_energies.csv",
            labeled_node_ids,
            labeled_labels,
            labeled_right,
            labeled_left,
        )
    if args.plot:
        save_energy_plot(
            output_dir / f"{stem}_energy_distribution.png",
            labeled_labels,
            right_node,
            left_node,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose the existing static EGNN spectral energies."
    )
    parser.add_argument(
        "--dataset",
        default="amazon",
        choices=["amazon", "yelp", "tfinance", "tsocial"],
    )
    parser.add_argument("--train_ratio", type=float, default=0.4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--undirected", action="store_true")
    parser.add_argument("--save_csv", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output_dir", default="results/spectral_energy")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
