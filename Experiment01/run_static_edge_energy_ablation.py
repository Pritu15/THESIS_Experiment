"""Run B5 across seeds and energy-mode/fusion combinations, so D1/D2/D3 and
gated/naive_concat can be compared under identical conditions.

Mirrors run_static_node_ablation.py / run_static_edge_ablation.py's
structure. Schema extends Step 7/8's with `energy_mode` and `fusion`
columns; does not modify either prior runner or its output files.

Requires a B0 node-score CSV for every (dataset, seed) pair being run here
(same requirement as Step 8's b4_train.py), used only for the trivial-
reference leakage diagnostic.
"""

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from b5_train import train_b5

RESULT_COLUMNS = [
    "model",
    "dataset",
    "seed",
    "level",
    "energy_mode",
    "fusion",
    "AUROC",
    "AUPRC",
    "F1",
    "training_time",
]


def build_model_args(args, seed, energy_mode, fusion):
    """Create a B5 training configuration for one (seed, energy_mode, fusion) run."""
    return SimpleNamespace(
        dataset=args.dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        undirected=args.undirected,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        gate_hidden_dim=args.gate_hidden_dim,
        energy_mode=energy_mode,
        fusion=fusion,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=seed,
        output_dir=str(Path(args.artifact_dir) / "b5"),
        node_scores_dir=args.node_scores_dir,
        node_scores_path=None,
        d3_edge_limit=args.d3_edge_limit,
    )


def save_results(records, csv_path, json_path):
    """Persist all completed runs so partial progress is not lost."""
    frame = pd.DataFrame(records, columns=RESULT_COLUMNS)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(frame.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    return frame


def print_results_table(frame):
    """Print one clean table containing only measured results."""
    display = frame.copy()
    for column in ("AUROC", "AUPRC", "F1"):
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    display["training_time"] = display["training_time"].map(
        lambda value: f"{value:.2f}s"
    )
    print("\n" + "=" * 90)
    print("STATIC EDGE ENERGY ABLATION RESULTS (B5)")
    print("=" * 90)
    print(display.to_string(index=False))


def run_ablation(args):
    """Execute every requested (seed, energy_mode, fusion) combination, sequentially."""
    output_dir = Path(args.output_dir)
    csv_path = output_dir / f"static_edge_energy_ablation_{args.dataset}.csv"
    json_path = output_dir / f"static_edge_energy_ablation_{args.dataset}.json"
    records = []

    combos = [(mode, fusion) for mode in args.energy_modes for fusion in args.fusions]
    total_runs = len(args.seeds) * len(combos)

    print("=" * 70)
    print("STATIC EDGE ENERGY ABLATION: B5 (D1/D2/D3 x gated/naive_concat)")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Seeds: {args.seeds}")
    print(f"Energy modes: {args.energy_modes}")
    print(f"Fusions: {args.fusions}")
    print(f"Node scores for trivial reference expected under: {args.node_scores_dir}")

    run_number = 0
    for seed in args.seeds:
        for energy_mode, fusion in combos:
            run_number += 1
            print("\n" + "#" * 70)
            print(
                f"RUN {run_number}/{total_runs}: model=B5 seed={seed} "
                f"energy_mode={energy_mode} fusion={fusion}"
            )
            print("#" * 70)

            model_args = build_model_args(args, seed, energy_mode, fusion)
            result = train_b5(model_args)
            records.append({column: result[column] for column in RESULT_COLUMNS})

            frame = save_results(records, csv_path, json_path)
            print_results_table(frame)
            print(f"CSV:  {csv_path}")
            print(f"JSON: {json_path}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    final_frame = save_results(records, csv_path, json_path)
    print_results_table(final_frame)
    print(f"\nCompleted {len(records)} measured runs.")
    print(f"Final CSV:  {csv_path}")
    print(f"Final JSON: {json_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run B5 across seeds and energy-mode/fusion combinations"
    )
    parser.add_argument(
        "--dataset",
        default="amazon",
        choices=["amazon", "yelp", "tfinance", "tsocial"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[47])
    parser.add_argument(
        "--energy_modes", nargs="+", default=["d3"], choices=["d1", "d2", "d3"]
    )
    parser.add_argument(
        "--fusions", nargs="+", default=["gated"], choices=["gated", "naive_concat"]
    )
    parser.add_argument("--train_ratio", type=float, default=0.4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--undirected", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--gate_hidden_dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--output_dir", default="results/static_edge_energy_ablation")
    parser.add_argument("--artifact_dir", default="results/static_edge_energy_models")
    parser.add_argument("--node_scores_dir", default="results/b0")
    parser.add_argument("--d3_edge_limit", type=int, default=200_000)
    return parser.parse_args()


if __name__ == "__main__":
    run_ablation(parse_args())
