"""Run the B4 static edge-anomaly arm across seeds, in Step 7's results schema plus a level column.

This mirrors run_static_node_ablation.py's structure (same CLI pattern,
same save_results/print_results_table approach) but targets the edge-level
arm introduced in Step 8. It does not modify run_static_node_ablation.py
or any of its existing output files - node-level results stay exactly as
Step 7 produced them, in their original schema, at their original paths.

Requires a B0 node-score CSV for every (dataset, seed) pair being run
here, used only for the trivial-reference leakage diagnostic inside
b4_train.py. Produce those first, e.g. via:
    python run_static_node_ablation.py --models B0 --dataset amazon --seeds 47 48 49 ...
If a required file is missing, b4_train.py raises FileNotFoundError rather
than skipping the diagnostic silently.
"""

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from b4_train import train_b4

RESULT_COLUMNS = [
    "model",
    "dataset",
    "seed",
    "level",
    "AUROC",
    "AUPRC",
    "F1",
    "training_time",
]


def build_model_args(args, seed):
    """Create a B4 training configuration for one seed."""
    return SimpleNamespace(
        dataset=args.dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        undirected=args.undirected,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=seed,
        output_dir=str(Path(args.artifact_dir) / "b4"),
        node_scores_dir=args.node_scores_dir,
        node_scores_path=None,
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
    print("\n" + "=" * 86)
    print("STATIC EDGE ABLATION RESULTS")
    print("=" * 86)
    print(display.to_string(index=False))


def run_ablation(args):
    """Execute the B4 edge arm for every requested seed, sequentially."""
    output_dir = Path(args.output_dir)
    csv_path = output_dir / f"static_edge_ablation_{args.dataset}.csv"
    json_path = output_dir / f"static_edge_ablation_{args.dataset}.json"
    records = []

    print("=" * 70)
    print("STATIC EDGE ABLATION: B4 (derived UniGAD-style edge labels)")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Seeds: {args.seeds}")
    print(f"Epochs per run: {args.epochs}")
    print(f"Split: {args.train_ratio}/{args.val_ratio}/{1-args.train_ratio-args.val_ratio}")
    print(f"Node scores for trivial reference expected under: {args.node_scores_dir}")

    for run_number, seed in enumerate(args.seeds, start=1):
        print("\n" + "#" * 70)
        print(f"EDGE ABLATION RUN {run_number}/{len(args.seeds)}: model=B4, seed={seed}")
        print("#" * 70)

        model_args = build_model_args(args, seed)
        result = train_b4(model_args)
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
        description="Run the B4 static edge-anomaly arm across seeds"
    )
    parser.add_argument(
        "--dataset",
        default="amazon",
        choices=["amazon", "yelp", "tfinance", "tsocial"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[47])
    parser.add_argument("--train_ratio", type=float, default=0.4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--undirected", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--output_dir", default="results/static_edge_ablation")
    parser.add_argument("--artifact_dir", default="results/static_edge_models")
    parser.add_argument(
        "--node_scores_dir",
        default="results/b0",
        help="Directory containing b0_{dataset}_seed{seed}_node_scores.csv for each seed run here.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_ablation(parse_args())
