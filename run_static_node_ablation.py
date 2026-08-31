"""Run matched B0/B1/B2/B3 static node-anomaly ablations."""

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from b0_train import train_b0
from b1_train import train_b1
from b2_train import train_b2
from b3_train import train_b3


TRAINERS = {
    "B0": train_b0,
    "B1": train_b1,
    "B2": train_b2,
    "B3": train_b3,
}

RESULT_COLUMNS = [
    "model",
    "dataset",
    "seed",
    "AUROC",
    "AUPRC",
    "F1",
    "training_time",
]


def build_model_args(args, model_name, seed):
    """Create the same training configuration for every ablation model."""
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
        output_dir=str(Path(args.artifact_dir) / model_name.lower()),
        gate_type=args.gate_type,
        gate_hidden_dim=args.gate_hidden_dim,
        gate_num_layers=args.gate_num_layers,
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
    print("STATIC NODE ABLATION RESULTS")
    print("=" * 86)
    print(display.to_string(index=False))


def run_ablation(args):
    """Execute every requested model/seed pair sequentially."""
    output_dir = Path(args.output_dir)
    csv_path = output_dir / f"static_node_ablation_{args.dataset}.csv"
    json_path = output_dir / f"static_node_ablation_{args.dataset}.json"
    records = []

    print("=" * 70)
    print("STATIC NODE ABLATION: B0 / B1 / B2 / B3")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Seeds: {args.seeds}")
    print(f"Epochs per run: {args.epochs}")
    print(f"Split: {args.train_ratio}/{args.val_ratio}/{1-args.train_ratio-args.val_ratio}")

    total_runs = len(args.models) * len(args.seeds)
    run_number = 0
    for seed in args.seeds:
        for model_name in args.models:
            run_number += 1
            print("\n" + "#" * 70)
            print(
                f"ABLATION RUN {run_number}/{total_runs}: "
                f"model={model_name}, seed={seed}"
            )
            print("#" * 70)

            model_args = build_model_args(args, model_name, seed)
            result = TRAINERS[model_name](model_args)
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
        description="Run matched B0-B3 static node-anomaly ablations"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["B0", "B1", "B2", "B3"],
        choices=list(TRAINERS),
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
    parser.add_argument(
        "--gate_type",
        default="per_feature",
        choices=["per_node", "per_feature"],
    )
    parser.add_argument("--gate_hidden_dim", type=int, default=16)
    parser.add_argument(
        "--gate_num_layers", type=int, default=2, choices=[2, 3, 4]
    )
    parser.add_argument("--output_dir", default="results/static_node_ablation")
    parser.add_argument("--artifact_dir", default="results/static_node_models")
    return parser.parse_args()


if __name__ == "__main__":
    run_ablation(parse_args())
