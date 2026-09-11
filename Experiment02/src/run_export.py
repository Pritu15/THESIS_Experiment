"""
Thin wrapper around UniGAD's unmodified `work()` (main.py) that writes results in the CSV
schema Experiment02/CLAUDE.md requires (results/{phase}_{dataset}_{seed}.csv with columns
phase,dataset,seed,level,f1_macro,auroc,auprc,epoch,runtime_s,peak_gpu_mb) instead of the
repo's own `save_results()`, which writes one averaged .xlsx row per dataset/kernel/cross_mode
and discards per-seed values (see ARCHITECTURE.md Sec. 8, landmine #5).

Does not modify main.py/utils.py/etc. Reuses `get_args()` and `work()` as-is; this file only
adds the CSV-writing and dataset-name-resolution glue main.py's own `main()` keeps inline.

Known limitation (fine for Phase 0b's single-seed smoke test; revisit for Phase 0c/0d):
`work()` only returns metrics averaged across `args.trials` trials, not each trial's raw
score. With --trials 1 the "mean" *is* the single trial's value, so seed is unambiguous
(seed_list[0] = 3407). For multi-seed runs, work()'s trial loop would need to be
duplicated here to capture true per-seed rows -- deferred until Phase 0c/0d actually need it.
"""
import argparse
import os
import sys

import pandas as pd
import torch

# Peel off our own flags before handing the rest of sys.argv to UniGAD's own get_args().
_own_parser = argparse.ArgumentParser(add_help=False)
_own_parser.add_argument("--phase", type=str, required=True)
_own_parser.add_argument("--protocol", type=str, default="")
_own_parser.add_argument("--out_dir", type=str, default="../results")
_own_args, _remaining_argv = _own_parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining_argv

from utils import get_args, Dataset, NAME_MAP, DATASETS  # noqa: E402
from main import work, seed_list  # noqa: E402

# Mirrors main.py:194-214's dataset-name whitelist (kept in sync manually; main.py is
# unmodified so this can't import a shared helper -- see ARCHITECTURE.md Sec. 6/8).
_UNIFIED_NAMES = {
    "uni-tsocial", "mnist/dgl/mnist0", "mnist/dgl/mnist1",
    "mutag/dgl/mutag0", "bm/dgl/bm_mn_dgl", "bm/dgl/bm_ms_dgl", "bm/dgl/bm_mt_dgl",
}
_EDGE_LABEL_NAMES = {
    "reddit", "weibo", "amazon", "yelp", "tfinace", "tolokers", "questions", "tfinance",
}

# Mirrors utils.py Dataset.prepare_dataset()'s hardcoded ratio table (src/utils.py:549-557),
# for CSV annotation only -- not used to drive training, which reads the ratio straight off
# `dataset.name` at run time (see ARCHITECTURE.md Sec. 8, landmine #1: that lookup is keyed
# on the raw name and does NOT match the "-els"-suffixed name main.py actually constructs
# for amazon/yelp, so those two silently get the 0.4/0.2 default rather than 0.7/0.1).
def resolved_split_ratio(dataset_name):
    if dataset_name in ("tolokers", "questions"):
        return "0.5/0.25"
    if dataset_name in ("uni-tsocial", "tsocial", "tfinance", "tfinace", "reddit", "weibo"):
        return "0.4/0.2"
    if dataset_name in ("amazon", "yelp"):
        return "0.4/0.2  # BUG: paper wants 0.7/0.1 but '-els' suffix breaks the name match"
    return "0.4/0.2 (default)"


def resolve_dataset(dataset_name, sp_type):
    if dataset_name in _UNIFIED_NAMES:
        return Dataset(dataset_name, prefix="../datasets/unified/", sp_type=sp_type)
    elif dataset_name in _EDGE_LABEL_NAMES:
        return Dataset(dataset_name + "-els", prefix="../datasets/edge_labels/",
                        sp_type=sp_type, labels_have="ne")
    else:
        return Dataset(dataset_name)


def main():
    args = get_args()

    if args.khop == 1:
        sp_type = "star+norm"
    elif args.khop == 2:
        sp_type = "convtree+norm"
    else:
        raise NotImplementedError("--khop must be 1 or 2")

    if "-" in args.datasets:
        st, ed = args.datasets.split("-")
        dataset_names = DATASETS[int(st):int(ed) + 1]
    else:
        dataset_names = [DATASETS[int(t)] for t in args.datasets.split(",")]

    kernels = args.kernels.split(",")
    cross_modes = args.cross_modes.split(",")

    if args.trials != 1:
        print(f"WARNING: run_export.py only writes one CSV row per (dataset, kernel, "
              f"cross_mode) using work()'s mean over {args.trials} trials -- true per-seed "
              f"disaggregation is not implemented yet (see module docstring). Proceeding "
              f"with the averaged value as a single row.")

    os.makedirs(args.out_dir, exist_ok=True)

    for dataset_name in dataset_names:
        dataset = resolve_dataset(dataset_name, sp_type)
        for kernel in kernels:
            for cross_mode in cross_modes:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                model_result = work(dataset, kernel, cross_mode, args)
                runtime_s = model_result["time cost"].iloc[0]
                peak_gpu_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                               if torch.cuda.is_available() else 0.0)

                output_route = [c for c in cross_mode.split("2")[1]]
                seed = seed_list[0] if args.trials == 1 else f"mean_of_{args.trials}"

                rows = []
                for k in output_route:
                    level = NAME_MAP[k]
                    rows.append({
                        "phase": _own_args.phase,
                        "protocol": _own_args.protocol,
                        "dataset": dataset_name,
                        "split": resolved_split_ratio(dataset_name),
                        "encoder": kernel,
                        "khop": args.khop,
                        "seed": seed,
                        "level": level,
                        "f1_macro": model_result[f"MacroF1 {level} mean"].iloc[0],
                        "auroc": model_result[f"AUROC {level} mean"].iloc[0],
                        "auprc": model_result[f"AUPRC {level} mean"].iloc[0],
                        "epoch": args.epoch_ft,
                        "epoch_pretrain": args.epoch_pretrain,
                        "runtime_s": runtime_s,
                        "peak_gpu_mb": peak_gpu_mb,
                        "device": args.device,
                    })

                out_name = f"{_own_args.phase}_{dataset_name}_{seed}.csv"
                out_path = os.path.join(args.out_dir, out_name)
                pd.DataFrame(rows).to_csv(out_path, index=False)
                print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
