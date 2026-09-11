# Progress log

## Phase 0a — Repository map
**Status: CONFIRMED by user (2026-09-11).**

Deliverable: [ARCHITECTURE.md](ARCHITECTURE.md) (already present in the repo, committed in `67c09e4`).

Verified against current `src/` on 2026-09-11 (no drift):
- `main.py` entry point, `work()` flow, `--khop` handling (§1) — matches.
- Split-ratio landmine (§8 #1): `main.py:203-212` builds `Dataset(dataset_name + '-els', ...)`, so
  `self.name` becomes `"amazon-els"` / `"yelp-els"`, which does **not** match the `['amazon','yelp']` /
  `['tolokers','questions']` / `[...,'tfinance','reddit',...]` checks in
  `utils.py:549-557` — confirmed still present, unfixed. Amazon/YelpChi silently fall through to the
  0.4/0.2 default instead of the UniGAD paper's 0.7/0.1 split.

No code changed this phase (as required). Kaggle numbers: n/a (this phase produces no runnable code).

## Phase 0b — Kaggle environment
**Status: code written, awaiting your Kaggle output.**

What was added (no changes to any existing `src/*.py` — all new files):
- [src/data_prep/build_els_datasets.py](src/data_prep/build_els_datasets.py) — builds
  `datasets/edge_labels/<name>-els` for Reddit/Weibo/Amazon/YelpChi/Tolokers/Questions/T-Finance
  from GADBench's raw graphs (Tang et al., NeurIPS 2024, cited as [49] in the UniGAD paper).
  UniGAD ships no data or conversion script for these — see the module docstring for the
  full provenance chain (verified against arXiv:2411.06427 p.7 and github.com/squareRoot3/GADBench,
  not guessed). Edge labels are derived via the paper's own stated formula
  `P_anom(i,j) = avg(P_i_anom, P_j_anom)`; the exact binarization threshold isn't specified
  in the paper, so the script prints both candidate rules' resulting edge-anomaly % next to
  the paper's Table 1 targets for calibration before committing to one.
- [src/run_export.py](src/run_export.py) — wraps `main.py`'s unmodified `work()` to write
  CSVs in the schema CLAUDE.md wants (`results/{phase}_{dataset}_{seed}.csv`), since the
  repo's own `save_results()` writes averaged `.xlsx` rows, not per-seed CSVs (landmine #5).
  Per-seed disaggregation for `--trials > 1` is deferred to Phase 0c/0d (documented
  limitation in the file — `work()` only returns the trials-averaged score).
- [kaggle/phase0b_cells.md](kaggle/phase0b_cells.md) — 6 sequenced cells: repo pull → CUDA
  probe → pinned install (torch+dgl cu118, matched, with CPU fallback) → DGL/CUDA
  verification → GADBench download + `-els` build (with the rule-calibration dry run) →
  smoke test (Amazon, GCN, khop=1, 1 seed, 2 epochs).

Decisions confirmed with user (2026-09-11):
- Build Amazon/YelpChi/T-Finance from GADBench + the paper's formula (not hunting for an
  unpublished original file).
- For Phase 0c's B0-orig grid, use the paper's own cross_mode (e.g. `ne2ne`), accepting the
  risk that a reconstructed-edge-label mismatch could affect node-level numbers too if it
  ever comes up — to be treated as a debugging variable if B0-orig misses its target, not
  worked around preemptively.

**Waiting on you to run kaggle/phase0b_cells.md and paste back the output of each cell**,
especially: `nvidia-smi`/CUDA version (Cell 2), the install result (Cell 3), the DGL/CUDA
verification (Cell 4), the GADBench zip's extracted file names and the edge-rule calibration
numbers (Cell 5), and the smoke-test CSV (Cell 6).

## Phase 0c — B0-orig
Status: not started.

## Phase 0d — B0-egnn
Status: not started.

## Phase 1 — Energy module
Status: not started.

## Phase 2 — B2 (energy attached)
Status: not started.

## Phase 2.5 — Camouflage diagnostic
Status: not started.

## Phase 3 — Dual-MRQSampler
Status: not started.

## Phase 4 — Static ablations
Status: not started.
