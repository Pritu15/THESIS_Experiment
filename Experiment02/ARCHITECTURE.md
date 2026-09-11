# UniGAD codebase — architecture map (Phase 0a)

Scope: static-graph, node-level path only (Amazon / YelpChi / T-Finance / Reddit), since that is
what B0-orig, B0-egnn, and the energy work (Phase 1+) target. No code was changed for this phase.

## 1. Entry point and CLI flags

Entry point: `src/main.py`, run as `python main.py --datasets ... --kernels ... [flags]` **from inside `src/`** —
every path in the codebase (`../pretrained_models/`, `../results/`, `../datasets/...`) is relative to that cwd.

`main()` (`src/main.py:160`):
1. Parses `args = get_args()` (`src/utils.py:865`).
2. Resolves `--datasets` into names via the `DATASETS` index list (`src/utils.py:30`) — either a single
   comma list of indices (`--datasets 2,3`) or a range (`--datasets 2-4`).
3. Resolves `--khop` into `sp_type`: `1` → `"star+norm"`, `2` → `"convtree+norm"`, anything else raises
   `NotImplementedError`. **`--khop` defaults to `0`**, which is not a valid choice for `main()` — it must
   always be passed explicitly as `1` or `2`.
4. For each dataset name, builds a `Dataset` (§6) with a prefix/`labels_have` chosen by a hardcoded
   name whitelist, then for each `--kernels` entry and each `--cross_modes` entry calls `work(...)`.

`work()` (`src/main.py:55`) per (dataset, kernel, cross_mode):
1. `dataset.prepare_dataset(total_trials=args.trials)` — builds train/val/test splits for all trials at once (§6).
2. `dataset.make_sp_matrix_graph_list(khop=hop, load_kg=(not args.force_remake_sp))` — builds/loads the
   MRQSampler neighbor-weight graph (§4).
3. Builds one `GraphMAE` pretrain model (§2) and pretrains it once (or loads a cached checkpoint), **before**
   the trial loop.
4. For each of `args.trials` trials: reloads the pretrained weights fresh, sets `seed = seed_list[t]`
   (`seed_list = range(3407, 10000, 10)`, `src/main.py:16` — **not** an arbitrary seed, a fixed deterministic
   sequence), gets dataloaders for that trial's split, builds a `UnifyMLPDetector` (wraps `UNIMLP_E2E`, §5)
   and trains it, collecting `MacroF1`/`AUROC`/`AUPRC` per output level.
5. Averages metrics across trials and writes one row to an Excel file via `save_results()` (`src/utils.py:917`,
   writes `../results/<name>.xlsx` — **not CSV**, see §8).

Full flag list is defined in `get_args()` (`src/utils.py:865-914`); the ones that matter for the thesis phases:

| Flag | Default | Meaning |
|---|---|---|
| `--datasets` | `''` | index or index-range into `DATASETS` |
| `--kernels` | `gcn` | comma list, e.g. `gcn,bwgnn`; encoder(-decoder) via `kernel` or `encoder-decoder` |
| `--khop` | `0` | MRQSampler tree depth; **must be set to 1 or 2** |
| `--trials` | `1` | number of seeded repeats, averaged in the output row |
| `--cross_modes` | `ng2ng` | `"<input_route>2<output_route>"`, e.g. `n2n`, `ne2ne` — controls which levels the GraphStitch net reads/predicts |
| `--epoch_pretrain` | `100` | GraphMAE pretrain epochs (paper default 50, per README) |
| `--lr` | `0.01` | pretrain LR (paper search space is `5e-4`–`1e-2`, per CLAUDE.md) |
| `--epoch_ft` | `200` | GraphStitch fine-tune epochs |
| `--lr_ft` | `0.003` | GraphStitch fine-tune LR |
| `--hid_dim` | `32` | pretrain hidden dim |
| `--act` | `leakyrelu` | pretrain encoder activation |
| `--stitch_mlp_layers` | `1` | layer count inside each GraphStitch "isolated" block (used twice — the paper's "GraphStitch layers ∈ {1,2,3}×2") |
| `--final_mlp_layers` | `2` | layer count in the final classifier MLP |
| `--mask_ratio` | `0.5` | GraphMAE masking ratio |
| `--device` | `cuda` | no CPU auto-fallback exists — pass `--device cpu` explicitly if needed |
| `--metric` | `AUROC` | metric used to select the best epoch on val (governs early stopping/model selection, so it should be `AUROC` per the paper's own protocol) |
| `--force_remake_sp` | off | forces the MRQSampler cache to be recomputed instead of loaded (see §8, Kaggle read-only landmine) |
| `--save_model` / `--load_model` | off / `''` | `--load_model` is only checked for **non-empty**, the string value itself is discarded — the actual checkpoint path is always auto-derived (`model_path` in `work()`), never user-supplied |

## 2. GraphMAE pretraining path

`GraphMAE` (`src/pretrain_models.py:22`) is a masked-feature-reconstruction SSL model, adapted from the
official GraphMAE/CogDL implementation:

- `encoding_mask_noise` randomly masks `mask_ratio` of nodes' input features (replacing with a learned
  `enc_mask_token`, with `replace_ratio` of them instead replaced by another random node's features).
- `mask_attr_prediction`: runs the masked graph through `self.encoder` (§3) → `encoder_to_decoder` linear
  projection → zeroes the masked nodes' representations again (re-mask) unless the decoder is `mlp`/`linear`
  → `self.decoder` reconstructs the original features only at the masked nodes → SCE loss
  (`sce_loss`, `src/utils.py:152`, a cosine-similarity power loss, default `alpha_l=2` — but instantiated with
  the module default `alpha_l=2`, not the CogDL usual `alpha_l=3`... check if this matters for reproduction).
- `embed(g, x)` (`src/pretrain_models.py:233`) — **just calls `self.encoder(g, x)`**, i.e. the unmasked full
  encoder forward pass. This is the function called downstream to get node representations (§7).
- Pretraining loop: `pretrain()` in `src/main.py:18`. Runs `args.epoch_pretrain` epochs, Adam + StepLR,
  over `dataset.get_pretrain_dataloaders(args.batch_size)` (all graphs in `dataset.graph_list`, unlabeled).
  Pretrain always uses a fixed `pretrain_seed = 42` (`src/main.py:75`), independent of the trial's seed.

## 3. Encoder / decoder (`src/edcoders.py`)

Three GNN backbones, selectable via `--kernels`:

- **GCN** (`GCN`/`GraphConv`, line 18/90) — a from-scratch symmetric-normalized GCN (not `dgl.nn.GraphConv`),
  with optional residual and norm. Used for both `gcn` encoder and `gcn` decoder (decoder always non-encoding,
  no final activation).
- **BWGNN** (`BWGNN`/`PolyConv`, line 447/388) — Beta-Wavelet GNN, the paper's main "BWG" encoder. Computes a
  bank of spectral polynomial filters (`calculate_theta2`, Bernstein-basis approximation of a beta
  distribution over the normalized Laplacian spectrum) and concatenates their outputs
  (`enc_out_dim = hid_dim * len(self.conv)`, i.e. **BWGNN inflates the embedding width** by `d+1=3` — this
  propagates into `embed_dim` used everywhere downstream). Only usable as an encoder in `GraphMAE`
  (`decoder_type` auto-forced to `gcn` when `kernel` has no `-` and `encoder_type == 'bwgnn'`,
  `src/main.py:63-64`).
- **GIN** (`GIN`/`GINConv`, line 188) — present but not mentioned in CLAUDE.md's scope; ignore for this thesis.

`GraphMAE.__init__` wires `encoder_type`/`decoder_type` from the `kernels` flag (`kernel` or
`encoder-decoder` split on `-`), each independently selectable.

## 4. MRQSampler ("subpooling matrix")

There is no class literally named `MRQSampler`; per the README ("the MRQSampler in our code is implemented
as two unfolded versions dedicated to orders 1 and 2 rather than a recursive version"), it's two hardcoded,
non-recursive greedy-selection routines in `src/utils.py`, both of which are literally maximizing a
**Rayleigh-quotient-shaped ratio** over candidate neighbor sets — this is the closest existing analogue to
the thesis's `E_R` sampler and is the natural place to look when building the dual `S*_R`/`S*_L` sampler in
Phase 3:

- `khop=1` → `sp_type="star+norm"` → 1-hop star graph via `dgl.KHopGraph(1)` + `select_topk_star_normft`
  (`src/utils.py:250`): for the center node `x0` and each 1-hop neighbor `xi`, greedily includes neighbors in
  descending order of `(x0-xi)² / xi²` while the cumulative ratio keeps increasing.
- `khop=2` → `sp_type="convtree+norm"` → `get_convtree_topk_nbs_norm` (`src/utils.py:317`, decorated with
  `@profile` from `line_profiler` — this import is a **hard runtime dependency** even outside profiling
  runs): a 2-hop greedy tree search, explicitly computing and tracking `RQ_max = ai/bi` while adding
  1-hop then 2-hop neighbors, essentially a depth-2 unrolled Rayleigh-quotient greedy maximizer.

Both operate on `graph.ndata['feature_normed']`, a per-node scalar (L2 norm of min-max-normalized features,
computed in `make_sp_matrix_graph_list`, `src/utils.py:479-487`) — i.e. **the existing sampler scores
neighbors using a single scalar summary of each node's feature vector, not the full per-feature localized
energy** the thesis equations define. This is expected — extending it to real per-feature `E_R`/`E_L` is
exactly Phase 3's job — but it means Phase 3 cannot just "swap the scoring function" as a drop-in without
also changing what per-node quantity is fed in (currently a single normed scalar, `feature_normed`) to a
per-node-per-feature tensor.

Output: `dataset.sp_matrix_graph_list`, one `dgl.graph` per input graph, empty of node features, with one
edge per selected (neighbor → center) pair and `edata['pw']` holding that neighbor's aggregation weight.
Cached to disk as `<full_name>.khop_<khop>.sp_type_<sp_type>.sp_matrix` next to the source dataset file
(§8 — Kaggle read-only landmine).

## 5. GraphStitch network (`UNIMLP_E2E`, `src/predictors.py:79`)

Wrapped by `UnifyMLPDetector` (`src/e2e_models.py:29`), which owns the training loop, multi-task loss
(via `pareto_fn`, `src/Pareto_fn.py` — Pareto-optimal task weighting; `pcgrad_fn.py` is imported but its call
is commented out / dead code), and evaluation (`get_best_f1` threshold sweep + `roc_auc_score` +
`average_precision_score`).

`UNIMLP_E2E` structure, keyed by `input_route`/`output_route` (single chars from `'n'|'e'|'g'`, parsed out of
`cross_mode = "<input>2<output>"`):

- `layer1` / `layer3`: per-route isolated `stitch_mlp_layers`-deep MLP blocks (`nn.ModuleDict`, one entry per
  input-route letter).
- `layer2` / `layer4`: **the actual "stitch"** — `nn.ParameterDict` of scalar weights, one per
  `(output_route, input_route)` pair, initialized to `1.0` when routes match and small random when they
  cross levels. Forward pass sums `weight[o,i] * layer_{1,3}[i](state[o])` over all input routes `i`, for each
  output route `o` — this is literally the "cross-stitch" mixing of per-level representations.
- `layer56`: final `dropout` → `final_mlp_layers`-deep MLP → linear classifier head (2 classes), applied
  per output route.

`forward()` has two structurally near-duplicated branches gated on `self.single_graph`
(set by `UnifyMLPDetector.__init__` when `dataset.is_single_graph`): the non-single-graph branch uses the
whole batched node/edge set as `inner_state['n']`/`['e']`; the single-graph branch instead indexes with
`self.mask_dicts['n'/'e'][scen]` (train/val/test node/edge id masks) since the "batch" is always the same
one graph. Both branches are otherwise identical and both start with `h = self.pretrain_model.embed(g, h)`
(§7).

## 6. Data loaders and expected dataset format

`Dataset` (`src/utils.py:409`) wraps `dgl.data.utils.load_graphs(path)`, which returns `(graph_list, label_dict)`.
Three prefixes, chosen in `main()` by a **hardcoded dataset-name whitelist** (`src/main.py:194-214`):

- `datasets/edge_labels/<name>-els` — **this is the one that matters for Amazon/YelpChi/T-Finance/Reddit**
  (`main.py:203-212`, `labels_have='ne'`). Expects a single big graph (`len(graph_list) == 1`) with
  `graph.ndata['feature']` (float), `graph.ndata['node_label']`, `graph.edata['edge_label']`. No graph-level
  label is expected/used for these (`labels_have` has no `'g'`).
- `datasets/unified/<name>` — multi-graph benchmark sets (mnist/mutag/bm/uni-tsocial), `label_dict['glabel']`
  used for graph labels, default `labels_have="ng"` (node + graph, no edge labels).
- anything else — raw path under `datasets/`, single graph, no special label wiring beyond
  `graph.ndata['feature']`.

`Dataset.__init__` also picks the MRQSampler backend from `sp_type` (`"<method>+<agg_ft>"`, e.g.
`"star+norm"`): `select_topk_fn` / `get_sp_adj_list` per §4.

`prepare_dataset(total_trials)` (`src/utils.py:527`) builds **all trials' splits up front**, not lazily per
trial:

- Casts `ndata['feature']` to float; collects `node_label`/`edge_label` lists.
- Picks `(train_ratio, val_ratio)` from a **hardcoded if/elif chain keyed on `self.name`**
  (`src/utils.py:549-557`) — `tolokers`/`questions` → `0.5/0.25`; `uni-tsocial`/`tsocial`/`tfinance`/
  `reddit`/`weibo` → `0.4/0.2`; `amazon`/`yelp` → `0.7/0.1`; `mnist0`/`mnist1` substring → `0.1/0.1`; else
  default `0.4/0.2`. **See §8 — this check is broken for the exact datasets the thesis needs.**
- For multi-graph datasets: stratified `train_test_split` over graph indices, three boolean mask matrices
  `[graph_num, trials]`.
- For single-graph datasets (Amazon/YelpChi/T-Finance/Reddit all fall here): stratified `train_test_split`
  over **node** indices (stratified on `node_label`), then builds `train/val/test` **edge** id sets by taking
  `dgl.node_subgraph` of each split and reading back `edata[dgl.EID]`. Per trial, appends the *same* graph
  object to `self.graph_list` three times (train/val/test placeholders) — see §8, this explodes the
  pretraining dataloader.

`split(trial_id)` slices `train/val/test_graphs`, label dicts, and matching `sp_matrix_graph_list` entries
for one trial. `get_graph_and_sp_dataloaders` / `get_pretrain_dataloaders` wrap these in DGL
`GraphDataLoader`s with custom collate functions (`collate_with_sp`, `collate_pretrain`,
`src/utils.py:164-192`).

## 7. Where node representations are formed (energy attachment point)

**`UNIMLP_E2E.forward()`, both branches, first line inside `with g.local_scope():`**
(`src/predictors.py:144` and `src/predictors.py:189`):

```python
h = self.pretrain_model.embed(g, h)
```

This is the pretrained encoder's raw node embedding — exactly the tensor CLAUDE.md's Phase 2 instruction
("attach the gated energy vector to the encoder output before the sampler") means. Immediately after this
line:

- if `'g' in output_route`: a graph-level mean-pool is taken from **this same `h`**, before any subgraph
  pooling — so a graph-level energy feature, if ever added, must also branch off at this exact point, not
  after `SubgraphPooling`.
- if `self.khop != 0`: `h = SubgraphPooling(h, sg_matrix)` (`src/predictors.py:26`) — this is the MRQSampler
  aggregation (§4) applied to `h`, i.e. **the sampler step CLAUDE.md refers to happens immediately after this
  line**, using `sg_matrix.edata['pw']` weights computed in §4.
- `'n'`/`'e'` routes then read off `h` (post-pooling).

So the additive energy vector for `--use_energy` belongs immediately after `h = self.pretrain_model.embed(g, h)`
and before the `if self.khop != 0: h = SubgraphPooling(...)` line, in both the single-graph and
non-single-graph branches (they are separate code paths, not shared — any patch here has to touch both).

## 8. Hardcoded landmines that will fight a flag-based extension

Ranked by how much they'll hurt the phases already scoped in CLAUDE.md.

1. **HIGH — split-ratio dataset-name check is broken for Amazon/YelpChi (breaks B0-orig, Phase 0c).**
   `main.py:212` constructs `Dataset(dataset_name + '-els', ...)`, so `self.name` becomes e.g. `"amazon-els"`,
   `"yelp-els"`. But `prepare_dataset()`'s ratio table (`src/utils.py:550-555`) checks
   `self.name in ['amazon', 'yelp']` / `['tolokers','questions']` / `[...,'tfinance','reddit',...]` — none of
   which match the `-els`-suffixed name. Every edge-label dataset silently falls through to the `0.4/0.2`
   default. For T-Finance/Reddit that happens to match CLAUDE.md's intended 40% split anyway, but **Amazon
   and YelpChi will silently get a 40% train split instead of the UniGAD paper's 70%**, invalidating B0-orig's
   whole purpose (reproducing the paper's own protocol) unless this is fixed before Phase 0c runs. This needs
   a real fix (e.g. strip `-els` before the check, or match on `self.name.split('-')[0]`), not a workaround —
   flagging now per your working style, not patching since Phase 0a is read-only.

2. **HIGH — pretraining silently 3×trials-plies the whole graph for every single-graph dataset (GPU-hour
   budget for Phase 0c/0d grid).** For Amazon/YelpChi/T-Finance/Reddit, `prepare_dataset()`'s single-graph
   branch (`src/utils.py:574-617`) appends the **same graph object** to `self.graph_list` three times per
   trial (train/val/test placeholders), before pretraining ever runs. `work()` then pretrains once over
   `dataset.get_pretrain_dataloaders(args.batch_size)`, which iterates this bloated list — so one pretrain
   epoch does `3 * args.trials` forward/backward passes over the *entire* graph instead of 1. With
   `--trials 10` (needed for B0-egnn's 10 seeds) that's a **30× inflation** of pretraining cost, and because
   `dgl.batch()` physically concatenates node features per DataLoader batch, **`--batch_size` > 1 on a
   single-graph dataset batches multiple full copies of the graph into one GPU forward pass** — likely to OOM
   Kaggle's T4/P100 on T-Finance/Reddit-scale graphs well before hitting the paper's numbers. Concretely:
   pretraining these four datasets needs `--batch_size 1`, and the estimated-GPU-hours number promised for
   Phase 0c's sequenced grid should account for this 3×trials multiplier, not just "epochs × dataset size".

3. **MEDIUM — Kaggle-specific: the MRQSampler cache and pretrained-model cache both write next to their
   *inputs*.** `sp_matrix_graphs_filename` (§4) is written beside the source dataset file
   (`self.full_name + ...`), and `model_path` for pretrained GraphMAE checkpoints is `../pretrained_models/...`
   relative to cwd. If datasets are mounted read-only from a private Kaggle Dataset (as CLAUDE.md's Phase 0b
   plans), the default `load_kg=True` path (`not args.force_remake_sp`) will try to load a cached
   `.sp_matrix` file that doesn't exist yet, fall through to recomputing it, and then unconditionally
   `save_graphs(self.sp_matrix_graphs_filename, ...)` into the read-only mount — a guaranteed crash on first
   run of any dataset/khop combination not pre-baked into the Kaggle Dataset. Phase 0b needs to either
   pre-bake `.sp_matrix` files into the Kaggle Dataset for every `(dataset, khop)` pair that will be run, or
   patch the write target to `/kaggle/working/`.

4. **MEDIUM — `--khop` defaults to `0`, which `main()` rejects.** Every invocation must explicitly pass
   `--khop 1` or `--khop 2`; there is no safe default. Trivial once known, but worth a note in the runner
   cells so a bare copy-paste doesn't immediately `NotImplementedError`.

5. **LOW — results are written as `.xlsx`, not CSV.** `save_results()` (`src/utils.py:917`) calls
   `results.transpose().to_excel(...)`. CLAUDE.md's Phase 0b/0c/0d acceptance criteria all want
   `results/*.csv` with a specific column schema (`phase,dataset,seed,...`) — the existing save path produces
   neither that filename pattern nor CSV, nor per-trial rows (it's one row per dataset/kernel/cross_mode,
   already averaged across trials, discarding per-seed values). The Phase 0b runner will need its own results
   writer rather than reusing `save_results()` as-is, and per-seed (not just mean/std) rows should be captured
   inside `work()`'s trial loop if per-seed CSVs are wanted.

6. **LOW — `--load_model`'s value is discarded.** It's only checked for non-emptiness
   (`if args.load_model == "":`); the actual checkpoint path loaded is always the auto-derived `model_path`
   in `work()`. Passing `--load_model /some/other/path.pt` silently loads the *auto-derived* path instead,
   not the one given — relevant if Phase 0b's resumable-pretraining plan wants to point at a specific
   checkpoint.

7. **LOW — `DATASETS` list has a real entry and a typo'd duplicate for the same dataset**
   (`'tfinance'` at index 4, `'tfinace'` at index 19, `src/utils.py:30-36`). `main.py`'s prefix-selection
   `elif` chain (`main.py:203-212`) checks both spellings so it happens to work either way, but `--datasets`
   index arithmetic (`DATASETS[int(t)]`) means index `19` and index `4` both resolve to T-Finance under
   different (mis)spellings — worth using index `4` consistently to avoid confusion in runner cells.

8. **LOW — `@profile` decorator on the hot-path sampler function is a hard import, not conditional.**
   `src/utils.py:22` does `from line_profiler import profile` unconditionally, and decorates
   `get_convtree_topk_nbs_norm` (§4) with it at line 316. `line_profiler` is in `requirements.txt`, so this
   works as long as that package installs on the Kaggle image, but it's an easy-to-miss pinned dependency for
   Phase 0b's install cell, and (depending on `line_profiler` version behavior) `@profile` outside a
   `kernprof` run may add per-call overhead to every sampler invocation.

## Summary for later phases

- **Phase 1 (energy.py)**: standalone, no landmine here — it doesn't touch this file.
- **Phase 2 (`--use_energy`)**: attach at `src/predictors.py:144` and `:189` (both branches), right after
  `h = self.pretrain_model.embed(g, h)` and before the `SubgraphPooling` call — see §7.
- **Phase 3 (dual sampler)**: the two functions to extend/replace are `select_topk_star_normft`
  (`khop=1`) and `get_convtree_topk_nbs_norm` (`khop=2`), both in `src/utils.py` — see §4. They currently
  score on a single per-node scalar (`feature_normed`), not a per-feature vector, so "swap only the scoring
  function" will also require changing what's precomputed into `graph.ndata['feature_normed']` upstream in
  `make_sp_matrix_graph_list`.
- **Before Phase 0c can trust its own acceptance criterion**, landmine #1 (Amazon/YelpChi split ratio) must
  be fixed, and before its GPU-hour estimate can be trusted, landmine #2 (3×trials pretrain blowup) must be
  accounted for or fixed.
