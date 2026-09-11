Project context

Undergrad's thesis extending UniGAD (multi-level graph anomaly detection) with bidirectional spectral energy, borrowing the energy formulation from the EGNN paper ("Modeling Spectral Energy Shifts in Spatio-Temporal Graph Anomaly Detection"). The UniGAD codebase is in this repo, unmodified.

Temporal modeling comes later. This entire prompt covers static graphs only.

Equations in scope for this phase
Right energy (Rayleigh quotient): E_R = xᵀLx / xᵀx
Left energy via flipped operator L_L = 2I − L, so E_L = 2 − E_R — never compute a second Laplacian product, reuse E_R
Localized per-node per-feature form: E_R(Ni,f) = Enum/Eden where Enum = Σ_{j∈N(i)} w_ij (x_i,f/√d_i − x_j,f/√d_j)² Eden = (x_i,f)²/d_i + Σ_{j∈N(i)} (x_j,f)²/d_j
Feature-wise gate: g_i,f = σ(MLP(h_i)), z_i,f = g·E_L + (1−g)·E_R
Dual sampler: S*_R = argmax E_R(S) and S*_L = argmax E_L(S)

Out of scope, do not implement: sliding windows, ΔE_t, temporal pooling, temporal GraphStitch, λ_T.

HARD CONSTRAINT: I cannot run anything locally

Everything executes in a Kaggle notebook (free tier: T4 x2 or P100, ~13 GB RAM, 9-hour session cap, /kaggle/working is the only writable persistent path, internet must be toggled on manually).

Rules this imposes on you:

You never run training. You have no visibility into my runs. Write code, hand me a copy-pasteable Kaggle cell, then stop and wait for me to paste back output or the traceback. Never assume a run succeeded.
Each phase ships a runner: committed code changes plus kaggle/phaseN_cells.md containing cells that pull the repo, install pinned deps, run the phase, and write results into /kaggle/working/results/.
Assume no internet during training. Install in a setup cell while internet is on; datasets come from a cached private Kaggle Dataset.
Checkpoint every epoch to /kaggle/working/ckpt/ with --resume support. The 9-hour cap will kill runs. Never write outside /kaggle/working.
CPU-testable unit tests wherever possible (energy math, sampler scoring, tensor shapes) so I can validate in seconds without spending GPU quota.
Results to CSV, not stdout: results/{phase}_{dataset}_{seed}.csv with phase,dataset,seed,level,f1_macro,auroc,auprc,epoch,runtime_s,peak_gpu_mb.
HARD CONSTRAINT: one phase at a time, gated on numbers

My thesis states: "Do not implement everything at once. Each step must produce a measurable baseline." I am enforcing that.

Do not scaffold future phases. No temporal code, at all, in this prompt's scope.
Do not refactor UniGAD broadly. Smallest additive change, behind a flag (--use_energy, --dual_sampler) so every earlier baseline stays runnable and reproducible from the same entry point.
After each phase: stop. I report the Kaggle numbers. You do not begin the next phase until I confirm the acceptance criterion is met.
If a phase misses its criterion, we debug that phase. We do not proceed and hope it washes out later.
The comparison target — TWO SEPARATE BASELINE RUNS

This is important and I want it done in this exact order. There are two different protocols in play and conflating them makes any gap uninterpretable.

B0-orig — reproduce UniGAD under the UniGAD paper's OWN protocol

Purpose: prove the code is correct. Nothing else. Use the UniGAD paper's own settings, not EGNN's:

Splits as the UniGAD paper specifies per dataset: Amazon 70% train, YelpChi 70%, T-Finance 40%, Reddit 40%. These differ per dataset — do not unify them.
Epochs and hyperparameters from UniGAD's own search space: learning rate in (5e-4, 1e-2), activation ∈ {ReLU, LeakyReLU, Tanh}, hidden ∈ {16, 32, 64}, MRQSampler tree depth ∈ {1, 2}, GraphStitch layers ∈ {1,2,3}×2, epochs ∈ {100, 200, 300, 400, 500}. Model selection by highest validation AUROC, as the paper does. GraphMAE pretraining at its default settings, 50 epochs.
Run both encoders, UniGAD-GCN and UniGAD-BWG, since the paper reports them separately and they differ enormously on some datasets.
Node-level targets from the UniGAD paper's own tables (F1-macro / AUPRC):
Dataset	UniGAD-GCN	UniGAD-BWG
Amazon	69.39 / 38.06	91.33 / 87.28
YelpChi	58.23 / 61.00	70.16 / 27.42
T-Finance	84.92 / 75.30	89.75 / 85.34
Reddit	56.70 / 9.73	54.08 / 5.19
Note the wild encoder split on YelpChi and Amazon AUPRC — if my numbers land near one encoder's column but not the other, that is a signal about which encoder path is actually wired up, so always report which encoder produced each row.

Acceptance for B0-orig: node-level F1-macro and AUPRC within ~1–2 points of the matching cell above. Emit results/B0_orig.csv with a protocol=unigad column and a generated markdown table showing my value, the paper value, and the delta, per dataset per encoder.

Then stop. I review before anything else happens. If this does not match, we have a code or data problem and there is no point running any other protocol.

B0-egnn — re-run the same unmodified code under EGNN's protocol

Only after B0-orig is confirmed. Purpose: establish the anchor my own contributions get measured against.

40/20/40 for every dataset, 10 seeds, mean ± std.
Fixed config from EGNN's hyperparameter table (no grid search): h_feats=32, num_layers=2, encoder=bwgnn, epoch_pretrain=50, mask_ratio=0.5.
Emit results/B0_egnn.csv with protocol=egnn.

Target is the UniGAD row of EGNN's Table 2 (F1-m / AUROC / AUPRC); the "Ours" row is the ceiling my later phases aim at:

Dataset	UniGAD row (my B0-egnn target)	"Ours" row (my ceiling)
Amazon	90.46 / 96.60 / 86.65	91.52 / 96.32 / 89.40
YelpChi	71.23 / 83.69 / 53.97	76.89 / 88.10 / 66.26
T-Finance	89.34 / 95.14 / 84.71	89.60 / 95.39 / 84.39
T-Social	78.67 / 91.65 / 58.33	95.40 / 99.69 / 95.89

Expect B0-egnn to differ from B0-orig — notably lower on Amazon and YelpChi, where training data drops from 70% to 40%. That difference is a result worth recording in the thesis, not a bug to chase. Produce a short results/protocol_comparison.md quantifying it.

Every CSV row must carry protocol, split, encoder, and seed columns so these two families of numbers can never be accidentally mixed.

T-Social is likely out of Kaggle's reach (5M+ nodes, 73M+ edges vs ~13 GB RAM). Attempt it last, in B0-egnn only. If it OOMs, say so plainly and we scope it out of the thesis rather than fighting it.

All later phases (B1, B2, dual sampler, ablations) use the EGNN protocol only, so they compare cleanly against B0_egnn.csv.

Phase plan (static only — follow exactly)
Phase 0a — Repository map (no code)

Read the repo and give me: the entry point and its CLI flags, the GraphMAE pretraining path, the encoder, the MRQSampler, the GraphStitch network, the data loaders and expected dataset format, and precisely where node representations are formed (this is where energy features will attach later). Flag anything hardcoded that will fight a flag-based extension.

Deliverable: ARCHITECTURE.md. No other code this step.

Phase 0b — Kaggle environment

Pinned dependency install that works on the current Kaggle image. Expect DGL / PyTorch / CUDA version conflicts — this is the single most likely place the project stalls. Verify DGL sees CUDA; provide a CPU fallback. Then give me instructions to prepare Amazon, YelpChi, T-Finance once as a private Kaggle Dataset so later runs need no internet.

Acceptance: a smoke run (few epochs, one seed, Amazon) completes end to end and writes a CSV. Correctness not yet expected — only that the pipeline runs.

Phase 0c — B0-orig (UniGAD's own protocol)

Unmodified UniGAD under the UniGAD paper's own splits, grid search, and epoch range, both encoders. Order: Amazon → YelpChi → T-Finance (cheapest first); Reddit optional as an extra check since it is the smallest.

Because the full grid is expensive on Kaggle's 9-hour cap, give me a sequenced search: run one config per cell block, append to CSV, resumable, so I can spread the grid across multiple sessions without losing work. Tell me the estimated GPU hours before I start.

Acceptance: matches the UniGAD paper's own node-level table within ~1–2 points. results/B0_orig.csv + delta table.

Then stop. I review before anything else.

Phase 0d — B0-egnn (EGNN protocol, fixed config)

Same unmodified code, 40/20/40, 10 seeds, no grid search, encoder=bwgnn. Emit results/B0_egnn.csv, the delta table against EGNN's Table 2 UniGAD row, and results/protocol_comparison.md quantifying B0-orig vs B0-egnn.

Acceptance: near the EGNN Table 2 UniGAD row, or a clearly reasoned account of the gap. This file is the anchor for every later phase.

Then stop. I review before we build anything new.

Phase 1 — Energy module, standalone and unit-tested

Write energy.py fully independent of UniGAD first.

Mandatory test before anything else: path graph A−B−C−D, unnormalized Laplacian, x = [1, 1.5, 0.5, 1.2] must give numerator 1.74 (= 0.25 + 1 + 0.49) and denominator 4.94. Assert both. Then assert E_L + E_R == 2 within floating tolerance.

Then localized per-node per-feature energy over 3-hop neighborhoods (EGNN's hop ablation shows 3-hop beats 1/5/7), plus the gating MLP.

Acceptance: all tests pass in a CPU-only Kaggle cell in under a minute.

Phase 2 — B2 = UniGAD + right/left energy (static)

Attach the gated energy vector to the encoder output before the sampler, behind --use_energy. Nothing else changes.

Acceptance: results/B2.csv on all three datasets, 10 seeds, in a table against both B0_egnn.csv and EGNN's "Ours" row.

Phase 2.5 — Camouflage diagnostic (do not skip)

This is what defends the contribution in my viva. On Amazon and YelpChi (which the paper says exhibit mixed left/right spectral patterns), record and plot E_R, E_L, and the learned gate g for anomalies vs normals.

Acceptance: figures/camouflage.pdf showing E_R(anom) < E_R(normal) while E_L(anom) > E_L(normal) for the left-shift subpopulation. If this does not appear, Phase 1 is wrong and we fix it rather than continuing.

Phase 3 — Dual-MRQSampler

Swap only the sampler's scoring function; keep the existing tree-depth machinery (depth 1–2). Produce S*_R and S*_L. Ablate merge vs concat.

Acceptance: results/B2_dual.csv, with runtime and peak GPU memory reported so I know it still fits Kaggle.

Phase 4 — Static ablations

Remove left branch; remove right branch; fixed gate vs learned gate; one-level vs three-level; remove GraphStitch. Two datasets, 10 seeds, mean ± std, plus a script emitting LaTeX tables straight from the CSVs.

Stop here. Temporal work is a separate conversation.

Working style
Before each phase, state your plan in ≤10 lines and wait for my approval.
Ask for Kaggle output; never guess what happened.
When I paste a traceback, fix the root cause — do not wrap it in try/except.
Maintain PROGRESS.md: phase, status, the numbers I reported, and the exact Kaggle cell that produced them. I will lose sessions and must resume from it.
Commit after every working phase (phase N: <what>). I push to GitHub and pull into Kaggle, so the repo is the transfer mechanism — never leave uncommitted code that Kaggle needs.

Begin with Phase 0a only: produce ARCHITECTURE.md. Write no other code.