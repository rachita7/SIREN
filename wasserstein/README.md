# Wasserstein analysis: do CKA and optimal transport agree on which methods find the same neurons?

Companion to `cka/`. Same neuron selections, same held-out activations, same
controls (layer-matched null, global null, disjoint-half ceiling, three
residualization variants), but the similarity measure is an exact Wasserstein
distance between the two neuron populations instead of linear CKA. Every run
also computes CKA on the identical prepared matrices and null draws, so the
two measures can be compared row by row without worrying about seeds or
preprocessing differences.

## What is being measured, and why this construction

Each method gives `X ∈ R^{N×k}`: N held-out prompts (rows, shared across
methods) × k selected neurons (columns, different per method). After
`cka_core.prepare` (dead neurons dropped, nuisance covariates residualized,
columns centered and z-scored) **each neuron is a point in prompt space
R^N** — its activation profile. Two methods are then two point clouds in the
*same* space, so a Wasserstein distance is well defined:

    W(A, B) = min over one-to-one matchings π of  mean_i  (1 − |corr(a_i, b_π(i))|)

With uniform weights and (near-)equal sizes this is a linear assignment
problem, solved exactly with `scipy.optimize.linear_sum_assignment` in
~0.1–0.6 s. For z-scored columns `1 − corr = ‖a − b‖² / 2N`, so the `--signed`
version is literally the squared 2-Wasserstein distance between the clouds;
the default takes `|corr|` because a neuron and its negation are the same
feature (CKA treats them identically too). `matched_corr = 1 − W` is the
readable form: the mean |correlation| between each of A's neurons and its
optimally matched partner in B.

**How this relates to CKA.** Both measures are functions of the same
`k_A × k_B` cross-correlation matrix `R`:

| | formula | invariant to | asks |
|---|---|---|---|
| linear CKA | `‖R_AB‖²_F / (‖R_AA‖_F ‖R_BB‖_F)` — an L2 average over *all* cross-correlations | rotations within either population | do the two populations span the same directions? |
| Wasserstein | `min_π mean_i c(i, π(i))` — an assignment over the same matrix | permutations and sign flips only | can the neurons be paired off as individuals with the same profile? |

W-similarity implies high CKA; the converse fails. Two different bases of the
same subspace score CKA = 1 and W ≫ 0 (`smoke_test.py` constructs exactly
this). So the honest expectation is: **the two measures should agree on
ranking, and disagree on pairs that share a subspace without sharing
individual neurons.** Either outcome is informative; agreement is not the
success criterion.

**Constructions that look like "Wasserstein between neuron distributions" but
answer the wrong question** (and are therefore not what is coded):

- W over `(layer, neuron_index)` coordinates — neuron index has no metric.
  The only metric coordinate is depth; that is `layer_w1` below, kept as a
  separate *location* measure.
- Discrete OT with cost 0 for the same neuron, 1 otherwise — this is
  `1 − Jaccard`, already known (0.02–0.16 across methods).
- Gromov–Wasserstein between the two prompt clouds — discards the fact that
  prompts are aligned across methods, the one thing CKA exploits.
- Sliced or entropic approximations — unnecessary here (exact OT is cheap)
  and they lose the matching itself, which is the interpretable by-product
  (`--save_matching`).

Two things this analysis gives that CKA cannot:

- **Overlap decomposition.** Neurons selected by both methods match
  themselves at cost 0. `frac_identical` is the share of such matches;
  `matched_corr_nonidentical` is the score on the genuinely different neurons
  alone. Wang / Wang-robust (Jaccard 0.6) looks similar for a trivial reason;
  this column says whether anything is left once that is removed.
- **Per-neuron match quality** (`wd_matchhist_*.png`): whether a similarity
  is carried by a few perfectly matched neurons or by many weak ones.

`layer_w1` (earth-mover's distance between the two depth histograms, in
layers) is reported alongside as a location measure. CKA's layer-matched null
divides depth out by construction — the null draw has W1 = 0 to its own
method — so this is complementary, not comparable. At N=2294 Yang sits
12–16 layers from everyone else; all other pairs are within 1–4 layers.

## Soft Matching on AdvBench (Khosla & Williams)

Isolated experiment: signed Pearson assignment, top 1% (N=4588), all 7
methods, raw AdvBench activations, one layer-matched random control.
Does not touch WildGuard or `run_wasserstein.py`.

```bash
python wasserstein/run_soft_matching.py --activations cka/activations/advbench_mean.npy
sbatch wasserstein/soft_matching_advbench.sbatch
```

Writes only `wasserstein/results/soft_matching_advbench_N4588.{png,csv}`.

## Setup

```bash
conda activate siren                    # numpy, scipy, pandas, matplotlib
python wasserstein/smoke_test.py        # known-answer checks + end-to-end, ~30 s, no GPU
```

## Run

**Depth-profile W1 only** (no activations, seconds, runs on a laptop):

```bash
python wasserstein/run_wasserstein.py --layer_only --methods all
python wasserstein/run_wasserstein.py --layer_only --methods all --budget 4588
```

**Full analysis** reuses the activation tensors from `cka/` (`cka/activations/{dataset}_{pooling}.npy` + `.meta.csv`). If they already exist, nothing needs a GPU:

```bash
python wasserstein/run_wasserstein.py --activations cka/activations/wildguard_mean.npy
python wasserstein/run_wasserstein.py --activations cka/activations/wildguard_mean.npy \
    --methods all --budget 2294 --save_matching
```

If they do not exist, extract them once exactly as for CKA (GPU, ~10 min):

```bash
python cka/build_prompts.py --dataset wildguard --max_prompts 2000
python cka/extract_activations.py --prompts cka/prompts/wildguard.csv --pooling mean --batch_size 8
```

**Agreement with CKA.** Runs automatically in `run_all.sh`; by hand:

```bash
# against the CKA computed inside the W run (same inputs, same seeds)
python wasserstein/compare_to_cka.py --wasserstein wasserstein/results/wd_wildguard_mean_N2294.csv
# against the reference cka/run_cka.py table (must be same dataset/pooling/budget/methods)
python wasserstein/compare_to_cka.py --wasserstein wasserstein/results/wd_wildguard_mean_N2294.csv \
    --cka cka/results/cka_wildguard_mean_N2294.csv
```

**Drivers**, with the same environment variables as `cka/run_all.sh`
(`DATASETS`, `POOLINGS`, `BUDGETS`, `METHODS`, `RUN_TAG`, `SKIP_EXTRACT`,
`NULL_SEEDS`; plus `WD_FLAGS` for extra `run_wasserstein.py` flags):

```bash
bash wasserstein/run_all.sh                                          # local, defaults
METHODS=all BUDGETS="459 2294 4588 9175" bash wasserstein/run_all.sh  # everything
# cluster, CPU only, activations must already exist (SKIP_EXTRACT=1 default):
RUN_TAG=all7 DATASETS=wildguard METHODS=all BUDGETS="459 2294 4588 9175" \
    sbatch --export=ALL,RUN_TAG,DATASETS,METHODS,BUDGETS wasserstein/wasserstein.sbatch
```

Use the **same** `RUN_TAG`/`DATASETS`/`METHODS`/`BUDGETS` as the CKA job so
`compare_to_cka.py` finds a matching `cka/results/cka_{tag}_N{budget}.csv`;
otherwise it falls back to the internal CKA columns and says so.

### AdvBench (alongside an already-running WildGuard job)

AdvBench is the official GitHub CSV (not the gated `walledai/AdvBench` Hub
mirror), harmful-only (~520 prompts). Every
file is tagged `advbench`, so it cannot overwrite `wd_wildguard_*`. The
shared layer-W1 tables are skipped (`SKIP_LAYER_W1=1`).

On a login node if compute nodes are offline:

```bash
conda activate siren
python cka/build_prompts.py --dataset advbench --max_prompts 2000
```

Then from the repo root (GPU: extracts `cka/activations/advbench_mean.npy` if
needed, then runs Wasserstein):

```bash
sbatch wasserstein/wasserstein_advbench.sbatch
METHODS=all sbatch --export=ALL,METHODS wasserstein/wasserstein_advbench.sbatch
```

Read `wasserstein/results/wd_advbench_mean_N2294.csv` and
`wd_vs_cka_advbench_mean_N2294.csv`. Because every prompt is harmful, the
`class` variant is uninformative; use `raw` and `class+length`.

Runtime: ~1000 assignment problems per residualization variant (nulls +
ceilings + observed). N=2294 with 7 methods: ~10 min per dataset. Assignment
is O(k³), so N=9175 is ~30× slower per call — lower `NULL_SEEDS` there.
Memory: the k×k cost matrix is < 1 GB at N=9175.

## Reading the results

Everything lands in `wasserstein/results/`. Main table `wd_{tag}_N{budget}.csv`, one row per (variant, pair):

| column | meaning |
|---|---|
| `wasserstein` | the distance, 0 = identical clouds. Raw values sit near `1 − chance` for *any* two sets (two random sets of 2294 neurons on 1500 prompts already reach matched \|corr\| ≈ 0.09), so never report without the null |
| `matched_corr` | `1 − wasserstein`. All controls below are on this scale so signs read like CKA (higher = more similar) |
| `wd_z_vs_null` | **primary statistic**, SDs above (+) / below (−) the layer-matched random null; \|z\| > 3 is significant |
| `wd_normalized` | 0 = null, 1 = same-method ceiling (disjoint halves, with the half-size null correction as in CKA). NaN when the ceiling does not sit above the null |
| `frac_identical`, `matched_corr_nonidentical` | overlap decomposition, see above |
| `frac_strong` | share of matched pairs with \|corr\| ≥ 0.5 (`--strong`) |
| `cka`, `cka_z_vs_null`, `cka_normalized` | linear CKA on the same prepared matrices, nulls and ceiling splits |
| `layer_w1` | depth disagreement in layers |
| `jaccard`, `family_pair` | as in `cka/` |

Figures: `wd_matrix_*` (matched\|corr\| / null / W-normalized / CKA-normalized
heatmaps), `wd_pairs_*` (bars vs null with ceiling), `wd_matchhist_*`
(per-neuron match quality vs null), `wd_vs_cka_*` (scatter of the two
normalized scores), `wd_variants_*`, `wd_layer_*` (depth W1 heatmap and
profiles). `--save_matching` writes the optimal neuron pairing of each
observed pair to `results/matchings/`, sorted by \|corr\| — which SIREN
neuron corresponds to which Wang neuron.

**The supervisor's question** is answered by `wd_vs_cka_{tag}_N{budget}.csv`
from `compare_to_cka.py`, per variant:

| statistic | reading |
|---|---|
| `spearman_normalized`, `spearman_z` | rank agreement between the two measures across pairs. ≳ 0.7: they order the pairs alike. ≲ 0.3: they disagree — a finding, not an error |
| `sig_both / sig_cka_only / sig_wd_only / sig_neither` | how many pairs each measure calls significant (z > 3) |
| `spearman_jaccard_vs_*_z` | how much each measure is tracking raw index overlap |
| `spearman_depth_vs_*_z` | how much each is tracking depth proximity (should be ~0 for both, given the layer-matched null) |

Per pair:

| CKA | W | meaning |
|---|---|---|
| high | high | same subspace **and** neurons interchangeable as individuals — the strong result |
| high | ≈ null | same subspace, different individual neurons (CKA's rotation invariance). Expected to be common |
| ≈ null | ≈ null | the methods find different things; Jaccard was the whole story |
| ≈ null | high | check `frac_identical` — shared neurons matching themselves |

Read the strict variant (`class+length`) for claims, as in `cka/`.

## Files

| file | role |
|---|---|
| `wd_core.py` | correlation cost, exact OT matching, overlap decomposition, layer W1; the CKA ↔ W relationship in the docstring |
| `run_wasserstein.py` | cross-method analysis with all controls + CKA on identical inputs; `--layer_only` |
| `compare_to_cka.py` | rank agreement / significance agreement, internal or external CKA table |
| `wd_plots.py` | figures (reuses generic heatmap/profile helpers from `cka/plots.py`) |
| `smoke_test.py` | known-answer checks (rotation, permutation, sign, W₂ identity, decomposition) + end-to-end run |
| `run_all.sh`, `wasserstein.sbatch` | drivers (local / CPU SLURM) |
