# Refusal-direction / DFA: do the safety-neuron methods feed the same mechanism?

Companion to `cka/`, based on Arditi et al. 2024
([arXiv:2406.11717](https://arxiv.org/abs/2406.11717)). CKA asks whether the
selected populations encode similar *representations*; this asks the
mechanistic question: **do the neurons selected by SIREN, Wang, Zhao and
Yang write into the same refusal direction of the residual stream?**

## Method

1. **Direction.** `r = mean_harmful(x) − mean_harmless(x)` over
   residual-stream states at the last prompt token; the layer is chosen by
   held-out AUROC, capped at 80% of depth (per the paper).
2. **Per-neuron DFA.** FFN neuron `(l, j)` writes
   `a_{l,j}(x) · W_down^(l)[:, j]` into the residual stream, so
   `DFA_{l,j} = (E_harmful[a] − E_benign[a]) · (W_down^(l)[:, j] · r̂)` —
   the paper's Direct Feature Attribution pushed down to single neurons.
3. **Write vectors.** Summing a method's contributions gives one
   residual-stream vector `v_M`; the pairwise `cos(v_A, v_B)` matrix (plus
   `cos(v_M, r̂)`) is the functional analogue of the CKA matrix. Two methods
   can share zero neurons and still score cos ≈ 1 here.
4. **Ablation.** Zero each method's neurons during generation: does `x · r̂`
   drop, does refusal on harmful prompts drop, does benign behavior survive?
   Attribution says what the neurons write; ablation says whether it matters.

Selections are the same `(layer, neuron)` sets as everywhere in the repo
(input to `mlp.down_proj`, 32 × 14336 neurons, loaded via
`cka/neuron_sets.py`). Budgets N = 459 / 2294 / 4588 / 9175
(0.1 / 0.5 / 1 / 2%); default N=2294 with the four canonical methods
`siren wang zhao_topk yang_refusal`; `METHODS=all` runs all 7 variants.

## Data: one WildGuard train/val/test framework

| split | purpose |
|---|---|
| WildGuardTrain → fit subset | construct r |
| WildGuardTrain → val subset | choose r's layer |
| **WildGuardTest** = `cka/prompts/wildguard.csv` (1508 balanced prompts) | **everything reported**: CKA, DFA, cosines, ablation |

The official train/test split provides the one principle that matters:
prompts that construct/select the direction never appear in the evaluation
(`build_direction_prompts.py` additionally drops any exact-text overlap).
CKA, DFA and the ablation all read the same frozen evaluation file, so
disagreements between measures cannot be blamed on the data. HarmBench +
Alpaca are upstream provenance only (neuron selection, before this
experiment starts).

## Run

```bash
conda activate siren
python refusal_direction/smoke_test.py     # verify pipeline, no GPU (~1 min)
```

**1. Prompt sets — login node (needs internet):**

```bash
python refusal_direction/build_direction_prompts.py --dataset wildguard_train
python cka/build_prompts.py --dataset wildguard      # skip if it exists
```

**2. First pass without ablation — GPU job, ~30–40 min** (residuals →
direction → WildGuardTest activations → DFA analysis):

```bash
RUN_ABLATION=0 sbatch --export=ALL,RUN_ABLATION refusal_direction/refusal.sbatch
```

**3. Check for signal:**

```bash
grep -A2 "Chosen hidden state" logs/refusal-dfa_<jobid>.out    # want AUROC ~0.9+
column -s, -t refusal_direction/results/dfa_summary_wildguard_last_dir-wildguard_train_N2294.csv
column -s, -t refusal_direction/results/writevec_cos_wildguard_last_dir-wildguard_train_N2294.csv
```

Signal = `z_sum_vs_null` (and pairwise `z_vs_null`) well beyond ±3.
If everything sits at |z| < 3, skip the ablation — the finding is already
"not preferentially on the refusal circuit".

**4. Causal ablation — GPU job, ~1 h** (all earlier artifacts are skipped;
only the ablation runs):

```bash
sbatch refusal_direction/refusal.sbatch
```

**5. Sweeps** (all artifacts exist; `SKIP_EXTRACT=1` means no GPU extraction):

```bash
# all 7 method variants at 1% budget
SKIP_EXTRACT=1 RUN_ABLATION=0 METHODS=all BUDGETS="4588" RUN_TAG=all7 \
    sbatch --export=ALL,SKIP_EXTRACT,RUN_ABLATION,METHODS,BUDGETS,RUN_TAG \
    refusal_direction/refusal.sbatch

# budget sweep, canonical 4 methods
SKIP_EXTRACT=1 RUN_ABLATION=0 BUDGETS="459 4588 9175" \
    sbatch --export=ALL,SKIP_EXTRACT,RUN_ABLATION,BUDGETS \
    refusal_direction/refusal.sbatch

# robustness: mean pooling, per-layer directions, different direction corpus
SKIP_EXTRACT=1 RUN_ABLATION=0 POOLING=mean \
    sbatch --export=ALL,SKIP_EXTRACT,RUN_ABLATION,POOLING \
    refusal_direction/refusal.sbatch
SKIP_EXTRACT=1 RUN_ABLATION=0 DIRECTION_MODE=per_layer RUN_TAG=perlayer \
    sbatch --export=ALL,SKIP_EXTRACT,RUN_ABLATION,DIRECTION_MODE,RUN_TAG \
    refusal_direction/refusal.sbatch
RUN_ABLATION=0 DIRECTION_DATASETS=openai_moderation \
    sbatch --export=ALL,RUN_ABLATION,DIRECTION_DATASETS \
    refusal_direction/refusal.sbatch
```

Fetch results: `scp "euler:~/SIREN/refusal_direction/results/*" ~/Downloads/refusal-results/`

## Controls (raw numbers are never reported alone)

- **Layer-matched random null** — the whole MLP stack writes refusal on
  harmful prompts, so random neuron subsets already show positive DFA and
  cosines. Only the excess counts: every value comes with a null mean/std
  and `z_vs_null`; |z| > 3 is significant.
- **All-MLP reference** — the write vector of all 458,752 neurons; the
  ceiling any subset is pulled toward.
- **Ablation references** — `direction ablation` (Arditi's full removal of
  r̂) is the maximum effect; `random[method]` is each method's own control;
  benign-prompt refusal must stay near baseline or the ablation just broke
  the model.

## Reading the results (`refusal_direction/results/`)

| file | content |
|---|---|
| `dfa_summary_{tag}.csv` | per method: DFA sum + `z_sum_vs_null`, `frac_positive`, `cos_r` + null + z |
| `writevec_cos_{tag}.csv`, `writevec_matrix_*.png` | method×method cosine matrix: observed / null / **z (read this panel)** |
| `direction_profile_*.png` | direction health: val AUROC by depth — check this first |
| `direction_alignment_*.png`, `dfa_dist_*.png`, `dfa_layers_*.png` | alignment bars, per-neuron distributions, depth profile |
| `dfa_map_{label}.npy`, `writevecs_{tag}.npz`, `dfa_neurons_{tag}.csv` | raw artifacts — rerun any selection offline for free |
| `ablation_{tag}.csv/.png`, `ablation_generations_*.csv` | causal results + raw completions (skim them; substring refusal detection is crude) |

| observation | meaning |
|---|---|
| `z_sum_vs_null` ≈ 0 | the method's neurons write no more refusal than random neurons in the same layers |
| cross-method cos high with z ≫ 3 | different neurons, same functional direction — the convergence result |
| method X aligns, Y does not | Y tracks a different component of safety than the refusal circuit |
| DFA z ≫ 3 but ablation ≈ its random control | writes refusal but redundantly — attribution without necessity |
| within-family pairs (`METHODS=all`) | empirical ceiling; cross-family scores at that level are the strongest result |

Compare z-scores within one budget only — null spreads differ across
selection sizes.

## Caveats

- **Refusal ≠ all of safety**: r̂ mediates refusal specifically; a
  non-aligned method may track another safety component (a refusal
  *subspace* à la Wollschläger et al. 2025 is the natural extension).
- **No z-scoring, deliberately**: DFA is a physical quantity; outlier
  neurons dominating a write vector is real. `dfa_median`/`frac_positive`
  are the robust companions.

## Files

| file | role |
|---|---|
| `refusal_core.py` | direction math, AUROC layer selection, DFA identities, refusal markers |
| `downproj.py` | streams `down_proj` weights from safetensors (no full-model load) |
| `build_direction_prompts.py` | WildGuardTrain fit/val prompt sets, eval overlap removed |
| `extract_residuals.py` | residual-stream states at the last prompt token (GPU) |
| `fit_direction.py` | difference-in-means direction + validated layer choice |
| `run_dfa.py` | per-neuron DFA, write vectors, cosine matrix, all controls (CPU) |
| `run_ablation.py` | causal neuron/direction ablation with generation (GPU) |
| `plots.py`, `smoke_test.py` | figures; correctness + end-to-end checks |
| `run_all.sh`, `refusal.sbatch` | drivers (local / SLURM) |

## References

Arditi et al. 2024 (refusal direction, DFA, directional ablation) ·
Wollschläger et al. 2025 (refusal concept cones) · Elhage et al. 2021
(residual stream as a shared write space).
