# Refusal-direction / DFA: do the safety-neuron methods feed the same mechanism?

Companion analysis to `cka/`, based on Arditi et al. 2024, *Refusal in
Language Models Is Mediated by a Single Direction*
([arXiv:2406.11717](https://arxiv.org/abs/2406.11717)). CKA asks whether the
selected populations encode similar *representations*; this folder asks a
mechanistically stronger question: **do the neurons selected by SIREN, Wang,
Zhao and Yang write into the same refusal direction of the residual
stream?** Two methods can share zero neurons (Jaccard ≈ 0.02–0.16), score
mid-range CKA, and still converge here — or vice versa.

## Method in one paragraph

A refusal direction `r = mean_harmful(x) − mean_harmless(x)` is fitted from
residual-stream states at the last prompt token on a corpus none of the
methods ever saw, with the layer chosen by held-out AUROC (capped at 80% of
depth, following the paper). Each MLP neuron `(l, j)` writes
`a_{l,j}(x) · W_down^(l)[:, j]` into the residual stream, so its per-neuron
refusal contribution — the paper's Direct Feature Attribution pushed down to
neurons — is `DFA_{l,j} = (E_harmful[a] − E_benign[a]) · (W_down^(l)[:, j] · r̂)`,
evaluated on the held-out prompts already used by the CKA analysis. Summing
each method's contributions gives one residual-stream **write vector** `v_M`
per method; the pairwise `cos(v_A, v_B)` matrix (plus each `cos(v_M, r̂)`) is
the functional analogue of the CKA matrix. Finally, `run_ablation.py` zeroes
each method's neurons during generation and checks whether `x · r̂` and
refusal behavior actually drop — attribution says what the neurons write,
ablation says whether it matters.

All selections live in the same space as everywhere else in the repo: the
input to `mlp.down_proj` of Llama-3-8B-Instruct (32 × 14336 neurons), loaded
through `cka/neuron_sets.py`. Budgets N = 459 / 2294 / 4588 / 9175.

## One dataset framework: WildGuard train/val/test

The whole comparison lives inside WildGuard's **official** splits — our
controlled adaptation of Arditi et al.'s train → validation → evaluation
protocol (they use different corpora; the principle that matters is that
prompts used to construct/select the direction are never the prompts used
for final evaluation):

| split | purpose |
|---|---|
| WildGuardTrain → fit subset | construct the difference-in-means direction r |
| WildGuardTrain → val subset | choose r's layer (held-out AUROC, 80%-depth cap) |
| **WildGuardTest** (= `cka/prompts/wildguard.csv`) | **everything that is reported**: CKA, DFA, write-vector cosines, ablation |

`cka/prompts/wildguard.csv` is the frozen master evaluation file: CKA and
DFA read the very same activation arrays and the ablation samples its
prompts from the same CSV, so if CKA and DFA disagree about a method pair,
the dataset cannot be blamed — both saw identical prompts. WildGuardTest is
never split further; its 1,508 balanced prompts stay intact as the official
human-annotated test set (the train split's labels are GPT-4-derived with
auditing, which is fine for fitting a mean difference).

HarmBench + Alpaca appear only as upstream provenance: the four neuron sets
were selected on them before this experiment starts, and they are used
nowhere inside it. As a belt-and-braces guard on top of the official split,
`build_direction_prompts.py` drops any direction prompt whose text also
appears in the evaluation CSVs. Fitting on a different corpus entirely
(e.g. `--dataset openai_moderation`) remains available as a robustness
check that r is not WildGuard-specific.

## Setup and run

```bash
conda activate siren                      # everything needed is already there
python refusal_direction/smoke_test.py    # verify pipeline, no GPU (~1 min)

bash refusal_direction/run_all.sh         # local (needs GPU for 3 steps)
sbatch refusal_direction/refusal.sbatch   # same on SLURM
```

Step by step, if you prefer:

```bash
# 1. direction prompts from WildGuardTrain (CPU, HF access): balanced fit/val
#    splits, any exact-text overlap with the eval CSVs removed
python refusal_direction/build_direction_prompts.py --dataset wildguard_train

# 2. residual-stream states at the last prompt token (GPU, ~5 min)
python refusal_direction/extract_residuals.py \
    --prompts refusal_direction/prompts/wildguard_train_fit.csv
python refusal_direction/extract_residuals.py \
    --prompts refusal_direction/prompts/wildguard_train_val.csv

# 3. fit r on the fit split, select its layer on the val split (CPU)
python refusal_direction/fit_direction.py \
    --fit_residuals refusal_direction/residuals/wildguard_train_fit_last.npy \
    --val_residuals refusal_direction/residuals/wildguard_train_val_last.npy

# 4. WildGuardTest activations -- same script/format as the CKA analysis (GPU, ~10 min)
python cka/extract_activations.py --prompts cka/prompts/wildguard.csv --pooling last

# 5. the headline DFA analysis (CPU; streams down_proj weights from the HF cache)
python refusal_direction/run_dfa.py \
    --activations cka/activations/wildguard_last.npy \
    --direction refusal_direction/directions/wildguard_train_last.npz

# 6. causal validation (GPU, ~1 h at the defaults)
python refusal_direction/run_ablation.py \
    --prompts cka/prompts/wildguard.csv \
    --direction refusal_direction/directions/wildguard_train_last.npz
```

Sweeps via environment variables, mirroring `cka/run_all.sh`:

```bash
BUDGETS="459 2294 4588 9175" bash refusal_direction/run_all.sh
METHODS=all RUN_ABLATION=0 bash refusal_direction/run_all.sh
DIRECTION_MODE=per_layer RUN_TAG=perlayer bash refusal_direction/run_all.sh
POOLING=mean bash refusal_direction/run_all.sh   # reuse the CKA activations
DIRECTION_DATASETS=openai_moderation bash refusal_direction/run_all.sh
    # robustness: fit r on a different corpus; alignment surviving a corpus
    # swap shows r is not WildGuard-specific
```

## Controls (why raw numbers are never reported alone)

- **Layer-matched random null** — the entire MLP stack writes a substantial
  refusal component on harmful prompts, so random neurons from the same
  layers already show positive DFA sums and positive write-vector cosines.
  Every observed value is reported with the null mean/std and `z_vs_null`
  (20 seeds for write vectors, 200 for scalar sums); |z| > 3 is significant.
- **All-MLP reference** — the write vector of *all* 458,752 neurons: the
  ceiling direction any subset is pulled toward, and the yardstick for "the
  method captures the stack's refusal writing with 0.5% of neurons".
- **Direction-ablation reference** (ablation only) — Arditi et al.'s full
  removal of r̂ from the residual stream: the maximum behavioral effect the
  direction can account for, against which diffuse neuron ablations are read.
- **Benign-prompt check** (ablation only) — refusal on benign prompts must
  stay near baseline, otherwise the ablation merely broke the model.
- **`per_layer` direction mode** — recomputes projections against the
  direction fitted at each layer's own output, guarding against the single
  chosen direction rotating across depth.

## Evaluating the results

Everything lands in `refusal_direction/results/`.

| file | content |
|---|---|
| `dfa_summary_{tag}.csv` | per method: DFA sum, `z_sum_vs_null`, `frac_positive`, `cos_r` + its null and z |
| `writevec_cos_{tag}.csv` + `writevec_matrix_*.png` | the 4×4 (+ All MLP) functional-similarity matrix: observed / null / z |
| `direction_alignment_*.png` | `cos(v_M, r̂)` per method vs its random control |
| `dfa_dist_*.png`, `dfa_layers_*.png`, `dfa_neurons_{tag}.csv` | per-neuron distributions and where in depth each method's refusal-writing sits |
| `dfa_map_{label}.npy` | full 32×14336 DFA map — rerun any selection/null offline for free |
| `writevecs_{tag}.npz` | the raw write vectors and r̂ |
| `ablation_{tag}.csv/.png`, `ablation_generations_*.csv` | causal results + raw completions |

| observation | meaning |
|---|---|
| `z_sum_vs_null` ≈ 0 | the method's neurons write no more refusal than random neurons in the same layers |
| cross-method `cos(v_A, v_B)` high with z ≫ 0 | different neurons, same functional direction — the convergence result |
| method X aligns, method Y does not | Y tracks a different component of safety than the refusal circuit (plausible for Yang's DPO/toxicity neurons) |
| DFA z ≫ 0 but ablation ≈ its random control | the neurons write refusal but redundantly — attribution without necessity |

Read the DFA and CKA results together: same-representation (CKA) and
same-functional-direction (DFA) are logically independent, and the four
combinations mean different things.

## Caveats

- **Refusal ≠ all of safety.** r̂ mediates *refusal* specifically. A method
  whose neurons ignore r̂ may still capture harmfulness perception or other
  safety components; later work (Wollschläger et al. 2025, concept cones)
  suggests replacing the single direction with a small subspace — the
  natural extension if the 1-D version shows signal.
- **Pooled activations.** DFA uses the same pooled-per-prompt activations as
  the CKA analysis. `--pooling last` (the default here) matches the token
  position where r̂ was fitted; `mean` blurs across positions but lets you
  reuse existing CKA activation files.
- **Substring refusal detection is crude.** Skim
  `ablation_generations_*.csv` before trusting refusal rates.
- **No z-scoring, deliberately.** DFA is a physical quantity in
  residual-stream units; massive-activation neurons dominating a write
  vector is a real effect, not an artifact. `dfa_median` and
  `frac_positive` in the summary are the outlier-robust companions.

## Files

| file | role |
|---|---|
| `refusal_core.py` | direction math, AUROC layer selection, DFA identities, refusal markers |
| `downproj.py` | streams `down_proj` weights from safetensors (no full-model load) |
| `build_direction_prompts.py` | disjoint fit/val prompt sets |
| `extract_residuals.py` | residual-stream states at the last prompt token (GPU) |
| `fit_direction.py` | difference-in-means direction + validated layer choice |
| `run_dfa.py` | per-neuron DFA, write vectors, cosine matrix, all controls |
| `run_ablation.py` | causal neuron/direction ablation with generation (GPU) |
| `plots.py`, `smoke_test.py` | figures; correctness + end-to-end checks |
| `run_all.sh`, `refusal.sbatch` | drivers (local / SLURM) |

## References

Arditi et al. 2024 (refusal direction, DFA, directional ablation) ·
Wollschläger et al. 2025 (refusal concept cones) · nostalgebraist 2020 /
Elhage et al. 2021 (residual stream as a shared write space).
