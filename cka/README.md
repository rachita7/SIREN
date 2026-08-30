# CKA analysis: do the safety-neuron methods find the same representation?

The methods select almost disjoint neurons (Jaccard ≈ 0.02–0.16). This folder
tests whether their *populations* nevertheless encode the same information,
using linear CKA on held-out prompts. Full methodology and report-ready prose:
`METHODS.md`.

All selections live in the same space: the input to
`model.layers[l].mlp.down_proj` of Llama-3-8B-Instruct (32 × 14336 = 458,752
neurons), matching `mlpneuron_mean` in `utils/model_hooks.py`.

| key | method |
|---|---|
| `siren` | SIREN — Jiao et al., *LLM Safety From Within* |
| `wang`, `wang_robust` | Wang — *Neuron-Level Safety Alignment for LLMs* |
| `zhao_topk`, `zhao_eps` | Zhao — *Understanding and Enhancing Safety Mechanisms* |
| `yang_refusal`, `yang_harmfulness` | Yang et al. — *How Does DPO Reduce Toxicity?* (EMNLP 2025) |

Budgets N = **459 / 2294 / 4588 / 9175** (0.1 / 0.5 / 1 / 2 % of all MLP
neurons). Defaults: `siren wang zhao_topk yang_refusal` at N=2294;
`--methods all` runs all 7.

## Setup

```bash
conda activate siren            # everything needed is already there
python cka/smoke_test.py        # verify pipeline, no GPU (~30 s)
python cka/neuron_sets.py       # inspect selections + Jaccard, no GPU
```

HuggingFace (one-time): accept terms for
[Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
and [allenai/wildguardmix](https://huggingface.co/datasets/allenai/wildguardmix),
then `huggingface-cli login`. XSTest is ungated. On Euler, run
`build_prompts.py` on a **login node** (compute nodes may be offline).

## Run

```bash
bash cka/run_all.sh          # local: 4 methods, N=2294, wildguard + xstest
sbatch cka/cka.sbatch        # same on SLURM (GPU)
```

Sweeps via environment variables:

```bash
METHODS=all BUDGETS="459 2294 4588 9175" bash cka/run_all.sh   # everything
POOLINGS="mean last" bash cka/run_all.sh                       # pooling robustness
DATASETS="wildguard+openai_moderation xstest" bash cka/run_all.sh  # pool corpora
```

The GPU steps (prompts + activation extraction) run once per dataset; every
budget/method/seed sweep after that is CPU-only and rereads the saved
activations.

Step by step, if you prefer:

```bash
python cka/build_prompts.py --dataset wildguard --max_prompts 2000   # CPU, HF access
python cka/extract_activations.py --prompts cka/prompts/wildguard.csv \
    --pooling mean --batch_size 8                                    # GPU, ~10 min
python cka/run_cka.py --activations cka/activations/wildguard_mean.npy      # CPU
python cka/run_cross_layer.py --activations cka/activations/wildguard_mean.npy \
    --variant class+length                                           # CPU
python cka/check_overlap_effect.py \
    --activations cka/activations/wildguard_mean.npy                 # CPU
```

### Concurrent runs

Outputs are named by dataset + pooling, so a second configuration on the same
dataset must set `RUN_TAG` (or it overwrites the first) and `SKIP_EXTRACT=1`
(reuse existing activations; no GPU needed):

```bash
RUN_TAG=all7 DATASETS=wildguard METHODS=all BUDGETS="459 2294 4588 9175" \
    sbatch --export=ALL,RUN_TAG,DATASETS,METHODS,BUDGETS cka/cka_analysis.sbatch
```

### Runtime (2000 prompts, 24 GB GPU)

Prompts 1–5 min; extraction 5–10 min (GPU); cross-method CKA 5–10 min;
cross-layer 10–20 min. Defaults ≈ 1 h total. `METHODS=all` over all budgets
≈ 3–4 h. Levers: `--skip_rsa`, `RUN_CROSS_LAYER=0`, `--null_seeds`.

## Controls (why raw CKA is never reported alone)

- **Layer-matched random null** — random neurons with each method's per-layer
  counts. Two random subsets of one model already score CKA ≈ 0.9+, so a raw
  number means nothing without this. 20 seeds; reported as mean, std and
  `z_vs_null`.
- **Ceiling** — CKA between two disjoint halves of one method's *own*
  selection: what "same information, same procedure" scores. Includes a
  half-size random baseline, since halving lowers CKA by itself.
- **Residualization variants** — `raw`; `class` (harmful/benign means
  projected out); `class+length` (also token count, and dataset identity when
  corpora are pooled). The strict variant is the one that supports claims.
- **Robustness measures** — unbiased-HSIC CKA (small-sample), Spearman RSA
  (outlier neurons), per-neuron z-scoring (`--no_zscore` to check).
- **Overlap control** — `check_overlap_effect.py` deletes the neurons two
  methods *share* and recomputes. A pair whose z collapses was only scoring
  high because both matrices contained identical columns.

## Evaluating the results

Everything lands in `cka/results/`. Main table: `cka_{tag}_N{budget}.csv`,
one row per (variant, pair). Figures: `cka_matrix_*` (observed/null/normalized
heatmaps), `cka_pairs_*` (bars vs null with ceiling), `cka_variants_*`
(normalized across variants), `crosslayer_*` (32×32 maps),
`layer_profiles_*`.

**Primary statistic: `z_vs_null`** — standard deviations above (+) or below
(−) the layer-matched random null. |z| > 3 is significant.

| observation | meaning |
|---|---|
| z ≈ 0 | Pair is indistinguishable from random neurons in the same layers. The near-zero Jaccard is the whole story. |
| z ≫ 0 and `normalized` ≈ 1 | Populations are representationally interchangeable despite disjoint neurons — the interesting positive result. Confirm with `check_overlap_effect.py`. |
| z ≪ 0 | Selections are *less* mutually aligned than random sets: each method finds distinctive but different structure. |

`normalized` rescales CKA so 0 = null and 1 = ceiling. It is NaN when the
ceiling does not sit above the null (small datasets, e.g. XSTest's 400
prompts, cannot resolve the scale) — fall back to z.

Then read across variants: high on `raw` but collapsing on `class` means the
methods only share the coarse harmful/benign axis; surviving `class+length` is
the strong claim.

Consistency checks: `cka` ≈ `cka_unbiased` (else add prompts);
`rsa_spearman` moves with `cka` (else outlier neurons drive it). With
`--methods all`, compare `family_pair` `within` vs `cross` rows: cross-method
scores at within-method level is the strongest form of the result.

Cross-layer maps: read the **z panel only**; the raw panel shows a diagonal
band for any subsets because neighboring layers share the residual stream.
Off-diagonal z peaks = same structure at different depths. Cells with fewer
than `--min_neurons` (10) are masked — Yang concentrates ~80% of its neurons
in layers 29–31.

## Data

Neurons were selected on HarmBench + Alpaca, so those are excluded here.
Defaults: **wildguard** (primary; gated; balanced to 1508 prompts by its 754
harmful rows) and **xstest** (stress test; 400 prompts whose surface form is
decoupled from the label). Ungated alternatives: `openai_moderation`,
`beavertails`, `advbench`. Pool corpora (`--dataset a b` or `DATASETS="a+b"`)
to get past a small minority class; pooled runs deduplicate and residualize
dataset identity automatically.

## Troubleshooting

- **`RuntimeWarning ... in matmul` on macOS** — spurious (Accelerate BLAS);
  results are exact. `python -W ignore::RuntimeWarning` silences it.
- **CUDA OOM during extraction** — `--batch_size 4`.
- **CPU OOM in `run_cka.py`** — reduce budget/methods, or `--skip_rsa`.
- **Missing SIREN JSONs on the cluster** — `git pull`; `.gitignore` needs the
  `!results/**/*.json` negation. Verify with `python cka/neuron_sets.py`.

## Files

| file | role |
|---|---|
| `neuron_sets.py` | selection loaders, nulls, half-splits; runnable summary |
| `cka_core.py` | CKA (biased/unbiased), RSA, residualization, normalization |
| `build_prompts.py` | held-out, balanced, chat-templated prompt sets |
| `extract_activations.py` | `[N, 32, 14336]` float16 activations + metadata |
| `run_cka.py` | cross-method analysis with all controls |
| `run_cross_layer.py` | 32×32 cross-layer analysis, per-layer-pair nulls |
| `check_overlap_effect.py` | shared-neuron control |
| `plots.py`, `smoke_test.py` | figures; correctness + end-to-end checks |
| `run_all.sh`, `cka.sbatch`, `cka_analysis.sbatch` | drivers (local / GPU / CPU-only) |
| `METHODS.md` | report-ready methodology write-up |

## References

Kornblith et al. 2019 (linear CKA) · Song et al. 2012 (unbiased HSIC) ·
Davari et al. 2022 (CKA outlier sensitivity) · Kriegeskorte et al. 2008 (RSA).
