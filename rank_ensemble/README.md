# Rank-ensemble of identified safety neurons

Does combining the **exact ranks** of the safety neurons already identified
by several methods produce a more causally effective safety-neuron set than
any single method, at the same neuron budget N?

This folder is a **new experiment**. It does not modify or reuse the
tier-based approximation in `ensemble/`.

Two sources of data, with different jobs:

| Source | Role |
|---|---|
| Full-ranking CSV | Exact rank `r_m(j)` of every FFN neuron under method `m` |
| Existing top-N safety-neuron file | Eligibility: the neurons method `m` actually calls safety neurons at budget N |

**A neuron may enter the ensemble at budget N only if it appears in at least
one method's existing top-N safety-neuron file.** Average rank over the
full universe is not enough.

```
C_N = topN_method1 ∪ … ∪ topN_methodK
ensemble_N = N lowest-scoring (or quota-selected) neurons from C_N
```

Default budgets are 459 / 2294 / 4588 / 9175 (0.1 / 0.5 / 1 / 2 % of
32 × 14336 = 458,752 Llama-3-8B FFN neurons). They are **not** hard-coded:
pass `--budgets 459 2294 4588 9175` or any other N for which the top-N
files exist. Method identities live in `methods.json` (JSON so the
default `siren` env does not need PyYAML; `--config foo.yaml` still
works if PyYAML is installed).

---

## 1. Full-ranking files

Configured in `methods.json` (`ranking.path`). Current defaults, all
relative to the SIREN repo root:

| Method id | File | Rank column | Neuron columns | Score column |
|---|---|---|---|---|
| `siren` | `results/siren_mlpneuron_full_ranking.csv` | `global_rank` | `layer`, `neuron_index` | `abs_weight` |
| `yang_harmfulness` | `results/yang_full_ranking_delta_harmfulness.csv` | `rank` | `layer`, `within_layer_index` | `abs_score` |
| `yang_refusal` | `results/full_ranking_delta_refusal.csv` | `rank` | `layer`, `within_layer_index` | `abs_score` |
| `zhao_topk` | `results/full_ranking_neurons_zhao_topk_matching.csv` | `rank` | `layer`, `neuron_index` | `score` |
| `zhao_relative_epsilon` | `results/full_ranking_neurons_zhao_relative_epsilon_matching.csv` | `rank` | `layer`, `neuron_index` | `score` |
| `wang` | `results/full_ranking_neurons_wang_matching.csv` | `rank` | `layer`, `neuron_index` | `score` |
| `wang_robust` | `results/full_ranking_neurons_wang_robust_matching.csv` | `rank` | `layer`, `neuron_index` | `score` |

**Rank 1 = highest-ranked.** Formats are not assumed identical; column
names are read from `methods.json`. Internally every neuron is `(layer, neuron_index)`.
Yang's `within_layer_index` is that neuron index.

SIREN's official order is `global_rank` (per-layer entry-threshold), **not**
a global sort of `abs_weight`.

All seven current rankings contain the same 458,752 `(layer, neuron_index)`
pairs, each exactly once, with unique ranks `1 … 458752`.

---

## 2. Top-N safety-neuron files

Configured in `methods.json` (`topn.path`), `{n}` substituted for the budget.

| Method id | File | Format |
|---|---|---|
| `siren` | `results/rachita_neurons/llama3-8b-instruct_mlpneuron_mean-std-mlpneuron_mean-clean_selected_neurons_top{n}.json` | `{"layer0": [idx, …], …}` |
| `yang_harmfulness` | `results/tengerleg_neurons/neurons_delta_harmfulness_N{n}.csv` | `variant,layer,neuron_index` |
| `yang_refusal` | `results/tengerleg_neurons/neurons_delta_refusal_N{n}.csv` | same |
| `zhao_topk` | `results/svea_neurons/fulltest_neurons_zhao_topk_N{n}.csv` | same |
| `zhao_relative_epsilon` | `results/svea_neurons/fulltest_neurons_zhao_relative_epsilon_N{n}.csv` | same |
| `wang` | `results/svea_neurons/fulltest_neurons_wang_N{n}.csv` | same |
| `wang_robust` | `results/svea_neurons/fulltest_neurons_wang_robust_N{n}.csv` | same |

Available budgets today: **459, 2294, 4588, 9175**. Zhao's published files
are a few rows off N at some cutoffs because of score ties
(`zhao_topk` 2298 / 4590; `zhao_relative_epsilon` 2293). Those sets still
match the ranking prefix of the same length. The ensemble still emits
**exactly N** neurons; `C_N` uses the published file.

---

## 3. Ranking vs eligibility

- **Full ranking** = exact rank information used to score / order neurons.
- **Top-N file** = the method's identified safety neurons at that N.
  Only these neurons (across methods) may enter `C_N`.

---

## 4. Candidate pool

```
C_N = ⋃_m  { existing top-N safety neurons of method m }
```

Example: at N=2294, `C_N` is the union of the seven existing top-2294
files. Every selected ensemble neuron at that N is in `C_N`.

---

## 5. Ensemble A — exact-rank consensus

For every neuron `j` and method `m`:

```
p_m(j) = (r_m(j) − 1) / (M_m − 1)
consensus_score(j) = mean_m p_m(j)
```

`M_m` is the number of neurons ranked by method `m`. All seven current
rankings have `M_m = 458752`, so the denominator is shared. Lower score =
ranked more highly overall.

A global consensus table (`results/consensus_scores_universe.csv`, ~100 MB)
can be written with `--write-global-consensus` for diagnostics. **The
ensemble is not selected from that table.**

At each N:

1. Build `C_N` from the top-N files.
2. Look up each candidate's exact rank under every full ranking.
3. Sort `C_N` by `consensus_score` (tie-break: `layer`, `neuron_index`).
4. Take the N lowest-scoring candidates.

Output: `selections/rank_consensus_N{n}.csv` and the full ranked pool
`selections/candidate_pool_N{n}.csv`.

Columns include `layer`, `neuron_index`, `rank_<method>`,
`percentile_<method>`, `consensus_score`, `methods_that_selected_it_at_N`.
There is no tier 1/2/3/4/5 column.

---

## 6. Ensemble B — quota / round-robin

At budget N, method `m` may propose **only** from its own existing top-N
file, ordered by that method's full ranking. It never continues down the
full ranking after that file is exhausted.

Deterministic method order = order in `methods.json`.

1. Method 1 proposes its highest-ranked unused neuron from its top-N set.
2. Method 2 does the same.
3. … through every configured method, then repeat.

If the proposed neuron is already in the ensemble, that proposal is a
**duplicate**: skip it and immediately continue down **that method's**
top-N list (the method does not lose its turn). If a method has no unused
top-N neurons left, drop it from the round-robin and keep going with the
rest. Stop at exactly N unique neurons.

Every selected neuron still belongs to at least one original top-N set.
Unfilled quota slots are redistributed automatically.

Recorded per neuron: `(layer, neuron_index)`, `entered_via` (which method
caused it to enter), exact ranks under all methods, `also_in` (other
methods' top-N sets that contain it), `selection_step`.

Also reported: neurons contributed by each method, skipped duplicates,
whether any method exhausted its top-N list.

Output: `selections/quota_N{n}.csv`.

---

## 7. Duplicate handling

- **Consensus:** unique `(layer, neuron_index)`; `C_N` is a set.
- **Quota:** a later method proposing an already-selected neuron is a
  skipped duplicate; that method immediately offers its next unused
  top-N neuron.
- Rankings themselves must contain no duplicate neurons (validated).

---

## 8. Configurable budgets

```bash
python rank_ensemble/build_ensembles.py --budgets 459 2294 4588 9175
python rank_ensemble/build_ensembles.py --budgets 1000   # needs top-1000 files
```

`--methods` restricts which configured methods participate. Adding a
method is a `methods.json` edit, not an aggregator rewrite.

---

## 9. Validation (rankings vs top-N files)

```bash
python rank_ensemble/validate_rankings.py
```

Writes `results/validation_report.md`. For every method and N it checks
that the **first N neurons of the full ranking**, as a set, equal the
existing top-N safety-neuron file. `build_ensembles.py` stops if they
disagree.

Wang / Zhao use the corrected `*_matching.csv` rankings. Score ties are
recorded; percentiles use the assigned unique rank.

---

## 10. Ablation mechanism (from Repo B)

Reproduced from
[`safety-neurons-where-you-look`](https://github.com/SveaReuter/safety-neurons-where-you-look)
`src/eval/ablation_validation.py`.

A selected neuron is an **input channel of `mlp.down_proj`** at that
layer — the gated FFN activation `SiLU(gate_proj(x)) * up_proj(x)`, width
14336. A `forward_pre_hook` clones that input, zeros the selected
channels, and returns the modified args. That is equivalent to zeroing
the corresponding `down_proj` weight column (removing the neuron's
residual-stream contribution) without changing gate/up computation.

- Registered on `decoder.layers[l].mlp.down_proj` before the condition
  runs; removed after.
- Fires on **every forward pass**, so every generated token and the
  single-token MMLU logit read see the same zeros.
- AdvBench and MMLU share the same hooks.
- `(layer, neuron_index)` → flat index `layer * 14336 + neuron_index`.
- Repo B used a Peft-wrapped DPO model (`model.model.model.layers`).
  The SIREN default is plain Instruct (`model.model.layers`).
  `ablation.decoder_layers()` handles both.

The ensemble CSV simply lists the N `(layer, neuron_index)` pairs that
undergo this same operation.

---

## 11. AdvBench evaluation

Reproduced from Repo B `ablation_validation_advbench.py` /
`run_ablation_pilot_target.py`.

- Local cache: `rank_ensemble/data/advbench_harmful_behaviors.csv`
  (520 rows from [llm-attacks](https://github.com/llm-attacks/llm-attacks);
  fetched automatically if missing).
- Column: `goal`.
- 100 prompts, `random.seed(EVAL_SEED=4242)` then `random.sample`.
- Chat template: user turn + generation prompt.
- `max_new_tokens=64`, greedy (`do_sample=False`).
- Detector: `refusal_labels.is_refusal` (Repo B's keyword list, first 200
  characters). **Not** a new heuristic.
- Metric: **AdvBench refusal rate**.

---

## 12. MMLU evaluation

Reproduced from Repo B `run_ablation_pilot_target.py`.

- `datasets.load_dataset("cais/mmlu", "all", split="test")`.
- **2000 questions**, shuffled with `EVAL_SEED=4242`. Repo B's CLI
  default is 500; the teammate's matched runs used 2000
  (`run_zhao_relative_epsilon_ablation.py`).
- Prompt: question + `A./B./C./D.` choices + `Answer:` — **no** chat
  template (lm-eval-harness convention).
- Next-token logits of `" A"`, `" B"`, `" C"`, `" D"`; argmax vs the
  integer gold answer.
- Metric: **MMLU accuracy**. Not generation-based multiple choice.

---

## 13. Random control

Repo B **does** have a random baseline:
`src/build_ablation_pilot_targets.py` builds `random_{n}` as

```python
torch.randperm(num_layers * intermediate_size, generator=Generator().manual_seed(1000))[:n]
```

(`RANDOM1_SEED = 1000`). This folder reproduces that draw. It is a
uniform sample from the valid FFN universe, not layer-matched. Written to
`selections/random_N{n}.csv` and `results/targets_N{n}/random.json`.

---

## 14. Commands — build ensembles

```bash
conda activate siren
python rank_ensemble/smoke_test.py
python rank_ensemble/validate_rankings.py
python rank_ensemble/build_ensembles.py
python rank_ensemble/build_ensembles.py --budgets 459 2294 4588 9175
```

Individual builders:

```bash
python rank_ensemble/build_rank_consensus.py --budgets 2294
python rank_ensemble/build_quota.py --budgets 2294
```

---

## 15. Commands — run ablations

One GPU job evaluates, at a fixed N: unablated, each configured method's
existing top-N, rank-consensus, quota, and random. Same model, hooks,
prompt sample, seed, decoding, detector, and datasets. **Only the neuron
set changes.**

```bash
python rank_ensemble/run_ablation_target.py --budget 2294
python rank_ensemble/run_ablation_target.py --budget 2294 --target-name rank_consensus
BUDGETS="459 2294 4588 9175" sbatch --export=ALL,BUDGETS rank_ensemble/rank_ensemble.sbatch
```

Optional DPO adapter (Repo B's cluster path is not required at runtime):

```bash
python rank_ensemble/run_ablation_target.py --budget 2294 \
    --lora-adapter /path/to/dpo_hh_run4/final_adapter
```

Results: `rank_ensemble/results/N{n}/target_{name}/result.json`.
Already-finished targets are skipped (resume-safe).

On offline compute nodes, cache MMLU on a login node first:

```bash
python -c "from datasets import load_dataset; load_dataset('cais/mmlu','all',split='test')"
```

---

## 16. Commands — aggregate and plot

```bash
python rank_ensemble/analyze_results.py --budgets 459 2294 4588 9175
```

Writes `results/summary_N{n}.csv`, `results/summary.md`,
`refusal_vs_N.png`, `mmlu_vs_N.png`, `refusal_vs_mmlu.png` (and per-N
planes). No extra metrics beyond AdvBench refusal and MMLU accuracy.

### Runtime

Building ensembles is CPU-only and takes about **20 seconds**.

The GPU ablation is the long step. Repo B evaluates one prompt at a time
(100 AdvBench generations + 2000 MMLU forwards) per neuron set. On a
24 GB card expect **about 5–10 minutes per condition**.

| What you run | Conditions | Rough wall time |
|---|---:|---|
| One budget (default `N=2294`) | 11 (unablated + 7 methods + consensus + quota + random) | **1.5–2.5 h** |
| All four default budgets | 44 (baseline is re-run per N) | **6–10 h** |

`run_ablation_target.py` skips any `target_*/result.json` that already
exists, so a killed job can be resubmitted. Warm the MMLU cache on a
login node first if compute nodes are offline.

Suggested order: run `N=2294` first. If unablated refusal is high and
MMLU is ~0.6, submit the other three budgets.

### How to read the results

1. Open `results/summary.md` (or the per-N CSV). Compare **only inside
   one N**. Do not compare SIREN at 459 against consensus at 2294.
2. Look at `refusal_vs_mmlu.png`. The interesting region is **lower
   refusal, MMLU still near unablated**.
3. A strong ensemble, at the same N as the single methods:
   - AdvBench refusal drops **at least as much as the best single method**
   - MMLU stays close to the unablated point (and well above chance 0.25)
4. If refusal goes to ~0 and MMLU is near 0.25, the model collapsed.
   That is not a better safety-neuron set — read it the same way as
   random / Wang-style collapse.
5. `results/composition_N{n}.csv` is diagnostic only (pool size, Jaccard,
   layer counts, quota duplicates). It is not the causal answer.

Read refusal **together with** MMLU. A selection that drives refusal to
zero only because general capability collapses is not a better
safety-neuron set.

---

## 17. What was reproduced from Repo B

Inspected, not modified:
https://github.com/SveaReuter/safety-neurons-where-you-look

| Component | Repo B file | Repo A file |
|---|---|---|
| Ablation hook (zero `down_proj` input) | `src/eval/ablation_validation.py` `make_zero_input_hook`, `register_ablation_hooks` | `ablation.py` |
| Chat formatting | `ablation_validation.format_chat` | `ablation.py` |
| AdvBench load + sample | `ablation_validation_advbench.load_advbench_prompts` | `ablation.py` |
| Greedy generation | `ablation_validation_advbench.generate_one` | `ablation.py` |
| Refusal detector | `src/refusal_labels.py` `is_refusal` | `refusal_labels.py` |
| MMLU prompt / logits | `run_ablation_pilot_target.py` | `ablation.py` |
| Driver + `result.json` | `src/eval/run_ablation_pilot_target.py` | `run_ablation_target.py` |
| Random-N | `src/build_ablation_pilot_targets.py` (`RANDOM1_SEED=1000`) | `aggregators.random_pairs` |
| EVAL_SEED | 4242 | `methods.json` / `ablation.EVAL_SEED` |

Repo B's `load_model_and_tokenizer` hard-codes a cluster DPO LoRA. The
port defaults to `meta-llama/Meta-Llama-3-8B-Instruct` (the model the
SIREN selections were built on) and accepts `--lora-adapter`. Across
conditions the checkpoint is shared.

---

## Scientific question

At the same neuron budget N, does ensembling the safety neurons identified
by the configured methods produce a set that is more causally important
for safety behaviour than the individual methods, while remaining
selective to safety rather than general capability?

A strong ensemble: substantial drop in AdvBench refusal after ablation,
comparatively preserved MMLU. Compare only matched N (never N=459 vs
N=2294). Composition / Jaccard / layer histograms in
`results/composition_N{n}.csv` are diagnostics, not the primary
evaluation.

Sets available at each N: each method's existing top-N, rank-consensus
top-N, quota top-N, random-N, unablated baseline.

---

## Layout

```
rank_ensemble/
  methods.json              # methods, paths, default budgets, seeds
  config.py
  load_rankings.py
  validate_rankings.py
  aggregators.py            # consensus, quota, random, composition
  build_rank_consensus.py
  build_quota.py
  build_ensembles.py
  refusal_labels.py         # Repo B detector
  ablation.py               # Repo B hooks + AdvBench + MMLU
  run_ablation_target.py
  analyze_results.py
  smoke_test.py
  run_all.sh
  rank_ensemble.sbatch
  data/advbench_harmful_behaviors.csv
  selections/
  results/
```
