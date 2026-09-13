# Rank-ensemble validation report

Full rankings supply exact ranks. Existing top-N safety-neuron
files supply eligibility. This report checks both, and checks
whether the first N neurons of each ranking reproduce that
method's published top-N set.

- Config: `/Users/rachitaagrawal/Code/SIREN/rank_ensemble/methods.json`
- Methods: siren, yang_harmfulness, yang_refusal, zhao_topk, zhao_relative_epsilon, wang, wang_robust
- Expected neuron universe: 32 layers × 14336 = 458752
- Budgets checked: [459]
- Rank convention: **1 = highest-ranked**

## 1. Full-ranking integrity

| Method | N ranked | Unique | Dup neurons | Tied scores (values / neurons) | Score monotone ↓ | Issues |
|---|---:|---:|---:|---:|---|---|
| SIREN (`siren`) | 458752 | 458752 | 0 | 1998 / 4002 | no | none |
| Yang (harmfulness) (`yang_harmfulness`) | 458752 | 458752 | 0 | 1435 / 2873 | yes | none |

Score ties are recorded here. Every ranking already assigns a
unique integer rank, so percentile ranks use that assigned rank
rather than re-breaking ties.

SIREN's `global_rank` follows per-layer entry-threshold order,
not a global sort of `abs_weight`. That is why SIREN's score
column is not monotone in rank. Rank 1 is still the official
highest-ranked neuron.

## 2. Shared neuron universe

All configured methods rank **exactly the same** 458752 `(layer, neuron_index)` pairs.

## 4. Ranking prefix vs existing top-N safety-neuron files

For each method and budget N, the ranking prefix of length
`|top-N file|` is compared as a **set** to that file.
If the published file is a few neurons off N (ties at the
cutoff), the file must still be an exact prefix of the ranking.

| Method | N | \|rank prefix\| | \|top-N file\| | size−N | Jaccard | only in ranking | only in top-N | Match |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| siren | 459 | 459 | 459 | +0 | 1.0000 | 0 | 0 | yes |
| yang_harmfulness | 459 | 459 | 459 | +0 | 1.0000 | 0 | 0 | yes |

Every configured method's ranking prefix reproduces its
published top-N file at every checked budget.

## 5. Verdict

PASS: rankings are well-formed, share one universe, and
agree with the existing top-N safety-neuron files.
