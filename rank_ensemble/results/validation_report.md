# Rank-ensemble validation report

Full rankings supply exact ranks. Existing top-N safety-neuron
files supply eligibility. This report checks both, and checks
whether the first N neurons of each ranking reproduce that
method's published top-N set.

- Config: `/Users/rachitaagrawal/Code/SIREN/rank_ensemble/methods.json`
- Methods: siren, yang_harmfulness, yang_refusal, zhao_topk, zhao_relative_epsilon, wang, wang_robust
- Expected neuron universe: 32 layers × 14336 = 458752
- Budgets checked: [459, 2294, 4588, 9175]
- Rank convention: **1 = highest-ranked**

## 1. Full-ranking integrity

| Method | N ranked | Unique | Dup neurons | Tied scores (values / neurons) | Score monotone ↓ | Issues |
|---|---:|---:|---:|---:|---|---|
| SIREN (`siren`) | 458752 | 458752 | 0 | 1998 / 4002 | no | none |
| Yang (harmfulness) (`yang_harmfulness`) | 458752 | 458752 | 0 | 1435 / 2873 | yes | none |
| Yang (refusal) (`yang_refusal`) | 458752 | 458752 | 0 | 1494 / 2992 | yes | none |
| Zhao (top-k) (`zhao_topk`) | 458752 | 458752 | 0 | 13962 / 458437 | no | none |
| Zhao (rel-eps) (`zhao_relative_epsilon`) | 458752 | 458752 | 0 | 84233 / 281878 | yes | none |
| Wang (`wang`) | 458752 | 458752 | 0 | 62 / 124 | yes | none |
| Wang (robust) (`wang_robust`) | 458752 | 458752 | 0 | 68 / 136 | yes | none |

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
| siren | 2294 | 2294 | 2294 | +0 | 1.0000 | 0 | 0 | yes |
| siren | 4588 | 4588 | 4588 | +0 | 1.0000 | 0 | 0 | yes |
| siren | 9175 | 9175 | 9175 | +0 | 1.0000 | 0 | 0 | yes |
| yang_harmfulness | 459 | 459 | 459 | +0 | 1.0000 | 0 | 0 | yes |
| yang_harmfulness | 2294 | 2294 | 2294 | +0 | 1.0000 | 0 | 0 | yes |
| yang_harmfulness | 4588 | 4588 | 4588 | +0 | 1.0000 | 0 | 0 | yes |
| yang_harmfulness | 9175 | 9175 | 9175 | +0 | 1.0000 | 0 | 0 | yes |
| yang_refusal | 459 | 459 | 459 | +0 | 1.0000 | 0 | 0 | yes |
| yang_refusal | 2294 | 2294 | 2294 | +0 | 1.0000 | 0 | 0 | yes |
| yang_refusal | 4588 | 4588 | 4588 | +0 | 1.0000 | 0 | 0 | yes |
| yang_refusal | 9175 | 9175 | 9175 | +0 | 1.0000 | 0 | 0 | yes |
| zhao_topk | 459 | 459 | 459 | +0 | 1.0000 | 0 | 0 | yes |
| zhao_topk | 2294 | 2298 | 2298 | +4 | 1.0000 | 0 | 0 | yes |
| zhao_topk | 4588 | 4590 | 4590 | +2 | 1.0000 | 0 | 0 | yes |
| zhao_topk | 9175 | 9175 | 9175 | +0 | 1.0000 | 0 | 0 | yes |
| zhao_relative_epsilon | 459 | 459 | 459 | +0 | 1.0000 | 0 | 0 | yes |
| zhao_relative_epsilon | 2294 | 2293 | 2293 | -1 | 1.0000 | 0 | 0 | yes |
| zhao_relative_epsilon | 4588 | 4588 | 4588 | +0 | 1.0000 | 0 | 0 | yes |
| zhao_relative_epsilon | 9175 | 9175 | 9175 | +0 | 1.0000 | 0 | 0 | yes |
| wang | 459 | 459 | 459 | +0 | 1.0000 | 0 | 0 | yes |
| wang | 2294 | 2294 | 2294 | +0 | 1.0000 | 0 | 0 | yes |
| wang | 4588 | 4588 | 4588 | +0 | 1.0000 | 0 | 0 | yes |
| wang | 9175 | 9175 | 9175 | +0 | 1.0000 | 0 | 0 | yes |
| wang_robust | 459 | 459 | 459 | +0 | 1.0000 | 0 | 0 | yes |
| wang_robust | 2294 | 2294 | 2294 | +0 | 1.0000 | 0 | 0 | yes |
| wang_robust | 4588 | 4588 | 4588 | +0 | 1.0000 | 0 | 0 | yes |
| wang_robust | 9175 | 9175 | 9175 | +0 | 1.0000 | 0 | 0 | yes |

Published Zhao top-N files are slightly off the nominal
budget at a few cutoffs (ties). Those files still match
the ranking prefix of the same length. Eligibility uses
the published file; the ensemble still emits exactly N
neurons.

Every configured method's ranking prefix reproduces its
published top-N file at every checked budget.

## 5. Verdict

PASS: rankings are well-formed, share one universe, and
agree with the existing top-N safety-neuron files.
