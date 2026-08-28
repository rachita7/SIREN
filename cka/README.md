# CKA analysis: do the four safety-neuron methods find the same representation?

The four methods select almost disjoint neurons. The question this folder
answers is whether their *populations* nevertheless encode the same
information. CKA is the right tool for that, because it compares two
representations of the same prompts without requiring the individual dimensions
to correspond.

Methods compared (all on Llama-3-8B-Instruct):

| key | method | paper |
|---|---|---|
| `siren` | SIREN | Jiao et al., *LLM Safety From Within: Detecting Harmful Content with Internal Representations* |
| `wang`, `wang_robust` | Wang | *Neuron-Level Safety Alignment for LLMs* |
| `zhao_topk`, `zhao_eps` | Zhao | *Understanding and Enhancing Safety Mechanisms* |
| `yang_rms`, `yang_refusal`, `yang_harmfulness` | Yang | Yang, Sondej, Mayne, Lee & Mahdi (EMNLP 2025), *How Does DPO Reduce Toxicity? A Mechanistic Neuron-Level Analysis* |

---

## Assessment of the proposed plan

### What was right

**The activation definition is correct and, importantly, it is forced.** All
four selections index the same coordinate space, verified against every file in
`results/`: layers span 0–31 and neuron indices span 0–14335, i.e. the input to
`model.layers[l].mlp.down_proj`, which is exactly
`SiLU(W_gate x) ⊙ (W_up x)`. This is not a modelling choice you get to make —
it is what the saved indices mean. It also happens to be the tensor
`utils/model_hooks.py` already captures as `mlpneuron_mean`, so the analysis
sits in the same space SIREN's own probes were fit in. `cka/extract_activations.py`
reuses that hook definition.

**One shared pooling rule for all methods is right,** and mean pooling is the
correct primary choice since SIREN's selection used `mlpneuron_mean`.
`--pooling last` is implemented as a robustness check.

**Class-residualization is a genuinely important control,** and the intuition
about the two outcomes (agreement only on the coarse harmful/benign axis vs.
agreement on within-class structure) is exactly the right framing.

### What was wrong or badly underweighted

**1. The random control is the entire experiment, not a caveat.** This is the
big one. All four methods select subsets of *one* population of 32 × 14336 =
458,752 neurons in a single model. Two arbitrary subsets of that population
already share the model's global variance structure, so they will score a high
CKA before anyone mentions safety. The smoke test in this folder demonstrates
the failure mode concretely: on synthetic activations with a shared global
factor, **layer-matched random neuron sets score CKA = 0.997**. A reported
"CKA(SIREN, Wang) = 0.85" under those conditions would be a null result
misread as a discovery. So every number here is reported against a
layer-matched random null over 20 seeds, and the headline quantity is a
normalized score, not a raw CKA.

**2. Prompt length is a confound at least as serious as the class label.** Mean
pooling divides by the token count, so length leaks into every neuron and is
often a leading principal component of the pooled representation. HarmBench and
Alpaca prompts differ systematically in length, and so do the harmful and
benign halves of essentially every safety benchmark. Removing class means does
not remove this. So the strictest variant here, `class+length`, projects out
class dummies *and* a polynomial in the token count. `extract_activations.py`
prints the per-class token-count gap so you can see how big the problem is on
your data.

**3. There is no ceiling reference in the plan, so "high" is undefined.** CKA
between two representations of the same 2000 prompts is never going to be 1.0
even for genuinely equivalent populations. This folder computes the ceiling
empirically: split one method's own selection into two disjoint halves and
measure CKA between them. That is what "same information, same procedure"
scores on this data. Cross-method CKA is then reported as
`(observed − null) / (ceiling − null)`: 0 means indistinguishable from random
neurons, 1 means as similar as a method is to itself.

**4. The 32 × 32 cross-layer heatmap, as proposed, would have produced a
spurious finding.** Layer *l* and layer *m* of a residual network are
intrinsically correlated — consecutive layers add small increments to a shared
stream — so *any* neuron subset of layer *l* scores high against *any* subset
of a nearby layer *m*. A raw heatmap shows a broad diagonal band no matter what
the methods did, and the hoped-for result ("SIREN layer 7 aligns with Yang
layer 25") is far more likely to be the model's own layer geometry than a fact
about the selections. `run_cross_layer.py` therefore recomputes the whole
32 × 32 map with layer-matched random neurons *per layer pair* and reports the
z-score of the excess. The raw panel is still plotted, next to the z panel, so
you can see how misleading it is.

**5. Yang's selection is degenerate per layer** and would have produced
garbage cells. `yang_rms` at N=2500 puts 1641 of its 2500 neurons in layer 31
and only 2–4 neurons in each of layers 14–22. CKA from 2 neurons is noise.
`--min_neurons` (default 10) masks those cells. Worth knowing before you
interpret anything: Yang's method is concentrated almost entirely in the last
three layers, whereas SIREN, Wang and Zhao spread across all 32.

**6. Use the unbiased HSIC estimator for headline numbers.** The biased
estimator that the proposed snippet implements inflates when the prompt count
is small, and it inflates by different amounts for matrices of different width
— which matters here because the methods have different per-layer structure.
Both are reported (`cka` and `cka_unbiased`); if they disagree, trust the
unbiased one and add prompts.

**7. CKA is known to be dominated by a handful of very-high-variance
directions** (Davari et al. 2022, *Reliability of CKA as a Similarity Measure*).
On Llama-3 that is a live hazard, because the model has massive-activation
outlier neurons. Two consequences: per-neuron z-scoring is not cosmetic, it
changes the answer (so it is the default, with `--no_zscore` as a sensitivity
check), and a second, rank-based measure is reported alongside — Spearman RSA
on the prompt-similarity matrices, which cannot be driven by outlier scaling.
If CKA is high and RSA is not, the CKA is an artifact.

**8. It is not a 4 × 4 comparison unless you choose canonical variants.**
`results/` holds 8 selections: two Wang variants, two Zhao variants, three Yang
rankings. The default is one per method (`siren wang zhao_topk yang_rms`), but
`--methods all` gives the full 8 × 8 — and the *within*-method comparisons
there are a useful extra reference. If Wang vs. Wang-robust only reaches 0.6,
then 0.6 across methods is high.

**9. Two smaller things.** Dead/near-constant neurons must be dropped before
z-scoring or they become pure amplified noise (handled, and the count is
printed). And 1000 prompts is too few: with a 2500-neuron budget the
prompt-similarity matrix is rank-limited by the prompt count, which inflates
biased CKA. Use ≥ 2000.

### Held-out data

You selected the neurons on HarmBench + Alpaca (`data_files/`), so the
recommendation to use different data is right. The defaults:

- **`wildguard`** (primary) — WildGuard test split, prompt-level harm labels,
  diverse, both classes, disjoint from HarmBench/Alpaca.
- **`xstest`** (stress test) — safe prompts that *look* harmful plus genuinely
  unsafe ones. This is the more valuable of the two for your question: because
  surface form is decoupled from the label, a high CKA here cannot be explained
  away as "all four methods encode harmful-sounding wording".

| dataset | rows | classes | HF access |
|---|---|---|---|
| `wildguard` | 1,725 | both | **gated** — accept terms + token |
| `xstest` | 450 | both | ungated |
| `aegis2` | 1,964 | both | **gated** |
| `toxic_chat` | 5,082 | both | **gated** |
| `openai_moderation` | 1,680 | both | ungated |
| `beavertails` | 3,021 | both (response-level labels) | ungated |
| `advbench` | 520 | harmful only | ungated |

`build_prompts.py` loads these directly rather than through
`train/preprocess.py`, for two reasons. `preprocess.py` splits every corpus
into train/val/test for probe training, but none of these corpora were used to
select any method's neurons, so the whole dataset is held out and splitting it
just discards prompts — for XSTest that would have left 90 prompts, far too few.
And `preprocess.py`'s wildguard path also loads `wildguardtrain` (~87k rows),
a large download of a gated config with no use here.

Prompts are wrapped in the Llama-3 chat template with the backbone's own
tokenizer, reproducing the `formatted_input` column of `data_files/*.csv`, so
the activations sit in the same regime the neurons were selected under.

---

## Setup

Everything needed is already in the repo's conda environment:

```bash
conda activate siren
```

### HuggingFace access

The backbone (`meta-llama/Meta-Llama-3-8B-Instruct`) is gated, and so are the
two recommended prompt sets. One-time setup:

1. While logged in to HuggingFace, click *Agree and access* on each page you
   need: [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct),
   [allenai/wildguardmix](https://huggingface.co/datasets/allenai/wildguardmix).
   XSTest needs nothing.
2. Authenticate on the cluster:

```bash
huggingface-cli login          # or: export HF_TOKEN=hf_...
```

3. Confirm access before queueing a GPU job:

```bash
python -c "from datasets import load_dataset; \
d=load_dataset('allenai/wildguardmix','wildguardtest')['test']; \
print(len(d), d.column_names)"
```

If that fails, `build_prompts.py` prints the fix and lists the ungated
alternatives; nothing else in the pipeline cares which held-out set you use.
On ETH Euler, run `build_prompts.py` on a **login node** — compute nodes may not
reach the Hub, and the resulting CSV is all the GPU step needs.

Verify the pipeline without a GPU or any downloads (~25 s). This checks the CKA
implementation against cases with known answers, the neuron loaders against all
8 selections × 3 budgets, and runs both analysis scripts end to end on
synthetic activations:

```bash
python cka/smoke_test.py
```

Inspect what is in `results/` before running anything expensive:

```bash
python cka/neuron_sets.py --budget 2500
```

This prints each method's size and layer profile and the pairwise index-level
Jaccard — the near-zero overlap that motivates the whole analysis.

## Runtime

Wall-clock for the default configuration (2 datasets, 2000 prompts each,
N=2500, 4 methods, 20 null seeds), on one 24 GB GPU:

| step | time | hardware |
|---|---|---|
| 1. build prompts | 1–5 min per dataset (mostly download) | CPU, needs network |
| 2. extract activations | 5–10 min per dataset (incl. ~2 min model load) | GPU |
| 3. cross-method CKA | 5–10 min per dataset | CPU |
| 4. cross-layer CKA | 10–20 min per dataset | CPU |
| **total** | **≈ 45–75 min for both datasets** | |

The 6 h wall time in `cka.sbatch` is deliberate headroom. Things that change
this materially: `--budget 10000` roughly triples steps 3–4; `--methods all`
takes 28 pairs instead of 6, so ~4× on steps 3–4 (and cross-layer becomes ~1 h+);
`--null_seeds 100` for publication figures roughly quintuples step 3. Steps 3
and 4 reread the saved activations, so you can iterate on them freely without
re-running the GPU step.

## Running it

The short version, from the repo root:

```bash
bash cka/run_all.sh
```

or on SLURM:

```bash
sbatch cka/cka.sbatch
```

If your compute nodes are offline, build the prompt sets on a login node first
(`python cka/build_prompts.py --dataset wildguard`).

### Step by step

**1. Build a held-out prompt set** (CPU, needs HF access):

```bash
python cka/build_prompts.py --dataset wildguard --max_prompts 2000
python cka/build_prompts.py --dataset xstest   --max_prompts 2000
```

Writes `cka/prompts/{dataset}.csv`, class-balanced and chat-templated.

**2. Extract activations** (GPU, ~5–10 min for 2000 prompts on a 24 GB card):

```bash
python cka/extract_activations.py \
    --prompts cka/prompts/wildguard.csv \
    --pooling mean --batch_size 8
```

Writes `cka/activations/wildguard_mean.npy`, a float16
`[N, 32, 14336]` array (~1.8 GB), plus a `.meta.csv` with labels and token
counts. Every neuron is stored, not just the selected ones, so all downstream
analysis — any method, any budget, any random seed, the full cross-layer sweep
— reruns on CPU without touching the GPU again.

**3. Cross-method CKA** (CPU, ~5–15 min):

```bash
python cka/run_cka.py --activations cka/activations/wildguard_mean.npy
```

**4. Cross-layer CKA** (CPU, ~10–20 min):

```bash
python cka/run_cross_layer.py \
    --activations cka/activations/wildguard_mean.npy \
    --variant class+length
```

### Useful variations

```bash
# All 8 selections, so within-method variants give an extra reference point
python cka/run_cka.py --activations ... --methods all

# Does the conclusion depend on the neuron budget?
for N in 2500 5000 10000; do python cka/run_cka.py --activations ... --budget $N; done

# Sensitivity to the z-score, given Llama-3's outlier neurons
python cka/run_cka.py --activations ... --no_zscore --label wildguard_nozscore

# Robustness to the pooling rule
python cka/extract_activations.py --prompts cka/prompts/wildguard.csv --pooling last
python cka/run_cka.py --activations cka/activations/wildguard_last.npy

# Publication-quality nulls (slower)
python cka/run_cka.py --activations ... --null_seeds 100 --ceiling_seeds 20
```

## Output

Everything lands in `cka/results/`:

| file | contents |
|---|---|
| `cka_{tag}_N{budget}.csv` | one row per (variant, method pair): `cka`, `cka_unbiased`, `rsa_spearman`, both nulls with std, ceiling, `z_vs_null`, `normalized`, `jaccard` |
| `cka_{tag}_N{budget}.json` | the same plus per-method ceilings and run configuration |
| `cka_matrix_{tag}_N{budget}_{variant}.png` | observed / null / normalized matrices side by side |
| `cka_pairs_{tag}_N{budget}_{variant}.png` | observed vs. null bars with the ceiling marked |
| `cka_variants_{tag}_N{budget}.png` | normalized score across the three residualizations |
| `layer_profiles_{tag}_N{budget}.png` | each method's layer distribution |
| `crosslayer_{...}_{a}_vs_{b}.png` | raw 32×32 CKA next to the null-normalized z map |
| `crosslayer_{...}_{a}_vs_{b}_{cka,zscore}.csv` | the underlying matrices |
| `crosslayer_long_{...}.csv` | tidy per-(layer, layer) rows for your own plotting |

## Reading the results

The column to look at is **`normalized`**, not `cka`:

- **≈ 0** — the selected neurons are no more similar to each other than random
  neurons from the same layers. The methods genuinely find different things,
  and the near-zero Jaccard is the whole story.
- **≈ 1** — as similar as two disjoint halves of one method's own selection.
  The populations are representationally interchangeable despite selecting
  almost disjoint neurons. This is the interesting result.
- **intermediate** — partial overlap in what is encoded; report the number
  rather than a verdict.

Then read across the three variants:

- high on `raw`, collapses on `class` → the methods only agree on the coarse
  harmful-vs-benign distinction. Real, but weak.
- still high on `class+length` → they agree on fine-grained structure *within*
  each class that is not prompt length. This is the strong claim, and the one
  worth writing up.

Two consistency checks before you believe any of it: `cka` and `cka_unbiased`
should agree (if not, add prompts), and `rsa_spearman` should move with `cka`
(if not, a few outlier neurons are driving the CKA).

For the cross-layer maps, read the **z panel, not the raw panel**. The raw
panel will show a diagonal band regardless. Off-diagonal z peaks are the
finding worth chasing: the same structure recovered at different network
depths.

## Troubleshooting

**`RuntimeWarning: overflow/divide-by-zero encountered in matmul` on macOS.**
Spurious. NumPy 2.x with Apple's Accelerate BLAS raises bogus floating-point
warnings on large matmuls; the results are exact (the smoke test's
`CKA(X, X) == 1.000000` checks run through the same code path). It does not
occur on the Linux cluster. Silence it with `python -W ignore::RuntimeWarning`.

**Out of memory in `run_cka.py`.** Peak usage is roughly
`num_methods × num_prompts × budget × 4 bytes × 3`. At the defaults that is
under 1 GB, but `--methods all --budget 10000` needs ~10 GB. Reduce the budget,
the method count, or pass `--skip_rsa` (RSA is the one step that materializes
an N × N matrix).

**Gated HuggingFace datasets.** `wildguard`, `aegis2` and `toxic_chat` require
accepting terms on the Hub plus a token (see *HuggingFace access* above). Use
`xstest`, `openai_moderation` or `beavertails` instead — the analysis does not
care which held-out set you use, only that the four methods did not select on it.

**`FileNotFoundError: siren @ N=2500: missing results/rachita_neurons/...`**
The repo's `.gitignore` has a blanket `*.json` rule, which silently excluded
SIREN's exported neuron JSONs while the other methods' CSVs were committed. Now
fixed with a `!results/**/*.json` negation. If you hit it on an older checkout,
`git pull` and confirm with `python cka/neuron_sets.py`.

**CUDA out of memory during extraction.** Drop to `--batch_size 4`. Activations
are pooled inside the hook rather than after the forward pass, so peak memory
is ~3.7 GB lower than the extractor in `utils/model_hooks.py`, but batch 8 at
512 tokens next to the 16 GB bf16 model is still a snug fit on a 24 GB card.

## Files

| file | role |
|---|---|
| `neuron_sets.py` | loaders for all 8 selections; layer-matched and global nulls; disjoint-half splits; matrix construction. Runnable for a summary. |
| `cka_core.py` | linear CKA (biased and unbiased HSIC), Spearman RSA, residualization, preprocessing, normalized scores |
| `build_prompts.py` | held-out, class-balanced, chat-templated prompt sets |
| `extract_activations.py` | `down_proj`-input activations for all 32 × 14336 neurons |
| `run_cka.py` | cross-method analysis with all controls |
| `run_cross_layer.py` | 32 × 32 cross-layer analysis with per-layer-pair nulls |
| `plots.py` | figures |
| `smoke_test.py` | correctness checks and an end-to-end synthetic run |
| `run_all.sh`, `cka.sbatch` | drivers |

## References

- Kornblith, Norouzi, Lee & Hinton (2019). *Similarity of Neural Network
  Representations Revisited.* ICML. — linear CKA.
- Song, Smola, Gretton, Bedo & Borgwardt (2012). *Feature Selection via
  Dependence Maximization.* JMLR. — unbiased HSIC estimator.
- Davari, Horoi, Natik, Lajoie, Wolf & Belilovsky (2022). *Reliability of CKA as
  a Similarity Measure in Deep Learning.* — the outlier-direction sensitivity
  that motivates z-scoring and the RSA cross-check.
- Kriegeskorte, Mur & Bandettini (2008). *Representational similarity
  analysis.* — the RSA measure.
