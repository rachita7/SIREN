# Methods write-up: representational similarity of safety-neuron selections

Report-ready description of what `cka/` does. Adapt the prose; the numbers in
brackets are the defaults, so update them if you change flags.

---

## 1. Motivation

Four independent methods localize safety-relevant computation in Llama-3-8B-Instruct
to sets of individual neurons. Their selections barely overlap: at a budget of
2,500 neurons the index-level Jaccard similarity between methods ranges from
0.024 (SIREN vs. Zhao) to 0.153 (SIREN vs. Wang). Taken at face value this
suggests the methods disagree about where safety lives.

Neuron-identity overlap is, however, a very strict criterion. Two populations
can encode the same information through different coordinates. We therefore ask
a weaker and more meaningful question: **do the selected populations induce the
same similarity structure over prompts, even where they share no neurons?**
Centered Kernel Alignment (CKA; Kornblith et al., 2019) is designed for exactly
this comparison, as it is invariant to rotation and isotropic scaling of either
representation and does not require the two representations to have equal
dimensionality.

## 2. Common activation space

All four methods index the same space, so no conversion is required. Each
selection is a set of `(layer, neuron_index)` pairs with layers in 0–31 and
indices in 0–14,335, addressing the per-neuron MLP activation of Llama-3-8B,
i.e. the input to `mlp.down_proj`:

    h_l = SiLU(W_gate x) ⊙ (W_up x) ∈ R^14336,     l = 0..31

This gives a shared population of 32 × 14,336 = **458,752 neurons**. We extract
`h_l` for every layer with forward pre-hooks on `down_proj` and pool over the
prompt's non-padding tokens by taking the mean, matching the pooling under which
SIREN's probes were fitted. Prompts are truncated at 512 tokens.

For a method selecting neurons S = {(l_1, j_1), ..., (l_k, j_k)} and a prompt
set of size N, this yields

    X_S ∈ R^{N × k},    (X_S)_{i,t} = h_{l_t, j_t}(prompt i)

Row *i* is the same prompt in every method's matrix; columns are whatever
neurons that method selected. All methods are compared at the same budget *k*,
so set size cannot drive any single comparison.

## 3. Preprocessing

Applied identically to every representation:

1. **Drop near-constant columns.** Neurons with essentially zero variance across
   prompts carry no signal and would be amplified into pure noise by step 4.
2. **Residualization.** Columns are regressed on a nuisance design matrix *Z*
   and replaced by the residuals (three variants, §5).
3. **Column centering**, required by the CKA estimators.
4. **Per-neuron z-scoring.** Neurons from different layers differ in scale by
   orders of magnitude, and Llama-3 contains massive-activation outlier neurons.
   Since CKA is known to be dominated by a small number of high-variance
   directions (Davari et al., 2022), without this step CKA would largely measure
   whether two methods happened to include the same outlier neuron. Reported
   without z-scoring as a sensitivity check.

## 4. Similarity measures

**Linear CKA.** For column-centered *X*, *Y* with Gram matrices K = XXᵀ,
L = YYᵀ:

    CKA(X, Y) = ‖XᵀY‖²_F / (‖XᵀX‖_F ‖YᵀY‖_F)

We report both the biased estimator and the **unbiased HSIC** estimator (Song
et al., 2012) as the headline value; the biased form inflates at small prompt
counts and by different amounts for matrices of different width.

**Spearman RSA** (Kriegeskorte et al., 2008). Spearman correlation between the
two prompt-by-prompt cosine-similarity matrices. Rank-based, hence insensitive
to the per-neuron scaling that CKA is sensitive to; used as a robustness check.
Agreement between CKA and RSA indicates no single outlier neuron is responsible.

## 5. Confound controls (residualization variants)

Every comparison is computed three times, removing progressively more nuisance
structure:

| variant | *Z* contains |
|---|---|
| `raw` | nothing |
| `class` | intercept, harmful/benign dummy |
| `class+length` | the above, plus a degree-2 polynomial in the prompt's token count (and dataset dummies when corpora are pooled) |

The class control tests whether apparent agreement is only the coarse
harmful-vs-benign axis. **The length control is equally necessary**: mean pooling
divides by the token count, so prompt length leaks into every neuron and is
typically a leading principal component, while harmful and benign prompts differ
systematically in length. Removing class means does not remove it.

## 6. Reference distributions

A raw CKA value is uninterpretable here, because all four methods select subsets
of one shared 458,752-neuron population and therefore inherit the model's global
variance structure before safety is considered. (On synthetic activations with a
single dominant shared factor, random neuron sets reach CKA ≈ 0.997.) Each
comparison is therefore reported against:

- **Layer-matched random null** [20 draws]. Each method's neurons are replaced
  by random neurons with the *same per-layer counts*, which also removes "the
  methods simply chose similar layers" as an explanation. This is the floor.
- **Global random null** [20 draws]. Same total size, uniform over all layers.
  Comparing the two nulls isolates the contribution of layer placement alone.
- **Same-method ceiling** [10 splits]. CKA between two disjoint, layer-matched
  halves of a *single* method's own selection: what "the same information
  recovered by the same procedure" scores on this data. Always below 1.0.
- **Within-method variants** (when all eight selections are used). Wang vs.
  Wang-robust, Zhao top-k vs. relative-epsilon, and the three Yang rankings are
  genuine alternative implementations of the same method, giving a stricter
  reference than disjoint halves. Rows are tagged `within` / `cross`.

The headline statistic is the **normalized score**

    normalized = (CKA_observed − CKA_null) / (CKA_ceiling − CKA_null)

with 0 = indistinguishable from layer-matched random neurons and 1 = as similar
as a method is to itself. A z-score against the null draws is also reported.

## 7. Cross-layer analysis

To test whether the methods encode comparable structure at *different network
depths*, CKA is computed between method A restricted to layer *l* and method B
restricted to layer *m*, for all 32 × 32 pairs (CKA does not require equal
dimensionality, so differing per-layer counts are fine).

A raw cross-layer map is not interpretable on its own: consecutive layers of a
residual network add small increments to a shared stream and are intrinsically
correlated, so any neuron subset of layer *l* scores high against any subset of
a nearby layer *m*, producing a broad diagonal band regardless of the
selections. We therefore recompute the entire map with layer-matched random
neurons **per layer pair** [5 draws] and report

    z(l, m) = (CKA_selected(l, m) − mean CKA_random(l, m)) / sd CKA_random(l, m)

which is the excess similarity attributable to *which* neurons were selected,
with the model's own layer geometry divided out. Cells where either method has
fewer than 10 neurons in the layer are masked: Yang's RMS ranking places 1,641
of its 2,500 neurons in layer 31 and only 2–4 in each of layers 14–22, where CKA
would be noise.

## 8. Data

Neurons were selected on HarmBench (harmful) + Alpaca (benign). To avoid
measuring structure that a method may have fitted to its own selection set, CKA
is evaluated on **held-out corpora that no method used**:

| set | rows used | composition |
|---|---|---|
| WildGuard (test) | 1,508 | class-balanced; the split has 1,725 prompts of which 754 are harmful, which is what caps the balanced size |
| XSTest | 400 | 200 safe-but-harmful-looking + 200 genuinely unsafe |

Prompt sets are class-balanced (per-class means are subtracted, so unbalanced
classes would give one class a much noisier mean), deduplicated (identical
prompts would produce identical rows, a block of maximal similarity shared by
all methods irrespective of neuron choice), and wrapped in the Llama-3 chat
template with the backbone's own tokenizer so activations occur in the same
regime as during selection. Corpora can be pooled to exceed a single dataset's
minority-class limit, in which case dataset identity is projected out alongside
class and length.

XSTest is the more diagnostic of the two: because surface form is deliberately
decoupled from the label, high similarity there cannot be attributed to all
methods merely encoding harmful-sounding wording. It is small, so absolute CKA
values are inflated by rank limitation; the normalized score is unaffected
because null and ceiling are computed on the same prompts.

## 9. Methods compared

| key | method | source |
|---|---|---|
| `siren` | SIREN | Jiao et al., *LLM Safety From Within: Detecting Harmful Content with Internal Representations* |
| `wang`, `wang_robust` | Wang | *Neuron-Level Safety Alignment for LLMs* |
| `zhao_topk`, `zhao_eps` | Zhao | *Understanding and Enhancing Safety Mechanisms* |
| `yang_rms`, `yang_refusal`, `yang_harmfulness` | Yang | Yang, Sondej, Mayne, Lee & Mahdi (EMNLP 2025), *How Does DPO Reduce Toxicity? A Mechanistic Neuron-Level Analysis* |

Budgets: N = 2,500 / 5,000 / 10,000 selected neurons.

## 10. Interpreting the results

Read the **normalized** column, not raw CKA.

| observation | conclusion |
|---|---|
| normalized ≈ 0 | The selections are no more similar than random neurons from the same layers. The methods genuinely identify different structure, and the near-zero Jaccard is the whole story. |
| normalized ≈ 1 | As similar as two disjoint halves of one method's own selection: the populations are representationally interchangeable despite sharing almost no neurons. |
| intermediate | Partial agreement in what is encoded; report the value rather than a verdict. |

Then read across variants:

- High on `raw`, collapsing on `class` → agreement is confined to the coarse
  harmful-vs-benign distinction. Real but weak.
- Still high on `class+length` → agreement extends to fine-grained structure
  *within* each class that is not prompt length or corpus identity. This is the
  strong claim.

Validity checks to state alongside any number: the biased and unbiased CKA
estimators agree (otherwise the prompt count is too low); Spearman RSA moves
with CKA (otherwise a few outlier neurons drive the result); and conclusions
hold across neuron budgets (a sharp rise with N indicates convergence merely
because larger subsets of a shared population must overlap in what they encode).

For the cross-layer maps, interpret the z panel rather than the raw panel.
Off-diagonal z peaks would indicate the same structure recovered at different
depths — the most interesting possible outcome, and the reason the raw map alone
would be misleading.

## 11. References

- Kornblith, Norouzi, Lee & Hinton (2019). *Similarity of Neural Network
  Representations Revisited.* ICML.
- Song, Smola, Gretton, Bedo & Borgwardt (2012). *Feature Selection via
  Dependence Maximization.* JMLR.
- Davari, Horoi, Natik, Lajoie, Wolf & Belilovsky (2022). *Reliability of CKA as
  a Similarity Measure in Deep Learning.*
- Kriegeskorte, Mur & Bandettini (2008). *Representational similarity analysis.*
