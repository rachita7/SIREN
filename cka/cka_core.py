"""Representational-similarity measures for comparing two neuron populations.

Everything here operates on matrices X of shape [num_prompts, num_neurons].
Row i is the SAME prompt in every matrix; columns need not correspond and the
two matrices need not have the same number of columns.

Measures
--------
linear_cka          Kornblith et al. (2019) linear CKA, biased HSIC estimator.
linear_cka_unbiased Same, with Song et al. (2012) unbiased HSIC. Preferred for
                    headline numbers: the biased estimator inflates when the
                    prompt count is small, and it inflates by different amounts
                    for matrices of different width.
rsa_spearman        Spearman correlation between the two prompt-by-prompt
                    similarity matrices. Rank-based, so unlike CKA it cannot be
                    driven by a handful of huge-variance "massive activation"
                    neurons (the known failure mode of CKA reported by Davari
                    et al. 2022). Use it as a robustness check on CKA.

Preprocessing (prepare)
-----------------------
1. drop near-constant columns   -- dead neurons would otherwise be amplified
                                   to pure noise by the z-score step
2. residualize against Z        -- Z is a design matrix of nuisance variables;
                                   pass class dummies to remove the
                                   harmful/benign axis, plus token count to
                                   remove the prompt-length axis that mean
                                   pooling unavoidably injects
3. column-center                -- required by both CKA estimators
4. optional per-column z-score  -- neurons from different layers differ in
                                   scale by orders of magnitude; without this,
                                   CKA measures whether both methods happened
                                   to include one of the model's outlier
                                   neurons rather than anything about safety
"""
import numpy as np


# ------------------------------------------------------------ preprocessing

def design_matrix(labels=None, token_counts=None, extra=None,
                  polynomial_degree=2):
    """Nuisance design matrix Z (with intercept) for residualization.

    labels        : [N] binary/integer class labels -> one dummy per class.
                    Removing these means CKA can no longer be explained by
                    "all methods separate harmful from benign".
    token_counts  : [N] number of real tokens per prompt -> polynomial terms.
                    Mean pooling divides by the token count, so prompt length
                    leaks into every neuron and is typically a leading
                    principal component. If harmful and benign prompts differ
                    in length (they do, for HarmBench vs Alpaca and for most
                    safety benchmarks) this confound is at least as serious as
                    the class confound.
    extra         : [N, p] any further covariates to project out.
    """
    cols = []
    n = None
    for arr in (labels, token_counts):
        if arr is not None:
            n = len(np.asarray(arr))
            break
    if extra is not None:
        n = len(np.asarray(extra))
    if n is None:
        return None

    cols.append(np.ones((n, 1), dtype=np.float64))
    if labels is not None:
        labels = np.asarray(labels)
        for value in np.unique(labels)[1:]:  # drop one level: intercept covers it
            cols.append((labels == value).astype(np.float64)[:, None])
    if token_counts is not None:
        t = np.asarray(token_counts, dtype=np.float64)
        t = (t - t.mean()) / (t.std() + 1e-12)
        for d in range(1, polynomial_degree + 1):
            cols.append((t ** d)[:, None])
    if extra is not None:
        cols.append(np.asarray(extra, dtype=np.float64).reshape(n, -1))
    return np.concatenate(cols, axis=1)


def residualize(X, Z):
    """Remove from every column of X the part linearly predictable from Z."""
    if Z is None:
        return X
    Z = np.asarray(Z, dtype=np.float64)
    coef, *_ = np.linalg.lstsq(Z, X.astype(np.float64), rcond=None)
    return (X.astype(np.float64) - Z @ coef).astype(np.float32)


def prepare(X, zscore=True, Z=None, var_tol=1e-6, return_dropped=False):
    """Full preprocessing pipeline; returns a float32 [N, k'] matrix."""
    X = np.asarray(X, dtype=np.float32)

    std = X.std(axis=0)
    keep = std > max(1e-8, var_tol * float(np.median(std)) if std.size else 1e-8)
    dropped = int((~keep).sum())
    if dropped:
        X = X[:, keep]
    if X.shape[1] == 0:
        raise ValueError("all columns were near-constant across prompts")

    X = residualize(X, Z)
    X = X - X.mean(axis=0, keepdims=True)
    if zscore:
        std = X.std(axis=0, keepdims=True)
        X = X / (std + 1e-8)
        X = X - X.mean(axis=0, keepdims=True)

    return (X, dropped) if return_dropped else X


# --------------------------------------------------------------- linear CKA

def _fro_sq_cross(X, Y):
    """||X^T Y||_F^2 for column-centered X, Y, computed the cheap way.

    The feature-space form costs O(N k1 k2); the Gram form costs O(N^2 (k1+k2)).
    Pick whichever is smaller -- this is what makes the 32x32 cross-layer sweep
    (thousands of CKA evaluations) tractable.
    """
    n, k1 = X.shape
    k2 = Y.shape[1]
    if k1 * k2 <= n * n:
        cross = X.T @ Y
        return float(np.sum(cross.astype(np.float64) ** 2))
    K = X @ X.T
    L = Y @ Y.T
    return float(np.sum(K.astype(np.float64) * L.astype(np.float64)))


def linear_cka(X, Y):
    """Biased linear CKA. Inputs must already be column-centered (see prepare)."""
    num = _fro_sq_cross(X, Y)
    den = np.sqrt(_fro_sq_cross(X, X)) * np.sqrt(_fro_sq_cross(Y, Y))
    return float(num / den) if den > 0 else float("nan")


class Representation:
    """A prepared matrix plus its cached prompt-by-prompt Gram matrix.

    Comparing M methods means M(M-1)/2 pairs -- 28 of them for all eight
    selections. Calling linear_cka() directly recomputes both self-norms every
    time, so the dominant O(N^2 k) work is repeated ~M times per matrix. Caching
    the Gram matrix K = X X^T once reduces every pairwise comparison to
    sum(K * L), which is O(N^2) and effectively free.

    Memory is N^2 floats per representation: 16 MB at N=2000, so all eight
    observed representations cost ~128 MB. That is the right trade whenever
    several pairs are needed; the cross-layer sweep does NOT use this, because
    there it would mean 32 layers x 8 methods x 16 MB = 4 GB.
    """

    __slots__ = ("X", "gram", "norm")

    def __init__(self, X):
        self.X = X
        self.gram = X @ X.T
        self.norm = float(np.sqrt(np.sum(self.gram.astype(np.float64) ** 2)))

    def cka(self, other):
        den = self.norm * other.norm
        if den <= 0:
            return float("nan")
        num = float(np.sum(self.gram.astype(np.float64)
                           * other.gram.astype(np.float64)))
        return num / den

    def cka_unbiased(self, other):
        if self.gram.shape[0] < 4:
            return float("nan")
        num = _hsic_unbiased(self.gram, other.gram)
        den = np.sqrt(max(_hsic_unbiased(self.gram, self.gram), 0.0)
                      * max(_hsic_unbiased(other.gram, other.gram), 0.0))
        return float(num / den) if den > 0 else float("nan")


def _hsic_unbiased(K, L):
    """Song et al. (2012) unbiased HSIC estimator from uncentered Gram matrices."""
    n = K.shape[0]
    K = K.astype(np.float64).copy()
    L = L.astype(np.float64).copy()
    np.fill_diagonal(K, 0.0)
    np.fill_diagonal(L, 0.0)
    k_sum = K.sum()
    l_sum = L.sum()
    # tr(K L) = sum(K * L) for symmetric K, L -- avoids an O(N^3) matmul.
    term = (float(np.sum(K * L))
            + k_sum * l_sum / ((n - 1) * (n - 2))
            - 2.0 * (K.sum(axis=0) @ L.sum(axis=1)) / (n - 2))
    return term / (n * (n - 3))


def linear_cka_unbiased(X, Y):
    """Linear CKA built from the unbiased HSIC estimator.

    Costs O(N^2) memory, so this is used for the small headline matrix rather
    than the cross-layer sweep.
    """
    if X.shape[0] < 4:
        return float("nan")
    K = X @ X.T
    L = Y @ Y.T
    num = _hsic_unbiased(K, L)
    den = np.sqrt(max(_hsic_unbiased(K, K), 0.0) * max(_hsic_unbiased(L, L), 0.0))
    return float(num / den) if den > 0 else float("nan")


# ----------------------------------------------------------------- rank RSA

def _upper_triangle(M, max_pairs, rng):
    iu = np.triu_indices(M.shape[0], k=1)
    values = M[iu]
    if max_pairs and values.size > max_pairs:
        pick = rng.choice(values.size, size=max_pairs, replace=False)
        values = values[pick]
        return values, pick
    return values, None


def rsa_spearman(X, Y, max_pairs=400_000, seed=0):
    """Spearman correlation of the two prompt-similarity matrices.

    Cosine similarity is used rather than the raw inner product so that a few
    very long prompts cannot dominate. Rank-based and therefore insensitive to
    the per-neuron scaling that CKA is famously sensitive to.
    """
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)

    def sim(M):
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
        return M @ M.T

    a = sim(X)
    b = sim(Y)
    va, pick = _upper_triangle(a, max_pairs, rng)
    iu = np.triu_indices(b.shape[0], k=1)
    vb = b[iu]
    if pick is not None:
        vb = vb[pick]
    rho, _ = spearmanr(va, vb)
    return float(rho)


# ------------------------------------------------------------------ summary

def normalized_score(observed, null_mean, ceiling_mean):
    """Where the observed similarity sits between chance and the ceiling.

    0 = indistinguishable from layer-matched random neurons.
    1 = as similar as two disjoint halves of the SAME method's own selection.
    Values are not clipped; >1 or <0 are informative and should be reported.
    """
    span = ceiling_mean - null_mean
    if not np.isfinite(span) or abs(span) < 1e-12:
        return float("nan")
    return float((observed - null_mean) / span)


def null_zscore(observed, null_values):
    null_values = np.asarray(null_values, dtype=np.float64)
    sd = null_values.std(ddof=1) if null_values.size > 1 else 0.0
    if sd < 1e-12:
        return float("nan")
    return float((observed - null_values.mean()) / sd)
