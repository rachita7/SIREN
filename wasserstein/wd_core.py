"""Wasserstein (optimal-transport) distances between two safety-neuron
selections, designed to be read side by side with the CKA numbers in cka/.

Both measures start from the same object
----------------------------------------
Every method gives a prepared matrix X in R^{N x k} (cka_core.prepare: dead
neurons dropped, nuisance variables residualized, columns centered and
z-scored). Rows are the SAME N held-out prompts for every method; columns are
that method's k neurons. Column j is one neuron's *activation profile* over
the prompt set.

For two methods the k_A x k_B cross-correlation matrix

    R[i, j] = corr(profile of A's neuron i, profile of B's neuron j)

contains everything either measure looks at:

    linear CKA(A, B)  =  ||R_AB||_F^2 / (||R_AA||_F ||R_BB||_F)

i.e. an L2 AVERAGE over every cross-correlation, weighted by the within-method
correlation structure. It is invariant to rotating either population: two
different bases of the same subspace give CKA = 1.

The Wasserstein distance coded here is instead an ASSIGNMENT over the same
matrix. Each selection is the uniform empirical measure over its neurons,
viewed as points (profiles) in prompt space, and

    W(A, B) = min over one-to-one matchings pi of  mean_i  c(i, pi(i))
    c(i, j) = 1 - |R[i, j]|                        (default, sign-invariant)
            = 1 - R[i, j]                          (--signed)

For z-scored columns ||x||^2 = N, so 1 - R[i, j] = ||x_i - y_j||^2 / (2N):
the signed version is exactly the squared 2-Wasserstein distance between the
two neuron point clouds, up to the constant 2N. The sign-invariant version is
the same thing on the projective sphere (a neuron and its negation are the
same feature, which is also how CKA treats them, since X X^T is invariant to
column sign flips).

The transported cost has a direct reading: ``matched_corr = 1 - W`` is the
mean |correlation| between each of A's neurons and its optimally matched
partner in B. CKA asks "do the two populations span the same directions?";
W asks "can the neurons be paired off as individuals with the same profile?".
W-similarity implies high CKA, but not conversely -- a rotated basis breaks
the matching while leaving CKA at 1.

Why this specific construction and not another "Wasserstein between neuron
distributions"
----------------------------------------------------------------------------
* W over (layer, neuron_index) coordinates: neuron index has no metric;
  index 5 is not closer to 6 than to 9000. The only metric coordinate is
  depth, which is `layer_w1` below -- kept as a separate, cheap LOCATION
  measure, not a representational one.
* Discrete OT with cost 0 if same neuron else 1 reduces to 1 - overlap, i.e.
  Jaccard, which is already known to be ~0.02-0.16.
* Gromov-Wasserstein between the two prompt clouds throws away the fact that
  the prompts are aligned across methods, which is the one thing CKA uses.
* Uniform weights on neurons: the methods' importance scores live on
  incomparable scales, so weighting by them would compare scoring functions,
  not selections. A side effect is that OT treats every neuron equally, so it
  is immune to the massive-activation outlier neurons that can dominate CKA
  (Davari et al. 2022).

Exact OT is used, not sliced/entropic approximations: with uniform marginals
and (near-)equal sizes it is a linear assignment problem, 0.1 s at k = 2294
and 0.6 s at k = 4588 with scipy. When k_A != k_B (Zhao's tie handling adds a
few rows) the rectangular assignment matches min(k_A, k_B) neurons and the
remainder is reported as unmatched.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance


# --------------------------------------------------------------- utilities

def keep_mask(X, var_tol=1e-6):
    """Columns cka_core.prepare keeps. Same rule, exposed so that the
    (layer, neuron) identity of every surviving column is known."""
    X = np.asarray(X, dtype=np.float32)
    std = X.std(axis=0)
    thresh = max(1e-8, var_tol * float(np.median(std)) if std.size else 1e-8)
    return std > thresh


def unit_columns(X):
    """Column-normalize so that X^T Y is a correlation matrix.

    prepare() already centers every column; after z-scoring every column has
    norm sqrt(N), so this is a no-op up to the constant. Doing it explicitly
    makes the correlation cost correct with --no_zscore as well.
    """
    X = np.asarray(X, dtype=np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    return X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-8)


def cross_correlation(XA, XB):
    """[k_A, k_B] matrix of Pearson correlations between neuron profiles."""
    return (unit_columns(XA).T @ unit_columns(XB)).astype(np.float64)


# ---------------------------------------------------------- profile OT / W

def match(R, signed=False):
    """Optimal one-to-one matching under cost 1 - |R| (or 1 - R).

    Returns (rows, cols, cost) with cost[i] = c(rows[i], cols[i]).
    """
    cost_matrix = 1.0 - (R if signed else np.abs(R))
    rows, cols = linear_sum_assignment(cost_matrix)
    return rows, cols, cost_matrix[rows, cols]


def profile_wasserstein(XA, XB, signed=False, ids_a=None, ids_b=None,
                        strong=0.5):
    """Exact Wasserstein distance between two neuron populations in prompt
    space, plus the decomposition of the optimal matching.

    XA, XB : prepared [N, k] matrices (same N).
    ids_a, ids_b : optional [k, 2] int arrays of (layer, neuron_index) for
        every column, used to flag matched pairs that are literally the SAME
        neuron. Shared neurons match themselves at cost 0, so a pair with
        large index overlap looks similar for a trivial reason; the
        `_nonidentical` fields give the answer on the genuinely different
        neurons alone. (cka/check_overlap_effect.py plays this role for CKA.)

    Returns dict with
        wasserstein          mean transported cost = W (0 = identical clouds)
        matched_corr         1 - W: mean |rho| of optimally matched neurons
        matched_corr_median
        frac_strong          share of matches with |rho| >= `strong`
        n_matched, n_unmatched
        frac_identical       share of matches that are the same neuron
        matched_corr_nonidentical  mean |rho| over non-identical matches only
        rows, cols, rho      the matching itself
    """
    R = cross_correlation(XA, XB)
    rows, cols, cost = match(R, signed=signed)
    rho = R[rows, cols]
    sim = 1.0 - cost
    out = {
        "wasserstein": float(cost.mean()),
        "matched_corr": float(sim.mean()),
        "matched_corr_median": float(np.median(sim)),
        "frac_strong": float((sim >= strong).mean()),
        "n_matched": int(rows.size),
        "n_unmatched": int(max(R.shape) - rows.size),
        "rows": rows, "cols": cols, "rho": rho,
    }
    if ids_a is not None and ids_b is not None:
        ident = np.all(ids_a[rows] == ids_b[cols], axis=1)
        out["identical"] = ident
        out["frac_identical"] = float(ident.mean())
        out["matched_corr_nonidentical"] = (
            float(sim[~ident].mean()) if (~ident).any() else float("nan"))
    else:
        out["identical"] = np.zeros(rows.size, dtype=bool)
        out["frac_identical"] = 0.0
        out["matched_corr_nonidentical"] = out["matched_corr"]
    return out


def cka_from_correlations(R_AB, R_AA, R_BB):
    """Linear CKA written in terms of correlation matrices -- identical to
    cka_core.linear_cka on z-scored inputs. Provided to make the shared
    starting point of the two measures explicit and testable."""
    num = float(np.sum(R_AB ** 2))
    den = np.sqrt(float(np.sum(R_AA ** 2)) * float(np.sum(R_BB ** 2)))
    return num / den if den > 0 else float("nan")


def chance_matched_corr(n_prompts, k, seed=0):
    """Expected matched_corr for two INDEPENDENT Gaussian populations of k
    neurons on n_prompts prompts. Grows with k (more candidates to match
    against) and shrinks like 1/sqrt(n_prompts). Analytical guidance only; the
    real null in run_wasserstein.py uses actual model neurons."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n_prompts, k)).astype(np.float32)
    B = rng.normal(size=(n_prompts, k)).astype(np.float32)
    return profile_wasserstein(A, B)["matched_corr"]


# ------------------------------------------------------------ layer W1

def layer_w1(counts_a, counts_b):
    """Earth-mover's distance between two per-layer neuron histograms with
    ground metric |layer - layer'|. Units: layers.

    This is a LOCATION measure -- how far apart in depth the two selections
    sit. It is complementary to, not a substitute for, the profile W above:
    CKA's layer-matched null is constructed precisely to divide this out.
    """
    la = np.array(sorted(counts_a), dtype=np.float64)
    lb = np.array(sorted(counts_b), dtype=np.float64)
    wa = np.array([counts_a[int(l)] for l in la], dtype=np.float64)
    wb = np.array([counts_b[int(l)] for l in lb], dtype=np.float64)
    return float(wasserstein_distance(la, lb, wa, wb))


def layer_w1_to_uniform(counts, num_layers):
    """Distance of one method's depth profile from 'spread evenly over all
    layers' -- a per-method reference for reading the pairwise table."""
    layers = np.arange(num_layers, dtype=np.float64)
    la = np.array(sorted(counts), dtype=np.float64)
    wa = np.array([counts[int(l)] for l in la], dtype=np.float64)
    return float(wasserstein_distance(la, layers, wa, np.ones(num_layers)))
