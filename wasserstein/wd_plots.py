"""Figures specific to the Wasserstein analysis. Generic heatmap / layer-profile
helpers are reused from cka/plots.py."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path):
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def pair_bars(rows, output_path, title, ylabel, ceiling_by_pair=None):
    """Observed similarity vs the layer-matched null, one group per pair.

    rows: dicts with pair, observed, null_mean, null_std.
    """
    fig, ax = plt.subplots(figsize=(max(7.0, 1.5 * len(rows)), 5.0))
    x = np.arange(len(rows))
    width = 0.36
    ax.bar(x - width / 2, [r["observed"] for r in rows], width,
           label="observed (selected neurons)", color="tab:blue")
    ax.bar(x + width / 2, [r["null_mean"] for r in rows], width,
           yerr=[r["null_std"] for r in rows], capsize=3,
           label="layer-matched random neurons", color="tab:gray")
    if ceiling_by_pair:
        for i, r in enumerate(rows):
            c = ceiling_by_pair.get(r["pair"])
            if c is None or not np.isfinite(c):
                continue
            ax.hlines(c, i - 0.5, i + 0.5, color="tab:red", linestyle="--",
                      linewidth=1.4,
                      label="same-method ceiling (disjoint halves)" if i == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([r["pair"] for r in rows], rotation=30, ha="right",
                       fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, output_path)


def layer_w1_heatmap(matrix, labels, output_path, title):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(matrix, cmap="magma_r")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    mid = np.nanmean(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if v > mid else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="layer W1 (layers)")
    ax.set_title(title, fontsize=11)
    _save(fig, output_path)


def scatter_measures(x, y, labels, output_path, title, xlabel, ylabel,
                     annotate=True, family=None, stats_text=None):
    """One point per method pair; diagonal for reference; optional family
    coloring (within-method variants vs cross-method)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    if family is None:
        ax.scatter(x, y, s=36, color="tab:blue")
    else:
        family = np.asarray(family)
        for name, color in (("cross", "tab:blue"), ("within", "tab:orange")):
            sel = family == name
            if sel.any():
                ax.scatter(x[sel], y[sel], s=36, color=color,
                           label=f"{name}-method pair")
        ax.legend(fontsize=8)
    if annotate:
        for xi, yi, lab in zip(x, y, labels):
            if np.isfinite(xi) and np.isfinite(yi):
                ax.annotate(lab, (xi, yi), fontsize=6.5, alpha=0.8,
                            xytext=(3, 3), textcoords="offset points")
    # No y = x diagonal: the two scores are not expected to coincide, only to
    # order the pairs alike. 0 = each measure's own null, 1 = its own ceiling.
    for v, style in ((0.0, dict(color="black", linewidth=0.6, alpha=0.5)),
                     (1.0, dict(color="tab:red", linewidth=0.8, linestyle="--",
                                alpha=0.6))):
        ax.axhline(v, **style)
        ax.axvline(v, **style)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    if stats_text:
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, va="top",
                fontsize=8.5, bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax.grid(alpha=0.3)
    _save(fig, output_path)


def matched_corr_hist(rho_by_pair, output_path, title, null_rho=None):
    """Distribution of |rho| across matched neuron pairs, one curve per method
    pair. Shows whether a similarity is carried by a few perfectly matched
    neurons or by many weakly matched ones -- a mean cannot tell them apart."""
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    bins = np.linspace(0, 1, 41)
    for pair, rho in rho_by_pair.items():
        ax.hist(np.abs(rho), bins=bins, histtype="step", linewidth=1.3,
                density=True, label=pair)
    if null_rho is not None and len(null_rho):
        ax.hist(np.abs(null_rho), bins=bins, histtype="stepfilled", alpha=0.25,
                color="gray", density=True, label="layer-matched random")
    ax.set_xlabel("|correlation| between matched neurons")
    ax.set_ylabel("density")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    _save(fig, output_path)
