"""Plotting helpers for the CKA analysis."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _annotate(ax, matrix, labels, fmt="{:.2f}", threshold=None):
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    mid = threshold if threshold is not None else np.nanmean(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if not np.isfinite(v):
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                    color="white" if v > mid else "black")


def method_matrix_panel(matrices, labels, titles, cmaps, output_path,
                        suptitle=None, vranges=None):
    """Row of annotated method-by-method heatmaps."""
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 5.0))
    if n == 1:
        axes = [axes]
    for ax, matrix, title, cmap in zip(axes, matrices, titles, cmaps):
        vmin = vmax = None
        if vranges:
            vmin, vmax = vranges.pop(0)
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        _annotate(ax, matrix, labels)
        fig.colorbar(im, ax=ax, fraction=0.046)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {output_path}")


def pair_bars(rows, output_path, title, ceiling_by_pair=None):
    """Observed CKA vs the layer-matched null, one group per method pair.

    rows: list of dicts with keys pair, observed, null_mean, null_std,
          ceiling (optional).
    """
    fig, ax = plt.subplots(figsize=(max(7.0, 1.5 * len(rows)), 5.0))
    x = np.arange(len(rows))
    width = 0.36
    obs = [r["observed"] for r in rows]
    null = [r["null_mean"] for r in rows]
    err = [r["null_std"] for r in rows]

    ax.bar(x - width / 2, obs, width, label="observed (selected neurons)",
           color="tab:blue")
    ax.bar(x + width / 2, null, width, yerr=err, capsize=3,
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
    ax.set_ylabel("linear CKA")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {output_path}")


def variant_bars(variant_rows, output_path, title):
    """Normalized score per method pair, grouped by residualization variant.

    variant_rows: dict variant_name -> list of (pair, score)
    """
    variants = list(variant_rows)
    pairs = [p for p, _ in variant_rows[variants[0]]]
    fig, ax = plt.subplots(figsize=(max(8.0, 1.6 * len(pairs)), 5.0))
    x = np.arange(len(pairs))
    width = 0.8 / len(variants)
    for i, variant in enumerate(variants):
        values = [v for _, v in variant_rows[variant]]
        ax.bar(x + (i - (len(variants) - 1) / 2) * width, values, width,
               label=variant)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.axhline(1.0, color="tab:red", linestyle="--", linewidth=1,
               label="same-method ceiling")
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("normalized similarity\n(0 = random neurons, 1 = same-method ceiling)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {output_path}")


def cross_layer_panel(observed, zscore, label_a, label_b, output_path,
                      mask=None):
    """Raw cross-layer CKA next to its null-normalized version.

    The raw panel almost always shows a broad diagonal band, because adjacent
    layers of a residual network are intrinsically correlated regardless of
    which neurons you pick. The z-scored panel is the interpretable one: it
    shows where the SELECTED neurons align more than layer-matched random
    neurons from the very same layer pair do.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    for ax, matrix, title, cmap, center in (
            (axes[0], observed, f"Raw linear CKA\n{label_a} layer vs {label_b} layer",
             "viridis", False),
            (axes[1], zscore,
             f"Excess over layer-matched random neurons (z)\n{label_a} vs {label_b}",
             "RdBu_r", True)):
        m = np.array(matrix, dtype=float)
        if mask is not None:
            m = np.where(mask, m, np.nan)
        if center:
            lim = np.nanmax(np.abs(m)) if np.isfinite(m).any() else 1.0
            im = ax.imshow(m, origin="lower", cmap=cmap, vmin=-lim, vmax=lim)
        else:
            im = ax.imshow(m, origin="lower", cmap=cmap)
        ax.set_xlabel(f"{label_b} layer")
        ax.set_ylabel(f"{label_a} layer")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {output_path}")


def layer_profiles(selections, output_path, title):
    """Per-layer neuron counts for each method, normalized per method."""
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for label, counts in selections.items():
        layers = sorted(counts)
        vals = np.array([counts[l] for l in layers], dtype=float)
        ax.plot(layers, vals / vals.sum(), marker="o", markersize=3.5,
                label=label)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Share of the method's selected neurons")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {output_path}")
