"""Plotting helpers for the refusal-direction / DFA analysis."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, output_path):
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {output_path}")


def direction_profile(aurocs, cohens_d, chosen, max_frac, output_path, title):
    """Validation AUROC and Cohen's d of the projection, per hidden state."""
    h = len(aurocs)
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.plot(range(h), aurocs, marker="o", markersize=3.5, color="tab:blue",
            label="val AUROC of x . r_hat")
    ax.axvline(chosen, color="tab:red", linestyle="--", linewidth=1.2,
               label=f"chosen (h={chosen})")
    ax.axvspan(max_frac * h, h - 1, color="gray", alpha=0.15,
               label="beyond depth cap")
    ax.axhline(0.5, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("hidden state index (0 = embeddings, h = output of block h-1)")
    ax.set_ylabel("validation AUROC", color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(range(h), cohens_d, marker="s", markersize=3, color="tab:orange",
             alpha=0.6, label="Cohen's d")
    ax2.set_ylabel("Cohen's d", color="tab:orange")
    ax.set_title(title, fontsize=11)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    _save(fig, output_path)


def matrix_panel(matrices, labels, titles, cmaps, output_path, suptitle=None,
                 vranges=None, fmt="{:.2f}"):
    """Row of annotated square heatmaps (same layout as cka/plots.py)."""
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 5.0))
    if n == 1:
        axes = [axes]
    vranges = list(vranges) if vranges else [(None, None)] * n
    for ax, matrix, title, cmap, (vmin, vmax) in zip(axes, matrices, titles,
                                                     cmaps, vranges):
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        mid = np.nanmean(matrix)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v = matrix[i, j]
                if not np.isfinite(v):
                    continue
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=8, color="white" if v > mid else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    _save(fig, output_path)


def alignment_bars(names, observed, null_mean, null_std, output_path, title,
                   ylabel="cosine(v_method, refusal direction)"):
    """Observed alignment per method vs its layer-matched random null."""
    fig, ax = plt.subplots(figsize=(max(7.0, 1.6 * len(names)), 5.0))
    x = np.arange(len(names))
    width = 0.36
    ax.bar(x - width / 2, observed, width, color="tab:blue",
           label="observed (selected neurons)")
    ax.bar(x + width / 2, null_mean, width, yerr=null_std, capsize=3,
           color="tab:gray", label="layer-matched random neurons")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, output_path)


def dfa_distributions(values_by_method, null_by_method, output_path, title):
    """Per-neuron DFA distributions: selected vs layer-matched random.

    Symmetric-log x axis: DFA scores span orders of magnitude and both signs.
    """
    names = list(values_by_method)
    fig, axes = plt.subplots(len(names), 1,
                             figsize=(9.0, 2.1 * len(names)), sharex=True)
    if len(names) == 1:
        axes = [axes]
    all_vals = np.concatenate([np.abs(v) for v in values_by_method.values()])
    linthresh = max(np.percentile(all_vals[all_vals > 0], 25), 1e-8) \
        if (all_vals > 0).any() else 1e-8
    for ax, name in zip(axes, names):
        obs = np.asarray(values_by_method[name], dtype=np.float64)
        nul = np.asarray(null_by_method[name], dtype=np.float64)
        bins = np.concatenate([
            -np.geomspace(max(np.abs(obs).max(), np.abs(nul).max(), linthresh),
                          linthresh, 40),
            [0.0],
            np.geomspace(linthresh,
                         max(np.abs(obs).max(), np.abs(nul).max(), linthresh),
                         40)])
        ax.hist(nul, bins=bins, density=True, alpha=0.45, color="tab:gray",
                label="layer-matched random")
        ax.hist(obs, bins=bins, density=True, alpha=0.55, color="tab:blue",
                label="selected")
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_xscale("symlog", linthresh=linthresh)
        ax.set_ylabel(name, fontsize=9)
        ax.set_yticks([])
        if ax is axes[0]:
            ax.legend(fontsize=8)
    axes[-1].set_xlabel("per-neuron DFA score  (harmful - benign contribution "
                        "along the refusal direction)")
    fig.suptitle(title, fontsize=11)
    _save(fig, output_path)


def dfa_layer_profile(sums_by_method, output_path, title):
    """Per-layer total DFA of each method's selected neurons."""
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for name, sums in sums_by_method.items():
        ax.plot(range(len(sums)), sums, marker="o", markersize=3.5, label=name)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("layer")
    ax.set_ylabel("total DFA of the method's neurons in that layer")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, output_path)


def ablation_summary(rows, output_path, title):
    """Refusal rates and direction projections per ablation condition.

    rows: list of dicts with keys condition, refusal_harmful, refusal_benign,
          proj_harmful, proj_benign.
    """
    conditions = [r["condition"] for r in rows]
    x = np.arange(len(conditions))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(max(12.0, 1.7 * len(rows)), 5.2))

    ax = axes[0]
    ax.bar(x - width / 2, [r["refusal_harmful"] for r in rows], width,
           color="tab:red", label="harmful prompts")
    ax.bar(x + width / 2, [r["refusal_benign"] for r in rows], width,
           color="tab:green", label="benign prompts")
    ax.set_ylabel("refusal rate (substring match)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Refusal behavior", fontsize=10)

    ax2 = axes[1]
    ax2.bar(x - width / 2, [r["proj_harmful"] for r in rows], width,
            color="tab:red", label="harmful prompts")
    ax2.bar(x + width / 2, [r["proj_benign"] for r in rows], width,
            color="tab:green", label="benign prompts")
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.set_ylabel("mean x . r_hat at the chosen layer")
    ax2.set_title("Residual-stream projection onto the refusal direction",
                  fontsize=10)

    for ax_ in axes:
        ax_.set_xticks(x)
        ax_.set_xticklabels(conditions, rotation=30, ha="right", fontsize=8)
        ax_.legend(fontsize=8)
        ax_.grid(alpha=0.3, axis="y")
    fig.suptitle(title, fontsize=12)
    _save(fig, output_path)
