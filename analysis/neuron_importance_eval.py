"""
Neuron ranking and ablation evaluation from:

    "What Is One Grain of Sand in the Desert? Analyzing Individual Neurons in
    Deep NLP Models" (Dalvi et al., AAAI 2019, arXiv:1812.09355)

The paper's Linguistic Correlation Analysis trains a linear classifier on
neuron activations and ranks neurons by their absolute classifier weights
(Algorithm 1 for multi-class probes). The ranking is evaluated by:

  * masking-out ablation (their Table 2): keep only the top/bottom N% of
    neurons, zero the rest, and re-evaluate the already-trained classifier;
  * retraining ablation (their Table 4): retrain the classifier on only the
    kept neurons and evaluate it.

This script applies that methodology independently to one linear probe per
layer (the paper trains a single classifier over one D-dimensional
representation; the layer-wise application is an adaptation). How faithful
the ranking is also depends on how the probes were trained: the paper uses
elastic-net (L1+L2) regularization; e.g. SIREN probes are L1-only.

Outputs:
    {label}_neuron_ranking.npz   - full neuron ranking per layer
    {label}_salient_neurons.csv/.png - per-layer count of salient neurons (the
        paper's criterion: top neurons cumulatively contributing --mass_threshold
        of the weight mass, 25% in their focused-vs-distributed analysis) and
        the cumulative weight-mass curves behind it
    {label}_ablation_masking.csv/.png   (with --features)
    {label}_ablation_retrain.csv/.png   (with --train_features --retrain)

The masking results also include the majority-class baseline the paper
compares against (MAJ in their Table 1). The expected picture for a
meaningful ranking: top N% well above bottom N%, both probes above majority.

-------------------------------------------------------------------------------
Input formats (two options)
-------------------------------------------------------------------------------

A) SIREN probes pickle (this repo):

    python analysis/neuron_importance_eval.py \
        --probes train/probes/llama3-8b-instruct_general_probes.pkl \
        --pooling_type residual_mean --label llama3-8b-instruct

B) Generic npz of probe weights (for any codebase). Save one array per layer;
   1-D (D,) for binary probes or 2-D (num_classes, D) for multi-class probes.
   Optional per-layer bias under "<key>__bias":

    np.savez("weights.npz",
             layer0=w0, layer1=w1, ...,          # (D,) or (L, D)
             layer0__bias=b0, ...)               # optional, scalar or (L,)

    python analysis/neuron_importance_eval.py --weights weights.npz --label mymodel

-------------------------------------------------------------------------------
Activations for the ablation evaluation
-------------------------------------------------------------------------------

Provide an npz with one (N, D) activation matrix per layer under the *same
keys* as the weights, plus an (N,) integer "labels" array:

    np.savez("eval_features.npz", labels=y,
             **{key: X_layer for key, X_layer in features.items()})

    python analysis/neuron_importance_eval.py --weights weights.npz \
        --features eval_features.npz \
        --train_features train_features.npz --retrain

In SIREN, the per-layer matrices come from stacking
rep[layer_idx]["residual_mean"] over samples (see
train.probe_trainer.extract_layer_features).

IMPORTANT: the activations must be exactly the representation the probe was
trained on. If your pipeline standardizes/normalizes features before probing,
apply the same preprocessing before exporting them here; otherwise the
masking-out evaluation of the trained probe is meaningless.

Note the masking here ablates neurons *in the probe* (equivalent to zeroing
the activation columns of a linear model); it does not reproduce the paper's
separate experiment of ablating neurons inside the underlying model itself.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import io
import pickle
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class CPUUnpickler(pickle.Unpickler):
    """Unpickler that maps CUDA tensors to CPU, so GPU-trained probes can be
    analyzed on machines without a GPU."""

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            import torch
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu",
                                        weights_only=False)
        return super().find_class(module, name)


def load_siren_probes(probe_path, pooling_type):
    """Return ({layer_key: (weights, bias)}, {layer_key: best_C}) from a SIREN
    *_general_probes.pkl."""
    with open(probe_path, "rb") as f:
        data = CPUUnpickler(f).load()
    best_probes = data["best_probes"]

    layers, best_Cs = {}, {}
    suffix = f"_{pooling_type}"
    for key, entry in best_probes.items():
        if not key.endswith(suffix):
            continue
        model = entry["probe"].model
        w = model.weight.detach().cpu().numpy().squeeze()
        b = model.bias.detach().cpu().numpy().squeeze()
        layers[key] = (w, float(b) if b.ndim == 0 else b)
        best_Cs[key] = entry.get("best_C", 1.0)
    if not layers:
        raise ValueError(f"No probes with pooling_type={pooling_type!r} found in {probe_path}")
    return layers, best_Cs


def load_generic_weights(npz_path):
    """Return {layer_key: (weights, bias)} from a plain npz file."""
    data = np.load(npz_path)
    layers = {}
    for key in data.files:
        if key.endswith("__bias") or key == "labels":
            continue
        w = np.asarray(data[key], dtype=np.float64).squeeze()
        bias_key = f"{key}__bias"
        b = np.asarray(data[bias_key], dtype=np.float64).squeeze() if bias_key in data.files else 0.0
        layers[key] = (w, b)
    if not layers:
        raise ValueError(f"No weight arrays found in {npz_path}")
    return layers


def load_features(npz_path):
    data = np.load(npz_path)
    if "labels" not in data.files:
        raise ValueError(f"{npz_path} must contain a 'labels' array")
    y = np.asarray(data["labels"]).astype(int)
    X = {k: np.asarray(data[k], dtype=np.float64) for k in data.files if k != "labels"}
    return X, y


def sort_layer_keys(keys):
    """Order layer keys by the first integer they contain (fallback: as-is)."""
    def layer_num(key):
        m = re.search(r"\d+", key)
        return int(m.group()) if m else 0
    return sorted(keys, key=layer_num)


def layer_number(key):
    m = re.search(r"\d+", key)
    return int(m.group()) if m else None


# ---------------------------------------------------------------------------
# Ranking (paper: Methodology + Algorithm 1)
# ---------------------------------------------------------------------------

def rank_neurons(weights, alpha=0.5):
    """Rank neurons by decreasing importance.

    Binary probes (1-D weights): descending |weight|.
    Multi-class probes (num_classes x D): the paper's Algorithm 1 - starting
    at p=1% and increasing by `alpha`, take the neurons that are in the
    top-p% cumulative weight mass of *any* class, appending newly discovered
    neurons to the ordering.

    Neurons discovered in the same iteration are appended in order of their
    max-|weight| across classes; the paper leaves this tie-break unspecified,
    so it is an implementation choice.
    """
    absw = np.abs(weights)
    if absw.ndim == 1:
        return np.argsort(-absw, kind="stable")

    num_classes, dim = absw.shape
    order = np.argsort(-absw, axis=1, kind="stable")
    sorted_w = np.take_along_axis(absw, order, axis=1)
    cumsum = np.cumsum(sorted_w, axis=1)
    totals = cumsum[:, -1]

    max_abs = absw.max(axis=0)
    ordering = []
    seen = np.zeros(dim, dtype=bool)
    for p in np.arange(1.0, 100.0 + alpha, alpha):
        new = set()
        for c in range(num_classes):
            if totals[c] <= 0:  # all-zero class contributes no neurons
                continue
            n_top = int(np.searchsorted(cumsum[c], p / 100.0 * totals[c])) + 1
            for idx in order[c, :n_top]:
                if not seen[idx]:
                    new.add(int(idx))
        for idx in sorted(new, key=lambda i: -max_abs[i]):
            seen[idx] = True
            ordering.append(idx)
        if len(ordering) == dim:
            break
    # Neurons never discovered (e.g. zero weight in every class) go last.
    if len(ordering) < dim:
        remaining = np.where(~seen)[0]
        ordering.extend(remaining[np.argsort(-max_abs[remaining], kind="stable")])
    return np.array(ordering, dtype=int)


def count_salient_neurons(weights, threshold):
    """Number of salient neurons by the paper's criterion: the top neurons
    that cumulatively contribute `threshold` of the total absolute weight
    mass (the paper uses 25% for its focused-vs-distributed analysis).

    For multi-class probes this is the size of the union of the per-class
    top-threshold sets (GETTOPNEURONSPERTAG in Algorithm 1)."""
    absw = np.atleast_2d(np.abs(weights))
    salient = set()
    for class_w in absw:
        total = class_w.sum()
        if total <= 0:
            continue
        order = np.argsort(-class_w, kind="stable")
        cumsum = np.cumsum(class_w[order])
        n_top = int(np.searchsorted(cumsum, threshold * total)) + 1
        salient.update(order[:n_top].tolist())
    return len(salient)


# ---------------------------------------------------------------------------
# Ablation evaluation (paper: Evaluation using Neuron Ablation)
# ---------------------------------------------------------------------------

def predict(X, weights, bias):
    # np.errstate: Apple Accelerate emits spurious divide/overflow/invalid
    # RuntimeWarnings for matmul on macOS; the results are correct.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        if weights.ndim == 1:
            return (X @ weights + bias > 0).astype(int)
        return np.argmax(X @ weights.T + bias, axis=1)


def masked_weights(weights, keep_indices):
    """Zeroing masked activations is equivalent to zeroing their weights."""
    masked = np.zeros_like(weights)
    if weights.ndim == 1:
        masked[keep_indices] = weights[keep_indices]
    else:
        masked[:, keep_indices] = weights[:, keep_indices]
    return masked


def score(y_true, y_pred, metric):
    if metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    return f1_score(y_true, y_pred, average="macro")


def majority_baseline(y_eval, metric):
    """Score of always predicting the most frequent eval label (MAJ in the
    paper's Table 1, simplified to a global majority class)."""
    majority_label = np.bincount(y_eval).argmax()
    return score(y_eval, np.full_like(y_eval, majority_label), metric)


def masking_ablation(layers, ranking, X_eval, y_eval, percentages, metric):
    """Table 2 of the paper: keep top/bottom N% of neurons, zero the rest,
    re-evaluate the already-trained probe."""
    majority = majority_baseline(y_eval, metric)
    rows = []
    for key in sort_layer_keys(layers):
        if key not in X_eval:
            continue
        w, b = layers[key]
        X = X_eval[key]
        dim = len(ranking[key])
        full = score(y_eval, predict(X, w, b), metric)
        for pct in percentages:
            n_keep = max(1, int(round(pct / 100.0 * dim)))
            for which, keep in (("top", ranking[key][:n_keep]),
                                ("bottom", ranking[key][-n_keep:])):
                masked = score(y_eval, predict(X, masked_weights(w, keep), b), metric)
                rows.append({"layer_key": key, "layer": layer_number(key),
                             "percent_kept": pct, "which": which, "n_kept": n_keep,
                             metric: masked, f"full_{metric}": full,
                             f"majority_{metric}": majority})
    return pd.DataFrame(rows)


def make_retrainer(backend, c_values, l1_ratio, best_Cs=None, seed=42):
    """Return fit_predict(X_tr, y_tr, X_ev, key) -> predictions.

    backend "elasticnet": logistic regression with the paper's elastic-net
    (L1+L2) objective. The paper's lambda values do not transfer directly to
    sklearn's C, so if several C values are given, the best one is chosen on
    an internal 80/20 validation split before refitting on the full data.

    backend "siren": reuse SIREN's own LinearProbe trainer (L1, torch) with
    the best_C the original probe of that layer was trained with, so the
    retrained classifier matches the original probe formulation exactly.
    """
    if backend == "siren":
        from train.probe_trainer import LinearProbe

        def fit_predict(X_tr, y_tr, X_ev, key):
            probe = LinearProbe(C=(best_Cs or {}).get(key, 1.0), device="cpu")
            probe.train(X_tr.astype(np.float32), y_tr, quick_eval=True)
            return probe.predict(X_ev.astype(np.float32))
        return fit_predict

    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    def make_clf(C):
        kwargs = {"solver": "saga", "l1_ratio": l1_ratio, "C": C, "max_iter": 3000}
        # 'penalty' is deprecated since sklearn 1.8 (l1_ratio alone implies
        # elastic-net) but required for elastic-net in older versions.
        major, minor = (int(v) for v in sklearn.__version__.split(".")[:2])
        if (major, minor) < (1, 8):
            kwargs["penalty"] = "elasticnet"
        return LogisticRegression(**kwargs)

    def fit_predict(X_tr, y_tr, X_ev, key):
        # np.errstate: see predict() - suppresses spurious Accelerate matmul
        # warnings raised inside sklearn on macOS. ConvergenceWarnings are
        # also silenced: saga routinely fails to fully converge on the
        # noise-only bottom-N% subsets, which is expected and harmless.
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"), \
                warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            best_C = c_values[0]
            if len(c_values) > 1:
                X_fit, X_val, y_fit, y_val = train_test_split(
                    X_tr, y_tr, test_size=0.2, random_state=seed, stratify=y_tr)
                best_val = -1.0
                for C in c_values:
                    clf = make_clf(C)
                    clf.fit(X_fit, y_fit)
                    val = accuracy_score(y_val, clf.predict(X_val))
                    if val > best_val:
                        best_val, best_C = val, C
            clf = make_clf(best_C)
            clf.fit(X_tr, y_tr)
            return clf.predict(X_ev)
    return fit_predict


def retraining_ablation(layers, ranking, X_train, y_train, X_eval, y_eval,
                        percentages, metric, fit_predict):
    """Table 4 of the paper: retrain a fresh classifier on only the kept
    neurons and evaluate it."""
    majority = majority_baseline(y_eval, metric)
    rows = []
    for key in sort_layer_keys(layers):
        if key not in X_train or key not in X_eval:
            continue
        dim = len(ranking[key])
        for pct in percentages:
            n_keep = max(1, int(round(pct / 100.0 * dim)))
            for which, keep in (("top", ranking[key][:n_keep]),
                                ("bottom", ranking[key][-n_keep:])):
                preds = fit_predict(X_train[key][:, keep], y_train,
                                    X_eval[key][:, keep], key)
                rows.append({"layer_key": key, "layer": layer_number(key),
                             "percent_kept": pct, "which": which, "n_kept": n_keep,
                             metric: score(y_eval, preds, metric),
                             f"majority_{metric}": majority})
        print(f"  retrained {key}")
    return pd.DataFrame(rows)


def plot_salient_neurons(layers, mass_threshold, label, output_dir):
    """Visualize the ranking results from the probe weights alone:
    (1) the per-layer number of salient neurons at the paper's cumulative
    weight-mass criterion (their Figure 6 analog), and (2) the cumulative
    weight-mass curves that criterion is read off from."""
    keys = sort_layer_keys(layers)
    counts = {k: count_salient_neurons(layers[k][0], mass_threshold) for k in keys}
    dims = {k: np.atleast_2d(layers[k][0]).shape[1] for k in keys}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    layer_ids = [layer_number(k) for k in keys]
    axes[0].plot(layer_ids, [counts[k] for k in keys], marker="o", markersize=4,
                 color="tab:red")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel(f"# salient neurons ({int(mass_threshold * 100)}% of weight mass)")
    axes[0].set_title("Salient neurons per layer\n(low = focused, high = distributed)")
    axes[0].grid(alpha=0.3)

    cmap = plt.get_cmap("viridis")
    for i, key in enumerate(keys):
        # For multi-class probes, plot the mass curve of the summed |weights|.
        w = np.sort(np.atleast_2d(np.abs(layers[key][0])).sum(axis=0))[::-1]
        total = max(w.sum(), 1e-12)
        frac_neurons = np.arange(1, len(w) + 1) / len(w)
        axes[1].plot(frac_neurons, np.cumsum(w) / total,
                     color=cmap(i / max(len(keys) - 1, 1)), linewidth=1)
    axes[1].axhline(mass_threshold, color="gray", linestyle=":", linewidth=1.5,
                    label=f"{int(mass_threshold * 100)}% mass criterion")
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=min(layer_ids), vmax=max(layer_ids)))
    fig.colorbar(sm, ax=axes[1], label="Layer")
    axes[1].set_xlabel("Fraction of neurons (ranked by |weight|)")
    axes[1].set_ylabel("Cumulative share of weight mass")
    axes[1].set_title("Cumulative weight mass per layer")
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle(label)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{label}_salient_neurons.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")

    csv_path = os.path.join(output_dir, f"{label}_salient_neurons.csv")
    pd.DataFrame([{"layer_key": k, "layer": layer_number(k),
                   "num_neurons": dims[k], "num_salient": counts[k],
                   "salient_fraction": counts[k] / dims[k]} for k in keys]
                 ).to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")


def plot_ablation(df, metric, focus_percent, label, output_dir, kind="masking"):
    """Visualize the Table 2 / Table 4 numbers."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Panel 1: across layers at the focus percentage.
    focus = df[df["percent_kept"] == focus_percent]
    if f"full_{metric}" in df.columns:
        full = df.drop_duplicates("layer_key").sort_values("layer")
        axes[0].plot(full["layer"], full[f"full_{metric}"], color="black",
                     linewidth=1.5, label="all neurons")
    for which, color in (("top", "tab:green"), ("bottom", "tab:orange")):
        sub = focus[focus["which"] == which].sort_values("layer")
        axes[0].plot(sub["layer"], sub[metric], marker="o", markersize=3,
                     color=color, label=f"{which} {focus_percent}% kept")
    axes[0].axhline(df[f"majority_{metric}"].iloc[0], color="gray",
                    linestyle=":", linewidth=1.5, label="majority baseline")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel(metric)
    axes[0].set_title(f"{kind.capitalize()} ablation across layers ({focus_percent}% kept)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Panel 2: across percentages at the best layer.
    if f"full_{metric}" in df.columns:
        best_key = df.loc[df[f"full_{metric}"].idxmax(), "layer_key"]
    else:
        best_key = df.loc[df[metric].idxmax(), "layer_key"]
    best = df[df["layer_key"] == best_key]
    for which, color in (("top", "tab:green"), ("bottom", "tab:orange")):
        sub = best[best["which"] == which].sort_values("percent_kept")
        axes[1].plot(sub["percent_kept"], sub[metric], marker="o", markersize=4,
                     color=color, label=which)
    if f"full_{metric}" in df.columns:
        axes[1].axhline(best[f"full_{metric}"].iloc[0], color="black",
                        linewidth=1.5, linestyle="--", label="all neurons")
    axes[1].axhline(best[f"majority_{metric}"].iloc[0], color="gray",
                    linestyle=":", linewidth=1.5, label="majority baseline")
    axes[1].set_xlabel("% of neurons kept")
    axes[1].set_ylabel(metric)
    axes[1].set_title(f"{kind.capitalize()} ablation at {best_key}")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle(label)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{label}_ablation_{kind}.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Neuron ranking and ablation evaluation "
                    "(Dalvi et al. 2019, arXiv:1812.09355)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--probes", type=str,
                        help="SIREN *_general_probes.pkl (this repo's format)")
    source.add_argument("--weights", type=str,
                        help="Generic npz of per-layer probe weights (see module docstring)")
    parser.add_argument("--pooling_type", type=str, default="residual_mean",
                        help="Pooling type inside the SIREN pickle (ignored for --weights)")
    parser.add_argument("--features", type=str, default=None,
                        help="npz with per-layer eval activations + 'labels'; enables the "
                             "masking-out ablation (paper Table 2)")
    parser.add_argument("--train_features", type=str, default=None,
                        help="npz with per-layer train activations + 'labels'; needed for --retrain")
    parser.add_argument("--retrain", action="store_true",
                        help="Also run the retraining ablation (paper Table 4); slow")
    parser.add_argument("--retrain_backend", type=str, default="elasticnet",
                        choices=["elasticnet", "siren"],
                        help="elasticnet: L1+L2 logistic regression as in the paper; "
                             "siren: reuse SIREN's LinearProbe trainer (requires --probes)")
    parser.add_argument("--retrain_C", type=float, nargs="+", default=[0.1, 1.0, 10.0],
                        help="Inverse regularization strength(s) for elasticnet retraining; "
                             "if several are given, tuned on an internal validation split "
                             "(pass a single value to speed things up)")
    parser.add_argument("--retrain_l1_ratio", type=float, default=0.5,
                        help="L1/L2 mix for elasticnet retraining (0=ridge, 1=lasso)")
    parser.add_argument("--percentages", type=float, nargs="+",
                        default=[2, 5, 10, 15, 20, 50],
                        help="Percentages of neurons to keep in the ablations")
    parser.add_argument("--focus_percent", type=float, default=10,
                        help="Percentage highlighted in the across-layers ablation panel")
    parser.add_argument("--mass_threshold", type=float, default=0.9,
                        help="Cumulative weight-mass criterion for counting salient "
                             "neurons (the paper uses 0.25)")
    parser.add_argument("--metric", type=str, default="accuracy",
                        choices=["accuracy", "f1_macro"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", type=str, default=None,
                        help="Prefix for output files (default: input filename stem)")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "results",
                                             "neuron_importance"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    source_path = args.probes or args.weights
    label = args.label or os.path.splitext(os.path.basename(source_path))[0]

    best_Cs = {}
    if args.probes:
        layers, best_Cs = load_siren_probes(args.probes, args.pooling_type)
    else:
        layers = load_generic_weights(args.weights)
    keys = sort_layer_keys(layers)
    print(f"Loaded {len(keys)} layers from {source_path} "
          f"(dims: {layers[keys[0]][0].shape[-1]})")

    ranking = {k: rank_neurons(layers[k][0]) for k in keys}
    ranking_path = os.path.join(args.output_dir, f"{label}_neuron_ranking.npz")
    np.savez(ranking_path, **{k: ranking[k] for k in keys})
    print(f"Saved {ranking_path} (full neuron ranking per layer)")

    plot_salient_neurons(layers, args.mass_threshold, label, args.output_dir)

    if not args.features:
        print("\nNo --features given: skipping the ablation evaluation (paper Table 2/4). "
              "See the module docstring for the expected npz format.")
        return

    X_eval, y_eval = load_features(args.features)
    matched = [k for k in keys if k in X_eval]
    if not matched:
        raise ValueError("No feature keys match the weight keys; they must be identical "
                         f"(weights: {keys[:3]}..., features: {list(X_eval)[:3]}...)")
    print(f"\nMasking-out ablation on {len(matched)} layers, "
          f"{len(y_eval)} eval samples...")
    mask_df = masking_ablation(layers, ranking, X_eval, y_eval,
                               args.percentages, args.metric)
    mask_path = os.path.join(args.output_dir, f"{label}_ablation_masking.csv")
    mask_df.to_csv(mask_path, index=False)
    print(f"Saved {mask_path}")
    plot_ablation(mask_df, args.metric, args.focus_percent, label,
                  args.output_dir, kind="masking")

    pivot = mask_df.pivot_table(index="layer", columns=["percent_kept", "which"],
                                values=args.metric)
    print(f"\nMasking-out ablation ({args.metric}), paper Table 2 style "
          f"(majority baseline: {mask_df[f'majority_{args.metric}'].iloc[0]:.3f}):")
    print(pivot.round(3).to_string())

    if args.retrain:
        if not args.train_features:
            print("\n--retrain requires --train_features; skipping the retraining ablation.")
        elif args.retrain_backend == "siren" and not args.probes:
            print("\n--retrain_backend siren requires --probes; skipping the "
                  "retraining ablation.")
        else:
            X_train, y_train = load_features(args.train_features)
            fit_predict = make_retrainer(args.retrain_backend, args.retrain_C,
                                         args.retrain_l1_ratio, best_Cs, args.seed)
            print(f"\nRetraining ablation ({args.retrain_backend}, "
                  f"{len(y_train)} train samples)...")
            retrain_df = retraining_ablation(layers, ranking, X_train, y_train,
                                             X_eval, y_eval, args.percentages,
                                             args.metric, fit_predict)
            retrain_path = os.path.join(args.output_dir,
                                        f"{label}_ablation_retrain.csv")
            retrain_df.to_csv(retrain_path, index=False)
            print(f"Saved {retrain_path}")
            plot_ablation(retrain_df, args.metric, args.focus_percent, label,
                          args.output_dir, kind="retrain")


if __name__ == "__main__":
    main()
