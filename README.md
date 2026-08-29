# SIREN — Safety Neuron Identification

SIREN trains one L1-regularized linear probe per transformer layer on hidden
activations (harmful vs. harmless prompts) and ranks each layer's features by
probe |weight|. All experiments use the standardized **HarmBench (harmful) +
Alpaca (safe)** benchmark in `data_files/` and Llama-3-8B-Instruct with
`mlpneuron_mean` pooling (the 14336-dim input to `mlp.down_proj`, i.e. real
FFN neurons — index-comparable with the other methods).

## Repo layout

| Path | Contents |
|---|---|
| `train/` | `preprocess.py` (dataset loaders), `probe_trainer.py` (L1 probe), `train_general_siren.py` (main trainer), `run_standard_siren.sh` (normal run), `run_clean_stability_siren.sh` (3 stability runs) |
| `cluster/` | `setup_env.sh` + `download_assets.py` (one-time env/model setup), `train_standard_siren.sbatch`, `train_clean_stability_siren.sbatch` |
| `analysis/` | `plot_layer_probes.py` (F1 + neuron count per layer), `plot_siren_neurons.py` (counts per threshold or exact top-N budget), `export_topn_neurons.py` (exact top-N JSONs), `clean_stability_intersection.py` (3-run intersection + plots). Outputs land in `analysis/results/` |
| `utils/` | `config.py` (model configs), `model_hooks.py` (activation extraction hooks) |
| `data_files/` | Standardized, chat-templated CSVs: `{harmbench,alpaca}_{train,val,test}.csv`, the cleaned `alpaca_train-clean.csv`, and the stability thirds `{harmbench,alpaca}_train_split{1,2,3}.csv`. Self-contained |
| `results/` | Other methods' neuron selections (Svea/Wang/Tengerleg pkls, intersection CSVs) for comparisons. Self-contained |
| `cka/` | CKA representation-similarity analysis; has its own README. Self-contained |

## One-time cluster setup

```bash
bash cluster/setup_env.sh        # conda env + HF cache + model prefetch
```

## 1. Normal training (clean Alpaca, mlpneuron)

```bash
SIREN_ALPACA_TRAIN=alpaca_train-clean \
C_VALUES="1000.0 5000.0 20000.0" THRESHOLDS="0.3 0.6 0.9" \
OUTPUT_SUFFIX="-std-mlpneuron_mean-clean" \
  sbatch cluster/train_standard_siren.sbatch llama3-8b-instruct mlpneuron_mean
```

Trains on `harmbench_train` + `alpaca_train-clean` (val/test: standard files).
Probes are saved to
`train/probes/llama3-8b-instruct_general_probes-std-mlpneuron_mean-clean.pkl`.
`OUTPUT_SUFFIX` keeps runs from overwriting each other.

## 2. Top-N neuron JSONs (no retraining needed)

Budgets 459 / 2294 / 4588 / 9175 = 0.1 / 0.5 / 1 / 2 % of the FFN space
(32 layers x 14336 neurons). Selections live in the saved probe weights, so
this runs in seconds on a login node:

```bash
python analysis/export_topn_neurons.py --model llama3-8b-instruct \
    --pooling_type mlpneuron_mean --suffix=-std-mlpneuron_mean-clean \
    --targets 459 2294 4588 9175
```

Writes `analysis/results/..._selected_neurons_top{N}.json`
(`{"layer0": [neuron indices], ...}`, descending importance per layer).

## 3. The two diagrams (normal run)

```bash
# probe F1 (left) + neuron count per layer (right)
python analysis/plot_layer_probes.py --models llama3-8b-instruct \
    --pooling_type mlpneuron_mean --suffix=-std-mlpneuron_mean-clean --threshold 0.9

# neuron counts per layer at the exact budgets, one curve per N
python analysis/plot_siren_neurons.py --model llama3-8b-instruct \
    --pooling_type mlpneuron_mean --suffix=-std-mlpneuron_mean-clean \
    --top_n 459 2294 4588 9175
```

(`plot_siren_neurons.py` also supports the cumulative-importance view via
`--thresholds 0.3 0.6 0.9` instead of `--top_n`.)

## 4. Stability training (3 clean train thirds)

```bash
sbatch cluster/train_clean_stability_siren.sbatch llama3-8b-instruct mlpneuron_mean
```

Trains one probe set per split on `{harmbench,alpaca}_train_split{i}`
(shared standard val/test), saving
`train/probes/..._general_probes-clean_stability-mlpneuron_mean-split{1,2,3}.pkl`.

## 5. Stability intersection: test results, top-N neurons, and plots

```bash
python analysis/clean_stability_intersection.py --model llama3-8b-instruct \
    --pooling_type mlpneuron_mean --targets 459 2294 4588 9175
```

Prints each split model's mean val/test F1 and saves to `analysis/results/`
(all names contain `clean_stability`):

- `..._clean_stability_probe_f1.csv` — per-layer val/test F1 for all 3 models
- `..._clean_stability_intersection_top{N}.json` — the N neurons most robustly
  selected by ALL three runs (worst-rank tuned intersection)
- `..._clean_stability_intersection_summary.csv` — per-run budget M per target
- `..._clean_stability_layer_probes.png` — 3-split F1 (left) + intersection
  counts per budget (right)
- `..._clean_stability_intersection_counts.png` — standalone counts plot

## Fetching outputs to your laptop

```bash
scp "euler:~/SIREN/analysis/results/*clean*" ~/Downloads/siren-analysis-results/
```
