"""
Prefetch all datasets and model weights into the HF cache (and data/ for the
benchmark CSVs). Run this on a login node with internet access so that offline
compute nodes only need to read the cache.

    python cluster/download_assets.py [--skip_instruct]
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse

from datasets import load_dataset
from huggingface_hub import snapshot_download

from train.preprocess import _load_csv, preprocess_dataset, HARMBENCH_CSV_URL, ADVBENCH_CSV_URL
from utils.config import MODEL_CONFIGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_instruct", action="store_true",
                        help="Skip the gated meta-llama Instruct model")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Config names from utils/config.py to prefetch "
                             "(default: llama3-8b-sft llama3-8b-instruct)")
    parser.add_argument("--paper_datasets", type=str, nargs="+", default=None,
                        help="Paper datasets to prefetch (e.g. wildguard aegis2). "
                             "Some are gated on HF and need access + HF_TOKEN.")
    args = parser.parse_args()

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        print(f"HF cache: {hf_home}")
    else:
        print("WARNING: HF_HOME is not set — downloads will go to ~/.cache/huggingface,")
        print("which on Euler fills your $HOME quota. Set it first, e.g.:")
        print("  export HF_HOME=/cluster/scratch/$USER/hf_cache")
        if input("Continue anyway? [y/N] ").strip().lower() != "y":
            sys.exit(1)

    print("Downloading HH-RLHF (harmless-base)...")
    load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base")

    print("Downloading Alpaca (safe negatives for HarmBench/AdvBench eval)...")
    load_dataset("tatsu-lab/alpaca")

    print("Downloading HarmBench behaviors CSV...")
    _load_csv(HARMBENCH_CSV_URL)

    print("Downloading AdvBench behaviors CSV...")
    _load_csv(ADVBENCH_CSV_URL)

    if args.paper_datasets:
        for ds_name in args.paper_datasets:
            print(f"Prefetching paper dataset: {ds_name} ...")
            try:
                preprocess_dataset(ds_name)
            except Exception as e:
                print(f"ERROR prefetching {ds_name}: {e}")
                print("If this is a gating error, accept the dataset terms on its "
                      "HF page and make sure HF_TOKEN is set.")
                raise

    if args.models:
        models = args.models
    else:
        models = ["llama3-8b-sft"]
        if not args.skip_instruct:
            models.append("llama3-8b-instruct")

    for name in models:
        repo_id = MODEL_CONFIGS[name]["model_path"]
        print(f"Downloading model weights: {repo_id} ...")
        try:
            # original/* holds Meta's raw-format checkpoints (consolidated*.pth,
            # ~16GB) which transformers never loads; skip them.
            snapshot_download(repo_id, ignore_patterns=["original/*", "*.pth"])
        except Exception as e:
            print(f"ERROR downloading {repo_id}: {e}")
            if "meta-llama" in repo_id:
                print("This model is gated. Request access at "
                      f"https://huggingface.co/{repo_id} and set HF_TOKEN.")
            raise

    print("\nAll assets cached.")


if __name__ == "__main__":
    main()
