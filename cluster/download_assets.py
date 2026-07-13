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

from train.preprocess import _load_csv, HARMBENCH_CSV_URL, ADVBENCH_CSV_URL
from utils.config import MODEL_CONFIGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_instruct", action="store_true",
                        help="Skip the gated meta-llama Instruct model")
    args = parser.parse_args()

    print("Downloading HH-RLHF (harmless-base)...")
    load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base")

    print("Downloading Alpaca (safe negatives for HarmBench/AdvBench eval)...")
    load_dataset("tatsu-lab/alpaca")

    print("Downloading HarmBench behaviors CSV...")
    _load_csv(HARMBENCH_CSV_URL)

    print("Downloading AdvBench behaviors CSV...")
    _load_csv(ADVBENCH_CSV_URL)

    models = ["llama3-8b-sft"]
    if not args.skip_instruct:
        models.append("llama3-8b-instruct")

    for name in models:
        repo_id = MODEL_CONFIGS[name]["model_path"]
        print(f"Downloading model weights: {repo_id} ...")
        try:
            snapshot_download(repo_id)
        except Exception as e:
            print(f"ERROR downloading {repo_id}: {e}")
            if "meta-llama" in repo_id:
                print("This model is gated. Request access at "
                      f"https://huggingface.co/{repo_id} and set HF_TOKEN.")
            raise

    print("\nAll assets cached.")


if __name__ == "__main__":
    main()
