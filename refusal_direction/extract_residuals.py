"""Extract residual-stream hidden states at the last prompt token.

The refusal direction lives in the residual stream, not in MLP-neuron space,
so this is a different tensor from cka/extract_activations.py: for every
prompt it stores hidden_states[h][last real token] for every hidden-state
index h = 0..num_layers (h = 0 is the embedding output, h = l+1 the output
of decoder block l).

The last prompt token is the position right after the chat template's
assistant header -- the "post-instruction" position where Arditi et al. found
the difference-in-means direction to be strongest. It is also where the model
commits to refusing or complying.

Output:
  {out_dir}/{tag}_{position}.npy       float16 [N, num_layers+1, d_model]
  {out_dir}/{tag}_{position}.meta.csv  label, n_tokens, dataset, text

Size is small: 1024 prompts x 33 x 4096 x 2 bytes = 0.28 GB.
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(HERE, "residuals")


@torch.no_grad()
def extract(texts, model, tokenizer, device, position, max_length, batch_size):
    num_hidden = len(model.model.layers) + 1
    d_model = model.config.hidden_size
    out = np.zeros((len(texts), num_hidden, d_model), dtype=np.float16)
    token_counts = np.zeros(len(texts), dtype=np.int32)

    for start in range(0, len(texts), batch_size):
        chunk = [t if (isinstance(t, str) and t.strip()) else " "
                 for t in texts[start:start + batch_size]]
        batch = tokenizer(chunk, return_tensors="pt", truncation=True,
                          max_length=max_length, padding=True,
                          add_special_tokens=False)
        batch = {k: v.to(device) for k, v in batch.items()}
        lengths = batch["attention_mask"].sum(dim=1)

        result = model(**batch, output_hidden_states=True)
        # hidden_states: tuple of num_hidden tensors [B, T, D]
        for h, states in enumerate(result.hidden_states):
            states = states.float()
            if position == "last":
                rows = states[torch.arange(states.shape[0], device=device),
                              lengths - 1]
            else:  # mean over real tokens
                m = batch["attention_mask"].unsqueeze(-1).to(states.dtype)
                rows = (states * m).sum(dim=1) / lengths.unsqueeze(-1).float()
            out[start:start + len(chunk), h] = rows.cpu().numpy().astype(np.float16)
        token_counts[start:start + len(chunk)] = lengths.cpu().numpy()
        if (start // batch_size) % 10 == 0:
            print(f"  {min(start + batch_size, len(texts))}/{len(texts)}",
                  flush=True)
    return out, token_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--prompts", required=True,
                        help="CSV from build_direction_prompts.py (or any CSV "
                             "with formatted_input and label columns).")
    parser.add_argument("--model_path",
                        default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--text_column", default="formatted_input",
                        help="'formatted_input' (chat-templated; tokenized "
                             "with add_special_tokens=False) is what the "
                             "direction should be fitted on.")
    parser.add_argument("--position", default="last", choices=["last", "mean"],
                        help="'last' matches Arditi et al.'s post-instruction "
                             "position and is the primary setting.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tag", default=None,
                        help="Output basename; defaults to the prompts filename.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tag = args.tag or os.path.splitext(os.path.basename(args.prompts))[0]
    os.makedirs(args.output_dir, exist_ok=True)

    frame = pd.read_csv(args.prompts)
    if args.text_column not in frame.columns:
        raise SystemExit(f"column '{args.text_column}' not in {args.prompts} "
                         f"(have {list(frame.columns)})")
    if args.limit:
        frame = frame.iloc[:args.limit].reset_index(drop=True)
    texts = frame[args.text_column].astype(str).tolist()

    print(f"{len(texts)} prompts | model={args.model_path} | device={device} "
          f"| position={args.position}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map={"": device},
        trust_remote_code=True)
    model.eval()

    resid, token_counts = extract(texts, model, tokenizer, device,
                                  args.position, args.max_length,
                                  args.batch_size)

    meta = pd.DataFrame({
        "label": frame["label"].astype(int) if "label" in frame else 0,
        "n_tokens": token_counts,
        "dataset": frame["dataset"] if "dataset" in frame else tag,
        "text": frame["text"] if "text" in frame else texts,
    })

    # Atomic rename, .npy last -- same completeness convention as cka/.
    resid_path = os.path.join(args.output_dir, f"{tag}_{args.position}.npy")
    meta_path = os.path.join(args.output_dir, f"{tag}_{args.position}.meta.csv")
    tmp_resid = f"{resid_path}.{os.getpid()}.tmp.npy"
    tmp_meta = f"{meta_path}.{os.getpid()}.tmp"
    np.save(tmp_resid, resid)
    meta.to_csv(tmp_meta, index=False)
    os.replace(tmp_meta, meta_path)
    os.replace(tmp_resid, resid_path)

    print(f"\nSaved {resid_path}  ({resid.nbytes / 1e9:.2f} GB)")
    print(f"Saved {meta_path}")


if __name__ == "__main__":
    main()
