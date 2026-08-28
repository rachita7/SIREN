"""Extract the full per-neuron MLP activation matrix for a prompt set.

Activation definition
---------------------
For every layer l we capture the input to `mlp.down_proj`:

    h_l = SiLU(W_gate x) * (W_up x)          in R^14336

This is the vector each method's (layer, neuron_index) pair indexes, and it is
byte-for-byte the same hook utils/model_hooks.py registers for
`mlpneuron_mean`. Because every method is scored from this one tensor, the
comparison cannot be contaminated by differing activation conventions.

Pooling
-------
`--pooling mean` averages over the prompt's real (non-pad) tokens, matching
SIREN's `mlpneuron_mean`, and is the primary setting.
`--pooling last` takes the final real token, as a robustness check. Mean
pooling divides by the token count and so leaks prompt length into every
neuron; run_cka.py can regress that out, but the last-token variant is a
useful independent check that the conclusions do not hinge on it.

Output
------
{out_dir}/{tag}_{pooling}.npy    float16 [N, num_layers, intermediate_size]
{out_dir}/{tag}_{pooling}.meta.csv  label, n_tokens, dataset, text

Size: 2000 prompts x 32 x 14336 x 2 bytes = 1.8 GB. Storing every neuron (not
just the selected ones) is deliberate: it lets every downstream analysis --
any method, any neuron budget, any random-control seed, the whole cross-layer
sweep -- rerun on CPU without touching the GPU again.
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
DEFAULT_OUTPUT_DIR = os.path.join(HERE, "activations")


class MlpNeuronExtractor:
    """Captures the down_proj input of every layer via forward pre-hooks."""

    def __init__(self, model_path, device="cuda", dtype=torch.bfloat16,
                 max_length=512):
        self.device = device
        self.max_length = max_length
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map={"": device},
            trust_remote_code=True)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                       trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.layers = self.model.model.layers
        self.num_layers = len(self.layers)
        self.intermediate_size = self.layers[0].mlp.down_proj.in_features
        self._captured = [None] * self.num_layers
        self._hooks = []

    def _hook(self, layer_idx):
        def fn(module, inputs):
            self._captured[layer_idx] = inputs[0].detach()
        return fn

    def __enter__(self):
        for i, layer in enumerate(self.layers):
            self._hooks.append(
                layer.mlp.down_proj.register_forward_pre_hook(self._hook(i)))
        return self

    def __exit__(self, *exc):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    @torch.no_grad()
    def pooled(self, texts, pooling, add_special_tokens):
        """([B, num_layers, intermediate_size] float32, [B] token counts)."""
        texts = [t if (isinstance(t, str) and t.strip()) else " " for t in texts]
        batch = self.tokenizer(texts, return_tensors="pt", truncation=True,
                              max_length=self.max_length, padding=True,
                              add_special_tokens=add_special_tokens)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        self._captured = [None] * self.num_layers
        self.model(**batch)

        mask = batch["attention_mask"]
        lengths = mask.sum(dim=1)
        out = torch.empty(mask.shape[0], self.num_layers, self.intermediate_size,
                          dtype=torch.float32, device=self.device)
        for layer_idx, acts in enumerate(self._captured):
            acts = acts.float()
            if pooling == "mean":
                m = mask.unsqueeze(-1).to(acts.dtype)
                out[:, layer_idx] = (acts * m).sum(dim=1) / lengths.unsqueeze(-1)
            elif pooling == "last":
                out[:, layer_idx] = acts[torch.arange(acts.shape[0]),
                                         lengths - 1]
            else:
                raise ValueError(f"unknown pooling '{pooling}'")
        return out.cpu().numpy(), lengths.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--prompts", required=True,
                        help="CSV from cka/build_prompts.py (or any CSV with a "
                             "text column and a label column).")
    parser.add_argument("--model_path",
                        default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--text_column", default="formatted_input",
                        help="'formatted_input' (chat-templated -> tokenized "
                             "with add_special_tokens=False) or 'text' (raw).")
    parser.add_argument("--pooling", default="mean", choices=["mean", "last"])
    parser.add_argument("--batch_size", type=int, default=8,
                        help="8 keeps peak GPU memory near 20GB on Llama-3-8B.")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Matches SIREN's extraction setting.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--tag", default=None,
                        help="Output basename; defaults to the prompts filename.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0,
                        help="Debug: only process the first N prompts.")
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

    # Chat-templated text already contains literal <|begin_of_text|> etc., so a
    # second BOS must not be added -- same rule as the rest of the repo.
    add_special_tokens = args.text_column != "formatted_input"

    print(f"{len(texts)} prompts | model={args.model_path} | device={device} | "
          f"pooling={args.pooling} | add_special_tokens={add_special_tokens}")

    with MlpNeuronExtractor(args.model_path, device=device,
                            max_length=args.max_length) as extractor:
        n_layers = extractor.num_layers
        width = extractor.intermediate_size
        print(f"  captured space: {n_layers} layers x {width} neurons "
              f"({n_layers * width} total)")

        acts = np.zeros((len(texts), n_layers, width), dtype=np.float16)
        token_counts = np.zeros(len(texts), dtype=np.int32)
        for start in range(0, len(texts), args.batch_size):
            end = min(start + args.batch_size, len(texts))
            pooled, lengths = extractor.pooled(texts[start:end], args.pooling,
                                               add_special_tokens)
            acts[start:end] = pooled.astype(np.float16)
            token_counts[start:end] = lengths
            if (start // args.batch_size) % 25 == 0:
                print(f"  {end}/{len(texts)}", flush=True)

    n_overflow = int(np.isinf(acts.astype(np.float32)).sum())
    if n_overflow:
        print(f"  WARNING: {n_overflow} values overflowed float16 storage.")

    acts_path = os.path.join(args.output_dir, f"{tag}_{args.pooling}.npy")
    np.save(acts_path, acts)

    meta = pd.DataFrame({
        "label": frame["label"].astype(int) if "label" in frame else 0,
        "n_tokens": token_counts,
        "dataset": frame["dataset"] if "dataset" in frame else tag,
        "text": frame["text"] if "text" in frame else texts,
    })
    meta_path = os.path.join(args.output_dir, f"{tag}_{args.pooling}.meta.csv")
    meta.to_csv(meta_path, index=False)

    print(f"\nSaved {acts_path}  ({acts.nbytes / 1e9:.2f} GB)")
    print(f"Saved {meta_path}")
    print(f"  token counts: mean={token_counts.mean():.1f} "
          f"min={token_counts.min()} max={token_counts.max()}")
    if "label" in frame:
        for label, sub in meta.groupby("label"):
            print(f"  class {label}: n={len(sub)} "
                  f"mean tokens={sub['n_tokens'].mean():.1f}")
        print("  (a large per-class token-count gap is exactly the confound "
              "the length-residualized CKA variant controls for)")


if __name__ == "__main__":
    main()
