"""
ToxicChat: https://huggingface.co/datasets/lmsys/toxic-chat
OpenAIModeration: https://huggingface.co/datasets/walledai/openai-moderation-dataset
Aegis: https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0
Aegis 2.0: https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0
SimpleSafetyTests: https://huggingface.co/datasets/Bertievidgen/SimpleSafetyTests           # Only has test set
HarmBench: https://huggingface.co/datasets/walledai/HarmBench                               # Only has harmful examples
WildguardTest: https://huggingface.co/datasets/allenai/wildguardmix
SafeRLHF: https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF
Beavertails: https://huggingface.co/datasets/PKU-Alignment/BeaverTails
XSTest: https://huggingface.co/datasets/Paul/XSTest       # Only has test set
HH-RLHF: https://huggingface.co/datasets/Anthropic/hh-rlhf                                  # harmless-base; chosen=safe / rejected=harmful
HarmBench (CSV): https://github.com/centerforaisafety/HarmBench                             # Only has harmful examples
AdvBench (CSV): https://github.com/llm-attacks/llm-attacks                                  # Only has harmful examples
"""
import os
import random

from datasets import Dataset, DatasetDict, load_dataset

HARMBENCH_CSV_URL = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
ADVBENCH_CSV_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
_DATA_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def preprocess_dataset(dataset: str, val_ratio: float = 0.2) -> DatasetDict:
    if dataset == "toxic_chat":
        return _preprocess_toxic_chat(val_ratio)
    elif dataset == "openai_moderation":
        return _preprocess_openai_moderation(val_ratio)
    elif dataset == "aegis":
        return _preprocess_aegis(val_ratio)
    elif dataset == "aegis2":
        return _preprocess_aegis2(val_ratio)
    elif dataset == "wildguard":
        return _preprocess_wildguard(val_ratio)
    elif dataset == "safe_rlhf":
        return _preprocess_safe_rlhf(val_ratio)
    elif dataset == "beavertails":
        return _preprocess_beavertails(val_ratio)
    elif dataset == "xstest":
        return _preprocess_xstest(val_ratio)
    elif dataset == "hh_rlhf":
        return _preprocess_hh_rlhf(val_ratio)
    elif dataset == "harmbench":
        return _preprocess_harmbench(val_ratio)
    elif dataset == "advbench":
        return _preprocess_advbench(val_ratio)
    else:
        raise ValueError(f"Invalid dataset: {dataset}")


def _preprocess_toxic_chat(val_ratio: float = 0.2) -> DatasetDict:
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124")

    split = ds["train"].train_test_split(test_size=val_ratio, shuffle=True)
    out = DatasetDict(
        {
            "train": split["train"],
            "validation": split["test"],
            "test": ds["test"],
        }
    )

    for k in out.keys():
        out[k] = out[k].rename_column("user_input", "text")
        out[k] = out[k].rename_column("toxicity", "label")

    return out


def _preprocess_openai_moderation(val_ratio: float = 0.2) -> DatasetDict:
    ds = load_dataset("walledai/openai-moderation-dataset")

    test_val_size = val_ratio * 2
    split_60_40 = ds["train"].train_test_split(test_size=test_val_size, shuffle=True)
    split_20_20 = split_60_40["test"].train_test_split(test_size=0.5, shuffle=True)
    out = DatasetDict(
        {
            "train": split_60_40["train"],
            "validation": split_20_20["train"],
            "test": split_20_20["test"],
        }
    )

    label_cols = ["S", "H", "V", "HR", "SH", "S3", "H2", "V2"]

    def to_binary_label(batch):
        n = len(batch[label_cols[0]])
        labels = []
        for i in range(n):
            val = 0
            for c in label_cols:
                if batch[c][i] == 1:
                    val = 1
                    break
            labels.append(val)
        return {"label": labels}

    for k in out.keys():
        out[k] = out[k].rename_column("prompt", "text")
        out[k] = out[k].map(to_binary_label, batched=True)

    return out


def _preprocess_aegis(val_ratio: float = 0.2) -> DatasetDict:
    ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-1.0")

    split = ds["train"].train_test_split(test_size=val_ratio, shuffle=True)
    out = DatasetDict(
        {
            "train": split["train"],
            "validation": split["test"],
            "test": ds["test"],
        }
    )

    label_cols = ["labels_0", "labels_1", "labels_2", "labels_3", "labels_4"]

    def convert_and_vote(batch):
        n = len(batch[label_cols[0]])
        cols = {c: [] for c in label_cols}
        final_labels = []

        for i in range(n):
            safe_cnt = 0
            unsafe_cnt = 0

            for c in label_cols:
                v = batch[c][i]
                if v is None:
                    cols[c].append(None)
                    continue
                if v == "Safe":
                    cols[c].append(0)
                    safe_cnt += 1
                else:
                    cols[c].append(1)
                    unsafe_cnt += 1

            if safe_cnt == 0 and unsafe_cnt == 0:
                final_labels.append(None)
            else:
                # tie -> 1
                final_labels.append(1 if unsafe_cnt >= safe_cnt else 0)

        cols["label"] = final_labels
        return cols

    for k in out.keys():
        out[k] = out[k].map(convert_and_vote, batched=True)
        none_count = out[k].filter(lambda x: x["label"] is None).num_rows
        assert none_count == 0

    return out


def _preprocess_aegis2(val_ratio: float = 0.2) -> DatasetDict:
    """
    Aegis 2.0 (Nemotron Content Safety Dataset V2). Full official splits are
    30,007 train / 1,445 validation / 1,964 test. AEGIS2_MAX_TRAIN caps the
    TRAIN split only (validation/test kept full so metrics stay comparable);
    default 10000 keeps the job well within the 18h wall time. Set to 0 to use
    the full train split.
    """
    ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0")

    train_split = ds["train"]
    max_train = int(os.environ.get("AEGIS2_MAX_TRAIN", "10000"))
    if max_train and max_train < train_split.num_rows:
        train_split = train_split.shuffle(seed=42).select(range(max_train))

    out = DatasetDict(
        {
            "train": train_split,
            "validation": ds["validation"],
            "test": ds["test"],
        }
    )

    # Keep only rows with a usable binary prompt label.
    out = DatasetDict({k: v.filter(lambda x: x["prompt_label"] in ("safe", "unsafe"))
                       for k, v in out.items()})

    def to_binary_label(example):
        return {"label": 1 if example["label"] == "unsafe" else 0}

    for k in out.keys():
        out[k] = out[k].rename_column("prompt", "text")
        out[k] = out[k].rename_column("prompt_label", "label")
        out[k] = out[k].map(to_binary_label)

    return out


def _preprocess_wildguard(val_ratio: float = 0.2) -> DatasetDict:
    train_ds = load_dataset("allenai/wildguardmix", "wildguardtrain")
    test_ds = load_dataset("allenai/wildguardmix", "wildguardtest")

    train_split = train_ds["train"]
    test_split = test_ds["test"]

    split = train_split.train_test_split(test_size=val_ratio, shuffle=True)
    out = DatasetDict({"train": split["train"], "validation": split["test"], "test": test_split})

    def to_binary_label(example):
        v = example["label"]
        if v is None:
            return {"label": None}
        if v == "unharmful":
            return {"label": 0}
        if v == "harmful":
            return {"label": 1}
        raise ValueError(f"Invalid dataset: {v}")

    for k in out.keys():
        out[k] = out[k].rename_column("prompt", "text")
        out[k] = out[k].rename_column("prompt_harm_label", "label")
        out[k] = out[k].map(to_binary_label)
        out[k] = out[k].filter(lambda x: x["label"] is not None)

    return out


def _preprocess_safe_rlhf(val_ratio: float = 0.2) -> DatasetDict:
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", "default")

    split = ds["train"].train_test_split(test_size=val_ratio, shuffle=True)
    out = DatasetDict(
        {
            "train": split["train"],
            "validation": split["test"],
            "test": ds["test"],
        }
    )

    def to_binary_label(example):
        v = example["label"]
        if v is True:
            label = 0
        elif v is False:
            label = 1
        else:
            raise ValueError(f"Invalid dataset: {v}")

        return {"label": label, "text": f"{example['prompt']}\n{example['response_0']}"}

    for k in out.keys():
        out[k] = out[k].rename_column("is_response_0_safe", "label")
        out[k] = out[k].map(to_binary_label)

    return out


def _preprocess_beavertails(val_ratio: float = 0.2) -> DatasetDict:
    ds = load_dataset("PKU-Alignment/BeaverTails")
    base = DatasetDict({"train": ds["30k_train"], "test": ds["30k_test"]})

    split = base["train"].train_test_split(test_size=val_ratio, shuffle=True)
    out = DatasetDict({"train": split["train"], "validation": split["test"], "test": base["test"]})

    def to_label_and_text(example):
        v = example["label"]
        if v is True:
            label = 0
        elif v is False:
            label = 1
        else:
            raise ValueError(f"Invalid dataset: {v}")
        return {"label": label, "text": f"{example['prompt']}\n{example['response']}"}

    for k in out.keys():
        out[k] = out[k].rename_column("is_safe", "label")
        out[k] = out[k].map(to_label_and_text)

    return out


def _preprocess_xstest(val_ratio: float = 0.2) -> DatasetDict:
    ds = load_dataset("Paul/XSTest")

    test_val_size = val_ratio * 2
    split_60_40 = ds["train"].train_test_split(test_size=test_val_size, shuffle=True)
    split_20_20 = split_60_40["test"].train_test_split(test_size=0.5, shuffle=True)
    out = DatasetDict(
        {
            "train": split_60_40["train"],
            "validation": split_20_20["train"],
            "test": split_20_20["test"],
        }
    )

    def to_binary_label(example):
        v = example["label"]
        if v == "safe":
            return {"label": 0}
        if v == "unsafe":
            return {"label": 1}
        raise ValueError(f"Invalid dataset: {v}")

    for k in out.keys():
        out[k] = out[k].rename_column("prompt", "text")
        out[k] = out[k].map(to_binary_label)

    return out


def _first_exchange(transcript: str) -> str:
    """Keep only the first Human/Assistant exchange of an HH-RLHF transcript."""
    assistant_pos = transcript.find("\n\nAssistant:")
    if assistant_pos == -1:
        return transcript.strip()
    next_human = transcript.find("\n\nHuman:", assistant_pos)
    if next_human == -1:
        return transcript.strip()
    return transcript[:next_human].strip()


def _preprocess_hh_rlhf(val_ratio: float = 0.2) -> DatasetDict:
    """
    Anthropic HH-RLHF, harmless-base split. Weak labels: the first exchange of
    the `chosen` transcript is treated as safe (0) and of the `rejected`
    transcript as harmful (1). Pairs whose first exchanges are identical
    (preference differs only in later turns) are dropped.

    HH_RLHF_MAX_ROWS caps the number of preference pairs sampled from the train
    split (each pair yields 2 examples). Default 8000 -> 16k train+val texts.
    """
    max_rows = int(os.environ.get("HH_RLHF_MAX_ROWS", "8000"))
    ds = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base")

    def build(split, cap=None):
        rows = ds[split]
        indices = list(range(len(rows)))
        random.Random(42).shuffle(indices)
        if cap is not None:
            indices = indices[:cap]
        texts, labels = [], []
        for i in indices:
            row = rows[int(i)]
            chosen = _first_exchange(row["chosen"])
            rejected = _first_exchange(row["rejected"])
            if not chosen or not rejected or chosen == rejected:
                continue
            texts.extend([chosen, rejected])
            labels.extend([0, 1])
        return Dataset.from_dict({"text": texts, "label": labels})

    train_full = build("train", cap=max_rows)
    split = train_full.train_test_split(test_size=val_ratio, shuffle=True, seed=42)
    return DatasetDict(
        {
            "train": split["train"],
            "validation": split["test"],
            "test": build("test"),
        }
    )


def _load_csv(url: str):
    """Read a CSV, caching the file under data/ so compute nodes without
    internet access can reuse a copy fetched on the login node."""
    import pandas as pd

    os.makedirs(_DATA_CACHE_DIR, exist_ok=True)
    local_path = os.path.join(_DATA_CACHE_DIR, os.path.basename(url))
    if not os.path.exists(local_path):
        import urllib.request
        urllib.request.urlretrieve(url, local_path)
    df = pd.read_csv(local_path)
    return df


def _safe_prompts_alpaca(n: int, seed: int = 42) -> list:
    ds = load_dataset("tatsu-lab/alpaca")["train"]
    ds = ds.filter(lambda x: x["input"] == "")
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    return [ds[int(i)]["instruction"] for i in indices[:n]]


def _harmful_only_eval_dataset(harmful_prompts: list) -> DatasetDict:
    """Build an eval-only DatasetDict from a harmful-only prompt list.

    HarmBench/AdvBench contain no safe examples, so by default the eval set is
    harmful-only and recall == detection rate is the primary metric
    (F1/precision are degenerate). Set HARM_EVAL_ADD_SAFE=1 to mix in an equal
    number of safe Alpaca instructions (label 0) so F1/precision are meaningful.
    """
    texts = list(harmful_prompts)
    labels = [1] * len(texts)
    if os.environ.get("HARM_EVAL_ADD_SAFE", "0") == "1":
        safe = _safe_prompts_alpaca(len(texts))
        texts.extend(safe)
        labels.extend([0] * len(safe))
    d = Dataset.from_dict({"text": texts, "label": labels})
    return DatasetDict({"train": d, "validation": d, "test": d})


def _preprocess_harmbench(val_ratio: float = 0.2) -> DatasetDict:
    df = _load_csv(HARMBENCH_CSV_URL)
    if "FunctionalCategory" in df.columns:
        df = df[df["FunctionalCategory"] == "standard"]
    prompts = df["Behavior"].dropna().astype(str).tolist()
    return _harmful_only_eval_dataset(prompts)


def _preprocess_advbench(val_ratio: float = 0.2) -> DatasetDict:
    df = _load_csv(ADVBENCH_CSV_URL)
    prompts = df["goal"].dropna().astype(str).tolist()
    return _harmful_only_eval_dataset(prompts)
