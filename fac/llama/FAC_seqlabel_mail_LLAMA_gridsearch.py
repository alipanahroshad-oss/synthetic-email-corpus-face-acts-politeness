#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export HF_TOKEN=<your_hf_token>   # Llama 3.2 is gated: accept the licence on Hugging Face first
export CUDA_VISIBLE_DEVICES=0

nohup bash -lc '
for s in 13 21 42 87 100; do
  echo "===== RUN_SEED=$s ====="
  RUN_SEED=$s python FAC_seqlabel_mail_LLAMA_gridsearch.py
done
' > seqlabel_req_or_rep_llama_lora_multiseed.log 2>&1 &

tail -f seqlabel_req_or_rep_llama_lora_multiseed.log
"""
# FAC_seqlabel_mail_LLAMA_gridsearch.py

import os
import re
import ast
import gc
import json
import math
import random
import shutil
import inspect
import textwrap
import itertools
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

print("SeqLabel + Req-or-Rep — LLaMA-3.2-1B + LoRA + BCE — validation micro-F1 grid search")

# Multi-seed control: keep the train/val/test split fixed, but vary training randomness.
RUN_SEED = int(os.environ.get("RUN_SEED", "42"))

CFG = {
    "train_csv": "data/splits/train_seed42.csv",
    "val_csv":   "data/splits/val_seed42.csv",
    "test_csv":  "data/splits/test_seed42.csv",

    "TEXT_COL":     "covered_text",
    "LABEL_COL":    "GoldFaceAct",
    "EMAIL_COL":    "email_id",
    "SENTIDX_COL":  "sentence_idx",
    "ISREQ_COL":    "is_request",
    "SEED_COL":     "seed",
    "PAIR_COL":     "pair_idx",

    "filter_is_request": None,

    "LABELS": ["HNeg+","HNeg-","HPos+","HPos-","Neutral","SNeg+","SNeg-","SPos+","SPos-"],

    # --- model ---
    "model_name": "meta-llama/Llama-3.2-1B",
    "max_length": 1024,

    # --- LoRA ---
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],

    # --- output ---
    "output_dir": f"outputs/llama32_1b_lora_seqlabel_req_or_rep_xent_runseed{RUN_SEED}",

    # --- grid ---
    "lr_grid": [5e-5, 1e-4, 2e-4],
    "epochs_grid": [3, 5, 8],
    "batch_grid": [2, 4],
    "dropout_grid": [0.1, 0.2],

    "weight_decay": 0.0,
    "early_stopping_patience": 2,
    "effective_batch_size": 8,

    # --- threshold search on validation ---
    "tau_grid_start": 0.05,
    "tau_grid_end": 0.95,
    "tau_grid_steps": 91,

    # --- precision ---
    "use_bf16": True,

    "seed": RUN_SEED,
    "debug_sequences": 3,
    "save_test_predictions": True,
    "save_trial_models": False,
}

os.makedirs(CFG["output_dir"], exist_ok=True)
print(f"[INFO] RUN_SEED={RUN_SEED}")
print(f"[INFO] Output dir: {CFG['output_dir']}")
torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"]); random.seed(CFG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG["seed"])

LABELS = CFG["LABELS"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for i, l in enumerate(LABELS)}

# ---- detect correct tokenizer kwarg for this transformers version ----
_TRAINER_PARAMS = set(inspect.signature(Trainer.__init__).parameters.keys())
if "processing_class" in _TRAINER_PARAMS:
    TOK_KW = "processing_class"
elif "tokenizer" in _TRAINER_PARAMS:
    TOK_KW = "tokenizer"
else:
    TOK_KW = None
print(f"[INFO] Trainer tokenizer kwarg detected: {TOK_KW}")

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def sigmoid_stable(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    np.clip(x, -50.0, 50.0, out=x)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)

def parse_labels(cell, _row_idx: int = -1) -> List[str]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    s = str(cell).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = ast.literal_eval(s)
            return [str(x).strip() for x in arr]
        except Exception as e:
            print(f"[WARN] parse_labels: literal_eval failed for row {_row_idx}: {s!r} ({e}); falling back to comma split")
    return [t.strip() for t in re.split(r"[;,]", s) if t.strip()]

def to_multi_hot(names: List[str]) -> np.ndarray:
    vec = np.zeros(len(LABELS), dtype=np.float32)
    for n in names:
        if n in label2id:
            vec[label2id[n]] = 1.0
    return vec

def load_split(csv_path: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    need = [cfg["TEXT_COL"], cfg["LABEL_COL"], cfg["EMAIL_COL"], cfg["SENTIDX_COL"],
            cfg["ISREQ_COL"], cfg["SEED_COL"], cfg["PAIR_COL"]]
    for c in need:
        assert c in df.columns, f"Column '{c}' not found in {csv_path}"
    if cfg["filter_is_request"] in (0, 1):
        df = df[df[cfg["ISREQ_COL"]] == cfg["filter_is_request"]].copy()
    for c in [cfg["EMAIL_COL"], cfg["SENTIDX_COL"], cfg["ISREQ_COL"], cfg["SEED_COL"], cfg["PAIR_COL"]]:
        df[c] = df[c].astype(int)
    df["gold_list"] = [parse_labels(v, i) for i, v in enumerate(df[cfg["LABEL_COL"]].tolist())]
    df["y_vec"] = df["gold_list"].map(to_multi_hot)
    df[cfg["TEXT_COL"]] = df[cfg["TEXT_COL"]].astype(str).fillna("")
    unknown_rows = sum(1 for names in df["gold_list"] if any(n not in label2id for n in names))
    if unknown_rows > 0:
        print(f"[WARN] {unknown_rows} rows in {os.path.basename(csv_path)} contained at least one unknown label name (silently dropped from y_vec)")
    return df.reset_index(drop=True)

df_train = load_split(CFG["train_csv"], CFG)
df_val   = load_split(CFG["val_csv"],   CFG)
df_test  = load_split(CFG["test_csv"],  CFG)
print(f"[INFO] Loaded rows: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

def check_split_integrity(df_train, df_val, df_test, seed_col):
    s_tr = set(df_train[seed_col].unique())
    s_va = set(df_val[seed_col].unique())
    s_te = set(df_test[seed_col].unique())
    print("\n[SEED GROUPS]")
    print(f"Train ({len(s_tr)}): {sorted(list(s_tr))}")
    print(f"Val   ({len(s_va)}): {sorted(list(s_va))}")
    print(f"Test  ({len(s_te)}): {sorted(list(s_te))}")
    print("\n[Overlap checks]")
    for a,b,name in [(s_tr,s_va,"Train ∩ Val"),(s_tr,s_te,"Train ∩ Test"),(s_va,s_te,"Val ∩ Test")]:
        ov = sorted(list(a & b))
        print(f"{name} ({len(ov)}): {ov}")
    if (s_tr & s_va) or (s_tr & s_te) or (s_va & s_te):
        print("[LEAKAGE WARNING] Some seeds appear in multiple splits!")
    else:
        print("[OK] No seeds in common across splits.")

check_split_integrity(df_train, df_val, df_test, CFG["SEED_COL"])

# ---------------------------------------------------------------------------
# Tokenizer — RIGHT padding (we need explicit per-sentence positions).
# ---------------------------------------------------------------------------
tok = AutoTokenizer.from_pretrained(CFG["model_name"], use_fast=True)
assert tok.is_fast, "Use a fast tokenizer."
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
tok.padding_side = "right"
print(f"[INFO] tokenizer pad_token={tok.pad_token!r} (id={tok.pad_token_id}); padding_side={tok.padding_side}")

# ---------------------------------------------------------------------------
# Diagnostic: how long are typical emails after tokenization?
# (No truncation here — just measure.)
# ---------------------------------------------------------------------------
def measure_email_lengths(df: pd.DataFrame, cfg: Dict[str, Any], split_name: str):
    lens = []
    nsents = []
    for eid, grp in df.sort_values([cfg["EMAIL_COL"], cfg["SENTIDX_COL"]]).groupby(cfg["EMAIL_COL"], sort=False):
        texts = grp[cfg["TEXT_COL"]].astype(str).tolist()
        joined = " ".join(texts)
        n_tokens = len(tok(joined, add_special_tokens=False)["input_ids"])
        lens.append(n_tokens)
        nsents.append(len(texts))
    arr = np.array(lens)
    print(f"[DIAG][{split_name}] email-level token-length stats (no truncation): "
          f"n={len(arr)}, min={arr.min()}, p50={int(np.percentile(arr,50))}, "
          f"p90={int(np.percentile(arr,90))}, p99={int(np.percentile(arr,99))}, "
          f"max={arr.max()}; sentence-count mean={np.mean(nsents):.1f}")
    return arr

_ = measure_email_lengths(df_train, CFG, "train")
_ = measure_email_lengths(df_val,   CFG, "val")
_ = measure_email_lengths(df_test,  CFG, "test")

# ---------------------------------------------------------------------------
# Build per-email tokenized examples by tokenizing each sentence individually,
# then concatenating token IDs with a single space-token in between sentences,
# and recording each sentence's last-token POSITION directly during build.
#
# This avoids any offset_mapping / char-range matching, so "lost" can only
# mean genuine overflow (sentence didn't fit in the remaining budget after
# BOS + earlier sentences + per-sentence separators).
# ---------------------------------------------------------------------------
def _tokenize_no_specials(s: str) -> List[int]:
    return tok(s, add_special_tokens=False)["input_ids"]

# Pre-tokenize a single space-token (used as separator between sentences in
# the concatenated sequence). LLaMA tokenizers turn " " into a single special
# Ġ token in BPE; we keep it as the literal sequence the tokenizer produces.
SEP_IDS: List[int] = _tokenize_no_specials(" ")
if not SEP_IDS:
    # Defensive: if the tokenizer eats lone spaces, use empty separator.
    SEP_IDS = []
print(f"[INFO] Inter-sentence separator tokens: {SEP_IDS} (len={len(SEP_IDS)})")

# BOS token (LLaMA always has one)
BOS_ID = tok.bos_token_id
if BOS_ID is None:
    print("[WARN] No BOS token; sequences will start with the first content token.")
    PRELUDE: List[int] = []
else:
    PRELUDE = [BOS_ID]
print(f"[INFO] BOS prelude: {PRELUDE}")

def encode_email_sentences(sentence_texts: List[str], max_length: int):
    """
    Build the input_ids for an email by tokenizing each sentence and concatenating
    them with SEP_IDS in between, prefixed by PRELUDE (BOS).

    Returns:
        input_ids                 : list of token ids, length <= max_length
        attention_mask            : 1s for every position (no padding here)
        sentence_end_positions    : last-token absolute positions (in input_ids)
                                    for each KEPT sentence, in order
        kept_sentence_indices     : which input sentences were kept (oldest first)

    A sentence is "kept" only if at least one of its tokens fits in the budget.
    If a sentence is partially included (its tail truncated by max_length), we
    still keep it and use the last token actually present as its end position.

    If even one token of a sentence wouldn't fit, the sentence is dropped, and
    no further sentences are added (preserves chronological prefix).
    """
    input_ids: List[int] = list(PRELUDE)
    sentence_end_positions: List[int] = []
    kept: List[int] = []

    for s_idx, s_text in enumerate(sentence_texts):
        # Per-sentence token IDs.
        s_ids = _tokenize_no_specials(s_text)
        if len(s_ids) == 0:
            # Empty sentence; nothing to add. Still mark as kept with end at
            # the last existing position (or skip if there's nothing yet).
            if input_ids:
                kept.append(s_idx)
                sentence_end_positions.append(len(input_ids) - 1)
            else:
                # No tokens at all yet (no PRELUDE, empty first sentence).
                # Skip — there's no valid "end" position to record.
                pass
            continue

        # Add separator before this sentence (except the very first content sentence).
        # We add the separator only if there is already at least one content token.
        # PRELUDE alone counts as "no content yet".
        already_have_content = (len(input_ids) > len(PRELUDE))
        sep = SEP_IDS if already_have_content else []

        # Compute remaining budget.
        budget = max_length - len(input_ids)
        addition_required = len(sep) + 1   # need at least the separator + one content token
        if budget < addition_required:
            # No room for even a partial sentence. Stop here.
            break

        # How many tokens of s_ids can we fit?
        room_for_s = max_length - len(input_ids) - len(sep)
        if room_for_s < 1:
            break
        take = s_ids[:room_for_s]

        # Append.
        input_ids.extend(sep)
        start_pos = len(input_ids)
        input_ids.extend(take)
        end_pos = len(input_ids) - 1   # last token of this sentence

        kept.append(s_idx)
        sentence_end_positions.append(end_pos)

        # If we couldn't fit the full sentence, that's the last sentence we'll add.
        if len(take) < len(s_ids):
            break

    attention_mask = [1] * len(input_ids)
    return input_ids, attention_mask, sentence_end_positions, kept

class SeqLabelLlamaDataset(Dataset):
    """
    Per-email seqlabel dataset using direct sentence-by-sentence tokenization.
    """
    def __init__(self, df: pd.DataFrame, tokenizer, cfg: Dict[str, Any], split_name: str = ""):
        self.examples = []
        self.split_name = split_name
        self._fully_dropped_seqs = 0
        self._partial_truncated_seqs = 0
        self._sentences_dropped_total = 0
        self._sentences_input_total = 0
        self._sentences_kept_total = 0

        for eid, grp in df.sort_values([cfg["EMAIL_COL"], cfg["SENTIDX_COL"]]).groupby(cfg["EMAIL_COL"], sort=False):
            grp_sorted = grp.sort_values(cfg["SENTIDX_COL"])
            texts = grp_sorted[cfg["TEXT_COL"]].astype(str).tolist()
            labels = [np.array(v, dtype=np.float32) for v in grp_sorted["y_vec"].tolist()]
            sentidx_list = grp_sorted[cfg["SENTIDX_COL"]].astype(int).tolist()
            is_request_v = int(grp_sorted[cfg["ISREQ_COL"]].iloc[0])
            seed_v = int(grp_sorted[cfg["SEED_COL"]].iloc[0])
            pair_v = int(grp_sorted[cfg["PAIR_COL"]].iloc[0])

            n_in = len(texts)
            self._sentences_input_total += n_in

            input_ids, attn, sent_ends, kept = encode_email_sentences(texts, cfg["max_length"])
            n_kept = len(kept)
            self._sentences_kept_total += n_kept
            self._sentences_dropped_total += (n_in - n_kept)

            if n_kept == 0:
                self._fully_dropped_seqs += 1
                continue
            if n_kept < n_in:
                self._partial_truncated_seqs += 1

            labels_kept = [labels[i] for i in kept]
            sentidx_kept = [sentidx_list[i] for i in kept]

            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": attn,
                "sentence_ends": sent_ends,
                "labels": labels_kept,
                "email_id": int(eid),
                "is_request": is_request_v,
                "seed": seed_v,
                "pair_idx": pair_v,
                "sentence_idx_list": sentidx_kept,
            })

        if self._fully_dropped_seqs:
            print(f"[WARN][{split_name}] Dropped {self._fully_dropped_seqs} sequences with zero surviving sentences after truncation.")
        if self._partial_truncated_seqs:
            print(f"[WARN][{split_name}] {self._partial_truncated_seqs} sequences had partial truncation; "
                  f"{self._sentences_dropped_total}/{self._sentences_input_total} sentences total lost "
                  f"({100.0 * self._sentences_dropped_total / max(1, self._sentences_input_total):.2f}%).")
        else:
            print(f"[OK][{split_name}] No partial truncation. {self._sentences_kept_total}/{self._sentences_input_total} sentences kept.")

    def truncation_report(self) -> Dict[str, int]:
        return {
            "split": self.split_name,
            "n_sequences_kept": len(self.examples),
            "n_sequences_fully_dropped": int(self._fully_dropped_seqs),
            "n_sequences_partial_truncation": int(self._partial_truncated_seqs),
            "sentences_input_total": int(self._sentences_input_total),
            "sentences_kept_total": int(self._sentences_kept_total),
            "sentences_dropped_total": int(self._sentences_dropped_total),
            "pct_sentences_dropped": float(
                100.0 * self._sentences_dropped_total / max(1, self._sentences_input_total)
            ),
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        e = self.examples[idx]
        labels = torch.tensor(np.stack(e["labels"], axis=0), dtype=torch.float32)
        return {
            "input_ids": torch.tensor(e["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(e["attention_mask"], dtype=torch.long),
            "sentence_ends": torch.tensor(e["sentence_ends"], dtype=torch.long),
            "labels": labels,
        }

train_ds = SeqLabelLlamaDataset(df_train, tok, CFG, split_name="train")
val_ds   = SeqLabelLlamaDataset(df_val,   tok, CFG, split_name="val")
test_ds  = SeqLabelLlamaDataset(df_test,  tok, CFG, split_name="test")

trunc_report = {
    "run_seed": int(CFG["seed"]),
    "train": train_ds.truncation_report(),
    "val":   val_ds.truncation_report(),
    "test":  test_ds.truncation_report(),
    "max_length": CFG["max_length"],
    "model_name": CFG["model_name"],
}
with open(os.path.join(CFG["output_dir"], "truncation_report.json"), "w", encoding="utf-8") as f:
    json.dump(trunc_report, f, indent=2)
print(f"[INFO] Truncation report -> {os.path.join(CFG['output_dir'], 'truncation_report.json')}")

# Hard guard: if more than 30% of test sentences are still being lost, abort.
test_pct_lost = trunc_report["test"]["pct_sentences_dropped"]
if test_pct_lost > 30.0:
    raise RuntimeError(
        f"[ABORT] Test split is still losing {test_pct_lost:.2f}% of sentences "
        f"to truncation at max_length={CFG['max_length']}. Bump max_length further."
    )
elif test_pct_lost > 5.0:
    print(f"[NOTE] Test split is losing {test_pct_lost:.2f}% of sentences to truncation. Acceptable but monitor.")
else:
    print(f"[OK] Test split sentence loss: {test_pct_lost:.2f}%.")

# Sanity-check: print a few examples to verify sentence_ends point at non-pad positions.
for split_name, ds in [("train", train_ds)]:
    for i in range(min(2, len(ds.examples))):
        ex = ds.examples[i]
        ids = ex["input_ids"]
        ends = ex["sentence_ends"]
        S = len(ex["labels"])
        print(f"[DEBUG][{split_name}] ex{i}: email_id={ex['email_id']} L={len(ids)} S={S}")
        for j, end in enumerate(ends[:3]):
            tok_str = tok.decode([ids[end]])
            print(f"  sentence {j} (orig sidx={ex['sentence_idx_list'][j]}): end_pos={end}, end_token={tok_str!r}")

# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------
class CollatorSeqLabelLlama:
    def __init__(self, pad_token_id: int, label_dim: int):
        self.pad_token_id = pad_token_id
        self.label_dim = label_dim

    def __call__(self, batch):
        max_len = max(x["input_ids"].shape[0] for x in batch)
        input_ids, attention_mask = [], []
        for x in batch:
            pad = max_len - x["input_ids"].shape[0]
            input_ids.append(F.pad(x["input_ids"], (0, pad), value=self.pad_token_id))
            attention_mask.append(F.pad(x["attention_mask"], (0, pad), value=0))
        input_ids = torch.stack(input_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)

        max_S = max(x["labels"].shape[0] for x in batch)
        label_grid, sent_end_grid = [], []
        for x in batch:
            S = x["labels"].shape[0]
            if S < max_S:
                pad_rows = torch.full((max_S - S, self.label_dim), -1.0, dtype=torch.float32)
                label_grid.append(torch.cat([x["labels"], pad_rows], dim=0))
                sent_end_grid.append(F.pad(x["sentence_ends"], (0, max_S - S), value=-1))
            else:
                label_grid.append(x["labels"])
                sent_end_grid.append(x["sentence_ends"])
        labels = torch.stack(label_grid, dim=0)
        sentence_ends = torch.stack(sent_end_grid, dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sentence_ends": sentence_ends,
            "labels": labels,
        }

collator = CollatorSeqLabelLlama(pad_token_id=tok.pad_token_id, label_dim=len(LABELS))

# ---------------------------------------------------------------------------
# Custom model
# ---------------------------------------------------------------------------
class LlamaSeqLabelClassifier(nn.Module):
    def __init__(self, base_model: nn.Module, hidden_size: int, num_labels: int, dropout: float):
        super().__init__()
        self.backbone = base_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, sentence_ends=None, labels=None, **_):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        H = out.last_hidden_state
        B, T, Hdim = H.shape

        valid_mask = (sentence_ends >= 0)
        safe_idx = sentence_ends.clamp(min=0)

        idx_expanded = safe_idx.unsqueeze(-1).expand(-1, -1, Hdim)
        sent_reps = torch.gather(H, dim=1, index=idx_expanded)

        logits = self.classifier(self.dropout(sent_reps))

        loss = None
        if labels is not None:
            label_valid = (labels[..., 0] != -1)
            full_mask = label_valid & valid_mask
            if full_mask.any():
                z = logits[full_mask, :]
                y = labels[full_mask, :]
                loss = self.loss_fn(z.float(), y.float())
            else:
                loss = torch.zeros([], device=logits.device, dtype=torch.float32)

        return {"loss": loss, "logits": logits}

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def flatten_valid(pred_logits: np.ndarray, labels: np.ndarray):
    pred_logits = np.asarray(pred_logits)
    labels = np.asarray(labels)
    padded_mask = (labels == -1).all(axis=-1)
    keep_mask = ~padded_mask
    probs2d = pred_logits.reshape(-1, pred_logits.shape[-1])[keep_mask.reshape(-1)]
    gold2d = labels.reshape(-1, labels.shape[-1])[keep_mask.reshape(-1)]
    gold2d = (gold2d > 0).astype(int)
    return probs2d, gold2d

def flatten_valid_np(pred_logits: np.ndarray, labels: np.ndarray):
    probs2d_raw, gold2d = flatten_valid(pred_logits, labels)
    return sigmoid_stable(probs2d_raw), gold2d

def compute_micro_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
    }

def search_best_global_tau_microf1(logits: np.ndarray, labels: np.ndarray) -> Tuple[float, Dict[str, float]]:
    probs, gold2d = flatten_valid_np(logits, labels)
    best_tau = 0.50
    best = None
    tau_grid = np.linspace(CFG["tau_grid_start"], CFG["tau_grid_end"], CFG["tau_grid_steps"])
    for tau in tau_grid:
        pred = (probs >= float(tau)).astype(int)
        m = compute_micro_metrics(gold2d, pred)
        row = {"tau": float(tau), **m}
        if best is None:
            best_tau = float(tau); best = row
        else:
            if row["micro_f1"] > best["micro_f1"]:
                best_tau = float(tau); best = row
            elif math.isclose(row["micro_f1"], best["micro_f1"], rel_tol=0.0, abs_tol=1e-12):
                if row["micro_precision"] > best["micro_precision"]:
                    best_tau = float(tau); best = row
                elif math.isclose(row["micro_precision"], best["micro_precision"], rel_tol=0.0, abs_tol=1e-12):
                    if row["micro_recall"] > best["micro_recall"]:
                        best_tau = float(tau); best = row
    return best_tau, best

def compute_metrics_epoch(eval_pred):
    logits, labels = eval_pred
    best_tau, best = search_best_global_tau_microf1(logits, labels)
    return {
        "micro/f1_tau_best": best["micro_f1"],
        "micro/precision_tau_best": best["micro_precision"],
        "micro/recall_tau_best": best["micro_recall"],
        "best_tau": best_tau,
    }

def classwise_table(gold: np.ndarray, pred: np.ndarray, subset_name: str) -> pd.DataFrame:
    rows = []
    for j, lab in enumerate(LABELS):
        yt = gold[:, j].astype(int)
        yp = pred[:, j].astype(int)
        rows.append({
            "subset": subset_name,
            "label": lab,
            "support_gold": int(yt.sum()),
            "support_pred": int(yp.sum()),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "gold_absent": bool(yt.sum() == 0),
            "pred_absent": bool(yp.sum() == 0),
        })
    return pd.DataFrame(rows)

class TableLoggerCallback(TrainerCallback):
    def __init__(self):
        self.rows = []
        self.last_train_loss = None
        self._printed = False
        self._seen_epochs = set()
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.last_train_loss = float(logs["loss"])
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        metrics = metrics or {}
        g = lambda k: metrics.get(f"eval_{k}", metrics.get(k))
        epoch = int(round(state.epoch or 0)) if state.epoch is not None else (len(self.rows)+1)
        if epoch in self._seen_epochs:
            return
        self._seen_epochs.add(epoch)
        row = {
            "epoch": epoch,
            "train_loss": self.last_train_loss,
            "val_loss": g("loss"),
            "micro/f1_tau_best": g("micro/f1_tau_best"),
            "micro/precision_tau_best": g("micro/precision_tau_best"),
            "micro/recall_tau_best": g("micro/recall_tau_best"),
            "best_tau": g("best_tau"),
        }
        self.rows.append(row)
        def fmt(x): return "—" if x is None else f"{x:.6f}"
        if not self._printed:
            print("\nEpoch\tTraining Loss\tValidation Loss\tMicro/f1_bestTau\tMicro/precision_bestTau\tMicro/recall_bestTau\tBestTau")
            self._printed = True
        print(f"{row['epoch']}\t{fmt(row['train_loss'])}\t{fmt(row['val_loss'])}\t{fmt(row['micro/f1_tau_best'])}\t{fmt(row['micro/precision_tau_best'])}\t{fmt(row['micro/recall_tau_best'])}\t{fmt(row['best_tau'])}")
    def on_train_end(self, args, state, control, **kwargs):
        if self.rows:
            out_csv = os.path.join(args.output_dir, "epoch_log_summary.csv")
            pd.DataFrame(self.rows).to_csv(out_csv, index=False)

# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_model(classifier_dropout: float):
    config = AutoConfig.from_pretrained(CFG["model_name"])
    config.pad_token_id = tok.pad_token_id

    backbone = AutoModel.from_pretrained(
        CFG["model_name"],
        config=config,
        torch_dtype=torch.bfloat16 if CFG["use_bf16"] else torch.float32,
    )
    backbone.config.pad_token_id = tok.pad_token_id

    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=CFG["lora_r"],
        lora_alpha=CFG["lora_alpha"],
        lora_dropout=CFG["lora_dropout"],
        target_modules=CFG["lora_target_modules"],
        bias="none",
    )
    backbone = get_peft_model(backbone, lora_cfg)

    hidden_size = config.hidden_size
    model = LlamaSeqLabelClassifier(
        base_model=backbone,
        hidden_size=hidden_size,
        num_labels=len(LABELS),
        dropout=classifier_dropout,
    )

    for name, param in model.named_parameters():
        if name.startswith("classifier"):
            param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Trainable params: {n_trainable:,} / {n_total:,} ({100.0 * n_trainable / n_total:.4f}%)")
    return model

# ---------------------------------------------------------------------------
# Custom save / load
# ---------------------------------------------------------------------------
def _write_adapter_and_head(model_ref: "LlamaSeqLabelClassifier", target_dir: str):
    adapter_dir = os.path.join(target_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    try:
        model_ref.backbone.save_pretrained(adapter_dir)
    except Exception as e:
        print(f"[WARN] adapter save failed at {adapter_dir}: {e}")
    head_path = os.path.join(target_dir, "cls_head.pt")
    try:
        head_state = {
            "classifier.weight": model_ref.classifier.weight.detach().cpu(),
            "classifier.bias": model_ref.classifier.bias.detach().cpu(),
        }
        torch.save(head_state, head_path)
    except Exception as e:
        print(f"[WARN] classifier head save failed at {head_path}: {e}")

class SaveAdapterAndHeadCallback(TrainerCallback):
    """Write adapter + head into every kept checkpoint dir."""
    def __init__(self, model_ref):
        self.model_ref = model_ref
    def on_save(self, args, state, control, **kwargs):
        if not os.path.isdir(args.output_dir):
            return
        kept_dirs = []
        for n in os.listdir(args.output_dir):
            if n.startswith("checkpoint-") and os.path.isdir(os.path.join(args.output_dir, n)):
                kept_dirs.append(os.path.join(args.output_dir, n))
        if not kept_dirs:
            return
        for d in kept_dirs:
            _write_adapter_and_head(self.model_ref, d)
    def on_train_end(self, args, state, control, **kwargs):
        best = state.best_model_checkpoint
        if best and os.path.isdir(best):
            adapter_dir = os.path.join(best, "adapter")
            head_path = os.path.join(best, "cls_head.pt")
            if not (os.path.isdir(adapter_dir) and os.path.isfile(head_path)):
                _write_adapter_and_head(self.model_ref, best)

def load_best_into(model: LlamaSeqLabelClassifier, best_checkpoint_dir: str):
    adapter_dir = os.path.join(best_checkpoint_dir, "adapter")
    head_path = os.path.join(best_checkpoint_dir, "cls_head.pt")
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"Missing adapter dir at {adapter_dir}")
    if not os.path.isfile(head_path):
        raise FileNotFoundError(f"Missing classifier head at {head_path}")

    base = AutoModel.from_pretrained(
        CFG["model_name"],
        torch_dtype=torch.bfloat16 if CFG["use_bf16"] else torch.float32,
    )
    base.config.pad_token_id = tok.pad_token_id
    backbone = PeftModel.from_pretrained(base, adapter_dir)
    model.backbone = backbone

    head_state = torch.load(head_path, map_location="cpu")
    with torch.no_grad():
        model.classifier.weight.copy_(head_state["classifier.weight"])
        model.classifier.bias.copy_(head_state["classifier.bias"])
    return model

def select_better_trial(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if b is None:
        return True
    if a["val_micro_f1_best_tau"] > b["val_micro_f1_best_tau"]:
        return True
    if a["val_micro_f1_best_tau"] < b["val_micro_f1_best_tau"]:
        return False
    if a["val_micro_precision_best_tau"] > b["val_micro_precision_best_tau"]:
        return True
    if a["val_micro_precision_best_tau"] < b["val_micro_precision_best_tau"]:
        return False
    if a["val_micro_recall_best_tau"] > b["val_micro_recall_best_tau"]:
        return True
    return False

def make_trainer_kwargs(extra: Dict[str, Any]) -> Dict[str, Any]:
    if TOK_KW is not None:
        extra[TOK_KW] = tok
    return extra

# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------
grid_rows = []
best_trial = None
best_trial_dir = None
trial_id = 0

for lr, epochs, batch_size, dropout in itertools.product(
    CFG["lr_grid"], CFG["epochs_grid"], CFG["batch_grid"], CFG["dropout_grid"]
):
    trial_id += 1
    trial_name = f"trial_{trial_id:03d}_lr{lr}_ep{epochs}_bs{batch_size}_do{dropout}"
    trial_dir = os.path.join(CFG["output_dir"], "grid_trials", trial_name)
    os.makedirs(trial_dir, exist_ok=True)
    print(f"\n[GRID] {trial_name}")

    cleanup_memory()
    model = build_model(classifier_dropout=dropout)

    grad_accum = max(1, CFG["effective_batch_size"] // batch_size)

    args = TrainingArguments(
        output_dir=trial_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=CFG["weight_decay"],
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_micro/f1_tau_best",
        greater_is_better=True,
        bf16=CFG["use_bf16"] and torch.cuda.is_available(),
        fp16=False,
        seed=CFG["seed"],
        report_to=[],
        logging_steps=50,
        save_total_limit=2,
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
        gradient_checkpointing=False,
    )

    trainer_kwargs = make_trainer_kwargs(dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics_epoch,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=CFG["early_stopping_patience"]),
            TableLoggerCallback(),
            SaveAdapterAndHeadCallback(model),
        ],
    ))
    trainer = Trainer(**trainer_kwargs)

    trainer.train()

    best_checkpoint_dir = trainer.state.best_model_checkpoint or ""
    if not best_checkpoint_dir:
        raise RuntimeError(f"No best checkpoint found for {trial_name}")

    cleanup_memory()
    eval_model = build_model(classifier_dropout=dropout)
    eval_model = load_best_into(eval_model, best_checkpoint_dir)
    eval_model.eval()

    eval_args = TrainingArguments(
        output_dir=os.path.join(trial_dir, "eval_tmp"),
        per_device_eval_batch_size=batch_size,
        report_to=[],
        bf16=CFG["use_bf16"] and torch.cuda.is_available(),
        fp16=False,
        remove_unused_columns=False,
    )
    eval_trainer_kwargs = make_trainer_kwargs(dict(
        model=eval_model,
        args=eval_args,
        data_collator=collator,
    ))
    eval_trainer = Trainer(**eval_trainer_kwargs)

    val_out = eval_trainer.predict(val_ds)
    val_logits = val_out.predictions
    val_labels = val_out.label_ids
    val_tau_best, val_best = search_best_global_tau_microf1(val_logits, val_labels)

    trial_row = {
        "trial_id": trial_id,
        "trial_name": trial_name,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "dropout": dropout,
        "weight_decay": CFG["weight_decay"],
        "checkpoint_metric_micro_f1_best_tau": float(trainer.state.best_metric) if trainer.state.best_metric is not None else np.nan,
        "val_best_tau": float(val_tau_best),
        "val_micro_f1_best_tau": float(val_best["micro_f1"]),
        "val_micro_precision_best_tau": float(val_best["micro_precision"]),
        "val_micro_recall_best_tau": float(val_best["micro_recall"]),
        "best_checkpoint_dir": best_checkpoint_dir,
    }
    grid_rows.append(trial_row)

    if select_better_trial(trial_row, best_trial):
        best_trial = trial_row
        best_trial_dir = trial_dir

    pd.DataFrame(grid_rows).to_csv(
        os.path.join(CFG["output_dir"], "grid_search_results_partial.csv"),
        index=False,
    )

    del trainer, eval_trainer, model, eval_model, val_out, val_logits, val_labels
    cleanup_memory()

# ---------------------------------------------------------------------------
# Aggregate grid results
# ---------------------------------------------------------------------------
grid_df = pd.DataFrame(grid_rows).sort_values(
    ["val_micro_f1_best_tau", "val_micro_precision_best_tau", "val_micro_recall_best_tau"],
    ascending=[False, False, False]
).reset_index(drop=True)
grid_csv = os.path.join(CFG["output_dir"], "grid_search_results.csv")
grid_df.to_csv(grid_csv, index=False)

best_trial["run_seed"] = int(CFG["seed"])
best_json = os.path.join(CFG["output_dir"], "best_config.json")
with open(best_json, "w", encoding="utf-8") as f:
    json.dump(best_trial, f, indent=2)

print(f"\n[BEST] {best_trial['trial_name']}")
print(json.dumps(best_trial, indent=2))

# ---------------------------------------------------------------------------
# Final test evaluation with best validation-learned tau
# ---------------------------------------------------------------------------
best_ckpt = best_trial["best_checkpoint_dir"]
best_tau = float(best_trial["val_best_tau"])
best_bs = int(best_trial["batch_size"])
best_dropout = float(best_trial["dropout"])

cleanup_memory()
final_model = build_model(classifier_dropout=best_dropout)
final_model = load_best_into(final_model, best_ckpt)
final_model.eval()

final_args = TrainingArguments(
    output_dir=os.path.join(CFG["output_dir"], "final_eval_tmp"),
    per_device_eval_batch_size=best_bs,
    report_to=[],
    bf16=CFG["use_bf16"] and torch.cuda.is_available(),
    fp16=False,
    remove_unused_columns=False,
)
final_trainer_kwargs = make_trainer_kwargs(dict(
    model=final_model,
    args=final_args,
    data_collator=collator,
))
final_trainer = Trainer(**final_trainer_kwargs)

test_out = final_trainer.predict(test_ds)
test_logits = test_out.predictions
test_labels = test_out.label_ids

def flatten_pred_gold_isreq_per_example(ds: SeqLabelLlamaDataset, logits, tau: float):
    logits = np.asarray(logits)
    preds_list, golds_list, isreq_list, meta_rows = [], [], [], []
    for i, ex in enumerate(ds.examples):
        S_b = len(ex["labels"])
        if S_b == 0:
            continue
        probs_b = sigmoid_stable(logits[i][:S_b])
        pred_b = (probs_b >= tau).astype(int)
        gold_b = (np.stack(ex["labels"][:S_b]) > 0.5).astype(int)
        is_req_b = np.array([int(ex["is_request"])] * S_b, dtype=int)
        preds_list.append(pred_b)
        golds_list.append(gold_b)
        isreq_list.append(is_req_b)
        for j in range(S_b):
            meta_rows.append({
                "email_id": int(ex["email_id"]),
                "is_request": int(ex["is_request"]),
                "seed": int(ex["seed"]),
                "pair_idx": int(ex["pair_idx"]),
                "sentence_idx": int(ex["sentence_idx_list"][j]),
            })
    pred2d = np.vstack(preds_list) if preds_list else np.zeros((0, len(LABELS)), dtype=int)
    gold2d = np.vstack(golds_list) if golds_list else np.zeros((0, len(LABELS)), dtype=int)
    is_req_flags = np.concatenate(isreq_list) if isreq_list else np.zeros((0,), dtype=int)
    meta_df = pd.DataFrame(meta_rows)
    assert pred2d.shape == gold2d.shape
    assert pred2d.shape[0] == is_req_flags.shape[0] == len(meta_df)
    return pred2d, gold2d, is_req_flags, meta_df

pred_all, gold_all, is_req_flags, meta_df = flatten_pred_gold_isreq_per_example(
    test_ds, test_logits, best_tau
)

diag = {
    "df_test_rows": int(len(df_test)),
    "df_test_request_rows": int((df_test[CFG["ISREQ_COL"]].astype(int) == 1).sum()),
    "df_test_reply_rows": int((df_test[CFG["ISREQ_COL"]].astype(int) == 0).sum()),
    "test_ds_sequences": int(len(test_ds.examples)),
    "evaluated_sentence_rows": int(len(gold_all)),
    "evaluated_request_rows": int((is_req_flags == 1).sum()),
    "evaluated_reply_rows": int((is_req_flags == 0).sum()),
    "best_tau": float(best_tau),
    "test_truncation_report": test_ds.truncation_report(),
}
with open(os.path.join(CFG["output_dir"], "row_count_diagnostics.json"), "w", encoding="utf-8") as f:
    json.dump(diag, f, indent=2)

meta_df.to_csv(os.path.join(CFG["output_dir"], "test_eval_sentence_metadata.csv"), index=False)

mask_req = is_req_flags == 1
mask_rep = is_req_flags == 0

rows = []
class_rows = []
for subset_name, mask in [
    ("OVERALL", np.ones(len(gold_all), dtype=bool)),
    ("REQUEST", mask_req),
    ("REPLY", mask_rep),
]:
    y_true = gold_all[mask]
    y_pred = pred_all[mask]
    m = compute_micro_metrics(y_true, y_pred)
    rows.append({
        "subset": subset_name,
        "n_rows": int(mask.sum()),
        "tau_used": float(best_tau),
        "micro_f1": float(m["micro_f1"]),
        "micro_precision": float(m["micro_precision"]),
        "micro_recall": float(m["micro_recall"]),
    })
    class_rows.append(classwise_table(y_true, y_pred, subset_name))

test_table = pd.DataFrame(rows)
class_df = pd.concat(class_rows, ignore_index=True)

test_table.to_csv(os.path.join(CFG["output_dir"], "test_micro_metrics_overall_request_reply.csv"), index=False)
class_df.to_csv(os.path.join(CFG["output_dir"], "test_classwise_f1_support_for_appendix.csv"), index=False)

tex_lines = [
    "\\begin{tabular}{lcccc}",
    "\\toprule",
    "\\textbf{Subset} & \\textbf{N} & \\textbf{mF1} & \\textbf{Micro-P} & \\textbf{Micro-R} \\\\",
    "\\midrule",
]
for _, r in test_table.iterrows():
    tex_lines.append(
        f"{r['subset'].title()} & {int(r['n_rows'])} & {r['micro_f1']:.3f} & {r['micro_precision']:.3f} & {r['micro_recall']:.3f} \\\\"
    )
tex_lines += ["\\bottomrule", "\\end{tabular}"]
with open(os.path.join(CFG["output_dir"], "test_micro_metrics_overall_request_reply.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(tex_lines) + "\n")

if CFG["save_test_predictions"]:
    rows_pred = []
    logits_np = np.asarray(test_logits)
    for i, ex in enumerate(test_ds.examples):
        S_b = len(ex["labels"])
        if S_b == 0:
            continue
        probs_S = sigmoid_stable(logits_np[i][:S_b])
        pred_S = (probs_S >= best_tau).astype(int)
        gold_S = (np.stack(ex["labels"][:S_b]) > 0.5).astype(int)
        S_ok = min(gold_S.shape[0], probs_S.shape[0], len(ex["sentence_idx_list"]))
        for j in range(S_ok):
            row = {
                "email_id": int(ex["email_id"]),
                "is_request": int(ex["is_request"]),
                "seed": int(ex["seed"]),
                "pair_idx": int(ex["pair_idx"]),
                "sentence_idx": int(ex["sentence_idx_list"][j]),
                "gold_labels": ",".join([l for k, l in enumerate(LABELS) if gold_S[j][k] == 1]),
                "best_val_tau": float(best_tau),
                f"pred_labels_tau_{best_tau:.2f}": ",".join([l for k, l in enumerate(LABELS) if pred_S[j][k] == 1]),
            }
            for k, lbl in enumerate(LABELS):
                row[f"pred_prob_{lbl}"] = float(probs_S[j][k])
            rows_pred.append(row)
    out_csv = os.path.join(CFG["output_dir"], "test_predictions_per_sentence_best_validation_tau.csv")
    pd.DataFrame(rows_pred).to_csv(out_csv, index=False)

manifest = {
    "run_seed": int(CFG["seed"]),
    "model": "LLaMA-3.2 + LoRA, custom multi-label sentence-classifier head with BCE-with-logits (xent)",
    "base_model": CFG["model_name"],
    "lora": {
        "r": CFG["lora_r"],
        "alpha": CFG["lora_alpha"],
        "dropout": CFG["lora_dropout"],
        "target_modules": CFG["lora_target_modules"],
    },
    "precision": "bf16" if CFG["use_bf16"] else "fp32",
    "max_length": CFG["max_length"],
    "selection_metric_for_hyperparameters": "validation micro-F1 at validation-learned tau",
    "checkpoint_metric_within_each_run": "validation micro-F1 at validation-learned tau",
    "threshold_learning": "best global validation tau maximizing validation micro-F1",
    "pooling": "last token of each sentence (decoder-only-friendly equivalent of [CLS]/mean-pool for BERT)",
    "tokenizer": {
        "pad_token": tok.pad_token,
        "pad_token_id": tok.pad_token_id,
        "padding_side": tok.padding_side,
    },
    "best_trial": best_trial,
    "truncation_report": trunc_report,
}
with open(os.path.join(CFG["output_dir"], "run_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

if not CFG["save_trial_models"]:
    trial_root = os.path.join(CFG["output_dir"], "grid_trials")
    if os.path.isdir(trial_root):
        for name in os.listdir(trial_root):
            p = os.path.join(trial_root, name)
            if os.path.abspath(p) != os.path.abspath(best_trial_dir):
                shutil.rmtree(p, ignore_errors=True)

print("\n=== FINAL TEST SUMMARY ===")
print(test_table.to_string(index=False))
print(f"\n[DONE] Wrote grid results          -> {grid_csv}")
print(f"[DONE] Wrote best config           -> {best_json}")
print(f"[DONE] Wrote test summary CSV      -> {os.path.join(CFG['output_dir'], 'test_micro_metrics_overall_request_reply.csv')}")
print(f"[DONE] Wrote classwise CSV         -> {os.path.join(CFG['output_dir'], 'test_classwise_f1_support_for_appendix.csv')}")
print(f"[DONE] Wrote LaTeX table           -> {os.path.join(CFG['output_dir'], 'test_micro_metrics_overall_request_reply.tex')}")
print(f"[DONE] Wrote truncation report     -> {os.path.join(CFG['output_dir'], 'truncation_report.json')}")
print(f"[DONE] Wrote row count diagnostics -> {os.path.join(CFG['output_dir'], 'row_count_diagnostics.json')}")
print(f"[DONE] Wrote run manifest          -> {os.path.join(CFG['output_dir'], 'run_manifest.json')}")
