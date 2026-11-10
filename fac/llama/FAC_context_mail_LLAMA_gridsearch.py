#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export HF_TOKEN=<your_hf_token>   # Llama 3.2 is gated: accept the licence on Hugging Face first
export CUDA_VISIBLE_DEVICES=2

nohup bash -lc '
for s in 13 21 42 87 100; do
  echo "===== RUN_SEED=$s ====="
  RUN_SEED=$s python FAC_context_mail_LLAMA_gridsearch.py
done
' > past_same_email_req_or_rep_llama_lora_multiseed.log 2>&1 &

tail -f past_same_email_req_or_rep_llama_lora_multiseed.log
"""
# FAC_context_mail_LLAMA_gridsearch.py

import os
import re
import ast
import gc
import json
import math
import random
import shutil
import inspect
import itertools
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
    set_seed,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

print("Past Same Email — Req-or-Rep — LLaMA-3.2_1B + LoRA + BCE — validation micro-F1 grid search")

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
    "max_length": 256,

    # --- TargetFirst formatting ---
    "context_prefix": "\n\nContext: ",     # joins target -> context
    "context_sep":    " ",                  # joins past sentences within context

    # --- LoRA ---
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],

    # --- output ---
    "output_dir": "outputs/llama32_1b_lora_past_same_email_req_or_rep_xent",

    # --- grid ---
    "lr_grid": [5e-5, 1e-4, 2e-4],
    "epochs_grid": [3, 5, 8],
    "batch_grid": [4, 8],
    "dropout_grid": [0.1, 0.2],

    "weight_decay": 0.0,
    "early_stopping_patience": 2,
    "effective_batch_size": 16,

    # --- threshold search on validation ---
    "tau_grid_start": 0.05,
    "tau_grid_end": 0.95,
    "tau_grid_steps": 91,

    # --- precision ---
    "use_bf16": True,

    "seed": 42,
    "save_test_predictions": True,
    "save_trial_models": False,
}

# ---------------------------------------------------------------------------
# Multi-seed control
# ---------------------------------------------------------------------------
# Keep the data split fixed (train_seed42/val_seed42/test_seed42),
# but change the *training* random seed via the RUN_SEED environment variable.
# Example:
#   RUN_SEED=13 python FAC_context_mail_LLAMA_gridsearch.py
# ---------------------------------------------------------------------------
BASE_OUTPUT_DIR = CFG["output_dir"]
RUN_SEED = int(os.environ.get("RUN_SEED", CFG["seed"]))
CFG["seed"] = RUN_SEED
CFG["output_dir"] = f"{BASE_OUTPUT_DIR}_runseed{RUN_SEED}"

os.makedirs(CFG["output_dir"], exist_ok=True)

def seed_everything(seed: int):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True

seed_everything(CFG["seed"])
print(f"[INFO] Training random seed: {CFG['seed']}")
print("[INFO] Data split is fixed: train_seed42 / val_seed42 / test_seed42")
print(f"[INFO] Output directory: {CFG['output_dir']}")

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
    # Defensive: cast text column to str so downstream string ops never barf on float NaN.
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
# Tokenizer — LEFT padding for sentence classification.
# (LLaMA's seq-cls head reads the LAST non-pad token at index -1.)
# ---------------------------------------------------------------------------
tok = AutoTokenizer.from_pretrained(CFG["model_name"], use_fast=True)
assert tok.is_fast, "Use a fast tokenizer."
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
tok.padding_side = "left"
print(f"[INFO] tokenizer pad_token={tok.pad_token!r} (id={tok.pad_token_id}); padding_side={tok.padding_side}")

# ---------------------------------------------------------------------------
# Build TargetFirst inputs WITH explicit context-truncation accounting.
#
# For each row in df:
#   target = row[TEXT_COL]
#   past_sentences_in_same_email_oldest_first = sentences with the same
#       email_id and a smaller sentence_idx, ordered oldest -> newest
#
# We then form input_ids manually so we can:
#   (a) ALWAYS keep the target intact (truncation="only_second" semantics),
#   (b) drop most-recent past sentences when budget is exceeded
#       (keep oldest, drop newest — preserves chronological prefix),
#   (c) count how many past sentences got dropped per row.
#
# We also pre-pad to max_length for shape stability across train/val/test.
# ---------------------------------------------------------------------------
def precompute_past_sentences(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[List[str]]:
    """For each row in df, return the list of past sentences (oldest first)."""
    out = [None] * len(df)
    for eid, grp in df.groupby(cfg["EMAIL_COL"], sort=False):
        # Sort within email by sentence_idx ascending.
        grp_sorted = grp.sort_values(cfg["SENTIDX_COL"])
        idx_list = grp_sorted.index.tolist()
        sidx_list = grp_sorted[cfg["SENTIDX_COL"]].astype(int).tolist()
        text_list = grp_sorted[cfg["TEXT_COL"]].astype(str).tolist()
        # For each row, past = all entries with sentence_idx strictly less than its own.
        for i, row_idx in enumerate(idx_list):
            past_texts = []
            target_sidx = sidx_list[i]
            for j, s_other in enumerate(sidx_list):
                if s_other < target_sidx:
                    past_texts.append(text_list[j])
                else:
                    break  # sorted; once >= target, no more past
            out[row_idx] = past_texts
    return out

class TargetFirstLlamaDataset(Dataset):
    """
    Pre-tokenizes every (target, past_sentences) row with KEEP-OLDEST truncation.

    Records per-row diagnostics:
      - n_past_sentences_input : number of past sentences available
      - n_past_sentences_kept  : number whose tokens (at least partially) made it
      - n_past_sentences_lost  : input - kept

    We tokenize the full assembled string with the tokenizer's max_length
    truncation, but we know which sentences would have been dropped because
    we tokenize each past sentence individually under a remaining budget and
    stop adding once it would exceed the budget.
    """
    def __init__(self, df: pd.DataFrame, past_lists: List[List[str]], tokenizer, cfg: Dict[str, Any], split_name: str = ""):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.split_name = split_name
        self.max_length = cfg["max_length"]
        self.context_prefix = cfg["context_prefix"]
        self.context_sep = cfg["context_sep"]

        # We pre-build all (input_ids, attention_mask) and per-row diagnostics.
        self._all_input_ids = []
        self._all_attn = []
        self._all_labels = []
        self._diag_rows = []

        # BOS handling: LLaMA tokenizer typically adds a single BOS when
        # add_special_tokens=True. We tokenize each piece without specials
        # and prepend BOS exactly once at assembly time.
        bos = tokenizer.bos_token_id
        if bos is None:
            # Fallback: some configs may not have BOS; we just skip prepending.
            bos_list = []
        else:
            bos_list = [bos]

        prefix_ids = tokenizer(self.context_prefix, add_special_tokens=False)["input_ids"]
        sep_ids    = tokenizer(self.context_sep, add_special_tokens=False)["input_ids"]

        n_total_past_input = 0
        n_total_past_kept  = 0
        n_total_past_lost  = 0
        n_rows_with_loss   = 0

        for i in range(len(self.df)):
            target_text = str(self.df.iloc[i][cfg["TEXT_COL"]])
            past_list = past_lists[i] or []

            # Tokenize the target without specials.
            target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]

            # Reserve room: BOS + target + (prefix only if there's any context).
            # Always cap target so it fits with at least 1 BOS slot.
            budget_for_target = self.max_length - len(bos_list) - 1  # leave 1 slot
            if len(target_ids) > budget_for_target:
                # If target alone overflows, hard-truncate it at the END.
                # (Per the BERT only_second policy, target is kept; here we
                # protect against the pathological case where target alone
                # is longer than max_length. We truncate the tail — the head
                # of the target is most informative for classification.)
                target_ids = target_ids[:budget_for_target]

            # Now decide whether we can include any context at all.
            base_used = len(bos_list) + len(target_ids)
            remaining = self.max_length - base_used

            n_past_input = len(past_list)
            n_past_kept = 0
            n_past_lost = 0

            ctx_ids: List[int] = []
            if n_past_input > 0 and remaining > len(prefix_ids) + 1:
                # We have room for the prefix and at least one context token.
                ctx_ids.extend(prefix_ids)
                budget = remaining - len(prefix_ids)

                # Add past sentences oldest-first; stop when adding the next
                # one would exceed the budget. This is the "keep oldest"
                # policy: we never drop early sentences in favor of later ones.
                for p_i, past_text in enumerate(past_list):
                    pids = tokenizer(past_text, add_special_tokens=False)["input_ids"]
                    addition = len(pids) + (len(sep_ids) if p_i > 0 else 0)
                    if addition <= budget:
                        if p_i > 0:
                            ctx_ids.extend(sep_ids)
                            budget -= len(sep_ids)
                        ctx_ids.extend(pids)
                        budget -= len(pids)
                        n_past_kept += 1
                    else:
                        # No room for this past sentence; the rest are also lost.
                        # (Conservative: even if a *later* sentence might fit,
                        # we want to preserve oldest-first contiguity. Stop here.)
                        n_past_lost = n_past_input - p_i
                        break
            else:
                # No room for any context. All past sentences are "lost"
                # (they weren't included). This is rare unless target is huge.
                n_past_lost = n_past_input
                ctx_ids = []

            # Assemble: BOS + target + context_with_prefix
            full_ids = bos_list + target_ids + ctx_ids
            assert len(full_ids) <= self.max_length, f"assembly overshot: {len(full_ids)} > {self.max_length}"

            # Pad to max_length on the LEFT (so last real token sits at index -1).
            pad_count = self.max_length - len(full_ids)
            if pad_count > 0:
                input_ids = [tokenizer.pad_token_id] * pad_count + full_ids
                attn = [0] * pad_count + [1] * len(full_ids)
            else:
                input_ids = full_ids
                attn = [1] * len(full_ids)

            self._all_input_ids.append(input_ids)
            self._all_attn.append(attn)
            self._all_labels.append(np.array(self.df.iloc[i]["y_vec"], dtype=np.float32))

            self._diag_rows.append({
                "row_idx": int(i),
                "email_id": int(self.df.iloc[i][cfg["EMAIL_COL"]]),
                "sentence_idx": int(self.df.iloc[i][cfg["SENTIDX_COL"]]),
                "is_request": int(self.df.iloc[i][cfg["ISREQ_COL"]]),
                "n_past_sentences_input": int(n_past_input),
                "n_past_sentences_kept": int(n_past_kept),
                "n_past_sentences_lost": int(n_past_lost),
            })

            n_total_past_input += n_past_input
            n_total_past_kept  += n_past_kept
            n_total_past_lost  += n_past_lost
            if n_past_lost > 0:
                n_rows_with_loss += 1

        # Aggregate diagnostics
        self._summary = {
            "split": split_name,
            "rows": int(len(self.df)),
            "rows_with_any_context_loss": int(n_rows_with_loss),
            "past_sentences_total_input": int(n_total_past_input),
            "past_sentences_total_kept":  int(n_total_past_kept),
            "past_sentences_total_lost":  int(n_total_past_lost),
            "pct_past_sentences_lost": float(
                100.0 * n_total_past_lost / max(1, n_total_past_input)
            ),
        }
        if n_rows_with_loss > 0:
            print(f"[INFO][{split_name}] {n_rows_with_loss}/{len(self.df)} rows had context truncation; "
                  f"{n_total_past_lost}/{n_total_past_input} past sentences lost "
                  f"({self._summary['pct_past_sentences_lost']:.2f}%).")
        else:
            print(f"[INFO][{split_name}] No context truncation occurred (all available past sentences fit).")

    def context_truncation_summary(self) -> Dict[str, Any]:
        return dict(self._summary)

    def context_truncation_per_row(self) -> List[Dict[str, Any]]:
        return list(self._diag_rows)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self._all_input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self._all_attn[idx], dtype=torch.long),
            "labels": torch.tensor(self._all_labels[idx], dtype=torch.float32),
        }

print("[INFO] Building past-sentence lists per row...")
past_train = precompute_past_sentences(df_train, CFG)
past_val   = precompute_past_sentences(df_val,   CFG)
past_test  = precompute_past_sentences(df_test,  CFG)

print("[INFO] Encoding train split...")
train_ds = TargetFirstLlamaDataset(df_train, past_train, tok, CFG, split_name="train")
print("[INFO] Encoding val split...")
val_ds   = TargetFirstLlamaDataset(df_val,   past_val,   tok, CFG, split_name="val")
print("[INFO] Encoding test split...")
test_ds  = TargetFirstLlamaDataset(df_test,  past_test,  tok, CFG, split_name="test")

# Persist context-truncation diagnostics up front.
ctx_trunc_report = {
    "train": train_ds.context_truncation_summary(),
    "val":   val_ds.context_truncation_summary(),
    "test":  test_ds.context_truncation_summary(),
    "max_length": CFG["max_length"],
    "model_name": CFG["model_name"],
    "policy": "KEEP OLDEST: target is kept intact; past sentences added oldest-first; once budget is exceeded, all remaining (newer) past sentences are dropped.",
}
with open(os.path.join(CFG["output_dir"], "context_truncation_report.json"), "w", encoding="utf-8") as f:
    json.dump(ctx_trunc_report, f, indent=2)
print(f"[INFO] Context truncation report -> {os.path.join(CFG['output_dir'], 'context_truncation_report.json')}")

# Per-row diagnostic CSV (test split is most useful, but we save all three).
for split_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
    pd.DataFrame(ds.context_truncation_per_row()).to_csv(
        os.path.join(CFG["output_dir"], f"context_truncation_per_row_{split_name}.csv"),
        index=False,
    )

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_micro_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
    }

def search_best_global_tau_microf1(logits: np.ndarray, labels: np.ndarray) -> Tuple[float, Dict[str, float]]:
    probs = sigmoid_stable(logits)
    best_tau = 0.50
    best = None
    tau_grid = np.linspace(CFG["tau_grid_start"], CFG["tau_grid_end"], CFG["tau_grid_steps"])
    for tau in tau_grid:
        pred = (probs >= float(tau)).astype(int)
        m = compute_micro_metrics(labels, pred)
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

def eval_subset_micro(y_true: np.ndarray, probs: np.ndarray, tau: float) -> Dict[str, float]:
    pred = (probs >= float(tau)).astype(int)
    return compute_micro_metrics(y_true, pred)

def classwise_table(y_true: np.ndarray, probs: np.ndarray, tau: float, subset_name: str) -> pd.DataFrame:
    pred = (probs >= float(tau)).astype(int)
    rows = []
    for j, lab in enumerate(LABELS):
        yt = y_true[:, j].astype(int)
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
# Uses LlamaForSequenceClassification (built-in BCE for multi_label_classification)
# wrapped with LoRA. The seq-cls head reads the last non-pad token.
# ---------------------------------------------------------------------------
def build_model(classifier_dropout: float):
    config = AutoConfig.from_pretrained(CFG["model_name"])
    config.num_labels = len(LABELS)
    config.problem_type = "multi_label_classification"
    config.id2label = id2label
    config.label2id = label2id
    config.pad_token_id = tok.pad_token_id

    if hasattr(config, "classifier_dropout"):
        config.classifier_dropout = float(classifier_dropout)
    if hasattr(config, "hidden_dropout"):
        config.hidden_dropout = float(classifier_dropout)

    base = AutoModelForSequenceClassification.from_pretrained(
        CFG["model_name"],
        config=config,
        torch_dtype=torch.bfloat16 if CFG["use_bf16"] else torch.float32,
    )
    base.config.pad_token_id = tok.pad_token_id

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=CFG["lora_r"],
        lora_alpha=CFG["lora_alpha"],
        lora_dropout=CFG["lora_dropout"],
        target_modules=CFG["lora_target_modules"],
        bias="none",
    )
    model = get_peft_model(base, lora_cfg)

    # Unfreeze the classifier head explicitly (PEFT freezes everything except LoRA).
    for name, param in model.named_parameters():
        if "score" in name or "classifier" in name:
            param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Trainable params: {n_trainable:,} / {n_total:,} ({100.0 * n_trainable / n_total:.4f}%)")
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
        data_seed=CFG["seed"],
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
        compute_metrics=compute_metrics_epoch,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=CFG["early_stopping_patience"]),
            TableLoggerCallback(),
        ],
    ))
    trainer = Trainer(**trainer_kwargs)

    trainer.train()

    best_checkpoint_dir = trainer.state.best_model_checkpoint or ""
    if not best_checkpoint_dir:
        raise RuntimeError(f"No best checkpoint found for {trial_name}")

    # Reload best checkpoint cleanly via PEFT.
    base_for_eval = AutoModelForSequenceClassification.from_pretrained(
        CFG["model_name"],
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
        torch_dtype=torch.bfloat16 if CFG["use_bf16"] else torch.float32,
    )
    base_for_eval.config.pad_token_id = tok.pad_token_id
    best_model = PeftModel.from_pretrained(base_for_eval, best_checkpoint_dir)
    best_model.eval()

    eval_args = TrainingArguments(
        output_dir=os.path.join(trial_dir, "eval_tmp"),
        per_device_eval_batch_size=batch_size,
        report_to=[],
        bf16=CFG["use_bf16"] and torch.cuda.is_available(),
        fp16=False,
        remove_unused_columns=False,
    )
    eval_trainer_kwargs = make_trainer_kwargs(dict(
        model=best_model,
        args=eval_args,
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

    del trainer, eval_trainer, model, best_model, base_for_eval, val_out, val_logits, val_labels
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
best_batch = int(best_trial["batch_size"])

base_final = AutoModelForSequenceClassification.from_pretrained(
    CFG["model_name"],
    num_labels=len(LABELS),
    problem_type="multi_label_classification",
    id2label=id2label,
    label2id=label2id,
    torch_dtype=torch.bfloat16 if CFG["use_bf16"] else torch.float32,
)
base_final.config.pad_token_id = tok.pad_token_id
best_model = PeftModel.from_pretrained(base_final, best_ckpt)
best_model.eval()

final_args = TrainingArguments(
    output_dir=os.path.join(CFG["output_dir"], "final_eval_tmp"),
    per_device_eval_batch_size=best_batch,
    report_to=[],
    bf16=CFG["use_bf16"] and torch.cuda.is_available(),
    fp16=False,
    remove_unused_columns=False,
)
final_trainer_kwargs = make_trainer_kwargs(dict(
    model=best_model,
    args=final_args,
))
final_trainer = Trainer(**final_trainer_kwargs)

test_out = final_trainer.predict(test_ds)
test_logits = test_out.predictions
test_labels = test_out.label_ids
test_probs = sigmoid_stable(test_logits)

is_req = df_test[CFG["ISREQ_COL"]].astype(int).to_numpy()
mask_overall = np.ones(len(df_test), dtype=bool)
mask_request = (is_req == 1)
mask_reply = (is_req == 0)

summary_rows = []
for subset_name, mask in [
    ("OVERALL", mask_overall),
    ("REQUEST", mask_request),
    ("REPLY", mask_reply),
]:
    m = eval_subset_micro(test_labels[mask], test_probs[mask], best_tau)
    summary_rows.append({
        "subset": subset_name,
        "n_rows": int(mask.sum()),
        "tau": float(best_tau),
        "micro_f1": m["micro_f1"],
        "micro_precision": m["micro_precision"],
        "micro_recall": m["micro_recall"],
    })
summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(CFG["output_dir"], "test_micro_metrics_overall_request_reply.csv")
summary_df.to_csv(summary_csv, index=False)

classwise_df = pd.concat([
    classwise_table(test_labels[mask_overall], test_probs[mask_overall], best_tau, "OVERALL"),
    classwise_table(test_labels[mask_request], test_probs[mask_request], best_tau, "REQUEST"),
    classwise_table(test_labels[mask_reply],   test_probs[mask_reply],   best_tau, "REPLY"),
], ignore_index=True)
classwise_csv = os.path.join(CFG["output_dir"], "test_classwise_f1_support_for_appendix.csv")
classwise_df.to_csv(classwise_csv, index=False)

latex_lines = [
    "\\begin{tabular}{lcccc}",
    "\\toprule",
    "\\textbf{Subset} & \\textbf{N} & \\textbf{mF1} & \\textbf{micro-P} & \\textbf{micro-R} \\\\",
    "\\midrule",
]
for _, r in summary_df.iterrows():
    latex_lines.append(
        f"{str(r['subset']).title()} & {int(r['n_rows'])} & {r['micro_f1']:.3f} & {r['micro_precision']:.3f} & {r['micro_recall']:.3f} \\\\"
    )
latex_lines += ["\\bottomrule", "\\end{tabular}"]
latex_path = os.path.join(CFG["output_dir"], "test_micro_metrics_overall_request_reply.tex")
with open(latex_path, "w", encoding="utf-8") as f:
    f.write("\n".join(latex_lines) + "\n")

if CFG["save_test_predictions"]:
    pred_bin = (test_probs >= best_tau).astype(int)
    save_df = df_test.copy()
    for col in ("gold_list", "y_vec"):
        if col in save_df.columns:
            save_df = save_df.drop(columns=[col])
    for k, lbl in enumerate(LABELS):
        save_df[f"pred_prob_{lbl}"] = test_probs[:, k]
    save_df[f"pred_labels_tau_{best_tau:.2f}"] = [
        ",".join([LABELS[k] for k, v in enumerate(test_probs[i]) if v >= best_tau])
        for i in range(len(save_df))
    ]
    save_df["best_tau_from_validation"] = float(best_tau)
    save_df["pred_cardinality"] = pred_bin.sum(axis=1)
    save_df["gold_cardinality"] = test_labels.sum(axis=1)
    pred_csv = os.path.join(CFG["output_dir"], "test_predictions_past_same_email_req_or_rep_llama_lora_xent_best_validation_tau.csv")
    save_df.to_csv(pred_csv, index=False)

manifest = {
    "model": "LLaMA-3.2 (configurable via CFG['model_name']) + LoRA, LlamaForSequenceClassification with BCE-with-logits (xent)",
    "base_model": CFG["model_name"],
    "run_seed": CFG["seed"],
    "base_output_dir": BASE_OUTPUT_DIR,
    "lora": {
        "r": CFG["lora_r"],
        "alpha": CFG["lora_alpha"],
        "dropout": CFG["lora_dropout"],
        "target_modules": CFG["lora_target_modules"],
    },
    "precision": "bf16" if CFG["use_bf16"] else "fp32",
    "max_length": CFG["max_length"],
    "input_format": "<target>\\n\\nContext: <past_sent_1 past_sent_2 ... past_sent_N>",
    "context_truncation_policy": "KEEP OLDEST: target kept intact; past sentences added oldest-first; once budget exceeded, all remaining (newer) past sentences are dropped. (Equivalent to BERT truncation='only_second' in spirit.)",
    "selection_metric_for_hyperparameters": "validation micro-F1 at validation-learned tau",
    "checkpoint_metric_within_each_run": "validation micro-F1 at validation-learned tau",
    "threshold_learning": "best global validation tau maximizing validation micro-F1",
    "tokenizer": {
        "pad_token": tok.pad_token,
        "pad_token_id": tok.pad_token_id,
        "padding_side": tok.padding_side,
    },
    "best_trial": best_trial,
    "context_truncation_report": ctx_trunc_report,
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
print(summary_df.to_string(index=False))
print(f"\n[DONE] Wrote grid results              -> {grid_csv}")
print(f"[DONE] Wrote best config               -> {best_json}")
print(f"[DONE] Wrote test summary CSV          -> {summary_csv}")
print(f"[DONE] Wrote classwise CSV             -> {classwise_csv}")
print(f"[DONE] Wrote LaTeX table               -> {latex_path}")
print(f"[DONE] Wrote context truncation report -> {os.path.join(CFG['output_dir'], 'context_truncation_report.json')}")
