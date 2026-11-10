#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export CUDA_VISIBLE_DEVICES=0

nohup bash -lc '
for s in 13 21 42 87 100; do
  echo "===== RUN_SEED=$s ====="
  RUN_SEED=$s python FAC_context_mail_BERT_gridsearch.py
done
' > past_same_email_req_or_rep_bert_multiseed.log 2>&1 &

tail -f past_same_email_req_or_rep_bert_multiseed.log

"""
# FAC_context_mail_BERT_gridsearch.py
# ============================================================
# BERT multi-label classifier (9 labels) — TargetFirst: PastSameEmail
# ============================================================

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
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
)
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

print("TargetFirst: PastSameEmail Req-or-Rep — BERT + ASL — validation micro-F1 grid search [MULTI-SEED]")

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

    "filter_is_request": None,

    "LABELS": ["HNeg+","HNeg-","HPos+","HPos-","Neutral","SNeg+","SNeg-","SPos+","SPos-"],

    "model_name": "bert-base-uncased",
    "max_length": 256,

    "output_dir": "outputs/bert_targetfirst_past_same_email_asl",

    # Grid
    "lr_grid": [1e-5, 2e-5, 3e-5],
    "epochs_grid": [3, 5, 8],
    "batch_grid": [8, 16],
    "dropout_grid": [0.1, 0.2],

    "weight_decay": 0.01,
    "early_stopping_patience": 2,

    "use_pos_weight": False,
    "asl_gamma_pos": 0.0,
    "asl_gamma_neg": 4.0,
    "asl_clip": 0.05,

    "tau_grid_start": 0.05,
    "tau_grid_end": 0.95,
    "tau_grid_steps": 91,

    "seed": 42,
    "debug_samples": 3,
    "save_test_predictions": True,
    "save_trial_models": False,
}

# ---------------------------------------------------------------------------
# Multi-seed control
# ---------------------------------------------------------------------------
# Keep the grouped train/val/test split fixed, but vary the training seed.
# Usage:
#   RUN_SEED=13 python FAC_context_mail_BERT_gridsearch.py
#   RUN_SEED=21 python FAC_context_mail_BERT_gridsearch.py
# etc.
BASE_OUTPUT_DIR = CFG["output_dir"]
RUN_SEED = int(os.environ.get("RUN_SEED", CFG["seed"]))
CFG["seed"] = RUN_SEED
CFG["run_seed"] = RUN_SEED
CFG["base_output_dir"] = BASE_OUTPUT_DIR
CFG["output_dir"] = f"{BASE_OUTPUT_DIR}_runseed{RUN_SEED}"

print(f"[INFO] Fixed split files: {CFG['train_csv']}, {CFG['val_csv']}, {CFG['test_csv']}")
print(f"[INFO] Training RUN_SEED={RUN_SEED}")
print(f"[INFO] Output directory: {CFG['output_dir']}")

os.makedirs(CFG["output_dir"], exist_ok=True)
torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
random.seed(CFG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG["seed"])

LABELS = CFG["LABELS"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for i, l in enumerate(LABELS)}

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
    need = [cfg["TEXT_COL"], cfg["LABEL_COL"], cfg["EMAIL_COL"], cfg["SENTIDX_COL"], cfg["ISREQ_COL"], cfg["SEED_COL"]]
    for c in need:
        assert c in df.columns, f"Missing column '{c}' in {csv_path}"
    if cfg["filter_is_request"] in (0, 1):
        df = df[df[cfg["ISREQ_COL"]] == cfg["filter_is_request"]].copy()

    for c in (cfg["EMAIL_COL"], cfg["SENTIDX_COL"], cfg["ISREQ_COL"], cfg["SEED_COL"]):
        df[c] = df[c].astype(int)

    df["gold_list"] = [parse_labels(v, i) for i, v in enumerate(df[cfg["LABEL_COL"]].tolist())]
    df["y_vec"] = df["gold_list"].map(to_multi_hot)
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
    ov_tr_va = sorted(list(s_tr & s_va))
    ov_tr_te = sorted(list(s_tr & s_te))
    ov_va_te = sorted(list(s_va & s_te))
    print("\n[Overlap checks]")
    print(f"Train ∩ Val  ({len(ov_tr_va)}): {ov_tr_va}")
    print(f"Train ∩ Test ({len(ov_tr_te)}): {ov_tr_te}")
    print(f"Val   ∩ Test ({len(ov_va_te)}): {ov_va_te}")
    if ov_tr_va or ov_tr_te or ov_va_te:
        print("[LEAKAGE WARNING] Some seeds appear in multiple splits!")
    else:
        print("[OK] No seeds in common across splits.")

def print_is_request_counts(df_train, df_val, df_test, col):
    print("\n[TRAIN is_request counts]\n", df_train[col].value_counts(dropna=False))
    print("[VAL   is_request counts]\n", df_val[col].value_counts(dropna=False))
    print("[TEST  is_request counts]\n", df_test[col].value_counts(dropna=False))

check_split_integrity(df_train, df_val, df_test, CFG["SEED_COL"])
print_is_request_counts(df_train, df_val, df_test, CFG["ISREQ_COL"])

def add_target_and_context(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    email_col, sent_col, text_col = cfg["EMAIL_COL"], cfg["SENTIDX_COL"], cfg["TEXT_COL"]
    df2 = df.sort_values([email_col, sent_col]).copy()
    df2["target"] = df2[text_col].astype(str)
    df2["context"] = ""
    for eid, grp in df2.groupby(email_col, sort=False):
        texts = grp[text_col].astype(str).tolist()
        ctx_vals, pasts = [], []
        for s in texts:
            ctx_vals.append(" ".join(pasts) if pasts else "")
            pasts.append(s)
        df2.loc[grp.index, "context"] = ctx_vals
    df2["context"] = df2["context"].astype(str).fillna("")
    df2["target"] = df2["target"].astype(str).fillna("")
    return df2.reset_index(drop=True)

df_train_ctx = add_target_and_context(df_train, CFG)
df_val_ctx   = add_target_and_context(df_val,   CFG)
df_test_ctx  = add_target_and_context(df_test,  CFG)

tok = AutoTokenizer.from_pretrained(CFG["model_name"])

# ---------------------------------------------------------------------------
# Context-truncation diagnostic
#
# The tokenizer uses truncation="only_second", which drops tokens from the END
# of segment B (the context). Since context is built in chronological order
# (oldest first), this means the MOST RECENT past sentences are the ones lost
# when the budget is exceeded. This diagnostic reports how often that happens
# and how many tokens are dropped.
# ---------------------------------------------------------------------------
def context_truncation_report(df_ctx: pd.DataFrame, tokenizer, max_length: int, split_name: str) -> Dict[str, Any]:
    targets = df_ctx["target"].astype(str).tolist()
    contexts = df_ctx["context"].astype(str).tolist()

    # length-only encodings (no padding, no truncation) to measure the *intended*
    # full-pair length before any cuts.
    n_rows = len(df_ctx)
    n_truncated = 0
    n_truncated_with_nonempty_ctx = 0
    total_target_tokens = 0
    total_context_tokens_intended = 0
    total_context_tokens_kept_after_trunc = 0
    over_by_examples = []  # for histogramming

    # Tokenize targets alone (with [CLS] target [SEP] form by passing as text only)
    # and contexts alone, to count tokens. We don't need full pair encoding; pair
    # length = 1([CLS]) + len_target_tokens + 1([SEP]) + len_context_tokens + 1([SEP]).
    target_token_lens = [len(tokenizer.encode(t, add_special_tokens=False)) for t in targets]
    context_token_lens = [len(tokenizer.encode(c, add_special_tokens=False)) if c else 0 for c in contexts]

    SPECIAL_OVERHEAD = 3  # [CLS], [SEP], [SEP]

    for tgt_len, ctx_len in zip(target_token_lens, context_token_lens):
        total_target_tokens += tgt_len
        total_context_tokens_intended += ctx_len
        full_len = SPECIAL_OVERHEAD + tgt_len + ctx_len
        if full_len > max_length:
            n_truncated += 1
            if ctx_len > 0:
                n_truncated_with_nonempty_ctx += 1
            ctx_budget = max(0, max_length - SPECIAL_OVERHEAD - tgt_len)
            kept = min(ctx_len, ctx_budget)
            total_context_tokens_kept_after_trunc += kept
            over_by_examples.append(full_len - max_length)
        else:
            total_context_tokens_kept_after_trunc += ctx_len

    pct_truncated = 100.0 * n_truncated / max(1, n_rows)
    ctx_lost = total_context_tokens_intended - total_context_tokens_kept_after_trunc
    pct_ctx_lost = 100.0 * ctx_lost / max(1, total_context_tokens_intended)

    over_arr = np.array(over_by_examples, dtype=int) if over_by_examples else np.array([0])
    report = {
        "split": split_name,
        "max_length": int(max_length),
        "n_rows": int(n_rows),
        "n_rows_truncated": int(n_truncated),
        "n_rows_truncated_with_nonempty_context": int(n_truncated_with_nonempty_ctx),
        "pct_rows_truncated": float(pct_truncated),
        "total_target_tokens": int(total_target_tokens),
        "total_context_tokens_intended": int(total_context_tokens_intended),
        "total_context_tokens_kept_after_truncation": int(total_context_tokens_kept_after_trunc),
        "context_tokens_lost": int(ctx_lost),
        "pct_context_tokens_lost": float(pct_ctx_lost),
        "overflow_tokens_p50": int(np.percentile(over_arr, 50)) if len(over_by_examples) else 0,
        "overflow_tokens_p90": int(np.percentile(over_arr, 90)) if len(over_by_examples) else 0,
        "overflow_tokens_max": int(over_arr.max()) if len(over_by_examples) else 0,
        "truncation_policy_note": (
            "truncation='only_second' drops tokens from the END of context. "
            "Context is built chronologically (oldest first), so the lost tokens are "
            "the MOST RECENT past sentences. This is informational only; behavior is unchanged."
        ),
    }
    print(f"[CTX-TRUNC][{split_name}] {n_truncated}/{n_rows} rows truncated ({pct_truncated:.2f}%); "
          f"context tokens lost: {ctx_lost}/{total_context_tokens_intended} ({pct_ctx_lost:.2f}%); "
          f"overflow p50/p90/max = {report['overflow_tokens_p50']}/{report['overflow_tokens_p90']}/{report['overflow_tokens_max']}")
    return report

print("[INFO] Computing context-truncation diagnostics (this is read-only; behavior unchanged)...")
ctx_trunc_report = {
    "train": context_truncation_report(df_train_ctx, tok, CFG["max_length"], "train"),
    "val":   context_truncation_report(df_val_ctx,   tok, CFG["max_length"], "val"),
    "test":  context_truncation_report(df_test_ctx,  tok, CFG["max_length"], "test"),
}
with open(os.path.join(CFG["output_dir"], "context_truncation_report.json"), "w", encoding="utf-8") as f:
    json.dump(ctx_trunc_report, f, indent=2)
print(f"[INFO] Context-truncation report written to {os.path.join(CFG['output_dir'], 'context_truncation_report.json')}")

def tokenize_pairs(target_list: List[str], context_list: List[str], max_length: int):
    return tok(
        target_list,
        context_list,
        padding="max_length",
        truncation="only_second",
        max_length=max_length,
        return_token_type_ids=True,
        return_tensors=None,
    )

class PastReqRepDataset(Dataset):
    def __init__(self, df_ctx: pd.DataFrame, cfg: Dict[str, Any]):
        self.df = df_ctx.reset_index(drop=True)
        self.target = self.df["target"].astype(str).tolist()
        self.context = self.df["context"].astype(str).tolist()
        self.labels = np.stack(self.df["y_vec"].values)
        self.enc = tokenize_pairs(self.target, self.context, cfg["max_length"])
    def __len__(self):
        return len(self.target)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item

train_ds = PastReqRepDataset(df_train_ctx, CFG)
val_ds   = PastReqRepDataset(df_val_ctx,   CFG)
test_ds  = PastReqRepDataset(df_test_ctx,  CFG)

pos_weight_tensor = None
if CFG["use_pos_weight"]:
    y = np.stack(df_train_ctx["y_vec"].values)
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    pos_weight = (neg / (pos + 1e-6)).astype(np.float32)
    pos_weight_tensor = torch.tensor(pos_weight)

class AsymmetricLoss(nn.Module):
    """Paper ASL (Ridnik et al. 2021)."""
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gp = gamma_pos
        self.gn = gamma_neg
        self.clip = clip
        self.eps = eps
    def forward(self, logits, targets, pos_weight=None):
        p = torch.sigmoid(logits)

        log_pos = torch.log(p.clamp(min=self.eps))
        if self.gp > 0:
            log_pos = log_pos * (1.0 - p) ** self.gp

        if self.clip is not None and self.clip > 0:
            x_neg = (1.0 - p - self.clip).clamp(min=0.0)
        else:
            x_neg = 1.0 - p

        log_neg = torch.log(x_neg.clamp(min=self.eps))
        if self.gn > 0:
            log_neg = log_neg * (1.0 - x_neg) ** self.gn

        loss = -(targets * log_pos + (1.0 - targets) * log_neg)
        if pos_weight is not None:
            loss = loss * (1.0 + targets * (pos_weight.to(loss.device) - 1.0))
        return loss.mean()

class ASLTrainer(Trainer):
    def __init__(self, *args, pos_weight=None, asl_gamma_pos=0.0, asl_gamma_neg=4.0, asl_clip=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight
        self.asl = AsymmetricLoss(gamma_pos=asl_gamma_pos, gamma_neg=asl_gamma_neg, clip=asl_clip)
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.asl(logits, labels, pos_weight=self.pos_weight)
        return (loss, outputs) if return_outputs else loss

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

class TableLoggerCallback(TrainerCallback):
    """Dedupe on epoch only."""
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
        epoch = int(round(state.epoch or 0)) if state.epoch is not None else (len(self.rows) + 1)
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

def debug_print_samples(df_ctx: pd.DataFrame, ds: PastReqRepDataset, cfg: Dict[str, Any], n=3):
    """Uses an isolated RNG, does not perturb global random state."""
    if n <= 0:
        return
    rng = random.Random(cfg.get("seed", 42))

    print("\n" + "="*72)
    print(f"[DEBUG] Showing {n} examples. LABEL ORDER: {LABELS}")
    print("="*72)

    rq0 = df_ctx[(df_ctx[cfg["ISREQ_COL"]] == 1) & (df_ctx["context"].str.len() == 0)].index.tolist()
    rp0 = df_ctx[(df_ctx[cfg["ISREQ_COL"]] == 0) & (df_ctx["context"].str.len() == 0)].index.tolist()
    rqa = df_ctx[(df_ctx[cfg["ISREQ_COL"]] == 1) & (df_ctx["context"].str.len() > 0)].index.tolist()
    rpa = df_ctx[(df_ctx[cfg["ISREQ_COL"]] == 0) & (df_ctx["context"].str.len() > 0)].index.tolist()

    chosen = []
    if rq0:
        chosen.append(rng.choice(rq0))
    if rp0 and len(chosen) < 2:
        chosen.append(rng.choice(rp0))
    pool = rqa + rpa
    rng.shuffle(pool)
    for idx in pool:
        if len(chosen) >= n:
            break
        if idx not in chosen:
            chosen.append(idx)
    if len(chosen) < n:
        all_idx = list(range(len(df_ctx)))
        rng.shuffle(all_idx)
        for idx in all_idx:
            if len(chosen) >= n:
                break
            if idx not in chosen:
                chosen.append(idx)

    def _to_pylist(x):
        if isinstance(x, list):
            return x
        if hasattr(x, "tolist"):
            return x.tolist()
        return list(x)

    for i in chosen[:n]:
        row = df_ctx.iloc[i]
        enc_i = {k: v[i] for k, v in ds.enc.items()}
        input_ids = _to_pylist(enc_i["input_ids"])
        tokens = tok.convert_ids_to_tokens(input_ids)
        typeids = _to_pylist(enc_i.get("token_type_ids", [0] * len(input_ids)))
        gold_vec = row["y_vec"].astype(int).tolist()
        print("-"*72)
        print(f"[Row {i}] email_id={row[cfg['EMAIL_COL']]} | sentence_idx={row[cfg['SENTIDX_COL']]} | is_request={row[cfg['ISREQ_COL']]}")
        print("TARGET :", textwrap.shorten(row["target"],  width=160, placeholder="…"))
        print("CONTEXT:", textwrap.shorten(row["context"], width=160, placeholder="…"))
        print("Gold vec :", gold_vec, "=>", [lbl for k, lbl in enumerate(LABELS) if gold_vec[k] == 1])
        show = min(120, len(tokens))
        print("TOKENS    :", " ".join(tokens[:show]))
        print("TYPE_IDS  :", " ".join(map(str, typeids[:show])))
    print("="*72 + "\n")

debug_print_samples(df_train_ctx, train_ds, CFG, CFG["debug_samples"])

def build_model(dropout: float):
    config = AutoConfig.from_pretrained(CFG["model_name"])
    if hasattr(config, "hidden_dropout_prob"):
        config.hidden_dropout_prob = float(dropout)
    if hasattr(config, "attention_probs_dropout_prob"):
        config.attention_probs_dropout_prob = float(dropout)
    if hasattr(config, "classifier_dropout"):
        config.classifier_dropout = float(dropout)
    config.num_labels = len(LABELS)
    config.problem_type = "multi_label_classification"
    config.id2label = id2label
    config.label2id = label2id

    model = AutoModelForSequenceClassification.from_pretrained(
        CFG["model_name"],
        config=config,
    )
    return model

def make_trainer_kwargs(extra: Dict[str, Any]) -> Dict[str, Any]:
    """Inject the right tokenizer kwarg for the installed transformers."""
    if TOK_KW is not None:
        extra[TOK_KW] = tok
    return extra

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
    model = build_model(dropout=dropout)

    args = TrainingArguments(
        output_dir=trial_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=CFG["weight_decay"],
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_micro/f1_tau_best",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        seed=CFG["seed"],
        report_to=[],
        logging_steps=50,
        save_total_limit=2,
    )

    trainer_kwargs = make_trainer_kwargs(dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics_epoch,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=CFG["early_stopping_patience"]),
                   TableLoggerCallback()],
    ))
    trainer = ASLTrainer(
        **trainer_kwargs,
        pos_weight=(pos_weight_tensor if CFG["use_pos_weight"] else None),
        asl_gamma_pos=CFG["asl_gamma_pos"],
        asl_gamma_neg=CFG["asl_gamma_neg"],
        asl_clip=CFG["asl_clip"],
    )

    trainer.train()

    best_checkpoint_dir = trainer.state.best_model_checkpoint or ""
    if not best_checkpoint_dir:
        raise RuntimeError(f"No best checkpoint found for {trial_name}")

    best_model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint_dir)
    eval_args = TrainingArguments(
        output_dir=os.path.join(trial_dir, "eval_tmp"),
        per_device_eval_batch_size=batch_size,
        report_to=[],
        fp16=torch.cuda.is_available(),
    )
    eval_trainer_kwargs = make_trainer_kwargs(dict(
        model=best_model,
        args=eval_args,
    ))
    eval_trainer = ASLTrainer(
        **eval_trainer_kwargs,
        pos_weight=(pos_weight_tensor if CFG["use_pos_weight"] else None),
        asl_gamma_pos=CFG["asl_gamma_pos"],
        asl_gamma_neg=CFG["asl_gamma_neg"],
        asl_clip=CFG["asl_clip"],
    )

    val_out = eval_trainer.predict(val_ds)
    val_logits = val_out.predictions
    val_labels = val_out.label_ids
    val_tau_best, val_best = search_best_global_tau_microf1(val_logits, val_labels)

    trial_row = {
        "run_seed": int(CFG["run_seed"]),
        "trial_id": trial_id,
        "trial_name": trial_name,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
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

    if best_trial is None:
        best_trial = trial_row
        best_trial_dir = trial_dir
    else:
        if trial_row["val_micro_f1_best_tau"] > best_trial["val_micro_f1_best_tau"]:
            best_trial = trial_row; best_trial_dir = trial_dir
        elif math.isclose(trial_row["val_micro_f1_best_tau"], best_trial["val_micro_f1_best_tau"], rel_tol=0.0, abs_tol=1e-12):
            if trial_row["val_micro_precision_best_tau"] > best_trial["val_micro_precision_best_tau"]:
                best_trial = trial_row; best_trial_dir = trial_dir
            elif math.isclose(trial_row["val_micro_precision_best_tau"], best_trial["val_micro_precision_best_tau"], rel_tol=0.0, abs_tol=1e-12):
                if trial_row["val_micro_recall_best_tau"] > best_trial["val_micro_recall_best_tau"]:
                    best_trial = trial_row; best_trial_dir = trial_dir

    # incremental save in case of crash
    pd.DataFrame(grid_rows).to_csv(
        os.path.join(CFG["output_dir"], "grid_search_results_partial.csv"),
        index=False,
    )

    del trainer, eval_trainer, model, best_model, val_out, val_logits, val_labels
    cleanup_memory()

grid_df = pd.DataFrame(grid_rows).sort_values(
    ["val_micro_f1_best_tau", "val_micro_precision_best_tau", "val_micro_recall_best_tau"],
    ascending=[False, False, False]
).reset_index(drop=True)
grid_df.to_csv(os.path.join(CFG["output_dir"], "grid_search_results.csv"), index=False)

with open(os.path.join(CFG["output_dir"], "best_config.json"), "w", encoding="utf-8") as f:
    json.dump(best_trial, f, indent=2)

print(f"\n[BEST] {best_trial['trial_name']}")
print(json.dumps(best_trial, indent=2))

best_ckpt = best_trial["best_checkpoint_dir"]
best_tau = float(best_trial["val_best_tau"])
best_bs = int(best_trial["batch_size"])

best_model = AutoModelForSequenceClassification.from_pretrained(best_ckpt)
final_args = TrainingArguments(
    output_dir=os.path.join(CFG["output_dir"], "final_eval_tmp"),
    per_device_eval_batch_size=best_bs,
    report_to=[],
    fp16=torch.cuda.is_available(),
)
final_trainer_kwargs = make_trainer_kwargs(dict(
    model=best_model,
    args=final_args,
))
final_trainer = ASLTrainer(
    **final_trainer_kwargs,
    pos_weight=(pos_weight_tensor if CFG["use_pos_weight"] else None),
    asl_gamma_pos=CFG["asl_gamma_pos"],
    asl_gamma_neg=CFG["asl_gamma_neg"],
    asl_clip=CFG["asl_clip"],
)

test_out = final_trainer.predict(test_ds)
test_logits = test_out.predictions
test_labels = test_out.label_ids
test_probs = sigmoid_stable(test_logits)
is_req = df_test_ctx[CFG["ISREQ_COL"]].astype(int).to_numpy()

rows = []
class_rows = []
for subset_name, mask in [
    ("OVERALL", np.ones(len(df_test_ctx), dtype=bool)),
    ("REQUEST", is_req == 1),
    ("REPLY", is_req == 0),
]:
    y_true = test_labels[mask]
    probs = test_probs[mask]
    pred = (probs >= best_tau).astype(int)
    m = compute_micro_metrics(y_true, pred)
    rows.append({
        "run_seed": int(CFG["run_seed"]),
        "subset": subset_name,
        "n_rows": int(mask.sum()),
        "tau": float(best_tau),
        "micro_f1": float(m["micro_f1"]),
        "micro_precision": float(m["micro_precision"]),
        "micro_recall": float(m["micro_recall"]),
    })
    class_rows.append(classwise_table(y_true, probs, best_tau, subset_name))

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
    save_df = df_test_ctx.copy()
    for col in ("gold_list", "y_vec"):
        if col in save_df.columns:
            save_df = save_df.drop(columns=[col])
    for j, lab in enumerate(LABELS):
        save_df[f"pred_prob_{lab}"] = test_probs[:, j]
    save_df["best_val_tau"] = float(best_tau)
    save_df[f"pred_labels_tau_{best_tau:.2f}"] = [
        ",".join([LABELS[j] for j, v in enumerate(test_probs[i]) if v >= best_tau])
        for i in range(len(save_df))
    ]
    out_csv = os.path.join(CFG["output_dir"], "test_predictions_targetfirst_pastsameemail_best_validation_tau.csv")
    save_df.to_csv(out_csv, index=False)

manifest = {
    "run_seed": int(CFG["run_seed"]),
    "base_output_dir": CFG["base_output_dir"],
    "output_dir": CFG["output_dir"],
    "selection_metric_for_hyperparameters": "validation micro-F1",
    "checkpoint_metric_within_each_run": "validation micro-F1 at best validation-learned tau",
    "threshold_learning": "best global validation tau maximizing validation micro-F1",
    "asl_implementation": "paper (Ridnik et al. 2021): negative branch uses x_neg = clamp(1-p-clip, min=0); weight (1 - x_neg)^gamma_neg",
    "context_truncation_policy": (
        "truncation='only_second' — drops tokens from the END of the context segment. "
        "Context is built chronologically (oldest first), so when the budget is exceeded, "
        "the MOST RECENT past sentences are lost. Behavior is unchanged in this run; "
        "see context_truncation_report.json for impact metrics."
    ),
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

print("[DONE] Wrote:")
print(" - grid_search_results.csv")
print(" - best_config.json")
print(" - test_micro_metrics_overall_request_reply.csv")
print(" - test_micro_metrics_overall_request_reply.tex")
print(" - test_classwise_f1_support_for_appendix.csv")
print(" - test_predictions_targetfirst_pastsameemail_best_validation_tau.csv")
print(" - context_truncation_report.json")
print(" - run_manifest.json")
