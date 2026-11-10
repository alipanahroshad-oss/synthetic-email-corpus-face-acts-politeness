#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import gc
import ast
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
    AutoModelForSequenceClassification,
    AutoConfig,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

print("SENTENCE ONLY + ASL — BERT — validation micro-F1 grid search (patched)")

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

    "LABELS": ["HNeg+","HNeg-","HPos+","HPos-","Neutral","SNeg+","SNeg-","SPos+","SPos-"],

    "model_name": "bert-base-uncased",
    "max_length": 128,

    # Grid
    "lr_grid": [1e-5, 2e-5, 3e-5],
    "epochs_grid": [3, 5, 8],
    "batch_grid": [8, 16],
    "dropout_grid": [0.1, 0.2],

    # Fixed
    "weight_decay": 0.01,
    "early_stopping_patience": 2,
    "use_pos_weight": False,
    "asl_gamma_pos": 0.0,
    "asl_gamma_neg": 4.0,
    "asl_clip": 0.05,

    # Threshold search on validation
    "tau_grid_start": 0.05,
    "tau_grid_end": 0.95,
    "tau_grid_steps": 91,  # step 0.01

    "seed": 42,
    "output_dir": "outputs/bert_multi_label_faceacts_sentence_only_asl",
    "save_test_predictions": True,
    "save_trial_models": False,
}

# ============================================================
# MULTI-SEED PATCH
# ------------------------------------------------------------
# Keep the data split fixed (train_seed42/val_seed42/test_seed42),
# but change the *training* random seed via the RUN_SEED environment variable.
# Example:
#   RUN_SEED=13 python FAC_no_context_BERT_gridsearch.py
# ============================================================
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
    # More reproducible CUDA behavior.
    # This may slightly reduce speed, but is useful for comparable repeated runs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG["seed"])
print(f"[INFO] Training random seed: {CFG['seed']}")
print("[INFO] Data split is fixed: train_seed42 / val_seed42 / test_seed42")
print(f"[INFO] Output directory: {CFG['output_dir']}")

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
    for c in [cfg["EMAIL_COL"], cfg["SENTIDX_COL"], cfg["ISREQ_COL"], cfg["SEED_COL"], cfg["PAIR_COL"]]:
        df[c] = df[c].astype(int)
    df["gold_list"] = [parse_labels(v, i) for i, v in enumerate(df[cfg["LABEL_COL"]].tolist())]
    df["y_vec"] = df["gold_list"].map(to_multi_hot)
    # sanity: count rows whose parsed labels include unknown names
    unknown_rows = 0
    for names in df["gold_list"]:
        if any(n not in label2id for n in names):
            unknown_rows += 1
    if unknown_rows > 0:
        print(f"[WARN] {unknown_rows} rows in {os.path.basename(csv_path)} contained at least one unknown label name (silently dropped from y_vec)")
    return df.reset_index(drop=True)

df_train = load_split(CFG["train_csv"], CFG)
df_val   = load_split(CFG["val_csv"],   CFG)
df_test  = load_split(CFG["test_csv"],  CFG)
print(f"[INFO] Loaded rows: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

tok = AutoTokenizer.from_pretrained(CFG["model_name"])

class SentDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: Dict[str, Any]):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.texts = self.df[cfg["TEXT_COL"]].astype(str).tolist()
        self.labels = np.stack(self.df["y_vec"].values)
        self.enc = tok(
            self.texts,
            padding="max_length",
            truncation=True,
            max_length=cfg["max_length"],
            return_token_type_ids=True,
            return_tensors=None,
        )
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item

train_ds = SentDataset(df_train, CFG)
val_ds   = SentDataset(df_val,   CFG)
test_ds  = SentDataset(df_test,  CFG)

pos_weight_tensor = None
if CFG["use_pos_weight"]:
    y = np.stack(df_train["y_vec"].values)
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    pos_weight = (neg / (pos + 1e-6)).astype(np.float32)
    pos_weight_tensor = torch.tensor(pos_weight)

class AsymmetricLoss(nn.Module):
    """
    Paper ASL (Ridnik et al., 2021).
    For positives:  loss_pos = -(1 - p)^gp * log(p)
    For negatives:  let p_m = max(p - clip, 0); x_neg = 1 - p_m
                    loss_neg = -(p_m)^gn * log(x_neg)
    Equivalently we work with x_neg = clamp(1 - p - clip, min=0) for the log term,
    and weight by (1 - x_neg)^gn.
    """
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gp = gamma_pos
        self.gn = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets, pos_weight=None):
        p = torch.sigmoid(logits)

        # Positive branch
        log_pos = torch.log(p.clamp(min=self.eps))
        if self.gp > 0:
            log_pos = log_pos * (1.0 - p) ** self.gp

        # shifted_neg_prob = clamp(1 - p - clip, min=0)
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
    def __init__(self, *args, pos_weight=None, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight
        self.asl = AsymmetricLoss(gamma_pos=gamma_pos, gamma_neg=gamma_neg, clip=clip)
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
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
            best_tau = float(tau)
            best = row
        else:
            if row["micro_f1"] > best["micro_f1"]:
                best_tau = float(tau)
                best = row
            elif math.isclose(row["micro_f1"], best["micro_f1"], rel_tol=0.0, abs_tol=1e-12):
                if row["micro_precision"] > best["micro_precision"]:
                    best_tau = float(tau)
                    best = row
                elif math.isclose(row["micro_precision"], best["micro_precision"], rel_tol=0.0, abs_tol=1e-12):
                    if row["micro_recall"] > best["micro_recall"]:
                        best_tau = float(tau)
                        best = row
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
    return AutoModelForSequenceClassification.from_pretrained(
        CFG["model_name"],
        config=config,
    )

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
    """[FIX-VER] Inject the right tokenizer kwarg for the installed transformers."""
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
        data_seed=CFG["seed"],
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=CFG["early_stopping_patience"])],
    ))
    trainer = ASLTrainer(
        **trainer_kwargs,
        pos_weight=(pos_weight_tensor if CFG["use_pos_weight"] else None),
        gamma_pos=CFG["asl_gamma_pos"],
        gamma_neg=CFG["asl_gamma_neg"],
        clip=CFG["asl_clip"],
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
        gamma_pos=CFG["asl_gamma_pos"],
        gamma_neg=CFG["asl_gamma_neg"],
        clip=CFG["asl_clip"],
    )

    val_out = eval_trainer.predict(val_ds)
    val_logits = val_out.predictions
    val_labels = val_out.label_ids

    val_tau_best, val_best = search_best_global_tau_microf1(val_logits, val_labels)

    trial_row = {
        "run_seed": CFG["seed"],
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

    if select_better_trial(trial_row, best_trial):
        best_trial = trial_row
        best_trial_dir = trial_dir

    # incremental save in case of crash
    pd.DataFrame(grid_rows).to_csv(
        os.path.join(CFG["output_dir"], "grid_search_results_partial.csv"),
        index=False,
    )

    del trainer
    del eval_trainer
    del model
    del best_model
    gc.collect()  # [FIX-GC]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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

best_ckpt = best_trial["best_checkpoint_dir"]
best_tau = float(best_trial["val_best_tau"])
best_batch = int(best_trial["batch_size"])

best_model = AutoModelForSequenceClassification.from_pretrained(best_ckpt)
final_args = TrainingArguments(
    output_dir=os.path.join(CFG["output_dir"], "final_eval_tmp"),
    per_device_eval_batch_size=best_batch,
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
    gamma_pos=CFG["asl_gamma_pos"],
    gamma_neg=CFG["asl_gamma_neg"],
    clip=CFG["asl_clip"],
)

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
        "run_seed": CFG["seed"],
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
    for k, lbl in enumerate(LABELS):
        save_df[f"pred_prob_{lbl}"] = test_probs[:, k]
    save_df[f"pred_labels_tau_{best_tau:.2f}"] = [
        ",".join([LABELS[k] for k, v in enumerate(test_probs[i]) if v >= best_tau])
        for i in range(len(save_df))
    ]
    save_df["best_tau_from_validation"] = float(best_tau)
    save_df["pred_cardinality"] = pred_bin.sum(axis=1)
    save_df["gold_cardinality"] = test_labels.sum(axis=1)
    pred_csv = os.path.join(CFG["output_dir"], "test_predictions_sentence_only_asl_best_validation_tau.csv")
    save_df.to_csv(pred_csv, index=False)

manifest = {
    "run_seed": CFG["seed"],
    "base_output_dir": BASE_OUTPUT_DIR,
    "selection_metric_for_hyperparameters": "validation micro-F1",
    "checkpoint_metric_within_each_run": "validation micro-F1 at best validation-learned tau",
    "threshold_learning": "best global validation tau maximizing validation micro-F1",
    "asl_implementation": "paper (Ridnik et al. 2021): negative branch uses shifted x_neg = clamp(1-p-clip, min=0), weight (1-x_neg)^gamma_neg",
    "best_trial": best_trial,
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
print(f"\n[DONE] Wrote grid results  -> {grid_csv}")
print(f"[DONE] Wrote best config   -> {best_json}")
print(f"[DONE] Wrote test summary  -> {summary_csv}")
print(f"[DONE] Wrote classwise CSV -> {classwise_csv}")
print(f"[DONE] Wrote LaTeX table   -> {latex_path}")
