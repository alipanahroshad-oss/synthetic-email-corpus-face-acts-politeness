#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1. Run OPR BERT text-only with 5 OPR seeds

export CUDA_VISIBLE_DEVICES=1

nohup bash -lc '
for s in 13 21 42 87 100; do
  echo "===== BERT text-only OPR runseed$s ====="

  OPR_RUN_SEED="$s" \
  SPLIT_RANDOM_STATE=42 \
  OUT_DIR="outputs/opr_sentence_only_bert_text_only_opr_runseed$s" \
  RUN_BERT_TEXT=1 \
  RUN_PREDFA_MLP=0 \
  RUN_BERT_PLUS_PREDFA=0 \
  python OPR_BERT_PredFA_gridsearch.py

done
' > opr_sentence_only_bert_text_only_multiseed.log 2>&1 &

tail -f opr_sentence_only_bert_text_only_multiseed.log




2. Run PredFA-only and BERT+PredFA with paired FAC/OPR seeds

export CUDA_VISIBLE_DEVICES=1

export BASE=outputs/bert_multi_label_faceacts_sentence_only_asl

nohup bash -lc '
for s in 13 21 42 87 100; do
  echo "===== PredFA OPR with FAC runseed$s and OPR runseed$s ====="

  CKPT=$(python - "$BASE" "$s" <<PY
import json, sys
base, s = sys.argv[1], sys.argv[2]
p = f"{base}_runseed{s}/best_config.json"
with open(p) as f:
    d = json.load(f)
print(d["best_checkpoint_dir"])
PY
)

  echo "FAC checkpoint: $CKPT"

  FAC_RUN_SEED="$s" \
  OPR_RUN_SEED="$s" \
  SPLIT_RANDOM_STATE=42 \
  ASL_MODEL_DIR="$CKPT" \
  OUT_DIR="outputs/opr_sentence_only_bert_predfa_fac_runseed${s}_opr_runseed${s}" \
  RUN_BERT_TEXT=0 \
  RUN_PREDFA_MLP=1 \
  RUN_BERT_PLUS_PREDFA=1 \
  python OPR_BERT_PredFA_gridsearch.py

done
' > opr_sentence_only_predfa_fac_opr_paired_multiseed.log 2>&1 &

tail -f opr_sentence_only_predfa_fac_opr_paired_multiseed.log


aggregation:

python aggregate_opr_fac_opr_multiseed_results.py \
  --glob_pattern "outputs/opr_sentence_only_bert_predfa_fac_runseed*_opr_runseed*/opr_best_by_family.csv" \
  --out_dir outputs/opr_sentence_only_bert_predfa_fac_opr_paired_multiseed_summary

OPR_BERT_PredFA_gridsearch.py
=============================

BERT version of the OPR pipeline: BERT text-only, PredFA-only MLP and
BERT+PredFA, with PredFA features from the sentence-only BERT FAC model.
"""

# ============================================================
# USER PATHS
# ============================================================
DOC_CSV = "data/corpus/email_text_gold_three_dimensions_politeness_score_with_seed_correct.csv"
SENT_CSV = "data/corpus/sentences_with_golden_face_act.csv"

# UPDATED BEST SENTENCE-ONLY FAC CHECKPOINT
ASL_MODEL_DIR = __import__("os").environ.get(
    "ASL_MODEL_DIR",
    "outputs/bert_multi_label_faceacts_sentence_only_asl_gridsearch_microf1/grid_trials/trial_025_lr3e-05_ep3_bs8_do0.1/checkpoint-555"
)
ASL_TOK_NAME = "bert-base-uncased"

OUT_DIR = __import__("os").environ.get(
    "OUT_DIR",
    "outputs/opr_sentence_only_bert_predfa_gridsearch"
)

# ============================================================
# IMPORTS
# ============================================================
import os, ast, glob, json, math, random, itertools
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from pandas.api.types import CategoricalDtype
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, precision_score, recall_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    get_linear_schedule_with_warmup,
)

# ============================================================
# GLOBAL CONFIG
# ============================================================
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Keep the OPR document split fixed while varying only OPR training randomness.
# SPLIT_RANDOM_STATE controls the grouped train/val/test split.
# OPR_RUN_SEED controls model initialization, dropout, and DataLoader shuffling.
SPLIT_RANDOM_STATE = int(os.environ.get("SPLIT_RANDOM_STATE", "42"))
OPR_RUN_SEED = int(os.environ.get("OPR_RUN_SEED", "42"))
RANDOM_STATE = OPR_RUN_SEED  # backward-compatible name used inside training helpers

DOC_ID_COL = "email_id"
DOC_TEXT_COL = "text_email"
SEED_COL = "seed"
TARGETS = [
    "Directness_vs_Indirectness__GOLD",
    "Structural_Politeness_and_Politeness_Markers__GOLD",
    "Tone_and_Overall_Consideration__GOLD",
]
# "Directness": "Directness_vs_Indirectness__GOLD"
# "Positive Face": "Structural_Politeness_and_Politeness_Markers__GOLD"
# "Negative Face": "Tone_and_Overall_Consideration__GOLD"
SENT_TEXT_COL = "covered_text"
SENT_IDX_COL = "sentence_idx"
FA_COL = "GoldFaceAct"

FA_LABS = ["HNeg+", "HNeg-", "HPos+", "HPos-", "Neutral", "SNeg+", "SNeg-", "SPos+", "SPos-"]
SUM9_ORDER = [f"sumprob_{l}" for l in FA_LABS]
SUM13_EXTRA = ["sum_H", "sum_S", "sum_praise", "sum_threat"]
SUM13_ORDER = SUM9_ORDER + SUM13_EXTRA

DOC_BERT_NAME = "bert-base-uncased"
BERT_MAXLEN = 512
FAC_MAXLEN = 128
FAC_BATCH = 64

# Which systems to run
# Environment overrides use "1"/"0". This lets multi-FAC runs skip
# the pure BERT text-only baseline, which does not depend on FAC checkpoints.
RUN_BERT_TEXT = os.environ.get("RUN_BERT_TEXT", "1") == "1"
RUN_PREDFA_MLP = os.environ.get("RUN_PREDFA_MLP", "1") == "1"
RUN_BERT_PLUS_PREDFA = os.environ.get("RUN_BERT_PLUS_PREDFA", "1") == "1"
DO_MULTI_TASK = os.environ.get("DO_MULTI_TASK", "1") == "1"
DO_SINGLE_TASK = os.environ.get("DO_SINGLE_TASK", "1") == "1"

# IMPORTANT: threshold is learned on validation for hard PredFA.
GRID_PREDFA_MODES = ["hard"]
FAC_TAU_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2).tolist()

# OPR GRID SEARCH SPACE
BERT_GRID = {
    "lr": [2e-5, 3e-5],
    "epochs": [3, 5],
    "batch_size": [8],
    "dropout": [0.1, 0.2],
    "weight_decay": [0.01],
    "hidden_dim": [128],
}

MLP_GRID = {
    "lr": [1e-3, 5e-4],
    "epochs": [40, 60],
    "batch_size": [16, 32],
    "dropout": [0.1, 0.2],
    "weight_decay": [1e-4],
    "hidden_dim": [128, 256],
}

# ============================================================
# UTILS
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def make_dl_generator(seed: int) -> torch.Generator:
    """Dedicated DataLoader RNG for shuffle reproducibility."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _decategorize_series(s: pd.Series) -> pd.Series:
    if isinstance(s.dtype, CategoricalDtype):
        s = s.astype(object)
    return s


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if isinstance(df[col].dtype, CategoricalDtype):
            df[col] = df[col].astype(object)
    if DOC_ID_COL in df.columns:
        df[DOC_ID_COL] = df[DOC_ID_COL].astype(str)
    if SENT_TEXT_COL in df.columns:
        df[SENT_TEXT_COL] = df[SENT_TEXT_COL].astype(str)
    if FA_COL in df.columns:
        df[FA_COL] = df[FA_COL].astype(str)
    if SEED_COL in df.columns:
        df[SEED_COL] = pd.to_numeric(df[SEED_COL], errors="coerce")
    if SENT_IDX_COL in df.columns:
        df[SENT_IDX_COL] = pd.to_numeric(df[SENT_IDX_COL], errors="coerce")
    return df.reset_index(drop=True)


def grouped_split_70_15_15_index(groups: np.ndarray, random_state=42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(groups))
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=random_state)
    tr, tmp = next(gss1.split(idx, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=random_state)
    va_rel, te_rel = next(gss2.split(tmp, groups=groups[tmp]))
    va, te = tmp[va_rel], tmp[te_rel]
    return tr, va, te


def iter_grid(grid: Dict) -> List[Dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]


def parse_gold_faceacts(x) -> List[str]:
    """Robust parser for GoldFaceAct cells."""
    if pd.isna(x):
        return []
    if isinstance(x, list):
        vals = x
    else:
        s = str(x).strip()
        if s in {"", "nan", "None"}:
            return []
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, list):
                vals = obj
            else:
                vals = [str(obj)]
        except Exception:
            if ";" in s:
                vals = s.split(";")
            elif "," in s:
                vals = s.split(",")
            else:
                vals = [s]
    clean = []
    for v in vals:
        v = str(v).strip().strip("'\"")
        if v in FA_LABS:
            clean.append(v)
    return clean


def gold_multihot(df: pd.DataFrame) -> np.ndarray:
    Y = np.zeros((len(df), len(FA_LABS)), dtype=np.int32)
    lab2i = {l: i for i, l in enumerate(FA_LABS)}
    for i, x in enumerate(df[FA_COL].tolist()):
        for lab in parse_gold_faceacts(x):
            Y[i, lab2i[lab]] = 1
    return Y

# ============================================================
# FAC SENTENCE-ONLY INFERENCE
# ============================================================
def _has_model_files(d: str) -> bool:
    return any(
        os.path.isfile(os.path.join(d, fn))
        for fn in ["pytorch_model.bin", "model.safetensors", "tf_model.h5", "model.ckpt.index", "flax_model.msgpack"]
    )


def resolve_hf_model_dir(parent_dir: str) -> str:
    if _has_model_files(parent_dir):
        return parent_dir
    state_path = os.path.join(parent_dir, "trainer_state.json")
    if os.path.isfile(state_path):
        try:
            with open(state_path, "r") as f:
                state = json.load(f)
            best_ckpt = state.get("best_model_checkpoint", None)
            if best_ckpt and _has_model_files(best_ckpt):
                return best_ckpt
        except Exception:
            pass
    ckpts = sorted(glob.glob(os.path.join(parent_dir, "checkpoint-*")), key=os.path.getmtime, reverse=True)
    for ck in ckpts:
        if _has_model_files(ck):
            return ck
    raise OSError(f"No model files found in or under '{parent_dir}'")


class SentDataset(Dataset):
    def __init__(self, df: pd.DataFrame, text_col: str, tokenizer, max_length=128):
        df = df.copy()
        df[DOC_ID_COL] = df[DOC_ID_COL].astype(str)
        df[SENT_IDX_COL] = pd.to_numeric(df[SENT_IDX_COL], errors="coerce").fillna(0).astype(int)
        enc = tokenizer(
            df[text_col].astype(str).tolist(),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attn = enc["attention_mask"]
        self.doc_ids = df[DOC_ID_COL].tolist()
        self.sent_idx = df[SENT_IDX_COL].tolist()

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attn[i],
            "doc_id": self.doc_ids[i],
            "sent_idx": self.sent_idx[i],
        }


def predict_fa_probs(df_split: pd.DataFrame, model_dir: str, tok_name: str, batch=64, max_len=128) -> pd.DataFrame:
    if len(df_split) == 0:
        return pd.DataFrame(columns=[DOC_ID_COL, SENT_IDX_COL] + [f"prob_{l}" for l in FA_LABS])

    try:
        resolved_model_dir = resolve_hf_model_dir(model_dir)
    except OSError as e:
        raise OSError(
            f"[FAC] Could not resolve a Hugging Face checkpoint under '{model_dir}'. "
            f"Original error: {e}"
        ) from e
    print(f"[FAC] Loading sentence-only checkpoint: {resolved_model_dir}")
    tok = AutoTokenizer.from_pretrained(tok_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(resolved_model_dir).to(DEVICE)
    mdl.eval()

    ds = SentDataset(df_split, SENT_TEXT_COL, tok, max_length=max_len)
    dl = DataLoader(ds, batch_size=batch, shuffle=False)

    doc_ids, sids, allP = [], [], []
    with torch.no_grad():
        for b in dl:
            logits = mdl(
                input_ids=b["input_ids"].to(DEVICE),
                attention_mask=b["attention_mask"].to(DEVICE),
            ).logits
            P = torch.sigmoid(logits).cpu().numpy()
            allP.append(P)
            doc_ids.extend(b["doc_id"])
            sids.extend(b["sent_idx"])

    P = np.vstack(allP)
    out = {DOC_ID_COL: list(map(str, doc_ids)), SENT_IDX_COL: list(map(int, sids))}
    for j, l in enumerate(FA_LABS):
        out[f"prob_{l}"] = P[:, j]
    dfp = pd.DataFrame(out)
    dfp[DOC_ID_COL] = dfp[DOC_ID_COL].astype(str)
    dfp[SENT_IDX_COL] = pd.to_numeric(dfp[SENT_IDX_COL], errors="coerce").fillna(0).astype(int)
    return dfp


def tune_fac_tau_on_validation(val_pred: pd.DataFrame, val_sent: pd.DataFrame) -> Dict:
    """Choose tau maximizing validation micro-F1 for FAC multi-label prediction."""
    val_pred = val_pred.sort_values([DOC_ID_COL, SENT_IDX_COL], kind="mergesort").reset_index(drop=True)
    val_sent = val_sent.copy()
    val_sent[DOC_ID_COL] = val_sent[DOC_ID_COL].astype(str)
    val_sent[SENT_IDX_COL] = pd.to_numeric(val_sent[SENT_IDX_COL], errors="coerce").fillna(0).astype(int)
    val_sent = val_sent.sort_values([DOC_ID_COL, SENT_IDX_COL], kind="mergesort").reset_index(drop=True)

    assert len(val_pred) == len(val_sent), "val_pred and val_sent do not align by length."
    assert (val_pred[DOC_ID_COL].tolist() == val_sent[DOC_ID_COL].tolist()), "val_pred and val_sent doc_id order mismatch."
    assert (val_pred[SENT_IDX_COL].tolist() == val_sent[SENT_IDX_COL].tolist()), "val_pred and val_sent sentence_idx order mismatch."

    y_true = gold_multihot(val_sent)
    probs = val_pred[[f"prob_{l}" for l in FA_LABS]].to_numpy(dtype=np.float32)

    rows = []
    best = None
    for tau in FAC_TAU_GRID:
        y_hat = (probs >= tau).astype(np.int32)
        micro_f1 = f1_score(y_true, y_hat, average="micro", zero_division=0)
        micro_p = precision_score(y_true, y_hat, average="micro", zero_division=0)
        micro_r = recall_score(y_true, y_hat, average="micro", zero_division=0)
        row = {"tau": float(tau), "micro_f1": float(micro_f1), "micro_precision": float(micro_p), "micro_recall": float(micro_r)}
        rows.append(row)
        if best is None or row["micro_f1"] > best["micro_f1"]:
            best = row

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "fac_val_tau_search.csv"), index=False)
    with open(os.path.join(OUT_DIR, "fac_best_tau.json"), "w") as f:
        json.dump(best, f, indent=2)
    print(f"[FAC] Best validation tau = {best['tau']:.2f} | micro-F1 = {best['micro_f1']:.4f}")
    return best

# ============================================================
# PREDFA AGGREGATION
# ============================================================
def aggregate_predfa_hspt13(dfp: pd.DataFrame, mode: str = "hard", tau: Optional[float] = None) -> pd.DataFrame:
    """Build HSPT-13 PredFA features per email."""
    if mode not in {"soft", "hard"}:
        raise ValueError("mode must be 'soft' or 'hard'")
    if mode == "hard" and tau is None:
        raise ValueError("tau is required when mode='hard'")

    need = [DOC_ID_COL, SENT_IDX_COL] + [f"prob_{l}" for l in FA_LABS]
    for c in need:
        if c not in dfp.columns:
            raise ValueError(f"Missing PredFA column: {c}")

    dfp = dfp.copy()
    dfp[DOC_ID_COL] = _decategorize_series(dfp[DOC_ID_COL]).astype(str)
    dfp[SENT_IDX_COL] = pd.to_numeric(_decategorize_series(dfp[SENT_IDX_COL]), errors="coerce").fillna(0).astype(int)
    for l in FA_LABS:
        col = f"prob_{l}"
        vals = pd.to_numeric(_decategorize_series(dfp[col]), errors="coerce").fillna(0.0).astype(float)
        if mode == "hard":
            vals = (vals >= float(tau)).astype(float)
        dfp[col] = vals

    dfp = dfp.sort_values([DOC_ID_COL, SENT_IDX_COL], kind="mergesort", ignore_index=True)

    rows = []
    for doc_id, g in dfp.groupby(DOC_ID_COL, sort=False):
        row = {DOC_ID_COL: doc_id}
        for l in FA_LABS:
            row[f"sumprob_{l}"] = float(g[f"prob_{l}"].sum())

        row["sum_H"] = row["sumprob_HNeg+"] + row["sumprob_HNeg-"] + row["sumprob_HPos+"] + row["sumprob_HPos-"]
        row["sum_S"] = row["sumprob_SNeg+"] + row["sumprob_SNeg-"] + row["sumprob_SPos+"] + row["sumprob_SPos-"]
        row["sum_praise"] = row["sumprob_HPos+"] + row["sumprob_HNeg+"] + row["sumprob_SPos+"] + row["sumprob_SNeg+"]
        row["sum_threat"] = row["sumprob_HPos-"] + row["sumprob_HNeg-"] + row["sumprob_SPos-"] + row["sumprob_SNeg-"]
        rows.append(row)

    df_out = pd.DataFrame(rows)
    keep_cols = [DOC_ID_COL] + SUM13_ORDER
    for c in keep_cols:
        if c not in df_out.columns:
            df_out[c] = 0.0
    return df_out[keep_cols]


def align_predfa_to_split(F_pred: pd.DataFrame, df_doc: pd.DataFrame, idx: np.ndarray, split_name: str = "") -> np.ndarray:
    """Use set_index+reindex to guarantee row order matches idx,
    instead of relying on pandas merge preserving order with duplicate-edge cases.

    Warn loudly if any doc_id in the split is missing from F_pred
    (which would silently zero-fill the 13 features). Save the list of missing
    ids to disk for diagnosis.
    """
    want_doc_ids = df_doc.loc[idx, DOC_ID_COL].astype(str).tolist()

    F_pred_indexed = F_pred.copy()
    F_pred_indexed[DOC_ID_COL] = F_pred_indexed[DOC_ID_COL].astype(str)
    F_pred_indexed = F_pred_indexed.drop_duplicates(subset=[DOC_ID_COL], keep="first")
    F_pred_indexed = F_pred_indexed.set_index(DOC_ID_COL)

    # Coverage diagnostic.
    have_ids = set(F_pred_indexed.index.tolist())
    missing = [d for d in want_doc_ids if d not in have_ids]
    if missing:
        msg = (f"[ALIGN][{split_name}] WARNING: {len(missing)}/{len(want_doc_ids)} doc_ids missing "
               f"from PredFA features. Those rows will be zero-filled (potentially biased).")
        print(msg)
        if split_name:
            try:
                miss_path = os.path.join(OUT_DIR, f"predfa_missing_ids_{split_name}.json")
                with open(miss_path, "w") as f:
                    json.dump({"split": split_name, "n_missing": len(missing), "missing_doc_ids": missing}, f, indent=2)
            except Exception:
                pass

    # reindex preserves the order of want_doc_ids exactly.
    F_aligned = F_pred_indexed.reindex(want_doc_ids)
    # Zero-fill any missing rows (NaN -> 0 for the 13 feature columns).
    for c in SUM13_ORDER:
        if c not in F_aligned.columns:
            F_aligned[c] = 0.0
    F_aligned[SUM13_ORDER] = F_aligned[SUM13_ORDER].fillna(0.0)

    return F_aligned[SUM13_ORDER].to_numpy(dtype=np.float32)


def zscore_fit_transform(Xtr: np.ndarray, Xva: np.ndarray, Xte: np.ndarray):
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd == 0] = 1.0
    return (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd, mu, sd

# ============================================================
# DOC DATASETS / MODELS
# ============================================================
class DocBertDataset(Dataset):
    def __init__(self, df: pd.DataFrame, text_col: str, targets: List[str], tokenizer):
        enc = tokenizer(
            df[text_col].astype(str).tolist(),
            padding=True,
            truncation=True,
            max_length=BERT_MAXLEN,
            return_tensors="pt",
        )
        self.ids = enc["input_ids"]
        self.attn = enc["attention_mask"]
        self.y = torch.tensor(df[targets].to_numpy(np.float32))

    def __len__(self):
        return self.ids.size(0)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "attention_mask": self.attn[i], "targets": self.y[i]}


class DocBertFusionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, text_col: str, targets: List[str], tokenizer, predfa: np.ndarray):
        assert len(df) == len(predfa), "PredFA rows must align with doc rows."
        enc = tokenizer(
            df[text_col].astype(str).tolist(),
            padding=True,
            truncation=True,
            max_length=BERT_MAXLEN,
            return_tensors="pt",
        )
        self.ids = enc["input_ids"]
        self.attn = enc["attention_mask"]
        self.y = torch.tensor(df[targets].to_numpy(np.float32))
        self.predfa = torch.tensor(predfa, dtype=torch.float32)

    def __len__(self):
        return self.ids.size(0)

    def __getitem__(self, i):
        return {
            "input_ids": self.ids[i],
            "attention_mask": self.attn[i],
            "targets": self.y[i],
            "predfa": self.predfa[i],
        }


class DocPredFADataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, i):
        return {"x": self.X[i], "targets": self.Y[i]}


class BertDocRegressor(nn.Module):
    def __init__(self, encoder_name=DOC_BERT_NAME, n_targets=3, fusion_dim=0, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.enc = AutoModel.from_pretrained(encoder_name)
        hdim = self.enc.config.hidden_size
        self.fusion_dim = int(fusion_dim)
        in_dim = hdim + self.fusion_dim
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_targets)])

    def forward(self, input_ids, attention_mask, predfa: Optional[torch.Tensor] = None):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0, :]
        if self.fusion_dim > 0:
            if predfa is None:
                raise ValueError("predfa is required when fusion_dim > 0")
            z = torch.cat([cls, predfa], dim=-1)
        else:
            z = cls
        z = self.dropout(z)
        h = self.hidden(z)
        return torch.cat([head(h) for head in self.heads], dim=-1)


class PredFAMLP(nn.Module):
    def __init__(self, in_dim, out_dim=3, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)

# ============================================================
# METRICS / TRAIN HELPERS
# ============================================================
def mae_spearman(y_true: np.ndarray, y_pred: np.ndarray, target_names: List[str]) -> Dict:
    out = {"per_target": {}}
    maes, rhos = [], []
    for i, name in enumerate(target_names):
        yt, yp = y_true[:, i], y_pred[:, i]
        mae = float(np.mean(np.abs(yt - yp)))
        rho, _ = spearmanr(yt, yp)
        rho = float(rho) if rho is not None and not np.isnan(rho) else 0.0
        maes.append(mae)
        rhos.append(rho)
        out["per_target"][name] = {"MAE": mae, "Spearman": rho}
    out["mae_macro"] = float(np.mean(maes))
    out["rho_macro"] = float(np.mean(rhos))
    return out


def predict_bert(model, dl):
    model.eval()
    Ys, Ps = [], []
    with torch.no_grad():
        for b in dl:
            pf = b.get("predfa", None)
            pf = pf.to(DEVICE) if pf is not None else None
            pr = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), pf)
            Ys.append(b["targets"].numpy())
            Ps.append(pr.cpu().numpy())
    return np.concatenate(Ys, 0), np.concatenate(Ps, 0)


def predict_mlp(model, dl):
    model.eval()
    Ys, Ps = [], []
    with torch.no_grad():
        for b in dl:
            pr = model(b["x"].to(DEVICE))
            Ys.append(b["targets"].numpy())
            Ps.append(pr.cpu().numpy())
    return np.concatenate(Ys, 0), np.concatenate(Ps, 0)


def save_run(outdir: str, cfg: Dict, Yva, Pva, Yte, Pte):
    ensure_dir(outdir)
    metrics = {
        "config": cfg,
        "val": mae_spearman(Yva, Pva, TARGETS),
        "test": mae_spearman(Yte, Pte, TARGETS),
        "targets": TARGETS,
    }
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(outdir, "val_y.npy"), Yva)
    np.save(os.path.join(outdir, "val_pred.npy"), Pva)
    np.save(os.path.join(outdir, "test_y.npy"), Yte)
    np.save(os.path.join(outdir, "test_pred.npy"), Pte)
    return metrics


def train_bert_once(df_doc, tr_idx, va_idx, te_idx, cfg: Dict, predfa_feats=None, multi_task=True):
    set_seed(RANDOM_STATE)
    tk = AutoTokenizer.from_pretrained(DOC_BERT_NAME, use_fast=True)
    df_tr = df_doc.loc[tr_idx].reset_index(drop=True)
    df_va = df_doc.loc[va_idx].reset_index(drop=True)
    df_te = df_doc.loc[te_idx].reset_index(drop=True)

    if predfa_feats is None:
        ds_tr = DocBertDataset(df_tr, DOC_TEXT_COL, TARGETS, tk)
        ds_va = DocBertDataset(df_va, DOC_TEXT_COL, TARGETS, tk)
        ds_te = DocBertDataset(df_te, DOC_TEXT_COL, TARGETS, tk)
        fusion_dim = 0
    else:
        Xtr, Xva, Xte = predfa_feats
        ds_tr = DocBertFusionDataset(df_tr, DOC_TEXT_COL, TARGETS, tk, Xtr)
        ds_va = DocBertFusionDataset(df_va, DOC_TEXT_COL, TARGETS, tk, Xva)
        ds_te = DocBertFusionDataset(df_te, DOC_TEXT_COL, TARGETS, tk, Xte)
        fusion_dim = Xtr.shape[1]

    g = make_dl_generator(RANDOM_STATE)
    dl_tr = DataLoader(ds_tr, batch_size=cfg["batch_size"], shuffle=True, generator=g)
    dl_va = DataLoader(ds_va, batch_size=cfg["batch_size"], shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=cfg["batch_size"], shuffle=False)

    if multi_task:
        model = BertDocRegressor(
            encoder_name=DOC_BERT_NAME,
            n_targets=3,
            fusion_dim=fusion_dim,
            hidden_dim=cfg["hidden_dim"],
            dropout=cfg["dropout"],
        ).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        steps_total = max(1, len(dl_tr) * cfg["epochs"])
        sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps_total), steps_total)
        loss_fn = nn.SmoothL1Loss()

        best_score, best_state = -1e9, None
        for _ in range(cfg["epochs"]):
            model.train()
            for b in dl_tr:
                opt.zero_grad()
                pf = b.get("predfa", None)
                pf = pf.to(DEVICE) if pf is not None else None
                pr = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), pf)
                loss = loss_fn(pr, b["targets"].to(DEVICE))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sch.step()
            Yv, Pv = predict_bert(model, dl_va)
            score = mae_spearman(Yv, Pv, TARGETS)["rho_macro"]
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        Yva, Pva = predict_bert(model, dl_va)
        Yte, Pte = predict_bert(model, dl_te)
        return Yva, Pva, Yte, Pte

    # single-task
    Yva_all = df_doc.loc[va_idx, TARGETS].to_numpy(np.float32)
    Yte_all = df_doc.loc[te_idx, TARGETS].to_numpy(np.float32)
    Pva_all = np.zeros_like(Yva_all)
    Pte_all = np.zeros_like(Yte_all)

    for t_i in range(3):
        set_seed(RANDOM_STATE + t_i)
        g = make_dl_generator(RANDOM_STATE + t_i)
        dl_tr = DataLoader(ds_tr, batch_size=cfg["batch_size"], shuffle=True, generator=g)

        model = BertDocRegressor(
            encoder_name=DOC_BERT_NAME,
            n_targets=1,
            fusion_dim=fusion_dim,
            hidden_dim=cfg["hidden_dim"],
            dropout=cfg["dropout"],
        ).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        steps_total = max(1, len(dl_tr) * cfg["epochs"])
        sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps_total), steps_total)
        loss_fn = nn.SmoothL1Loss()
        best_score, best_state = -1e9, None

        for _ in range(cfg["epochs"]):
            model.train()
            for b in dl_tr:
                opt.zero_grad()
                pf = b.get("predfa", None)
                pf = pf.to(DEVICE) if pf is not None else None
                y = b["targets"][:, t_i].unsqueeze(1).to(DEVICE)
                pr = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), pf)
                loss = loss_fn(pr, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sch.step()

            model.eval()
            yv, pv = [], []
            with torch.no_grad():
                for b in dl_va:
                    pf = b.get("predfa", None)
                    pf = pf.to(DEVICE) if pf is not None else None
                    pr = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), pf)
                    yv.append(b["targets"][:, t_i].numpy())
                    pv.append(pr.cpu().numpy().squeeze(1))
            yv = np.concatenate(yv, 0)
            pv = np.concatenate(pv, 0)
            rho, _ = spearmanr(yv, pv)
            score = float(rho) if rho is not None and not np.isnan(rho) else 0.0
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        for dl, store in [(dl_va, Pva_all), (dl_te, Pte_all)]:
            preds = []
            with torch.no_grad():
                for b in dl:
                    pf = b.get("predfa", None)
                    pf = pf.to(DEVICE) if pf is not None else None
                    pr = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), pf)
                    preds.append(pr.cpu().numpy().squeeze(1))
            store[:, t_i] = np.concatenate(preds, 0)

    return Yva_all, Pva_all, Yte_all, Pte_all


def train_mlp_once(Xtr, Xva, Xte, Ytr, Yva, Yte, cfg: Dict, multi_task=True):
    set_seed(RANDOM_STATE)
    g = make_dl_generator(RANDOM_STATE)
    dl_tr = DataLoader(DocPredFADataset(Xtr, Ytr), batch_size=cfg["batch_size"], shuffle=True, generator=g)
    dl_va = DataLoader(DocPredFADataset(Xva, Yva), batch_size=cfg["batch_size"], shuffle=False)
    dl_te = DataLoader(DocPredFADataset(Xte, Yte), batch_size=cfg["batch_size"], shuffle=False)

    if multi_task:
        model = PredFAMLP(Xtr.shape[1], out_dim=3, hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"]).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        loss_fn = nn.SmoothL1Loss()
        best_score, best_state = -1e9, None
        for _ in range(cfg["epochs"]):
            model.train()
            for b in dl_tr:
                opt.zero_grad()
                pr = model(b["x"].to(DEVICE))
                loss = loss_fn(pr, b["targets"].to(DEVICE))
                loss.backward()
                opt.step()
            Yv, Pv = predict_mlp(model, dl_va)
            score = mae_spearman(Yv, Pv, TARGETS)["rho_macro"]
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        Yva_out, Pva = predict_mlp(model, dl_va)
        Yte_out, Pte = predict_mlp(model, dl_te)
        return Yva_out, Pva, Yte_out, Pte

    Pva = np.zeros_like(Yva)
    Pte = np.zeros_like(Yte)
    for t_i in range(3):
        set_seed(RANDOM_STATE + t_i)
        g = make_dl_generator(RANDOM_STATE + t_i)
        dl_tr = DataLoader(DocPredFADataset(Xtr, Ytr), batch_size=cfg["batch_size"], shuffle=True, generator=g)

        model = PredFAMLP(Xtr.shape[1], out_dim=1, hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"]).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        loss_fn = nn.SmoothL1Loss()
        best_score, best_state = -1e9, None
        for _ in range(cfg["epochs"]):
            model.train()
            for b in dl_tr:
                opt.zero_grad()
                y = b["targets"][:, t_i].unsqueeze(1).to(DEVICE)
                pr = model(b["x"].to(DEVICE))
                loss = loss_fn(pr, y)
                loss.backward()
                opt.step()
            model.eval()
            yv, pv = [], []
            with torch.no_grad():
                for b in dl_va:
                    pr = model(b["x"].to(DEVICE)).cpu().numpy().squeeze(1)
                    yv.append(b["targets"][:, t_i].numpy())
                    pv.append(pr)
            yv = np.concatenate(yv, 0)
            pv = np.concatenate(pv, 0)
            rho, _ = spearmanr(yv, pv)
            score = float(rho) if rho is not None and not np.isnan(rho) else 0.0
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        for dl, store in [(dl_va, Pva), (dl_te, Pte)]:
            preds = []
            with torch.no_grad():
                for b in dl:
                    preds.append(model(b["x"].to(DEVICE)).cpu().numpy().squeeze(1))
            store[:, t_i] = np.concatenate(preds, 0)
    return Yva, Pva, Yte, Pte

# ============================================================
# MAIN
# ============================================================
def _run_metadata() -> Dict:
    return {
        "opr_run_seed": int(OPR_RUN_SEED),
        "split_random_state": int(SPLIT_RANDOM_STATE),
        "fac_run_seed": os.environ.get("FAC_RUN_SEED", ""),
        "fac_checkpoint": ASL_MODEL_DIR,
    }


def _row_with_metadata(row: Dict) -> Dict:
    return {**row, **_run_metadata()}


def main():
    set_seed(OPR_RUN_SEED)
    ensure_dir(OUT_DIR)

    need_predfa = RUN_PREDFA_MLP or RUN_BERT_PLUS_PREDFA

    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Output dir: {OUT_DIR}")
    print(f"[INFO] Split random state: {SPLIT_RANDOM_STATE}")
    print(f"[INFO] OPR run seed: {OPR_RUN_SEED}")
    print(f"[INFO] FAC run seed: {os.environ.get('FAC_RUN_SEED', '')}")
    print(f"[INFO] FAC checkpoint: {ASL_MODEL_DIR if need_predfa else '[not used: PredFA systems disabled]'}")
    print(f"[INFO] RUN_BERT_TEXT={RUN_BERT_TEXT} RUN_PREDFA_MLP={RUN_PREDFA_MLP} RUN_BERT_PLUS_PREDFA={RUN_BERT_PLUS_PREDFA}")

    df_doc = sanitize_dataframe(pd.read_csv(DOC_CSV))
    df_sent = sanitize_dataframe(pd.read_csv(SENT_CSV))

    if SEED_COL not in df_doc.columns:
        seed_map = (
            df_sent[[DOC_ID_COL, SEED_COL]]
            .dropna()
            .drop_duplicates(subset=[DOC_ID_COL, SEED_COL])
            .groupby(DOC_ID_COL)[SEED_COL]
            .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0])
            .reset_index()
        )
        df_doc = sanitize_dataframe(df_doc.merge(seed_map, on=DOC_ID_COL, how="left"))

    for t in TARGETS:
        df_doc[t] = pd.to_numeric(df_doc[t], errors="coerce")
    pre_n = len(df_doc)
    df_doc = df_doc.dropna(subset=TARGETS).reset_index(drop=True)
    post_n = len(df_doc)
    if post_n < pre_n:
        print(f"[INFO] Dropped {pre_n - post_n} rows with non-numeric or NaN targets. {post_n} rows remain.")

    df_doc[DOC_ID_COL] = df_doc[DOC_ID_COL].astype(str)
    df_sent[DOC_ID_COL] = df_sent[DOC_ID_COL].astype(str)

    dup_count = df_doc[DOC_ID_COL].duplicated().sum()
    if dup_count:
        print(f"[WARN] {dup_count} duplicate {DOC_ID_COL} values in df_doc; keeping first.")
        df_doc = df_doc.drop_duplicates(subset=[DOC_ID_COL], keep="first").reset_index(drop=True)

    df_doc = df_doc.sort_values(DOC_ID_COL, kind="mergesort").reset_index(drop=True)

    if df_doc[SEED_COL].isna().any():
        n_bad = int(df_doc[SEED_COL].isna().sum())
        print(f"[WARN] {n_bad} rows have missing seed; dropping them.")
        df_doc = df_doc.dropna(subset=[SEED_COL]).reset_index(drop=True)
    df_doc[SEED_COL] = df_doc[SEED_COL].astype(int)

    if len(df_doc) == 0:
        raise RuntimeError("df_doc is empty after sanitization; nothing to train on.")

    groups = df_doc[SEED_COL].to_numpy().astype(int)
    # IMPORTANT: keep the split fixed across OPR_RUN_SEED values.
    tr_idx, va_idx, te_idx = grouped_split_70_15_15_index(groups, random_state=SPLIT_RANDOM_STATE)
    seeds_tr, seeds_va, seeds_te = set(groups[tr_idx]), set(groups[va_idx]), set(groups[te_idx])
    assert not (seeds_tr & seeds_va) and not (seeds_tr & seeds_te) and not (seeds_va & seeds_te)

    if len(tr_idx) == 0 or len(va_idx) == 0 or len(te_idx) == 0:
        raise RuntimeError(f"empty split: train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")

    split_info = {
        "split_random_state": SPLIT_RANDOM_STATE,
        "opr_run_seed": OPR_RUN_SEED,
        "n_docs_train": int(len(tr_idx)),
        "n_docs_val": int(len(va_idx)),
        "n_docs_test": int(len(te_idx)),
        "train_seeds": sorted(map(int, seeds_tr)),
        "val_seeds": sorted(map(int, seeds_va)),
        "test_seeds": sorted(map(int, seeds_te)),
        "fac_checkpoint": ASL_MODEL_DIR if need_predfa else "",
        "fac_run_seed": os.environ.get("FAC_RUN_SEED", ""),
        "run_bert_text": RUN_BERT_TEXT,
        "run_predfa_mlp": RUN_PREDFA_MLP,
        "run_bert_plus_predfa": RUN_BERT_PLUS_PREDFA,
    }
    with open(os.path.join(OUT_DIR, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"[INFO] Split: train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")

    # ---------- Targets
    Y = df_doc[TARGETS].to_numpy(np.float32)
    Y_tr, Y_va, Y_te = Y[tr_idx], Y[va_idx], Y[te_idx]

    # ---------- Run OPR grids
    all_rows = []
    best_by_family = {}

    # Family 1: BERT text-only. This does not depend on FAC checkpoints.
    # It can be run with RUN_BERT_TEXT=1 and RUN_PREDFA_MLP=RUN_BERT_PLUS_PREDFA=0.
    if RUN_BERT_TEXT:
        for multi_task in ([True] if DO_MULTI_TASK else []) + ([False] if DO_SINGLE_TASK else []):
            family = f"bert_text_{'mt' if multi_task else 'st'}"
            print(f"\n[GRID] {family}")
            for run_id, cfg in enumerate(iter_grid(BERT_GRID), start=1):
                print(f"[RUN] {family} #{run_id}: {cfg}")
                Yv, Pv, Yt, Pt = train_bert_once(df_doc, tr_idx, va_idx, te_idx, cfg, predfa_feats=None, multi_task=multi_task)
                run_dir = os.path.join(OUT_DIR, family, f"run_{run_id:03d}")
                cfg_save = _row_with_metadata({**cfg, "family": family, "predfa_mode": "none", "tau": None})
                metrics = save_run(run_dir, cfg_save, Yv, Pv, Yt, Pt)
                row = _row_with_metadata({**cfg, "family": family, "run_id": run_id, "predfa_mode": "none", "tau": None})
                row.update({
                    "val_mae_macro": metrics["val"]["mae_macro"],
                    "val_rho_macro": metrics["val"]["rho_macro"],
                    "test_mae_macro": metrics["test"]["mae_macro"],
                    "test_rho_macro": metrics["test"]["rho_macro"],
                    "run_dir": run_dir,
                })
                all_rows.append(row)
                if family not in best_by_family or row["val_rho_macro"] > best_by_family[family]["val_rho_macro"]:
                    best_by_family[family] = row

    if need_predfa:
        need_cols = [DOC_ID_COL, SENT_IDX_COL, SENT_TEXT_COL, SEED_COL, FA_COL]
        assert all(c in df_sent.columns for c in need_cols), f"Sentence CSV missing columns: {need_cols}"

        ids_tr = set(df_doc.loc[tr_idx, DOC_ID_COL].astype(str))
        ids_va = set(df_doc.loc[va_idx, DOC_ID_COL].astype(str))
        ids_te = set(df_doc.loc[te_idx, DOC_ID_COL].astype(str))

        df_tr_sent = df_sent[df_sent[DOC_ID_COL].isin(ids_tr)].reset_index(drop=True)
        df_va_sent = df_sent[df_sent[DOC_ID_COL].isin(ids_va)].reset_index(drop=True)
        df_te_sent = df_sent[df_sent[DOC_ID_COL].isin(ids_te)].reset_index(drop=True)

        # ---------- FAC inference using selected sentence-only checkpoint
        print("\n[FAC] Predicting sentence-level FA probabilities with selected sentence-only checkpoint...")
        tr_pred = predict_fa_probs(df_tr_sent, ASL_MODEL_DIR, ASL_TOK_NAME, batch=FAC_BATCH, max_len=FAC_MAXLEN)
        va_pred = predict_fa_probs(df_va_sent, ASL_MODEL_DIR, ASL_TOK_NAME, batch=FAC_BATCH, max_len=FAC_MAXLEN)
        te_pred = predict_fa_probs(df_te_sent, ASL_MODEL_DIR, ASL_TOK_NAME, batch=FAC_BATCH, max_len=FAC_MAXLEN)

        tr_pred.to_csv(os.path.join(OUT_DIR, "predfa_sentence_probs_train.csv"), index=False)
        va_pred.to_csv(os.path.join(OUT_DIR, "predfa_sentence_probs_val.csv"), index=False)
        te_pred.to_csv(os.path.join(OUT_DIR, "predfa_sentence_probs_test.csv"), index=False)

        # ---------- Learn FAC threshold on validation set
        best_tau_info = tune_fac_tau_on_validation(va_pred, df_va_sent)
        best_tau = float(best_tau_info["tau"])

        for predfa_mode in GRID_PREDFA_MODES:
            print(f"\n[PredFA] Building HSPT-13 features with mode='{predfa_mode}'")
            tau_for_mode = best_tau if predfa_mode == "hard" else None
            F_tr = aggregate_predfa_hspt13(tr_pred, mode=predfa_mode, tau=tau_for_mode)
            F_va = aggregate_predfa_hspt13(va_pred, mode=predfa_mode, tau=tau_for_mode)
            F_te = aggregate_predfa_hspt13(te_pred, mode=predfa_mode, tau=tau_for_mode)

            Xtr_raw = align_predfa_to_split(F_tr, df_doc, tr_idx, split_name="train")
            Xva_raw = align_predfa_to_split(F_va, df_doc, va_idx, split_name="val")
            Xte_raw = align_predfa_to_split(F_te, df_doc, te_idx, split_name="test")
            Xtr, Xva, Xte, mu, sd = zscore_fit_transform(Xtr_raw, Xva_raw, Xte_raw)

            mode_dir = os.path.join(OUT_DIR, f"predfa_{predfa_mode}")
            ensure_dir(mode_dir)
            pd.DataFrame({"feature": SUM13_ORDER, "mu": mu.tolist(), "sd": sd.tolist()}).to_csv(
                os.path.join(mode_dir, "hspt13_zscore_params.csv"), index=False
            )
            with open(os.path.join(mode_dir, "hspt13_feature_order.json"), "w") as f:
                json.dump(SUM13_ORDER, f, indent=2)

            # Family 2: PredFA-only MLP
            if RUN_PREDFA_MLP:
                for multi_task in ([True] if DO_MULTI_TASK else []) + ([False] if DO_SINGLE_TASK else []):
                    family = f"predfa_mlp_{predfa_mode}_{'mt' if multi_task else 'st'}"
                    print(f"\n[GRID] {family}")
                    for run_id, cfg in enumerate(iter_grid(MLP_GRID), start=1):
                        print(f"[RUN] {family} #{run_id}: {cfg}")
                        Yv, Pv, Yt, Pt = train_mlp_once(Xtr, Xva, Xte, Y_tr, Y_va, Y_te, cfg, multi_task=multi_task)
                        run_dir = os.path.join(OUT_DIR, family, f"run_{run_id:03d}")
                        cfg_save = _row_with_metadata({**cfg, "family": family, "predfa_mode": predfa_mode, "tau": tau_for_mode})
                        metrics = save_run(run_dir, cfg_save, Yv, Pv, Yt, Pt)
                        row = _row_with_metadata({**cfg, "family": family, "run_id": run_id, "predfa_mode": predfa_mode, "tau": tau_for_mode})
                        row.update({
                            "val_mae_macro": metrics["val"]["mae_macro"],
                            "val_rho_macro": metrics["val"]["rho_macro"],
                            "test_mae_macro": metrics["test"]["mae_macro"],
                            "test_rho_macro": metrics["test"]["rho_macro"],
                            "run_dir": run_dir,
                        })
                        all_rows.append(row)
                        if family not in best_by_family or row["val_rho_macro"] > best_by_family[family]["val_rho_macro"]:
                            best_by_family[family] = row

            # Family 3: BERT + PredFA
            if RUN_BERT_PLUS_PREDFA:
                for multi_task in ([True] if DO_MULTI_TASK else []) + ([False] if DO_SINGLE_TASK else []):
                    family = f"bert_plus_predfa_{predfa_mode}_{'mt' if multi_task else 'st'}"
                    print(f"\n[GRID] {family}")
                    for run_id, cfg in enumerate(iter_grid(BERT_GRID), start=1):
                        print(f"[RUN] {family} #{run_id}: {cfg}")
                        Yv, Pv, Yt, Pt = train_bert_once(
                            df_doc, tr_idx, va_idx, te_idx, cfg,
                            predfa_feats=(Xtr, Xva, Xte), multi_task=multi_task,
                        )
                        run_dir = os.path.join(OUT_DIR, family, f"run_{run_id:03d}")
                        cfg_save = _row_with_metadata({**cfg, "family": family, "predfa_mode": predfa_mode, "tau": tau_for_mode})
                        metrics = save_run(run_dir, cfg_save, Yv, Pv, Yt, Pt)
                        row = _row_with_metadata({**cfg, "family": family, "run_id": run_id, "predfa_mode": predfa_mode, "tau": tau_for_mode})
                        row.update({
                            "val_mae_macro": metrics["val"]["mae_macro"],
                            "val_rho_macro": metrics["val"]["rho_macro"],
                            "test_mae_macro": metrics["test"]["mae_macro"],
                            "test_rho_macro": metrics["test"]["rho_macro"],
                            "run_dir": run_dir,
                        })
                        all_rows.append(row)
                        if family not in best_by_family or row["val_rho_macro"] > best_by_family[family]["val_rho_macro"]:
                            best_by_family[family] = row
    else:
        print("[INFO] PredFA systems disabled; skipping FAC inference and HSPT-13 feature construction.")

    if not all_rows:
        raise RuntimeError("No OPR systems were run. Check RUN_BERT_TEXT/RUN_PREDFA_MLP/RUN_BERT_PLUS_PREDFA flags.")

    grid_df = pd.DataFrame(all_rows)
    grid_path = os.path.join(OUT_DIR, "opr_grid_search_results.csv")
    grid_df.to_csv(grid_path, index=False)

    best_df = pd.DataFrame(list(best_by_family.values())).sort_values("val_rho_macro", ascending=False)
    best_path = os.path.join(OUT_DIR, "opr_best_by_family.csv")
    best_df.to_csv(best_path, index=False)

    print("\n=== BEST BY FAMILY, selected by validation rho_macro ===")
    cols = [
        "family", "opr_run_seed", "fac_run_seed", "val_rho_macro", "val_mae_macro",
        "test_rho_macro", "test_mae_macro", "lr", "epochs", "batch_size",
        "dropout", "hidden_dim", "predfa_mode", "tau", "run_dir",
    ]
    cols = [c for c in cols if c in best_df.columns]
    if len(best_df):
        print(best_df[cols].to_string(index=False))

    print(f"\n[DONE] Wrote grid results -> {grid_path}")
    print(f"[DONE] Wrote best by family -> {best_path}")
    if need_predfa:
        print(f"[DONE] Wrote FAC tau search -> {os.path.join(OUT_DIR, 'fac_val_tau_search.csv')}")
    print(f"[DONE] Output dir -> {OUT_DIR}")


if __name__ == "__main__":
    main()
