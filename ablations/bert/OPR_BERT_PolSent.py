#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
OPR_BERT_PolSent_PredFA.py
==========================
export NLTK_DATA=$HOME/nltk_data

mkdir -p "$NLTK_DATA"

python - <<'PY'
import os
import nltk

download_dir = os.environ["NLTK_DATA"]
print("Downloading vader_lexicon to:", download_dir)
nltk.download("vader_lexicon", download_dir=download_dir)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
print(SentimentIntensityAnalyzer().polarity_scores("This is a polite and helpful email."))
print("VADER OK")
PY
-----------------
run:

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export NLTK_DATA=$HOME/nltk_data

export BASE=outputs/bert_multi_label_faceacts_sentence_only_asl

nohup bash -lc '
echo "======================================"
echo "START BERT Pol/Sent/PredFA OPR ST-only"
echo "======================================"

for s in 13 21 42 87 100; do
  echo ""
  echo "===== BERT Pol/Sent/PredFA ST runseed$s ====="

  CKPT=$(python - "$BASE" "$s" <<PY
import json, sys
base, s = sys.argv[1], sys.argv[2]
p = f"{base}_runseed{s}/best_config.json"
with open(p) as f:
    d = json.load(f)
print(d["best_checkpoint_dir"])
PY
)

  FAC_RUN_SEED="$s" \
  OPR_RUN_SEED="$s" \
  SPLIT_RANDOM_STATE=42 \
  ASL_MODEL_DIR="$CKPT" \
  OUT_DIR="outputs/opr_sentence_only_bert_polsent_predfa_ST_fac_runseed${s}_opr_runseed${s}" \
  RUN_POL_MLP=1 \
  RUN_SENT_MLP=1 \
  RUN_BERT_PLUS_POL_SENT=1 \
  RUN_BERT_PLUS_POL_SENT_PREDFA=1 \
  RUN_POL_PLUS_SENT_MLP=0 \
  DO_MULTI_TASK=0 \
  DO_SINGLE_TASK=1 \
  python -u OPR_BERT_PolSent_PredFA.py

  echo "===== DONE BERT Pol/Sent/PredFA ST runseed$s ====="
done

echo ""
echo "======================================"
echo "DONE BERT Pol/Sent/PredFA OPR ST-only"
echo "======================================"
' > opr_bert_polsent_predfa_ST_multiseed.log 2>&1 &

tail -f opr_bert_polsent_predfa_ST_multiseed.log
------------------------
python - <<'PY'
import os, glob, json, re
import pandas as pd

OUT_DIR = "outputs/opr_sentence_only_bert_polsent_predfa_ST_multiseed_summary"
os.makedirs(OUT_DIR, exist_ok=True)

TARGETS = {
    "Directness": "Directness_vs_Indirectness__GOLD",
    "Positive Face Saving": "Structural_Politeness_and_Politeness_Markers__GOLD",
    "Negative Face Saving": "Tone_and_Overall_Consideration__GOLD",
}

FAMILIES = [
    "pol_mlp_st",
    "sent_mlp_st",
    "bert_plus_pol_sent_st",
    "bert_plus_pol_sent_predfa_st",
]

paths = sorted(glob.glob(
    "outputs/opr_sentence_only_bert_polsent_predfa_ST_fac_runseed*_opr_runseed*/opr_best_by_family.csv"
))

if not paths:
    raise RuntimeError("No opr_best_by_family.csv files found.")

print("[INFO] Found files:")
for p in paths:
    print(" ", p)

rows = []
per_dim_rows = []

for p in paths:
    m = re.search(r"fac_runseed(\d+)_opr_runseed(\d+)", p)
    fac_seed = int(m.group(1)) if m else None
    opr_seed = int(m.group(2)) if m else None

    df = pd.read_csv(p)

    for fam in FAMILIES:
        r = df[df["family"] == fam]
        if len(r) != 1:
            print(f"[WARN] {p}: expected one row for {fam}, found {len(r)}")
            continue

        r = r.iloc[0].to_dict()
        r["fac_run_seed"] = fac_seed
        r["opr_run_seed"] = opr_seed
        rows.append(r)

        metrics_path = os.path.join(str(r["run_dir"]), "metrics.json")
        with open(metrics_path) as f:
            metrics = json.load(f)

        row = {
            "Model": fam,
            "fac_seed": fac_seed,
            "opr_seed": opr_seed,
            "Test ρ Macro": metrics["test"]["rho_macro"],
            "Test MAE Macro": metrics["test"]["mae_macro"],
        }

        for short, full in TARGETS.items():
            row[f"{short} ρ"] = metrics["test"]["per_target"][full]["Spearman"]
            row[f"{short} MAE"] = metrics["test"]["per_target"][full]["MAE"]

        per_dim_rows.append(row)

all_df = pd.DataFrame(rows)
per_df = pd.DataFrame(per_dim_rows)

all_path = os.path.join(OUT_DIR, "bert_polsent_predfa_ST_all_best_by_family.csv")
per_seed_path = os.path.join(OUT_DIR, "bert_polsent_predfa_ST_per_seed_per_dimension.csv")

all_df.to_csv(all_path, index=False)
per_df.to_csv(per_seed_path, index=False)

def ms(g, col):
    return f"{g[col].mean():.3f} ± {g[col].std(ddof=1):.3f}"

table_rows = []
for fam, g in per_df.groupby("Model"):
    table_rows.append({
        "Model": fam,
        "n_runs": len(g),
        "Test ρ Macro": ms(g, "Test ρ Macro"),
        "Test MAE Macro": ms(g, "Test MAE Macro"),
        "Directness ρ": ms(g, "Directness ρ"),
        "Directness MAE": ms(g, "Directness MAE"),
        "Positive Face ρ": ms(g, "Positive Face Saving ρ"),
        "Positive Face MAE": ms(g, "Positive Face Saving MAE"),
        "Negative Face ρ": ms(g, "Negative Face Saving ρ"),
        "Negative Face MAE": ms(g, "Negative Face Saving MAE"),
    })

table_df = pd.DataFrame(table_rows)

order = {
    "pol_mlp_st": 0,
    "sent_mlp_st": 1,
    "bert_plus_pol_sent_st": 2,
    "bert_plus_pol_sent_predfa_st": 3,
}
table_df["order"] = table_df["Model"].map(order)
table_df = table_df.sort_values("order").drop(columns=["order"])

table_path = os.path.join(OUT_DIR, "bert_polsent_predfa_ST_table_rows_per_dimension.csv")
table_df.to_csv(table_path, index=False)

summary = (
    all_df.groupby("family")
    .agg(
        n_runs=("opr_run_seed", "nunique"),
        val_rho_macro_mean=("val_rho_macro", "mean"),
        val_rho_macro_std=("val_rho_macro", "std"),
        val_mae_macro_mean=("val_mae_macro", "mean"),
        val_mae_macro_std=("val_mae_macro", "std"),
        test_rho_macro_mean=("test_rho_macro", "mean"),
        test_rho_macro_std=("test_rho_macro", "std"),
        test_mae_macro_mean=("test_mae_macro", "mean"),
        test_mae_macro_std=("test_mae_macro", "std"),
    )
    .reset_index()
)

for c in ["val_rho_macro", "val_mae_macro", "test_rho_macro", "test_mae_macro"]:
    summary[f"{c}_mean_std"] = summary.apply(
        lambda r: f"{r[f'{c}_mean']:.3f} ± {r[f'{c}_std']:.3f}", axis=1
    )

summary["order"] = summary["family"].map(order)
summary = summary.sort_values("order").drop(columns=["order"])

summary_path = os.path.join(OUT_DIR, "bert_polsent_predfa_ST_summary_mean_std.csv")
summary.to_csv(summary_path, index=False)

print("\n=== BERT POL/SENT/PREDFA ST MACRO SUMMARY ===")
print(summary[[
    "family", "n_runs",
    "val_rho_macro_mean_std", "val_mae_macro_mean_std",
    "test_rho_macro_mean_std", "test_mae_macro_mean_std"
]].to_string(index=False))

print("\n=== BERT POL/SENT/PREDFA ST TABLE ROWS ===")
print(table_df.to_string(index=False))

print("\n[DONE] wrote:")
print(all_path)
print(per_seed_path)
print(table_path)
print(summary_path)
PY
-----------------------------
Classical-features comparison for OPR.
Adds politeness lexicon (Pol) and VADER sentiment (Sent) features to
the OPR pipeline so we can compare FA-based features against classical
politeness/sentiment features.

Pol features: lexicon-count based, inspired by Danescu-Niculescu-Mizil
et al. (2013). 15 dimensions: 11 lexicon counts + 3 pronoun-group
counts + 1 imperative-verb-count. Citation: lexicon inspired by but
not identical to the original Stanford Politeness API features.

Sent features: VADER — 4 dimensions (neg, neu, pos, compound).

Extraction is DOC-LEVEL: features computed once over the whole
text_email, no per-sentence aggregation. Pol+Sent = 19 dims total.

This script runs ONLY the four new families (saves time — Text and
Text+PredFA are already in your existing PredFA runs and identical
by seed/config):

  pol_mlp_{mt,st}                       — Pol features only (15-dim MLP)
  sent_mlp_{mt,st}                      — Sent features only (4-dim MLP)
  bert_plus_pol_sent_{mt,st}            — BERT + Pol + Sent (text + classical)
  bert_plus_pol_sent_predfa_{mt,st}     — BERT + Pol + Sent + PredFA (full)

The last family needs FAC inference, so ASL_MODEL_DIR + FAC_RUN_SEED
env vars must be set (same as your existing PredFA script).

If you also want sent_plus_pol_mlp (combined classical MLP) or other
combinations, set RUN_POL_PLUS_SENT_MLP=1 etc.
"""

# ============================================================
# USER PATHS
# ============================================================
DOC_CSV = "data/corpus/email_text_gold_three_dimensions_politeness_score_with_seed_correct.csv"
SENT_CSV = "data/corpus/sentences_with_golden_face_act.csv"

import os

ASL_MODEL_DIR = os.environ.get(
    "ASL_MODEL_DIR",
    "outputs/bert_multi_label_faceacts_sentence_only_asl_gridsearch_microf1/grid_trials/trial_025_lr3e-05_ep3_bs8_do0.1/checkpoint-555"
)
ASL_TOK_NAME = "bert-base-uncased"

OUT_DIR = os.environ.get(
    "OUT_DIR",
    "outputs/opr_sentence_only_bert_polsent_predfa_gridsearch"
)

# ============================================================
# IMPORTS
# ============================================================
import ast, re, glob, json, math, random, itertools
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

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VADER
    VADER_AVAILABLE = True
except ImportError:
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer as VADER
        VADER_AVAILABLE = True
    except ImportError:
        VADER_AVAILABLE = False
        print("[WARN] VADER not available. Install: pip install vaderSentiment")

# ============================================================
# GLOBAL CONFIG
# ============================================================
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPLIT_RANDOM_STATE = int(os.environ.get("SPLIT_RANDOM_STATE", "42"))
OPR_RUN_SEED = int(os.environ.get("OPR_RUN_SEED", "42"))
RANDOM_STATE = OPR_RUN_SEED

DOC_ID_COL = "email_id"
DOC_TEXT_COL = "text_email"
SEED_COL = "seed"
TARGETS = [
    "Directness_vs_Indirectness__GOLD",
    "Structural_Politeness_and_Politeness_Markers__GOLD",
    "Tone_and_Overall_Consideration__GOLD",
]

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
RUN_POL_MLP            = os.environ.get("RUN_POL_MLP", "1") == "1"
RUN_SENT_MLP           = os.environ.get("RUN_SENT_MLP", "1") == "1"
RUN_BERT_PLUS_POL_SENT = os.environ.get("RUN_BERT_PLUS_POL_SENT", "1") == "1"
RUN_BERT_PLUS_POL_SENT_PREDFA = os.environ.get("RUN_BERT_PLUS_POL_SENT_PREDFA", "1") == "1"

# Optional extras (off by default — uncomment in shell if desired)
RUN_POL_PLUS_SENT_MLP  = os.environ.get("RUN_POL_PLUS_SENT_MLP", "0") == "1"

DO_MULTI_TASK = os.environ.get("DO_MULTI_TASK", "1") == "1"
DO_SINGLE_TASK = os.environ.get("DO_SINGLE_TASK", "1") == "1"

# Toggle: divide counts by doc token_count
NORMALIZE_POL_BY_TOKENS = os.environ.get("NORMALIZE_POL_BY_TOKENS", "0") == "1"

GRID_PREDFA_MODES = ["hard"]
FAC_TAU_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2).tolist()

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
# POLITENESS LEXICON (inspired by Danescu et al. 2013)
# ============================================================
LEXICONS = {
    "greetings": ["dear", "hi", "hello", "good morning", "good afternoon", "good evening"],
    "closings": ["regards", "best", "best regards", "kind regards", "sincerely",
                 "warm regards", "thanks", "thank you", "yours sincerely", "yours truly"],
    "thanks": ["thank", "thanks", "thank you", "appreciate", "grateful"],
    "apologies": ["sorry", "apologize", "apologies", "apologise"],
    "deference": ["please", "kindly", "if possible", "at your convenience", "would you mind"],
    "hedges": ["perhaps", "maybe", "possibly", "somewhat", "a bit", "i think", "i believe"],
    "soft_modals": ["could", "would", "might", "may", "can"],
    "indirect_patterns": ["could you", "would you", "would it be possible", "i was wondering",
                          "if you could", "at your convenience", "would you mind", "could you please"],
    "obligation": ["must", "need", "have to", "required", "should"],
    "urgency": ["urgent", "immediately", "asap", "crucial", "action"],
    "directive_patterns": ["please send", "do this", "let me know by", "i need you to", "send me",
                           "reply by", "provide", "submit", "review", "complete", "confirm", "share", "update me"],
}

PRONOUNS = {
    "second_person": ["you", "your", "yours"],
    "first_person_singular": ["i", "me", "my", "mine"],
    "first_person_plural": ["we", "us", "our", "ours"],
}

IMPERATIVE_START_VERBS = ["send", "share", "provide", "review", "submit", "complete", "reply",
                          "confirm", "check", "call", "update", "let", "do", "advise", "ensure",
                          "remember", "see", "consider", "finish"]

POL_FEATURE_ORDER = (
    [f"pol_{k}" for k in LEXICONS.keys()] +
    [f"pol_pron_{k}" for k in PRONOUNS.keys()] +
    ["pol_imperative_verb_count"]
)
SENT_FEATURE_ORDER = ["sent_neg", "sent_neu", "sent_pos", "sent_compound"]


def count_phrase(text_lower: str, phrase: str) -> int:
    """Count occurrences of a phrase (case-insensitive, word-boundary-aware for single words)."""
    if " " in phrase:
        return text_lower.count(phrase)
    # single word: use word boundaries to avoid 'i' matching inside 'this'
    return len(re.findall(rf"\b{re.escape(phrase)}\b", text_lower))


def compute_pol_features(text: str) -> np.ndarray:
    """Doc-level politeness feature vector."""
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    text_lower = text.lower()
    n_tokens = max(1, len(re.findall(r"\b\w+\b", text_lower)))
    feats = []
    for lex_name, words in LEXICONS.items():
        c = sum(count_phrase(text_lower, w) for w in words)
        feats.append(float(c))
    for pron_name, words in PRONOUNS.items():
        c = sum(count_phrase(text_lower, w) for w in words)
        feats.append(float(c))
    # imperative verbs (count anywhere in doc — see docstring for caveat)
    c = sum(count_phrase(text_lower, v) for v in IMPERATIVE_START_VERBS)
    feats.append(float(c))
    arr = np.array(feats, dtype=np.float32)
    if NORMALIZE_POL_BY_TOKENS:
        arr = arr / n_tokens
    return arr


_VADER_ANALYZER = None
def _get_vader():
    global _VADER_ANALYZER
    if _VADER_ANALYZER is None:
        if not VADER_AVAILABLE:
            raise RuntimeError("VADER not installed. pip install vaderSentiment")
        _VADER_ANALYZER = VADER()
    return _VADER_ANALYZER


def compute_sent_features(text: str) -> np.ndarray:
    """Doc-level VADER sentiment vector: [neg, neu, pos, compound]."""
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    s = _get_vader().polarity_scores(text)
    return np.array([s["neg"], s["neu"], s["pos"], s["compound"]], dtype=np.float32)


def compute_polsent_matrix(df_doc: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return (Pol_matrix N×P, Sent_matrix N×S) for the doc dataframe rows."""
    Pol = np.stack([compute_pol_features(t) for t in df_doc[DOC_TEXT_COL].tolist()], axis=0)
    Sent = np.stack([compute_sent_features(t) for t in df_doc[DOC_TEXT_COL].tolist()], axis=0)
    return Pol.astype(np.float32), Sent.astype(np.float32)


# ============================================================
# UTILS
# ============================================================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def ensure_dir(d): os.makedirs(d, exist_ok=True)


def make_dl_generator(seed):
    g = torch.Generator(); g.manual_seed(seed); return g


def _decategorize_series(s):
    if isinstance(s.dtype, CategoricalDtype):
        s = s.astype(object)
    return s


def sanitize_dataframe(df):
    df = df.copy()
    for col in df.columns:
        if isinstance(df[col].dtype, CategoricalDtype):
            df[col] = df[col].astype(object)
    if DOC_ID_COL in df.columns: df[DOC_ID_COL] = df[DOC_ID_COL].astype(str)
    if SENT_TEXT_COL in df.columns: df[SENT_TEXT_COL] = df[SENT_TEXT_COL].astype(str)
    if FA_COL in df.columns: df[FA_COL] = df[FA_COL].astype(str)
    if SEED_COL in df.columns: df[SEED_COL] = pd.to_numeric(df[SEED_COL], errors="coerce")
    if SENT_IDX_COL in df.columns: df[SENT_IDX_COL] = pd.to_numeric(df[SENT_IDX_COL], errors="coerce")
    return df.reset_index(drop=True)


def grouped_split_70_15_15_index(groups, random_state=42):
    idx = np.arange(len(groups))
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=random_state)
    tr, tmp = next(gss1.split(idx, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=random_state)
    va_rel, te_rel = next(gss2.split(tmp, groups=groups[tmp]))
    return tr, tmp[va_rel], tmp[te_rel]


def iter_grid(grid):
    keys = list(grid.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]


def parse_gold_faceacts(x):
    if pd.isna(x): return []
    if isinstance(x, list): vals = x
    else:
        s = str(x).strip()
        if s in {"", "nan", "None"}: return []
        try:
            obj = ast.literal_eval(s)
            vals = obj if isinstance(obj, list) else [str(obj)]
        except Exception:
            if ";" in s: vals = s.split(";")
            elif "," in s: vals = s.split(",")
            else: vals = [s]
    clean = []
    for v in vals:
        v = str(v).strip().strip("'\"")
        if v in FA_LABS: clean.append(v)
    return clean


def gold_multihot(df):
    Y = np.zeros((len(df), len(FA_LABS)), dtype=np.int32)
    lab2i = {l: i for i, l in enumerate(FA_LABS)}
    for i, x in enumerate(df[FA_COL].tolist()):
        for lab in parse_gold_faceacts(x):
            Y[i, lab2i[lab]] = 1
    return Y


# ============================================================
# FAC SENTENCE-ONLY INFERENCE (unchanged from base PredFA script)
# ============================================================
def _has_model_files(d):
    return any(os.path.isfile(os.path.join(d, fn)) for fn in
               ["pytorch_model.bin", "model.safetensors", "tf_model.h5", "model.ckpt.index", "flax_model.msgpack"])


def resolve_hf_model_dir(parent_dir):
    if _has_model_files(parent_dir): return parent_dir
    state_path = os.path.join(parent_dir, "trainer_state.json")
    if os.path.isfile(state_path):
        try:
            with open(state_path) as f: state = json.load(f)
            best_ckpt = state.get("best_model_checkpoint")
            if best_ckpt and _has_model_files(best_ckpt): return best_ckpt
        except Exception: pass
    ckpts = sorted(glob.glob(os.path.join(parent_dir, "checkpoint-*")), key=os.path.getmtime, reverse=True)
    for ck in ckpts:
        if _has_model_files(ck): return ck
    raise OSError(f"No model files in {parent_dir}")


class SentDataset(Dataset):
    def __init__(self, df, text_col, tokenizer, max_length=128):
        df = df.copy()
        df[DOC_ID_COL] = df[DOC_ID_COL].astype(str)
        df[SENT_IDX_COL] = pd.to_numeric(df[SENT_IDX_COL], errors="coerce").fillna(0).astype(int)
        enc = tokenizer(df[text_col].astype(str).tolist(),
                        padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        self.input_ids = enc["input_ids"]; self.attn = enc["attention_mask"]
        self.doc_ids = df[DOC_ID_COL].tolist(); self.sent_idx = df[SENT_IDX_COL].tolist()
    def __len__(self): return self.input_ids.size(0)
    def __getitem__(self, i):
        return {"input_ids": self.input_ids[i], "attention_mask": self.attn[i],
                "doc_id": self.doc_ids[i], "sent_idx": self.sent_idx[i]}


def predict_fa_probs(df_split, model_dir, tok_name, batch=64, max_len=128):
    if len(df_split) == 0:
        return pd.DataFrame(columns=[DOC_ID_COL, SENT_IDX_COL] + [f"prob_{l}" for l in FA_LABS])
    try:
        resolved = resolve_hf_model_dir(model_dir)
    except OSError as e:
        raise OSError(f"[FAC] Could not resolve checkpoint under {model_dir}: {e}") from e
    print(f"[FAC] Loading sentence-only checkpoint: {resolved}")
    tok = AutoTokenizer.from_pretrained(tok_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(resolved).to(DEVICE)
    mdl.eval()
    ds = SentDataset(df_split, SENT_TEXT_COL, tok, max_length=max_len)
    dl = DataLoader(ds, batch_size=batch, shuffle=False)
    doc_ids, sids, allP = [], [], []
    with torch.no_grad():
        for b in dl:
            logits = mdl(input_ids=b["input_ids"].to(DEVICE),
                         attention_mask=b["attention_mask"].to(DEVICE)).logits
            P = torch.sigmoid(logits).cpu().numpy()
            allP.append(P); doc_ids.extend(b["doc_id"]); sids.extend(b["sent_idx"])
    P = np.vstack(allP)
    out = {DOC_ID_COL: list(map(str, doc_ids)), SENT_IDX_COL: list(map(int, sids))}
    for j, l in enumerate(FA_LABS): out[f"prob_{l}"] = P[:, j]
    dfp = pd.DataFrame(out)
    dfp[DOC_ID_COL] = dfp[DOC_ID_COL].astype(str)
    dfp[SENT_IDX_COL] = pd.to_numeric(dfp[SENT_IDX_COL], errors="coerce").fillna(0).astype(int)
    return dfp


def tune_fac_tau_on_validation(val_pred, val_sent):
    val_pred = val_pred.sort_values([DOC_ID_COL, SENT_IDX_COL], kind="mergesort").reset_index(drop=True)
    val_sent = val_sent.copy()
    val_sent[DOC_ID_COL] = val_sent[DOC_ID_COL].astype(str)
    val_sent[SENT_IDX_COL] = pd.to_numeric(val_sent[SENT_IDX_COL], errors="coerce").fillna(0).astype(int)
    val_sent = val_sent.sort_values([DOC_ID_COL, SENT_IDX_COL], kind="mergesort").reset_index(drop=True)
    assert len(val_pred) == len(val_sent)
    assert val_pred[DOC_ID_COL].tolist() == val_sent[DOC_ID_COL].tolist()
    assert val_pred[SENT_IDX_COL].tolist() == val_sent[SENT_IDX_COL].tolist()
    y_true = gold_multihot(val_sent)
    probs = val_pred[[f"prob_{l}" for l in FA_LABS]].to_numpy(dtype=np.float32)
    rows, best = [], None
    for tau in FAC_TAU_GRID:
        y_hat = (probs >= tau).astype(np.int32)
        row = {"tau": float(tau),
               "micro_f1": float(f1_score(y_true, y_hat, average="micro", zero_division=0)),
               "micro_precision": float(precision_score(y_true, y_hat, average="micro", zero_division=0)),
               "micro_recall": float(recall_score(y_true, y_hat, average="micro", zero_division=0))}
        rows.append(row)
        if best is None or row["micro_f1"] > best["micro_f1"]: best = row
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "fac_val_tau_search.csv"), index=False)
    with open(os.path.join(OUT_DIR, "fac_best_tau.json"), "w") as f:
        json.dump(best, f, indent=2)
    print(f"[FAC] Best val tau = {best['tau']:.2f} | micro-F1 = {best['micro_f1']:.4f}")
    return best


# ============================================================
# PREDFA AGGREGATION (unchanged)
# ============================================================
def aggregate_predfa_hspt13(dfp, mode="hard", tau=None):
    if mode not in {"soft", "hard"}: raise ValueError("mode must be soft/hard")
    if mode == "hard" and tau is None: raise ValueError("tau required for hard mode")
    need = [DOC_ID_COL, SENT_IDX_COL] + [f"prob_{l}" for l in FA_LABS]
    for c in need:
        if c not in dfp.columns: raise ValueError(f"Missing column: {c}")
    dfp = dfp.copy()
    dfp[DOC_ID_COL] = _decategorize_series(dfp[DOC_ID_COL]).astype(str)
    dfp[SENT_IDX_COL] = pd.to_numeric(_decategorize_series(dfp[SENT_IDX_COL]), errors="coerce").fillna(0).astype(int)
    for l in FA_LABS:
        col = f"prob_{l}"
        vals = pd.to_numeric(_decategorize_series(dfp[col]), errors="coerce").fillna(0.0).astype(float)
        if mode == "hard": vals = (vals >= float(tau)).astype(float)
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
        if c not in df_out.columns: df_out[c] = 0.0
    return df_out[keep_cols]


def align_predfa_to_split(F_pred, df_doc, idx, split_name=""):
    want = df_doc.loc[idx, DOC_ID_COL].astype(str).tolist()
    F = F_pred.copy()
    F[DOC_ID_COL] = F[DOC_ID_COL].astype(str)
    F = F.drop_duplicates(subset=[DOC_ID_COL], keep="first").set_index(DOC_ID_COL)
    missing = [d for d in want if d not in set(F.index)]
    if missing:
        print(f"[ALIGN][{split_name}] WARNING: {len(missing)}/{len(want)} doc_ids missing from PredFA.")
        try:
            with open(os.path.join(OUT_DIR, f"predfa_missing_ids_{split_name}.json"), "w") as f:
                json.dump({"split": split_name, "n_missing": len(missing), "missing_doc_ids": missing}, f, indent=2)
        except Exception: pass
    F_a = F.reindex(want)
    for c in SUM13_ORDER:
        if c not in F_a.columns: F_a[c] = 0.0
    F_a[SUM13_ORDER] = F_a[SUM13_ORDER].fillna(0.0)
    return F_a[SUM13_ORDER].to_numpy(dtype=np.float32)


def zscore_fit_transform(Xtr, Xva, Xte):
    mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0); sd[sd == 0] = 1.0
    return (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd, mu, sd


# ============================================================
# DOC DATASETS / MODELS
# ============================================================
class DocBertDataset(Dataset):
    def __init__(self, df, text_col, targets, tokenizer):
        enc = tokenizer(df[text_col].astype(str).tolist(),
                        padding=True, truncation=True, max_length=BERT_MAXLEN, return_tensors="pt")
        self.ids = enc["input_ids"]; self.attn = enc["attention_mask"]
        self.y = torch.tensor(df[targets].to_numpy(np.float32))
    def __len__(self): return self.ids.size(0)
    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "attention_mask": self.attn[i], "targets": self.y[i]}


class DocBertFusionDataset(Dataset):
    """Generic fusion dataset; takes a single concatenated feature matrix."""
    def __init__(self, df, text_col, targets, tokenizer, fusion_feats: np.ndarray):
        assert len(df) == len(fusion_feats), "fusion rows must align with doc rows."
        enc = tokenizer(df[text_col].astype(str).tolist(),
                        padding=True, truncation=True, max_length=BERT_MAXLEN, return_tensors="pt")
        self.ids = enc["input_ids"]; self.attn = enc["attention_mask"]
        self.y = torch.tensor(df[targets].to_numpy(np.float32))
        self.fa = torch.tensor(fusion_feats, dtype=torch.float32)
    def __len__(self): return self.ids.size(0)
    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "attention_mask": self.attn[i],
                "targets": self.y[i], "predfa": self.fa[i]}


class DocFADataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32); self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self): return self.X.size(0)
    def __getitem__(self, i): return {"x": self.X[i], "targets": self.Y[i]}


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

    def forward(self, input_ids, attention_mask, predfa=None):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0, :]
        if self.fusion_dim > 0:
            if predfa is None: raise ValueError("fusion features required when fusion_dim > 0")
            z = torch.cat([cls, predfa], dim=-1)
        else:
            z = cls
        z = self.dropout(z); h = self.hidden(z)
        return torch.cat([head(h) for head in self.heads], dim=-1)


class FAMLP(nn.Module):
    def __init__(self, in_dim, out_dim=3, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x): return self.net(x)


# ============================================================
# METRICS / TRAIN HELPERS (unchanged)
# ============================================================
def mae_spearman(y_true, y_pred, target_names):
    out = {"per_target": {}}
    maes, rhos = [], []
    for i, name in enumerate(target_names):
        yt, yp = y_true[:, i], y_pred[:, i]
        mae = float(np.mean(np.abs(yt - yp)))
        rho, _ = spearmanr(yt, yp)
        rho = float(rho) if rho is not None and not np.isnan(rho) else 0.0
        maes.append(mae); rhos.append(rho)
        out["per_target"][name] = {"MAE": mae, "Spearman": rho}
    out["mae_macro"] = float(np.mean(maes))
    out["rho_macro"] = float(np.mean(rhos))
    return out


def predict_bert(model, dl):
    model.eval(); Ys, Ps = [], []
    with torch.no_grad():
        for b in dl:
            pf = b.get("predfa"); pf = pf.to(DEVICE) if pf is not None else None
            pr = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), pf)
            Ys.append(b["targets"].numpy()); Ps.append(pr.cpu().numpy())
    return np.concatenate(Ys, 0), np.concatenate(Ps, 0)


def predict_mlp(model, dl):
    model.eval(); Ys, Ps = [], []
    with torch.no_grad():
        for b in dl:
            pr = model(b["x"].to(DEVICE))
            Ys.append(b["targets"].numpy()); Ps.append(pr.cpu().numpy())
    return np.concatenate(Ys, 0), np.concatenate(Ps, 0)


def save_run(outdir, cfg, Yva, Pva, Yte, Pte):
    ensure_dir(outdir)
    metrics = {"config": cfg, "val": mae_spearman(Yva, Pva, TARGETS),
               "test": mae_spearman(Yte, Pte, TARGETS), "targets": TARGETS}
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(outdir, "val_y.npy"), Yva)
    np.save(os.path.join(outdir, "val_pred.npy"), Pva)
    np.save(os.path.join(outdir, "test_y.npy"), Yte)
    np.save(os.path.join(outdir, "test_pred.npy"), Pte)
    return metrics


def train_bert_once(df_doc, tr_idx, va_idx, te_idx, cfg, fusion_feats=None, multi_task=True):
    set_seed(RANDOM_STATE)
    tk = AutoTokenizer.from_pretrained(DOC_BERT_NAME, use_fast=True)
    df_tr = df_doc.loc[tr_idx].reset_index(drop=True)
    df_va = df_doc.loc[va_idx].reset_index(drop=True)
    df_te = df_doc.loc[te_idx].reset_index(drop=True)

    if fusion_feats is None:
        ds_tr = DocBertDataset(df_tr, DOC_TEXT_COL, TARGETS, tk)
        ds_va = DocBertDataset(df_va, DOC_TEXT_COL, TARGETS, tk)
        ds_te = DocBertDataset(df_te, DOC_TEXT_COL, TARGETS, tk)
        fusion_dim = 0
    else:
        Xtr, Xva, Xte = fusion_feats
        ds_tr = DocBertFusionDataset(df_tr, DOC_TEXT_COL, TARGETS, tk, Xtr)
        ds_va = DocBertFusionDataset(df_va, DOC_TEXT_COL, TARGETS, tk, Xva)
        ds_te = DocBertFusionDataset(df_te, DOC_TEXT_COL, TARGETS, tk, Xte)
        fusion_dim = Xtr.shape[1]

    g = make_dl_generator(RANDOM_STATE)
    dl_tr = DataLoader(ds_tr, batch_size=cfg["batch_size"], shuffle=True, generator=g)
    dl_va = DataLoader(ds_va, batch_size=cfg["batch_size"], shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=cfg["batch_size"], shuffle=False)

    if multi_task:
        model = BertDocRegressor(encoder_name=DOC_BERT_NAME, n_targets=3, fusion_dim=fusion_dim,
                                 hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"]).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        steps_total = max(1, len(dl_tr) * cfg["epochs"])
        sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps_total), steps_total)
        loss_fn = nn.SmoothL1Loss()
        best_score, best_state = -1e9, None
        for _ in range(cfg["epochs"]):
            model.train()
            for b in dl_tr:
                opt.zero_grad()
                pf = b.get("predfa"); pf = pf.to(DEVICE) if pf is not None else None
                pr = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), pf)
                loss = loss_fn(pr, b["targets"].to(DEVICE))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step()
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
    Pva_all = np.zeros_like(Yva_all); Pte_all = np.zeros_like(Yte_all)

    for t_i in range(3):
        set_seed(RANDOM_STATE + t_i)
        g = make_dl_generator(RANDOM_STATE + t_i)
        dl_tr = DataLoader(ds_tr, batch_size=cfg["batch_size"], shuffle=True, generator=g)
        model = BertDocRegressor(encoder_name=DOC_BERT_NAME, n_targets=1, fusion_dim=fusion_dim,
                                 hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"]).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        steps_total = max(1, len(dl_tr) * cfg["epochs"])
        sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps_total), steps_total)
        loss_fn = nn.SmoothL1Loss()
        best_score, best_state = -1e9, None
        for _ in range(cfg["epochs"]):
            model.train()
            for b in dl_tr:
                opt.zero_grad()
                pf = b.get("predfa"); pf = pf.to(DEVICE) if pf is not None else None
                y = b["targets"][:, t_i].unsqueeze(1).to(DEVICE)
                pr = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), pf)
                loss = loss_fn(pr, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step()
            model.eval(); yv, pv = [], []
            with torch.no_grad():
                for b in dl_va:
                    pf = b.get("predfa"); pf = pf.to(DEVICE) if pf is not None else None
                    pr = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), pf)
                    yv.append(b["targets"][:, t_i].numpy())
                    pv.append(pr.cpu().numpy().squeeze(1))
            yv = np.concatenate(yv, 0); pv = np.concatenate(pv, 0)
            rho, _ = spearmanr(yv, pv)
            score = float(rho) if rho is not None and not np.isnan(rho) else 0.0
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state); model.eval()
        for dl, store in [(dl_va, Pva_all), (dl_te, Pte_all)]:
            preds = []
            with torch.no_grad():
                for b in dl:
                    pf = b.get("predfa"); pf = pf.to(DEVICE) if pf is not None else None
                    pr = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), pf)
                    preds.append(pr.cpu().numpy().squeeze(1))
            store[:, t_i] = np.concatenate(preds, 0)
    return Yva_all, Pva_all, Yte_all, Pte_all


def train_mlp_once(Xtr, Xva, Xte, Ytr, Yva, Yte, cfg, multi_task=True):
    set_seed(RANDOM_STATE)
    g = make_dl_generator(RANDOM_STATE)
    dl_tr = DataLoader(DocFADataset(Xtr, Ytr), batch_size=cfg["batch_size"], shuffle=True, generator=g)
    dl_va = DataLoader(DocFADataset(Xva, Yva), batch_size=cfg["batch_size"], shuffle=False)
    dl_te = DataLoader(DocFADataset(Xte, Yte), batch_size=cfg["batch_size"], shuffle=False)

    if multi_task:
        model = FAMLP(Xtr.shape[1], out_dim=3, hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"]).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        loss_fn = nn.SmoothL1Loss()
        best_score, best_state = -1e9, None
        for _ in range(cfg["epochs"]):
            model.train()
            for b in dl_tr:
                opt.zero_grad(); pr = model(b["x"].to(DEVICE))
                loss = loss_fn(pr, b["targets"].to(DEVICE))
                loss.backward(); opt.step()
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

    Pva = np.zeros_like(Yva); Pte = np.zeros_like(Yte)
    for t_i in range(3):
        set_seed(RANDOM_STATE + t_i)
        g = make_dl_generator(RANDOM_STATE + t_i)
        dl_tr = DataLoader(DocFADataset(Xtr, Ytr), batch_size=cfg["batch_size"], shuffle=True, generator=g)
        model = FAMLP(Xtr.shape[1], out_dim=1, hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"]).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        loss_fn = nn.SmoothL1Loss()
        best_score, best_state = -1e9, None
        for _ in range(cfg["epochs"]):
            model.train()
            for b in dl_tr:
                opt.zero_grad(); y = b["targets"][:, t_i].unsqueeze(1).to(DEVICE)
                pr = model(b["x"].to(DEVICE))
                loss = loss_fn(pr, y); loss.backward(); opt.step()
            model.eval(); yv, pv = [], []
            with torch.no_grad():
                for b in dl_va:
                    pr = model(b["x"].to(DEVICE)).cpu().numpy().squeeze(1)
                    yv.append(b["targets"][:, t_i].numpy()); pv.append(pr)
            yv = np.concatenate(yv, 0); pv = np.concatenate(pv, 0)
            rho, _ = spearmanr(yv, pv)
            score = float(rho) if rho is not None and not np.isnan(rho) else 0.0
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state); model.eval()
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
def _run_metadata():
    return {
        "opr_run_seed": int(OPR_RUN_SEED),
        "split_random_state": int(SPLIT_RANDOM_STATE),
        "fac_run_seed": os.environ.get("FAC_RUN_SEED", ""),
        "fac_checkpoint": ASL_MODEL_DIR,
        "normalize_pol_by_tokens": NORMALIZE_POL_BY_TOKENS,
    }


def _row_with_metadata(row):
    return {**row, **_run_metadata()}


def main():
    set_seed(OPR_RUN_SEED); ensure_dir(OUT_DIR)
    need_predfa = RUN_BERT_PLUS_POL_SENT_PREDFA

    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Output dir: {OUT_DIR}")
    print(f"[INFO] Split random state: {SPLIT_RANDOM_STATE}")
    print(f"[INFO] OPR run seed: {OPR_RUN_SEED}")
    print(f"[INFO] FAC run seed: {os.environ.get('FAC_RUN_SEED','')}")
    print(f"[INFO] FAC checkpoint: {ASL_MODEL_DIR if need_predfa else '[disabled]'}")
    print(f"[INFO] Normalize Pol by tokens: {NORMALIZE_POL_BY_TOKENS}")
    print(f"[INFO] Families: pol_mlp={RUN_POL_MLP} sent_mlp={RUN_SENT_MLP} "
          f"bert+pol+sent={RUN_BERT_PLUS_POL_SENT} bert+pol+sent+predfa={RUN_BERT_PLUS_POL_SENT_PREDFA}")

    df_doc = sanitize_dataframe(pd.read_csv(DOC_CSV))
    df_sent = sanitize_dataframe(pd.read_csv(SENT_CSV))

    if SEED_COL not in df_doc.columns:
        seed_map = (df_sent[[DOC_ID_COL, SEED_COL]].dropna()
                    .drop_duplicates(subset=[DOC_ID_COL, SEED_COL])
                    .groupby(DOC_ID_COL)[SEED_COL]
                    .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0])
                    .reset_index())
        df_doc = sanitize_dataframe(df_doc.merge(seed_map, on=DOC_ID_COL, how="left"))

    for t in TARGETS:
        df_doc[t] = pd.to_numeric(df_doc[t], errors="coerce")
    pre_n = len(df_doc)
    df_doc = df_doc.dropna(subset=TARGETS).reset_index(drop=True)
    if len(df_doc) < pre_n:
        print(f"[INFO] dropped {pre_n - len(df_doc)} rows.")

    df_doc[DOC_ID_COL] = df_doc[DOC_ID_COL].astype(str)
    df_sent[DOC_ID_COL] = df_sent[DOC_ID_COL].astype(str)
    dup = df_doc[DOC_ID_COL].duplicated().sum()
    if dup:
        print(f"[WARN] {dup} dups; keeping first.")
        df_doc = df_doc.drop_duplicates(subset=[DOC_ID_COL], keep="first").reset_index(drop=True)
    df_doc = df_doc.sort_values(DOC_ID_COL, kind="mergesort").reset_index(drop=True)
    if df_doc[SEED_COL].isna().any():
        df_doc = df_doc.dropna(subset=[SEED_COL]).reset_index(drop=True)
    df_doc[SEED_COL] = df_doc[SEED_COL].astype(int)
    if len(df_doc) == 0: raise RuntimeError("df_doc empty.")

    groups = df_doc[SEED_COL].to_numpy().astype(int)
    tr_idx, va_idx, te_idx = grouped_split_70_15_15_index(groups, random_state=SPLIT_RANDOM_STATE)
    seeds_tr, seeds_va, seeds_te = set(groups[tr_idx]), set(groups[va_idx]), set(groups[te_idx])
    assert not (seeds_tr & seeds_va) and not (seeds_tr & seeds_te) and not (seeds_va & seeds_te)

    split_info = {
        "split_random_state": SPLIT_RANDOM_STATE, "opr_run_seed": OPR_RUN_SEED,
        "fac_run_seed": os.environ.get("FAC_RUN_SEED", ""),
        "fac_checkpoint": ASL_MODEL_DIR if need_predfa else "",
        "n_docs_train": int(len(tr_idx)), "n_docs_val": int(len(va_idx)), "n_docs_test": int(len(te_idx)),
        "train_seeds": sorted(map(int, seeds_tr)),
        "val_seeds": sorted(map(int, seeds_va)),
        "test_seeds": sorted(map(int, seeds_te)),
        "pol_features": POL_FEATURE_ORDER,
        "sent_features": SENT_FEATURE_ORDER,
        "normalize_pol_by_tokens": NORMALIZE_POL_BY_TOKENS,
    }
    with open(os.path.join(OUT_DIR, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"[INFO] Split: train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")

    Y = df_doc[TARGETS].to_numpy(np.float32)
    Y_tr, Y_va, Y_te = Y[tr_idx], Y[va_idx], Y[te_idx]

    # ----------------------------------------------------
    # Compute Pol and Sent features at the DOC level
    # ----------------------------------------------------
    print("\n[FEATS] Computing doc-level Pol (lexicon) features...")
    Pol_all, Sent_all = compute_polsent_matrix(df_doc)
    print(f"[FEATS] Pol shape: {Pol_all.shape}, Sent shape: {Sent_all.shape}")

    Pol_tr, Pol_va, Pol_te = Pol_all[tr_idx], Pol_all[va_idx], Pol_all[te_idx]
    Sent_tr, Sent_va, Sent_te = Sent_all[tr_idx], Sent_all[va_idx], Sent_all[te_idx]

    Pol_tr_z, Pol_va_z, Pol_te_z, pol_mu, pol_sd = zscore_fit_transform(Pol_tr, Pol_va, Pol_te)
    Sent_tr_z, Sent_va_z, Sent_te_z, sent_mu, sent_sd = zscore_fit_transform(Sent_tr, Sent_va, Sent_te)

    # Save feature manifests
    feat_dir = os.path.join(OUT_DIR, "feature_manifests")
    ensure_dir(feat_dir)
    pd.DataFrame({"feature": POL_FEATURE_ORDER, "mu": pol_mu.tolist(), "sd": pol_sd.tolist()}).to_csv(
        os.path.join(feat_dir, "pol_zscore.csv"), index=False)
    pd.DataFrame({"feature": SENT_FEATURE_ORDER, "mu": sent_mu.tolist(), "sd": sent_sd.tolist()}).to_csv(
        os.path.join(feat_dir, "sent_zscore.csv"), index=False)
    # Also save raw feature matrices for downstream analysis
    np.save(os.path.join(feat_dir, "pol_train.npy"), Pol_tr)
    np.save(os.path.join(feat_dir, "pol_val.npy"),   Pol_va)
    np.save(os.path.join(feat_dir, "pol_test.npy"),  Pol_te)
    np.save(os.path.join(feat_dir, "sent_train.npy"), Sent_tr)
    np.save(os.path.join(feat_dir, "sent_val.npy"),   Sent_va)
    np.save(os.path.join(feat_dir, "sent_test.npy"),  Sent_te)

    all_rows = []
    best_by_family = {}

    # ============================================================
    # Family 1: Pol-only MLP
    # ============================================================
    if RUN_POL_MLP:
        for multi_task in ([True] if DO_MULTI_TASK else []) + ([False] if DO_SINGLE_TASK else []):
            family = f"pol_mlp_{'mt' if multi_task else 'st'}"
            print(f"\n[GRID] {family}")
            for run_id, cfg in enumerate(iter_grid(MLP_GRID), start=1):
                print(f"[RUN] {family} #{run_id}: {cfg}")
                Yv, Pv, Yt, Pt = train_mlp_once(Pol_tr_z, Pol_va_z, Pol_te_z, Y_tr, Y_va, Y_te, cfg, multi_task=multi_task)
                run_dir = os.path.join(OUT_DIR, family, f"run_{run_id:03d}")
                cfg_save = _row_with_metadata({**cfg, "family": family, "feat_set": "pol", "feat_dim": Pol_tr_z.shape[1]})
                metrics = save_run(run_dir, cfg_save, Yv, Pv, Yt, Pt)
                row = _row_with_metadata({**cfg, "family": family, "run_id": run_id, "feat_set": "pol", "feat_dim": Pol_tr_z.shape[1]})
                row.update({"val_mae_macro": metrics["val"]["mae_macro"], "val_rho_macro": metrics["val"]["rho_macro"],
                            "test_mae_macro": metrics["test"]["mae_macro"], "test_rho_macro": metrics["test"]["rho_macro"],
                            "run_dir": run_dir})
                all_rows.append(row)
                if family not in best_by_family or row["val_rho_macro"] > best_by_family[family]["val_rho_macro"]:
                    best_by_family[family] = row

    # ============================================================
    # Family 2: Sent-only MLP
    # ============================================================
    if RUN_SENT_MLP:
        for multi_task in ([True] if DO_MULTI_TASK else []) + ([False] if DO_SINGLE_TASK else []):
            family = f"sent_mlp_{'mt' if multi_task else 'st'}"
            print(f"\n[GRID] {family}")
            for run_id, cfg in enumerate(iter_grid(MLP_GRID), start=1):
                print(f"[RUN] {family} #{run_id}: {cfg}")
                Yv, Pv, Yt, Pt = train_mlp_once(Sent_tr_z, Sent_va_z, Sent_te_z, Y_tr, Y_va, Y_te, cfg, multi_task=multi_task)
                run_dir = os.path.join(OUT_DIR, family, f"run_{run_id:03d}")
                cfg_save = _row_with_metadata({**cfg, "family": family, "feat_set": "sent", "feat_dim": Sent_tr_z.shape[1]})
                metrics = save_run(run_dir, cfg_save, Yv, Pv, Yt, Pt)
                row = _row_with_metadata({**cfg, "family": family, "run_id": run_id, "feat_set": "sent", "feat_dim": Sent_tr_z.shape[1]})
                row.update({"val_mae_macro": metrics["val"]["mae_macro"], "val_rho_macro": metrics["val"]["rho_macro"],
                            "test_mae_macro": metrics["test"]["mae_macro"], "test_rho_macro": metrics["test"]["rho_macro"],
                            "run_dir": run_dir})
                all_rows.append(row)
                if family not in best_by_family or row["val_rho_macro"] > best_by_family[family]["val_rho_macro"]:
                    best_by_family[family] = row

    # Optional: Pol+Sent combined MLP
    if RUN_POL_PLUS_SENT_MLP:
        for multi_task in ([True] if DO_MULTI_TASK else []) + ([False] if DO_SINGLE_TASK else []):
            family = f"pol_plus_sent_mlp_{'mt' if multi_task else 'st'}"
            print(f"\n[GRID] {family}")
            Xtr_ps = np.concatenate([Pol_tr_z, Sent_tr_z], axis=1)
            Xva_ps = np.concatenate([Pol_va_z, Sent_va_z], axis=1)
            Xte_ps = np.concatenate([Pol_te_z, Sent_te_z], axis=1)
            for run_id, cfg in enumerate(iter_grid(MLP_GRID), start=1):
                print(f"[RUN] {family} #{run_id}: {cfg}")
                Yv, Pv, Yt, Pt = train_mlp_once(Xtr_ps, Xva_ps, Xte_ps, Y_tr, Y_va, Y_te, cfg, multi_task=multi_task)
                run_dir = os.path.join(OUT_DIR, family, f"run_{run_id:03d}")
                cfg_save = _row_with_metadata({**cfg, "family": family, "feat_set": "pol+sent", "feat_dim": Xtr_ps.shape[1]})
                metrics = save_run(run_dir, cfg_save, Yv, Pv, Yt, Pt)
                row = _row_with_metadata({**cfg, "family": family, "run_id": run_id, "feat_set": "pol+sent", "feat_dim": Xtr_ps.shape[1]})
                row.update({"val_mae_macro": metrics["val"]["mae_macro"], "val_rho_macro": metrics["val"]["rho_macro"],
                            "test_mae_macro": metrics["test"]["mae_macro"], "test_rho_macro": metrics["test"]["rho_macro"],
                            "run_dir": run_dir})
                all_rows.append(row)
                if family not in best_by_family or row["val_rho_macro"] > best_by_family[family]["val_rho_macro"]:
                    best_by_family[family] = row

    # ============================================================
    # Family 3: BERT + Pol + Sent fusion (no PredFA)
    # ============================================================
    if RUN_BERT_PLUS_POL_SENT:
        Xtr_bps = np.concatenate([Pol_tr_z, Sent_tr_z], axis=1)
        Xva_bps = np.concatenate([Pol_va_z, Sent_va_z], axis=1)
        Xte_bps = np.concatenate([Pol_te_z, Sent_te_z], axis=1)
        for multi_task in ([True] if DO_MULTI_TASK else []) + ([False] if DO_SINGLE_TASK else []):
            family = f"bert_plus_pol_sent_{'mt' if multi_task else 'st'}"
            print(f"\n[GRID] {family}")
            for run_id, cfg in enumerate(iter_grid(BERT_GRID), start=1):
                print(f"[RUN] {family} #{run_id}: {cfg}")
                Yv, Pv, Yt, Pt = train_bert_once(df_doc, tr_idx, va_idx, te_idx, cfg,
                                                  fusion_feats=(Xtr_bps, Xva_bps, Xte_bps), multi_task=multi_task)
                run_dir = os.path.join(OUT_DIR, family, f"run_{run_id:03d}")
                cfg_save = _row_with_metadata({**cfg, "family": family, "feat_set": "text+pol+sent", "feat_dim": Xtr_bps.shape[1]})
                metrics = save_run(run_dir, cfg_save, Yv, Pv, Yt, Pt)
                row = _row_with_metadata({**cfg, "family": family, "run_id": run_id, "feat_set": "text+pol+sent", "feat_dim": Xtr_bps.shape[1]})
                row.update({"val_mae_macro": metrics["val"]["mae_macro"], "val_rho_macro": metrics["val"]["rho_macro"],
                            "test_mae_macro": metrics["test"]["mae_macro"], "test_rho_macro": metrics["test"]["rho_macro"],
                            "run_dir": run_dir})
                all_rows.append(row)
                if family not in best_by_family or row["val_rho_macro"] > best_by_family[family]["val_rho_macro"]:
                    best_by_family[family] = row

    # ============================================================
    # Family 4: BERT + Pol + Sent + PredFA (full kitchen sink)
    # ============================================================
    if RUN_BERT_PLUS_POL_SENT_PREDFA:
        # Need FAC inference for PredFA
        need_cols = [DOC_ID_COL, SENT_IDX_COL, SENT_TEXT_COL, SEED_COL, FA_COL]
        assert all(c in df_sent.columns for c in need_cols), f"Sent CSV missing: {need_cols}"

        ids_tr = set(df_doc.loc[tr_idx, DOC_ID_COL].astype(str))
        ids_va = set(df_doc.loc[va_idx, DOC_ID_COL].astype(str))
        ids_te = set(df_doc.loc[te_idx, DOC_ID_COL].astype(str))
        df_tr_sent = df_sent[df_sent[DOC_ID_COL].isin(ids_tr)].reset_index(drop=True)
        df_va_sent = df_sent[df_sent[DOC_ID_COL].isin(ids_va)].reset_index(drop=True)
        df_te_sent = df_sent[df_sent[DOC_ID_COL].isin(ids_te)].reset_index(drop=True)

        print("\n[FAC] Predicting sentence-level FA probabilities...")
        tr_pred = predict_fa_probs(df_tr_sent, ASL_MODEL_DIR, ASL_TOK_NAME, batch=FAC_BATCH, max_len=FAC_MAXLEN)
        va_pred = predict_fa_probs(df_va_sent, ASL_MODEL_DIR, ASL_TOK_NAME, batch=FAC_BATCH, max_len=FAC_MAXLEN)
        te_pred = predict_fa_probs(df_te_sent, ASL_MODEL_DIR, ASL_TOK_NAME, batch=FAC_BATCH, max_len=FAC_MAXLEN)

        tr_pred.to_csv(os.path.join(OUT_DIR, "predfa_sentence_probs_train.csv"), index=False)
        va_pred.to_csv(os.path.join(OUT_DIR, "predfa_sentence_probs_val.csv"), index=False)
        te_pred.to_csv(os.path.join(OUT_DIR, "predfa_sentence_probs_test.csv"), index=False)

        best_tau_info = tune_fac_tau_on_validation(va_pred, df_va_sent)
        best_tau = float(best_tau_info["tau"])

        F_tr = aggregate_predfa_hspt13(tr_pred, mode="hard", tau=best_tau)
        F_va = aggregate_predfa_hspt13(va_pred, mode="hard", tau=best_tau)
        F_te = aggregate_predfa_hspt13(te_pred, mode="hard", tau=best_tau)

        Xpf_tr = align_predfa_to_split(F_tr, df_doc, tr_idx, split_name="train")
        Xpf_va = align_predfa_to_split(F_va, df_doc, va_idx, split_name="val")
        Xpf_te = align_predfa_to_split(F_te, df_doc, te_idx, split_name="test")
        Xpf_tr_z, Xpf_va_z, Xpf_te_z, pf_mu, pf_sd = zscore_fit_transform(Xpf_tr, Xpf_va, Xpf_te)

        pf_dir = os.path.join(OUT_DIR, "predfa_features")
        ensure_dir(pf_dir)
        pd.DataFrame({"feature": SUM13_ORDER, "mu": pf_mu.tolist(), "sd": pf_sd.tolist()}).to_csv(
            os.path.join(pf_dir, "hspt13_zscore.csv"), index=False)

        # Concatenate ALL three: Pol (15) + Sent (4) + PredFA (13) = 32 dims
        Xtr_full = np.concatenate([Pol_tr_z, Sent_tr_z, Xpf_tr_z], axis=1)
        Xva_full = np.concatenate([Pol_va_z, Sent_va_z, Xpf_va_z], axis=1)
        Xte_full = np.concatenate([Pol_te_z, Sent_te_z, Xpf_te_z], axis=1)

        for multi_task in ([True] if DO_MULTI_TASK else []) + ([False] if DO_SINGLE_TASK else []):
            family = f"bert_plus_pol_sent_predfa_{'mt' if multi_task else 'st'}"
            print(f"\n[GRID] {family}")
            for run_id, cfg in enumerate(iter_grid(BERT_GRID), start=1):
                print(f"[RUN] {family} #{run_id}: {cfg}")
                Yv, Pv, Yt, Pt = train_bert_once(df_doc, tr_idx, va_idx, te_idx, cfg,
                                                  fusion_feats=(Xtr_full, Xva_full, Xte_full), multi_task=multi_task)
                run_dir = os.path.join(OUT_DIR, family, f"run_{run_id:03d}")
                cfg_save = _row_with_metadata({**cfg, "family": family,
                                               "feat_set": "text+pol+sent+predfa",
                                               "feat_dim": Xtr_full.shape[1], "tau": best_tau})
                metrics = save_run(run_dir, cfg_save, Yv, Pv, Yt, Pt)
                row = _row_with_metadata({**cfg, "family": family, "run_id": run_id,
                                          "feat_set": "text+pol+sent+predfa",
                                          "feat_dim": Xtr_full.shape[1], "tau": best_tau})
                row.update({"val_mae_macro": metrics["val"]["mae_macro"], "val_rho_macro": metrics["val"]["rho_macro"],
                            "test_mae_macro": metrics["test"]["mae_macro"], "test_rho_macro": metrics["test"]["rho_macro"],
                            "run_dir": run_dir})
                all_rows.append(row)
                if family not in best_by_family or row["val_rho_macro"] > best_by_family[family]["val_rho_macro"]:
                    best_by_family[family] = row

    if not all_rows:
        raise RuntimeError("No families ran. Check RUN_* env vars.")

    grid_df = pd.DataFrame(all_rows)
    grid_path = os.path.join(OUT_DIR, "opr_grid_search_results.csv")
    grid_df.to_csv(grid_path, index=False)

    best_df = pd.DataFrame(list(best_by_family.values())).sort_values("val_rho_macro", ascending=False)
    best_path = os.path.join(OUT_DIR, "opr_best_by_family.csv")
    best_df.to_csv(best_path, index=False)

    print("\n=== BEST BY FAMILY ===")
    cols = ["family", "opr_run_seed", "fac_run_seed", "val_rho_macro", "val_mae_macro",
            "test_rho_macro", "test_mae_macro", "feat_set", "feat_dim",
            "lr", "epochs", "batch_size", "dropout", "hidden_dim", "run_dir"]
    cols = [c for c in cols if c in best_df.columns]
    if len(best_df):
        print(best_df[cols].to_string(index=False))

    print(f"\n[DONE] Grid -> {grid_path}")
    print(f"[DONE] Best by family -> {best_path}")
    print(f"[DONE] Output dir -> {OUT_DIR}")


if __name__ == "__main__":
    main()
