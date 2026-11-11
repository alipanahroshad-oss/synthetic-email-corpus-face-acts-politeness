#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
OPR_Llama_PolSent_PredFA_GoldFA.py
==================================


export HF_TOKEN=<your_hf_token>   # Llama 3.2 is gated: accept the licence on Hugging Face first
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=1

export FAC_REGISTRY=outputs/llama_lora_sentence_only_multiseed/llama_sentence_only_best_checkpoints.csv

ls -lh OPR_Llama_PolSent_PredFA_GoldFA.py
ls -lh "$FAC_REGISTRY"

python -m py_compile \
  OPR_Llama_PolSent_PredFA_GoldFA.py

nohup bash -lc '
echo "===================================================="
echo "START LLaMA Pol+Sent / PredFA / GoldFA OPR ST-only"
echo "===================================================="

for s in 13 21 42 87 100; do
  echo ""
  echo "===== LLaMA PolSent PredFA GoldFA ST FAC=$s OPR=$s ====="

  FAC_RUN_SEED="$s" \
  OPR_RUN_SEED="$s" \
  SPLIT_RANDOM_STATE=42 \
  FAC_REGISTRY_CSV="$FAC_REGISTRY" \
  OUT_DIR="outputs/opr_sentence_only_llama_polsent_predfa_goldfa_ST_fac_runseed${s}_opr_runseed${s}" \
  RUN_POL_MLP=0 \
  RUN_SENT_MLP=0 \
  RUN_LLAMA_PLUS_POL_SENT=1 \
  RUN_LLAMA_PLUS_POL_SENT_PREDFA=1 \
  RUN_LLAMA_PLUS_POL_SENT_GOLDFA=1 \
  DO_MULTI_TASK=0 \
  DO_SINGLE_TASK=1 \
  python -u OPR_Llama_PolSent_PredFA_GoldFA.py

  echo "===== DONE LLaMA PolSent PredFA GoldFA ST seed$s ====="
done

echo ""
echo "===================================================="
echo "DONE LLaMA Pol+Sent / PredFA / GoldFA OPR ST-only"
echo "===================================================="
' > opr_llama_polsent_predfa_goldfa_ST_multiseed.log 2>&1 &

tail -f opr_llama_polsent_predfa_goldfa_ST_multiseed.log
-------------------
aggregate:
python - <<'PY'
import os, glob, json, re
import pandas as pd

OUT_DIR = "outputs/opr_sentence_only_llama_polsent_predfa_goldfa_ST_multiseed_summary"
os.makedirs(OUT_DIR, exist_ok=True)

TARGETS = {
    "Directness": "Directness_vs_Indirectness__GOLD",
    "Positive Face Saving": "Structural_Politeness_and_Politeness_Markers__GOLD",
    "Negative Face Saving": "Tone_and_Overall_Consideration__GOLD",
}

FAMILIES = [
    "llama_plus_pol_sent_st",
    "llama_plus_pol_sent_predfa_st",
    "llama_plus_pol_sent_goldfa_st",
]

paths = sorted(glob.glob(
    "outputs/opr_sentence_only_llama_polsent_predfa_goldfa_ST_fac_runseed*_opr_runseed*/opr_best_by_family.csv"
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

all_path = os.path.join(OUT_DIR, "llama_polsent_predfa_goldfa_ST_all_best_by_family.csv")
per_seed_path = os.path.join(OUT_DIR, "llama_polsent_predfa_goldfa_ST_per_seed_per_dimension.csv")

all_df.to_csv(all_path, index=False)
per_df.to_csv(per_seed_path, index=False)

def ms(g, col):
    # mean +/- standard error (SEM), matching the main OPR tables
    n = len(g)
    sem = (g[col].std(ddof=1) / n ** 0.5) if n > 1 else 0.0
    return f"{g[col].mean():.3f} ± {sem:.3f}"

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
    "llama_plus_pol_sent_st": 0,
    "llama_plus_pol_sent_predfa_st": 1,
    "llama_plus_pol_sent_goldfa_st": 2,
}
table_df["order"] = table_df["Model"].map(order)
table_df = table_df.sort_values("order").drop(columns=["order"])

table_path = os.path.join(OUT_DIR, "llama_polsent_predfa_goldfa_ST_table_rows_per_dimension.csv")
table_df.to_csv(table_path, index=False)

summary = (
    all_df.groupby("family")
    .agg(
        n_runs=("opr_run_seed", "nunique"),
        val_rho_macro_mean=("val_rho_macro", "mean"),
        val_rho_macro_sem=("val_rho_macro", "sem"),
        val_mae_macro_mean=("val_mae_macro", "mean"),
        val_mae_macro_sem=("val_mae_macro", "sem"),
        test_rho_macro_mean=("test_rho_macro", "mean"),
        test_rho_macro_sem=("test_rho_macro", "sem"),
        test_mae_macro_mean=("test_mae_macro", "mean"),
        test_mae_macro_sem=("test_mae_macro", "sem"),
    )
    .reset_index()
)

for c in ["val_rho_macro", "val_mae_macro", "test_rho_macro", "test_mae_macro"]:
    summary[f"{c}_mean_sem"] = summary.apply(
        lambda r: f"{r[f'{c}_mean']:.3f} ± {r[f'{c}_std']:.3f}", axis=1
    )

summary["order"] = summary["family"].map(order)
summary = summary.sort_values("order").drop(columns=["order"])

summary_path = os.path.join(OUT_DIR, "llama_polsent_predfa_goldfa_ST_summary_mean_sem.csv")
summary.to_csv(summary_path, index=False)

print("\n=== LLAMA POL/SENT/PREDFA/GOLDFA ST MACRO SUMMARY ===")
print(summary[[
    "family", "n_runs",
    "val_rho_macro_mean_sem", "val_mae_macro_mean_sem",
    "test_rho_macro_mean_sem", "test_mae_macro_mean_sem"
]].to_string(index=False))

print("\n=== LLAMA POL/SENT/PREDFA/GOLDFA ST TABLE ROWS ===")
print(table_df.to_string(index=False))

print("\n[DONE] wrote:")
print(all_path)
print(per_seed_path)
print(table_path)
print(summary_path)
PY
------------

LLaMA version of the classical-features comparison for OPR. Mirrors
the BERT Pol+Sent+PredFA+GoldFA pipeline exactly, but with:

  • Doc-level regressor: Llama-3.2-1B + LoRA + last-non-pad-token pool
  • FAC sentence-level: Llama-3.2-1B + LoRA adapter from registry CSV
  • Pol features: same 15-dim lexicon-based vector (Danescu-inspired)
  • Sent features: same 4-dim VADER vector
  • GoldFA features: same 13-dim HSPT from human annotations

Five families:

  pol_mlp_{mt,st}                          — Pol-only MLP (no transformer)
  sent_mlp_{mt,st}                         — Sent-only MLP (no transformer)
  llama_plus_pol_sent_{mt,st}              — LLaMA + Pol + Sent
  llama_plus_pol_sent_predfa_{mt,st}       — LLaMA + Pol + Sent + PredFA
  llama_plus_pol_sent_goldfa_{mt,st}       — LLaMA + Pol + Sent + GoldFA (oracle)

NOTE: pol_mlp and sent_mlp produce IDENTICAL numbers to the BERT
version's pol_mlp / sent_mlp (no transformer involved). If you already
ran those on BERT, set RUN_POL_MLP=0 RUN_SENT_MLP=0 to save time.

NOTE: llama_plus_pol_sent (no FA) and llama_plus_pol_sent_predfa
overlap with what you might already have if you ran the LLaMA PredFA
pipeline; if so, set those toggles to 0 too.

PredFA family needs FAC LoRA registry + FAC_RUN_SEED env var.
GoldFA family needs no FAC model — uses human annotations directly.
"""

# ============================================================
# USER PATHS
# ============================================================
DOC_CSV = "data/corpus/email_text_gold_three_dimensions_politeness_score_with_seed_correct.csv"
SENT_CSV = "data/corpus/480_seeds_with_email_id_Sep.csv"

import os

FAC_REGISTRY_CSV = os.environ.get(
    "FAC_REGISTRY_CSV",
    "outputs/llama_lora_sentence_only_multiseed/llama_sentence_only_best_checkpoints.csv"
)
LLAMA_BASE_MODEL = os.environ.get("LLAMA_BASE_MODEL", "meta-llama/Llama-3.2-1B")

OUT_DIR = os.environ.get(
    "OUT_DIR",
    "outputs/opr_sentence_only_llama_polsent_predfa_goldfa_gridsearch"
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
    AutoConfig,
    get_linear_schedule_with_warmup,
)

try:
    from peft import PeftModel, LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("[WARN] peft not installed. LoRA will fail.")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VADER
    VADER_AVAILABLE = True
except ImportError:
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer as VADER
        VADER_AVAILABLE = True
    except ImportError:
        VADER_AVAILABLE = False
        print("[WARN] VADER not available. pip install vaderSentiment")

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

LLAMA_MAXLEN_DOC = 512
LLAMA_MAXLEN_SENT = 128
FAC_BATCH = 32

# Which systems to run
RUN_POL_MLP                    = os.environ.get("RUN_POL_MLP", "1") == "1"
RUN_SENT_MLP                   = os.environ.get("RUN_SENT_MLP", "1") == "1"
RUN_LLAMA_PLUS_POL_SENT        = os.environ.get("RUN_LLAMA_PLUS_POL_SENT", "1") == "1"
RUN_LLAMA_PLUS_POL_SENT_PREDFA = os.environ.get("RUN_LLAMA_PLUS_POL_SENT_PREDFA", "1") == "1"
RUN_LLAMA_PLUS_POL_SENT_GOLDFA = os.environ.get("RUN_LLAMA_PLUS_POL_SENT_GOLDFA", "1") == "1"

DO_MULTI_TASK = os.environ.get("DO_MULTI_TASK", "1") == "1"
DO_SINGLE_TASK = os.environ.get("DO_SINGLE_TASK", "1") == "1"

NORMALIZE_POL_BY_TOKENS = os.environ.get("NORMALIZE_POL_BY_TOKENS", "0") == "1"

GRID_PREDFA_MODES = ["hard"]
FAC_TAU_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2).tolist()

# Grids match the LLaMA PredFA pipeline (and BERT Pol+Sent for MLP_GRID)
# ============================================================
# Extended LoRA search + checkpointing
#   EXTENDED_LORA=1 (default) -> r in {8,16,32}, alpha=2r, epochs {5,8,12},
#                                lr fixed 2e-4 (stage-2 winner from main sweep).
#                                Set EXTENDED_LORA=0 for the earlier 4-config grid (r = 8).
#   LORA_TARGETS="attn" (default) -> q,k,v,o  (matches the main sweep: tm-attn)
#   SAVE_BEST_CHECKPOINTS=1 -> save best trainable state (LoRA+heads) per run
#                              under OUT_DIR/checkpoints/
#   Overrides: EXTLORA_LRS, EXTLORA_EPOCHS, EXTLORA_RANKS (comma-separated)
# ============================================================
EXTENDED_LORA = os.environ.get("EXTENDED_LORA", "1") == "1"
LORA_TARGETS = os.environ.get("LORA_TARGETS", "attn")
SAVE_BEST_CHECKPOINTS = os.environ.get("SAVE_BEST_CHECKPOINTS", "0") == "1"

if EXTENDED_LORA:
    _lrs = [float(x) for x in os.environ.get("EXTLORA_LRS", "2e-4").split(",")]
    _eps = [int(x) for x in os.environ.get("EXTLORA_EPOCHS", "5,8,12").split(",")]
    _rs = [int(x) for x in os.environ.get("EXTLORA_RANKS", "8,16,32").split(",")]
    LLAMA_GRID = {
        "lr": _lrs, "epochs": _eps, "batch_size": [4], "dropout": [0.1],
        "weight_decay": [0.01], "hidden_dim": [128],
        "lora_r": _rs, "lora_alpha": ["2r"], "lora_dropout": [0.05],
    }
    print(f"[EXTLORA] grid lr={_lrs} epochs={_eps} r={_rs} alpha=2r "
          f"targets={LORA_TARGETS} -> {len(_lrs)*len(_eps)*len(_rs)} configs")
else:
    LLAMA_GRID = {
        "lr": [1e-4, 2e-4], "epochs": [3, 5], "batch_size": [4], "dropout": [0.1],
        "weight_decay": [0.01], "hidden_dim": [128],
        "lora_r": [8], "lora_alpha": [16], "lora_dropout": [0.1],
    }

_CKPT_FAMILY = "llama"
def _set_ckpt_family(fam):
    global _CKPT_FAMILY
    _CKPT_FAMILY = str(fam)
def _save_ckpt(state_dict, tag):
    if not SAVE_BEST_CHECKPOINTS or state_dict is None:
        return
    d = os.path.join(OUT_DIR, "checkpoints"); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{tag}.pt"); torch.save(state_dict, p)
    print(f"[CKPT] saved {p}")

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
    if " " in phrase:
        return text_lower.count(phrase)
    return len(re.findall(rf"\b{re.escape(phrase)}\b", text_lower))


def compute_pol_features(text: str) -> np.ndarray:
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    text_lower = text.lower()
    n_tokens = max(1, len(re.findall(r"\b\w+\b", text_lower)))
    feats = []
    for lex_name, words in LEXICONS.items():
        feats.append(float(sum(count_phrase(text_lower, w) for w in words)))
    for pron_name, words in PRONOUNS.items():
        feats.append(float(sum(count_phrase(text_lower, w) for w in words)))
    feats.append(float(sum(count_phrase(text_lower, v) for v in IMPERATIVE_START_VERBS)))
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
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    s = _get_vader().polarity_scores(text)
    return np.array([s["neg"], s["neu"], s["pos"], s["compound"]], dtype=np.float32)


def compute_polsent_matrix(df_doc: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
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
    cfgs = [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]
    for c in cfgs:
        if c.get("lora_alpha") == "2r":
            c["lora_alpha"] = 2 * int(c["lora_r"])
    return cfgs


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


def resolve_fac_lora_adapter(registry_csv: str, fac_run_seed: int) -> str:
    if not os.path.isfile(registry_csv):
        raise FileNotFoundError(f"[FAC] Registry CSV not found: {registry_csv}")
    df = pd.read_csv(registry_csv)
    seed_col_candidates = ["run_seed", "fac_run_seed", "seed", "RUN_SEED"]
    path_col_candidates = ["adapter_dir", "best_checkpoint_dir", "checkpoint_dir", "path", "model_dir", "adapter_path"]
    seed_col = next((c for c in seed_col_candidates if c in df.columns), None)
    path_col = next((c for c in path_col_candidates if c in df.columns), None)
    if seed_col is None:
        raise ValueError(f"[FAC] Registry CSV has no seed column. Tried {seed_col_candidates}. Columns: {list(df.columns)}")
    if path_col is None:
        raise ValueError(f"[FAC] Registry CSV has no path column. Tried {path_col_candidates}. Columns: {list(df.columns)}")
    df[seed_col] = pd.to_numeric(df[seed_col], errors="coerce").astype("Int64")
    rows = df[df[seed_col] == int(fac_run_seed)]
    if len(rows) == 0:
        raise ValueError(f"[FAC] No row for seed={fac_run_seed}. Available: {sorted(df[seed_col].dropna().unique().tolist())}")
    adapter_dir = str(rows.iloc[0][path_col])
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"[FAC] Adapter dir not found: {adapter_dir}")
    print(f"[FAC] Resolved seed={fac_run_seed} -> LoRA adapter: {adapter_dir}")
    return adapter_dir


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    seq_lengths = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(last_hidden_states.size(0), device=last_hidden_states.device)
    return last_hidden_states[batch_idx, seq_lengths]


def _llama_tok_with_pad():
    tok = AutoTokenizer.from_pretrained(LLAMA_BASE_MODEL, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    return tok


# ============================================================
# FAC SENTENCE-LEVEL INFERENCE (LLaMA + LoRA)
# ============================================================
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


def predict_fa_probs_llama(df_split, adapter_dir, batch=32, max_len=128):
    if len(df_split) == 0:
        return pd.DataFrame(columns=[DOC_ID_COL, SENT_IDX_COL] + [f"prob_{l}" for l in FA_LABS])
    if not PEFT_AVAILABLE: raise RuntimeError("peft required")
    print(f"[FAC] Loading LLaMA base + LoRA adapter: {adapter_dir}")
    tok = _llama_tok_with_pad()
    try:
        base = AutoModelForSequenceClassification.from_pretrained(
            LLAMA_BASE_MODEL, num_labels=9,
            problem_type="multi_label_classification", torch_dtype=torch.float16,
        )
        base.config.pad_token_id = tok.pad_token_id
        mdl = PeftModel.from_pretrained(base, adapter_dir).to(DEVICE)
    except Exception as e:
        raise OSError(f"[FAC] Could not load adapter from {adapter_dir}: {e}") from e
    mdl.eval()

    ds = SentDataset(df_split, SENT_TEXT_COL, tok, max_length=max_len)
    dl = DataLoader(ds, batch_size=batch, shuffle=False)
    doc_ids, sids, allP = [], [], []
    with torch.no_grad():
        for b in dl:
            logits = mdl(input_ids=b["input_ids"].to(DEVICE),
                         attention_mask=b["attention_mask"].to(DEVICE)).logits
            P = torch.sigmoid(logits.float()).cpu().numpy()
            allP.append(P); doc_ids.extend(b["doc_id"]); sids.extend(b["sent_idx"])

    del mdl, base
    if torch.cuda.is_available(): torch.cuda.empty_cache()

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
# GOLDFA + PREDFA AGGREGATION (HSPT-13)
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


def gold_multihot_sentences(df_sent: pd.DataFrame) -> pd.DataFrame:
    df = df_sent.copy()
    df[DOC_ID_COL] = df[DOC_ID_COL].astype(str)
    df[SENT_IDX_COL] = pd.to_numeric(df[SENT_IDX_COL], errors="coerce").fillna(0).astype(int)
    Y = np.zeros((len(df), len(FA_LABS)), dtype=np.float32)
    lab2i = {l: i for i, l in enumerate(FA_LABS)}
    for i, x in enumerate(df[FA_COL].tolist()):
        for lab in parse_gold_faceacts(x):
            Y[i, lab2i[lab]] = 1.0
    out = {DOC_ID_COL: df[DOC_ID_COL].tolist(), SENT_IDX_COL: df[SENT_IDX_COL].tolist()}
    for j, l in enumerate(FA_LABS):
        out[f"prob_{l}"] = Y[:, j]
    return pd.DataFrame(out)


def aggregate_goldfa_hspt13(dfp: pd.DataFrame) -> pd.DataFrame:
    """Gold is already binary — reuse PredFA aggregator in soft mode."""
    return aggregate_predfa_hspt13(dfp, mode="soft", tau=None)


def align_predfa_to_split(F_pred, df_doc, idx, split_name=""):
    want = df_doc.loc[idx, DOC_ID_COL].astype(str).tolist()
    F = F_pred.copy()
    F[DOC_ID_COL] = F[DOC_ID_COL].astype(str)
    F = F.drop_duplicates(subset=[DOC_ID_COL], keep="first").set_index(DOC_ID_COL)
    missing = [d for d in want if d not in set(F.index)]
    if missing:
        print(f"[ALIGN][{split_name}] WARNING: {len(missing)}/{len(want)} doc_ids missing.")
        try:
            with open(os.path.join(OUT_DIR, f"missing_ids_{split_name}.json"), "w") as f:
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
# DOC DATASETS / MODELS (LLaMA)
# ============================================================
class DocLlamaDataset(Dataset):
    def __init__(self, df, text_col, targets, tokenizer, max_length=512):
        enc = tokenizer(df[text_col].astype(str).tolist(),
                        padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        self.ids = enc["input_ids"]; self.attn = enc["attention_mask"]
        self.y = torch.tensor(df[targets].to_numpy(np.float32))
    def __len__(self): return self.ids.size(0)
    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "attention_mask": self.attn[i], "targets": self.y[i]}


class DocLlamaFusionDataset(Dataset):
    """Generic fusion: takes a single concatenated feature matrix."""
    def __init__(self, df, text_col, targets, tokenizer, fusion_feats: np.ndarray, max_length=512):
        assert len(df) == len(fusion_feats), "Fusion rows must align with doc rows."
        enc = tokenizer(df[text_col].astype(str).tolist(),
                        padding=True, truncation=True, max_length=max_length, return_tensors="pt")
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


class LlamaDocRegressor(nn.Module):
    def __init__(self, base_model_name, n_targets=3, fusion_dim=0,
                 hidden_dim=128, dropout=0.1,
                 lora_r=8, lora_alpha=16, lora_dropout=0.1,
                 pad_token_id=None):
        super().__init__()
        if not PEFT_AVAILABLE: raise RuntimeError("peft required")
        config = AutoConfig.from_pretrained(base_model_name)
        if pad_token_id is not None: config.pad_token_id = pad_token_id
        base = AutoModel.from_pretrained(base_model_name, config=config, torch_dtype=torch.float16)
        lora_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha,
            target_modules=(["q_proj", "k_proj", "v_proj", "o_proj"]
                            + (["gate_proj", "up_proj", "down_proj"] if LORA_TARGETS == "attn_mlp" else [])),
            lora_dropout=lora_dropout, bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.enc = get_peft_model(base, lora_cfg)
        self.enc.print_trainable_parameters()
        hdim = base.config.hidden_size
        self.fusion_dim = int(fusion_dim)
        in_dim = hdim + self.fusion_dim
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_targets)])

    def forward(self, input_ids, attention_mask, predfa=None):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = last_token_pool(out.last_hidden_state, attention_mask).float()
        if self.fusion_dim > 0:
            if predfa is None: raise ValueError("Fusion features required when fusion_dim > 0")
            z = torch.cat([pooled, predfa.float()], dim=-1)
        else:
            z = pooled
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
# METRICS / TRAIN HELPERS
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


def predict_llama(model, dl):
    model.eval(); Ys, Ps = [], []
    with torch.no_grad():
        for b in dl:
            pf = b.get("predfa"); pf = pf.to(DEVICE) if pf is not None else None
            pr = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), pf)
            Ys.append(b["targets"].numpy()); Ps.append(pr.float().cpu().numpy())
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


def _cleanup_model(model):
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def train_llama_once(df_doc, tr_idx, va_idx, te_idx, cfg, fusion_feats=None, multi_task=True):
    set_seed(RANDOM_STATE)
    tk = _llama_tok_with_pad()
    df_tr = df_doc.loc[tr_idx].reset_index(drop=True)
    df_va = df_doc.loc[va_idx].reset_index(drop=True)
    df_te = df_doc.loc[te_idx].reset_index(drop=True)

    if fusion_feats is None:
        ds_tr = DocLlamaDataset(df_tr, DOC_TEXT_COL, TARGETS, tk, max_length=LLAMA_MAXLEN_DOC)
        ds_va = DocLlamaDataset(df_va, DOC_TEXT_COL, TARGETS, tk, max_length=LLAMA_MAXLEN_DOC)
        ds_te = DocLlamaDataset(df_te, DOC_TEXT_COL, TARGETS, tk, max_length=LLAMA_MAXLEN_DOC)
        fusion_dim = 0
    else:
        Xtr, Xva, Xte = fusion_feats
        ds_tr = DocLlamaFusionDataset(df_tr, DOC_TEXT_COL, TARGETS, tk, Xtr, max_length=LLAMA_MAXLEN_DOC)
        ds_va = DocLlamaFusionDataset(df_va, DOC_TEXT_COL, TARGETS, tk, Xva, max_length=LLAMA_MAXLEN_DOC)
        ds_te = DocLlamaFusionDataset(df_te, DOC_TEXT_COL, TARGETS, tk, Xte, max_length=LLAMA_MAXLEN_DOC)
        fusion_dim = Xtr.shape[1]

    g = make_dl_generator(RANDOM_STATE)
    dl_tr = DataLoader(ds_tr, batch_size=cfg["batch_size"], shuffle=True, generator=g)
    dl_va = DataLoader(ds_va, batch_size=cfg["batch_size"], shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=cfg["batch_size"], shuffle=False)

    if multi_task:
        model = LlamaDocRegressor(
            base_model_name=LLAMA_BASE_MODEL, n_targets=3, fusion_dim=fusion_dim,
            hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"],
            lora_r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"], pad_token_id=tk.pad_token_id,
        ).to(DEVICE)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=cfg["lr"], weight_decay=cfg["weight_decay"])
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
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sch.step()
            Yv, Pv = predict_llama(model, dl_va)
            score = mae_spearman(Yv, Pv, TARGETS)["rho_macro"]
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                              if v.requires_grad or "head" in k or "hidden" in k}
        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        _save_ckpt(best_state, f"{_CKPT_FAMILY}_mt_r{cfg['lora_r']}_a{cfg['lora_alpha']}_ep{cfg['epochs']}_seed{RANDOM_STATE}")
        model.load_state_dict(best_state, strict=False)
        Yva, Pva = predict_llama(model, dl_va)
        Yte, Pte = predict_llama(model, dl_te)
        _cleanup_model(model)
        return Yva, Pva, Yte, Pte

    # single-task
    Yva_all = df_doc.loc[va_idx, TARGETS].to_numpy(np.float32)
    Yte_all = df_doc.loc[te_idx, TARGETS].to_numpy(np.float32)
    Pva_all = np.zeros_like(Yva_all); Pte_all = np.zeros_like(Yte_all)

    for t_i in range(3):
        set_seed(RANDOM_STATE + t_i)
        g = make_dl_generator(RANDOM_STATE + t_i)
        dl_tr = DataLoader(ds_tr, batch_size=cfg["batch_size"], shuffle=True, generator=g)
        model = LlamaDocRegressor(
            base_model_name=LLAMA_BASE_MODEL, n_targets=1, fusion_dim=fusion_dim,
            hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"],
            lora_r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"], pad_token_id=tk.pad_token_id,
        ).to(DEVICE)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=cfg["lr"], weight_decay=cfg["weight_decay"])
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
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sch.step()
            model.eval(); yv, pv = [], []
            with torch.no_grad():
                for b in dl_va:
                    pf = b.get("predfa"); pf = pf.to(DEVICE) if pf is not None else None
                    pr = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), pf)
                    yv.append(b["targets"][:, t_i].numpy())
                    pv.append(pr.float().cpu().numpy().squeeze(1))
            yv = np.concatenate(yv, 0); pv = np.concatenate(pv, 0)
            rho, _ = spearmanr(yv, pv)
            score = float(rho) if rho is not None and not np.isnan(rho) else 0.0
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                              if v.requires_grad or "head" in k or "hidden" in k}
        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        _save_ckpt(best_state, f"{_CKPT_FAMILY}_st_t{t_i}_r{cfg['lora_r']}_a{cfg['lora_alpha']}_ep{cfg['epochs']}_seed{RANDOM_STATE + t_i}")
        model.load_state_dict(best_state, strict=False); model.eval()
        for dl, store in [(dl_va, Pva_all), (dl_te, Pte_all)]:
            preds = []
            with torch.no_grad():
                for b in dl:
                    pf = b.get("predfa"); pf = pf.to(DEVICE) if pf is not None else None
                    pr = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), pf)
                    preds.append(pr.float().cpu().numpy().squeeze(1))
            store[:, t_i] = np.concatenate(preds, 0)
        _cleanup_model(model)
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
def _run_metadata(adapter_dir=""):
    return {
        "opr_run_seed": int(OPR_RUN_SEED),
        "split_random_state": int(SPLIT_RANDOM_STATE),
        "fac_run_seed": os.environ.get("FAC_RUN_SEED", ""),
        "fac_adapter_dir": adapter_dir,
        "llama_base_model": LLAMA_BASE_MODEL,
        "normalize_pol_by_tokens": NORMALIZE_POL_BY_TOKENS,
    }


def _row_with_metadata(row, adapter_dir=""):
    return {**row, **_run_metadata(adapter_dir)}


def main():
    set_seed(OPR_RUN_SEED); ensure_dir(OUT_DIR)
    need_predfa = RUN_LLAMA_PLUS_POL_SENT_PREDFA
    need_goldfa = RUN_LLAMA_PLUS_POL_SENT_GOLDFA

    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Output dir: {OUT_DIR}")
    print(f"[INFO] LLaMA base: {LLAMA_BASE_MODEL}")
    print(f"[INFO] Split random state: {SPLIT_RANDOM_STATE}")
    print(f"[INFO] OPR run seed: {OPR_RUN_SEED}")
    print(f"[INFO] FAC run seed: {os.environ.get('FAC_RUN_SEED','')}")
    print(f"[INFO] FAC registry: {FAC_REGISTRY_CSV if need_predfa else '[not used]'}")
    print(f"[INFO] Families: pol_mlp={RUN_POL_MLP} sent_mlp={RUN_SENT_MLP} "
          f"llama+pol+sent={RUN_LLAMA_PLUS_POL_SENT} +predfa={RUN_LLAMA_PLUS_POL_SENT_PREDFA} "
          f"+goldfa={RUN_LLAMA_PLUS_POL_SENT_GOLDFA}")

    fac_adapter_dir = ""
    if need_predfa:
        fac_seed_env = os.environ.get("FAC_RUN_SEED", "")
        if not fac_seed_env:
            raise RuntimeError("FAC_RUN_SEED env var required for PredFA family.")
        fac_adapter_dir = resolve_fac_lora_adapter(FAC_REGISTRY_CSV, int(fac_seed_env))

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
        "fac_adapter_dir": fac_adapter_dir,
        "llama_base_model": LLAMA_BASE_MODEL,
        "n_docs_train": int(len(tr_idx)), "n_docs_val": int(len(va_idx)), "n_docs_test": int(len(te_idx)),
        "train_seeds": sorted(map(int, seeds_tr)),
        "val_seeds": sorted(map(int, seeds_va)),
        "test_seeds": sorted(map(int, seeds_te)),
        "pol_features": POL_FEATURE_ORDER,
        "sent_features": SENT_FEATURE_ORDER,
    }
    with open(os.path.join(OUT_DIR, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"[INFO] Split: train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")

    Y = df_doc[TARGETS].to_numpy(np.float32)
    Y_tr, Y_va, Y_te = Y[tr_idx], Y[va_idx], Y[te_idx]

    # Compute Pol and Sent features (doc-level)
    print("\n[FEATS] Computing doc-level Pol+Sent features...")
    Pol_all, Sent_all = compute_polsent_matrix(df_doc)
    print(f"[FEATS] Pol shape: {Pol_all.shape}, Sent shape: {Sent_all.shape}")
    Pol_tr, Pol_va, Pol_te = Pol_all[tr_idx], Pol_all[va_idx], Pol_all[te_idx]
    Sent_tr, Sent_va, Sent_te = Sent_all[tr_idx], Sent_all[va_idx], Sent_all[te_idx]
    Pol_tr_z, Pol_va_z, Pol_te_z, pol_mu, pol_sd = zscore_fit_transform(Pol_tr, Pol_va, Pol_te)
    Sent_tr_z, Sent_va_z, Sent_te_z, sent_mu, sent_sd = zscore_fit_transform(Sent_tr, Sent_va, Sent_te)

    feat_dir = os.path.join(OUT_DIR, "feature_manifests")
    ensure_dir(feat_dir)
    pd.DataFrame({"feature": POL_FEATURE_ORDER, "mu": pol_mu.tolist(), "sd": pol_sd.tolist()}).to_csv(
        os.path.join(feat_dir, "pol_zscore.csv"), index=False)
    pd.DataFrame({"feature": SENT_FEATURE_ORDER, "mu": sent_mu.tolist(), "sd": sent_sd.tolist()}).to_csv(
        os.path.join(feat_dir, "sent_zscore.csv"), index=False)

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

    # ============================================================
    # Family 3: LLaMA + Pol + Sent (no FA)
    # ============================================================
    if RUN_LLAMA_PLUS_POL_SENT:
        Xtr_lps = np.concatenate([Pol_tr_z, Sent_tr_z], axis=1)
        Xva_lps = np.concatenate([Pol_va_z, Sent_va_z], axis=1)
        Xte_lps = np.concatenate([Pol_te_z, Sent_te_z], axis=1)
        for multi_task in ([True] if DO_MULTI_TASK else []) + ([False] if DO_SINGLE_TASK else []):
            family = f"llama_plus_pol_sent_{'mt' if multi_task else 'st'}"
            _set_ckpt_family(family)
            print(f"\n[GRID] {family}")
            for run_id, cfg in enumerate(iter_grid(LLAMA_GRID), start=1):
                print(f"[RUN] {family} #{run_id}: {cfg}")
                Yv, Pv, Yt, Pt = train_llama_once(df_doc, tr_idx, va_idx, te_idx, cfg,
                                                   fusion_feats=(Xtr_lps, Xva_lps, Xte_lps), multi_task=multi_task)
                run_dir = os.path.join(OUT_DIR, family, f"run_{run_id:03d}")
                cfg_save = _row_with_metadata({**cfg, "family": family, "feat_set": "text+pol+sent", "feat_dim": Xtr_lps.shape[1]})
                metrics = save_run(run_dir, cfg_save, Yv, Pv, Yt, Pt)
                row = _row_with_metadata({**cfg, "family": family, "run_id": run_id, "feat_set": "text+pol+sent", "feat_dim": Xtr_lps.shape[1]})
                row.update({"val_mae_macro": metrics["val"]["mae_macro"], "val_rho_macro": metrics["val"]["rho_macro"],
                            "test_mae_macro": metrics["test"]["mae_macro"], "test_rho_macro": metrics["test"]["rho_macro"],
                            "run_dir": run_dir})
                all_rows.append(row)
                if family not in best_by_family or row["val_rho_macro"] > best_by_family[family]["val_rho_macro"]:
                    best_by_family[family] = row

    # ============================================================
    # Family 4: LLaMA + Pol + Sent + PredFA (FAC inference)
    # ============================================================
    if RUN_LLAMA_PLUS_POL_SENT_PREDFA:
        need_cols = [DOC_ID_COL, SENT_IDX_COL, SENT_TEXT_COL, SEED_COL, FA_COL]
        assert all(c in df_sent.columns for c in need_cols), f"Sent CSV missing: {need_cols}"

        ids_tr = set(df_doc.loc[tr_idx, DOC_ID_COL].astype(str))
        ids_va = set(df_doc.loc[va_idx, DOC_ID_COL].astype(str))
        ids_te = set(df_doc.loc[te_idx, DOC_ID_COL].astype(str))
        df_tr_sent = df_sent[df_sent[DOC_ID_COL].isin(ids_tr)].reset_index(drop=True)
        df_va_sent = df_sent[df_sent[DOC_ID_COL].isin(ids_va)].reset_index(drop=True)
        df_te_sent = df_sent[df_sent[DOC_ID_COL].isin(ids_te)].reset_index(drop=True)

        print(f"\n[FAC] Predicting sentence-level FA probabilities (FAC seed={os.environ.get('FAC_RUN_SEED')})...")
        tr_pred = predict_fa_probs_llama(df_tr_sent, fac_adapter_dir, batch=FAC_BATCH, max_len=LLAMA_MAXLEN_SENT)
        va_pred = predict_fa_probs_llama(df_va_sent, fac_adapter_dir, batch=FAC_BATCH, max_len=LLAMA_MAXLEN_SENT)
        te_pred = predict_fa_probs_llama(df_te_sent, fac_adapter_dir, batch=FAC_BATCH, max_len=LLAMA_MAXLEN_SENT)

        tr_pred.to_csv(os.path.join(OUT_DIR, "predfa_sentence_probs_train.csv"), index=False)
        va_pred.to_csv(os.path.join(OUT_DIR, "predfa_sentence_probs_val.csv"), index=False)
        te_pred.to_csv(os.path.join(OUT_DIR, "predfa_sentence_probs_test.csv"), index=False)

        best_tau_info = tune_fac_tau_on_validation(va_pred, df_va_sent)
        best_tau = float(best_tau_info["tau"])

        F_tr = aggregate_predfa_hspt13(tr_pred, mode="hard", tau=best_tau)
        F_va = aggregate_predfa_hspt13(va_pred, mode="hard", tau=best_tau)
        F_te = aggregate_predfa_hspt13(te_pred, mode="hard", tau=best_tau)

        Xpf_tr = align_predfa_to_split(F_tr, df_doc, tr_idx, split_name="train_predfa")
        Xpf_va = align_predfa_to_split(F_va, df_doc, va_idx, split_name="val_predfa")
        Xpf_te = align_predfa_to_split(F_te, df_doc, te_idx, split_name="test_predfa")
        Xpf_tr_z, Xpf_va_z, Xpf_te_z, pf_mu, pf_sd = zscore_fit_transform(Xpf_tr, Xpf_va, Xpf_te)

        pf_dir = os.path.join(OUT_DIR, "predfa_features")
        ensure_dir(pf_dir)
        pd.DataFrame({"feature": SUM13_ORDER, "mu": pf_mu.tolist(), "sd": pf_sd.tolist()}).to_csv(
            os.path.join(pf_dir, "hspt13_zscore.csv"), index=False)

        # Pol (15) + Sent (4) + PredFA (13) = 32 dims fused with LLaMA pooled
        Xtr_full = np.concatenate([Pol_tr_z, Sent_tr_z, Xpf_tr_z], axis=1)
        Xva_full = np.concatenate([Pol_va_z, Sent_va_z, Xpf_va_z], axis=1)
        Xte_full = np.concatenate([Pol_te_z, Sent_te_z, Xpf_te_z], axis=1)

        for multi_task in ([True] if DO_MULTI_TASK else []) + ([False] if DO_SINGLE_TASK else []):
            family = f"llama_plus_pol_sent_predfa_{'mt' if multi_task else 'st'}"
            _set_ckpt_family(family)
            print(f"\n[GRID] {family}")
            for run_id, cfg in enumerate(iter_grid(LLAMA_GRID), start=1):
                print(f"[RUN] {family} #{run_id}: {cfg}")
                Yv, Pv, Yt, Pt = train_llama_once(df_doc, tr_idx, va_idx, te_idx, cfg,
                                                   fusion_feats=(Xtr_full, Xva_full, Xte_full), multi_task=multi_task)
                run_dir = os.path.join(OUT_DIR, family, f"run_{run_id:03d}")
                cfg_save = _row_with_metadata({**cfg, "family": family,
                                               "feat_set": "text+pol+sent+predfa",
                                               "feat_dim": Xtr_full.shape[1], "tau": best_tau}, fac_adapter_dir)
                metrics = save_run(run_dir, cfg_save, Yv, Pv, Yt, Pt)
                row = _row_with_metadata({**cfg, "family": family, "run_id": run_id,
                                          "feat_set": "text+pol+sent+predfa",
                                          "feat_dim": Xtr_full.shape[1], "tau": best_tau}, fac_adapter_dir)
                row.update({"val_mae_macro": metrics["val"]["mae_macro"], "val_rho_macro": metrics["val"]["rho_macro"],
                            "test_mae_macro": metrics["test"]["mae_macro"], "test_rho_macro": metrics["test"]["rho_macro"],
                            "run_dir": run_dir})
                all_rows.append(row)
                if family not in best_by_family or row["val_rho_macro"] > best_by_family[family]["val_rho_macro"]:
                    best_by_family[family] = row

    # ============================================================
    # Family 5: LLaMA + Pol + Sent + GoldFA (oracle)
    # ============================================================
    if RUN_LLAMA_PLUS_POL_SENT_GOLDFA:
        need_cols = [DOC_ID_COL, SENT_IDX_COL, SENT_TEXT_COL, SEED_COL, FA_COL]
        assert all(c in df_sent.columns for c in need_cols), f"Sent CSV missing: {need_cols}"

        ids_tr = set(df_doc.loc[tr_idx, DOC_ID_COL].astype(str))
        ids_va = set(df_doc.loc[va_idx, DOC_ID_COL].astype(str))
        ids_te = set(df_doc.loc[te_idx, DOC_ID_COL].astype(str))
        df_tr_sent_g = df_sent[df_sent[DOC_ID_COL].isin(ids_tr)].reset_index(drop=True)
        df_va_sent_g = df_sent[df_sent[DOC_ID_COL].isin(ids_va)].reset_index(drop=True)
        df_te_sent_g = df_sent[df_sent[DOC_ID_COL].isin(ids_te)].reset_index(drop=True)

        print("\n[GoldFA] Building sentence-level multi-hot vectors from human annotations...")
        tr_gold = gold_multihot_sentences(df_tr_sent_g)
        va_gold = gold_multihot_sentences(df_va_sent_g)
        te_gold = gold_multihot_sentences(df_te_sent_g)

        tr_gold.to_csv(os.path.join(OUT_DIR, "goldfa_sentence_multihot_train.csv"), index=False)
        va_gold.to_csv(os.path.join(OUT_DIR, "goldfa_sentence_multihot_val.csv"), index=False)
        te_gold.to_csv(os.path.join(OUT_DIR, "goldfa_sentence_multihot_test.csv"), index=False)

        print("\n[GoldFA] Aggregating to HSPT-13 doc-level features (oracle, no tau)...")
        Fg_tr = aggregate_goldfa_hspt13(tr_gold)
        Fg_va = aggregate_goldfa_hspt13(va_gold)
        Fg_te = aggregate_goldfa_hspt13(te_gold)

        Xgf_tr = align_predfa_to_split(Fg_tr, df_doc, tr_idx, split_name="train_goldfa")
        Xgf_va = align_predfa_to_split(Fg_va, df_doc, va_idx, split_name="val_goldfa")
        Xgf_te = align_predfa_to_split(Fg_te, df_doc, te_idx, split_name="test_goldfa")
        Xgf_tr_z, Xgf_va_z, Xgf_te_z, gf_mu, gf_sd = zscore_fit_transform(Xgf_tr, Xgf_va, Xgf_te)

        gf_dir = os.path.join(OUT_DIR, "goldfa_features")
        ensure_dir(gf_dir)
        pd.DataFrame({"feature": SUM13_ORDER, "mu": gf_mu.tolist(), "sd": gf_sd.tolist()}).to_csv(
            os.path.join(gf_dir, "hspt13_zscore.csv"), index=False)

        # Pol (15) + Sent (4) + GoldFA (13) = 32 dims
        Xtr_oracle = np.concatenate([Pol_tr_z, Sent_tr_z, Xgf_tr_z], axis=1)
        Xva_oracle = np.concatenate([Pol_va_z, Sent_va_z, Xgf_va_z], axis=1)
        Xte_oracle = np.concatenate([Pol_te_z, Sent_te_z, Xgf_te_z], axis=1)

        for multi_task in ([True] if DO_MULTI_TASK else []) + ([False] if DO_SINGLE_TASK else []):
            family = f"llama_plus_pol_sent_goldfa_{'mt' if multi_task else 'st'}"
            _set_ckpt_family(family)
            print(f"\n[GRID] {family}")
            for run_id, cfg in enumerate(iter_grid(LLAMA_GRID), start=1):
                print(f"[RUN] {family} #{run_id}: {cfg}")
                Yv, Pv, Yt, Pt = train_llama_once(df_doc, tr_idx, va_idx, te_idx, cfg,
                                                   fusion_feats=(Xtr_oracle, Xva_oracle, Xte_oracle), multi_task=multi_task)
                run_dir = os.path.join(OUT_DIR, family, f"run_{run_id:03d}")
                cfg_save = _row_with_metadata({**cfg, "family": family,
                                               "feat_set": "text+pol+sent+goldfa",
                                               "feat_dim": Xtr_oracle.shape[1], "tau": None})
                metrics = save_run(run_dir, cfg_save, Yv, Pv, Yt, Pt)
                row = _row_with_metadata({**cfg, "family": family, "run_id": run_id,
                                          "feat_set": "text+pol+sent+goldfa",
                                          "feat_dim": Xtr_oracle.shape[1], "tau": None})
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
            "lr", "epochs", "batch_size", "dropout", "hidden_dim", "lora_r", "lora_alpha", "run_dir"]
    cols = [c for c in cols if c in best_df.columns]
    if len(best_df):
        print(best_df[cols].to_string(index=False))

    print(f"\n[DONE] Grid -> {grid_path}")
    print(f"[DONE] Best by family -> {best_path}")
    print(f"[DONE] Output dir -> {OUT_DIR}")


if __name__ == "__main__":
    main()
