#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OPR_LLAMA_PredFA_gridsearch.py
==============================
aggregation:
python - <<'PY'
import os, glob, json
import pandas as pd

out_dir = "outputs/opr_sentence_only_llama_full_multiseed_summary"
os.makedirs(out_dir, exist_ok=True)

paths = sorted(glob.glob("outputs/opr_sentence_only_llama_text_only_opr_runseed*/opr_best_by_family.csv"))
paths += sorted(glob.glob("outputs/opr_sentence_only_llama_predfa_fac_runseed*_opr_runseed*/opr_best_by_family.csv"))

rows = []
for p in paths:
    df = pd.read_csv(p)
    df["source_file"] = p
    rows.append(df)

all_df = pd.concat(rows, ignore_index=True)
all_path = os.path.join(out_dir, "llama_opr_all_best_by_family.csv")
all_df.to_csv(all_path, index=False)

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

for col in ["val_rho_macro", "val_mae_macro", "test_rho_macro", "test_mae_macro"]:
    summary[f"{col}_mean_std"] = summary.apply(
        lambda r: f"{r[f'{col}_mean']:.3f} ± {r[f'{col}_std']:.3f}",
        axis=1
    )

summary_path = os.path.join(out_dir, "llama_opr_summary_mean_std.csv")
summary.to_csv(summary_path, index=False)

target_map = {
    "Directness": "Directness_vs_Indirectness__GOLD",
    "Positive Face": "Structural_Politeness_and_Politeness_Markers__GOLD",
    "Negative Face": "Tone_and_Overall_Consideration__GOLD",
}

per_seed_rows = []
for _, r in all_df.iterrows():
    metrics_path = os.path.join(str(r["run_dir"]), "metrics.json")
    if not os.path.isfile(metrics_path):
        continue
    with open(metrics_path) as f:
        m = json.load(f)

    test = m["test"]["per_target"]
    row = {
        "family": r["family"],
        "opr_run_seed": r.get("opr_run_seed", ""),
        "fac_run_seed": r.get("fac_run_seed", ""),
        "test_rho_macro": m["test"]["rho_macro"],
        "test_mae_macro": m["test"]["mae_macro"],
    }

    for short, full in target_map.items():
        row[f"{short}_rho"] = test[full]["Spearman"]
        row[f"{short}_mae"] = test[full]["MAE"]

    per_seed_rows.append(row)

per_df = pd.DataFrame(per_seed_rows)
per_path = os.path.join(out_dir, "llama_opr_per_seed_per_dimension.csv")
per_df.to_csv(per_path, index=False)

def ms(g, col):
    return f"{g[col].mean():.3f} ± {g[col].std(ddof=1):.3f}"

table_rows = []
for fam, g in per_df.groupby("family"):
    table_rows.append({
        "Model": fam,
        "Test ρ Macro": ms(g, "test_rho_macro"),
        "Test MAE Macro": ms(g, "test_mae_macro"),
        "Directness ρ": ms(g, "Directness_rho"),
        "Directness MAE": ms(g, "Directness_mae"),
        "Positive Face ρ": ms(g, "Positive Face_rho"),
        "Positive Face MAE": ms(g, "Positive Face_mae"),
        "Negative Face ρ": ms(g, "Negative Face_rho"),
        "Negative Face MAE": ms(g, "Negative Face_mae"),
    })

table_df = pd.DataFrame(table_rows)
table_path = os.path.join(out_dir, "llama_opr_table_rows_per_dimension.csv")
table_df.to_csv(table_path, index=False)

print("\n=== MACRO SUMMARY ===")
print(summary[[
    "family", "n_runs",
    "val_rho_macro_mean_std", "val_mae_macro_mean_std",
    "test_rho_macro_mean_std", "test_mae_macro_mean_std"
]].to_string(index=False))

print("\n=== TABLE ROWS ===")
print(table_df.to_string(index=False))

print("\n[DONE] wrote:")
print(all_path)
print(summary_path)
print(per_path)
print(table_path)
PY
--------------------------
to run:

export HF_TOKEN=<your_hf_token>   # Llama 3.2 is gated: accept the licence on Hugging Face first
export CUDA_VISIBLE_DEVICES=0

# Paper grid (Supplementary Table II): lr 2e-4, epochs {5, 8, 12}, LoRA r {8, 16, 32}, alpha = 2r
export EXTENDED_LORA=1
export EXTLORA_LRS=2e-4

export FAC_REGISTRY=outputs/llama_lora_sentence_only_multiseed/llama_sentence_only_best_checkpoints.csv

nohup bash -lc '
echo "=============================="
echo "START LLaMA OPR FULL PIPELINE"
echo "=============================="

echo ""
echo "### STEP 1: LLaMA TEXT-ONLY OPR ###"

for s in 13 21 42 87 100; do
  echo "===== LLaMA text-only OPR runseed$s ====="

  OPR_RUN_SEED="$s" \
  SPLIT_RANDOM_STATE=42 \
  OUT_DIR="outputs/opr_sentence_only_llama_text_only_opr_runseed$s" \
  RUN_LLAMA_TEXT=1 \
  RUN_PREDFA_MLP=0 \
  RUN_LLAMA_PLUS_PREDFA=0 \
  python -u OPR_LLAMA_PredFA_gridsearch.py

  echo "===== DONE LLaMA text-only OPR runseed$s ====="
done

echo ""
echo "### STEP 2: LLaMA PredFA-only + LLaMA+PredFA OPR ###"

for s in 13 21 42 87 100; do
  echo "===== LLaMA PredFA OPR FAC seed$s + OPR seed$s ====="

  FAC_RUN_SEED="$s" \
  OPR_RUN_SEED="$s" \
  SPLIT_RANDOM_STATE=42 \
  FAC_REGISTRY_CSV="$FAC_REGISTRY" \
  OUT_DIR="outputs/opr_sentence_only_llama_predfa_fac_runseed${s}_opr_runseed${s}" \
  RUN_LLAMA_TEXT=0 \
  RUN_PREDFA_MLP=1 \
  RUN_LLAMA_PLUS_PREDFA=1 \
  python -u OPR_LLAMA_PredFA_gridsearch.py

  echo "===== DONE LLaMA PredFA OPR seed$s ====="
done

echo ""
echo "=============================="
echo "DONE LLaMA OPR FULL PIPELINE"
echo "=============================="
' > opr_llama_full_text_and_predfa_pipeline.log 2>&1 &

tail -f opr_llama_full_text_and_predfa_pipeline.log




LLaMA version of the OPR pipeline.

What's LLaMA-ified vs the BERT version:
  • Doc-level regressor: Llama-3.2-1B + LoRA, mean/last-token pooled head,
    3 regression outputs (SmoothL1).
  • FAC sentence-level inference: Llama-3.2-1B + LoRA adapter from your
    FAC sentence-only registry CSV. One LoRA adapter is selected per run
    via FAC_RUN_SEED (paired with OPR_RUN_SEED in the shell loop).

Pooling: last non-pad token's last hidden state.
LoRA targets: q_proj, k_proj, v_proj, o_proj (Llama attention projections).
"""

# ============================================================
# USER PATHS
# ============================================================
DOC_CSV = "data/corpus/email_text_gold_three_dimensions_politeness_score_with_seed_correct.csv"
SENT_CSV = "data/corpus/sentences_with_golden_face_act.csv"

import os

# FAC LoRA registry — one row per seed with path to LoRA adapter dir
FAC_REGISTRY_CSV = os.environ.get(
    "FAC_REGISTRY_CSV",
    "outputs/llama_lora_sentence_only_multiseed/llama_sentence_only_best_checkpoints.csv"
)

# Base LLaMA model (LoRA adapters load on top of this)
LLAMA_BASE_MODEL = os.environ.get("LLAMA_BASE_MODEL", "meta-llama/Llama-3.2-1B")

OUT_DIR = os.environ.get(
    "OUT_DIR",
    "outputs/opr_sentence_only_llama_predfa_gridsearch"
)

# ============================================================
# IMPORTS
# ============================================================
import ast, glob, json, math, random, itertools
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
    print("[WARN] peft not installed. LoRA features will fail.")

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

LLAMA_MAXLEN_DOC = 512
LLAMA_MAXLEN_SENT = 128
FAC_BATCH = 32        # smaller than BERT (LLaMA is bigger)

# Which systems to run
RUN_LLAMA_TEXT = os.environ.get("RUN_LLAMA_TEXT", "1") == "1"
RUN_PREDFA_MLP = os.environ.get("RUN_PREDFA_MLP", "1") == "1"
RUN_LLAMA_PLUS_PREDFA = os.environ.get("RUN_LLAMA_PLUS_PREDFA", "1") == "1"
DO_MULTI_TASK = os.environ.get("DO_MULTI_TASK", "1") == "1"
DO_SINGLE_TASK = os.environ.get("DO_SINGLE_TASK", "1") == "1"

# FAC threshold learned on validation
GRID_PREDFA_MODES = ["hard"]
FAC_TAU_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2).tolist()

# OPR GRID (LLaMA-tuned: smaller batches than BERT to fit memory; LoRA lr)
# ============================================================
# LoRA search controls
#   EXTENDED_LORA=1   -> widen the grid: r in {8,16,32} (alpha tied = 2r),
#                        longer training {5,8,12} epochs, lr up to 3e-4
#   LORA_TARGETS      -> "attn" (paper OPR: q,k,v,o) | "attn_mlp" (FAC-style:
#                        + gate,up,down projections)
#   LLAMA_DTYPE       -> "fp16" (paper) | "bf16" (safer numerics if supported)
#   LOG_CURVES=1      -> save per-epoch train/val loss + Spearman curves
#                        (CSV + PNG) under OUT_DIR/learning_curves/
# Defaults give the earlier 4-config grid (r = 8); the paper uses
# EXTENDED_LORA=1 EXTLORA_LRS=2e-4 (9 configs).
# ============================================================
EXTENDED_LORA = os.environ.get("EXTENDED_LORA", "0") == "1"
LORA_TARGETS  = os.environ.get("LORA_TARGETS", "attn")
LLAMA_DTYPE   = os.environ.get("LLAMA_DTYPE", "fp16")
LOG_CURVES    = os.environ.get("LOG_CURVES", "1") == "1"

if EXTENDED_LORA:
    # lr list overridable: EXTLORA_LRS="2e-4" fixes lr (stage-2: 9 configs total)
    _lrs = [float(x) for x in os.environ.get("EXTLORA_LRS", "1e-4,2e-4,3e-4").split(",")]
    _eps = [int(x) for x in os.environ.get("EXTLORA_EPOCHS", "5,8,12").split(",")]
    _rs  = [int(x) for x in os.environ.get("EXTLORA_RANKS", "8,16,32").split(",")]
    LLAMA_GRID = {
        "lr": _lrs,
        "epochs": _eps,     # LoRA needs more steps than full FT on ~330 docs
        "batch_size": [4],
        "dropout": [0.1],
        "weight_decay": [0.01],
        "hidden_dim": [128],
        "lora_r": _rs,      # rank is a hyperparameter
        "lora_alpha": ["2r"],     # tied: alpha = 2 * r (same ratio as FAC r=16/a=32)
        "lora_dropout": [0.05],
    }
    print(f"[EXTLORA] grid: lr={_lrs} epochs={_eps} r={_rs} alpha=2r targets={LORA_TARGETS} "
          f"-> {len(_lrs)*len(_eps)*len(_rs)} configs")
else:
    LLAMA_GRID = {
        "lr": [1e-4, 2e-4],          # LoRA tolerates higher lr than full fine-tune
        "epochs": [3, 5],
        "batch_size": [4],            # smaller for LLaMA 1B
        "dropout": [0.1],
        "weight_decay": [0.01],
        "hidden_dim": [128],
        "lora_r": [8],
        "lora_alpha": [16],
        "lora_dropout": [0.1],
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


def grouped_split_70_15_15_index(groups: np.ndarray, random_state=42):
    idx = np.arange(len(groups))
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=random_state)
    tr, tmp = next(gss1.split(idx, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=random_state)
    va_rel, te_rel = next(gss2.split(tmp, groups=groups[tmp]))
    va, te = tmp[va_rel], tmp[te_rel]
    return tr, va, te


def iter_grid(grid: Dict) -> List[Dict]:
    keys = list(grid.keys())
    cfgs = [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]
    # resolve tied lora_alpha = 2 * lora_r
    for c in cfgs:
        if c.get("lora_alpha") == "2r":
            c["lora_alpha"] = 2 * int(c["lora_r"])
    return cfgs


# ============================================================
# LEARNING-CURVE LOGGING (convergence diagnostics)
# ============================================================
def _curve_dir():
    d = os.path.join(OUT_DIR, "learning_curves")
    os.makedirs(d, exist_ok=True)
    return d


def _smoothl1_np(y, p) -> float:
    t = torch.from_numpy(np.asarray(y, dtype=np.float32))
    q = torch.from_numpy(np.asarray(p, dtype=np.float32))
    return float(nn.SmoothL1Loss()(q, t).item())


def _save_curve(history_rows: List[Dict], tag: str):
    """Save one run's per-epoch history as CSV + PNG under OUT_DIR/learning_curves/."""
    if not LOG_CURVES or not history_rows:
        return
    import csv as _csv
    d = _curve_dir()
    csv_path = os.path.join(d, f"{tag}.csv")
    with open(csv_path, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(history_rows[0].keys()))
        w.writeheader()
        for r in history_rows:
            w.writerow(r)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep = [r["epoch"] for r in history_rows]
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(ep, [r["train_loss"] for r in history_rows], marker="o", label="train loss")
        ax[0].plot(ep, [r["val_loss"] for r in history_rows], marker="o", label="val loss")
        ax[0].set_xlabel("epoch"); ax[0].set_ylabel("SmoothL1"); ax[0].legend()
        ax[0].set_title("Loss")
        ax[1].plot(ep, [r["train_rho"] for r in history_rows], marker="o", label="train Spearman")
        ax[1].plot(ep, [r["val_rho"] for r in history_rows], marker="o", label="val Spearman")
        ax[1].set_xlabel("epoch"); ax[1].set_ylabel("Spearman rho"); ax[1].legend()
        ax[1].set_title("Rank correlation")
        fig.suptitle(tag, fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(d, f"{tag}.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"[CURVE] plot skipped for {tag} ({e})")


def parse_gold_faceacts(x) -> List[str]:
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
# FAC LoRA registry lookup
# ============================================================
def resolve_fac_lora_adapter(registry_csv: str, fac_run_seed: int) -> str:
    """
    Look up the LoRA adapter directory for the given FAC seed.
    Tolerant about column naming:
      - path columns: adapter_dir, best_checkpoint_dir, checkpoint_dir, path, model_dir
      - seed columns: run_seed, fac_run_seed, seed
    """
    if not os.path.isfile(registry_csv):
        raise FileNotFoundError(f"[FAC] Registry CSV not found: {registry_csv}")

    df = pd.read_csv(registry_csv)

    seed_col_candidates = ["run_seed", "fac_run_seed", "seed", "RUN_SEED"]
    path_col_candidates = ["adapter_dir", "best_checkpoint_dir", "checkpoint_dir", "path", "model_dir", "adapter_path"]

    seed_col = next((c for c in seed_col_candidates if c in df.columns), None)
    path_col = next((c for c in path_col_candidates if c in df.columns), None)

    if seed_col is None:
        raise ValueError(
            f"[FAC] Registry CSV has no seed column. Tried {seed_col_candidates}. "
            f"Columns present: {list(df.columns)}"
        )
    if path_col is None:
        raise ValueError(
            f"[FAC] Registry CSV has no path column. Tried {path_col_candidates}. "
            f"Columns present: {list(df.columns)}"
        )

    df[seed_col] = pd.to_numeric(df[seed_col], errors="coerce").astype("Int64")
    rows = df[df[seed_col] == int(fac_run_seed)]
    if len(rows) == 0:
        raise ValueError(
            f"[FAC] No row in registry for seed={fac_run_seed}. "
            f"Available seeds: {sorted(df[seed_col].dropna().unique().tolist())}"
        )
    if len(rows) > 1:
        print(f"[FAC] WARN: {len(rows)} rows for seed={fac_run_seed} in registry; using first.")
    adapter_dir = str(rows.iloc[0][path_col])
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(
            f"[FAC] Registry says adapter for seed={fac_run_seed} is at '{adapter_dir}' "
            f"but that path is not a directory."
        )
    print(f"[FAC] Resolved seed={fac_run_seed} -> LoRA adapter: {adapter_dir}")
    return adapter_dir


# ============================================================
# Last non-pad token pooling
# ============================================================
def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Pool by last non-pad token. Works for any padding side.
    """
    seq_lengths = attention_mask.sum(dim=1) - 1   # index of last real token, per row
    batch_idx = torch.arange(last_hidden_states.size(0), device=last_hidden_states.device)
    return last_hidden_states[batch_idx, seq_lengths]


# ============================================================
# FAC SENTENCE-LEVEL INFERENCE (LLaMA + LoRA adapter)
# ============================================================
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


def _llama_tok_with_pad():
    """Tokenizer with pad_token set."""
    tok = AutoTokenizer.from_pretrained(LLAMA_BASE_MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def predict_fa_probs_llama(df_split: pd.DataFrame, adapter_dir: str, batch=32, max_len=128) -> pd.DataFrame:
    if len(df_split) == 0:
        return pd.DataFrame(columns=[DOC_ID_COL, SENT_IDX_COL] + [f"prob_{l}" for l in FA_LABS])

    if not PEFT_AVAILABLE:
        raise RuntimeError("peft library required for LoRA adapter loading; pip install peft")

    print(f"[FAC] Loading LLaMA base + LoRA adapter: {adapter_dir}")
    tok = _llama_tok_with_pad()

    try:
        base = AutoModelForSequenceClassification.from_pretrained(
            LLAMA_BASE_MODEL,
            num_labels=9,
            problem_type="multi_label_classification",
            torch_dtype=torch.float16,
        )
        base.config.pad_token_id = tok.pad_token_id
        mdl = PeftModel.from_pretrained(base, adapter_dir)
        mdl = mdl.to(DEVICE)
    except Exception as e:
        raise OSError(
            f"[FAC] Could not load LLaMA + LoRA adapter from '{adapter_dir}'. "
            f"Original error: {e}"
        ) from e
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
            P = torch.sigmoid(logits.float()).cpu().numpy()
            allP.append(P)
            doc_ids.extend(b["doc_id"])
            sids.extend(b["sent_idx"])

    # Free GPU memory before downstream OPR training
    del mdl, base
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    P = np.vstack(allP)
    out = {DOC_ID_COL: list(map(str, doc_ids)), SENT_IDX_COL: list(map(int, sids))}
    for j, l in enumerate(FA_LABS):
        out[f"prob_{l}"] = P[:, j]
    dfp = pd.DataFrame(out)
    dfp[DOC_ID_COL] = dfp[DOC_ID_COL].astype(str)
    dfp[SENT_IDX_COL] = pd.to_numeric(dfp[SENT_IDX_COL], errors="coerce").fillna(0).astype(int)
    return dfp


def tune_fac_tau_on_validation(val_pred: pd.DataFrame, val_sent: pd.DataFrame) -> Dict:
    val_pred = val_pred.sort_values([DOC_ID_COL, SENT_IDX_COL], kind="mergesort").reset_index(drop=True)
    val_sent = val_sent.copy()
    val_sent[DOC_ID_COL] = val_sent[DOC_ID_COL].astype(str)
    val_sent[SENT_IDX_COL] = pd.to_numeric(val_sent[SENT_IDX_COL], errors="coerce").fillna(0).astype(int)
    val_sent = val_sent.sort_values([DOC_ID_COL, SENT_IDX_COL], kind="mergesort").reset_index(drop=True)

    assert len(val_pred) == len(val_sent), "val_pred and val_sent do not align by length."
    assert (val_pred[DOC_ID_COL].tolist() == val_sent[DOC_ID_COL].tolist()), "doc_id order mismatch."
    assert (val_pred[SENT_IDX_COL].tolist() == val_sent[SENT_IDX_COL].tolist()), "sentence_idx order mismatch."

    y_true = gold_multihot(val_sent)
    probs = val_pred[[f"prob_{l}" for l in FA_LABS]].to_numpy(dtype=np.float32)

    rows = []
    best = None
    for tau in FAC_TAU_GRID:
        y_hat = (probs >= tau).astype(np.int32)
        micro_f1 = f1_score(y_true, y_hat, average="micro", zero_division=0)
        micro_p = precision_score(y_true, y_hat, average="micro", zero_division=0)
        micro_r = recall_score(y_true, y_hat, average="micro", zero_division=0)
        row = {"tau": float(tau), "micro_f1": float(micro_f1),
               "micro_precision": float(micro_p), "micro_recall": float(micro_r)}
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
    want_doc_ids = df_doc.loc[idx, DOC_ID_COL].astype(str).tolist()
    F_pred_indexed = F_pred.copy()
    F_pred_indexed[DOC_ID_COL] = F_pred_indexed[DOC_ID_COL].astype(str)
    F_pred_indexed = F_pred_indexed.drop_duplicates(subset=[DOC_ID_COL], keep="first")
    F_pred_indexed = F_pred_indexed.set_index(DOC_ID_COL)

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

    F_aligned = F_pred_indexed.reindex(want_doc_ids)
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
class DocLlamaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, text_col: str, targets: List[str], tokenizer, max_length=512):
        enc = tokenizer(
            df[text_col].astype(str).tolist(),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.ids = enc["input_ids"]
        self.attn = enc["attention_mask"]
        self.y = torch.tensor(df[targets].to_numpy(np.float32))

    def __len__(self):
        return self.ids.size(0)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "attention_mask": self.attn[i], "targets": self.y[i]}


class DocLlamaFusionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, text_col: str, targets: List[str], tokenizer, predfa: np.ndarray, max_length=512):
        assert len(df) == len(predfa), "PredFA rows must align with doc rows."
        enc = tokenizer(
            df[text_col].astype(str).tolist(),
            padding=True,
            truncation=True,
            max_length=max_length,
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
    def __len__(self): return self.X.size(0)
    def __getitem__(self, i): return {"x": self.X[i], "targets": self.Y[i]}


class LlamaDocRegressor(nn.Module):
    """
    Llama-3.2-1B + LoRA → last-non-pad-token pool → optional fusion w/ PredFA → MLP → n_targets.
    """
    def __init__(self, base_model_name: str, n_targets: int = 3, fusion_dim: int = 0,
                 hidden_dim: int = 128, dropout: float = 0.1,
                 lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1,
                 pad_token_id: int = None):
        super().__init__()
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft library required for LoRA; pip install peft")

        config = AutoConfig.from_pretrained(base_model_name)
        if pad_token_id is not None:
            config.pad_token_id = pad_token_id
        # Load base in fp16 (paper default) or bf16 via LLAMA_DTYPE; LoRA layers fp32 (peft default)
        _dtype = torch.float16
        if LLAMA_DTYPE == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            _dtype = torch.bfloat16
        base = AutoModel.from_pretrained(base_model_name, config=config, torch_dtype=_dtype)

        # "attn" = paper OPR setting; "attn_mlp" = FAC-style, adds MLP projections
        _tm = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if LORA_TARGETS == "attn_mlp":
            _tm += ["gate_proj", "up_proj", "down_proj"]
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=_tm,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.enc = get_peft_model(base, lora_cfg)
        self.enc.print_trainable_parameters()

        hdim = base.config.hidden_size
        self.fusion_dim = int(fusion_dim)
        in_dim = hdim + self.fusion_dim
        # Heads in fp32 for stability
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_targets)])

    def forward(self, input_ids, attention_mask, predfa: Optional[torch.Tensor] = None):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = last_token_pool(out.last_hidden_state, attention_mask)
        pooled = pooled.float()  # cast to fp32 for the head
        if self.fusion_dim > 0:
            if predfa is None:
                raise ValueError("predfa is required when fusion_dim > 0")
            z = torch.cat([pooled, predfa.float()], dim=-1)
        else:
            z = pooled
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
    def forward(self, x): return self.net(x)


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


def predict_llama(model, dl):
    model.eval()
    Ys, Ps = [], []
    with torch.no_grad():
        for b in dl:
            pf = b.get("predfa", None)
            pf = pf.to(DEVICE) if pf is not None else None
            pr = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), pf)
            Ys.append(b["targets"].numpy())
            Ps.append(pr.float().cpu().numpy())
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


def _cleanup_model(model):
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_llama_once(df_doc, tr_idx, va_idx, te_idx, cfg: Dict, predfa_feats=None, multi_task=True):
    set_seed(RANDOM_STATE)
    tk = _llama_tok_with_pad()
    df_tr = df_doc.loc[tr_idx].reset_index(drop=True)
    df_va = df_doc.loc[va_idx].reset_index(drop=True)
    df_te = df_doc.loc[te_idx].reset_index(drop=True)

    if predfa_feats is None:
        ds_tr = DocLlamaDataset(df_tr, DOC_TEXT_COL, TARGETS, tk, max_length=LLAMA_MAXLEN_DOC)
        ds_va = DocLlamaDataset(df_va, DOC_TEXT_COL, TARGETS, tk, max_length=LLAMA_MAXLEN_DOC)
        ds_te = DocLlamaDataset(df_te, DOC_TEXT_COL, TARGETS, tk, max_length=LLAMA_MAXLEN_DOC)
        fusion_dim = 0
    else:
        Xtr, Xva, Xte = predfa_feats
        ds_tr = DocLlamaFusionDataset(df_tr, DOC_TEXT_COL, TARGETS, tk, Xtr, max_length=LLAMA_MAXLEN_DOC)
        ds_va = DocLlamaFusionDataset(df_va, DOC_TEXT_COL, TARGETS, tk, Xva, max_length=LLAMA_MAXLEN_DOC)
        ds_te = DocLlamaFusionDataset(df_te, DOC_TEXT_COL, TARGETS, tk, Xte, max_length=LLAMA_MAXLEN_DOC)
        fusion_dim = Xtr.shape[1]

    g = make_dl_generator(RANDOM_STATE)
    dl_tr = DataLoader(ds_tr, batch_size=cfg["batch_size"], shuffle=True, generator=g)
    dl_va = DataLoader(ds_va, batch_size=cfg["batch_size"], shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=cfg["batch_size"], shuffle=False)
    # deterministic train loader for learning-curve evaluation
    dl_tr_eval = DataLoader(ds_tr, batch_size=cfg["batch_size"], shuffle=False)

    if multi_task:
        model = LlamaDocRegressor(
            base_model_name=LLAMA_BASE_MODEL,
            n_targets=3,
            fusion_dim=fusion_dim,
            hidden_dim=cfg["hidden_dim"],
            dropout=cfg["dropout"],
            lora_r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            pad_token_id=tk.pad_token_id,
        ).to(DEVICE)

        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        )
        steps_total = max(1, len(dl_tr) * cfg["epochs"])
        sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps_total), steps_total)
        loss_fn = nn.SmoothL1Loss()

        # learning-curve bookkeeping
        curve_tag = (f"llama_mt_lr{cfg['lr']}_ep{cfg['epochs']}_bs{cfg['batch_size']}"
                     f"_r{cfg['lora_r']}_a{cfg['lora_alpha']}_ld{cfg['lora_dropout']}"
                     f"_tm-{LORA_TARGETS}_fus{fusion_dim}_seed{RANDOM_STATE}")
        history = []
        best_score, best_state = -1e9, None
        for ep_i in range(cfg["epochs"]):
            model.train()
            ep_losses = []
            for b in dl_tr:
                opt.zero_grad()
                pf = b.get("predfa", None)
                pf = pf.to(DEVICE) if pf is not None else None
                pr = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), pf)
                loss = loss_fn(pr, b["targets"].to(DEVICE))
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sch.step()
                ep_losses.append(float(loss.item()))
            Yv, Pv = predict_llama(model, dl_va)
            mv = mae_spearman(Yv, Pv, TARGETS)
            score = mv["rho_macro"]
            if LOG_CURVES:
                Ytr_e, Ptr_e = predict_llama(model, dl_tr_eval)
                mt_ = mae_spearman(Ytr_e, Ptr_e, TARGETS)
                history.append({
                    "epoch": ep_i + 1,
                    "train_loss": float(np.mean(ep_losses)) if ep_losses else float("nan"),
                    "val_loss": _smoothl1_np(Yv, Pv),
                    "train_rho": mt_["rho_macro"],
                    "val_rho": mv["rho_macro"],
                    "train_mae": mt_["mae_macro"],
                    "val_mae": mv["mae_macro"],
                    "lr_now": float(sch.get_last_lr()[0]),
                })
            if score > best_score:
                best_score = score
                # Only save trainable params (LoRA + heads) for memory
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if v.requires_grad or "head" in k or "hidden" in k}
        _save_curve(history, curve_tag)

        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        # load_state_dict with strict=False because we only stored trainable + head params
        model.load_state_dict(best_state, strict=False)
        Yva, Pva = predict_llama(model, dl_va)
        Yte, Pte = predict_llama(model, dl_te)
        _cleanup_model(model)
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

        model = LlamaDocRegressor(
            base_model_name=LLAMA_BASE_MODEL,
            n_targets=1,
            fusion_dim=fusion_dim,
            hidden_dim=cfg["hidden_dim"],
            dropout=cfg["dropout"],
            lora_r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            pad_token_id=tk.pad_token_id,
        ).to(DEVICE)

        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        )
        steps_total = max(1, len(dl_tr) * cfg["epochs"])
        sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps_total), steps_total)
        loss_fn = nn.SmoothL1Loss()
        best_score, best_state = -1e9, None
        # learning-curve bookkeeping (one curve per target dimension)
        curve_tag = (f"llama_st_t{t_i}_{TARGETS[t_i].replace(' ', '')[:12]}"
                     f"_lr{cfg['lr']}_ep{cfg['epochs']}_bs{cfg['batch_size']}"
                     f"_r{cfg['lora_r']}_a{cfg['lora_alpha']}_ld{cfg['lora_dropout']}"
                     f"_tm-{LORA_TARGETS}_fus{fusion_dim}_seed{RANDOM_STATE + t_i}")
        history = []

        for ep_i in range(cfg["epochs"]):
            model.train()
            ep_losses = []
            for b in dl_tr:
                opt.zero_grad()
                pf = b.get("predfa", None)
                pf = pf.to(DEVICE) if pf is not None else None
                y = b["targets"][:, t_i].unsqueeze(1).to(DEVICE)
                pr = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), pf)
                loss = loss_fn(pr, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sch.step()
                ep_losses.append(float(loss.item()))

            model.eval()
            yv, pv = [], []
            with torch.no_grad():
                for b in dl_va:
                    pf = b.get("predfa", None)
                    pf = pf.to(DEVICE) if pf is not None else None
                    pr = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), pf)
                    yv.append(b["targets"][:, t_i].numpy())
                    pv.append(pr.float().cpu().numpy().squeeze(1))
            yv = np.concatenate(yv, 0)
            pv = np.concatenate(pv, 0)
            rho, _ = spearmanr(yv, pv)
            score = float(rho) if rho is not None and not np.isnan(rho) else 0.0
            if LOG_CURVES:
                ytr, ptr = [], []
                with torch.no_grad():
                    for b in dl_tr_eval:
                        pf = b.get("predfa", None)
                        pf = pf.to(DEVICE) if pf is not None else None
                        pr = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), pf)
                        ytr.append(b["targets"][:, t_i].numpy())
                        ptr.append(pr.float().cpu().numpy().squeeze(1))
                ytr = np.concatenate(ytr, 0)
                ptr = np.concatenate(ptr, 0)
                rtr, _ = spearmanr(ytr, ptr)
                rtr = float(rtr) if rtr is not None and not np.isnan(rtr) else 0.0
                history.append({
                    "epoch": ep_i + 1,
                    "train_loss": float(np.mean(ep_losses)) if ep_losses else float("nan"),
                    "val_loss": _smoothl1_np(yv.reshape(-1, 1), pv.reshape(-1, 1)),
                    "train_rho": rtr,
                    "val_rho": score,
                    "train_mae": float(np.mean(np.abs(ytr - ptr))),
                    "val_mae": float(np.mean(np.abs(yv - pv))),
                    "lr_now": float(sch.get_last_lr()[0]),
                })
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if v.requires_grad or "head" in k or "hidden" in k}
        _save_curve(history, curve_tag)

        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state, strict=False)
        model.eval()
        for dl, store in [(dl_va, Pva_all), (dl_te, Pte_all)]:
            preds = []
            with torch.no_grad():
                for b in dl:
                    pf = b.get("predfa", None)
                    pf = pf.to(DEVICE) if pf is not None else None
                    pr = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), pf)
                    preds.append(pr.float().cpu().numpy().squeeze(1))
            store[:, t_i] = np.concatenate(preds, 0)
        _cleanup_model(model)

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
def _run_metadata(adapter_dir: str = "") -> Dict:
    return {
        "opr_run_seed": int(OPR_RUN_SEED),
        "split_random_state": int(SPLIT_RANDOM_STATE),
        "fac_run_seed": os.environ.get("FAC_RUN_SEED", ""),
        "fac_adapter_dir": adapter_dir,
        "llama_base_model": LLAMA_BASE_MODEL,
    }


def _row_with_metadata(row: Dict, adapter_dir: str = "") -> Dict:
    return {**row, **_run_metadata(adapter_dir)}


def main():
    set_seed(OPR_RUN_SEED)
    ensure_dir(OUT_DIR)

    need_predfa = RUN_PREDFA_MLP or RUN_LLAMA_PLUS_PREDFA

    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Output dir: {OUT_DIR}")
    print(f"[INFO] LLaMA base: {LLAMA_BASE_MODEL}")
    print(f"[INFO] Split random state: {SPLIT_RANDOM_STATE}")
    print(f"[INFO] OPR run seed: {OPR_RUN_SEED}")
    print(f"[INFO] FAC run seed: {os.environ.get('FAC_RUN_SEED', '')}")
    print(f"[INFO] FAC registry: {FAC_REGISTRY_CSV if need_predfa else '[not used: PredFA disabled]'}")
    print(f"[INFO] RUN_LLAMA_TEXT={RUN_LLAMA_TEXT} RUN_PREDFA_MLP={RUN_PREDFA_MLP} RUN_LLAMA_PLUS_PREDFA={RUN_LLAMA_PLUS_PREDFA}")

    # Resolve FAC LoRA adapter if needed
    fac_adapter_dir = ""
    if need_predfa:
        fac_seed_env = os.environ.get("FAC_RUN_SEED", "")
        if not fac_seed_env:
            raise RuntimeError("FAC_RUN_SEED env var required when PredFA systems are enabled.")
        fac_adapter_dir = resolve_fac_lora_adapter(FAC_REGISTRY_CSV, int(fac_seed_env))

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
    if len(df_doc) < pre_n:
        print(f"[INFO] Dropped {pre_n - len(df_doc)} rows with non-numeric targets.")

    df_doc[DOC_ID_COL] = df_doc[DOC_ID_COL].astype(str)
    df_sent[DOC_ID_COL] = df_sent[DOC_ID_COL].astype(str)

    dup_count = df_doc[DOC_ID_COL].duplicated().sum()
    if dup_count:
        print(f"[WARN] {dup_count} duplicate {DOC_ID_COL} in df_doc; keeping first.")
        df_doc = df_doc.drop_duplicates(subset=[DOC_ID_COL], keep="first").reset_index(drop=True)

    df_doc = df_doc.sort_values(DOC_ID_COL, kind="mergesort").reset_index(drop=True)

    if df_doc[SEED_COL].isna().any():
        n_bad = int(df_doc[SEED_COL].isna().sum())
        print(f"[WARN] {n_bad} rows have missing seed; dropping.")
        df_doc = df_doc.dropna(subset=[SEED_COL]).reset_index(drop=True)
    df_doc[SEED_COL] = df_doc[SEED_COL].astype(int)

    if len(df_doc) == 0:
        raise RuntimeError("df_doc is empty after sanitization.")

    groups = df_doc[SEED_COL].to_numpy().astype(int)
    tr_idx, va_idx, te_idx = grouped_split_70_15_15_index(groups, random_state=SPLIT_RANDOM_STATE)
    seeds_tr, seeds_va, seeds_te = set(groups[tr_idx]), set(groups[va_idx]), set(groups[te_idx])
    assert not (seeds_tr & seeds_va) and not (seeds_tr & seeds_te) and not (seeds_va & seeds_te)

    if len(tr_idx) == 0 or len(va_idx) == 0 or len(te_idx) == 0:
        raise RuntimeError(f"empty split: train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")

    split_info = {
        "split_random_state": SPLIT_RANDOM_STATE,
        "opr_run_seed": OPR_RUN_SEED,
        "fac_run_seed": os.environ.get("FAC_RUN_SEED", ""),
        "fac_adapter_dir": fac_adapter_dir,
        "llama_base_model": LLAMA_BASE_MODEL,
        "n_docs_train": int(len(tr_idx)),
        "n_docs_val": int(len(va_idx)),
        "n_docs_test": int(len(te_idx)),
        "train_seeds": sorted(map(int, seeds_tr)),
        "val_seeds": sorted(map(int, seeds_va)),
        "test_seeds": sorted(map(int, seeds_te)),
        "run_llama_text": RUN_LLAMA_TEXT,
        "run_predfa_mlp": RUN_PREDFA_MLP,
        "run_llama_plus_predfa": RUN_LLAMA_PLUS_PREDFA,
    }
    with open(os.path.join(OUT_DIR, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"[INFO] Split: train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")

    Y = df_doc[TARGETS].to_numpy(np.float32)
    Y_tr, Y_va, Y_te = Y[tr_idx], Y[va_idx], Y[te_idx]

    all_rows = []
    best_by_family = {}

    # Family 1: LLaMA text-only (no FAC needed)
    if RUN_LLAMA_TEXT:
        for multi_task in ([True] if DO_MULTI_TASK else []) + ([False] if DO_SINGLE_TASK else []):
            family = f"llama_text_{'mt' if multi_task else 'st'}"
            print(f"\n[GRID] {family}")
            for run_id, cfg in enumerate(iter_grid(LLAMA_GRID), start=1):
                print(f"[RUN] {family} #{run_id}: {cfg}")
                Yv, Pv, Yt, Pt = train_llama_once(df_doc, tr_idx, va_idx, te_idx, cfg, predfa_feats=None, multi_task=multi_task)
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

        print(f"\n[FAC] Predicting sentence-level FA probabilities with LLaMA+LoRA adapter (FAC seed={os.environ.get('FAC_RUN_SEED')})...")
        tr_pred = predict_fa_probs_llama(df_tr_sent, fac_adapter_dir, batch=FAC_BATCH, max_len=LLAMA_MAXLEN_SENT)
        va_pred = predict_fa_probs_llama(df_va_sent, fac_adapter_dir, batch=FAC_BATCH, max_len=LLAMA_MAXLEN_SENT)
        te_pred = predict_fa_probs_llama(df_te_sent, fac_adapter_dir, batch=FAC_BATCH, max_len=LLAMA_MAXLEN_SENT)

        tr_pred.to_csv(os.path.join(OUT_DIR, "predfa_sentence_probs_train.csv"), index=False)
        va_pred.to_csv(os.path.join(OUT_DIR, "predfa_sentence_probs_val.csv"), index=False)
        te_pred.to_csv(os.path.join(OUT_DIR, "predfa_sentence_probs_test.csv"), index=False)

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
                        cfg_save = _row_with_metadata({**cfg, "family": family, "predfa_mode": predfa_mode, "tau": tau_for_mode}, fac_adapter_dir)
                        metrics = save_run(run_dir, cfg_save, Yv, Pv, Yt, Pt)
                        row = _row_with_metadata({**cfg, "family": family, "run_id": run_id, "predfa_mode": predfa_mode, "tau": tau_for_mode}, fac_adapter_dir)
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

            # Family 3: LLaMA + PredFA fusion
            if RUN_LLAMA_PLUS_PREDFA:
                for multi_task in ([True] if DO_MULTI_TASK else []) + ([False] if DO_SINGLE_TASK else []):
                    family = f"llama_plus_predfa_{predfa_mode}_{'mt' if multi_task else 'st'}"
                    print(f"\n[GRID] {family}")
                    for run_id, cfg in enumerate(iter_grid(LLAMA_GRID), start=1):
                        print(f"[RUN] {family} #{run_id}: {cfg}")
                        Yv, Pv, Yt, Pt = train_llama_once(
                            df_doc, tr_idx, va_idx, te_idx, cfg,
                            predfa_feats=(Xtr, Xva, Xte), multi_task=multi_task,
                        )
                        run_dir = os.path.join(OUT_DIR, family, f"run_{run_id:03d}")
                        cfg_save = _row_with_metadata({**cfg, "family": family, "predfa_mode": predfa_mode, "tau": tau_for_mode}, fac_adapter_dir)
                        metrics = save_run(run_dir, cfg_save, Yv, Pv, Yt, Pt)
                        row = _row_with_metadata({**cfg, "family": family, "run_id": run_id, "predfa_mode": predfa_mode, "tau": tau_for_mode}, fac_adapter_dir)
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
        print("[INFO] PredFA disabled; skipping FAC inference and HSPT-13.")

    if not all_rows:
        raise RuntimeError("No OPR systems were run. Check RUN_LLAMA_TEXT/RUN_PREDFA_MLP/RUN_LLAMA_PLUS_PREDFA flags.")

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
        "dropout", "hidden_dim", "lora_r", "lora_alpha", "predfa_mode", "tau", "run_dir",
    ]
    cols = [c for c in cols if c in best_df.columns]
    if len(best_df):
        print(best_df[cols].to_string(index=False))

    print(f"\n[DONE] Grid results -> {grid_path}")
    print(f"[DONE] Best by family -> {best_path}")
    if need_predfa:
        print(f"[DONE] FAC tau search -> {os.path.join(OUT_DIR, 'fac_val_tau_search.csv')}")
    print(f"[DONE] Output dir -> {OUT_DIR}")


if __name__ == "__main__":
    main()
