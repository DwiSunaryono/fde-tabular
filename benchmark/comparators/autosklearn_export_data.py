"""Data export for the Auto-sklearn baseline (its environment, Python 3.9 / scikit-learn 0.24, cannot load the data
with the study's loader). For every dataset of the primary list, writes askl_data/<dataset>.npz with:
  X        float matrix: numbers as loaded (NaN kept), categories coded 0..m-1 in sorted order, missing category = m
  cat      boolean mask of categorical columns (passed to Auto-sklearn as feat_type "Categorical")
  y        labels 0..K-1 (as in the study)
The TRAIN/TEST indices come from each library (train_idx, test_idx), so the split equals the other baselines.
Usage: python tools_run/askl_prep.py <primary list.json> <out dir>
"""
import json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.getcwd())
from fde2 import data  # noqa: E402

names = json.load(open(sys.argv[1])); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
for n in names:
    m = json.load(open(f"conf2_final/{n}_seed0.json"))["identity"]
    X, y = data.load(n, m["source"], m["data_id"])
    cat = np.array([not pd.api.types.is_numeric_dtype(X[c]) for c in X.columns])
    M = np.zeros(X.shape)
    for j, c in enumerate(X.columns):
        if cat[j]:
            v = X[c].astype("object"); levels = sorted({str(t) for t in v.dropna()})
            code = {t: i for i, t in enumerate(levels)}
            M[:, j] = [len(levels) if pd.isna(t) else code[str(t)] for t in v]
        else:
            M[:, j] = pd.to_numeric(X[c], errors="coerce").to_numpy(dtype=float)
    np.savez_compressed(os.path.join(out, f"{n}.npz"), X=M, cat=cat, y=y)
    print(n, M.shape, int(cat.sum()))
