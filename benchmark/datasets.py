"""Loader of the 53 evaluation datasets of the accompanying paper, and the outer TRAIN/TEST split of each run.

Specifications are in datasets.json: source (openml, uci, kaggle, open_data), identifier, target column, removed
columns, and the stratified row limit. Sources:
  openml     fetched with scikit-learn's fetch_openml by data id
  uci        UCI Machine Learning Repository by dataset id: the table returned by ucimlrepo (data.original), or a
             file <uci_dir>/<id>/original.csv holding the same table
  kaggle     files downloaded with the Kaggle API into <kaggle_dir>/<owner>_<dataset>/ (licences forbid redistribution)
  open_data  tables built from public-agency and open-licence sources, shipped in data_open/ (see SOURCES.md)

Usage:
  from datasets import load, holdout
  X, y = load("msha-accidents")              # X: DataFrame, y: int labels 0..K-1
  train, test = holdout(y, seed)             # seeds 0 to 4, as in the paper
"""
import json, os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = json.load(open(os.path.join(HERE, "datasets.json")))
SUBSAMPLE_SEED = 20261022          # stratified row limit of the large OpenML and Kaggle tables


def _raw(name, e, uci_dir=None, kaggle_dir=None, openml_home=None):
    s = e["source"]
    if s == "open_data":
        d = pd.read_csv(os.path.join(HERE, "data_open", f"{name}.csv.gz"), skipinitialspace=True)
        return d.iloc[:, :-1], d.iloc[:, -1].astype(str).str.strip()
    if s == "uci":
        if uci_dir:
            d = pd.read_csv(os.path.join(uci_dir, str(e["id"]), "original.csv"))
        else:
            from ucimlrepo import fetch_ucirepo
            d = fetch_ucirepo(id=int(e["id"])).data.original
        d = d[d[e["target"]].notna()]
        return d.drop(columns=[e["target"]] + [c for c in e.get("drop", []) if c in d.columns]), d[e["target"]]
    if s == "kaggle":
        if not kaggle_dir:
            raise ValueError(f"{name}: download kaggle dataset {e['file'].split('/')[0].replace('_', '/', 1)} and pass kaggle_dir")
        d = pd.read_csv(os.path.join(kaggle_dir, e["file"]), low_memory=False)
        t = e["target"]
        y = (d[t] > 0).map({True: "claim", False: "none"}) if e.get("rule") == "positive" else d[t]
        return d.drop(columns=[t] + e.get("drop", [])), y
    from sklearn.datasets import fetch_openml
    home = openml_home or os.path.expanduser("~/scikit_learn_data")
    if "target_dropna" in e:
        fr = fetch_openml(data_id=int(e["id"]), as_frame=True, parser="liac-arff", data_home=home, target_column=None).frame
        t = e["target_dropna"]; fr = fr[fr[t].notna()].reset_index(drop=True)
        return fr.drop(columns=[t]), fr[t]
    d = fetch_openml(data_id=int(e["id"]), as_frame=True, parser="liac-arff", data_home=home)
    return d.data, d.target


def load(name, uci_dir=None, kaggle_dir=None, openml_home=None):
    """(X, y) of one evaluation dataset, prepared as in the paper: row limit, removed columns, text columns kept raw,
    numbers stored as text converted, identifier-like text columns removed, labels mapped to 0..K-1 in sorted order."""
    e = SPEC[name]
    X, y = _raw(name, e, uci_dir, kaggle_dir, openml_home)
    if "subsample" in e and len(y) > e["subsample"]:
        keep, _ = train_test_split(np.arange(len(y)), train_size=e["subsample"], stratify=np.asarray(y), random_state=SUBSAMPLE_SEED)
        keep = np.sort(keep); X, y = X.iloc[keep], pd.Series(np.asarray(y)[keep])
    X = X.copy(); X.columns = [str(c) for c in X.columns]
    if e["source"] != "uci":
        X = X.drop(columns=[c for c in e.get("drop", []) if c in X.columns])
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("object").where(X[c].notna(), np.nan)
    for c in X.columns:                      # numbers stored as text -> numeric
        if not pd.api.types.is_numeric_dtype(X[c]):
            v = pd.to_numeric(X[c].astype("string").str.strip(), errors="coerce")
            nn = X[c].notna() & (X[c].astype("string").str.strip() != "")
            if nn.sum() > 0 and v[nn].notna().mean() >= 0.95:
                X[c] = v.astype(float)
    # identifier-like columns: text with unique values in more than half of the rows
    X = X.drop(columns=[c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() > 0.5 * len(X)])
    y = pd.Series(np.asarray(y)).astype(str)
    classes = sorted(y.unique())
    return X.reset_index(drop=True), y.map({c: i for i, c in enumerate(classes)}).to_numpy(dtype=np.int64)


def holdout(y, seed, test_size=0.2):
    """Outer split of one run: stratified 80/20 TRAIN/TEST indices for seed 0 to 4."""
    tr, te = train_test_split(np.arange(len(y)), test_size=test_size, stratify=y, random_state=seed)
    return np.sort(tr), np.sort(te)
