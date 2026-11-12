"""Extended members: large-budget GBDTs (ordinal or native categorical coding), an MLP, and large forests.

Every member is bagged over 3 stratified folds of the rows it is fitted on; a class with fewer rows reduces the
number of folds. A fold model is trained on every fold except one. The GBDTs and the MLP watch log loss on the fold
they do not see and stop early. Those held-out predictions become the member's out-of-fold probabilities for the
combiner, while new rows are scored by the mean of the fold models.

  catn_cat   CatBoost, depth 6, categories kept as text, missing values treated as a category of their own
  big_lgbm   LightGBM, 63 leaves, subsample 0.8, ordinal-coded categories, numbers median-imputed
  big_xgb    XGBoost, depth 6, subsample 0.8, coded the same way
  catn_lgbm  LightGBM as big_lgbm, but with native categorical handling
  catn_xgb   XGBoost as big_xgb, but with native categorical handling
  nn_mlp     PyTorch MLP: numbers median-imputed, then quantile-transformed to a normal distribution; categories
             one-hot coded once they occur at least 5 times, rarer ones grouped; 3 hidden layers of 256 units with
             ReLU and dropout 0.1; Adam (learning rate 1e-3), batch 512, at most 200 epochs, patience 10
  big_rf     random forest, 500 trees, sqrt(F) columns per split, ordinal-coded categories
  big_et     extra trees, same settings
GBDTs: at most 5,000 trees, learning rate 0.05, early stopping after 100 rounds. Each encoder is fitted on the rows
of its own fold model only.
"""
import sys

import numpy as np
import pandas as pd

from . import device
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, QuantileTransformer

KINDS = ("catn_cat", "big_lgbm", "big_xgb", "catn_lgbm", "catn_xgb", "nn_mlp", "big_rf", "big_et")
BAG_K = 3


def raw_frame(X, cat=None):
    """Columns renamed f0, f1, ... (LightGBM rejects some characters); non-numeric columns as text, with missing
    values forming a category of their own. `cat` fixes the categorical columns found at fit time."""
    X = pd.DataFrame(X).reset_index(drop=True)
    X = X.set_axis([f"f{i}" for i in range(X.shape[1])], axis=1)
    if cat is None:
        cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    X = X.copy()
    for c in cat:
        X[c] = np.array(["__missing__" if pd.isna(v) else str(v) for v in X[c]], dtype=object)
    return X, cat


def _proba(m, X, K, cls=None):
    out = np.zeros((len(X), K)); c = np.asarray(m.classes_).astype(int)
    out[:, c if cls is None else cls[c]] = m.predict_proba(X)
    return out


def _coder(Xf, cat, kind):
    num = [c for c in Xf.columns if c not in cat]
    if kind == "nn_mlp":
        nt = Pipeline([("imp", SimpleImputer(strategy="median")),
                       ("q", QuantileTransformer(n_quantiles=min(1000, max(10, len(Xf))), output_distribution="normal", random_state=0))])
        ct = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                       ("enc", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5, sparse_output=False))])
    else:
        nt = SimpleImputer(strategy="median")
        ct = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))])
    return ColumnTransformer(([("num", nt, num)] if num else []) + ([("cat", ct, cat)] if cat else []))


# macOS: PyTorch and LightGBM/XGBoost bring separate OpenMP libraries, and the MLP crashes in a process holding both, so
# the MLP is trained in a separate process and predicts in numpy there (see _torch_mlp)
ISOLATE_TORCH = sys.platform == "darwin"


class _Net:
    """Fitted MLP plus its input coder; predicts class probabilities. `net` is a PyTorch network or, when trained in a
    separate process, the list of its layer weights."""

    def __init__(self, ct, net):
        self.ct, self.net = ct, net

    def predict_proba(self, X):
        A = self.ct.transform(X).astype(np.float32)
        if isinstance(self.net, list) or ISOLATE_TORCH:
            from ._torch_mlp import forward, weights
            W = self.net if isinstance(self.net, list) else weights(self.net)
            return forward(W, A)
        import torch
        with torch.no_grad():
            return torch.softmax(self.net(torch.tensor(A, dtype=torch.float32)), 1).numpy()


def _mlp(A, ya, H, yh, K, seed, threads):
    if not ISOLATE_TORCH:
        from ._torch_mlp import train
        return train(A, ya, H, yh, K, seed, threads, "cuda" if device.gpu() else "cpu")
    import os, subprocess, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "in.npz"), os.path.join(tmp, "out.npz")
        np.savez(src, A=A, ya=ya, H=H, yh=yh, K=K, seed=seed, threads=threads)
        p = subprocess.run([sys.executable, "-m", "fde._torch_mlp", src, dst], capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"training the MLP member in a separate process failed (exit {p.returncode}): {p.stderr[-2000:]}")
        d = np.load(dst)
        return [(d[f"w{i}"], d[f"b{i}"]) for i in range(int(d["n"]))]


class Bagged:
    """One extended member: the fold models of its 3-fold bagging, with their preprocessing."""

    def __init__(self, kind, cat, K):
        self.kind, self.cat, self.K, self.parts = kind, cat, K, []

    def _prep(self, X, i):
        conv, ct = self.parts[i][0]
        if conv == "native":
            return X.assign(**{c: X[c].astype(ct[c]) for c in self.cat})
        if conv == "ct":
            return ct.transform(X).astype(np.float32)
        return X

    def _one(self, i, X):
        _, m, cls = self.parts[i]
        if self.kind in ("nn_mlp", "big_rf", "big_et"):
            out = np.zeros((len(X), self.K))
            if self.kind == "nn_mlp":
                out[:] = m.predict_proba(X)
            else:
                out[:, m.classes_] = m.predict_proba(self._prep(X, i))
            return out
        return _proba(m, self._prep(X, i), self.K, cls)

    def predict_proba(self, X):
        """X: a frame from raw_frame, carrying the categorical columns found at fit time."""
        return sum(self._one(i, X) for i in range(len(self.parts))) / len(self.parts)


def fit_member(kind, Xf, yf, K, seed, cat, threads=4):
    """Bagged member fitted on the rows (Xf, yf); returns (out-of-fold probabilities, fitted member)."""
    oof = np.zeros((len(yf), K)); mem = Bagged(kind, cat, K)
    cnt = np.bincount(yf); k = int(max(2, min(BAG_K, cnt[cnt > 0].min())))
    for a, h0 in StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(yf)), yf):
        if kind in ("nn_mlp", "big_rf", "big_et"):
            h = h0
            ct = _coder(Xf, cat, kind).fit(Xf.iloc[a])
            A, H = (ct.transform(x).astype(np.float32) for x in (Xf.iloc[a], Xf.iloc[h]))
            if kind == "nn_mlp":
                net = _Net(ct, _mlp(A, yf[a], H, yf[h], K, seed, threads))
                mem.parts.append((("none", None), net, None)); oof[h] = net.predict_proba(Xf.iloc[h])
            else:
                from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
                M = RandomForestClassifier if kind == "big_rf" else ExtraTreesClassifier
                m = M(n_estimators=500, max_features="sqrt", n_jobs=threads, random_state=seed).fit(A, yf[a])
                mem.parts.append((("ct", ct), m, None)); ph = np.zeros((len(H), K)); ph[:, m.classes_] = m.predict_proba(H); oof[h] = ph
            continue
        # a class with too few rows for the fitting part is dropped from that fold model (labels re-coded to 0..k-1);
        # its held-out rows still receive out-of-fold probabilities, but early stopping only sees rows of the classes
        # that remained
        cls = np.unique(yf[a]); code = np.full(K, -1); code[cls] = np.arange(len(cls)); ya = code[yf[a]]
        h = h0[code[yf[h0]] >= 0]; yh = code[yf[h]]; cls = None if len(cls) == K else cls
        if kind == "catn_cat":
            from catboost import CatBoostClassifier
            gpu = dict(task_type="GPU", devices="0", gpu_ram_part=device.GPU_RAM_PART) if device.gpu() else {}
            m = CatBoostClassifier(iterations=5000, learning_rate=0.05, depth=6, thread_count=threads, random_seed=seed, verbose=0,
                                   allow_writing_files=False, od_type="Iter", od_wait=100, cat_features=cat, **gpu)
            m.fit(Xf.iloc[a], ya, eval_set=(Xf.iloc[h], yh), use_best_model=True)
            mem.parts.append((("none", None), m, cls)); oof[h0] = _proba(m, Xf.iloc[h0], K, cls)
            continue
        if kind in ("catn_lgbm", "catn_xgb"):
            cats = {c: pd.CategoricalDtype(sorted(set(Xf[c].iloc[a]))) for c in cat}
            conv = lambda D: D.assign(**{c: D[c].astype(cats[c]) for c in cat})
            A, H, H0 = conv(Xf.iloc[a]), conv(Xf.iloc[h]), conv(Xf.iloc[h0])
            if kind == "catn_lgbm":
                from lightgbm import LGBMClassifier, early_stopping
                m = LGBMClassifier(n_estimators=5000, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                   n_jobs=threads, random_state=seed, verbose=-1)
                m.fit(A, ya, eval_set=[(H, yh)], callbacks=[early_stopping(100, verbose=False)])
            else:
                from xgboost import XGBClassifier
                m = XGBClassifier(n_estimators=5000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                  enable_categorical=True, max_cat_to_onehot=1, n_jobs=threads, random_state=seed, verbosity=0, early_stopping_rounds=100,
                                  device="cuda" if device.gpu() else "cpu")
                m.fit(A, ya, eval_set=[(H, yh)], verbose=False); m.set_params(device="cpu")
            mem.parts.append((("native", cats), m, cls)); oof[h0] = _proba(m, H0, K, cls)
            continue
        ct = _coder(Xf, cat, kind).fit(Xf.iloc[a])
        A, H, H0 = (ct.transform(x).astype(np.float32) for x in (Xf.iloc[a], Xf.iloc[h], Xf.iloc[h0]))
        if kind == "big_lgbm":
            from lightgbm import LGBMClassifier, early_stopping
            m = LGBMClassifier(n_estimators=5000, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                               n_jobs=threads, random_state=seed, verbose=-1)
            m.fit(A, ya, eval_set=[(H, yh)], callbacks=[early_stopping(100, verbose=False)])
        else:
            from xgboost import XGBClassifier
            m = XGBClassifier(n_estimators=5000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                              n_jobs=threads, random_state=seed, verbosity=0, early_stopping_rounds=100,
                              device="cuda" if device.gpu() else "cpu")
            m.fit(A, ya, eval_set=[(H, yh)], verbose=False); m.set_params(device="cpu")
        mem.parts.append((("ct", ct), m, cls)); oof[h0] = _proba(m, H0, K, cls)
    return oof, mem
