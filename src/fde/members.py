"""The best tuned learner and the ensemble members, all built from the same training rows.

Best tuned learner: Optuna (TPE, 20 trials) tunes each of the twelve learner types against the run metric on inner
stratified 3-fold out-of-fold predictions, and the type with the highest inner score wins. Above 3,000 rows, SVC is
skipped (`svc_max_n`).
Members (all of them use every column):
  cwc2_{type}   the five next-best learner types (by inner score), with their tuned parameters and balanced class
                weights (k-nearest neighbours and MLP get no weights)
  imb_{type}    BalancedRandomForest (300 trees), EasyEnsemble (10 estimators) and RUSBoost (100 estimators)
  rbc2_{type}   the same five learner types, each fitted on every row of the smallest class and on at most twice as
                many rows of any other class (one random undersampling draw), with probabilities corrected to equal
                class priors
  c2_{type}     the same five learner types with no imbalance treatment (FDEClassifier does not use them)
Each member yields inner out-of-fold probabilities, which fit the combiner, and a model fitted on all rows.
"""
import time

import numpy as np
import optuna
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from .learners import LEARNER_TYPES, Member, default_params, suggest

optuna.logging.set_verbosity(optuna.logging.WARNING)
EPS = 1e-12
N_OTHER, RB_RATIO = 5, 2


def score(y, pred, metric):
    return f1_score(y, pred, average="weighted" if metric == "wF1" else "macro")


def inner_splits(y, seed, folds=3):
    k = int(max(2, min(folds, np.bincount(y).max())))
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def member_oof(make, X, y, splits, K):
    out = np.zeros((len(y), K))
    for tr, te in splits:
        out[te] = make().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])
    return out


class RBMember:
    """Member fitted on a class-rebalanced sample; its probabilities are corrected to equal class priors."""

    def __init__(self, mt, params, mask, K, seed, ratio=RB_RATIO):
        self.mt, self.params, self.mask, self.K, self.seed, self.ratio = mt, params, mask, K, seed, ratio

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed); cnt = np.bincount(y, minlength=self.K)
        n_min = cnt[cnt > 0].min(); idx = []
        for c in np.flatnonzero(cnt):
            rows = np.flatnonzero(y == c)
            idx.append(rows if len(rows) <= self.ratio * n_min else rng.choice(rows, int(self.ratio * n_min), replace=False))
        idx = np.sort(np.concatenate(idx))
        self.prior = np.bincount(y[idx], minlength=self.K) / len(idx)
        self.m = Member(self.mt, self.params, self.mask, self.K, self.seed).fit(X.iloc[idx], y[idx])
        return self

    def predict_proba(self, X):
        p = self.m.predict_proba(X) / np.where(self.prior > 0, self.prior, 1.0)
        return p / np.clip(p.sum(1, keepdims=True), 1e-12, None)


class CWMember:
    """Member fitted with balanced class weights."""

    def __init__(self, mt, params, mask, K, seed):
        self.m = Member(mt, params, mask, K, seed)

    def fit(self, X, y):
        self.m.fit(X, y, balanced=True); return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)


def tune_best_learner(X, y, K, metric, seed, splits, trials=20, svc_max_n=3000, n_ref=0):
    """Returns (type, parameters, inner score, {type: (inner score, parameters)}, tuning seconds per type)."""
    cols, best, pool, tune = list(X.columns), None, {}, {}
    n = n_ref or len(y)
    for mt in [t for t in LEARNER_TYPES if not (t == "svc" and n > svc_max_n)]:
        t0 = time.time()
        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))

        def objective(t, mt=mt):
            pred = np.zeros(len(y), dtype=np.int64)
            for tr, te in splits:
                pred[te] = Member(mt, suggest(t, mt), cols, K, seed, proba=False).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
            return score(y, pred, metric)
        st.optimize(objective, n_trials=trials)
        p = dict(st.best_params)
        pool[mt] = (st.best_value, p); tune[mt] = round(time.time() - t0, 3)
        if best is None or st.best_value > best[2] + EPS:
            best = (mt, p, st.best_value)
    return best[0], best[1], best[2], pool, tune


def recipes(X, K, seed, best_type, best_params, pool, families):
    cols = list(X.columns)
    others = [mt for _, mt in sorted(((v[0], mt) for mt, v in pool.items() if mt != best_type), reverse=True)][:N_OTHER]
    out = []
    if "c2" in families:
        out += [(f"c2_{mt}", (lambda mt=mt: Member(mt, pool[mt][1], cols, K, seed))) for mt in others]
    if "rbc2" in families:
        out += [(f"rbc2_{mt}", (lambda mt=mt: RBMember(mt, pool[mt][1], cols, K, seed * 1000 + 91))) for mt in others]
    if "cwc2" in families:
        out += [(f"cwc2_{mt}", (lambda mt=mt: CWMember(mt, pool[mt][1], cols, K, seed * 1000 + 101))) for mt in others]
    if "imb" in families:
        out += [(f"imb_{mt}", (lambda mt=mt: Member(mt, default_params(mt), cols, K, seed * 1000 + 103))) for mt in ("brf", "eec", "rusb")]
    return out, others


def build(X, y, K, metric, seed, families, n_ref=0, trials=20, inner_folds=3, keep_models=False):
    """Best tuned learner and members on the rows (X, y): inner out-of-fold probabilities (float32 values, as stored
    by the study) and, when keep_models, the models fitted on all rows. Returns (oof dict, models dict, info)."""
    splits = inner_splits(y, seed, inner_folds)
    mt, p, inner, pool, tune = tune_best_learner(X, y, K, metric, seed, splits, trials, n_ref=n_ref)
    cols = list(X.columns)
    make_p1 = lambda: Member(mt, p, cols, K, seed)
    oof = {"P1": member_oof(make_p1, X, y, splits, K).astype(np.float32).astype(np.float64)}
    models = {"P1": make_p1().fit(X, y)} if keep_models else {}
    rec, others = recipes(X, K, seed, mt, p, pool, families)
    for lab, make in rec:
        oof[lab] = member_oof(make, X, y, splits, K).astype(np.float32).astype(np.float64)
        if keep_models:
            models[lab] = make().fit(X, y)
    info = {"best_type": mt, "best_params": p, "inner_score": inner, "pool": {k: round(v[0], 5) for k, v in pool.items()},
            "tune_seconds": tune, "other_types": others, "n_rows": len(y)}
    return oof, models, info
