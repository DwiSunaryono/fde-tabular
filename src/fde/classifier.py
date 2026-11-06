"""FDEClassifier: Feasible Diverse Ensemble.

Procedure (the configuration evaluated in the accompanying paper):
  1. Run metric: macro-F1 for two classes with a minority share below 0.15, otherwise weighted-F1.
  2. The training rows are split into 3 stratified outer folds. On the fit rows of each fold, and on all rows, the best
     tuned learner and the members are built: class-weighted, imbalance-specific and rebalanced members, and the
     extended members of fde.extended (large-budget GBDTs, an MLP, large forests).
  3. Combiner: stacking (multinomial logistic regression on the probabilities of the best learner and the members),
     then per-class probability scales, fitted on out-of-fold probabilities.
  4. Cross-fitted acceptance rule on the pooled outer-fold predictions: the ensemble is used only when the lower 5%
     bootstrap bound of its gain over the best tuned learner is above 0, its weight in the all-rows fit is not zero,
     and its predictions differ from the best learner's on at least one outer-fold row. Otherwise the best tuned
     learner is kept. The same rule applies to every imbalance level.
Option members="light" leaves out the extended members (step 2): a lighter and faster ensemble, somewhat less
accurate in the accompanying paper.
"""
import json, os, time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import __version__
from .combiners import GRID_WIDE, apply, fast_score, fit_combiner
from .deploy_test import OUTER_FOLDS, deployment_test, gate_metric
from .extended import KINDS, fit_member, raw_frame
from .learners import Member
from .members import build

FAMILIES = ("cwc2", "imb", "rbc2")
MARGIN = 0.0                       # the ensemble is used when the lower bound of its gain is above 0


def imbalance_group(y):
    cnt = np.bincount(y); cnt = cnt[cnt > 0]; ir = cnt.max() / cnt.min()
    return ("balanced" if ir < 3 else "low" if ir < 10 else "moderate" if ir < 50 else "extreme"), float(ir)


def prepare_frame(X):
    """Column names as strings; non-numeric columns as objects with NaN for missing values; numbers stored as text
    converted to numbers. Categorical columns stay raw (each learner encodes them on its own training rows)."""
    X = pd.DataFrame(X).copy()
    X.columns = [str(c) for c in X.columns]
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("object").where(X[c].notna(), np.nan)
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            v = pd.to_numeric(X[c].astype("string").str.strip(), errors="coerce")
            nn = X[c].notna() & (X[c].astype("string").str.strip() != "")
            if nn.sum() > 0 and v[nn].notna().mean() >= 0.95:
                X[c] = v.astype(float)
    return X.reset_index(drop=True)


def _fitset(X, y, K, metric, seed, n_ref, trials, Xr, cat, threads, extended):
    oof, models, info = build(X, y, K, metric, seed, FAMILIES, n_ref=n_ref, trials=trials, keep_models=True)
    for kind in KINDS if extended else ():              # extended members, on the raw frame
        o, m = fit_member(kind, Xr, y, K, seed, cat, threads)
        oof[kind], models[kind] = o.astype(np.float32).astype(np.float64), m
    labs = sorted(l for l in oof if l != "P1")
    spec, _, weight = fit_combiner("stack_scaled", oof, labs, y, K, metric, seed, GRID_WIDE)
    return {"models": models, "spec": spec, "weight": weight, "info": info, "labs": labs}


def _probs(models, labs, X, Xr=None):
    return {l: models[l].predict_proba(Xr if l in KINDS else X).astype(np.float32).astype(np.float64)
            for l in ["P1"] + labs if l in models}


class FDEClassifier:
    """Feasible Diverse Ensemble for tabular classification (scikit-learn style).

    Parameters
    ----------
    random_state : int, seed of every split, search and member.
    n_jobs : int, fit sets built in parallel (each tuned learner uses one thread).
    trials : int, Optuna trials per learner type (20 in the evaluated configuration).
    members : "full" (default) or "light" (imbalance-handling members only: lighter and faster).
    member_threads : int, threads of each extended member (4 in the evaluated configuration).
    """

    def __init__(self, random_state=0, n_jobs=4, trials=20, members="full", member_threads=4, verbose=0):
        self.random_state, self.n_jobs, self.trials = random_state, n_jobs, trials
        self.members, self.member_threads, self.verbose = members, member_threads, verbose

    def fit(self, X, y):
        t0 = time.time()
        if self.members not in ("full", "light"):
            raise ValueError('members must be "full" or "light"')
        Xr, self.cat_ = raw_frame(X)
        X = prepare_frame(X); y_raw = pd.Series(np.asarray(y)).astype(str)
        self.classes_ = np.array(sorted(y_raw.unique()))
        y = y_raw.map({c: i for i, c in enumerate(self.classes_)}).to_numpy(dtype=np.int64)
        self.columns_ = list(X.columns); K = len(self.classes_); seed = self.random_state
        if K < 2:
            raise ValueError("at least two classes are required")
        self.metric_ = gate_metric(y); self.group_, ir = imbalance_group(y)
        from joblib import Parallel, delayed
        rounds = list(StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
        sub = lambda D, i: D.iloc[i].reset_index(drop=True)
        jobs = [(sub(X, i), y[i], sub(Xr, i)) for i, _ in rounds] + [(X, y, Xr)]
        out = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(_fitset)(Xf, yf, K, self.metric_, seed, len(y), self.trials, Xrf, self.cat_, self.member_threads,
                                  self.members == "full") for Xf, yf, Xrf in jobs)
        cf_ref, cf_cand = np.zeros(len(y), int), np.zeros(len(y), int)
        for (i, j), r in zip(rounds, out[:3]):
            p = _probs(r["models"], r["labs"], sub(X, j), sub(Xr, j))
            cf_ref[j] = p["P1"].argmax(1); cf_cand[j] = apply(r["spec"], p).argmax(1)
        test = deployment_test(y, cf_ref, cf_cand, metric=self.metric_, seed=0, K=K, margin=MARGIN)
        full = out[3]
        self.uses_ensemble_ = bool(test.accept and full["weight"] > 0 and np.any(cf_cand != cf_ref))
        if self.uses_ensemble_:
            self.model_ = full
        else:
            self.model_ = {"models": {"P1": full["models"]["P1"]}, "spec": {"kind": "best"}, "labs": [], "info": full["info"]}
        self.crossfit_ = {"reference": fast_score(y, cf_ref, K, self.metric_), "ensemble": fast_score(y, cf_cand, K, self.metric_)}
        info = self.model_["info"]
        self.report_ = {"version": __version__, "rows": len(y), "classes": [str(c) for c in self.classes_], "imbalance_ratio": round(ir, 3),
                        "imbalance_group": self.group_, "metric": self.metric_, "best_learner": info["best_type"],
                        "best_learner_inner_score": round(info["inner_score"], 5),
                        "acceptance_rule": {"lower_bound": round(test.lower_bound, 5), "gain": round(test.gain, 5), "margin": MARGIN,
                                            "accepted": bool(test.accept)},
                        "crossfit_scores": self.crossfit_,
                        "uses_ensemble": self.uses_ensemble_, "models_in_predictor": 1 + (len(self.model_["labs"]) if self.uses_ensemble_ else 0),
                        "options": {"members": self.members, "trials": self.trials, "member_threads": self.member_threads},
                        "fit_seconds": round(time.time() - t0, 1)}
        if self.verbose:
            print(json.dumps(self.report_, indent=1))
        return self

    def predict_proba(self, X):
        Xr = raw_frame(X, self.cat_)[0]
        X = prepare_frame(X)[self.columns_]
        return apply(self.model_["spec"], _probs(self.model_["models"], self.model_["labs"], X, Xr))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
