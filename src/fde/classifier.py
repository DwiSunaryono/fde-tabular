"""FDEClassifier: Feasible Diverse Ensemble.

Procedure (the configuration evaluated in the accompanying paper):
  1. Metric of a run: macro-F1 when there are two classes and the minority share is below 0.15, weighted-F1
     otherwise.
  2. The training rows are cut into 3 stratified outer folds. For each fold, on its fit rows, and once on all rows,
     two things are built: the best tuned learner, and the members — class-weighted, imbalance-specific and
     rebalanced ones, plus those of fde.extended (large-budget GBDTs, an MLP, large forests).
  3. Combiner: stacking (multinomial logistic regression over the probabilities of the best learner and the
     members) followed by per-class probability scales; both are fitted on out-of-fold probabilities.
  4. Acceptance rule, cross-fitted over the pooled outer-fold predictions: the ensemble is used only if three
     conditions hold — the lower 5% bootstrap quantile of its gain over the best tuned learner is above 0, its
     weight in the all-rows fit is non-zero, and it differs from the best learner on at least one outer-fold row.
     Otherwise the best tuned learner is kept. No imbalance level is treated differently.
members="light" drops the extended members of step 2, which makes the ensemble lighter and faster but somewhat less
accurate in the accompanying paper.
"""
import json, os, sys, time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import __version__, device
from .combiners import GRID_WIDE, apply, fast_score, fit_combiner
from .deploy_test import OUTER_FOLDS, acceptance_rule, gate_metric
from .extended import KINDS, fit_member, raw_frame
from .learners import Member
from .members import inner_splits, member_task, pick_best, recipes, tune_one, tuned_types

FAMILIES = ("cwc2", "imb", "rbc2")
MARGIN = 0.0                       # the ensemble is used when the lower bound of its gain lies above 0


def imbalance_group(y):
    cnt = np.bincount(y); cnt = cnt[cnt > 0]; ir = cnt.max() / cnt.min()
    return ("balanced" if ir < 3 else "low" if ir < 10 else "moderate" if ir < 50 else "extreme"), float(ir)


def prepare_frame(X):
    """Column names as strings, non-numeric columns as objects holding NaN for missing values, and numbers that
    arrive as text converted to numbers. Categorical columns are left raw; each learner encodes them on its own
    training rows."""
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


class _Bar:
    """One-line progress bar on stderr (shown on a terminal only): stage, bar, tasks done, elapsed and remaining time,
    and the task that finished last."""

    def __init__(self, total, stage, on, work=None):
        self.total, self.stage, self.done, self.t0, self.what = max(total, 1), stage, 0, time.time(), ""
        self.work, self.got = max(work or total, 1e-9), 0.0
        self.on = on and sys.stderr.isatty()

    def step(self, what, weight=1.0):
        self.done = min(self.done + 1, self.total); self.got += weight; self.what = what; self.draw()

    def draw(self):
        if not self.on:
            return
        el = time.time() - self.t0; frac = min(self.got / self.work, 1.0); w = 24
        rem = el / frac - el if frac > 0 else 0
        fmt = lambda t: f"{int(t // 60)}:{int(t % 60):02d}"
        bar = "#" * int(w * frac) + "-" * (w - int(w * frac))
        sys.stderr.write(f"\r{self.stage} [{bar}] {self.done}/{self.total} {100 * frac:3.0f}%  elapsed {fmt(el)}  "
                         f"remaining ~{fmt(rem)}  {self.what[:28]:28s}")
        sys.stderr.flush()

    def close(self, say):
        if self.on:
            sys.stderr.write("\r" + " " * 125 + "\r")
        say(f"{self.stage}: {self.total} tasks in {time.time() - self.t0:.0f} s")


def plan_jobs(n_jobs, X):
    """Parallel jobs from what the machine has free: usable CPU cores (one is left for the system) and available
    memory (each job is given about 30 times the size of the data, at least 300 MB). Returns (jobs, description)."""
    import psutil
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:                               # not available on Windows and macOS
        cores = os.cpu_count() or 1
    free = psutil.virtual_memory().available
    per_job = max(300e6, 30 * float(pd.DataFrame(X).memory_usage(deep=True).sum()))
    auto = max(1, min(cores - 1, int(free // per_job)))
    jobs = auto if n_jobs in (None, -1) else max(1, min(int(n_jobs), cores))
    return jobs, f"{cores} CPU cores, {free / 1e9:.1f} GB memory free -> {jobs} parallel jobs"


def gpu_note(want):
    """Device actually used, and a line for the system check."""
    name = device.available()
    if want == "gpu" and name:
        return "gpu", f"{name}, used for XGBoost, CatBoost and the MLP (results can differ slightly from the CPU)"
    if want == "gpu":
        return "cpu", "no CUDA GPU found, so the CPU is used"
    return "cpu", (f"{name} found, not used (--device gpu uses it)" if name else "none found; the CPU is used")


def _call(key, fn, args, dev="cpu"):
    """One parallel task on device `dev`; library notices (deprecations, CUDA probing) are not shown."""
    import warnings
    device.use(dev)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return key, fn(*args)


# rough relative cost of tuning each learner type, used only to start the slow tasks first and to weight the progress bar
COST = {"cat": 20, "rf": 3, "et": 3, "xgb": 2, "lgbm": 2, "hgb": 2, "svc": 2, "mlp": 2, "catn_cat": 6, "nn_mlp": 3}


def _weight(key):
    return float(COST.get(key[-1] if isinstance(key, tuple) else key, 1))


def _run(tasks, n_jobs, stage, verbose, say, dev="cpu"):
    """Runs {key: (fn, args, label)} on n_jobs processes, slow tasks first; returns {key: result}."""
    from joblib import Parallel, delayed
    order = sorted(tasks, key=_weight, reverse=True)
    bar, out = _Bar(len(tasks), stage, verbose, sum(_weight(k) for k in order)), {}
    for key, r in Parallel(n_jobs=n_jobs, backend="loky", return_as="generator_unordered")(
            delayed(_call)(key, tasks[key][0], tasks[key][1], dev) for key in order):
        out[key] = r; bar.step(tasks[key][2], _weight(key))
    if verbose:
        bar.close(say)
    return out


def _probs(models, labs, X, Xr=None):
    return {l: models[l].predict_proba(Xr if l in KINDS else X).astype(np.float32).astype(np.float64)
            for l in ["P1"] + labs if l in models}


class FDEClassifier:
    """Feasible Diverse Ensemble for tabular classification, in scikit-learn style.

    Parameters
    ----------
    random_state : int, seed shared by every split, search and member.
    n_jobs : int, parallel tasks; -1 (default) chooses from the free CPU cores and memory (see plan_jobs). Each tuned
        learner is single-threaded. Results do not depend on n_jobs.
    trials : int, Optuna trials for each learner type (20 in the evaluated configuration).
    members : "full" (default) or "light", the latter keeping only the imbalance-handling members: lighter and
        faster.
    member_threads : int, threads per extended member (4 in the evaluated configuration).
    device : "cpu" (default; the results of the accompanying paper) or "gpu" (XGBoost, CatBoost and the MLP on the first
        CUDA GPU; faster on large data, results can differ slightly from the CPU; falls back to the CPU without a GPU).
    verbose : int, 0 silent; 1 progress messages on stderr; 2 also prints the report when fitting ends.
    """

    def __init__(self, random_state=0, n_jobs=-1, trials=20, members="full", member_threads=4, device="cpu", verbose=0):
        self.random_state, self.n_jobs, self.trials = random_state, n_jobs, trials
        self.members, self.member_threads, self.device, self.verbose = members, member_threads, device, verbose

    def fit(self, X, y):
        t0 = time.time()
        if self.members not in ("full", "light"):
            raise ValueError('members must be "full" or "light"')
        if self.device not in ("cpu", "gpu"):
            raise ValueError('device must be "cpu" or "gpu"')
        Xr, self.cat_ = raw_frame(X)
        X = prepare_frame(X); y_raw = pd.Series(np.asarray(y)).astype(str)
        self.classes_ = np.array(sorted(y_raw.unique()))
        y = y_raw.map({c: i for i, c in enumerate(self.classes_)}).to_numpy(dtype=np.int64)
        self.columns_ = list(X.columns); K = len(self.classes_); seed = self.random_state
        if K < 2:
            raise ValueError("at least two classes are required")
        self.metric_ = gate_metric(y); self.group_, ir = imbalance_group(y)
        rounds = list(StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
        sub = lambda D, i: D.iloc[i].reset_index(drop=True)
        sets = [(sub(X, i), y[i], sub(Xr, i)) for i, _ in rounds] + [(X, y, Xr)]
        names = [f"fold {k + 1}/3" for k in range(3)] + ["all rows"]
        say = (lambda msg: print(f"[fde] {msg}", file=sys.stderr, flush=True)) if self.verbose else (lambda msg: None)
        say(f"{len(y)} rows, {K} classes, metric {self.metric_}, members: {self.members}")
        say("checking the system ...")
        jobs, machine = plan_jobs(self.n_jobs, X)
        say(f"  CPU and memory: {machine}")
        self.device_used_, gpu_line = gpu_note(self.device)
        say(f"  GPU: {gpu_line}")
        splits = [inner_splits(yf, seed, 3) for _, yf, _ in sets]
        types = tuned_types(len(y))
        # stage 1: every learner type in every fit set is tuned as its own task
        tuned = _run({(k, mt): (tune_one, (Xf, yf, K, self.metric_, seed, splits[k], self.trials, mt), f"{names[k]}: {mt}")
                      for k, (Xf, yf, _) in enumerate(sets) for mt in types}, jobs, "Stage 1/3 Tuning learners", self.verbose, say, self.device_used_)
        # stage 2: the best learner and every member of every fit set, each as its own task
        best, tasks = [], {}
        for k, (Xf, yf, Xrf) in enumerate(sets):
            mt, p, inner, pool, tune = pick_best({t: tuned[(k, t)] for t in types})
            rec, others = recipes(Xf, K, seed, mt, p, pool, FAMILIES)
            best.append({"best_type": mt, "best_params": p, "inner_score": inner, "pool": {t: round(v[0], 5) for t, v in pool.items()},
                         "tune_seconds": tune, "other_types": others, "n_rows": len(yf)})
            cols = list(Xf.columns)
            tasks[(k, "P1")] = (member_task, ((lambda mt=mt, p=p: Member(mt, p, cols, K, seed)), Xf, yf, splits[k], K, True), f"{names[k]}: P1")
            for lab, make in rec:
                tasks[(k, lab)] = (member_task, (make, Xf, yf, splits[k], K, True), f"{names[k]}: {lab}")
            for kind in KINDS if self.members == "full" else ():
                tasks[(k, kind)] = (fit_member, (kind, Xrf, yf, K, seed, self.cat_, self.member_threads), f"{names[k]}: {kind}")
        built = _run(tasks, jobs, "Stage 2/3 Building members", self.verbose, say, self.device_used_)
        oofs, models = [{} for _ in sets], [{} for _ in sets]
        for (k, lab), (o, m) in built.items():
            oofs[k][lab] = o.astype(np.float32).astype(np.float64); models[k][lab] = m
        # stage 3: the combiner of every fit set
        labs = [sorted(l for l in oofs[k] if l != "P1") for k in range(4)]
        comb = _run({k: (fit_combiner, ("stack_scaled", oofs[k], labs[k], sets[k][1], K, self.metric_, seed, GRID_WIDE), names[k])
                     for k in range(4)}, jobs, "Stage 3/3 Combiners", self.verbose, say)
        out = [{"models": models[k], "spec": comb[k][0], "weight": comb[k][2], "info": best[k], "labs": labs[k]} for k in range(4)]
        cf_ref, cf_cand = np.zeros(len(y), int), np.zeros(len(y), int)
        for (i, j), r in zip(rounds, out[:3]):
            p = _probs(r["models"], r["labs"], sub(X, j), sub(Xr, j))
            cf_ref[j] = p["P1"].argmax(1); cf_cand[j] = apply(r["spec"], p).argmax(1)
        rule = acceptance_rule(y, cf_ref, cf_cand, metric=self.metric_, seed=0, K=K, margin=MARGIN)
        full = out[3]
        self.uses_ensemble_ = bool(rule.accept and full["weight"] > 0 and np.any(cf_cand != cf_ref))
        if self.uses_ensemble_:
            self.model_ = full
        else:
            self.model_ = {"models": {"P1": full["models"]["P1"]}, "spec": {"kind": "best"}, "labs": [], "info": full["info"]}
        self.crossfit_ = {"reference": fast_score(y, cf_ref, K, self.metric_), "ensemble": fast_score(y, cf_cand, K, self.metric_)}
        say(f"Acceptance rule: lower bound of the gain {rule.lower_bound:+.4f} (needs > 0) -> "
            + (f"ensemble used ({1 + len(full['labs'])} models)" if self.uses_ensemble_ else "best tuned learner kept")
            + f"; total {time.time() - t0:.0f} s")
        info = self.model_["info"]
        self.report_ = {"version": __version__, "rows": len(y), "classes": [str(c) for c in self.classes_], "imbalance_ratio": round(ir, 3),
                        "imbalance_group": self.group_, "metric": self.metric_, "best_learner": info["best_type"],
                        "best_learner_inner_score": round(info["inner_score"], 5),
                        "acceptance_rule": {"lower_bound": round(rule.lower_bound, 5), "gain": round(rule.gain, 5), "margin": MARGIN,
                                            "accepted": bool(rule.accept)},
                        "crossfit_scores": self.crossfit_,
                        "uses_ensemble": self.uses_ensemble_, "models_in_predictor": 1 + (len(self.model_["labs"]) if self.uses_ensemble_ else 0),
                        "options": {"members": self.members, "trials": self.trials, "member_threads": self.member_threads,
                                    "device": self.device_used_},
                        "fit_seconds": round(time.time() - t0, 1)}
        if self.verbose >= 2:
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
