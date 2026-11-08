"""Prediction-time check of the "Fast" claim on held-out rows.

For one FDEClassifier fitted with keep_candidates=True, four predictors built from the same fitted models are timed on
the same rows:
  fde     the deployed FDE predictor: the best tuned learner, the members kept by cost-aware pruning, and the combiner
  greedy  greedy ensemble selection (Caruana et al., 2004) over the same members: selection with replacement by Brier
          score on the same out-of-fold probabilities, started from the best tuned learner, at most 25 steps; the
          prediction averages the selected models, weighted by how often each was selected
  full    all members with the combiner fitted on them, without pruning
  best    the best tuned learner alone
Each predictor is timed as one call over all rows, on one CPU core where the system allows pinning (Linux) and with
the numerical libraries limited to one thread: one warm-up call, then the median of `repeats` calls, reported in ms
per 1,000 rows. Data preparation is included in every call; fitting is not. A predictor calls only the models it uses.
When the route keeps the best tuned learner, every predictor reduces to it, as in the accompanying paper.
"""
import os, sys, time
from contextlib import contextmanager

import numpy as np

from .combiners import apply, brier, fast_score
from .extended import raw_frame

# reference values of the accompanying paper (53 evaluation datasets, seed 0, one core per dataset)
PAPER = {"median_ratio": 3.34, "faster": 39, "datasets": 43}


def greedy_select(oof, labs, y, K, steps=25):
    """Greedy ensemble selection with replacement by Brier score on out-of-fold probabilities, started from the best
    tuned learner 'P1'. Returns {label: weight} or None when no member improves the Brier score."""
    pool = ["P1"] + list(labs); cnt = {"P1": 1}; s = oof["P1"].copy(); n = 1; cur = brier(y, s, K)
    for _ in range(steps):
        b, lab = min((brier(y, (s + oof[l]) / (n + 1), K), l) for l in pool)
        if b >= cur - 1e-12:
            break
        s, n, cur = s + oof[lab], n + 1, b; cnt[lab] = cnt.get(lab, 0) + 1
    return None if n == 1 else {l: c / n for l, c in cnt.items()}


@contextmanager
def one_core():
    """Pins the process to one CPU core where possible (Linux) and limits the numerical libraries to one thread.
    Yields True when the process was pinned."""
    from threadpoolctl import threadpool_limits
    old = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
    pinned = False
    if old:
        try:
            os.sched_setaffinity(0, {min(old)}); pinned = True
        except OSError:
            pinned = False
    torch = sys.modules.get("torch"); nt = torch.get_num_threads() if torch else None
    try:
        with threadpool_limits(1):
            if torch:
                torch.set_num_threads(1)
            yield pinned
    finally:
        if torch and nt:
            torch.set_num_threads(nt)
        if pinned:
            os.sched_setaffinity(0, old)


def _timed(fn, repeats):
    fn()                                                   # warm-up call
    w = []
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(); w.append(time.perf_counter() - t0)
    return float(np.median(w))


def speed_report(clf, X, y=None, repeats=5):
    """Times the four predictors of the module docstring on the rows X (and scores them when labels y are given).
    Returns a dict: rows, pinned, and for each predictor the models called, ms per 1,000 rows, and the score."""
    from .classifier import _probs, _used_labs, prepare_frame
    cand = getattr(clf, "candidates_", None)
    if cand is None:
        raise ValueError("fit the classifier with keep_candidates=True to compare prediction times")
    n = len(X); K = len(clf.classes_)
    prep = lambda: (prepare_frame(X)[clf.columns_], raw_frame(X, clf.cat_)[0])

    def probs(models, labs):
        Xp, Xr = prep(); return _probs(models, labs, Xp, Xr)

    used = clf.uses_ensemble_
    sel = greedy_select(cand["oof"], cand["labs"], cand["y"], K) if used else None
    g_labs = [l for l in (sel or {}) if l != "P1"]
    full_labs = _used_labs(cand["spec_full"], cand["labs"])
    def greedy():
        if not sel:
            return probs(cand["models"], [])["P1"]
        p = probs(cand["models"], g_labs)
        return sum(w * p[l] for l, w in sel.items())

    pipes = {"fde": (lambda: clf.predict_proba(X), 1 + len(_used_labs(clf.model_["spec"], clf.model_["labs"]))),
             "greedy": (greedy, 1 + len(g_labs)),
             "full": ((lambda: apply(cand["spec_full"], probs(cand["models"], full_labs))) if used
                      else (lambda: probs(cand["models"], [])["P1"]), 1 + len(full_labs) if used else 1),
             "best": (lambda: probs(cand["models"], [])["P1"], 1)}
    out = {"rows": n, "uses_ensemble": used, "pipelines": {}}
    yy = None
    if y is not None:
        idx = {c: i for i, c in enumerate(clf.classes_)}
        yy = np.array([idx.get(str(v), -1) for v in np.asarray(y)])
    with one_core() as pinned:
        out["pinned_to_one_core"] = pinned
        for name, (fn, models) in pipes.items():
            ms = 1e6 * _timed(fn, repeats) / max(n, 1)
            r = {"models": models, "ms_per_1000_rows": round(ms, 2)}
            if yy is not None:
                r["score"] = round(fast_score(yy, fn().argmax(1), K, clf.metric_), 5)
            out["pipelines"][name] = r
    P = out["pipelines"]
    out["ratio_greedy_to_fde"] = round(P["greedy"]["ms_per_1000_rows"] / P["fde"]["ms_per_1000_rows"], 2)
    out["ratio_full_to_fde"] = round(P["full"]["ms_per_1000_rows"] / P["fde"]["ms_per_1000_rows"], 2)
    out["metric"] = clf.metric_
    out["paper_reference"] = PAPER
    return out
