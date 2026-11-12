"""Outer cross-validation of FDE: every row is scored once, by a model fitted without it.

The data are split into `cv` stratified folds. For each fold, FDE is fitted on the other folds with all its inner
steps (tuning, members, combiner, route, sizing), and FDE and its best tuned learner are scored on the held-out fold.
The result lists the scores of every fold and their mean and standard deviation, so that a single split is not the
only evidence. The out-of-fold scores computed inside a fit are not used here, since they also served to choose the
members and are therefore optimistic.
"""
import numpy as np
import pandas as pd

METRICS = ("macro_f1", "weighted_f1", "balanced_accuracy")


def _scores(y, p):
    from sklearn.metrics import balanced_accuracy_score, f1_score
    return {"macro_f1": round(f1_score(y, p, average="macro"), 5), "weighted_f1": round(f1_score(y, p, average="weighted"), 5),
            "balanced_accuracy": round(balanced_accuracy_score(y, p), 5)}


def cross_validate(X, y, cv=5, random_state=0, on_fold=None, **params):
    """Score FDE and its best tuned learner by outer stratified `cv`-fold cross-validation.

    X : pandas DataFrame; y : labels; cv : number of outer folds (at least 2); random_state : seed of the fold split
    and of every fit; params : further FDEClassifier parameters (trials, members, n_jobs, device, verbose, ...);
    on_fold : optional callable(k, cv), called before fold k (1-based) is fitted.

    Returns a dict with "folds" (per fold: rows, route, ensemble use, models in the predictor, and the scores of
    "fde" and "best_tuned_learner"), "mean" and "sd" (over folds, per predictor and metric), "metric" (the F1 that
    FDE selects with on these data) and "cv".
    """
    from sklearn.model_selection import StratifiedKFold
    from .classifier import FDEClassifier, prepare_frame
    if int(cv) < 2:
        raise ValueError("cv must be at least 2")
    X = X.reset_index(drop=True)
    y = pd.Series(y).astype(str).to_numpy()
    folds, metric = [], None
    for k, (tr, te) in enumerate(StratifiedKFold(n_splits=int(cv), shuffle=True, random_state=random_state).split(X, y), 1):
        if on_fold:
            on_fold(k, int(cv))
        clf = FDEClassifier(random_state=random_state, **params).fit(X.iloc[tr], y[tr])
        best = clf.model_["models"]["P1"]
        pb = clf.classes_[best.predict_proba(prepare_frame(X.iloc[te])[clf.columns_]).argmax(1)]
        r = clf.report_; metric = r["metric"]
        folds.append({"fold": k, "rows": len(te), "route": r["route"], "uses_ensemble": r["uses_ensemble"],
                      "models_in_predictor": r["models_in_predictor"],
                      "fde": _scores(y[te], clf.predict(X.iloc[te])), "best_tuned_learner": _scores(y[te], pb)})
    summary = {s: {p: {m: round(float(f(np.array([x[p][m] for x in folds]))), 5) for m in METRICS}
                   for p in ("fde", "best_tuned_learner")}
               for s, f in (("mean", np.mean), ("sd", lambda v: np.std(v, ddof=1)))}
    return {"cv": int(cv), "metric": metric, "folds": folds, **summary}
