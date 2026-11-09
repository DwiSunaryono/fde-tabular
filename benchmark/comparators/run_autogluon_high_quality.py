"""AutoML baselines on the TRAIN/TEST split of a stored library (reporting only).

Tools: AutoGluon-Tabular with presets "best_quality" (bagging and stacking; uses the whole budget), each fitted on all TRAIN rows with a wall-clock budget
and CPU_THREADS threads, optimising the library's gate metric (weighted-F1, or macro-F1 for binary data with a minority
share below 0.15), and scored once on TEST. Budgets: 'b300' = 300 s; 'bp1' = the wall time P1 spent tuning on all
TRAIN rows (sum over its twelve learner types, from the library), clipped to [60, 3600] s - the budget of the
strongest comparator. Output: <library>.automl_<tool>_<budget>.json with the TEST score, fit and predict seconds and
the chosen model. Usage: python tools_run/automl_baseline_best.py <library.npz> autogluon_best <b300|bp1>
"""
import json, os, shutil, sys, tempfile, time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, os.getcwd())
from fde2 import data  # noqa: E402

CPU_THREADS = 4


def score(y, p, metric):
    return float(f1_score(y, p, average="macro" if metric == "macroF1" else "weighted"))


def main():
    npz, tool, bud = sys.argv[1], sys.argv[2], sys.argv[3]; b = npz[:-4]; out = f"{b}.automl_{tool}_{bud}.json"
    if os.path.exists(out):
        return
    meta = json.load(open(b + ".json")); z = np.load(npz); metric = meta["gate_metric"]
    idn = meta["identity"]; X, y = data.load(meta["dataset"], idn["source"], idn["data_id"])
    Xtr, Xte = X.iloc[z["train_idx"]].reset_index(drop=True), X.iloc[z["test_idx"]].reset_index(drop=True)
    ytr, yte = z["y_train"], z["y_test"]
    tune = sum(v[0] for v in meta["info"]["full"]["p1_spec"]["tune_seconds"].values())
    budget = 300 if bud == "b300" else int(min(3600, max(60, tune)))
    t0 = time.time(); best = ""
    if tool == "autogluon_best":
        from autogluon.tabular import TabularPredictor
        tmp = tempfile.mkdtemp(prefix="ag_")
        try:
            df = Xtr.copy(); df["__y__"] = ytr
            pr = TabularPredictor(label="__y__", eval_metric="f1_macro" if metric == "macroF1" else "f1_weighted", path=tmp,
                                  verbosity=0, problem_type="binary" if meta["K"] == 2 else "multiclass")
            pr.fit(df, time_limit=budget, presets="best_quality", num_cpus=CPU_THREADS, num_gpus=0)
            fit_s = time.time() - t0; t1 = time.time(); pred = np.asarray(pr.predict(Xte)); pred_s = time.time() - t1
            best = str(pr.model_best)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        from flaml import AutoML
        def wf1(X_val, y_val, estimator, labels, X_train, y_train, *args, **kw):
            v = 1 - score(y_val, estimator.predict(X_val), metric); return v, {"val_loss": v}
        am = AutoML()
        am.fit(Xtr, ytr, task="classification", metric=wf1, time_budget=budget, n_jobs=CPU_THREADS, verbose=0, seed=meta["seed"])
        fit_s = time.time() - t0; t1 = time.time(); pred = np.asarray(am.predict(Xte)); pred_s = time.time() - t1
        best = str(am.best_estimator)
    json.dump({"dataset": meta["dataset"], "seed": meta["seed"], "tool": tool, "budget_name": bud, "budget_s": budget,
               "metric": metric, "test_score": score(yte, pred.astype(int), metric), "fit_s": round(fit_s, 1),
               "predict_s": round(pred_s, 3), "best_model": best, "p1_tune_s": round(tune, 1), "threads": CPU_THREADS},
              open(out, "w"))
    print(meta["dataset"], meta["seed"], tool, bud, "done")


if __name__ == "__main__":
    main()
