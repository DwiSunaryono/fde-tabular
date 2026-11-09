"""Auto-sklearn baseline on the TRAIN/TEST split of a stored library (reporting only), in its own environment.

Auto-sklearn 1.0 (AutoSklearnClassifier, meta-learning and ensemble selection) is used because Auto-sklearn 2.0 supports
only accuracy, balanced accuracy, ROC AUC and log loss, not the F1 metrics of the study.

Same split, metric and budget as the other AutoML baselines: the run metric of the library (weighted-F1, or macro-F1
for two classes with a minority share below 0.15), the bp1 budget (the wall time P1 spent tuning on all TRAIN rows,
clipped to [60, 3600] s), 4 parallel workers, fitted on all TRAIN rows and scored once on TEST. Categorical columns
are declared as such (feat_type). Output: <out dir>/<dataset>_seed<s>.automl_autosklearn_bp1.json with the TEST score,
fit and predict seconds and the number of models in the final ensemble.
Usage: python tools_run/automl_autosklearn.py <library.npz> <askl data dir> <out dir>
"""
import json, os, shutil, sys, tempfile, time
import numpy as np

WORKERS, MEMORY_MB = 4, 6144


def score(y, p, metric):
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, average="macro" if metric == "macroF1" else "weighted"))


def main():
    npz, ddir, odir = sys.argv[1:4]; b = npz[:-4]; meta = json.load(open(b + ".json"))
    out = os.path.join(odir, f"{meta['dataset']}_seed{meta['seed']}.automl_autosklearn_bp1.json")
    if os.path.exists(out):
        return
    z = np.load(npz); d = np.load(os.path.join(ddir, meta["dataset"] + ".npz"))
    X, cat = d["X"], d["cat"]; tr, te = z["train_idx"], z["test_idx"]; ytr, yte = z["y_train"], z["y_test"]
    assert (d["y"][tr] == ytr).all() and (d["y"][te] == yte).all(), "labels differ from the library"
    metric = meta["gate_metric"]
    tune = sum(v[0] for v in meta["info"]["full"]["p1_spec"]["tune_seconds"].values())
    budget = int(min(3600, max(60, tune)))
    import autosklearn.metrics as M
    from autosklearn.classification import AutoSklearnClassifier
    tmp = tempfile.mkdtemp(prefix="askl_")
    try:
        am = AutoSklearnClassifier(time_left_for_this_task=budget, n_jobs=WORKERS, seed=int(meta["seed"]), memory_limit=MEMORY_MB,
                                    metric=M.f1_macro if metric == "macroF1" else M.f1_weighted, tmp_folder=os.path.join(tmp, "w"),
                                    delete_tmp_folder_after_terminate=True)
        t0 = time.time()
        am.fit(X[tr], ytr, feat_type=["Categorical" if c else "Numerical" for c in cat])
        fit_s = time.time() - t0; t1 = time.time(); pred = am.predict(X[te]); pred_s = time.time() - t1
        n_models = len(am.get_models_with_weights())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    json.dump({"dataset": meta["dataset"], "seed": meta["seed"], "tool": "autosklearn", "budget_name": "bp1", "budget_s": budget,
               "metric": metric, "test_score": score(yte, pred.astype(int), metric), "fit_s": round(fit_s, 1), "predict_s": round(pred_s, 3),
               "n_models": n_models, "workers": WORKERS}, open(out, "w"))
    print(meta["dataset"], meta["seed"], "autosklearn done")


if __name__ == "__main__":
    main()
