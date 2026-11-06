"""Command line: fde demo | fit | predict | evaluate.

  fde demo [--members full|light] [--trials 5] [--jobs -1]   (evaluate on the breast cancer data shipped with
              scikit-learn; no file or download needed; quick settings by default)

  fde fit --data train.csv --target y --out model.fde [--seed 0] [--jobs -1] [--trials 20] [--members full|light]
  fde predict --model model.fde --data new.csv [--proba] [--out predictions.csv]
  fde evaluate --data data.csv --target y [--test-size 0.2] [--seed 0] ...   (holds out a stratified part of the
              data and scores FDE against the best tuned learner on it, reporting macro-F1, weighted-F1 and
              balanced accuracy; the run metric is whichever F1 column applies)
"""
import argparse, json, sys


def _read(path):
    import pandas as pd
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, low_memory=False)


def _clf(a):
    from .classifier import FDEClassifier
    return FDEClassifier(random_state=a.seed, n_jobs=a.jobs, trials=a.trials, members=a.members, device=a.device,
                         verbose=0 if a.quiet else 1)


def _scores(y, p):
    from sklearn.metrics import balanced_accuracy_score, f1_score
    return {"macro_f1": round(f1_score(y, p, average="macro"), 5), "weighted_f1": round(f1_score(y, p, average="weighted"), 5),
            "balanced_accuracy": round(balanced_accuracy_score(y, p), 5)}


NAMES = {"hgb": "HistGradientBoosting", "xgb": "XGBoost", "lgbm": "LightGBM", "cat": "CatBoost", "rf": "random forest",
         "et": "extra trees", "svc": "SVC", "knn": "k-nearest neighbours", "lr": "logistic regression", "nb": "naive Bayes",
         "mlp": "MLP", "dt": "decision tree"}
METRIC = {"wF1": "weighted-F1", "macroF1": "macro-F1"}


def _summary(r, test=None, n_test=None, saved=None):
    """Readable result: data, best learner, acceptance rule, model, time, and the held-out scores when given."""
    o = r["options"]
    lines = ["", "FDE result",
             f"  Data           {r['rows']} training rows, {len(r['classes'])} classes ({', '.join(r['classes'][:6])}"
             f"{', ...' if len(r['classes']) > 6 else ''}), {r['imbalance_group']} (imbalance ratio {r['imbalance_ratio']:.1f})",
             f"  Best learner   {NAMES.get(r['best_learner'], r['best_learner'])} (inner {METRIC[r['metric']]} {r['best_learner_inner_score']:.4f})",
             ("  Ensemble       used: it predicted better than the best tuned learner on the training folds"
              if r["uses_ensemble"] else
              "  Ensemble       not needed: the best tuned learner alone was as good on the training folds"),
             f"  Model          {r['models_in_predictor']} model{'s' if r['models_in_predictor'] > 1 else ''} "
             f"({'best learner, members and combiner' if r['uses_ensemble'] else 'the best tuned learner alone'})",
             f"  Fit time       {r['fit_seconds']:.0f} s ({o['trials']} trials, {o['members']} members, {o['device'].upper()})"]
    if saved:
        lines.append(f"  Saved to       {saved}")
    lines += ["", f"  How {'it was evaluated' if test else 'the model was chosen'} (no row is used both to choose a model and to score it)"]
    step = 1
    if test:
        lines.append(f"    {step}. {n_test} rows were set aside before fitting and used only for the held-out scores below.")
        step += 1
    lines += [f"    {step}. On the training rows, 3 outer folds decided whether the ensemble is used, judged on rows",
              "       that each fold's models had not seen.",
              f"    {step + 1}. Inside every fold, the learners were tuned and the combiner fitted on 3 inner folds; every",
              "       imputer, encoder and scaler was fitted on its own training rows only."]
    if test:
        lines += ["", f"  {'Held-out rows (' + str(n_test) + ')':24s} {'FDE':>8s} {'best tuned learner':>20s}"]
        for k, lab in (("macro_f1", "macro-F1"), ("weighted_f1", "weighted-F1"), ("balanced_accuracy", "balanced accuracy")):
            lines.append(f"    {lab:22s} {test['fde'][k]:8.4f} {test['best_tuned_learner'][k]:20.4f}")
        lines.append("  FDE kept the best tuned learner, so both columns are the same." if not r["uses_ensemble"] else
                     "  FDE used the ensemble; the second column shows the best tuned learner alone.")
        lines += ["  To compare with another tool, score it on the same held-out rows; scores computed on rows that",
                  "  were also used for tuning come out higher than they will be on new data."]
    if not test:
        lines += ["  To see how the model does on new data, use `fde evaluate` or score your own held-out rows."]
    print("\n".join(lines) + "\n")


def _evaluate(X, y, a):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from .classifier import prepare_frame
    y = pd.Series(y).astype(str).to_numpy()
    tr, te = train_test_split(np.arange(len(y)), test_size=a.test_size, stratify=y, random_state=a.seed)
    clf = _clf(a).fit(X.iloc[tr], y[tr])
    p = clf.predict(X.iloc[te]); best = clf.model_["models"]["P1"]
    pb = clf.classes_[best.predict_proba(prepare_frame(X.iloc[te])[clf.columns_]).argmax(1)]
    test = {"fde": _scores(y[te], p), "best_tuned_learner": _scores(y[te], pb)}
    if a.json:
        print(json.dumps({"report": clf.report_, "test": test}, indent=1))
    else:
        _summary(clf.report_, test, len(te))


def _common(s, trials=20, members="full"):
    s.add_argument("--seed", type=int, default=0, help="seed shared by every split, search and member (default: 0)")
    s.add_argument("--jobs", type=int, default=-1, help="parallel tasks; -1 chooses from the free CPU cores and memory (default: -1)")
    s.add_argument("--trials", type=int, default=trials, help=f"Optuna trials for each learner type (default: {trials})")
    s.add_argument("--members", choices=("full", "light"), default=members,
                   help=f"light keeps the imbalance-handling members only: faster, slightly less accurate (default: {members})")
    s.add_argument("--device", choices=("cpu", "gpu"), default="cpu",
                   help="gpu trains XGBoost, CatBoost and the MLP on a CUDA GPU: faster on large data, results can differ "
                        "slightly from the CPU (default: cpu)")
    s.add_argument("--quiet", action="store_true", help="no progress messages")
    s.add_argument("--json", action="store_true", help="print the full report as JSON instead of the summary")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fde", description="Feasible Diverse Ensembles for tabular classification")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, text in (("fit", "fit FDE and save the model"),
                       ("evaluate", "fit on part of the data, then score the holdout")):
        s = sub.add_parser(name, help=text, description=text)
        s.add_argument("--data", required=True, help="input data (CSV or Parquet)")
        s.add_argument("--target", required=True, help="column that holds the labels")
        _common(s)
        if name == "fit":
            s.add_argument("--out", required=True, help="where to save the fitted model")
        else:
            s.add_argument("--test-size", type=float, default=0.2,
                           help="fraction of rows held out, stratified by class (default: 0.2)")
    text = ("evaluate FDE on the breast cancer data shipped with scikit-learn (no file or download needed); quick "
            "settings by default (5 trials, light members)")
    s = sub.add_parser("demo", help=text, description=text); _common(s, trials=5, members="light")
    s.add_argument("--test-size", type=float, default=0.2, help="fraction of rows held out, stratified by class (default: 0.2)")
    s = sub.add_parser("predict", help="score new rows with a fitted model, as labels or probabilities",
                       description="score new rows with a fitted model, as labels or probabilities")
    s.add_argument("--model", required=True, help="model saved by fde fit")
    s.add_argument("--data", required=True, help="input data (CSV or Parquet)")
    s.add_argument("--proba", action="store_true", help="write class probabilities rather than labels")
    s.add_argument("--out", default=None, help="CSV to write (default: standard output)")
    a = ap.parse_args(argv)
    if a.cmd in ("demo", "fit", "evaluate") and not a.quiet:
        print("[fde] loading the learner libraries (a few seconds on the first run) ...", file=sys.stderr, flush=True)
    if a.cmd == "fit":
        d = _read(a.data); clf = _clf(a).fit(d.drop(columns=[a.target]), d[a.target]); clf.save(a.out)
        if a.json:
            print(json.dumps(clf.report_, indent=1))
        else:
            _summary(clf.report_, saved=a.out)
    elif a.cmd == "predict":
        import pandas as pd
        from .classifier import FDEClassifier
        clf = FDEClassifier.load(a.model); d = _read(a.data)
        out = pd.DataFrame(clf.predict_proba(d), columns=[f"p_{c}" for c in clf.classes_]) if a.proba else pd.DataFrame({"prediction": clf.predict(d)})
        if a.out:
            out.to_csv(a.out, index=False)
        else:
            out.to_csv(sys.stdout, index=False)
    elif a.cmd == "demo":
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer(as_frame=True)
        _ = a.quiet or print(f"[fde] demo: breast cancer data (569 rows, 30 columns, 2 classes); quick settings: {a.trials} trials, "
              f"{a.members} members (the paper's settings: --trials 20 --members full)", file=sys.stderr, flush=True)
        _evaluate(d.data, d.target.map(dict(enumerate(d.target_names))), a)
    else:
        d = _read(a.data); _evaluate(d.drop(columns=[a.target]), d[a.target], a)


if __name__ == "__main__":
    main()
