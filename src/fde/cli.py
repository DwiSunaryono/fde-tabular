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

import numpy as np
import pandas as pd


def _read(path):
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, low_memory=False)


def _clf(a):
    from .classifier import FDEClassifier
    return FDEClassifier(random_state=a.seed, n_jobs=a.jobs, trials=a.trials, members=a.members, verbose=0 if a.quiet else 1)


def _scores(y, p):
    from sklearn.metrics import balanced_accuracy_score, f1_score
    return {"macro_f1": round(f1_score(y, p, average="macro"), 5), "weighted_f1": round(f1_score(y, p, average="weighted"), 5),
            "balanced_accuracy": round(balanced_accuracy_score(y, p), 5)}


def _evaluate(X, y, a):
    from sklearn.model_selection import train_test_split
    from .classifier import prepare_frame
    y = pd.Series(y).astype(str).to_numpy()
    tr, te = train_test_split(np.arange(len(y)), test_size=a.test_size, stratify=y, random_state=a.seed)
    clf = _clf(a).fit(X.iloc[tr], y[tr])
    p = clf.predict(X.iloc[te]); best = clf.model_["models"]["P1"]
    pb = clf.classes_[best.predict_proba(prepare_frame(X.iloc[te])[clf.columns_]).argmax(1)]
    print(json.dumps({"report": clf.report_, "test": {"fde": _scores(y[te], p), "best_tuned_learner": _scores(y[te], pb)}}, indent=1))


def _common(s, trials=20, members="full"):
    s.add_argument("--seed", type=int, default=0, help="seed shared by every split, search and member (default: 0)")
    s.add_argument("--jobs", type=int, default=-1, help="parallel tasks; -1 chooses from the free CPU cores and memory (default: -1)")
    s.add_argument("--trials", type=int, default=trials, help=f"Optuna trials for each learner type (default: {trials})")
    s.add_argument("--members", choices=("full", "light"), default=members,
                   help=f"light keeps the imbalance-handling members only: faster, slightly less accurate (default: {members})")
    s.add_argument("--quiet", action="store_true", help="no progress messages")


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
        print("[fde] loading the learners ...", file=sys.stderr, flush=True)
    if a.cmd == "fit":
        d = _read(a.data); clf = _clf(a).fit(d.drop(columns=[a.target]), d[a.target]); clf.save(a.out)
        print(json.dumps(clf.report_, indent=1))
    elif a.cmd == "predict":
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
        print(f"[fde] demo: breast cancer data (569 rows, 30 columns, 2 classes); quick settings: {a.trials} trials, "
              f"{a.members} members (the paper's settings: --trials 20 --members full)", file=sys.stderr, flush=True)
        _evaluate(d.data, d.target.map(dict(enumerate(d.target_names))), a)
    else:
        d = _read(a.data); _evaluate(d.drop(columns=[a.target]), d[a.target], a)


if __name__ == "__main__":
    main()
