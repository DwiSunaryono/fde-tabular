"""Command line: fde fit | predict | evaluate.

  fde fit --data train.csv --target y --out model.fde [--seed 0] [--jobs 4] [--members full|light]
  fde predict --model model.fde --data new.csv [--proba] [--out predictions.csv]
  fde evaluate --data data.csv --target y [--test-size 0.2] [--seed 0] ...   (stratified holdout; reports TEST scores
              of FDE and of the best tuned learner in the run metric, macro-F1, weighted-F1 and balanced accuracy)
"""
import argparse, json, sys

import numpy as np
import pandas as pd


def _read(path):
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, low_memory=False)


def _clf(a):
    from .classifier import FDEClassifier
    return FDEClassifier(random_state=a.seed, n_jobs=a.jobs, trials=a.trials, members=a.members)


def _scores(y, p):
    from sklearn.metrics import balanced_accuracy_score, f1_score
    return {"macro_f1": round(f1_score(y, p, average="macro"), 5), "weighted_f1": round(f1_score(y, p, average="weighted"), 5),
            "balanced_accuracy": round(balanced_accuracy_score(y, p), 5)}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fde", description="Feasible Diverse Ensembles for tabular classification")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fit", "evaluate"):
        s = sub.add_parser(name); s.add_argument("--data", required=True); s.add_argument("--target", required=True)
        s.add_argument("--seed", type=int, default=0); s.add_argument("--jobs", type=int, default=4); s.add_argument("--trials", type=int, default=20)
        s.add_argument("--members", choices=("full", "light"), default="full")
        if name == "fit":
            s.add_argument("--out", required=True)
        else:
            s.add_argument("--test-size", type=float, default=0.2)
    s = sub.add_parser("predict"); s.add_argument("--model", required=True); s.add_argument("--data", required=True)
    s.add_argument("--proba", action="store_true"); s.add_argument("--out", default=None)
    a = ap.parse_args(argv)
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
    else:
        from sklearn.model_selection import train_test_split
        d = _read(a.data); X, y = d.drop(columns=[a.target]), d[a.target].astype(str).to_numpy()
        tr, te = train_test_split(np.arange(len(y)), test_size=a.test_size, stratify=y, random_state=a.seed)
        clf = _clf(a).fit(X.iloc[tr], y[tr])
        p = clf.predict(X.iloc[te]); best = clf.model_["models"]["P1"]
        from .classifier import prepare_frame
        pb = clf.classes_[best.predict_proba(prepare_frame(X.iloc[te])[clf.columns_]).argmax(1)]
        print(json.dumps({"report": clf.report_, "test": {"fde": _scores(y[te], p), "best_tuned_learner": _scores(y[te], pb)}}, indent=1))


if __name__ == "__main__":
    main()
