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
    """Readable result on stdout: the decision and the held-out scores first (highlighted), then the details, the
    reasons and how the data were split. Plain text when stdout is not a terminal."""
    from rich import box
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    con, o, used = Console(highlight=False), r["options"], r["uses_ensemble"]
    n = r["models_in_predictor"]
    head = (Text(f"Ensemble used ({n} models: best learner, members and combiner)", style="bold green") if used else
            Text("Best tuned learner kept (one model; the ensemble was not needed)", style="bold cyan"))
    parts = [head]
    if test:
        t = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style="bold")
        t.add_column(f"held-out rows ({n_test})"); t.add_column("FDE", justify="right"); t.add_column("best tuned learner", justify="right")
        for k, lab in (("macro_f1", "macro-F1"), ("weighted_f1", "weighted-F1"), ("balanced_accuracy", "balanced accuracy")):
            a, b = test["fde"][k], test["best_tuned_learner"][k]
            t.add_row(lab, Text(f"{a:.4f}", style="bold green" if a > b else ""), Text(f"{b:.4f}", style="bold green" if b > a else ""))
        parts += [Text(""), t, Text("FDE kept the best tuned learner, so both columns are the same." if not used else
                                    "Higher scores are in green.", style="dim")]
    con.print(); con.print(Panel(Group(*parts), title="[bold]FDE result[/]", border_style="green" if used else "cyan", expand=False))
    g = Table.grid(padding=(0, 2)); g.add_column(style="bold"); g.add_column()
    cls = ", ".join(r["classes"][:6]) + (", ..." if len(r["classes"]) > 6 else "")
    imb = "balanced" if r["imbalance_group"] == "balanced" else f"{r['imbalance_group']} imbalance"
    g.add_row("Data", f"{r['rows']} training rows, {len(r['classes'])} classes ({cls}), {imb} (ratio {r['imbalance_ratio']:.1f})")
    g.add_row("Best learner", f"{NAMES.get(r['best_learner'], r['best_learner'])} (inner {METRIC[r['metric']]} {r['best_learner_inner_score']:.4f})")
    g.add_row("Fit time", f"{r['fit_seconds']:.0f} s ({o['trials']} trials, {o['members']} members, {o['device'].upper()})")
    if saved:
        g.add_row("Saved to", saved)
    g.add_row("Why", _why(r))
    con.print(g)
    steps = ([f"{n_test} rows were set aside before fitting and used only for the held-out scores."] if test else []) + [
        "On the training rows, 3 outer folds decided whether the ensemble is used, judged on rows that each fold's "
        "models had not seen.",
        "Inside every fold, the learners were tuned and the combiner fitted on 3 inner folds; every imputer, encoder "
        "and scaler was fitted on its own training rows only."]
    con.print(f"\n[bold]How {'it was evaluated' if test else 'the model was chosen'}[/] [dim](no row is used both to choose a model and to score it)[/]")
    from rich.padding import Padding
    lst = Table.grid(padding=(0, 1)); lst.add_column(justify="right", style="dim"); lst.add_column(style="dim")
    for i, x in enumerate(steps, 1):
        lst.add_row(f"{i}.", x)
    lst.add_row("", "To compare with another tool, score it on the same held-out rows; scores computed on rows that were "
                    "also used for tuning come out higher than they will be on new data." if test else
                    "To see how the model does on new data, use `fde evaluate` or score your own held-out rows.")
    con.print(Padding(lst, (0, 0, 0, 2)))
    con.print()


# share of runs that used the ensemble in the accompanying study (53 datasets x 5 seeds), by imbalance group
STUDY = {"balanced": 8, "low": 54, "moderate": 83, "extreme": 83}
GROUP = {"balanced": "balanced data (imbalance ratio below 3)", "low": "data with low imbalance (ratio 3 to 10)",
         "moderate": "data with moderate to extreme imbalance (ratio 10 or more)",
         "extreme": "data with moderate to extreme imbalance (ratio 10 or more)"}


def _why(r):
    """Plain-language reasons for the decision, from the cross-fitted scores of the training folds."""
    cf, m = r["crossfit_scores"], METRIC[r["metric"]]
    ref, ens, d = cf["reference"], cf["ensemble"], r.get("decision")
    seen = (f"On training rows that each fold's models had not seen, the ensemble scored {ens:.4f} and the best tuned "
            f"learner {ref:.4f} ({m}).")
    if d == "used":
        why = (" The ensemble was better, and its advantage held in at least 95% of the bootstrap resamples of those rows, "
               "so FDE uses it. Its members (learners refitted with class weights, on rebalanced samples, and "
               "imbalance-specific ensembles" + (", plus large-budget GBDTs, an MLP and large forests" if r["options"]["members"] == "full" else "")
               + ") correct some of the errors of the best tuned learner.")
    elif d == "not_consistent" and ens > ref:
        why = (" The ensemble scored a little higher, but the advantage did not hold in at least 95% of the bootstrap "
               "resamples, so it may be chance. FDE keeps the single model, which is simpler and faster to predict.")
    elif d == "not_consistent":
        why = (" The ensemble was not better, so FDE keeps the single model, which is simpler and faster to predict.")
    elif d == "same_predictions":
        why = " The ensemble made the same predictions as the best tuned learner, so FDE keeps the single model."
    else:
        why = " The combiner gave the members no weight, so FDE keeps the single model."
    ctx = (f" For comparison: in the accompanying study (full member set), the ensemble was used in "
           f"{STUDY[r['imbalance_group']]}% of runs on {GROUP[r['imbalance_group']]}.")
    return seen + why + ctx


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
    text = ("evaluate FDE on example data, no file or download needed; quick settings by default (5 trials, light "
            "members). --example breast-cancer: the best tuned learner is enough; --example imbalanced: the ensemble is used")
    s = sub.add_parser("demo", help=text, description=text); _common(s, trials=5, members="light")
    s.add_argument("--example", choices=("breast-cancer", "imbalanced"), default="breast-cancer",
                   help="breast-cancer: 569 rows, balanced classes; imbalanced: 2,000 synthetic rows, 10%% minority (default: breast-cancer)")
    s.add_argument("--test-size", type=float, default=0.2, help="fraction of rows held out, stratified by class (default: 0.2)")
    s = sub.add_parser("predict", help="score new rows with a fitted model, as labels or probabilities",
                       description="score new rows with a fitted model, as labels or probabilities")
    s.add_argument("--model", required=True, help="model saved by fde fit")
    s.add_argument("--data", required=True, help="input data (CSV or Parquet)")
    s.add_argument("--proba", action="store_true", help="write class probabilities rather than labels")
    s.add_argument("--out", default=None, help="CSV to write (default: standard output)")
    a = ap.parse_args(argv)
    if a.cmd in ("demo", "fit", "evaluate") and not a.quiet:
        print("fde loading the learner libraries (a few seconds on the first run) ...", file=sys.stderr, flush=True)
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
        quick = f"quick settings: {a.trials} trials, {a.members} members (the paper's settings: --trials 20 --members full)"
        if a.example == "breast-cancer":
            from sklearn.datasets import load_breast_cancer
            d = load_breast_cancer(as_frame=True); X, y = d.data, d.target.map(dict(enumerate(d.target_names)))
            what = "breast cancer data (569 rows, 30 columns, 2 balanced classes)"
        else:
            import numpy as np
            import pandas as pd
            from sklearn.datasets import make_classification
            Xa, ya = make_classification(n_samples=2000, n_features=20, n_informative=6, n_redundant=4, weights=[0.9, 0.1],
                                         flip_y=0.02, class_sep=0.8, random_state=1)
            X, y = pd.DataFrame(Xa, columns=[f"x{i}" for i in range(20)]), np.where(ya == 1, "minority", "majority")
            what = "synthetic imbalanced data (2,000 rows, 20 columns, 10% minority class)"
        if not a.quiet:
            from rich.console import Console
            Console(stderr=True, highlight=False).print(f"[bold cyan]fde[/] demo: {what}; [dim]{quick}[/]")
        _evaluate(X, y, a)
    else:
        d = _read(a.data); _evaluate(d.drop(columns=[a.target]), d[a.target], a)


if __name__ == "__main__":
    main()
