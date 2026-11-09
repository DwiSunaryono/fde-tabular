"""Command line: fde demo | fit | predict | evaluate.

  fde demo [--example imbalanced|breast-cancer] [--tables 5] [--quick] [--no-speed] [--members full|light] [--trials 5] [--jobs -1]
              (no file or download needed; quick settings by default. The default example is imbalanced, where the
              ensemble is used, and the prediction times are measured; breast-cancer shows the balanced case, where
              the best tuned learner is kept)

  fde fit --data train.csv --target y --out model.fde [--seed 0] [--jobs -1] [--trials 20] [--members full|light]
  fde predict --model model.fde --data new.csv [--proba] [--out predictions.csv]
  fde evaluate --data data.csv --target y [--test-size 0.2] [--seed 0] ...   (holds out a stratified part of the
              data and scores FDE against the best tuned learner on it, reporting macro-F1, weighted-F1 and
              balanced accuracy; the run metric is whichever F1 column applies)
  fde evaluate ... --speed, fde demo --speed   (also times, on the held-out rows, the FDE predictor against greedy
              ensemble selection over the same members, the full member set, and the best tuned learner; see fde.speed)
"""
import argparse, json, sys


def _read(path):
    import pandas as pd
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, low_memory=False)


def _clf(a):
    from .classifier import FDEClassifier
    return FDEClassifier(random_state=a.seed, n_jobs=a.jobs, trials=a.trials, members=a.members, device=a.device,
                         verbose=0 if a.quiet else 1, keep_candidates=getattr(a, "speed", False))


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
    head = (Text(f"Ensemble used ({n} models: best learner, members and combiner)", style="bold green") if used and r.get("stack_kept", True) else
            Text("Ensemble route: the stack was rejected, so the best tuned learner with the class-wise score adjustment is used (one model)",
                 style="bold green") if used else
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
        ("The classes are balanced, so 3 outer folds of the training rows decided whether the ensemble is used, judged on "
         "rows that each fold's models had not seen." if r["route"] == "acceptance_rule" else
         "The classes are imbalanced, so the ensemble is used without outer folds."),
        "In every fit, the learners were tuned and the combiner fitted on 3 inner folds, and the costliest members were dropped "
        "while the score stayed at that of all members; every imputer, encoder and scaler was fitted on its own training "
        "rows only."]
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
STUDY = {"balanced": 8, "low": 96, "moderate": 99, "extreme": 99}
GROUP = {"balanced": "balanced data (imbalance ratio below 3)", "low": "data with low imbalance (ratio 3 to 10)",
         "moderate": "data with moderate to extreme imbalance (ratio 10 or more)",
         "extreme": "data with moderate to extreme imbalance (ratio 10 or more)"}


def _why(r):
    """Plain-language reasons for the decision, from the cross-fitted scores of the training folds."""
    if r["route"] == "ensemble":
        return (f"The classes are imbalanced (ratio {r['imbalance_ratio']:.1f}), so FDE uses the ensemble. It built {r['models_built']} models and kept "
                f"{r['models_in_predictor']}, dropping the costliest members while the out-of-fold score stayed at that of all of them.")
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
    speed = None
    if getattr(a, "speed", False):
        from .speed import speed_report
        speed = speed_report(clf, X.iloc[te].reset_index(drop=True), y[te], repeats=a.repeats)
    check = None
    want_check = (bool(speed) and getattr(a, "cmd", "") == "demo" and a.example == "imbalanced"
                  and not getattr(a, "quick", False) and getattr(a, "tables", 1) > 1)
    if want_check and a.json:
        check = _speed_check(a, speed)
    if a.json:
        print(json.dumps({"report": clf.report_, "test": test, **({"speed": speed} if speed else {}),
                          **({"speed_check": check} if check else {})}, indent=1))
    else:
        _summary(clf.report_, test, len(te))
        if speed:
            _speed_print(speed)
        if want_check:
            check = _speed_check(a, speed)
            _speed_check_print(check)
        if getattr(a, "cmd", "") == "demo":
            _demo_takeaway(clf.report_, test, speed, a.example, check)


DEMO_SETS = [(20, 0.10, 1), (15, 0.15, 2), (30, 0.08, 3), (25, 0.12, 4), (20, 0.20, 5),       # columns, minority share, seed
             (40, 0.10, 6), (12, 0.18, 7), (35, 0.06, 8), (18, 0.12, 9), (28, 0.15, 10)]
GROUP_SIZE = 5


def _demo_set(k):
    """Settings of demo table k (0-based): the ten listed above, then columns and minority share in turn, seed k + 1."""
    if k < len(DEMO_SETS):
        return DEMO_SETS[k]
    return (12 + (7 * k) % 31, (0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20)[k % 7], k + 1)


def _demo_data(n_features, minority, seed):
    """One synthetic imbalanced table of 2,000 rows; the first set is the main example of the demo."""
    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_classification
    Xa, ya = make_classification(n_samples=2000, n_features=n_features, n_informative=6, n_redundant=4,
                                 weights=[1 - minority, minority], flip_y=0.02, class_sep=0.8, random_state=seed)
    return pd.DataFrame(Xa, columns=[f"x{i}" for i in range(n_features)]), np.where(ya == 1, "minority", "majority")


def _speed_check(a, first):
    """Prediction-time comparison with greedy ensemble selection on the demo tables (five by default), as the paper reports it: the
    median ratio over datasets, since a single dataset can go either way (FDE was slower on 4 of the paper's 43)."""
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from .classifier import FDEClassifier
    from .speed import speed_report
    n_tables = max(1, a.tables)
    rows = [{"columns": DEMO_SETS[0][0], "minority": DEMO_SETS[0][1], "fde_models": first["pipelines"]["fde"]["models"],
             "greedy_models": first["pipelines"]["greedy"]["models"], "ratio": first["ratio_greedy_to_fde"]}]
    import contextlib
    bar = contextlib.nullcontext(None)
    if not a.quiet:
        from rich.console import Console
        from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
        bar = Progress(SpinnerColumn(), TextColumn("[bold cyan]fde[/] prediction-time check"), BarColumn(), MofNCompleteColumn(),
                       TextColumn("{task.description}"), TimeElapsedColumn(), console=Console(stderr=True), transient=True)
    with bar as prog:
        task = prog.add_task("table 1 (the example above) done", total=n_tables, completed=1) if prog else None
        for k in range(2, n_tables + 1):
            nf, mino, seed = _demo_set(k - 1)
            if prog:
                prog.update(task, description=f"table {k}: fitting ({nf} columns, {mino:.0%} minority)")
            X, y = _demo_data(nf, mino, seed); y = pd.Series(y).astype(str).to_numpy()
            tr, te = train_test_split(np.arange(len(y)), test_size=a.test_size, stratify=y, random_state=a.seed)
            clf = FDEClassifier(random_state=a.seed, n_jobs=a.jobs, trials=a.trials, members=a.members, device=a.device,
                                verbose=0, keep_candidates=True).fit(X.iloc[tr], y[tr])
            if prog:
                prog.update(task, description=f"table {k}: timing the four predictors")
            sp = speed_report(clf, X.iloc[te].reset_index(drop=True), y[te], repeats=a.repeats)
            rows.append({"columns": nf, "minority": mino, "fde_models": sp["pipelines"]["fde"]["models"],
                         "greedy_models": sp["pipelines"]["greedy"]["models"], "ratio": sp["ratio_greedy_to_fde"],
                         "ensemble": sp["uses_ensemble"]})
            if prog:
                prog.advance(task)
    r = [x["ratio"] for x in rows]
    return {"tables": rows, "median_ratio": float(np.median(r)), "faster": int(sum(x > 1 for x in r)), "n": len(r),
            "paper_reference": first["paper_reference"]}


def _speed_check_print(c):
    """The tables in groups of five, each with its median, then the median over all tables."""
    import numpy as np
    from rich import box
    from rich.console import Console
    from rich.table import Table
    con = Console(highlight=False); T = c["tables"]
    for g0 in range(0, len(T), GROUP_SIZE):
        part = T[g0:g0 + GROUP_SIZE]; r = [x["ratio"] for x in part]
        t = Table(box=box.SIMPLE_HEAD, title=f"[bold]Prediction-time check, tables {g0 + 1}-{g0 + len(part)} of {len(T)}[/]",
                  title_justify="left")
        for col, j in (("table", "right"), ("columns", "right"), ("minority", "right"), ("FDE models", "right"),
                       ("greedy models", "right"), ("greedy / FDE time", "right")):
            t.add_column(col, justify=j)
        for i, x in enumerate(part, g0 + 1):
            g = x["ratio"]
            t.add_row(str(i), str(x["columns"]), f"{x['minority']:.0%}", str(x["fde_models"]), str(x["greedy_models"]),
                      f"[green]{g:.2f}[/]" if g > 1 else f"[red]{g:.2f}[/]")
        con.print(t)
        con.print(f"  Median of tables {g0 + 1}-{g0 + len(part)}: [bold]{float(np.median(r)):.2f}[/]; FDE faster on "
                  f"{sum(x > 1 for x in r)} of {len(r)}.\n")
    ref = c["paper_reference"]
    con.print(f"[bold]Over all {c['n']} tables: median greedy / FDE time {c['median_ratio']:.2f}; FDE faster on {c['faster']} of {c['n']}.[/]")
    con.print("Table 1 is the example above. A single table can go either way: greedy selection is faster when it keeps fewer "
              "or cheaper models, so the median over tables is the quantity to read, as in the paper "
              f"({ref['median_ratio']:.2f} on {ref['datasets']} datasets, FDE faster on {ref['faster']}). These quick tables use the "
              "light member set, whose members have similar prediction costs; ordering by cost matters most when costly members "
              "are present, as in the paper's full member set on real datasets.")
    con.print()


def _demo_takeaway(r, test, speed, example, check=None):
    """Three lines that say what the demo has just shown."""
    from rich.console import Console
    from rich.panel import Panel
    con = Console(highlight=False); m = "macro_f1" if r["metric"] == "macroF1" else "weighted_f1"; lab = METRIC[r["metric"]]
    f, b = test["fde"][m], test["best_tuned_learner"][m]
    lines = [f"1. Accuracy: on the held-out rows, FDE scores {f:.4f} {lab} against {b:.4f} for the best tuned learner"
             + (" (the same model here, because the route kept it)." if not r["uses_ensemble"] else ".")]
    if check:
        lines.append(f"2. Fast: over {check['n']} synthetic tables, FDE predicts {check['median_ratio']:.2f} times faster than greedy "
                     f"ensemble selection over the same members (median; FDE faster on {check['faster']} of {check['n']}; paper: median "
                     f"{check['paper_reference']['median_ratio']:.2f} on {check['paper_reference']['datasets']} datasets).")
    elif speed and r["uses_ensemble"]:
        g = speed["ratio_greedy_to_fde"]
        lines.append(f"2. Fast: FDE calls {speed['pipelines']['fde']['models']} models and predicts {g:.2f} times "
                     + ("faster than" if g >= 1 else "as fast as") + " greedy ensemble selection over the same members"
                     + f" (paper: median {speed['paper_reference']['median_ratio']:.2f} on {speed['paper_reference']['datasets']} datasets).")
    elif speed:
        lines.append("2. Fast: the route kept one model, which is the fastest predictor; see the imbalanced example for the ensemble.")
    else:
        lines.append("2. Fast: run the demo without --no-speed to compare prediction times with greedy ensemble selection.")
    lines.append("3. Balanced data: fde demo --example breast-cancer shows FDE keeping the best tuned learner when the ensemble does not help."
                 if example == "imbalanced" else
                 "3. Imbalanced data: fde demo (default example) shows the ensemble in use and its prediction time.")
    con.print(Panel("\n".join(lines), title="[bold]What this demo shows[/]", border_style="green", expand=False)); con.print()


def _speed_print(s):
    """Prediction times of the four predictors and how they were measured."""
    from rich import box
    from rich.console import Console
    from rich.table import Table
    con = Console(highlight=False); P = s["pipelines"]; m = METRIC[s["metric"]]
    t = Table(box=box.SIMPLE_HEAD, title=f"[bold]Prediction time on the {s['rows']} held-out rows[/]", title_justify="left")
    for c, j in (("predictor", "left"), ("models", "right"), ("ms/1,000 rows", "right"), (m, "right")):
        t.add_column(c, justify=j)
    lab = {"fde": "FDE", "greedy": "greedy selection", "full": "full member set", "best": "best tuned learner"}
    for k in ("fde", "greedy", "full", "best"):
        r = P[k]; t.add_row(lab[k], str(r["models"]), f"{r['ms_per_1000_rows']:.1f}", f"{r['score']:.4f}" if "score" in r else "-")
    con.print(t)
    g, f = s["ratio_greedy_to_fde"], s["ratio_full_to_fde"]
    if s["uses_ensemble"]:
        con.print(f"[bold]Greedy selection / FDE time: {g:.2f}[/] ({'FDE faster' if g > 1 else 'greedy selection faster' if g < 1 else 'equal'}); "
                  f"full member set / FDE: {f:.2f}. A ratio above 1 means FDE predicts faster.")
    else:
        con.print("[bold]The route kept the best tuned learner[/], so every predictor reduces to it and the times differ only by noise.")
    ref = s["paper_reference"]
    notes = [("Each predictor is one call over all held-out rows, timed on one CPU core" +
              (" (the process was pinned to one core)" if s["pinned_to_one_core"] else " (numerical libraries limited to one thread; "
               "pinning to one core is available on Linux)") + ": one warm-up call, then the median of the repeated calls, divided by "
              "the number of rows. Data preparation is included; fitting is not."),
             ("All four predictors use the same fitted models. Greedy ensemble selection (Caruana et al., 2004) chooses members with "
              "replacement by Brier score on the same out-of-fold probabilities, starting from the best tuned learner."),
             ("FDE removes members in order of a fixed table of prediction costs while the out-of-fold score of the full member set "
              "is kept, so it is usually faster than greedy selection; greedy selection is faster when it keeps fewer or cheaper models, "
              "and FDE then keeps a member that its score needs."),
             (f"Reference from the accompanying paper: on {ref['datasets']} evaluation datasets on the ensemble route, the median ratio of "
              f"greedy selection to FDE time was {ref['median_ratio']:.2f}, with FDE faster on {ref['faster']}. Times depend on the "
              "machine; the ratio, measured on one core, is the comparable quantity.")]
    from rich.padding import Padding
    lst = Table.grid(padding=(0, 1)); lst.add_column(justify="right", style="dim"); lst.add_column(style="dim")
    for i, x in enumerate(notes, 1):
        lst.add_row(f"{i}.", x)
    con.print("\n[bold]How the times were measured[/]"); con.print(Padding(lst, (0, 0, 0, 2))); con.print()


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


def _speed_opts(s):
    s.add_argument("--speed", action="store_true",
                   help="also time, on the held-out rows and on one CPU core, the FDE predictor against greedy ensemble selection "
                        "over the same members, the full member set and the best tuned learner (the 'Fast' claim)")
    s.add_argument("--repeats", type=int, default=5, help="timed calls per predictor with --speed; the median is reported (default: 5)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fde", description="FDE: Fast-Deployment Ensemble for imbalanced tabular classification")
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
            _speed_opts(s)
    text = ("show what FDE does on example data, no file or download needed; quick settings by default (5 trials, light "
            "members). The default example is imbalanced: the ensemble is used and its prediction time is compared with greedy "
            "ensemble selection, then on five synthetic tables (--tables N), shown in groups of five, whose median ratio is "
            "reported as in the paper (--quick keeps the main example only). --example breast-cancer shows balanced data, where the best tuned learner is kept")
    s = sub.add_parser("demo", help=text, description=text); _common(s, trials=5, members="light")
    s.add_argument("--example", choices=("imbalanced", "breast-cancer"), default="imbalanced",
                   help="imbalanced: 2,000 synthetic rows, 10%% minority; breast-cancer: 569 rows, balanced classes (default: imbalanced)")
    s.add_argument("--test-size", type=float, default=0.2, help="fraction of rows held out, stratified by class (default: 0.2)")
    s.add_argument("--tables", type=int, default=5,
                   help="synthetic tables in the prediction-time check, the main example included; the median over them is "
                        "reported, in groups of five (default: 5; for example --tables 100 for a wider check)")
    s.add_argument("--quick", action="store_true",
                   help="time only the main example, the same as --tables 1 (faster demo)")
    s.add_argument("--no-speed", dest="speed", action="store_false",
                   help="skip the prediction-time comparison that the demo runs by default")
    s.add_argument("--repeats", type=int, default=5, help="timed calls per predictor; the median is reported (default: 5)")
    s.set_defaults(speed=True)
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
            X, y = _demo_data(*DEMO_SETS[0])
            what = "synthetic imbalanced data (2,000 rows, 20 columns, 10% minority class)"
        if not a.quiet:
            from rich.console import Console
            Console(stderr=True, highlight=False).print(f"[bold cyan]fde[/] demo: {what}; [dim]{quick}[/]")
        _evaluate(X, y, a)
    else:
        d = _read(a.data); _evaluate(d.drop(columns=[a.target]), d[a.target], a)


if __name__ == "__main__":
    main()
