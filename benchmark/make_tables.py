"""Recomputes the main comparison of the accompanying paper (Table 4) from results/runs.csv.

Dataset as unit: for comparator c and dataset d, the gain is the mean over the five seeds of (FDE - c) on the run metric,
in percentage points. For each comparator: mean gain, 95% percentile bootstrap interval over datasets (9,999 resamples,
one generator with seed 20261103 drawn in the order of the comparators), wins/ties/losses at 0.1 pp, the mean gain on
imbalanced and balanced datasets, and the two-sided Wilcoxon signed-rank p-value with Holm's correction over the eight
comparators.
Usage: python make_tables.py [results/runs.csv]"""
import csv, os, sys

import numpy as np
from scipy.stats import wilcoxon

PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "runs.csv")
COMPARATORS = [("AutoGluon (default preset)", "autogluon_default"), ("Auto-sklearn", "autosklearn"), ("FLAML", "flaml"),
               ("Best tuned learner M*", "best_tuned_learner"), ("M* with class-wise score adjustment", "best_tuned_learner_adjusted"),
               ("Best tuned learner or TabPFN M+", "best_tuned_learner_or_tabpfn"), ("TabPFN", "tabpfn"),
               ("AutoGluon (high-quality preset)", "autogluon_high_quality")]

runs = list(csv.DictReader(open(PATH)))
datasets = sorted({r["dataset"] for r in runs})
imbalanced = np.array([next(r["imbalance_group"] for r in runs if r["dataset"] == d) != "balanced" for d in datasets])


def gain(col):
    return np.array([100 * np.mean([float(r["fde"]) - float(r[col]) for r in runs if r["dataset"] == d]) for d in datasets])


def holm(p):
    p = np.asarray(p); order = np.argsort(p); out = np.empty(len(p)); run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (len(p) - i) * p[j])); out[j] = run
    return out


rng = np.random.default_rng(20261103)
G = {col: gain(col) for _, col in COMPARATORS}
P = holm([wilcoxon(G[col][G[col] != 0]).pvalue for _, col in COMPARATORS])
print(f"{len(datasets)} datasets ({imbalanced.sum()} imbalanced, {(~imbalanced).sum()} balanced), {len(runs)} runs\n")
print(f"{'comparator':38s} {'gain':>6s} {'95% CI':>17s} {'W/T/L':>9s} {'imbal.':>7s} {'bal.':>6s} {'Holm p':>9s}")
for (name, col), p in zip(COMPARATORS, P):
    e = G[col]; lo, hi = np.percentile(e[rng.integers(0, len(e), (9999, len(e)))].mean(1), [2.5, 97.5])
    wtl = f"{(e > .1).sum()}/{(abs(e) <= .1).sum()}/{(e < -.1).sum()}"
    print(f"{name:38s} {e.mean():+6.2f} [{lo:+6.2f}, {hi:+6.2f}] {wtl:>9s} {e[imbalanced].mean():+7.2f} {e[~imbalanced].mean():+6.2f} {p:9.2g}")
