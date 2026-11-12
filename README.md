# FDE: Fast-Deployment Ensemble for imbalanced tabular classification
FDE is a machine learning framework designed to improve tabular classification through diverse ensemble learning.
FDE combines multiple machine learning models, addresses class imbalance, and adaptively selects between ensemble predictions and the best-performing individual model.

This repository provides the implementation, documentation, and experimental resources for FDE.
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23997868.svg)](https://doi.org/10.5281/zenodo.23997868)

FDE builds a diverse ensemble around the best tuned single learner and keeps only as many members as its score requires.
Imbalanced data receive the ensemble; balanced data receive it only when a cross-fitted acceptance rule supports a gain.
Otherwise the best tuned learner is used on its own. *Fast deployment* refers to prediction time: members are removed
in order of their prediction cost, so the deployed ensemble calls few models. `fde evaluate --speed` measures this on
your own data (see [Checking the Fast claim](#checking-the-fast-claim)).

- **Best tuned learner.** Optuna tunes twelve learner types on inner cross-validation (CatBoost, LightGBM, XGBoost,
  HistGradientBoosting, random forest, extra trees, SVC, k-nearest neighbours, logistic regression, naive Bayes, MLP
  and decision tree) and the highest-scoring type is kept. Above 3,000 rows, SVC is left out.
- **Members that handle class imbalance.** Three groups join the ensemble: the five next-best learner types refitted
  with class weights; BalancedRandomForest, EasyEnsemble and RUSBoost; and the same five learner types refitted on
  rebalanced samples, with their probabilities corrected to equal class priors.
- **Extended members.** Eight members work with a larger budget: CatBoost, LightGBM and XGBoost (ordinal or native
  categorical coding, up to 5,000 trees with early stopping), an MLP with three hidden layers of 256 units, and
  random forest and extra trees with 500 trees. Each is bagged over 3 folds of its training rows.
- **Combiner.** A stack (multinomial logistic regression) over the probabilities of the best learner and the
  members, with a class-wise score adjustment of the resulting probabilities on top. The stack is kept only when its
  cross-validated Brier score is lower than that of the best learner; when it is rejected and only the adjustment is
  kept, the members are not called at prediction.
- **Class-ratio route.** With a majority-to-minority ratio of 3 or more in the training rows, the ensemble is used
  when the combiner changes the probabilities of the best learner. Below 3, the cross-fitted acceptance rule decides.
- **Cross-fitted acceptance rule.** Three outer cross-fitted folds of the training rows decide on balanced data. The
  ensemble is kept only if the lower 5% bootstrap quantile of its gain over the best tuned learner lies above 0, if its
  weight in the all-rows fit is non-zero, and if it changes at least one outer-fold prediction. Otherwise the best tuned
  learner is kept, which is the usual outcome on balanced data.
- **Ensemble sizing.** From all members, the costliest member is dropped as long as the inner out-of-fold score stays at
  that of all members, with the combiner refitted at each step and a fixed table of member costs. The best learner stays.

A run is scored with macro-F1 when there are two classes and the minority share is below 0.15, and with weighted-F1
in every other case.

## Installation

Python 3.10 or later (check with `python3 --version`). Installing into a fresh virtual environment keeps FDE apart from other projects. FDE runs on the
CPU by default, so the CPU build of PyTorch is enough and much smaller than the GPU build. Choose your system:

<details open>
<summary><b>Linux</b></summary>

```bash
python3 -m venv ~/fde-env
source ~/fde-env/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fde-tabular
fde demo
```

If `python3 -m venv` reports that `ensurepip` is missing (Debian, Ubuntu), install it first with
`sudo apt install python3-venv`.
</details>

<details>
<summary><b>macOS</b></summary>

```bash
brew install libomp python@3.12
python3.12 -m venv ~/fde-env
source ~/fde-env/bin/activate
pip install --upgrade pip
pip install torch
pip install fde-tabular
fde demo
```

The `python3` that ships with macOS is Python 3.9, which is too old for FDE (pip then reports "No matching distribution
found for fde-tabular"); the commands above use Python 3.12 from Homebrew. LightGBM and XGBoost need the OpenMP library
(`libomp`). On macOS the default PyTorch build already runs on the CPU.
</details>

<details>
<summary><b>Windows (Command Prompt)</b></summary>

Type or paste **one line at a time** and press Enter after each line:

```bat
py -m venv %USERPROFILE%\fde-env
%USERPROFILE%\fde-env\Scripts\activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fde-tabular
fde demo
```

If the first line gives an error, for example "'py' is not recognized" or "Unable to create process using ...", use
`python` instead of `py`:

```bat
python -m venv %USERPROFILE%\fde-env
```

In PowerShell, activate with `$env:USERPROFILE\fde-env\Scripts\Activate.ps1` instead (if scripts are blocked, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once). The Command Prompt does not treat `#` as a comment, so do
not paste comments after a command. Commands written for Linux, such as `source` or `~/`, do not work here.
</details>

<details>
<summary><b>With a CUDA GPU (any system, optional)</b></summary>

Install the PyTorch build for your CUDA version from <https://pytorch.org> instead of the CPU build, then
`pip install fde-tabular`, and pass `--device gpu` (or `device="gpu"`). See the GPU note under Options before using it.
</details>

Each time you open a new terminal, activate the environment again (the second line above) before using `fde`.

Everything runs on the CPU unless `--device gpu` is given: each tuned learner gets one thread, each extended member
gets four (`member_threads`). Before fitting, FDE checks the free CPU cores and memory and runs that many tasks in
parallel (`n_jobs`); the progress bar shows the three stages (tuning the learners, building the members, fitting the
combiners) with an estimate of the remaining time.

## Quick start

```bash
fde demo                            # real imbalanced data: the ensemble is used, and its prediction time is measured
fde demo --example breast-cancer    # balanced data: the best tuned learner alone is enough
fde demo --cv 5                     # outer 5-fold cross-validation of the first example
```

The first demo evaluates FDE on real clinical data shipped with the package: 788 patients with severe fever with
thrombocytopenia syndrome, of whom 12% died in hospital (PLOS figshare, CC BY 4.0; one of the 53 evaluation datasets of
the accompanying paper). The second uses the breast cancer data that ships with scikit-learn. No file or download is
needed for either. The first demo scores FDE against the best tuned learner on held-out rows and times the FDE predictor
against greedy ensemble selection over the same members (see [Checking the Fast claim](#checking-the-fast-claim)); it
then repeats both comparisons on 5 stratified held-out splits of the same data and reports the mean score, the median
time ratio, and on how many splits FDE is better and faster, since a single split can go either way (`--splits N` sets
the number of splits, `--quick` keeps the first split only, and `--no-speed` skips the timing). With `--cv 5`, the demo
runs an outer 5-fold cross-validation instead and shows every fold with the mean and standard deviation. Both demos end
with a short panel, *What this demo shows*: the score of FDE against the best tuned learner, the prediction-time ratio,
where to see the other case, and how to obtain cross-validated scores. The summary explains why the ensemble was used
or not. The demos use 5 Optuna trials per learner type and the full member set, so a split takes well under a minute
on a laptop; `fde demo --trials 20` runs the settings evaluated in the paper. A progress bar shows the stage, the
elapsed time and an estimate of the remaining time.

## Use

```python
from fde import FDEClassifier

clf = FDEClassifier(random_state=0).fit(X_train, y_train)   # X_train: pandas DataFrame, y_train: labels
print(clf.report_)               # best learner, acceptance rule, whether the ensemble is used
pred = clf.predict(X_test)
proba = clf.predict_proba(X_test)
clf.save("model.fde"); clf = FDEClassifier.load("model.fde")
```

From the command line:

```bash
fde fit --data train.csv --target label --out model.fde
fde predict --model model.fde --data new.csv --proba --out scores.csv
fde evaluate --data data.csv --target label --test-size 0.2   # holdout scores for FDE and the best tuned learner
fde evaluate --data data.csv --target label --cv 5            # outer 5-fold cross-validation: every fold, mean and SD
fde fit --data train.csv --target label --out light.fde --members light   # fewer members: faster prediction
```

The command line prints a short summary: the best tuned learner, whether the ensemble was needed, the model size, the
fit time and, for `demo` and `evaluate`, the held-out scores. Add `--json` for the full report as JSON, and `--quiet`
to turn off the progress messages (they go to the standard error stream).

Text columns can be left as they are: every learner imputes and encodes them using only its own training rows, and
treats missing values the same way.

## Practical guidance

A performance-cost trade-off is the gain given up in exchange for faster prediction; a user makes this choice through
the member set. This section is the full version of Appendix G of the accompanying paper.

- **Input.** A labeled table in CSV or Parquet format with one target column is required, and no diagnosis of class
  imbalance is needed, since the route is computed from the training rows.
- **Example of use.** With `pip install fde-tabular` in place, a predictor is fitted by `fde fit` from a training table
  and the name of its target column, and new rows are scored by `fde predict`. The methods `fit`, `predict`, and
  `predict_proba` are provided by `FDEClassifier` in Python.
- **Settings.** By default, `fde fit` uses the settings of the paper, with 20 trials and the full member set, and every
  split, search, and member is fixed by `--seed`. The option `--device gpu` runs the fit on a GPU, which can cause small
  changes in results.
- **What is reported.** Each fit reports the route, the number of models built, kept, and called at prediction, the
  best tuned learner, and, on balanced data, the decision of the acceptance rule. In addition, `fde evaluate` scores FDE
  against the best tuned learner on a held-out stratified part of the data. The stored score of an audited evaluation
  run was reproduced by the package on the reference hardware (paper, Appendix I).
- **Checking prediction time.** The command `fde demo`, without options, fits FDE on the real imbalanced example and
  times, on held-out rows and one CPU core, the FDE predictor against greedy ensemble selection over the same members,
  the full member set, and the best tuned learner, with the protocol of Section IV-F of the paper, and then reports the
  median ratio over 5 held-out splits; `fde evaluate --speed`
  does the same on a user table. Times depend on the machine, so the ratio of greedy to FDE time is the quantity to
  compare.
- **Where care is needed.** Fit time can be long on data with many features; on balanced data the best tuned learner
  is retained in most runs, so a large gain should not be expected there; and a different member subset can be
  selected by a fit on another processor (paper, Appendix I).
- **Member set and cost.** A median of 3 models of the 22 built is already kept by cost-aware pruning. Large-budget
  GBDTs, the MLP, and large forests are also skipped by the option `--members light`, which shortens fitting; the
  scores of the paper refer to the full member set.

## Examples

Each example runs as it is (`pip install fde-tabular` first). Times are for a 32-core laptop; with fewer cores they
grow roughly in proportion. Predicted labels are returned as text (for example `"0"` and `"1"`).

**Two classes, quick settings (about 25 s).** Breast cancer data, shipped with scikit-learn:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from fde import FDEClassifier

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
clf = FDEClassifier(trials=5, members="light", verbose=1).fit(X_train, y_train)
print("accuracy:", (clf.predict(X_test) == y_test.astype(str)).mean())
print("ensemble used:", clf.uses_ensemble_, "| best learner:", clf.report_["best_learner"])
```

**Several classes, with probabilities (about 10 s).** Wine data, shipped with scikit-learn:

```python
from sklearn.datasets import load_wine
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from fde import FDEClassifier

X, y = load_wine(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
clf = FDEClassifier(trials=5, members="light", verbose=1).fit(X_train, y_train)
print("macro-F1:", round(f1_score(y_test.astype(str), clf.predict(X_test), average="macro"), 4))
print(clf.classes_)
print(clf.predict_proba(X_test)[:3].round(3))      # one column per class, in the order of clf.classes_
```

**Imbalanced classes, a text column and missing values; save and reload (about 15 s).** Synthetic data:

```python
import numpy as np, pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from fde import FDEClassifier

X, y = make_classification(n_samples=3000, n_features=12, n_informative=5, weights=[0.95, 0.05], random_state=1)
X = pd.DataFrame(X, columns=[f"x{i}" for i in range(12)])
X["region"] = np.where(X["x0"] > 0, "north", "south")     # text columns can stay as text
X.loc[::25, "x1"] = np.nan                               # missing values are imputed inside each learner
y = np.where(y == 1, "fraud", "normal")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
clf = FDEClassifier(trials=5, members="light", verbose=1).fit(X_train, y_train)
print("balanced accuracy:", round(balanced_accuracy_score(y_test, clf.predict(X_test)), 4))
clf.save("fraud.fde")
clf = FDEClassifier.load("fraud.fde")
print(clf.predict(X_test.head()))
```

**Real imbalanced data with the settings of the paper (about 50 s).** The pc4 software-defect data from OpenML
(downloaded once, about 0.1 MB; 1,458 rows, 12% defects). Here the ensemble is used:

```python
from sklearn.datasets import fetch_openml
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from fde import FDEClassifier

X, y = fetch_openml(data_id=1049, return_X_y=True, as_frame=True, parser="liac-arff")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
clf = FDEClassifier(verbose=1).fit(X_train, y_train)
print("macro-F1:", round(f1_score(y_test.astype(str), clf.predict(X_test), average="macro"), 4))
print("ensemble used:", clf.uses_ensemble_, "| models in predictor:", clf.report_["models_in_predictor"])
print("route:", clf.report_["route"])           # imbalanced data: the ensemble route, without the acceptance rule
```

**Command line with your own CSV file.** This writes a CSV from the wine data, then fits, predicts and evaluates:

```bash
python -c "from sklearn.datasets import load_wine; d = load_wine(as_frame=True).frame; d.to_csv('wine.csv', index=False)"
fde fit --data wine.csv --target target --out wine.fde --trials 5 --members light
fde predict --model wine.fde --data wine.csv --proba --out wine_scores.csv
fde evaluate --data wine.csv --target target --trials 5 --members light
fde evaluate --data wine.csv --target target --trials 5 --members light --cv 5
```

On Windows, run each line separately.

## How the data are split

No row is used both to choose a model and to score it:

1. `fde evaluate` and `fde demo` set aside a stratified share of the rows (20% by default) before fitting; these rows
   are used only for the held-out scores. With `fde evaluate --cv 5` (or `fde.cross_validate(X, y, cv=5)` in Python),
   the rows are instead split into 5 stratified folds, each fold is held out once while FDE is fitted on the others,
   and the scores of every fold are reported with their mean and standard deviation, so that every row is scored once
   by a model fitted without it.
2. On balanced training rows, 3 outer folds decide whether the ensemble is used. The decision is made on rows that
   each fold's models have not seen (the cross-fitted acceptance rule). Imbalanced data skip these folds.
3. In every fit, the learners are tuned and the combiner is fitted on 3 inner folds, and every imputer, encoder and
   scaler is fitted on its own training rows only.

To compare FDE with another tool, score both on the same held-out rows or folds. A score computed on rows that were
also used for tuning comes out higher than the score on new data; for this reason the out-of-fold scores inside a fit,
which also choose the members, are not reported as results.

## Checking the Fast claim

FDE removes members in order of a fixed table of prediction costs, from the costliest, as long as the out-of-fold
score of the full member set is kept. The option `--speed` shows what this does to prediction time on your data:

```
fde evaluate --data data.csv --target y --speed
fde demo                                       # no data needed; the demo measures the times by default
```

After the usual held-out scores, FDE times four predictors built from the same fitted models on the same held-out rows:

| Predictor | What it calls |
|---|---|
| FDE | the best tuned learner, the members kept by cost-aware pruning, and the combiner |
| greedy selection | greedy ensemble selection (Caruana et al., 2004) over the same members: selection with replacement by Brier score on the same out-of-fold probabilities, started from the best tuned learner |
| full member set | every member with the combiner, without pruning |
| best tuned learner | one model |

Each predictor is one call over all held-out rows on one CPU core: the process is pinned to one core on Linux, and the
numerical libraries are limited to one thread on every system. After one warm-up call, the median of five calls
(`--repeats`) is divided by the number of rows and reported in ms per 1,000 rows, with the score of each predictor on
the same rows. Data preparation is included and fitting is not. A ratio of greedy to FDE time above 1 means that FDE
predicts faster.

Times depend on the machine, so the ratio on one core is the quantity to compare. In the accompanying paper, on the 43
evaluation datasets on the ensemble route, the median ratio of greedy selection to FDE time was 3.34, and FDE was
faster on 39. Greedy selection is faster when it keeps fewer or cheaper models; FDE then keeps a member that its score
needs. When the route keeps the best tuned learner, every predictor reduces to that learner. In Python:

```python
from fde import FDEClassifier, speed_report
clf = FDEClassifier(keep_candidates=True).fit(X_train, y_train)
print(speed_report(clf, X_test, y_test))
```

`keep_candidates=True` keeps every fitted member for this comparison, so the saved model is larger; it does not change
the predictions of FDE.

## Options

| Option | Default | Meaning |
|---|---|---|
| `random_state` (`--seed`) | 0 | seed shared by every split, search and member |
| `n_jobs` (`--jobs`) | -1 | parallel tasks; -1 chooses from the free CPU cores and memory (results do not depend on it) |
| `members` (`--members`) | `full` | `light` drops the extended members: 14 models instead of 22, several times faster at prediction, with somewhat lower accuracy |
| `trials` (`--trials`) | 20 | Optuna trials for each learner type |
| `member_threads` | 4 | threads per extended member (Python API only) |
| `device` (`--device`) | `cpu` | `gpu` trains XGBoost, CatBoost and the MLP on the first CUDA GPU (LightGBM stays on the CPU: its GPU mode was slower in our tests). Results can differ slightly from the CPU; see the note below |
| `verbose` | 0 (1 on the command line) | 1 shows the system check and a progress bar with the stage and the remaining time; 2 also prints the report |
| `keep_candidates` (`--speed`) | `False` | keeps every fitted member so that `speed_report` can compare prediction times; predictions are unchanged |
| `--repeats` | 5 | timed calls per predictor with `--speed`; the median is reported |

These defaults are the configuration evaluated in the accompanying paper.

**GPU note.** The results of the accompanying paper were obtained on the CPU, and the CPU is the default. On our test
machine (32 CPU cores, one RTX 4070) fitting was faster on the CPU than with `--device gpu`: 26 s against 139 s for
`fde demo`, and 365 s against 704 s for 40,000 training rows, because many CPU tasks run at the same time while GPU
tasks wait for the one GPU.
The GPU option is offered for machines with few CPU cores; check the time on your own data. Models fitted with
`--device gpu` predict on the CPU, so they also run on machines without a GPU.

## Cost

Fitting costs more than tuning one learner, because up to twelve learner types are tuned once on imbalanced data and four times
on balanced data (once per outer fold and once on all rows). Prediction is cheap when the best tuned learner is kept: one model. When the ensemble is
used, the best learner, the members kept by sizing (a median of 2 of 21 in the accompanying paper; bagged members hold
three models each) and the combiner run. `--speed` measures this on your data.

## Benchmark artifacts

The folder [`benchmark/`](benchmark/) holds the evaluation of the accompanying paper: the 53 datasets and their loader,
a hash of the split of every run, the scores of FDE and of the eight comparators of the main table in all 265 runs, the
prediction-time records, the scripts and environments of the AutoML comparators, and a script that recomputes the main
comparison table. See [`benchmark/README.md`](benchmark/README.md).

## Citation

Please cite the accompanying article if you use FDE; the full reference will be added once it is published. The
software itself is archived on Zenodo: version 1.1.2, the configuration evaluated in the accompanying paper together
with its benchmark artifacts, has DOI [10.5281/zenodo.24075335](https://doi.org/10.5281/zenodo.24075335) (version 1.1.1:
[10.5281/zenodo.24066927](https://doi.org/10.5281/zenodo.24066927); version 1.1.0:
[10.5281/zenodo.24054150](https://doi.org/10.5281/zenodo.24054150); version 1.0.0:
[10.5281/zenodo.23997869](https://doi.org/10.5281/zenodo.23997869)), and
[10.5281/zenodo.23997868](https://doi.org/10.5281/zenodo.23997868) always points to the latest version. See `CITATION.cff`.

## License

MIT.
