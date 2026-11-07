# FDE: Feasible Diverse Ensembles for tabular classification
Feasible Diverse Ensembles (FDE) is a machine learning framework designed to improve tabular classification through diverse ensemble learning.
FDE combines multiple machine learning models, addresses class imbalance, and adaptively selects between ensemble predictions and the best-performing individual model.

This repository provides the implementation, documentation, and experimental resources for FDE.
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23997869.svg)](https://doi.org/10.5281/zenodo.23997869)

FDE wraps the best tuned single learner in a diverse ensemble and hands prediction to the ensemble only when the
training data support it over that learner. When they do not, the best tuned learner is used on its own.

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
  members, with a per-class rescaling of the resulting probabilities on top.
- **Cross-fitted acceptance rule.** Three outer cross-fitted folds of the training rows decide. The ensemble is kept
  only if the lower 5% bootstrap quantile of its gain over the best tuned learner lies above 0, if its weight in the
  all-rows fit is non-zero, and if it changes at least one outer-fold prediction. Otherwise the best tuned learner is
  kept. Every imbalance level follows the same rule; on balanced data the best tuned learner is usually the one kept.

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

Type or paste **one line at a time** and press Enter after each line (if `py` is not found, use `python` instead):

```bat
py -m venv %USERPROFILE%\fde-env
%USERPROFILE%\fde-env\Scripts\activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fde-tabular
fde demo
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
fde demo                          # breast cancer data: the best tuned learner alone is enough
fde demo --example imbalanced     # synthetic imbalanced data: the ensemble is used
```

The first demo evaluates FDE on the breast cancer data that ships with scikit-learn, the second on synthetic data with a
10% minority class (no file or download needed for either). The summary explains why the ensemble was used or not. The demos use
quick settings (5 Optuna trials per learner type and the light member set), so it finishes in about a minute on a
laptop; `fde demo --trials 20 --members full` runs the settings evaluated in the paper. A progress bar shows the
stage, the elapsed time and an estimate of the remaining time, and the result compares FDE with the best tuned
learner on held-out rows.

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
fde fit --data train.csv --target label --out light.fde --members light   # fewer members: faster prediction
```

The command line prints a short summary: the best tuned learner, whether the ensemble was needed, the model size, the
fit time and, for `demo` and `evaluate`, the held-out scores. Add `--json` for the full report as JSON, and `--quiet`
to turn off the progress messages (they go to the standard error stream).

Text columns can be left as they are: every learner imputes and encodes them using only its own training rows, and
treats missing values the same way.

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

X, y = fetch_openml(data_id=1049, return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
clf = FDEClassifier(verbose=1).fit(X_train, y_train)
print("macro-F1:", round(f1_score(y_test.astype(str), clf.predict(X_test), average="macro"), 4))
print("ensemble used:", clf.uses_ensemble_, "| models in predictor:", clf.report_["models_in_predictor"])
print(clf.report_["acceptance_rule"])           # lower bound of the gain on the cross-fitted training folds
```

**Command line with your own CSV file.** This writes a CSV from the wine data, then fits, predicts and evaluates:

```bash
python -c "from sklearn.datasets import load_wine; d = load_wine(as_frame=True).frame; d.to_csv('wine.csv', index=False)"
fde fit --data wine.csv --target target --out wine.fde --trials 5 --members light
fde predict --model wine.fde --data wine.csv --proba --out wine_scores.csv
fde evaluate --data wine.csv --target target --trials 5 --members light
```

On Windows, run each line separately.

## How the data are split

No row is used both to choose a model and to score it:

1. `fde evaluate` and `fde demo` set aside a stratified share of the rows (20% by default) before fitting; these rows
   are used only for the held-out scores.
2. On the training rows, 3 outer folds decide whether the ensemble is used. The decision is made on rows that each
   fold's models have not seen (the cross-fitted acceptance rule).
3. Inside every fold, the learners are tuned and the combiner is fitted on 3 inner folds, and every imputer, encoder
   and scaler is fitted on its own training rows only.

To compare FDE with another tool, score both on the same held-out rows. A score computed on rows that were also used
for tuning comes out higher than the score on new data.

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

These defaults are the configuration evaluated in the accompanying paper.

**GPU note.** The results of the accompanying paper were obtained on the CPU, and the CPU is the default. On our test
machine (32 CPU cores, one RTX 4070) fitting was faster on the CPU than with `--device gpu`: 26 s against 139 s for
`fde demo`, and 365 s against 704 s for 40,000 training rows, because many CPU tasks run at the same time while GPU
tasks wait for the one GPU.
The GPU option is offered for machines with few CPU cores; check the time on your own data. Models fitted with
`--device gpu` predict on the CPU, so they also run on machines without a GPU.

## Cost

Fitting costs more than tuning one learner, because up to twelve learner types are tuned four times (once per outer
fold and once on all rows). Prediction is cheap when the best tuned learner is kept: one model. When the ensemble is
used, the best learner, 21 members (eight of them bagged over 3 folds, three models each) and the combiner all run,
so prediction takes longer than with a single model.

## Citation

Please cite the accompanying article if you use FDE; the full reference will be added once it is published. The
software itself is archived on Zenodo: version 1.0.0 has DOI
[10.5281/zenodo.23997869](https://doi.org/10.5281/zenodo.23997869), and
[10.5281/zenodo.23997868](https://doi.org/10.5281/zenodo.23997868) always points to the latest version. See `CITATION.cff`.

## License

MIT.
