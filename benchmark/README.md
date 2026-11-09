# Benchmark artifacts of the accompanying paper

This folder holds what is needed to check the evaluation of *FDE: Fast-Deployment Ensemble Framework for Imbalanced
Tabular Classification*: the 53 evaluation datasets, a hash of the split of every run, the score of FDE and of the
eight comparators of Table 4 in all 265 runs (53 datasets x 5 seeds), the prediction-time records of Table 16, the
environment and invocation of each AutoML comparator, and a script that recomputes Table 4 from the stored scores.

| File | Content |
|---|---|
| `datasets.csv` | the 53 evaluation datasets: source, identifier, rows used, features, classes, imbalance ratio and group, run metric |
| `datasets.json` | the specification read by the loader: source and identifier, target column, removed columns, row limit |
| `datasets.py` | loader of each dataset as used in the paper (`load(name)`), and the generator of the outer TRAIN/TEST split of each run (`holdout(y, seed)`) |
| `data_open/` | the 16 tables built from public-agency and open-licence sources, as used in the paper; sources and licences in `SOURCES.md` |
| `results/runs.csv` | one row per run: TEST score of FDE and of the eight comparators of Table 4 on the run metric, route and ensemble use, models kept, and three fixed metrics of FDE and of the best tuned learner |
| `results/splits.csv` | one row per run: SHA-256 of the sorted TRAIN row indices, of the sorted TEST row indices, and of the TEST labels in row order, taken from the stored runs |
| `results/timing_pipeline_evaluation.csv` | Table 16b: prediction time of the full FDE pipeline and of greedy ensemble selection over the same members, 53 evaluation datasets, seed 0, one core |
| `results/timing_pipeline_development.csv` | Table 16a: the same comparison on 12 development datasets outside the cost table |
| `make_tables.py` | recomputes Table 4 (mean gain, 95% bootstrap interval, wins/ties/losses, Holm-adjusted Wilcoxon p) from `results/runs.csv` |
| `check_splits.py` | regenerates the split of each run with `holdout` and compares it with `results/splits.csv` |
| `comparators/` | the scripts that ran AutoGluon (default and high-quality presets), FLAML, and Auto-sklearn, the exported environments of these runs, and `make_run_records.py`, which prepares their input from this folder |

## Scope

The stored scores reproduce Table 4 and the timing records reproduce Table 16 and Fig. 3c. The scores of the twelve
tuned learners (Tables 13 and 14) and of the ablation variants (Table 9) are not part of this folder; they are produced
by the package and the protocol of the paper. Retraining all 265 runs repeats the whole benchmark; small numerical
differences across processors can change the members that pruning keeps, and with them the score of a single run
(see the paper, Appendix I).

## Datasets and splits

`load(name)` gives each dataset as used in the study; its output was checked against the study's own loader on all 53
datasets. OpenML datasets are fetched by data id (`sklearn.datasets.fetch_openml`); UCI datasets by dataset id (with
`ucimlrepo`, or from a folder of `original.csv` files); the 16 open-data tables are in `data_open/`. The seven Kaggle
datasets are not redistributed here: they are downloaded with the Kaggle API under the licence stated on each dataset
page, and their names and files are listed in `datasets.json`.

Before any split, `load` fixes the column types once from the feature columns, without labels: a text column whose
non-empty values are at least 95% numeric is read as numbers, and a text column with unique values in more than half
of the rows (an identifier) is removed. Imputation, scaling, and encoding are fitted later, on the fit rows of each
model, as described in the paper (Section III-F).

`holdout(y, seed)` regenerates the outer TRAIN/TEST split (stratified 80/20, random state equal to the seed, seeds 0
to 4). `results/splits.csv` holds hashes of the split stored with each run, and `python check_splits.py` compares the
two for every dataset that can be loaded. On the 46 datasets that need no Kaggle download, all 230 runs are identical.

```python
from datasets import load, holdout
X, y = load("msha-accidents")
train, test = holdout(y, seed=0)          # seeds 0 to 4
```

A run uses macro-F1 when there are two classes and the minority share in TRAIN is below 0.15, and weighted-F1 otherwise.

## Columns of `results/runs.csv`

| Column | Meaning |
|---|---|
| `dataset`, `seed` | the run |
| `imbalance_group` | low, moderate, extreme, or balanced, from the imbalance ratio |
| `run_metric` | `macro_f1` or `weighted_f1`, fixed from the TRAIN labels |
| `n_train` | rows in TRAIN |
| `route_ensemble` | 1 if the run is assigned to the ensemble route, 0 if FDE keeps the best tuned learner |
| `stack_kept` | 1 if the stack is kept, 0 if only the class-wise adjustment of the best tuned learner is kept; empty when `route_ensemble` is 0 (not applicable) |
| `models_kept_incl_best` | models kept by pruning, the best tuned learner included |
| `fde` and the eight comparator columns | TEST score on the run metric: `autogluon_default`, `autosklearn`, `flaml`, `best_tuned_learner`, `best_tuned_learner_adjusted`, `best_tuned_learner_or_tabpfn`, `tabpfn`, `autogluon_high_quality` |
| `automl_time_limit_s` | time limit given to Auto-sklearn, FLAML, and AutoGluon with its high-quality preset |
| `autogluon_default_time_limit_s` | time limit given to AutoGluon with its default preset (differs on 19 runs; see Methods and environments) |
| `autogluon_default_fit_s` | fit time of AutoGluon with its default preset, in seconds |
| `fde_weighted_f1`, `fde_macro_f1`, `fde_balanced_accuracy`, `best_*` | the three fixed metrics of FDE and of the best tuned learner |

## Methods and environments

- **FDE:** `fde-tabular` 1.1.0 with its default settings (20 Optuna trials per learner type, full member set, seed
  equal to the run seed), fitted on TRAIN and scored once on TEST. Software of the FDE runs, identical in all 265 run
  records: Python 3.12, scikit-learn 1.9.1, Optuna 5.0.0, LightGBM 4.7.0, XGBoost 3.4.1, CatBoost 1.2.10,
  imbalanced-learn 0.14.2, NumPy 2.5.3, pandas 3.0.6, and PyTorch 2.14.1.
- **Time limit of the AutoML systems:** the sum of the single-thread tuning times of the twelve learner types in the
  selection of the best tuned learner on that run, clipped to 60-3,600 s (`automl_time_limit_s`, median 830 s); four
  threads. AutoGluon with its default preset was run on several workstations, each reading the run record present
  there; on 19 runs that record was an earlier build of the same run (same data, seed, and split) whose tuning times
  differ, so its own limit is given in `autogluon_default_time_limit_s` (median 827 s). Most fits of this preset finish
  well below their caps: the median fit time is 42 s, and 1 of its 265 runs reaches 90% of its cap
  (`autogluon_default_fit_s`).
- **AutoGluon-Tabular 1.6.3:** default preset (`medium_quality`, `comparators/run_autogluon_flaml.py`) and high-quality
  preset (`best_quality`, `comparators/run_autogluon_high_quality.py`), with `eval_metric` set to the run metric and
  `problem_type` binary or multiclass. Environment: `comparators/environment_autogluon_flaml.txt`.
- **FLAML 2.7.0:** classification with the run metric as a custom metric, the same time limit, and four jobs
  (`comparators/run_autogluon_flaml.py`). Same environment as AutoGluon.
- **Auto-sklearn 0.15.0 (Auto-sklearn 1.0 interface, `AutoSklearnClassifier`):** the run metric, the same time limit,
  four workers, 6,144 MB per worker, and categorical columns declared (`comparators/run_autosklearn.py`). The package
  requires scikit-learn below 0.25, so it ran in a separate environment, Python 3.9.25 with scikit-learn 0.24.2
  (`comparators/environment_autosklearn.txt`), packed once and unpacked unchanged on every workstation; the data were
  exported for it by `comparators/autosklearn_export_data.py` with the same rows and split. Auto-sklearn 2.0 supports
  only accuracy, balanced accuracy, ROC AUC, and log loss, not the F1 metrics of the study.
- **TabPFN:** package 9.2.0 with checkpoint `tabpfn-v3-classifier-v3_default.ckpt` (SHA-256
  `d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988`), a stratified context of at most 10,000 rows
  of the fit set, ordinal-coded categories, and a GPU for the scores.

The scripts in `comparators/` are the code that ran, with one comment corrected. They read a run record (the TRAIN/TEST
row indices and labels, the run metric, called the gate metric there, and the Phase 1 tuning times that give the time
limit) and the study's loader. `comparators/make_run_records.py` builds these records from the public files of this
folder (split from `holdout`, checked against `results/splits.csv`; run metric and time limit from `results/runs.csv`),
and `comparators/fde2/data.py` connects the scripts to `datasets.py`. From the folder `comparators/`:

```
python make_run_records.py conf2_final                      # records of every dataset that can be loaded
python run_autogluon_flaml.py conf2_final/anneal_seed0.npz flaml bp1
python run_autogluon_high_quality.py conf2_final/anneal_seed0.npz autogluon_best bp1
python make_run_records.py conf2_final_ag --autogluon-default
python run_autogluon_flaml.py conf2_final_ag/anneal_seed0.npz autogluon bp1
python autosklearn_export_data.py conf2_final/names.json askl_data
<auto-sklearn environment>/bin/python run_autosklearn.py conf2_final/anneal_seed0.npz askl_data askl_out
```

The record folder is named `conf2_final` because `autosklearn_export_data.py` reads it under that name. Each script
writes its TEST score, fit and prediction seconds, and the selected model as JSON next to the record (Auto-sklearn: in
`askl_out/`). AutoML search under a wall-clock limit depends on the machine and its load, so a repeated run gives a
score close to, but not always equal to, the stored one.

## Paper item, file, and command

| Paper item | File | Command |
|---|---|---|
| Table 4 | `results/runs.csv` | `python make_tables.py` |
| Table 5 (fixed metrics against the best tuned learner) | `results/runs.csv`, columns `fde_*` and `best_*` | mean over seeds, then over datasets |
| Table 16a, 16b and Fig. 3c | `results/timing_pipeline_*.csv` | ratio `greedy_ms_per_1000_rows / fde_ms_per_1000_rows` per dataset |
| Table 15 | `datasets.csv`, `results/runs.csv` | mean over seeds per dataset |
| Splits of all runs | `results/splits.csv` | `python check_splits.py` |
