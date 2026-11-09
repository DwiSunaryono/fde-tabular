# Changelog

## 1.1.1
- Benchmark artifacts of the accompanying paper in `benchmark/`: the 53 evaluation datasets (loader, specifications,
  and the 16 open-data tables with their sources and licences), a SHA-256 hash of the TRAIN/TEST split of every run with
  `check_splits.py`, which regenerates and compares the splits, the scores of FDE and of the eight comparators of
  Table 4 in all 265 runs, the prediction-time records of Table 16, the scripts and environments of the AutoML
  comparators in `benchmark/comparators/`, with `make_run_records.py`, which prepares their input from the archive, and `make_tables.py`, which recomputes Table 4 from these scores.
- The package code is unchanged from 1.1.0; results are unchanged.

## 1.1.0
- Class-ratio route: with a majority-to-minority ratio of 3 or more in the training rows, the ensemble is used without
  outer folds, so one fit set is built instead of four. Balanced data keep the cross-fitted acceptance rule.
- Ensemble sizing: from all members, the costliest member is dropped as long as the out-of-fold score stays at that of
  all members, using a fixed table of member costs, which shortens prediction. The best learner always stays.
- Report: new keys `route` and `models_built`; `acceptance_rule` is `None` on imbalanced data.
- Prediction calls only the models that enter it: when the stack is rejected and only the class-wise adjustment of the
  best learner is kept, the members are no longer called. Predictions are unchanged.
- Report: new keys `models_called_at_prediction` and `stack_kept`.
- Checking the Fast claim: `fde evaluate --speed` and `fde demo` (Python: `keep_candidates=True` and
  `fde.speed_report`) time the FDE predictor against greedy ensemble selection over the same members, the full member
  set and the best tuned learner, on the held-out rows and one CPU core.
- `fde demo` now uses the imbalanced example by default, measures the prediction times, and ends with a short panel on
  what the demo has shown; `--example breast-cancer` shows balanced data, where the best tuned learner is kept, and
  `--no-speed` skips the timing.
- Name: FDE now stands for Fast-Deployment Ensemble, as in the accompanying paper.
- Results change from 1.0.x, since this is the configuration evaluated in the accompanying paper.

## 1.0.1
- macOS: LightGBM and XGBoost are loaded before PyTorch, which fixes a crash while the members are built; the GPU check
  no longer loads PyTorch unless `--device gpu` is given.
- README: installation on macOS with Python 3.12 from Homebrew (the system Python 3.9 is too old), and a pip upgrade
  step on every system.
- Requires joblib 1.4 or later.
- Results are unchanged.

## 1.0.0
- First release: FDEClassifier and the command line (fit, predict, evaluate) with the configuration evaluated in the
  accompanying paper.
