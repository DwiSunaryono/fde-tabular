# Changelog

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
