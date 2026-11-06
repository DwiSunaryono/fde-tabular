# FDE: Feasible Diverse Ensembles for tabular classification

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

```bash
pip install fde-tabular
```

Python 3.10 or later. Everything runs on the CPU: each tuned learner gets one thread, each extended member gets four
(`member_threads`), and the four fit sets — three outer folds plus all rows — are built in parallel (`n_jobs`).

FDE does not use a GPU. To avoid downloading the GPU build of PyTorch, install the CPU build first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fde-tabular
```

## Use

```python
from fde import FDEClassifier

clf = FDEClassifier(random_state=0, n_jobs=4).fit(X_train, y_train)   # X_train: pandas DataFrame, y_train: labels
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

Text columns can be left as they are: every learner imputes and encodes them using only its own training rows, and
treats missing values the same way.

## Options

| Option | Default | Meaning |
|---|---|---|
| `random_state` (`--seed`) | 0 | seed shared by every split, search and member |
| `n_jobs` (`--jobs`) | 4 | how many fit sets are built at the same time (each tuned learner is single-threaded) |
| `members` (`--members`) | `full` | `light` drops the extended members: 14 models instead of 22, several times faster at prediction, with somewhat lower accuracy |
| `trials` (`--trials`) | 20 | Optuna trials for each learner type |
| `member_threads` | 4 | threads per extended member (Python API only) |

These defaults are the configuration evaluated in the accompanying paper.

## Cost

Fitting costs more than tuning one learner, because up to twelve learner types are tuned four times (once per outer
fold and once on all rows). Prediction is cheap when the best tuned learner is kept: one model. When the ensemble is
used, the best learner, 21 members (eight of them bagged over 3 folds, three models each) and the combiner all run,
so prediction takes longer than with a single model.

## Citation

Please cite the accompanying article if you use FDE; the full reference will be added once it is published. See
`CITATION.cff`.

## License

MIT.
