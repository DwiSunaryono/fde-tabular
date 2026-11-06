# FDE: Feasible Diverse Ensembles for tabular classification

FDE builds a diverse ensemble around the best tuned single learner and uses it only when it is shown, on the
training data, to beat that learner. Otherwise the best tuned learner is kept.

- **Best tuned learner.** Twelve learner types (CatBoost, LightGBM, XGBoost, HistGradientBoosting, random forest,
  extra trees, SVC, k-nearest neighbours, logistic regression, naive Bayes, MLP, decision tree) are tuned with
  Optuna on inner cross-validation, and the best one is kept.
- **Members that handle class imbalance.** Class-weighted versions of the five next-best learner types,
  BalancedRandomForest, EasyEnsemble and RUSBoost, and rebalanced versions of the five learner types with
  probabilities corrected to the class priors.
- **Extended members.** Large-budget GBDTs (CatBoost, LightGBM and XGBoost, with ordinal or native categorical
  coding; up to 5,000 trees with early stopping), an MLP (3 x 256 units) and large forests (random forest and extra
  trees, 500 trees). Each is bagged over 3 folds of its training rows.
- **Combiner.** Stacking (multinomial logistic regression) on the probabilities of the best learner and the members,
  followed by per-class probability scales.
- **Cross-fitted acceptance rule.** On 3 outer cross-fitted folds of the training rows, the ensemble is used only
  when the lower 5% bootstrap bound of its gain over the best tuned learner is above 0, its weight is not zero and
  it changes at least one prediction. Otherwise the best tuned learner is kept. The rule is the same for every
  imbalance level; on balanced data the best tuned learner is usually kept.

The run metric is macro-F1 for two-class data with a minority share below 0.15, and weighted-F1 otherwise.

## Installation

```bash
pip install fde-tabular
```

Python 3.10 or later. All learners run on the CPU. Tuned learners use one thread each and the extended members four (`member_threads`); the fit sets are built in parallel (`n_jobs`).

## Use

```python
from fde import FDEClassifier

clf = FDEClassifier(random_state=0, n_jobs=4).fit(X_train, y_train)   # pandas DataFrame, labels
print(clf.report_)               # best learner, acceptance rule, whether the ensemble is used
pred = clf.predict(X_test)
proba = clf.predict_proba(X_test)
clf.save("model.fde"); clf = FDEClassifier.load("model.fde")
```

Command line:

```bash
fde fit --data train.csv --target label --out model.fde
fde predict --model model.fde --data new.csv --proba --out scores.csv
fde evaluate --data data.csv --target label --test-size 0.2   # holdout scores of FDE and the best tuned learner
```

Categorical columns may stay as text: each learner imputes and encodes them on its own training rows. Missing
values are imputed in the same way.

## Options

| Option | Default | Meaning |
|---|---|---|
| `members` (`--members`) | `full` | `light` keeps only the members that handle class imbalance: the ensemble has 14 models instead of 22 and predicts several times faster, with somewhat lower accuracy |
| `trials` (`--trials`) | 20 | Optuna trials per learner type |
| `member_threads` | 4 | threads of each extended member |

The defaults are the configuration evaluated in the accompanying paper.

## Cost

Fitting tunes twelve learner types four times (three outer folds and all rows), so it takes longer than tuning a
single learner. At prediction time, FDE evaluates one model when the best tuned learner is kept. When the ensemble is
used, it evaluates the best learner, 21 members (eight of them as 3 bagged fold models) and the combiner, so
prediction is slower than with one model.

## Citation

If you use FDE, please cite the accompanying article (reference added on publication). See `CITATION.cff`.

## License

MIT.
