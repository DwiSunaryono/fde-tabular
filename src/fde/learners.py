"""Learners, preprocessing and Optuna search spaces.

Every estimator is wrapped with its own preprocessing, fitted on the rows it is trained on (no statistic is
computed before a split). Columns are selected by name through `mask`.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# the twelve learner types from which the best tuned learner is chosen
LEARNER_TYPES = ("hgb", "xgb", "lgbm", "cat", "rf", "et", "svc", "knn", "lr", "nb", "mlp", "dt")
SCALE_SENSITIVE = {"svc", "knn", "lr", "nb", "mlp"}      # scaled numeric columns, one-hot categoricals
warnings.filterwarnings("ignore", message=".*probability.*deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def default_params(mt):
    return {"hgb": {"lr": 0.1, "leaf": 31, "l2": 0.0, "it": 200},
            "xgb": {"lr": 0.1, "md": 6, "ne": 200, "ss": 1.0, "cs": 1.0},
            "lgbm": {"lr": 0.1, "nl": 31, "ne": 200, "ss": 1.0, "cs": 1.0},
            "svc": {"c": 1.0, "g": "scale"},
            "knn": {"n": 5, "w": "uniform"},
            "dt": {"d": 8, "msl": 1},
            "cat": {"lr": 0.1, "dep": 6, "it": 300, "l2": 3.0},
            "rf": {"ne": 300, "mf": "sqrt", "msl": 1},
            "et": {"ne": 300, "mf": "sqrt", "msl": 1},
            "lr": {"c": 1.0},
            "nb": {"vs": 1e-9},
            "mlp": {"h": 64, "layers": 1, "a": 1e-4, "lri": 1e-3},
            "brf": {"ne": 300}, "eec": {"ne": 10}, "rusb": {"ne": 100}}[mt]


def suggest(trial, mt, pre=""):
    s = lambda n: pre + n
    if mt == "hgb":
        return {"lr": trial.suggest_float(s("lr"), 0.02, 0.3, log=True), "leaf": trial.suggest_int(s("leaf"), 8, 64, log=True),
                "l2": trial.suggest_float(s("l2"), 1e-4, 10.0, log=True), "it": trial.suggest_int(s("it"), 50, 400, log=True)}
    if mt == "xgb":
        return {"lr": trial.suggest_float(s("lr"), 0.02, 0.3, log=True), "md": trial.suggest_int(s("md"), 2, 8),
                "ne": trial.suggest_int(s("ne"), 50, 400, log=True), "ss": trial.suggest_float(s("ss"), 0.6, 1.0),
                "cs": trial.suggest_float(s("cs"), 0.6, 1.0)}
    if mt == "lgbm":
        return {"lr": trial.suggest_float(s("lr"), 0.02, 0.3, log=True), "nl": trial.suggest_int(s("nl"), 8, 64, log=True),
                "ne": trial.suggest_int(s("ne"), 50, 400, log=True), "ss": trial.suggest_float(s("ss"), 0.6, 1.0),
                "cs": trial.suggest_float(s("cs"), 0.6, 1.0)}
    if mt == "svc":
        return {"c": trial.suggest_float(s("c"), 1e-2, 1e3, log=True), "g": trial.suggest_float(s("g"), 1e-4, 1.0, log=True)}
    if mt == "knn":
        return {"n": trial.suggest_int(s("n"), 1, 40), "w": trial.suggest_categorical(s("w"), ["uniform", "distance"])}
    if mt == "dt":
        return {"d": trial.suggest_int(s("d"), 2, 24), "msl": trial.suggest_int(s("msl"), 1, 20, log=True)}
    if mt == "cat":
        return {"lr": trial.suggest_float(s("lr"), 0.02, 0.3, log=True), "dep": trial.suggest_int(s("dep"), 3, 8),
                "it": trial.suggest_int(s("it"), 100, 500, log=True), "l2": trial.suggest_float(s("l2"), 1.0, 10.0, log=True)}
    if mt in ("rf", "et"):
        return {"ne": trial.suggest_int(s("ne"), 100, 400, log=True),
                "mf": trial.suggest_categorical(s("mf"), ["sqrt", "log2", 0.5, 0.8]),
                "msl": trial.suggest_int(s("msl"), 1, 10, log=True)}
    if mt == "lr":
        return {"c": trial.suggest_float(s("c"), 1e-3, 1e2, log=True)}
    if mt == "nb":
        return {"vs": trial.suggest_float(s("vs"), 1e-11, 1e-3, log=True)}
    if mt == "mlp":
        return {"h": trial.suggest_categorical(s("h"), [32, 64, 128, 256]), "layers": trial.suggest_int(s("layers"), 1, 2),
                "a": trial.suggest_float(s("a"), 1e-6, 1e-1, log=True), "lri": trial.suggest_float(s("lri"), 1e-4, 1e-2, log=True)}
    raise ValueError(mt)


def _estimator(mt, p, K, seed, n, proba=True):
    if mt == "hgb":
        return HistGradientBoostingClassifier(learning_rate=p["lr"], max_leaf_nodes=p["leaf"], l2_regularization=p["l2"],
                                              max_iter=p["it"], early_stopping=False, random_state=seed)
    if mt == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(learning_rate=p["lr"], max_depth=p["md"], n_estimators=p["ne"], subsample=p["ss"],
                             colsample_bytree=p["cs"], tree_method="hist", device="cpu", n_jobs=1, random_state=seed, verbosity=0)
    if mt == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(learning_rate=p["lr"], num_leaves=p["nl"], n_estimators=p["ne"], subsample=p["ss"],
                              subsample_freq=1, colsample_bytree=p["cs"], n_jobs=1, random_state=seed, verbose=-1)
    if mt == "svc":
        # probability=True: Platt scaling fitted by internal 5-fold CV on the training rows only
        return SVC(C=p["c"], gamma=p["g"], kernel="rbf", probability=proba, random_state=seed,
                   max_iter=(500_000 if n <= 5000 else 200_000), cache_size=500)
    if mt == "knn":
        return KNeighborsClassifier(n_neighbors=p["n"], weights=p.get("w", "uniform"))
    if mt == "dt":
        return DecisionTreeClassifier(max_depth=p["d"], min_samples_leaf=p.get("msl", 1), random_state=seed)
    if mt == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(learning_rate=p["lr"], depth=p["dep"], iterations=p["it"], l2_leaf_reg=p["l2"],
                                  thread_count=1, random_seed=seed, verbose=0, allow_writing_files=False)
    if mt == "rf":
        return RandomForestClassifier(n_estimators=p["ne"], max_features=p["mf"], min_samples_leaf=p["msl"], n_jobs=1, random_state=seed)
    if mt == "et":
        return ExtraTreesClassifier(n_estimators=p["ne"], max_features=p["mf"], min_samples_leaf=p["msl"], n_jobs=1, random_state=seed)
    if mt == "lr":
        return LogisticRegression(C=p["c"], max_iter=2000)
    if mt == "nb":
        return GaussianNB(var_smoothing=p["vs"])
    if mt == "brf":                                        # imbalanced-learn ensembles (members only)
        from imblearn.ensemble import BalancedRandomForestClassifier
        return BalancedRandomForestClassifier(n_estimators=p["ne"], sampling_strategy="all", replacement=True, bootstrap=False,
                                              n_jobs=1, random_state=seed)
    if mt == "eec":
        from imblearn.ensemble import EasyEnsembleClassifier
        return EasyEnsembleClassifier(n_estimators=p["ne"], n_jobs=1, random_state=seed)
    if mt == "rusb":
        from imblearn.ensemble import RUSBoostClassifier
        return RUSBoostClassifier(n_estimators=p["ne"], random_state=seed)
    if mt == "mlp":
        return MLPClassifier(hidden_layer_sizes=(p["h"],) * p["layers"], alpha=p["a"], learning_rate_init=p["lri"],
                             early_stopping=True, max_iter=300, random_state=seed)
    raise ValueError(mt)


class Member:
    """One estimator with its own column subset and preprocessing; predict_proba is aligned to classes 0..K-1."""

    def __init__(self, mt, params, mask, K, seed=0, proba=True):
        # proba=False: label-only scoring during search (SVC skips Platt scaling)
        self.mt, self.params, self.mask, self.K, self.seed, self.proba = mt, dict(params), list(mask), K, seed, proba

    def fit(self, X, y, balanced=False):
        Xs = X[self.mask]
        cat = [c for c in self.mask if not pd.api.types.is_numeric_dtype(Xs[c])]
        num = [c for c in self.mask if c not in cat]
        steps = []
        if num:
            nt = [("imp", SimpleImputer(strategy="median"))]
            if self.mt in SCALE_SENSITIVE:
                nt.append(("sc", StandardScaler()))
            steps.append(("num", Pipeline(nt), num))
        if cat:
            enc = (OneHotEncoder(handle_unknown="ignore", sparse_output=False) if self.mt in SCALE_SENSITIVE
                   else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
            steps.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("enc", enc)]), cat))
        self.classes_ = np.unique(y)
        est = _estimator(self.mt, self.params, self.K, self.seed, len(y), self.proba)
        if self.mt == "knn":
            est.n_neighbors = max(1, min(est.n_neighbors, len(y) - 1))
        # labels re-indexed to 0..len(classes_)-1 so that XGBoost accepts a fold with an absent class
        self._map = {c: i for i, c in enumerate(self.classes_)}
        self.pipe = Pipeline([("pre", ColumnTransformer(steps)), ("est", est)])
        self.prior = None
        if Xs.nunique(dropna=True).max() <= 1 and len(self.classes_) > 1:
            # every selected column is constant on these rows: predict the class proportions
            self.prior = np.bincount(np.vectorize(self._map.get)(y), minlength=len(self.classes_)) / len(y)
            self.pipe = None
        elif len(self.classes_) == 1:
            self.pipe = None
        else:
            yy = np.vectorize(self._map.get)(y)
            try:
                if balanced and self.mt not in ("knn", "mlp"):     # balanced class weights (knn and mlp take none)
                    cnt = np.bincount(yy); w = (len(yy) / (len(cnt) * cnt))[yy]
                    self.pipe.fit(Xs, yy, est__sample_weight=w)
                else:
                    self.pipe.fit(Xs, yy)
            except ValueError:
                if self.mt not in ("eec", "rusb"):
                    raise
                # AdaBoost inside EasyEnsemble/RUSBoost can stop with a base learner worse than random: class proportions
                self.prior = np.bincount(yy, minlength=len(self.classes_)) / len(yy); self.pipe = None
        return self

    def predict(self, X):
        if self.pipe is None:
            return np.full(len(X), self.classes_[0] if self.prior is None else self.classes_[int(np.argmax(self.prior))])
        if self.proba:
            return np.argmax(self.predict_proba(X), 1)
        return self.classes_[np.asarray(self.pipe.predict(X[self.mask])).ravel().astype(int)]

    def predict_proba(self, X):
        out = np.zeros((len(X), self.K))
        if self.pipe is None:
            if self.prior is None:
                out[:, self.classes_[0]] = 1.0
            else:
                out[:, self.classes_] = self.prior
            return out
        out[:, self.classes_] = self.pipe.predict_proba(X[self.mask])
        return out
