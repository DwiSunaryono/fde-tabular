import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from fde import FDEClassifier
from fde.combiners import fast_score
from fde.deploy_test import acceptance_rule, gate_metric


def _data(n=400, weights=(0.85, 0.15), seed=0):
    X, y = make_classification(n_samples=n, n_features=8, n_informative=4, weights=list(weights), random_state=seed)
    X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    X["colour"] = np.where(X["x0"] > 0, "red", "blue")              # one text column
    X.loc[::17, "x1"] = np.nan                                       # some missing values
    return X, np.where(y == 1, "yes", "no")


def test_fast_score_matches_sklearn():
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(0); y = rng.integers(0, 3, 200); p = rng.integers(0, 3, 200)
    assert abs(fast_score(y, p, 3, "wF1") - f1_score(y, p, average="weighted")) < 1e-12
    assert abs(fast_score(y, p, 3, "macroF1") - f1_score(y, p, average="macro")) < 1e-12


def test_acceptance_rule_identical_predictions_rejected():
    y = np.array([0, 1] * 50); p = y.copy()
    r = acceptance_rule(y, p, p, metric="wF1", K=2)
    assert not r.accept and r.gain == 0


def test_gate_metric():
    assert gate_metric(np.array([0] * 90 + [1] * 10)) == "macroF1"
    assert gate_metric(np.array([0] * 50 + [1] * 50)) == "wF1"


@pytest.mark.slow
def test_fit_predict_save_load(tmp_path):
    X, y = _data()
    clf = FDEClassifier(random_state=0, n_jobs=2, trials=2).fit(X.iloc[:300], y[:300])
    assert clf.report_["imbalance_group"] in ("low", "moderate")
    p = clf.predict(X.iloc[300:]); pr = clf.predict_proba(X.iloc[300:])
    assert set(p) <= {"no", "yes"} and pr.shape == (100, 2)
    path = tmp_path / "m.fde"; clf.save(path)
    assert (FDEClassifier.load(path).predict(X.iloc[300:]) == p).all()


@pytest.mark.slow
def test_report_has_acceptance_rule():
    X, y = _data(weights=(0.5, 0.5))
    clf = FDEClassifier(random_state=0, n_jobs=1, trials=2).fit(X, y)
    r = clf.report_["acceptance_rule"]
    assert r["margin"] == 0.0 and clf.uses_ensemble_ == (clf.report_["models_in_predictor"] > 1)


def test_unknown_members_rejected():
    X, y = _data(n=60)
    with pytest.raises(ValueError):
        FDEClassifier(members="other").fit(X, y)


def test_unknown_device_rejected():
    X, y = _data(n=60)
    with pytest.raises(ValueError):
        FDEClassifier(device="tpu").fit(X, y)


def test_greedy_select_starts_from_best_learner():
    from fde.speed import greedy_select
    rng = np.random.default_rng(0); y = rng.integers(0, 2, 300)
    good = np.eye(2)[y] * 0.8 + 0.1; noisy = rng.dirichlet([1, 1], 300)
    sel = greedy_select({"P1": noisy, "m1": good, "m2": noisy}, ["m1", "m2"], y, 2)
    assert sel is not None and "P1" in sel and sel.get("m1", 0) > 0 and abs(sum(sel.values()) - 1) < 1e-12


def test_speed_report_needs_candidates():
    from fde.speed import speed_report
    clf = FDEClassifier(); clf.candidates_ = None
    with pytest.raises(ValueError):
        speed_report(clf, pd.DataFrame({"x": [1.0]}))


@pytest.mark.slow
def test_speed_report_on_holdout():
    from fde.speed import speed_report
    X, y = _data()
    clf = FDEClassifier(random_state=0, n_jobs=2, trials=2, members="light", keep_candidates=True).fit(X.iloc[:300], y[:300])
    s = speed_report(clf, X.iloc[300:].reset_index(drop=True), y[300:], repeats=2)
    assert set(s["pipelines"]) == {"fde", "greedy", "full", "best"} and s["ratio_greedy_to_fde"] > 0
    assert s["pipelines"]["fde"]["models"] == clf.report_["models_called_at_prediction"]
    assert (clf.predict(X.iloc[300:]) == clf.classes_[clf.predict_proba(X.iloc[300:]).argmax(1)]).all()


@pytest.mark.slow
def test_cross_validate_scores_every_row_once():
    from fde import cross_validate
    X, y = _data()
    r = cross_validate(X, y, cv=3, random_state=0, n_jobs=2, trials=2, members="light")
    assert r["cv"] == 3 and len(r["folds"]) == 3
    assert sum(f["rows"] for f in r["folds"]) == len(y)
    for p in ("fde", "best_tuned_learner"):
        vals = [f[p]["macro_f1"] for f in r["folds"]]
        assert abs(r["mean"][p]["macro_f1"] - np.mean(vals)) < 1e-4
    with pytest.raises(ValueError):
        cross_validate(X, y, cv=1)


def test_demo_data_shipped():
    from fde.cli import _demo_sfts
    X, y = _demo_sfts()
    assert X.shape == (788, 30) and set(y) == {"died", "survived"} and (y == "died").sum() == 96


def test_mlp_in_separate_process_matches(monkeypatch):
    # the macOS path (training in a separate process, numpy forward pass) gives the same probabilities
    from fde import _torch_mlp as tm, extended
    rng = np.random.default_rng(0); A = rng.normal(size=(300, 10)).astype("float32"); y = (A[:, 0] > 0).astype(int)
    net = tm.train(A[:240], y[:240], A[240:], y[240:], 2, 0, 2)
    monkeypatch.setattr(extended, "ISOLATE_TORCH", True)
    W = extended._mlp(A[:240], y[:240], A[240:], y[240:], 2, 0, 2)
    import torch
    with torch.no_grad():
        p = torch.softmax(net(torch.tensor(A)), 1).numpy()
    assert np.abs(tm.forward(W, A) - p).max() < 1e-5
