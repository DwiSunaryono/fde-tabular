"""Cross-fitted acceptance rule: paired bootstrap lower bound of the gain, compared with a margin.

The candidate replaces the reference model only when the lower 5% bootstrap quantile of its gain exceeds the
margin. The gain is computed from the joint counts of (true label, reference prediction, candidate prediction) on
the cross-fitted outer predictions, so a bootstrap replicate is a multinomial draw of those counts (Efron and
Tibshirani 1993). Metrics: weighted-F1 and macro-F1 for two classes, weighted-F1 for more classes.
"""
from dataclasses import dataclass

import numpy as np

MARGIN, QUANTILE, REPLICATES, OUTER_FOLDS = 0.005, 0.05, 20_000, 3
CHUNK = 2500                       # multiclass replicates are drawn in chunks of this size


@dataclass(frozen=True)
class TestResult:
    accept: bool
    lower_bound: float
    gain: float                    # gain on all rows (point estimate)
    n: int
    K: int
    metric: str


def joint_counts(y, reference, candidate, K):
    """K^3 counts of (y, reference, candidate), indexed by (y * K + reference) * K + candidate."""
    y, r, c = (np.asarray(v, dtype=np.int64) for v in (y, reference, candidate))
    K = max(int(K), 2)
    return np.bincount((y * K + r) * K + c, minlength=K ** 3)


def _safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(np.asarray(a, dtype=float)), where=np.asarray(b) > 0)


def _binary_score(m, metric):
    tn, fp, fn, tp = m[..., 0, 0], m[..., 0, 1], m[..., 1, 0], m[..., 1, 1]
    f0 = _safe_div(2 * tn, 2 * tn + fp + fn); f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    if metric == "macroF1":
        return 0.5 * (f0 + f1)
    if metric == "wF1":
        return _safe_div((tn + fp) * f0 + (tp + fn) * f1, tn + fp + fn + tp)
    raise ValueError(f"unknown metric {metric!r}")


def _weighted_f1(M):
    tp = np.diagonal(M, axis1=-2, axis2=-1); row = M.sum(-1); col = M.sum(-2)
    f1 = np.divide(2 * tp, row + col, out=np.zeros_like(tp), where=(row + col) > 0)
    return (row * f1).sum(-1) / np.maximum(row.sum(-1), 1e-300)


def gain_from_counts(c, K, metric):
    """Score of the candidate minus score of the reference; leading axes are vectorized."""
    c = np.asarray(c, dtype=float)
    if K == 2:
        t = c.reshape(c.shape[:-1] + (2, 2, 2))
        return _binary_score(t.sum(axis=-2), metric) - _binary_score(t.sum(axis=-1), metric)
    if metric != "wF1":
        raise ValueError("for more than two classes the metric is weighted-F1 ('wF1')")
    T = c.reshape(c.shape[:-1] + (K, K, K))
    return _weighted_f1(T.sum(-2)) - _weighted_f1(T.sum(-1))


def bootstrap_lower_bound(c, K, metric, seed=0, replicates=REPLICATES, quantile=QUANTILE):
    """Lower bootstrap quantile of the gain; counts are resampled from a multinomial distribution."""
    c = np.asarray(c); rng = np.random.default_rng(seed); n = int(c.sum())
    if K == 2:
        return float(np.quantile(gain_from_counts(rng.multinomial(n, c / n, size=replicates), 2, metric), quantile))
    nz = np.flatnonzero(c); out = []
    for s in range(0, replicates, CHUNK):
        m = min(CHUNK, replicates - s)
        W = np.zeros((m, K ** 3)); W[:, nz] = rng.multinomial(n, c[nz] / n, size=m)
        out.append(gain_from_counts(W, K, metric))
    return float(np.quantile(np.concatenate(out), quantile))


def deployment_test(y, pred_reference, pred_candidate, metric="wF1", K=None, seed=0, margin=MARGIN):
    """Accept the candidate when the lower bootstrap bound of its gain over the reference exceeds the margin."""
    y, r, c = (np.asarray(v) for v in (y, pred_reference, pred_candidate))
    K = max(2, int(max(y.max(), r.max(), c.max())) + 1) if K is None else int(K)
    cnt = joint_counts(y, r, c, K); lb = bootstrap_lower_bound(cnt, K, metric, seed)
    return TestResult(accept=lb > margin, lower_bound=lb, gain=float(gain_from_counts(cnt, K, metric)),
                      n=len(y), K=K, metric=metric)


MINORITY_IMBALANCED = 0.15


def gate_metric(y):
    """Metric of a run: macro-F1 for two classes with a minority share below 0.15, otherwise weighted-F1."""
    cnt = np.bincount(y)
    return "macroF1" if (len(cnt) == 2 and cnt.min() / cnt.sum() < MINORITY_IMBALANCED) else "wF1"
