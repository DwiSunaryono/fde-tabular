"""Combiners of the best tuned learner and the members, fitted on inner out-of-fold probabilities only.

  stack   multinomial logistic regression on the probabilities of the best learner and the members; C in
          {0.01, 0.1, 1, 10} chosen by cross-validated Brier score (stratified 3-fold on the out-of-fold rows); kept
          only when that Brier score is lower than the best learner's
  scale   per-class probability scales chosen by coordinate ascent (2 passes over the classes) on out-of-fold
          probabilities with the run metric; grid {0.5, 0.67, 1, 1.5, 2, 3} (the "wide" option uses
          {0.25, 0.5, 0.67, 1, 1.5, 2, 3, 5, 8, 12, 20})
  stack_scaled  stack followed by the scales chosen on its cross-validated probabilities
  mixed         the average of stack_scaled and the member mean, followed by scales chosen on that average
A combiner that keeps nothing (no stack, no scale) returns the best learner unchanged (weight 0).
The fitted combiner is returned as a small specification that is applied to new probabilities by `apply`.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

EPS = 1e-12
STACK_C = (0.01, 0.1, 1.0, 10.0)
GRID = (0.5, 0.67, 1.0, 1.5, 2.0, 3.0)
GRID_WIDE = (0.25, 0.5, 0.67, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)


def fast_score(y, p, K, metric):
    """Weighted or macro F1 from a K x K confusion matrix (same values as sklearn f1_score)."""
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(float); row = cm.sum(1); col = cm.sum(0)
    f1 = np.divide(2 * tp, row + col, out=np.zeros(K), where=(row + col) > 0)
    if metric == "wF1":
        return float((row * f1).sum() / max(row.sum(), 1))
    present = (row + col) > 0                     # sklearn macro averages over labels present in y or p
    return float(f1[present].mean())


def brier(y, p, K):
    return float(((p - np.eye(K)[y]) ** 2).sum(1).mean())


def _proba(m, Z, K):
    out = np.zeros((len(Z), K)); out[:, m.classes_] = m.predict_proba(Z); return out


def scales(o_sel, y, K, metric, grid=GRID):
    """Per-class scales by coordinate ascent (2 passes); None when no scale improves the score."""
    sc = lambda p: fast_score(y, p.argmax(1), K, metric)
    w = np.ones(K); best = sc(o_sel)
    for _ in range(2):
        for c in range(K):
            for v in grid:
                t = w.copy(); t[c] = v
                s_ = sc(o_sel * t)
                if s_ > best + EPS:
                    best, w = s_, t
    return None if np.all(w == 1) else w


def stack(oof, labs, y, K, seed):
    """(cv probabilities, fitted model or None). Columns: the best learner, then the members in `labs` order."""
    o = oof["P1"]
    if not labs or len(np.unique(y)) < 2:
        return o, None
    Zo = np.hstack([oof[l] for l in ["P1"] + labs])
    mc = np.bincount(y).min()
    k = int(max(2, min(3, mc))) if mc >= 2 else 0
    if k < 2:
        return o, None
    folds = list(StratifiedKFold(k, shuffle=True, random_state=seed).split(Zo, y))
    best = None
    for c in STACK_C:
        cv = np.zeros((len(y), K))
        for tr, te in folds:
            cv[te] = _proba(LogisticRegression(C=c, max_iter=1000).fit(Zo[tr], y[tr]), Zo[te], K)
        b_ = brier(y, cv, K)
        if best is None or b_ < best[0] - EPS:
            best = (b_, c, cv)
    b_, c, cv = best
    if b_ >= brier(y, o, K) - EPS:
        return o, None
    return cv, LogisticRegression(C=c, max_iter=1000).fit(Zo, y)


def fit_combiner(kind, oof, labs, y, K, metric, seed, grid=GRID):
    """Fits combiner `kind` ('stack_scaled' or 'mixed'). Returns (spec, cross-fitted/inner probabilities, weight)."""
    o = oof["P1"]
    cv, lr = stack(oof, labs, y, K, seed)
    w = scales(cv, y, K, metric, grid)
    spec = {"kind": "stack_scaled", "labs": list(labs), "lr": lr, "w1": w, "K": K}
    if w is None and lr is None:
        s_o, s_a = o, 0.0
    else:
        s_o, s_a = cv * (w if w is not None else 1.0), 1.0
    if kind == "stack_scaled":
        if s_a == 0:
            return {"kind": "best"}, o, 0.0
        return spec, s_o, 1.0
    # mixed: average of stack_scaled and the member mean, then scales
    if not labs:
        m_o, m_a = o, 0.0
    else:
        m_o, m_a = np.mean([oof[l] for l in labs], 0), 1.0
    if s_a == 0 and m_a == 0:
        return {"kind": "best"}, o, 0.0
    mix = 0.5 * (s_o + m_o)
    w2 = scales(mix, y, K, metric, grid)
    spec = {"kind": "mixed", "labs": list(labs), "lr": lr, "w1": w, "stack_used": s_a > 0, "mean_used": m_a > 0, "w2": w2, "K": K}
    return spec, mix * (w2 if w2 is not None else 1.0), 1.0


def apply(spec, probs):
    """Probabilities of the combined model; probs: {label: probabilities} of the best learner ('P1') and members."""
    if spec["kind"] == "best":
        return probs["P1"]
    K = spec["K"]
    if spec["lr"] is not None:
        s = _proba(spec["lr"], np.hstack([probs[l] for l in ["P1"] + spec["labs"]]), K)
    else:
        s = probs["P1"]
    if spec["w1"] is not None:
        s = s * spec["w1"]
    if spec["kind"] == "stack_scaled":
        return s
    if not spec["stack_used"]:
        s = probs["P1"]
    m = np.mean([probs[l] for l in spec["labs"]], 0) if spec["mean_used"] else probs["P1"]
    mix = 0.5 * (s + m)
    return mix * spec["w2"] if spec["w2"] is not None else mix
