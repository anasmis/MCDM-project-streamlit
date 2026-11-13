"""Crisp MCDM methods and weighting with explicit normalization conventions."""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog
from scipy.stats import rankdata

METHODS = {
    "TOPSIS": "Vector normalization; Euclidean distance to observed ideal solutions.",
    "SAW / WSM": "Weighted sum of benefit-oriented min–max values; constants map to 1.",
    "WPM": "Weighted product of positive benefit ratios x/max and cost ratios min/x.",
    "WASPAS": "Blend of ratio-normalized weighted sum and weighted product (lambda).",
    "VIKOR": "Compromise index Q from group utility and individual regret (v); lower is better.",
    "MOORA": "Vector-normalized benefit sum minus cost sum.",
    "COPRAS": "Positive column-sum normalization, benefit sums and reciprocal cost sums.",
    "EDAS": "Positive and negative distances from each criterion's average; positive data required.",
    "PROMETHEE II": "Net outranking flow with the usual (strict preference) function; ties give no preference.",
}


@dataclass
class Result:
    method: str
    scores: np.ndarray
    ranks: np.ndarray
    higher_is_better: bool = True
    details: dict = field(default_factory=dict)


def weights_vector(weights, count):
    w = np.asarray(weights, dtype=float)
    if w.shape != (count,) or not np.isfinite(w).all() or (w < 0).any() or not (w > 0).any():
        raise ValueError("Weights must be finite, nonnegative, and have a positive total.")
    w = w / w.max()
    return w / w.sum()


def validate(matrix, benefits):
    x = np.asarray(matrix, dtype=float)
    b = np.asarray(benefits)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError(
            "Enter a nonempty matrix of finite numeric values; missing values are not filled automatically."
        )
    if b.shape != (x.shape[1],) or not np.isin(b, [True, False]).all():
        raise ValueError("Set a benefit/cost direction for each criterion.")
    return x, b.astype(bool)


def minmax(matrix, benefits):
    x, b = validate(matrix, benefits)
    scale = np.max(np.abs(x), axis=0)
    x = x / np.where(scale == 0, 1, scale)
    lo, hi = x.min(axis=0), x.max(axis=0)
    span = hi - lo
    r = np.divide(x - lo, span, out=np.ones_like(x), where=span != 0)
    r[:, ~b] = 1 - r[:, ~b]
    r[:, span == 0] = 1
    return r


def make_result(method, scores, higher=True, **details):
    s = np.asarray(scores, dtype=float)
    if not np.isfinite(s).all():
        raise ValueError(
            "Calculation exceeded numerical limits. Rescale the criterion units and try again."
        )
    ranks = rankdata(-np.round(s, 12) if higher else np.round(s, 12), method="min").astype(int)
    return Result(method, s, ranks, higher, details)


def rank(matrix, weights, benefits, method="TOPSIS", *, v=0.5, lam=0.5):
    x, b = validate(matrix, benefits)
    w = weights_vector(weights, x.shape[1])
    if method not in METHODS:
        raise ValueError(f"Unknown ranking method: {method}")
    if not 0 <= v <= 1 or not 0 <= lam <= 1:
        raise ValueError("VIKOR v and WASPAS lambda must be between 0 and 1.")
    # Remove zero-weight columns so unsupported values in inactive criteria cannot affect a method.
    active = w > 0
    x, b, w = x[:, active], b[active], w[active]
    if method in {"WPM", "WASPAS", "COPRAS", "EDAS"} and (x <= 0).any():
        raise ValueError(
            f"{method} requires strictly positive values in all active criteria. Choose another method or revise the measurement scale."
        )
    # Scaling by a positive column constant keeps all these methods invariant and avoids overflow.
    scale = np.max(np.abs(x), axis=0)
    x = x / np.where(scale == 0, 1, scale)
    if method in {"TOPSIS", "MOORA"}:
        norms = np.linalg.norm(x, axis=0)
        r = np.divide(x, norms, out=np.zeros_like(x), where=norms != 0)
        y = r * w
        if method == "MOORA":
            return make_result(
                method,
                y[:, b].sum(axis=1) - y[:, ~b].sum(axis=1),
                normalized=r,
                weighted=y,
            )
        ideal = np.where(b, y.max(axis=0), y.min(axis=0))
        anti = np.where(b, y.min(axis=0), y.max(axis=0))
        dp, dn = np.linalg.norm(y - ideal, axis=1), np.linalg.norm(y - anti, axis=1)
        scores = np.divide(dn, dp + dn, out=np.full(len(x), 0.5), where=dp + dn != 0)
        return make_result(
            method,
            scores,
            normalized=r,
            weighted=y,
            positive_distance=dp,
            negative_distance=dn,
        )
    if method == "SAW / WSM":
        r = minmax(x, b)
        return make_result(method, r @ w, normalized=r, weighted=r * w)
    if method in {"WPM", "WASPAS"}:
        r = x / x.max(axis=0)
        r[:, ~b] = x[:, ~b].min(axis=0) / x[:, ~b]
        product = np.exp(np.log(r) @ w)
        score = product if method == "WPM" else lam * (r @ w) + (1 - lam) * product
        return make_result(method, score, normalized=r)
    if method == "VIKOR":
        regret = (1 - minmax(x, b)) * w
        s, r = regret.sum(axis=1), regret.max(axis=1)

        def unit(a):
            return (a - a.min()) / np.ptp(a) if np.ptp(a) else np.zeros_like(a)

        q = v * unit(s) + (1 - v) * unit(r)
        order = np.argsort(q, kind="stable")
        advantage = len(x) == 1 or q[order[1]] - q[order[0]] >= 1 / (len(x) - 1) - 1e-12
        stability = np.isclose(s[order[0]], s.min()) or np.isclose(r[order[0]], r.min())
        return make_result(
            method,
            q,
            False,
            utility=s,
            regret=r,
            acceptable_advantage=bool(advantage),
            acceptable_stability=bool(stability),
        )
    if method == "COPRAS":
        r = x / x.sum(axis=0)
        y = r * w
        plus, minus = y[:, b].sum(axis=1), y[:, ~b].sum(axis=1)
        score = plus.copy()
        if (~b).any():
            score += minus.sum() / (minus * (1 / minus).sum())
        return make_result(method, score / score.max(), normalized=r, weighted=y)
    if method == "EDAS":
        avg = x.mean(axis=0)
        delta = (x - avg) / avg * np.where(b, 1, -1)
        sp, sn = np.maximum(delta, 0) @ w, np.maximum(-delta, 0) @ w
        nsp = sp / sp.max() if sp.max() else np.zeros_like(sp)
        nsn = 1 - sn / sn.max() if sn.max() else np.ones_like(sn)
        return make_result(method, (nsp + nsn) / 2, positive_distance=sp, negative_distance=sn)
    # Usual PROMETHEE II can be computed through per-column ranks instead of an O(n^2) tensor.
    oriented = x * np.where(b, 1, -1)
    net = np.zeros(len(x))
    if len(x) > 1:
        for j in range(x.shape[1]):
            wins = rankdata(oriented[:, j], method="min") - 1
            losses = len(x) - rankdata(oriented[:, j], method="max")
            net += w[j] * (wins - losses) / (len(x) - 1)
    return make_result(method, net)


def objective_weights(matrix, benefits, method):
    x, b = validate(matrix, benefits)
    if method == "Equal":
        return np.full(x.shape[1], 1 / x.shape[1])
    r = minmax(x, b)
    varying = np.ptp(r, axis=0) > 1e-12
    if not varying.any():
        raise ValueError(
            f"{method} cannot derive weights: all criteria are constant. Use equal or manual weights."
        )
    info = np.zeros(x.shape[1])
    z = r[:, varying]
    if method == "Entropy":
        p = z / z.sum(axis=0)
        plogp = np.zeros_like(p)
        np.log(p, out=plogp, where=p > 0)
        info[varying] = np.maximum(0, 1 + (p * plogp).sum(axis=0) / np.log(len(x)))
    elif method == "CRITIC":
        corr = np.atleast_2d(np.corrcoef(z, rowvar=False)) if z.shape[1] > 1 else np.ones((1, 1))
        info[varying] = z.std(axis=0) * np.maximum(0, 1 - corr).sum(axis=0)
        if varying.sum() == 1:
            info[varying] = 1
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    if info.sum() <= 1e-12:
        raise ValueError(
            f"{method} found no distinguishing information (for example, perfectly correlated criteria). Use manual or equal weights."
        )
    return info / info.sum()


def ahp(pairwise):
    a = np.asarray(pairwise, dtype=float)
    if (
        a.ndim != 2
        or a.shape[0] != a.shape[1]
        or not len(a)
        or not np.isfinite(a).all()
        or (a <= 0).any()
    ):
        raise ValueError("AHP needs a finite positive square comparison matrix.")
    if not np.allclose(a * a.T, 1, atol=1e-7) or not np.allclose(a.diagonal(), 1):
        raise ValueError("AHP comparisons must be reciprocal and have a diagonal of 1.")
    values, vectors = np.linalg.eig(a)
    idx = np.argmax(values.real)
    w = np.abs(vectors[:, idx].real)
    n = len(a)
    ci = max(0, (values[idx].real - n) / (n - 1)) if n > 1 else 0
    ri = {
        1: 0,
        2: 0,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45,
        10: 1.49,
    }.get(n)
    cr = None if ri is None else ci / ri if ri else 0.0
    return w / w.sum(), cr


def bwm(best_to_others, others_to_worst, best, worst):
    a, c = np.asarray(best_to_others, float), np.asarray(others_to_worst, float)
    n = len(a)
    if c.shape != (n,) or n < 2 or best == worst or not (0 <= best < n and 0 <= worst < n):
        raise ValueError("BWM needs distinct best and worst criteria and matching comparisons.")
    if (
        not np.isfinite(a).all()
        or not np.isfinite(c).all()
        or (a < 1).any()
        or (a > 9).any()
        or (c < 1).any()
        or (c > 9).any()
    ):
        raise ValueError("BWM comparisons must be finite values between 1 and 9.")
    if (
        not np.isclose(a[best], 1)
        or not np.isclose(c[worst], 1)
        or not np.isclose(a[worst], c[best])
    ):
        raise ValueError(
            "BWM self-comparisons must equal 1 and both best/worst comparisons must agree."
        )
    rows = []
    for j in range(n):
        for left, right, ratio in [(best, j, a[j]), (j, worst, c[j])]:
            row = np.zeros(n + 1)
            row[left] += 1
            row[right] -= ratio
            for sign in (1, -1):
                constraint = sign * row
                constraint[-1] = -1
                rows.append(constraint)
    cost = np.zeros(n + 1)
    cost[-1] = 1
    eq = np.r_[np.ones(n), 0][None, :]
    result = linprog(
        cost,
        A_ub=rows,
        b_ub=np.zeros(len(rows)),
        A_eq=eq,
        b_eq=[1],
        bounds=[(0, None)] * (n + 1),
        method="highs",
    )
    if not result.success:
        raise ValueError("BWM optimization failed: " + result.message)
    return weights_vector(result.x[:n], n), float(result.x[-1])
