"""Triangular fuzzy arithmetic. No defuzzification before fuzzy TOPSIS distances."""

import re

import numpy as np

from .engine import make_result, weights_vector

DEFAULT_SCALE = {
    "Very low": (1, 1, 3),
    "Low": (1, 3, 5),
    "Medium": (3, 5, 7),
    "High": (5, 7, 9),
    "Very high": (7, 9, 9),
}
FUZZY_METHODS = {
    "Fuzzy TOPSIS": "Ratio-normalized triangles, fuzzy multiplication, observed scalar endpoint ideals, sum of vertex distances. Higher closeness is better.",
    "Fuzzy SAW": "Ratio-normalized triangular weighted sum, followed by centroid defuzzification. Higher centroid is better.",
}


def validate_triangles(values):
    a = np.asarray(values, dtype=float)
    if a.ndim < 1 or a.shape[-1] != 3 or not np.isfinite(a).all():
        raise ValueError("Fuzzy values must contain three finite numbers (lower, modal, upper).")
    if (a[..., 0] > a[..., 1]).any() or (a[..., 1] > a[..., 2]).any():
        raise ValueError("Every triangle must satisfy lower ≤ modal ≤ upper.")
    return a


def parse_cell(value, scale=None):
    if isinstance(value, (list, tuple, np.ndarray)):
        return validate_triangles(value)
    text = str(value).strip()
    lookup = {
        k.strip().casefold(): v for k, v in (DEFAULT_SCALE if scale is None else scale).items()
    }
    if text.casefold() in lookup:
        return validate_triangles(lookup[text.casefold()])
    try:
        parts = re.split(r"[,;]", text.strip("()[] "))
        numbers = [float(p.strip()) for p in parts]
    except (ValueError, TypeError):
        raise ValueError(
            f"Unknown value '{text}'. Enter a number, a triangle like 2;3;5, or a term from the linguistic scale."
        ) from None
    if len(numbers) == 1:
        numbers *= 3
    if len(numbers) != 3:
        raise ValueError(f"'{text}' needs one numeric value or three triangle endpoints.")
    return validate_triangles(numbers)


def parse_matrix(cells, scale=None):
    rows = []
    for i, row in enumerate(cells):
        parsed = []
        for j, cell in enumerate(row):
            try:
                parsed.append(parse_cell(cell, scale))
            except ValueError as exc:
                raise ValueError(f"Alternative {i + 1}, criterion {j + 1}: {exc}") from exc
        rows.append(parsed)
    result = np.asarray(rows)
    if result.ndim != 3:
        raise ValueError("Enter at least one alternative and one criterion.")
    return result


def fuzzy_weights(values, count):
    w = np.asarray(values, dtype=float)
    if w.ndim == 1:
        crisp = weights_vector(w, count)
        return np.repeat(crisp[:, None], 3, axis=1)
    w = validate_triangles(w)
    if w.shape != (count, 3) or (w < 0).any() or w[:, 1].sum() <= 0:
        raise ValueError("Fuzzy weights must be nonnegative triangles with a positive modal total.")
    # A common positive scalar preserves triangular ordering and has no effect on rankings.
    return w / w[:, 1].sum()


def fuzzy_rank(matrix, weights, benefits, method="Fuzzy TOPSIS"):
    x = validate_triangles(matrix)
    b = np.asarray(benefits, dtype=bool)
    if x.ndim != 3 or min(x.shape[:2]) < 1 or b.shape != (x.shape[1],):
        raise ValueError(
            "Fuzzy decision matrix must be alternatives × criteria × 3, with one direction per criterion."
        )
    if method not in FUZZY_METHODS:
        raise ValueError(f"Unknown fuzzy method: {method}")
    w = fuzzy_weights(weights, x.shape[1])
    active = w[:, 2] > 0
    x, b, w = x[:, active], b[active], w[active]
    if (x < 0).any():
        raise ValueError("Fuzzy ratio normalization requires nonnegative triangles.")
    if (~b).any() and (x[:, ~b, 0] <= 0).any():
        raise ValueError(
            "Fuzzy cost criteria require strictly positive lower endpoints for reciprocal normalization."
        )
    r = np.zeros_like(x)
    for j in range(x.shape[1]):
        if b[j]:
            hi = x[:, j, 2].max()
            r[:, j] = x[:, j] / hi if hi else 1
        else:
            r[:, j] = x[:, j, 0].min() / x[:, j, ::-1]
    weighted = r * w[None, :, :]
    if method == "Fuzzy SAW":
        totals = weighted.sum(axis=1)
        return make_result(
            method,
            totals.mean(axis=1),
            normalized=r,
            weighted=weighted,
            fuzzy_totals=totals,
        )
    ideal = weighted[:, :, 2].max(axis=0)[None, :, None]
    anti = weighted[:, :, 0].min(axis=0)[None, :, None]
    dp = np.sqrt(np.mean((weighted - ideal) ** 2, axis=2)).sum(axis=1)
    dn = np.sqrt(np.mean((weighted - anti) ** 2, axis=2)).sum(axis=1)
    scores = np.divide(dn, dp + dn, out=np.full(len(x), 0.5), where=dp + dn != 0)
    return make_result(
        method,
        scores,
        normalized=r,
        weighted=weighted,
        positive_distance=dp,
        negative_distance=dn,
    )


def fuzzy_ahp(pairwise):
    """Buckley geometric-mean weights with triangular inverse-sum normalization."""
    a = validate_triangles(pairwise)
    if a.ndim != 3 or a.shape[0] != a.shape[1] or (a <= 0).any():
        raise ValueError("Fuzzy AHP needs a positive square matrix of triangles.")
    if not np.allclose(a * a.transpose(1, 0, 2)[:, :, ::-1], 1, atol=1e-7):
        raise ValueError("Fuzzy AHP comparisons must be reciprocal: (l,m,u) ↔ (1/u,1/m,1/l).")
    if not np.allclose(a[np.arange(len(a)), np.arange(len(a))], 1):
        raise ValueError("Fuzzy AHP self-comparisons must be (1,1,1).")
    means = np.exp(np.log(a).mean(axis=1))
    w = means / means.sum(axis=0)[::-1]
    crisp = w.mean(axis=1)
    return crisp / crisp.sum(), w
