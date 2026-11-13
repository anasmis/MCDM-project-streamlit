import numpy as np
import pytest

from decisionlab.engine import METHODS, ahp, bwm, objective_weights, rank


@pytest.mark.parametrize("method", METHODS)
def test_dominance_costs_and_ties(method):
    result = rank([[9, 1], [5, 5], [1, 9], [9, 1]], [0.6, 0.4], [True, False], method)
    assert result.ranks.tolist() == [1, 3, 4, 1]


@pytest.mark.parametrize("method", METHODS)
def test_constants_zero_weights_single_alternative_and_scale_invariance(method):
    for x in [[[2, 2], [2, 2]], [[2, 2]]]:
        r = rank(x, [1, 1], [True, False], method)
        assert np.isfinite(r.scores).all()
        assert (r.ranks == 1).all()
    x = np.array([[2, 3, -8], [5, 6, 0], [9, 4, -1]], float)
    a = rank(x, [1, 2, 0], [True, False, True], method)
    c = rank(x * [1000, 0.001, 3], [1, 2, 0], [True, False, True], method)
    np.testing.assert_allclose(a.scores, c.scores, atol=1e-12)


def test_hand_calculated_scores():
    x, w, b = [[1, 3], [3, 1], [2, 2]], [0.5, 0.5], [True, False]
    np.testing.assert_allclose(rank(x, w, b, "TOPSIS").scores, [0, 1, 0.5])
    np.testing.assert_allclose(rank(x, w, b, "SAW / WSM").scores, [0, 1, 0.5])
    np.testing.assert_allclose(rank(x, w, b, "WPM").scores, [1 / 3, 1, np.sqrt(1 / 3)])
    np.testing.assert_allclose(rank(x, w, b, "PROMETHEE II").scores, [-1, 1, 0])
    np.testing.assert_allclose(rank(x, w, b, "VIKOR").scores, [1, 0, 0.5])
    for lam in [0, 0.3, 1]:
        r = rank(x, w, b, "WASPAS", lam=lam)
        expected = lam * np.array([1 / 3, 1, 7 / 12]) + (1 - lam) * np.array(
            [1 / 3, 1, np.sqrt(1 / 3)]
        )
        np.testing.assert_allclose(r.scores, expected)


def test_vikor_published_documentation_example():
    # pymcdm VIKOR public API example (see README references).
    x = [
        [78, 56, 34, 6],
        [4, 45, 3, 97],
        [18, 2, 50, 63],
        [9, 14, 11, 92],
        [85, 9, 100, 29],
    ]
    np.testing.assert_allclose(
        rank(x, [0.25] * 4, [True] * 4, "VIKOR").scores,
        [0.5679, 0.7667, 1, 0.7493, 0],
        atol=0.0001,
    )


def test_edas_published_documentation_example():
    x = [
        [3873, 39.55, 0.27, 0.87, 150, 0.07, 12, 2130],
        [5067, 67.26, 0.23, 0.23, 40, 0.02, 21, 2200],
        [2213, 24.69, 0.08, 0.17, 200, 0.04, 35, 570],
        [6243, 132, 0.07, 0.25, 100, 0.04, 16, 100],
        [8312, 460.47, 0.05, 0.21, 25, 0.1, 25, 200],
    ]
    weights = [0.131, 0.113, 0.126, 0.125, 0.126, 0.129, 0.132, 0.117]
    np.testing.assert_allclose(
        rank(x, weights, [False, False, False, True, True, False, True, True], "EDAS").scores,
        [0.841, 0.632, 0.883, 0.457, 0.104],
        atol=0.0006,
    )


def test_promethee_matches_pairwise_definition_with_ties():
    x = np.array([[1, 4], [1, 2], [3, 2], [5, 7]])
    w, b = np.array([0.3, 0.7]), np.array([True, False])
    oriented = x * np.where(b, 1, -1)
    pref = ((oriented[:, None, :] > oriented[None, :, :]) * w).sum(axis=2)
    expected = (pref.sum(axis=1) - pref.sum(axis=0)) / (len(x) - 1)
    np.testing.assert_allclose(rank(x, w, b, "PROMETHEE II").scores, expected)


@pytest.mark.parametrize("method", ["WPM", "WASPAS", "COPRAS", "EDAS"])
def test_positive_domain_is_enforced(method):
    with pytest.raises(ValueError, match="strictly positive"):
        rank([[0, 2], [3, 1]], [1, 1], [True, False], method)


@pytest.mark.parametrize("bad", [[0, 0], [-1, 2], [np.nan, 1], [1]])
def test_bad_weights_rejected(bad):
    with pytest.raises(ValueError):
        rank([[1, 2], [2, 1]], bad, [True, False])


def test_objective_weighting_and_degenerate_inputs():
    for method in ["Entropy", "CRITIC"]:
        w = objective_weights([[1, 2, 5], [2, 1, 5], [3, 4, 5]], [True] * 3, method)
        assert w[2] == 0
        assert np.isclose(w.sum(), 1)
        with pytest.raises(ValueError, match="constant"):
            objective_weights([[1, 2], [1, 2]], [True, False], method)
    with pytest.raises(ValueError, match="no distinguishing"):
        objective_weights([[1, 2], [2, 4], [3, 6]], [True] * 2, "CRITIC")


def test_ahp_consistency_and_bwm_exact_ratios():
    expected = np.array([0.6, 0.3, 0.1])
    a = expected[:, None] / expected
    weights, cr = ahp(a)
    np.testing.assert_allclose(weights, expected)
    assert cr < 1e-12
    weights, residual = bwm([1, 2, 6], [6, 3, 1], 0, 2)
    np.testing.assert_allclose(weights, expected)
    assert residual < 1e-12
    with pytest.raises(ValueError):
        ahp([[1, 2], [2, 1]])
    with pytest.raises(ValueError):
        bwm([1, 2, 5], [6, 3, 1], 0, 2)
