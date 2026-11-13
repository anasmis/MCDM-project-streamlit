import copy

import numpy as np
import pytest

from decisionlab.fuzzy import (
    FUZZY_METHODS,
    fuzzy_ahp,
    fuzzy_rank,
    parse_cell,
    parse_matrix,
)
from decisionlab.project import dump_json, example, load_csv, load_json


def test_fuzzy_parsing_and_errors():
    np.testing.assert_equal(parse_cell("high"), [5, 7, 9])
    np.testing.assert_equal(parse_cell("(2, 3, 5)"), [2, 3, 5])
    np.testing.assert_equal(parse_cell("4"), [4, 4, 4])
    np.testing.assert_equal(parse_cell("custom", {"Custom": [1, 2, 4]}), [1, 2, 4])
    for value in ["unknown", "5;3;2", "1;2", "nan", "inf"]:
        with pytest.raises(ValueError):
            parse_cell(value)


@pytest.mark.parametrize("method", FUZZY_METHODS)
def test_fuzzy_dominance_and_cost_direction(method):
    x = parse_matrix([["7;8;9", "1;2;3"], ["3;4;5", "5;6;7"], ["7;8;9", "1;2;3"]])
    result = fuzzy_rank(x, [1, 1], [True, False], method)
    assert result.ranks.tolist() == [1, 3, 1]
    result = fuzzy_rank(np.ones((2, 2, 3)), [1, 1], [True, False], method)
    assert result.ranks.tolist() == [1, 1]
    with pytest.raises(ValueError, match="positive lower"):
        fuzzy_rank(parse_matrix([["0;1;2"]]), [1], [False], method)


def test_fuzzy_vertex_distance_hand_calculation():
    x = parse_matrix([["1;2;3"], ["3;4;5"]])
    result = fuzzy_rank(x, [1], [True])
    positive = np.sqrt(np.mean((np.array([0.2, 0.4, 0.6]) - 1) ** 2))
    negative = np.sqrt(np.mean((np.array([0.2, 0.4, 0.6]) - 0.2) ** 2))
    assert result.scores[0] == pytest.approx(negative / (positive + negative))
    np.testing.assert_allclose(result.scores.sum(), 1)
    expected = np.array([[0.2, 0.4, 0.6], [0.6, 0.8, 1]])
    np.testing.assert_allclose(
        fuzzy_rank(x, [1], [True], "Fuzzy SAW").details["fuzzy_totals"], expected
    )


def test_uncertainty_and_fuzzy_weights_are_used():
    narrow = parse_matrix([["3;4;5", "1;2;3"], ["4;5;6", "2;3;4"]])
    wide = narrow.copy()
    wide[0, 0] = [0, 4, 8]  # Same centroid, different uncertainty.
    a = fuzzy_rank(narrow, [1, 1], [True, False]).scores
    b = fuzzy_rank(wide, [1, 1], [True, False]).scores
    assert not np.allclose(a, b)
    weighted = fuzzy_rank(narrow, [[0.1, 0.2, 0.4], [0.4, 0.8, 1]], [True, False])
    assert np.isfinite(weighted.scores).all()


def test_fuzzy_ahp_crisp_limit():
    expected = np.array([0.6, 0.3, 0.1])
    a = np.repeat((expected[:, None] / expected)[:, :, None], 3, axis=2)
    weights, triangles = fuzzy_ahp(a)
    np.testing.assert_allclose(weights, expected)
    np.testing.assert_allclose(triangles, np.repeat(expected[:, None], 3, axis=1))


@pytest.mark.parametrize(
    "name",
    ["Supplier selection", "Location planning", "Fuzzy assessment", "Blank problem"],
)
def test_project_roundtrip(name):
    p = example(name)
    p["settings"] = {"weighting": "AHP", "pairwise": {'["a", "b"]': "3"}, "v": 0.7}
    assert load_json(dump_json(p)) == p


def test_csv_delimiters_decimals_and_fuzzy_values():
    p = load_csv(b"Option;Price;Quality\nA;1,5;7\nB;2,5;8\n", sep=";", decimal=",")
    assert p["values"] == [["1.5", "7"], ["2.5", "8"]]
    p = load_csv(b'Alternative,Quality,Cost\nA,"1;2;3",5\nB,High,6\n')
    assert p["values"][0][0] == "1;2;3"
    assert all(c["direction"] == "Benefit" for c in p["criteria"])


def test_malformed_files_rejected():
    for content in [b"", b"A\nB\n", b"Alternative,X,X\nA,1,2\n"]:
        with pytest.raises(ValueError):
            load_csv(content)
    for mutate in [
        lambda p: p.update(alternatives=["A", "A", "B", "C"]),
        lambda p: p.update(values=[[1]]),
        lambda p: p["settings"].update(v=2),
        lambda p: p["criteria"][0].update(direction="bad"),
    ]:
        p = copy.deepcopy(example())
        mutate(p)
        with pytest.raises(ValueError):
            dump_json(p)
