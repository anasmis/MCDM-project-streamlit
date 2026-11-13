from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from decisionlab.engine import METHODS
from decisionlab.project import example

APP = Path(__file__).resolve().parents[1] / "app.py"


def choose(app, label, value):
    next(s for s in app.selectbox if s.label == label).select(value)
    return app.run()


def healthy(app):
    assert not app.exception, [e.message for e in app.exception]
    assert not app.error, [e.value for e in app.error]


def test_crisp_all_methods_weighting_and_sensitivity():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    healthy(app)
    app.multiselect[0].set_value(list(METHODS)).run()
    healthy(app)
    assert len(app.success) == len(METHODS)
    for method in ["Manual", "Entropy", "CRITIC", "AHP", "BWM", "Equal"]:
        choose(app, "Weighting method", method)
        healthy(app)
        assert len(app.success) == len(METHODS)
    next(b for b in app.button if b.label == "Run sensitivity analysis").click().run()
    healthy(app)
    assert len(app.get("download_button")) == 4
    app.run()
    assert len(app.get("download_button")) == 4


def test_fuzzy_example_and_weighting():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    choose(app, "Example or new problem", "Fuzzy assessment")
    next(b for b in app.button if b.label == "Load example / start new").click().run()
    healthy(app)
    assert len(app.success) == 2
    for method in ["Fuzzy manual", "Fuzzy AHP", "AHP", "Entropy", "CRITIC", "BWM"]:
        choose(app, "Weighting method", method)
        healthy(app)
        assert len(app.success) == 2
    next(b for b in app.button if b.label == "Run sensitivity analysis").click().run()
    healthy(app)


@pytest.mark.parametrize("name", ["Blank problem", "Location planning"])
def test_other_problem_templates(name):
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    choose(app, "Example or new problem", name)
    next(b for b in app.button if b.label == "Load example / start new").click().run()
    healthy(app)
    assert len(app.success) == 3


def test_invalid_input_is_reported_and_can_be_fixed():
    app = AppTest.from_file(str(APP), default_timeout=30)
    p = example()
    p["values"][0][0] = "unknown"
    app.session_state["project"] = p
    app.session_state["revision"] = 1
    app.session_state["settings_current"] = {}
    app.run()
    assert not app.exception
    assert "numeric performances" in app.error[0].value
    p["values"][0][0] = "120"
    app.session_state["project"] = p
    app.session_state["revision"] = 2
    for k in ["matrix_signature", "matrix_base", "matrix_current"]:
        del app.session_state[k]
    app.run()
    healthy(app)


def test_repository_entry_point():
    app = AppTest.from_file(str(APP.parents[1] / "app.py"), default_timeout=30).run()
    healthy(app)
    assert app.title[0].value == "Decision Lab"


def test_matrix_edits_survive_criterion_rename_and_schema_roundtrip():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.session_state["1_matrix_1"] = {
        "edited_rows": {0: {"Price": "1", "Quality": "10"}},
        "added_rows": [],
        "deleted_rows": [],
    }
    app.run()
    healthy(app)
    assert app.session_state["matrix_current"].iloc[0]["Price"] == "1"
    app.session_state["1_criteria"] = {
        "edited_rows": {0: {"name": "Budget"}},
        "added_rows": [],
        "deleted_rows": [],
    }
    app.run()
    healthy(app)
    assert app.session_state["matrix_current"].iloc[0]["Budget"] == "1"
    app.session_state["1_matrix_2"] = {
        "edited_rows": {0: {"Budget": "2"}},
        "added_rows": [],
        "deleted_rows": [],
    }
    # AppTest has no data_editor element API: supply the complete browser editor
    # state on each run, including unchanged edits in the other editor.
    app.session_state["1_criteria"] = {
        "edited_rows": {0: {"name": "Budget"}},
        "added_rows": [],
        "deleted_rows": [],
    }
    app.run()
    app.session_state["1_criteria"] = {"edited_rows": {}, "added_rows": [], "deleted_rows": []}
    app.run()
    healthy(app)
    assert app.session_state["matrix_current"].iloc[0]["Price"] == "2"


def test_weighting_settings_survive_switching_methods():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    choose(app, "Weighting method", "AHP")
    pair_key = next(k for k in app.session_state.filtered_state if k.startswith("1_AHP"))
    app.session_state[pair_key] = {
        "edited_rows": {0: {"Left / right": "3"}},
        "added_rows": [],
        "deleted_rows": [],
    }
    app.run()
    choose(app, "Weighting method", "Equal")
    healthy(app)
    assert "3" in app.session_state["settings_current"]["pairwise"].values()


def test_linguistic_scale_survives_mode_switch():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.radio[0].set_value("Fuzzy").run()
    app.session_state["1_scale_1"] = {
        "edited_rows": {0: {"Lower": 0.5}},
        "added_rows": [],
        "deleted_rows": [],
    }
    app.run()
    assert app.session_state["scale_current"]["Very low"][0] == 0.5
    app.radio[0].set_value("Crisp").run()
    app.radio[0].set_value("Fuzzy").run()
    healthy(app)
    assert app.session_state["scale_current"]["Very low"][0] == 0.5
