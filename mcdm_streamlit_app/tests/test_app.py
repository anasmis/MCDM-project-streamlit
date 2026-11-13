import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from decisionlab.engine import METHODS
from decisionlab.french import localized_example
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
        choose(app, "Méthode de pondération", method)
        healthy(app)
        assert len(app.success) == len(METHODS)
    next(b for b in app.button if b.label == "Lancer l’analyse de sensibilité").click().run()
    healthy(app)
    assert len(app.get("download_button")) == 4
    app.run()
    assert len(app.get("download_button")) == 4


def test_fuzzy_example_and_weighting():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    choose(app, "Exemple ou nouveau problème", "Fuzzy assessment")
    next(
        b for b in app.button if b.label == "Charger cet exemple / créer un problème"
    ).click().run()
    healthy(app)
    assert len(app.success) == 2
    for method in ["Fuzzy manual", "Fuzzy AHP", "AHP", "Entropy", "CRITIC", "BWM"]:
        choose(app, "Méthode de pondération", method)
        healthy(app)
        assert len(app.success) == 2
    next(b for b in app.button if b.label == "Lancer l’analyse de sensibilité").click().run()
    healthy(app)


@pytest.mark.parametrize("name", ["Blank problem", "Location planning"])
def test_other_problem_templates(name):
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    choose(app, "Exemple ou nouveau problème", name)
    next(
        b for b in app.button if b.label == "Charger cet exemple / créer un problème"
    ).click().run()
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
    assert "performances numériques" in app.error[0].value
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
    assert app.title[0].value == "Atelier de décision"


def test_matrix_edits_survive_criterion_rename_and_schema_roundtrip():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.session_state["1_matrix_1"] = {
        "edited_rows": {0: {"Prix": "1", "Qualité": "10"}},
        "added_rows": [],
        "deleted_rows": [],
    }
    app.run()
    healthy(app)
    assert app.session_state["matrix_current"].iloc[0]["Prix"] == "1"
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
    assert app.session_state["matrix_current"].iloc[0]["Prix"] == "2"


def test_weighting_settings_survive_switching_methods():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    choose(app, "Méthode de pondération", "AHP")
    pair_key = "1_AHP" + json.dumps([c["name"] for c in localized_example()["criteria"]])
    app.session_state[pair_key] = {
        "edited_rows": {0: {"Gauche / droite": "3"}},
        "added_rows": [],
        "deleted_rows": [],
    }
    app.run()
    choose(app, "Méthode de pondération", "Equal")
    healthy(app)
    assert "3" in app.session_state["settings_current"]["pairwise"].values()


def test_linguistic_scale_survives_mode_switch():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.radio[0].set_value("Fuzzy").run()
    app.session_state["1_scale_1"] = {
        "edited_rows": {0: {"Borne inférieure": 0.5}},
        "added_rows": [],
        "deleted_rows": [],
    }
    app.run()
    assert app.session_state["scale_current"]["Très faible"][0] == 0.5
    app.radio[0].set_value("Crisp").run()
    app.radio[0].set_value("Fuzzy").run()
    healthy(app)
    assert app.session_state["scale_current"]["Très faible"][0] == 0.5


def test_french_controls_and_guides():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert app.radio[0].options == ["Valeurs précises", "Logique floue"]
    assert (
        next(s for s in app.selectbox if s.label == "Méthode de pondération").options[0]
        == "Poids égaux"
    )
    choose(app, "Méthode à comprendre", "VIKOR")
    healthy(app)
    assert any("Le plus petit Q" in m.value for m in app.markdown)
    choose(app, "Méthode à comprendre", "Fuzzy TOPSIS")
    healthy(app)
    assert any("Les trois bornes interviennent" in m.value for m in app.markdown)


def test_existing_english_fuzzy_projects_keep_their_values():
    app = AppTest.from_file(str(APP), default_timeout=30)
    old_project = example("Fuzzy assessment")
    app.session_state["project"] = old_project
    app.session_state["revision"] = 1
    app.session_state["settings_current"] = {}
    app.run()
    healthy(app)
    assert len(app.success) == 2
    assert app.session_state["matrix_current"].iloc[0]["Quality"] == "High"
    assert app.session_state["project"] == old_project
