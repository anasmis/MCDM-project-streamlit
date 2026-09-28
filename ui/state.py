import numpy as np
import pandas as pd
import streamlit as st

from mcdm.io import example_problem
from mcdm.models import Criterion, DecisionProblem


def set_draft(problem: DecisionProblem, is_example: bool = False) -> None:
    revision = st.session_state.get("revision", 0) + 1
    st.session_state.update(
        {
            "step": 0,
            "problem": None,
            "weighting": None,
            "revision": revision,
            "draft_name": problem.name,
            "draft_description": problem.description,
            "draft_criteria": pd.DataFrame(
                {
                    "Critère": problem.criterion_names,
                    "Objectif": [
                        "Maximiser" if c.direction == "max" else "Minimiser"
                        for c in problem.criteria
                    ],
                    "Unité": [c.unit for c in problem.criteria],
                }
            ),
            "draft_matrix": problem.to_frame().rename_axis("Alternative").reset_index(),
            "is_example": is_example,
            "judgments": {},
            "weighting_method": "CRITIC",
            "preference": "objective",
            "ranking_method": "TOPSIS",
        }
    )


def initialize() -> None:
    if "step" not in st.session_state:
        set_draft(example_problem(), True)


def new_problem() -> None:
    set_draft(
        DecisionProblem(
            "Nouvelle étude",
            ("Alternative A", "Alternative B", "Alternative C"),
            (Criterion("Critère 1"), Criterion("Critère 2"), Criterion("Critère 3")),
            np.zeros((3, 3)),
        )
    )
    st.session_state.draft_matrix.iloc[:, 1:] = np.nan


def navigate(step: int) -> None:
    st.session_state.step = step
