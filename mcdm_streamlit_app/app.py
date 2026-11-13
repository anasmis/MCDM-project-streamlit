"""Streamlit workspace for domain-independent crisp and fuzzy MCDM problems."""

import copy
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from decisionlab.engine import (
    METHODS,
    ahp,
    bwm,
    objective_weights,
    rank,
    weights_vector,
)
from decisionlab.fuzzy import (
    DEFAULT_SCALE,
    FUZZY_METHODS,
    fuzzy_ahp,
    fuzzy_rank,
    fuzzy_weights,
    parse_cell,
    parse_matrix,
)
from decisionlab.project import (
    dump_json,
    example,
    load_csv,
    load_json,
    validate_project,
)

st.set_page_config(page_title="Decision Lab · MCDM", layout="wide")
stylesheet = Path(__file__).with_name("style.css").read_text(encoding="utf-8")
st.markdown(f"<style>{stylesheet}</style>", unsafe_allow_html=True)
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [
    "#1d4ed8",
    "#047857",
    "#a21caf",
    "#b45309",
    "#0e7490",
    "#b91c1c",
    "#6d28d9",
    "#475569",
]


def display_chart(figure):
    """Keep chart labels readable independently of the browser's theme."""
    figure.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"color": "#172b4d", "size": 14},
        margin={"l": 24, "r": 24, "t": 40, "b": 24},
    )
    st.plotly_chart(figure, theme=None, width="stretch")


def install_project(project):
    st.session_state.project = project
    st.session_state.settings_current = copy.deepcopy(project.get("settings", {}))
    st.session_state.scale_current = copy.deepcopy(project.get("scale", DEFAULT_SCALE))
    st.session_state.scale_visible = False
    st.session_state.revision = st.session_state.get("revision", 0) + 1
    for name in [
        "matrix_signature",
        "matrix_base",
        "matrix_current",
        "scale_signature",
    ]:
        st.session_state.pop(name, None)


if "project" not in st.session_state:
    install_project(example())
base = st.session_state.project
rev = st.session_state.revision


def key(name):
    return f"{rev}_{name}"


with st.sidebar:
    st.title("Decision Lab")
    st.caption("A workspace for multicriteria decisions")
    preset = st.selectbox(
        "Example or new problem",
        [
            "Supplier selection",
            "Location planning",
            "Fuzzy assessment",
            "Blank problem",
        ],
    )
    if st.button("Load example / start new", width="stretch"):
        install_project(example(preset))
        st.rerun()
    st.divider()
    st.subheader("Import a problem")
    upload = st.file_uploader("CSV matrix or project JSON", type=["csv", "json"], key=key("upload"))
    delimiter = st.selectbox("CSV delimiter", [",", ";", "Tab"], key=key("delimiter"))
    decimal = st.selectbox("CSV decimal separator", [".", ","], key=key("decimal"))
    if st.button("Import file", disabled=upload is None, width="stretch"):
        try:
            content = upload.getvalue()
            project = (
                load_json(content)
                if upload.name.lower().endswith(".json")
                else load_csv(
                    content,
                    sep="\t" if delimiter == "Tab" else delimiter,
                    decimal=decimal,
                )
            )
            install_project(project)
            st.rerun()
        except (ValueError, TypeError, KeyError) as exc:
            st.error(str(exc))
    st.caption(
        "CSV: first column = alternative names, remaining columns = criteria. Imported directions default to Benefit; review each one."
    )
    st.divider()
    mode = st.radio(
        "Decision model",
        ["Crisp", "Fuzzy"],
        index=0 if base.get("mode", "Crisp") == "Crisp" else 1,
        key=key("mode"),
    )
    st.caption(
        "Crisp: numeric performances. Fuzzy: numbers, linguistic terms, or triangular values such as 2;3;5."
    )

st.title("Decision Lab")
st.write(
    "Define your alternatives and criteria, choose your methods, and examine how the ranking changes."
)
title = st.text_input(
    "Problem title", value=base.get("title", "Decision problem"), key=key("title")
)
description = st.text_area(
    "Decision context (optional)",
    value=base.get("description", ""),
    height=70,
    key=key("description"),
)
data_tab, weight_tab, ranking_tab, compare_tab, sensitivity_tab, guide_tab = st.tabs(
    [
        "1 · Problem",
        "2 · Weights",
        "3 · Rankings",
        "4 · Compare",
        "5 · Sensitivity",
        "Method guide",
    ]
)

with data_tab:
    st.subheader("Criteria")
    st.caption(
        "Add or remove rows. Benefit means higher is better; Cost means lower is better. Units are descriptive. Manual weights are normalized automatically."
    )
    criteria_frame = st.data_editor(
        pd.DataFrame(base["criteria"])[["name", "direction", "weight", "unit"]],
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        key=key("criteria"),
        column_config={
            "name": st.column_config.TextColumn("Criterion", required=True),
            "direction": st.column_config.SelectboxColumn(
                "Direction", options=["Benefit", "Cost"], required=True
            ),
            "weight": st.column_config.NumberColumn("Manual weight", min_value=0.0, required=True),
            "unit": st.column_config.TextColumn("Unit"),
        },
    )
    try:
        criteria = criteria_frame.fillna({"unit": ""}).to_dict("records")
        trial = copy.deepcopy(base)
        trial.update(
            criteria=criteria,
            alternatives=["validation"],
            values=[["0"] * len(criteria)],
        )
        validate_project(trial)
    except (ValueError, TypeError) as exc:
        st.error(str(exc))
        st.stop()
    names = [c["name"].strip() for c in criteria]
    for c, name in zip(criteria, names, strict=False):
        c["name"] = name
    benefits = np.array([c["direction"] == "Benefit" for c in criteria])
    signature = tuple(names)
    if st.session_state.get("matrix_signature") != signature:
        previous = st.session_state.get("matrix_current")
        old_names = list(
            st.session_state.get("matrix_signature", [c["name"] for c in base["criteria"]])
        )
        if previous is None:
            previous = pd.DataFrame(base["values"], columns=old_names)
            previous.insert(0, "Alternative", base["alternatives"])
        new = pd.DataFrame({"Alternative": previous["Alternative"]})
        for j, name in enumerate(names):
            source = (
                name
                if name in previous
                else old_names[j]
                if j < len(old_names) and old_names[j] not in names
                else None
            )
            new[name] = previous[source] if source else "0"
        st.session_state.matrix_base = new.astype(str)
        st.session_state.matrix_signature = signature
        st.session_state.matrix_epoch = st.session_state.get("matrix_epoch", 0) + 1
    st.subheader("Decision matrix")
    st.caption(
        "Add or remove alternatives using the table. Rename alternatives in the first column. Each alternative and criterion must have a unique name."
    )
    matrix_frame = st.data_editor(
        st.session_state.matrix_base,
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        key=key("matrix_" + str(st.session_state.matrix_epoch)),
        column_config={
            c: st.column_config.TextColumn(c, required=True) for c in ["Alternative"] + names
        },
    )
    st.session_state.matrix_current = matrix_frame.copy()
    scale = copy.deepcopy(st.session_state.get("scale_current", base.get("scale", DEFAULT_SCALE)))
    if mode == "Fuzzy":
        if not st.session_state.get("scale_visible", False):
            st.session_state.scale_editor_base = copy.deepcopy(scale)
            st.session_state.scale_epoch = st.session_state.get("scale_epoch", 0) + 1
            st.session_state.scale_visible = True
        with st.expander("Linguistic scale · edit your own terms", expanded=True):
            st.caption(
                "Terms describe the magnitude of a performance, not its desirability. A High cost is still a cost. Use lower ≤ modal ≤ upper; cost triangles must have a positive lower endpoint."
            )
            initial_scale = pd.DataFrame(
                [
                    {"Term": term, "Lower": t[0], "Modal": t[1], "Upper": t[2]}
                    for term, t in st.session_state.scale_editor_base.items()
                ]
            )
            initial_scale = initial_scale.astype({"Lower": float, "Modal": float, "Upper": float})
            scale_frame = st.data_editor(
                initial_scale,
                num_rows="dynamic",
                hide_index=True,
                width="stretch",
                key=key("scale_" + str(st.session_state.scale_epoch)),
            )
            try:
                terms = scale_frame["Term"].astype(str).str.strip()
                if (
                    terms.eq("").any()
                    or terms.str.casefold().duplicated().any()
                    or scale_frame.isna().any().any()
                ):
                    raise ValueError(
                        "Linguistic terms must be nonempty, unique and have three endpoints."
                    )
                scale = {
                    row["Term"].strip(): [float(row[k]) for k in ["Lower", "Modal", "Upper"]]
                    for row in scale_frame.to_dict("records")
                }
                for t in scale.values():
                    parse_cell(t, {})
                st.session_state.scale_current = copy.deepcopy(scale)
            except (ValueError, TypeError) as exc:
                st.error(str(exc))
                st.stop()
    else:
        st.session_state.scale_visible = False
    try:
        if matrix_frame.isna().any().any():
            raise ValueError("Complete every matrix cell before solving.")
        problem = {
            "version": 1,
            "title": title,
            "description": description,
            "mode": mode,
            "alternatives": matrix_frame["Alternative"].astype(str).str.strip().tolist(),
            "criteria": criteria,
            "values": matrix_frame[names].astype(str).values.tolist(),
            "scale": scale,
            "settings": st.session_state.settings_current,
        }
        validate_project(problem)
        if mode == "Crisp":
            try:
                matrix = np.asarray(problem["values"], dtype=float)
            except ValueError:
                raise ValueError(
                    "Crisp mode requires numeric performances. Switch to Fuzzy for linguistic terms or triangular inputs."
                ) from None
            if not np.isfinite(matrix).all():
                raise ValueError(
                    "All performances must be finite; NaN and infinity are not accepted."
                )
            centroids = matrix
        else:
            matrix = parse_matrix(problem["values"], scale)
            centroids = (matrix / 3).sum(axis=2)
    except (ValueError, TypeError) as exc:
        st.error(str(exc))
        st.stop()
    labels = problem["alternatives"]
    constants = [names[j] for j in range(len(names)) if np.ptp(centroids[:, j]) == 0]
    if len(labels) == 1:
        st.info("There is only one alternative. Add more to compare choices.")
    if constants:
        st.info(
            "Constant centroids: "
            + ", ".join(constants)
            + ". These carry no discrimination for objective weights."
        )
    cols = st.columns(3)
    cols[0].metric("Alternatives", len(labels))
    cols[1].metric("Criteria", len(names))
    cols[2].metric("Model", mode)

settings = problem["settings"]
with weight_tab:
    st.subheader("How important is each criterion?")
    options = ["Equal", "Manual", "Entropy", "CRITIC", "AHP", "BWM"]
    if mode == "Fuzzy":
        options += ["Fuzzy manual", "Fuzzy AHP"]
    stored_weighting = settings.get("weighting", "Equal")
    weighting = st.selectbox(
        "Weighting method",
        options,
        index=options.index(stored_weighting) if stored_weighting in options else 0,
        key=key("weighting_" + mode),
    )
    settings["weighting"] = weighting
    fw = None
    try:
        if weighting in ["Equal", "Entropy", "CRITIC"]:
            weights = objective_weights(centroids, benefits, weighting)
            if mode == "Fuzzy" and weighting != "Equal":
                st.caption(
                    "Objective weights use triangle centroids. The fuzzy ranking still uses all three endpoints."
                )
        elif weighting == "Manual":
            weights = weights_vector([c["weight"] for c in criteria], len(names))
        elif weighting == "Fuzzy manual":
            prior = settings.get("fuzzy_weights", {})
            table = pd.DataFrame(
                {
                    "Criterion": names,
                    "Triangle": [str(prior.get(n, "1;1;1")) for n in names],
                }
            )
            edited = st.data_editor(
                table,
                disabled=["Criterion"],
                hide_index=True,
                key=key("fw_" + json.dumps(names)),
                width="stretch",
            )
            settings["fuzzy_weights"] = dict(
                zip(names, edited["Triangle"].astype(str), strict=False)
            )
            fw = fuzzy_weights([parse_cell(t, scale) for t in edited["Triangle"]], len(names))
            weights = weights_vector(fw.mean(axis=1), len(names))
        elif weighting in ["AHP", "Fuzzy AHP"]:
            fuzzy_pairwise = weighting == "Fuzzy AHP"
            st.caption(
                "Enter upper-triangle judgments; reciprocals are generated automatically. Use 1 for equal importance, 3/5/7/9 for increasing preference, or a decimal reciprocal for less importance."
            )
            pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]
            saved = settings.get("fuzzy_pairwise" if fuzzy_pairwise else "pairwise", {})
            if pairs:
                table = pd.DataFrame(
                    {
                        "Left criterion": [names[i] for i, j in pairs],
                        "Right criterion": [names[j] for i, j in pairs],
                        "Left / right": [
                            str(
                                saved.get(
                                    json.dumps([names[i], names[j]]),
                                    "1;1;1" if fuzzy_pairwise else "1",
                                )
                            )
                            for i, j in pairs
                        ],
                    }
                )
                edited = st.data_editor(
                    table,
                    disabled=["Left criterion", "Right criterion"],
                    hide_index=True,
                    width="stretch",
                    key=key(weighting + json.dumps(names)),
                )
                judgments = edited["Left / right"].tolist()
            else:
                judgments = []
            settings["fuzzy_pairwise" if fuzzy_pairwise else "pairwise"] = {
                json.dumps([names[i], names[j]]): str(t)
                for (i, j), t in zip(pairs, judgments, strict=False)
            }
            a = (
                np.ones((len(names), len(names), 3))
                if fuzzy_pairwise
                else np.ones((len(names), len(names)))
            )
            for (i, j), value in zip(pairs, judgments, strict=False):
                t = parse_cell(value, {}) if fuzzy_pairwise else float(value)
                if np.any(np.asarray(t) <= 0):
                    raise ValueError("Pairwise judgments must be positive.")
                a[i, j] = t
                a[j, i] = 1 / t[::-1] if fuzzy_pairwise else 1 / t
            if fuzzy_pairwise:
                weights, fw = fuzzy_ahp(a)
                _, cr = ahp(a[:, :, 1])
                st.caption(
                    "Fuzzy AHP uses Buckley geometric means and triangular inverse-sum normalization. Consistency below uses the modal comparison matrix."
                )
            else:
                weights, cr = ahp(a)
            if cr is None:
                st.info(
                    "Consistency ratio unavailable above 10 criteria; no random-index approximation is used."
                )
            else:
                st.metric(
                    "Modal consistency ratio" if fuzzy_pairwise else "Consistency ratio",
                    f"{cr:.3f}",
                )
                if cr > 0.10:
                    st.warning(
                        "Comparisons have CR > 0.10. Revise judgments before relying on the ranking."
                    )
        else:
            if len(names) < 2:
                raise ValueError(
                    "BWM needs at least two criteria. Select another weighting method."
                )
            saved = settings.get("bwm", {})
            best_name = st.selectbox(
                "Most important criterion",
                names,
                index=names.index(saved["best"]) if saved.get("best") in names else 0,
                key=key("best_" + json.dumps(names)),
            )
            worst_options = [n for n in names if n != best_name]
            worst_name = st.selectbox(
                "Least important criterion",
                worst_options,
                index=worst_options.index(saved["worst"])
                if saved.get("worst") in worst_options
                else len(worst_options) - 1,
                key=key("worst_" + best_name + json.dumps(names)),
            )
            best, worst = names.index(best_name), names.index(worst_name)
            shared = st.number_input(
                "Best / worst preference",
                1.0,
                9.0,
                float(saved.get("shared", 3)),
                key=key("bw_shared"),
            )
            st.caption(
                "Comparisons use 1–9. Self-comparisons and the shared best/worst judgment are set automatically."
            )
            a = saved.get("a", [1.0] * len(names))
            c = saved.get("c", [1.0] * len(names))
            if len(a) != len(names) or len(c) != len(names):
                a, c = [1.0] * len(names), [1.0] * len(names)
            table = pd.DataFrame(
                {"Criterion": names, "Best / criterion": a, "Criterion / worst": c}
            )
            edited = st.data_editor(
                table,
                disabled=["Criterion"],
                hide_index=True,
                width="stretch",
                key=key("bwm_" + json.dumps(names)),
                column_config={
                    n: st.column_config.NumberColumn(min_value=1.0, max_value=9.0)
                    for n in ["Best / criterion", "Criterion / worst"]
                },
            )
            a, c = (
                edited["Best / criterion"].to_numpy(float).copy(),
                edited["Criterion / worst"].to_numpy(float).copy(),
            )
            a[best], c[worst], a[worst], c[best] = 1, 1, shared, shared
            weights, residual = bwm(a, c, best, worst)
            settings["bwm"] = {
                "best": best_name,
                "worst": worst_name,
                "shared": shared,
                "a": a.tolist(),
                "c": c.tolist(),
            }
            st.metric("BWM optimal linear residual", f"{residual:.4f}")
            st.caption("This optimization residual is not an AHP consistency ratio.")
        weight_frame = pd.DataFrame({"Criterion": names, "Weight": weights})
        if fw is not None:
            for j, endpoint in enumerate(["Lower", "Modal", "Upper"]):
                weight_frame[endpoint] = fw[:, j]
        st.dataframe(weight_frame, hide_index=True, width="stretch")
        display_chart(
            px.bar(
                weight_frame,
                x="Criterion",
                y="Weight",
                color_discrete_sequence=["#1d4ed8"],
            ),
        )
    except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
        st.error(str(exc))
        st.stop()

registry = METHODS if mode == "Crisp" else FUZZY_METHODS
default_methods = (
    ["TOPSIS", "SAW / WSM", "VIKOR"] if mode == "Crisp" else ["Fuzzy TOPSIS", "Fuzzy SAW"]
)
with ranking_tab:
    st.subheader("Choose ranking methods")
    restored = settings.get("methods_" + mode, default_methods)
    selected = st.multiselect(
        "Ranking methods",
        list(registry),
        default=[m for m in restored if m in registry],
        key=key("methods_" + mode),
    )
    settings["methods_" + mode] = selected
    v, lam = float(settings.get("v", 0.5)), float(settings.get("lambda", 0.5))
    if "VIKOR" in selected:
        v = st.slider("VIKOR v · emphasis on group utility", 0.0, 1.0, v, 0.05, key=key("v"))
    if "WASPAS" in selected:
        lam = st.slider(
            "WASPAS lambda · weighted-sum share", 0.0, 1.0, lam, 0.05, key=key("lambda")
        )
    settings.update(v=v, **{"lambda": lam})
    st.caption(
        "Ranks are comparable across methods; raw scores have different meanings and scales. Ties share a rank (12-decimal tolerance)."
    )


def solve(method, w=None, fuzzy_w=None):
    if mode == "Fuzzy":
        effective = (
            fuzzy_w
            if fuzzy_w is not None
            else fw
            if fw is not None and w is None
            else weights
            if w is None
            else w
        )
        return fuzzy_rank(matrix, effective, benefits, method)
    return rank(matrix, weights if w is None else w, benefits, method, v=v, lam=lam)


results, failures, tables = {}, {}, {}
with ranking_tab:
    for method in selected:
        try:
            result = solve(method)
            results[method] = result
            table = pd.DataFrame(
                {"Alternative": labels, "Score": result.scores, "Rank": result.ranks}
            ).sort_values(["Rank", "Alternative"], kind="stable")
            tables[method] = table
            st.markdown(f"### {method}")
            st.caption(registry[method])
            winners = [labels[i] for i in range(len(labels)) if result.ranks[i] == 1]
            st.success("Top ranked: " + ", ".join(winners))
            if method == "VIKOR" and not (
                result.details["acceptable_advantage"] and result.details["acceptable_stability"]
            ):
                st.warning(
                    "VIKOR compromise conditions are not both satisfied. Treat the leading alternatives as a compromise set; Q rank alone does not establish a unique recommendation."
                )
            st.dataframe(table, hide_index=True, width="stretch")
            with st.expander("Calculation details · " + method):
                for name, values in result.details.items():
                    if isinstance(values, np.ndarray):
                        st.caption(name.replace("_", " ").title())
                        if values.ndim == 3:
                            detail = pd.DataFrame(
                                values.reshape(len(labels), -1),
                                columns=[
                                    f"{c} · {e}"
                                    for c, active in zip(names, weights > 0, strict=False)
                                    if active
                                    for e in ["l", "m", "u"]
                                ],
                            )
                        elif values.ndim == 2:
                            columns = [
                                c for c, active in zip(names, weights > 0, strict=False) if active
                            ]
                            if name == "fuzzy_totals" or values.shape[1] != len(columns):
                                columns = (
                                    ["Lower", "Modal", "Upper"]
                                    if values.shape[1] == 3
                                    else [str(i + 1) for i in range(values.shape[1])]
                                )
                            detail = pd.DataFrame(values, columns=columns)
                        else:
                            detail = pd.DataFrame({name: values})
                        detail.insert(0, "Alternative", labels)
                        st.dataframe(detail, hide_index=True, width="stretch")
                    else:
                        st.write(name.replace("_", " ").title(), values)
        except (ValueError, FloatingPointError) as exc:
            failures[method] = str(exc)
            st.warning(f"{method}: {exc}")
    if not selected:
        st.info("Select at least one ranking method.")

with compare_tab:
    st.subheader("Compare rankings")
    if results:
        ranks = pd.DataFrame({m: r.ranks for m, r in results.items()}, index=labels)
        ranks.index.name = "Alternative"
        st.dataframe(ranks.reset_index(), hide_index=True, width="stretch")
        long = ranks.reset_index().melt(id_vars="Alternative", var_name="Method", value_name="Rank")
        fig = px.line(long, x="Method", y="Rank", color="Alternative", markers=True)
        fig.update_yaxes(autorange="reversed", dtick=1)
        display_chart(fig)
        if len(results) > 1:
            st.caption(
                "Spearman rank correlation. A blank entry means correlation is undefined because all ranks are tied."
            )
            st.dataframe(ranks.corr(method="spearman"), width="stretch")
        st.caption(
            "Mean rank summarizes agreement across the selected successful methods. It is not an additional MCDM method."
        )
        consensus = pd.DataFrame(
            {"Alternative": labels, "Mean rank": ranks.mean(axis=1).values}
        ).sort_values("Mean rank", kind="stable")
        st.dataframe(consensus, hide_index=True, width="stretch")
    else:
        st.info("Choose a compatible method to compare rankings.")

with sensitivity_tab:
    st.subheader("How stable is the decision?")
    st.caption(
        "Vary one criterion's weight from 0 to 1; redistribute the remainder proportionally over the others. This examines weight sensitivity, not uncertainty in the performances."
    )
    if results and len(names) > 1:
        sensitivity_method = st.selectbox(
            "Sensitivity method", list(results), key=key("sensitivity_method_" + mode)
        )
        changed_name = st.selectbox(
            "Criterion to vary",
            names,
            key=key("sensitivity_criterion_" + json.dumps(names)),
        )
        changed = names.index(changed_name)
        sensitivity_signature = (
            json.dumps(problem, sort_keys=True),
            sensitivity_method,
            changed_name,
        )
        if st.button("Run sensitivity analysis", key=key("sensitivity_run")):
            rows, skipped = [], []
            for share in np.linspace(0, 1, 21):
                w = weights.copy()
                others = np.arange(len(w)) != changed
                total = w[others].sum()
                w[others] = w[others] / total * (1 - share) if total else (1 - share) / others.sum()
                w[changed] = share
                fuzzy_w = None
                if mode == "Fuzzy" and fw is not None:
                    fuzzy_w = np.repeat(w[:, None], 3, axis=1)
                    nonzero = weights > 0
                    fuzzy_w[nonzero] = fw[nonzero] * (w[nonzero] / weights[nonzero])[:, None]
                try:
                    r = solve(sensitivity_method, w, fuzzy_w)
                    rows += [
                        {"Weight": share, "Alternative": label, "Rank": int(r.ranks[i])}
                        for i, label in enumerate(labels)
                    ]
                except ValueError:
                    skipped.append(share)
            st.session_state.sensitivity_result = (sensitivity_signature, rows, skipped)
        stored_sensitivity = st.session_state.get("sensitivity_result")
        if stored_sensitivity and stored_sensitivity[0] == sensitivity_signature:
            _, rows, skipped = stored_sensitivity
            if rows:
                sensitivity = pd.DataFrame(rows)
                chart = px.line(
                    sensitivity, x="Weight", y="Rank", color="Alternative", markers=True
                )
                chart.update_yaxes(autorange="reversed", dtick=1)
                display_chart(chart)
                st.download_button(
                    "Download sensitivity CSV",
                    sensitivity.to_csv(index=False).encode(),
                    "sensitivity.csv",
                    "text/csv",
                    on_click="ignore",
                )
            if skipped:
                st.warning(f"{len(skipped)} weight scenarios were outside this method's domain.")
    else:
        st.info("Sensitivity needs at least two criteria and one successful ranking method.")

with guide_tab:
    st.subheader("Choose a method that fits your decision")
    st.write(
        "This workspace handles finite sets of alternatives evaluated against independent criteria. Encode a target criterion as a distance from your target and minimize it. Constraints, hierarchies with dependent criteria, continuous optimization, and other fuzzy set types require a different model."
    )
    for method, explanation in {**METHODS, **FUZZY_METHODS}.items():
        st.markdown(f"**{method}** — {explanation}")
    st.markdown("""
    **Weighting:** Equal and manual weights express direct preferences. Entropy measures dispersion;
    CRITIC combines contrast and correlation, using benefit-oriented min–max values. Constant criteria
    receive zero objective weight. AHP uses reciprocal pairwise judgments and a consistency check.
    BWM uses the linear best–worst optimization model. Fuzzy AHP uses triangular geometric means.

    **Fuzzy inputs:** A numeric value becomes `(x,x,x)`. A triangle `(l,m,u)` expresses a plausible range
    and a modal value; it is not a probability distribution. Customize the linguistic scale in Problem.
    Fuzzy TOPSIS uses observed endpoint ideals and vertex distances, summed across criteria; its crisp
    limit uses summed criterion distances and can differ from Euclidean crisp TOPSIS.

    **Data assumptions:** TOPSIS, SAW, VIKOR, MOORA, and usual PROMETHEE II accept signed values.
    WPM, WASPAS, COPRAS and EDAS require positive active values. Fuzzy methods require nonnegative
    triangles and strictly positive cost lower endpoints. Invalid inputs are reported, never imputed.
    Scores are relative to the current alternatives and may change when alternatives are added.
    """)
    st.markdown(
        "References: [MCDM method definitions and examples](https://pymcdm.readthedocs.io/en/master/pymcdm.methods.html), [triangular fuzzy numbers and vertex distance](https://www.mdpi.com/2071-1050/13/6/3020)."
    )

st.divider()
st.subheader("Save your work")
col1, col2, col3 = st.columns(3)
with col1:
    st.download_button(
        "Download project JSON",
        dump_json(problem),
        "decision_project.json",
        "application/json",
        width="stretch",
    )
with col2:
    st.download_button(
        "Download input CSV",
        matrix_frame.to_csv(index=False).encode("utf-8"),
        "decision_matrix.csv",
        "text/csv",
        width="stretch",
    )
with col3:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("project.json", dump_json(problem))
        z.writestr("weights.csv", weight_frame.to_csv(index=False))
        for i, (method, table) in enumerate(tables.items(), 1):
            z.writestr(
                f"ranking_{i}_{method.replace(' / ', '_').replace(' ', '_')}.csv",
                table.to_csv(index=False),
            )
        z.writestr(
            "report.json",
            json.dumps(
                {
                    "methods": list(results),
                    "unavailable": failures,
                    "weighting": weighting,
                    "mode": mode,
                    "parameters": {"v": v, "lambda": lam},
                },
                indent=2,
            ),
        )
    st.download_button(
        "Download results ZIP",
        archive.getvalue(),
        "decision_results.zip",
        "application/zip",
        width="stretch",
    )
st.caption(
    "Project JSON preserves inputs, directions, units, linguistic scales, pairwise judgments and method settings. Downloads contain the current edits."
)
