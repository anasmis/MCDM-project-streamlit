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
from decisionlab.french import (
    RANKING_GUIDES,
    WEIGHT_GUIDES,
    detail_label,
    error_message,
    label,
    localized_example,
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
    load_csv,
    load_json,
    validate_project,
)

st.set_page_config(page_title="Atelier de décision · MCDM", layout="wide")
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


def explain_method(method):
    guide = RANKING_GUIDES[method]
    st.write(guide["summary"])
    st.markdown("**Étapes du calcul**")
    st.markdown("\n".join(f"{i}. {step}" for i, step in enumerate(guide["steps"], 1)))
    st.latex(guide["formula"])
    st.markdown("**Comment lire le résultat ?**")
    st.write(guide["interpretation"])
    st.caption("Hypothèses et limites : " + guide["limits"])


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
    install_project(localized_example())
base = st.session_state.project
rev = st.session_state.revision


def key(name):
    return f"{rev}_{name}"


with st.sidebar:
    st.title("Atelier de décision")
    st.caption("Un espace d’aide à la décision multicritère")
    preset = st.selectbox(
        "Exemple ou nouveau problème",
        [
            "Supplier selection",
            "Location planning",
            "Fuzzy assessment",
            "Blank problem",
        ],
        format_func=label,
    )
    if st.button("Charger cet exemple / créer un problème", width="stretch"):
        install_project(localized_example(preset))
        st.rerun()
    st.divider()
    st.subheader("Importer un problème")
    upload = st.file_uploader("Matrice CSV ou projet JSON", type=["csv", "json"], key=key("upload"))
    delimiter = st.selectbox(
        "Séparateur du CSV",
        [",", ";", "Tab"],
        format_func=lambda v: "Tabulation" if v == "Tab" else v,
        key=key("delimiter"),
    )
    decimal = st.selectbox("Séparateur décimal du CSV", [".", ","], key=key("decimal"))
    if st.button("Importer le fichier", disabled=upload is None, width="stretch"):
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
            if not upload.name.lower().endswith(".json"):
                project["title"] = "Problème importé"
            install_project(project)
            st.rerun()
        except (ValueError, TypeError, KeyError) as exc:
            st.error(error_message(exc))
    st.caption(
        "CSV : la première colonne contient les alternatives, les suivantes les critères. Le sens initial est Maximiser ; vérifiez chaque critère après l’import."
    )
    st.divider()
    mode = st.radio(
        "Modèle de décision",
        ["Crisp", "Fuzzy"],
        index=0 if base.get("mode", "Crisp") == "Crisp" else 1,
        key=key("mode"),
        format_func=label,
    )
    st.caption(
        "Valeurs précises : performances numériques. Logique floue : nombres, termes linguistiques ou triangles tels que 2;3;5."
    )

st.title("Atelier de décision")
st.write(
    "Définissez vos alternatives et vos critères, choisissez vos méthodes et étudiez l’évolution du classement."
)
title = st.text_input(
    "Titre du problème", value=base.get("title", "Problème de décision"), key=key("title")
)
description = st.text_area(
    "Contexte de la décision (facultatif)",
    value=base.get("description", ""),
    height=70,
    key=key("description"),
)
data_tab, weight_tab, ranking_tab, compare_tab, sensitivity_tab, guide_tab = st.tabs(
    [
        "1 · Problème",
        "2 · Pondération",
        "3 · Classements",
        "4 · Comparaison",
        "5 · Sensibilité",
        "Comprendre les méthodes",
    ]
)

with data_tab:
    st.subheader("Critères")
    st.caption(
        "Ajoutez ou supprimez des critères. Maximiser privilégie une valeur élevée ; Minimiser privilégie une valeur faible. Les unités sont descriptives. Les poids manuels sont normalisés automatiquement."
    )
    criteria_frame = st.data_editor(
        pd.DataFrame(base["criteria"])[["name", "direction", "weight", "unit"]],
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        key=key("criteria"),
        column_config={
            "name": st.column_config.TextColumn("Critère", required=True),
            "direction": st.column_config.SelectboxColumn(
                "Sens", options=["Benefit", "Cost"], required=True, format_func=label
            ),
            "weight": st.column_config.NumberColumn("Poids manuel", min_value=0.0, required=True),
            "unit": st.column_config.TextColumn("Unité"),
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
        st.error(error_message(exc))
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
    st.subheader("Matrice de décision")
    st.caption(
        "Ajoutez ou supprimez des alternatives dans le tableau. Modifiez leur nom dans la première colonne. Chaque alternative et chaque critère doivent porter un nom unique."
    )
    st.caption(
        "Dans les cellules, utilisez un point décimal, par exemple 2.5. Pour un triangle flou, utilisez des points-virgules : 2;3;5."
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
        with st.expander("Échelle linguistique · personnalisez vos termes", expanded=True):
            st.caption(
                "Les termes décrivent la grandeur d’une performance. Un coût « Élevé » reste défavorable si le critère est à minimiser. Respectez : borne inférieure ≤ valeur modale ≤ borne supérieure. Les coûts exigent une borne inférieure strictement positive."
            )
            initial_scale = pd.DataFrame(
                [
                    {
                        "Terme": term,
                        "Borne inférieure": t[0],
                        "Valeur modale": t[1],
                        "Borne supérieure": t[2],
                    }
                    for term, t in st.session_state.scale_editor_base.items()
                ]
            )
            initial_scale = initial_scale.astype(
                {"Borne inférieure": float, "Valeur modale": float, "Borne supérieure": float}
            )
            scale_frame = st.data_editor(
                initial_scale,
                num_rows="dynamic",
                hide_index=True,
                width="stretch",
                key=key("scale_" + str(st.session_state.scale_epoch)),
            )
            try:
                terms = scale_frame["Terme"].astype(str).str.strip()
                if (
                    terms.eq("").any()
                    or terms.str.casefold().duplicated().any()
                    or scale_frame.isna().any().any()
                ):
                    raise ValueError(
                        "Les termes linguistiques doivent être renseignés, uniques et associés à trois bornes."
                    )
                scale = {
                    row["Terme"].strip(): [
                        float(row[k])
                        for k in ["Borne inférieure", "Valeur modale", "Borne supérieure"]
                    ]
                    for row in scale_frame.to_dict("records")
                }
                for t in scale.values():
                    parse_cell(t, {})
                st.session_state.scale_current = copy.deepcopy(scale)
            except (ValueError, TypeError) as exc:
                st.error(error_message(exc))
                st.stop()
    else:
        st.session_state.scale_visible = False
    try:
        if matrix_frame.isna().any().any():
            raise ValueError("Complétez toutes les cellules de la matrice avant le calcul.")
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
                    "Le modèle à valeurs précises exige des performances numériques. Passez en Logique floue pour utiliser des termes linguistiques ou des triangles."
                ) from None
            if not np.isfinite(matrix).all():
                raise ValueError(
                    "Toutes les performances doivent être finies ; NaN et l’infini ne sont pas acceptés."
                )
            centroids = matrix
        else:
            matrix = parse_matrix(problem["values"], scale)
            centroids = (matrix / 3).sum(axis=2)
    except (ValueError, TypeError) as exc:
        st.error(error_message(exc))
        st.stop()
    labels = problem["alternatives"]
    constants = [names[j] for j in range(len(names)) if np.ptp(centroids[:, j]) == 0]
    if len(labels) == 1:
        st.info("Il n’y a qu’une alternative. Ajoutez-en d’autres pour comparer les choix.")
    if constants:
        st.info(
            "Centres de gravité constants : "
            + ", ".join(constants)
            + ". Ils ne permettent pas de différencier les alternatives pour les pondérations objectives."
        )
    cols = st.columns(3)
    cols[0].metric("Alternatives", len(labels))
    cols[1].metric("Critères", len(names))
    cols[2].metric("Modèle", label(mode))

settings = problem["settings"]
with weight_tab:
    st.subheader("Quelle importance accorder à chaque critère ?")
    options = ["Equal", "Manual", "Entropy", "CRITIC", "AHP", "BWM"]
    if mode == "Fuzzy":
        options += ["Fuzzy manual", "Fuzzy AHP"]
    stored_weighting = settings.get("weighting", "Equal")
    weighting = st.selectbox(
        "Méthode de pondération",
        options,
        index=options.index(stored_weighting) if stored_weighting in options else 0,
        key=key("weighting_" + mode),
        format_func=label,
    )
    settings["weighting"] = weighting
    with st.expander("Comprendre cette pondération"):
        st.write(WEIGHT_GUIDES[weighting])
    fw = None
    try:
        if weighting in ["Equal", "Entropy", "CRITIC"]:
            weights = objective_weights(centroids, benefits, weighting)
            if mode == "Fuzzy" and weighting != "Equal":
                st.caption(
                    "Les poids objectifs sont calculés sur les centres de gravité des triangles. Le classement flou utilise toujours les trois bornes."
                )
        elif weighting == "Manual":
            weights = weights_vector([c["weight"] for c in criteria], len(names))
        elif weighting == "Fuzzy manual":
            prior = settings.get("fuzzy_weights", {})
            table = pd.DataFrame(
                {
                    "Critère": names,
                    "Triangle": [str(prior.get(n, "1;1;1")) for n in names],
                }
            )
            edited = st.data_editor(
                table,
                disabled=["Critère"],
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
                "Saisissez les comparaisons au-dessus de la diagonale ; les réciproques sont calculées automatiquement. Utilisez 1 pour une importance égale, 3/5/7/9 pour une préférence croissante, ou une valeur décimale inverse pour une importance moindre."
            )
            pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]
            saved = settings.get("fuzzy_pairwise" if fuzzy_pairwise else "pairwise", {})
            if pairs:
                table = pd.DataFrame(
                    {
                        "Critère de gauche": [names[i] for i, j in pairs],
                        "Critère de droite": [names[j] for i, j in pairs],
                        "Gauche / droite": [
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
                    disabled=["Critère de gauche", "Critère de droite"],
                    hide_index=True,
                    width="stretch",
                    key=key(weighting + json.dumps(names)),
                )
                judgments = edited["Gauche / droite"].tolist()
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
                    raise ValueError(
                        "Les comparaisons deux à deux doivent être strictement positives."
                    )
                a[i, j] = t
                a[j, i] = 1 / t[::-1] if fuzzy_pairwise else 1 / t
            if fuzzy_pairwise:
                weights, fw = fuzzy_ahp(a)
                _, cr = ahp(a[:, :, 1])
                st.caption(
                    "L’AHP flou utilise les moyennes géométriques de Buckley et une normalisation par somme triangulaire inversée. La cohérence ci-dessous porte sur la matrice des valeurs modales."
                )
            else:
                weights, cr = ahp(a)
            if cr is None:
                st.info(
                    "Le ratio de cohérence n’est pas disponible au-delà de 10 critères ; aucun indice aléatoire approximatif n’est utilisé."
                )
            else:
                st.metric(
                    "Ratio de cohérence modal" if fuzzy_pairwise else "Ratio de cohérence",
                    f"{cr:.3f}",
                )
                if cr > 0.10:
                    st.warning(
                        "Le ratio CR dépasse 0,10. Revoyez les comparaisons avant de vous appuyer sur le classement."
                    )
        else:
            if len(names) < 2:
                raise ValueError(
                    "BWM nécessite au moins deux critères. Choisissez une autre méthode de pondération."
                )
            saved = settings.get("bwm", {})
            best_name = st.selectbox(
                "Critère le plus important",
                names,
                index=names.index(saved["best"]) if saved.get("best") in names else 0,
                key=key("best_" + json.dumps(names)),
            )
            worst_options = [n for n in names if n != best_name]
            worst_name = st.selectbox(
                "Critère le moins important",
                worst_options,
                index=worst_options.index(saved["worst"])
                if saved.get("worst") in worst_options
                else len(worst_options) - 1,
                key=key("worst_" + best_name + json.dumps(names)),
            )
            best, worst = names.index(best_name), names.index(worst_name)
            shared = st.number_input(
                "Préférence meilleur / moins important",
                1.0,
                9.0,
                float(saved.get("shared", 3)),
                key=key("bw_shared"),
            )
            st.caption(
                "Les comparaisons vont de 1 à 9. Les comparaisons d’un critère avec lui-même et la préférence commune meilleur/moins important sont définies automatiquement."
            )
            a = saved.get("a", [1.0] * len(names))
            c = saved.get("c", [1.0] * len(names))
            if len(a) != len(names) or len(c) != len(names):
                a, c = [1.0] * len(names), [1.0] * len(names)
            table = pd.DataFrame(
                {"Critère": names, "Meilleur / critère": a, "Critère / moins important": c}
            )
            edited = st.data_editor(
                table,
                disabled=["Critère"],
                hide_index=True,
                width="stretch",
                key=key("bwm_" + json.dumps(names)),
                column_config={
                    n: st.column_config.NumberColumn(min_value=1.0, max_value=9.0)
                    for n in ["Meilleur / critère", "Critère / moins important"]
                },
            )
            a, c = (
                edited["Meilleur / critère"].to_numpy(float).copy(),
                edited["Critère / moins important"].to_numpy(float).copy(),
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
            st.metric("Résidu linéaire optimal BWM", f"{residual:.4f}")
            st.caption("Ce résidu d’optimisation n’est pas un ratio de cohérence AHP.")
        weight_frame = pd.DataFrame({"Critère": names, "Poids": weights})
        if fw is not None:
            for j, endpoint in enumerate(["Borne inférieure", "Valeur modale", "Borne supérieure"]):
                weight_frame[endpoint] = fw[:, j]
        st.dataframe(weight_frame, hide_index=True, width="stretch")
        display_chart(
            px.bar(
                weight_frame,
                x="Critère",
                y="Poids",
                color_discrete_sequence=["#1d4ed8"],
            ),
        )
    except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
        st.error(error_message(exc))
        st.stop()

registry = METHODS if mode == "Crisp" else FUZZY_METHODS
default_methods = (
    ["TOPSIS", "SAW / WSM", "VIKOR"] if mode == "Crisp" else ["Fuzzy TOPSIS", "Fuzzy SAW"]
)
with ranking_tab:
    st.subheader("Choisir les méthodes de classement")
    restored = settings.get("methods_" + mode, default_methods)
    selected = st.multiselect(
        "Méthodes de classement",
        list(registry),
        default=[m for m in restored if m in registry],
        key=key("methods_" + mode),
        format_func=label,
    )
    settings["methods_" + mode] = selected
    v, lam = float(settings.get("v", 0.5)), float(settings.get("lambda", 0.5))
    if "VIKOR" in selected:
        v = st.slider(
            "VIKOR : v · importance de la satisfaction globale", 0.0, 1.0, v, 0.05, key=key("v")
        )
    if "WASPAS" in selected:
        lam = st.slider(
            "WASPAS : λ · part de la somme pondérée", 0.0, 1.0, lam, 0.05, key=key("lambda")
        )
    settings.update(v=v, **{"lambda": lam})
    st.caption(
        "Comparez les rangs entre méthodes : les scores bruts ont des sens et des échelles différents. Les ex æquo partagent le même rang après arrondi à 12 décimales."
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
                {"Alternative": labels, "Score": result.scores, "Rang": result.ranks}
            ).sort_values(["Rang", "Alternative"], kind="stable")
            tables[method] = table
            st.markdown(f"### {label(method)}")
            st.caption(RANKING_GUIDES[method]["summary"])
            with st.expander("Comment fonctionne " + label(method) + " ?"):
                explain_method(method)
            winners = [labels[i] for i in range(len(labels)) if result.ranks[i] == 1]
            st.success("Premier rang : " + ", ".join(winners))
            if method == "VIKOR" and not (
                result.details["acceptable_advantage"] and result.details["acceptable_stability"]
            ):
                st.warning(
                    "Les conditions de compromis VIKOR ne sont pas toutes satisfaites. Examinez les alternatives de tête comme un ensemble de compromis ; le seul rang Q ne permet pas une recommandation unique."
                )
            st.dataframe(table, hide_index=True, width="stretch")
            with st.expander("Détail du calcul · " + label(method)):
                for name, values in result.details.items():
                    if isinstance(values, np.ndarray):
                        st.caption(detail_label(method, name))
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
                                    ["Borne inférieure", "Valeur modale", "Borne supérieure"]
                                    if values.shape[1] == 3
                                    else [str(i + 1) for i in range(values.shape[1])]
                                )
                            detail = pd.DataFrame(values, columns=columns)
                        else:
                            detail = pd.DataFrame({detail_label(method, name): values})
                        detail.insert(0, "Alternative", labels)
                        st.dataframe(detail, hide_index=True, width="stretch")
                    else:
                        st.write(label(name), "Oui" if values else "Non")
        except (ValueError, FloatingPointError) as exc:
            failures[method] = error_message(exc)
            st.warning(f"{label(method)} : {error_message(exc)}")
    if not selected:
        st.info("Sélectionnez au moins une méthode de classement.")

with compare_tab:
    st.subheader("Comparer les classements")
    if results:
        ranks = pd.DataFrame({label(m): r.ranks for m, r in results.items()}, index=labels)
        ranks.index.name = "Alternative"
        st.dataframe(ranks.reset_index(), hide_index=True, width="stretch")
        long = ranks.reset_index().melt(
            id_vars="Alternative", var_name="Méthode", value_name="Rang"
        )
        fig = px.line(long, x="Méthode", y="Rang", color="Alternative", markers=True)
        fig.update_yaxes(autorange="reversed", dtick=1)
        display_chart(fig)
        if len(results) > 1:
            st.caption(
                "Corrélation des rangs de Spearman. Une case vide indique une corrélation indéfinie, car tous les rangs sont identiques."
            )
            st.dataframe(ranks.corr(method="spearman"), width="stretch")
        st.caption(
            "Le rang moyen résume l’accord entre les méthodes calculées. Ce n’est pas une méthode MCDM supplémentaire."
        )
        consensus = pd.DataFrame(
            {"Alternative": labels, "Rang moyen": ranks.mean(axis=1).values}
        ).sort_values("Rang moyen", kind="stable")
        st.dataframe(consensus, hide_index=True, width="stretch")
    else:
        st.info("Choisissez une méthode compatible pour comparer les classements.")

with sensitivity_tab:
    st.subheader("La décision est-elle stable ?")
    st.caption(
        "Faites varier le poids d’un critère de 0 à 1 ; le poids restant est réparti proportionnellement entre les autres. Cette analyse mesure la sensibilité aux poids, pas l’incertitude des performances."
    )
    if results and len(names) > 1:
        sensitivity_method = st.selectbox(
            "Méthode à analyser",
            list(results),
            format_func=label,
            key=key("sensitivity_method_" + mode),
        )
        changed_name = st.selectbox(
            "Critère à faire varier",
            names,
            key=key("sensitivity_criterion_" + json.dumps(names)),
        )
        changed = names.index(changed_name)
        sensitivity_signature = (
            json.dumps(problem, sort_keys=True),
            sensitivity_method,
            changed_name,
        )
        if st.button("Lancer l’analyse de sensibilité", key=key("sensitivity_run")):
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
                        {"Poids": share, "Alternative": label, "Rang": int(r.ranks[i])}
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
                chart = px.line(sensitivity, x="Poids", y="Rang", color="Alternative", markers=True)
                chart.update_yaxes(autorange="reversed", dtick=1)
                display_chart(chart)
                st.download_button(
                    "Télécharger la sensibilité en CSV",
                    sensitivity.to_csv(index=False).encode(),
                    "sensitivity.csv",
                    "text/csv",
                    on_click="ignore",
                )
            if skipped:
                st.warning(
                    f"{len(skipped)} scénarios de pondération ne respectent pas les hypothèses de cette méthode."
                )
    else:
        st.info(
            "La sensibilité nécessite au moins deux critères et une méthode de classement calculable."
        )

with guide_tab:
    st.subheader("Comprendre le fonctionnement des méthodes")
    st.write(
        "Une alternative est un choix possible ; un critère est un axe d’évaluation. Le poids indique son importance, tandis que le sens indique s’il faut maximiser ou minimiser la performance. Les méthodes combinent ces informations pour établir un classement."
    )
    explained = st.selectbox(
        "Méthode à comprendre", list(RANKING_GUIDES), format_func=label, key=key("guide_method")
    )
    explain_method(explained)
    with st.expander("Comprendre la logique floue"):
        st.write(
            "Lorsque l’évaluation est imprécise, un triangle (l, m, u) représente une borne basse, une valeur la plus plausible et une borne haute. Par exemple, (5, 7, 9) exprime une appréciation autour de 7, avec une plage de 5 à 9. Il s’agit d’une possibilité, pas d’une distribution de probabilités."
        )
        st.write(
            "La normalisation rend les échelles comparables. La pondération applique l’importance des critères. La défuzzification résume un triangle par un nombre : elle intervient à la fin de la somme pondérée floue. TOPSIS flou conserve les trois bornes pour mesurer les distances."
        )
    with st.expander("Comment choisir une méthode ?"):
        st.markdown(
            "- **SAW/WSM** : une somme facile à expliquer, avec compensation entre critères.\n"
            "- **TOPSIS** : une comparaison à un profil idéal et à son opposé.\n"
            "- **VIKOR** : un compromis entre satisfaction globale et regret maximal.\n"
            "- **PROMETHEE II** : des préférences entre alternatives, sans seuil dans cette version.\n"
            "- **Méthodes floues** : des appréciations linguistiques ou des plages plausibles."
        )
        st.write(
            "Comparez plusieurs classements et utilisez la sensibilité pour vérifier l’effet des poids. Un accord entre méthodes ne garantit pas la justesse des données ni des préférences choisies."
        )
    st.caption(
        "Périmètre : matrices finies et complètes d’alternatives et de critères. Un objectif cible peut être transformé en écart à minimiser. Les contraintes de portefeuille, les critères interdépendants, l’optimisation continue et d’autres types d’ensembles flous demandent un modèle adapté."
    )
    st.markdown(
        "Références : [définitions et exemples des méthodes MCDM](https://pymcdm.readthedocs.io/en/master/pymcdm.methods.html), [nombres flous triangulaires et distance des sommets](https://www.mdpi.com/2071-1050/13/6/3020)."
    )

st.divider()
st.subheader("Enregistrer votre travail")
col1, col2, col3 = st.columns(3)
with col1:
    st.download_button(
        "Télécharger le projet JSON",
        dump_json(problem),
        "decision_project.json",
        "application/json",
        width="stretch",
    )
with col2:
    st.download_button(
        "Télécharger la matrice CSV",
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
        "Télécharger les résultats ZIP",
        archive.getvalue(),
        "decision_results.zip",
        "application/zip",
        width="stretch",
    )
st.caption(
    "Le projet JSON conserve les données, les sens, les unités, les échelles linguistiques, les comparaisons et les paramètres des méthodes. Les téléchargements incluent les modifications actuelles."
)
