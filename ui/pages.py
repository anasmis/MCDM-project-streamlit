import html
import json

import numpy as np
import pandas as pd
import streamlit as st

from mcdm import AHP, BWM, CRITIC, TOPSIS, WSM, Criterion, DecisionProblem, Entropy
from mcdm.analysis import compare_rankings, recommend_weighting, sensitivity
from mcdm.io import (
    csv_bytes,
    example_problem,
    export_bundle,
    matrix_from_csv,
    problem_from_json,
    problem_to_json,
)
from ui.charts import ranking_chart, sensitivity_chart, weight_chart
from ui.state import navigate, new_problem, set_draft
from ui.style import heading, note

CHART_CONFIG = {"displayModeBar": False, "locale": "fr"}
METHODS = {
    "AHP": (
        "Comparaisons par paires",
        "Exprimez vos priorités en comparant chaque paire de critères.",
    ),
    "BWM": (
        "Meilleur et moins important",
        "Ancrez vos comparaisons sur les deux critères extrêmes.",
    ),
    "CRITIC": (
        "Dispersion et corrélations",
        "Valorisez les critères contrastés qui apportent une information différente.",
    ),
    "Entropie": (
        "Information des performances",
        "Valorisez les critères dont les performances sont les plus discriminantes.",
    ),
}


def criteria_from_frame(frame: pd.DataFrame) -> tuple[Criterion, ...]:
    if frame["Critère"].isna().any() or frame["Objectif"].isna().any():
        raise ValueError("Renseignez le nom et l'objectif de chaque critère.")
    if not frame["Objectif"].isin(["Maximiser", "Minimiser"]).all():
        raise ValueError("Choisissez Maximiser ou Minimiser pour chaque critère.")
    return tuple(
        Criterion(
            row["Critère"],
            "max" if row["Objectif"] == "Maximiser" else "min",
            row["Unité"] if pd.notna(row["Unité"]) else "",
        )
        for _, row in frame.iterrows()
    )


def import_panel() -> None:
    with st.expander("Importer un problème ou une matrice"):
        st.caption(
            "JSON : un problème sauvegardé dans Arbitrage. CSV UTF-8 : première colonne = "
            "alternatives, suivantes = critères numériques. Séparateurs : ; , ou tabulation."
        )
        uploaded = st.file_uploader("Fichier de données", type=["json", "csv"])
        if st.button("Charger le fichier", disabled=uploaded is None):
            try:
                content = uploaded.getvalue()
                if uploaded.name.lower().endswith("json"):
                    imported = problem_from_json(content)
                else:
                    frame = matrix_from_csv(content)
                    imported = DecisionProblem(
                        uploaded.name.rsplit(".", 1)[0],
                        tuple(frame["Alternative"]),
                        tuple(Criterion(name) for name in frame.columns[1:]),
                        frame.iloc[:, 1:].to_numpy(dtype=float),
                    )
                set_draft(imported)
                st.session_state["import_notice"] = (
                    "Fichier chargé. Vérifiez le sens des critères avant de continuer. "
                    "Un CSV initialise tous les critères à Maximiser."
                )
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        st.download_button(
            "Télécharger un exemple CSV",
            csv_bytes(example_problem().to_frame().rename_axis("Alternative").reset_index()),
            "exemple_fournisseurs.csv",
            "text/csv",
        )


def problem_page() -> None:
    heading(
        "01 / Cadrage",
        "Toute décision commence par un problème.",
        "Définissez ce que vous comparez, les critères qui comptent et les performances de chaque alternative.",
    )
    actions = st.columns([1, 1, 2.3])
    actions[0].button("Nouvelle étude", width="stretch", on_click=new_problem)
    actions[1].button(
        "Charger l'exemple", width="stretch", on_click=set_draft, args=(example_problem(), True)
    )
    if st.session_state.is_example:
        actions[2].caption("EXEMPLE MODIFIABLE · Sélection d'un fournisseur")
    import_panel()
    if "import_notice" in st.session_state:
        st.info(st.session_state.pop("import_notice"))
    left, right = st.columns([3.7, 1.25], gap="large")
    revision = st.session_state.revision
    with left:
        st.subheader("Le cadre de l'étude")
        with st.form(f"structure_{revision}"):
            name = st.text_input(
                "Nom du problème", value=st.session_state.draft_name, max_chars=120
            )
            description = st.text_area(
                "Objectif de la décision",
                value=st.session_state.draft_description,
                height=85,
                placeholder="Quel choix souhaitez-vous éclairer ?",
            )
            st.markdown("**Critères d'évaluation**")
            criteria_frame = st.data_editor(
                st.session_state.draft_criteria,
                hide_index=True,
                num_rows="dynamic",
                width="stretch",
                key=f"criteria_editor_{revision}",
                column_config={
                    "Critère": st.column_config.TextColumn("Critère", required=True),
                    "Objectif": st.column_config.SelectboxColumn(
                        "Objectif", options=["Maximiser", "Minimiser"], required=True
                    ),
                    "Unité": st.column_config.TextColumn(
                        "Unité", help="Facultatif : €, jours, /100…"
                    ),
                },
            )
            structure_submit = st.form_submit_button("Enregistrer le cadre", width="stretch")
        if structure_submit:
            try:
                criteria = criteria_from_frame(criteria_frame)
                checked = DecisionProblem(
                    name, ("A", "B"), criteria, np.zeros((2, len(criteria))), description
                )
                old = st.session_state.draft_matrix
                names = checked.criterion_names
                new = old.reindex(columns=["Alternative", *names])
                st.session_state.update(
                    {
                        "draft_name": checked.name,
                        "draft_description": description,
                        "draft_criteria": pd.DataFrame(
                            {
                                "Critère": names,
                                "Objectif": [
                                    "Maximiser" if c.direction == "max" else "Minimiser"
                                    for c in criteria
                                ],
                                "Unité": [c.unit for c in criteria],
                            }
                        ),
                        "draft_matrix": new,
                        "problem": None,
                        "weighting": None,
                        "revision": revision + 1,
                        "judgments": {},
                        "is_example": False,
                    }
                )
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    with right:
        note(
            "Un critère, un sens",
            "Maximiser une qualité, une autonomie ou un rendement. Minimiser un coût, un délai ou un risque.",
        )
        note(
            "Votre matrice",
            "De 2 à 15 critères et de 2 à 500 alternatives. Les unités peuvent être différentes : les méthodes normalisent les valeurs.",
        )
        st.caption(
            "Enregistrez le cadre avant de saisir les performances. Un critère renommé ou ajouté crée une colonne à compléter."
        )
    st.subheader("Matrice des performances")
    st.caption(
        "Une ligne par alternative. Modifiez les cellules ; ajoutez ou supprimez des lignes dans le tableau."
    )
    with st.form(f"matrix_{revision}"):
        frame = st.data_editor(
            st.session_state.draft_matrix,
            hide_index=True,
            num_rows="dynamic",
            width="stretch",
            key=f"matrix_editor_{revision}",
            column_config={
                "Alternative": st.column_config.TextColumn("Alternative", required=True),
                **{
                    row["Critère"]: st.column_config.NumberColumn(
                        f"{row['Critère']} · {row['Unité']}" if row["Unité"] else row["Critère"],
                        required=True,
                        format="%.4f",
                    )
                    for _, row in st.session_state.draft_criteria.iterrows()
                },
            },
        )
        st.caption(
            "Le classement utilisera le dernier cadre enregistré et les performances ci-dessus."
        )
        submit = st.form_submit_button("Valider le problème et choisir les poids →", type="primary")
    if submit:
        try:
            problem = DecisionProblem(
                st.session_state.draft_name,
                tuple(frame["Alternative"]),
                criteria_from_frame(st.session_state.draft_criteria),
                frame.iloc[:, 1:].to_numpy(dtype=float),
                st.session_state.draft_description,
            )
            st.session_state.update(
                {"problem": problem, "draft_matrix": frame.copy(), "weighting": None, "step": 1}
            )
            st.rerun()
        except (ValueError, TypeError) as exc:
            st.error(
                str(exc)
                if isinstance(exc, ValueError)
                else "Complétez les noms et performances de chaque alternative."
            )


def ahp_inputs(problem: DecisionProblem):
    names = problem.criterion_names
    stored = st.session_state.judgments.setdefault("AHP", {})
    st.markdown("#### Comparez les critères deux à deux")
    st.caption(
        "1 = même importance · 3 = modérée · 5 = forte · 7 = très forte · 9 = extrême. Les réciproques favorisent le second critère."
    )
    scale = [1 / n for n in range(9, 1, -1)] + list(range(1, 10))
    matrix = np.eye(len(names))
    container = st.container(height=360) if len(names) > 5 else st.container()
    with container:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                key = f"{i}_{j}"
                saved = stored.get(key, 1)
                value = st.select_slider(
                    f"{names[i]} / {names[j]}",
                    options=scale,
                    value=saved,
                    format_func=lambda x: f"1/{round(1 / x)}" if x < 1 else str(x),
                    key=f"ahp_{st.session_state.revision}_{key}",
                )
                stored[key] = value
                matrix[i, j], matrix[j, i] = value, 1 / value
    return AHP().compute(matrix)


def bwm_inputs(problem: DecisionProblem):
    names = problem.criterion_names
    stored = st.session_state.judgments.setdefault("BWM", {})
    cols = st.columns(2)
    best = cols[0].selectbox(
        "Critère le plus important", names, index=stored.get("best", 0), key="bwm_best"
    )
    worst_options = [name for name in names if name != best]
    saved_worst = stored.get("worst_name", worst_options[-1])
    worst = cols[1].selectbox(
        "Critère le moins important",
        worst_options,
        index=worst_options.index(saved_worst)
        if saved_worst in worst_options
        else len(worst_options) - 1,
        key=f"bwm_worst_{best}",
    )
    b, w = names.index(best), names.index(worst)
    stored.update({"best": b, "worst_name": worst})
    pair_key = f"{b}_{w}"
    inputs = stored.setdefault(pair_key, {})
    bw = st.slider(
        f"Importance de {best} par rapport à {worst}",
        1,
        9,
        inputs.get("bw", 3),
        key=f"bwm_bw_{pair_key}",
    )
    inputs["bw"] = bw
    ab, aw = np.ones(len(names)), np.ones(len(names))
    ab[w], aw[b] = bw, bw
    st.caption(
        "1 = même importance · 9 = importance extrême. La comparaison des deux extrêmes est partagée entre les deux séries."
    )
    left, right = st.columns(2, gap="large")
    for j, name in enumerate(names):
        if j in (b, w):
            continue
        with left:
            ab[j] = st.slider(
                f"{best} / {name}", 1, 9, inputs.get(f"ab_{j}", 2), key=f"bwm_ab_{pair_key}_{j}"
            )
            inputs[f"ab_{j}"] = int(ab[j])
        with right:
            aw[j] = st.slider(
                f"{name} / {worst}", 1, 9, inputs.get(f"aw_{j}", 2), key=f"bwm_aw_{pair_key}_{j}"
            )
            inputs[f"aw_{j}"] = int(aw[j])
    return BWM().compute(b, w, ab, aw)


def weight_details(problem, result) -> None:
    with st.expander("Voir les calculs de pondération"):
        names = problem.criterion_names
        details = result.details
        if result.method == "AHP":
            st.dataframe(
                pd.DataFrame(details["comparisons"], index=names, columns=names), width="stretch"
            )
            st.caption(
                f"λ max = {details['lambda_max']:.6f} · IC = {details['consistency_index']:.6f} · RC = {details['consistency_ratio']:.2%}"
            )
        elif result.method == "BWM":
            st.dataframe(
                pd.DataFrame(
                    {
                        "Critère": names,
                        "Meilleur / critère": details["best_to_others"],
                        "Critère / moins important": details["others_to_worst"],
                    }
                ),
                hide_index=True,
            )
            st.caption(
                f"Résidu optimal ξ* = {details['residual']:.6f}. Plus il est proche de 0, mieux les poids respectent vos comparaisons. Ce résidu n'est pas un ratio de cohérence AHP."
            )
        elif result.method == "CRITIC":
            st.dataframe(
                pd.DataFrame(
                    {
                        "Critère": names,
                        "Écart-type": details["standard_deviation"],
                        "Information": details["information"],
                    }
                ),
                hide_index=True,
            )
            st.markdown("**Corrélations des critères variables**")
            active = details["standard_deviation"] > 0
            st.dataframe(
                pd.DataFrame(
                    details["correlation"][np.ix_(active, active)],
                    index=np.array(names)[active],
                    columns=np.array(names)[active],
                )
            )
        else:
            st.dataframe(
                pd.DataFrame(
                    {
                        "Critère": names,
                        "Entropie": details["entropy"],
                        "Divergence": details["divergence"],
                    }
                ),
                hide_index=True,
            )
            st.caption(
                "Convention : min–max orientée, puis proportions par colonne. Les termes 0 × ln(0) valent 0."
            )


def weighting_page() -> None:
    problem = st.session_state.problem
    if problem is None:
        navigate(0)
        st.rerun()
    heading(
        "02 / Pondération",
        "Donnez à chaque critère sa juste place.",
        "Choisissez comment exprimer l'importance des critères. Les poids sont normalisés pour totaliser 100 %.",
    )
    preference_label = st.radio(
        "Sur quoi souhaitez-vous fonder les poids ?",
        ["Les performances observées", "Mes priorités de décision"],
        horizontal=True,
        index=0 if st.session_state.preference == "objective" else 1,
    )
    preference = "objective" if preference_label == "Les performances observées" else "subjective"
    st.session_state.preference = preference
    recommended, reason = recommend_weighting(problem, preference)
    options = ["CRITIC", "Entropie"] if preference == "objective" else ["AHP", "BWM"]
    selected = st.session_state.weighting_method
    selected = selected if selected in options else recommended
    note(f"Méthode suggérée · {recommended}", reason + " Ce choix reste modifiable.")
    method = st.radio(
        "Méthode de pondération",
        options,
        index=options.index(selected),
        horizontal=True,
        format_func=lambda name: f"{name} · {METHODS[name][0]}",
    )
    st.session_state.weighting_method = method
    st.caption(METHODS[method][1])
    if method == "Entropie" and len(problem.alternatives) == 2:
        st.info(
            "Avec deux alternatives et une normalisation min–max, tous les critères variables reçoivent le même poids par entropie."
        )
    try:
        if method == "AHP":
            result = ahp_inputs(problem)
        elif method == "BWM":
            result = bwm_inputs(problem)
        else:
            result = (CRITIC() if method == "CRITIC" else Entropy()).compute(problem)
    except ValueError as exc:
        st.session_state.weighting = None
        st.error(str(exc))
        return
    signature = json.dumps(
        {"method": result.method, "details": result.details},
        default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value,
    )
    if signature != st.session_state.get("validated_signature"):
        st.session_state.weighting = None
    st.divider()
    left, right = st.columns([1.7, 1], gap="large")
    with left:
        st.subheader("Répartition des poids")
        st.plotly_chart(
            weight_chart(problem.criterion_names, result.weights),
            width="stretch",
            config=CHART_CONFIG,
        )
    with right:
        st.subheader("Poids calculés")
        st.dataframe(
            pd.DataFrame({"Critère": problem.criterion_names, "Poids (%)": result.weights * 100}),
            hide_index=True,
            width="stretch",
            column_config={"Poids (%)": st.column_config.NumberColumn(format="%.2f %%")},
        )
        st.caption(f"Total : {result.weights.sum():.0%}")
        if method == "AHP":
            cr = result.details["consistency_ratio"]
            st.metric("Ratio de cohérence", f"{cr:.1%}".replace(".", ","))
            if cr <= 0.1:
                st.caption("Comparaisons cohérentes au seuil usuel de 10 %.")
    for warning in result.warnings:
        st.warning(warning)
    accepted = True
    if method == "AHP" and result.details["consistency_ratio"] > 0.1:
        accepted = st.checkbox(
            "Je souhaite poursuivre avec ces comparaisons incohérentes.", key=f"accept_{signature}"
        )
    weight_details(problem, result)
    back, forward = st.columns([1, 2])
    back.button("← Modifier le problème", on_click=navigate, args=(0,))
    if forward.button(
        "Valider les poids et classer →", type="primary", disabled=not accepted, width="stretch"
    ):
        st.session_state.update({"weighting": result, "validated_signature": signature, "step": 2})
        st.rerun()


def ranking_page() -> None:
    problem, weighting = st.session_state.problem, st.session_state.weighting
    if problem is None or weighting is None:
        navigate(0 if problem is None else 1)
        st.rerun()
    heading(
        "03 / Classement",
        "Passez des critères au choix.",
        "Examinez le classement, comparez les méthodes et vérifiez l'effet d'une variation des poids.",
    )
    with st.expander("Rappel du problème et des poids"):
        st.write(problem.name)
        st.caption(problem.description)
        st.dataframe(problem.to_frame(), width="stretch")
        st.dataframe(
            pd.DataFrame(
                {
                    "Critère": problem.criterion_names,
                    "Objectif": [
                        "Maximiser" if c.direction == "max" else "Minimiser"
                        for c in problem.criteria
                    ],
                    "Poids (%)": weighting.weights * 100,
                }
            ),
            hide_index=True,
        )
    method = st.radio(
        "Méthode de classement",
        ["TOPSIS", "WSM"],
        horizontal=True,
        index=0 if st.session_state.ranking_method == "TOPSIS" else 1,
        format_func=lambda x: (
            "TOPSIS · Proximité de l'idéal" if x == "TOPSIS" else "WSM · Somme pondérée"
        ),
    )
    st.session_state.ranking_method = method
    if method == "TOPSIS":
        st.caption(
            "TOPSIS favorise la proximité du meilleur profil et l'éloignement du pire. Normalisation vectorielle, puis distances euclidiennes pondérées."
        )
    else:
        st.caption(
            "WSM additionne les performances normalisées min–max, pondérées par vos critères. Une bonne performance peut compenser une moins bonne."
        )
    result = (TOPSIS() if method == "TOPSIS" else WSM()).compute(problem, weighting.weights)
    table = result.to_frame(problem.alternatives)
    winners = table.loc[table["Rang"] == 1, "Alternative"].tolist()
    winner_label = "PREMIER RANG" if len(winners) == 1 else f"PREMIER RANG · {len(winners)} EX ÆQUO"
    st.markdown(
        f'<div class="winner"><div class="eyebrow">{winner_label}</div>'
        f"<h2>{html.escape(' · '.join(winners[:5]))}{'…' if len(winners) > 5 else ''}</h2>"
        f"<p>{method} · Pondération {weighting.method} · {len(problem.alternatives)} alternatives comparées</p></div>",
        unsafe_allow_html=True,
    )
    if len(winners) == len(problem.alternatives):
        st.info(
            "Les données et les poids retenus ne permettent pas de départager les alternatives."
        )
    tabs = st.tabs(["Classement", "Comparer WSM et TOPSIS", "Sensibilité", "Détail du calcul"])
    with tabs[0]:
        left, right = st.columns([1.55, 1], gap="large")
        with left:
            st.plotly_chart(ranking_chart(table.head(20)), width="stretch", config=CHART_CONFIG)
            if len(table) > 20:
                st.caption(
                    "Le graphique présente les 20 premières alternatives. Le tableau contient le classement complet."
                )
        with right:
            st.dataframe(
                table,
                hide_index=True,
                width="stretch",
                column_config={"Score": st.column_config.NumberColumn(format="%.4f")},
            )
            st.caption(
                "Un score plus élevé indique un meilleur classement. Les ex æquo partagent le même rang."
            )
            st.download_button(
                "Exporter le classement · CSV",
                csv_bytes(table),
                "classement.csv",
                "text/csv",
                width="stretch",
            )
    with tabs[1]:
        comparison = compare_rankings(problem, weighting.weights)
        agreement = np.array_equal(comparison["Rang WSM"], comparison["Rang TOPSIS"])
        st.markdown(
            "**Les deux méthodes donnent le même classement.**"
            if agreement
            else "**Les méthodes conduisent à des classements différents.**"
        )
        st.caption(
            "Comparez les rangs : les scores de WSM et de TOPSIS n'ont pas la même interprétation. Écart de rang = rang TOPSIS − rang WSM."
        )
        st.dataframe(
            comparison,
            hide_index=True,
            width="stretch",
            column_config={
                "Score WSM": st.column_config.NumberColumn(format="%.4f"),
                "Score TOPSIS": st.column_config.NumberColumn(format="%.4f"),
            },
        )
    with tabs[2]:
        st.markdown("#### Le premier rang résiste-t-il à un changement de priorité ?")
        criterion = st.selectbox("Critère à faire varier", problem.criterion_names)
        index = problem.criterion_names.index(criterion)
        st.caption(
            "Le poids sélectionné est multiplié par 0,7 à 1,3, puis tous les poids sont renormalisés. Les proportions des autres critères sont conservées."
        )
        if weighting.weights[index] == 0:
            st.info(
                "Ce critère a un poids nul : une variation multiplicative ne modifie pas les poids."
            )
        frame = sensitivity(problem, weighting.weights, method, index)
        leaders = frame[frame["Rang"] == 1]
        stable = all(
            set(group["Alternative"]) == set(winners)
            for _, group in leaders.groupby("Variation (%)")
        )
        st.caption(
            "Premier rang inchangé sur les sept scénarios testés."
            if stable
            else "Le premier rang change dans au moins un des sept scénarios testés."
        )
        shown = table.head(10)["Alternative"]
        st.plotly_chart(
            sensitivity_chart(frame[frame["Alternative"].isin(shown)]),
            width="stretch",
            config=CHART_CONFIG,
        )
        if len(table) > 10:
            st.caption(
                "Courbes limitées aux 10 premières alternatives du classement initial ; tous les scénarios sont exportables."
            )
        st.download_button(
            "Exporter les scénarios · CSV", csv_bytes(frame), "sensibilite.csv", "text/csv"
        )
        st.caption(
            "Cette exploration locale ne garantit pas la stabilité pour d'autres variations ni pour l'ajout d'alternatives."
        )
    with tabs[3]:
        st.markdown("**Matrice normalisée**")
        st.dataframe(
            pd.DataFrame(
                result.normalized_matrix,
                index=problem.alternatives,
                columns=problem.criterion_names,
            ),
            width="stretch",
        )
        st.markdown("**Matrice normalisée et pondérée**")
        st.dataframe(
            pd.DataFrame(
                result.weighted_matrix, index=problem.alternatives, columns=problem.criterion_names
            ),
            width="stretch",
        )
        if method == "TOPSIS":
            st.dataframe(
                pd.DataFrame(
                    {
                        "Alternative": problem.alternatives,
                        "Distance à l'idéal": result.details["distance_best"],
                        "Distance à l'anti-idéal": result.details["distance_worst"],
                    }
                ),
                hide_index=True,
            )
            st.caption(
                "Score = distance à l'anti-idéal / somme des deux distances. Si les deux distances sont nulles, le score conventionnel vaut 0,5."
            )
        else:
            st.caption(
                "Score = somme de chaque ligne de la matrice pondérée. Un critère constant est normalisé à 1 pour toutes les alternatives."
            )
    st.divider()
    left, middle, right = st.columns([1, 1, 1.3])
    left.button("← Réviser les poids", on_click=navigate, args=(1,))
    middle.download_button(
        "Sauvegarder le problème",
        problem_to_json(problem),
        "probleme.json",
        "application/json",
        width="stretch",
    )
    right.download_button(
        "Télécharger le dossier complet",
        export_bundle(problem, weighting, result),
        "arbitrage_resultats.zip",
        "application/zip",
        type="primary",
        width="stretch",
    )
    st.caption(
        "Le dossier comprend les données, les poids, les comparaisons, les résultats et une synthèse. Le fichier problème JSON permet de reprendre le cadrage ; les poids sont recalculés lors d'une nouvelle session."
    )
