import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from mcdm.models import DecisionProblem, validate_weights
from mcdm.ranking import TOPSIS, WSM


def recommend_weighting(problem: DecisionProblem, preference: str) -> tuple[str, str]:
    if preference == "subjective":
        if len(problem.criteria) > 5:
            return "BWM", "Avec plus de cinq critères, BWM réduit le nombre de comparaisons à saisir."
        return "AHP", "AHP permet de comparer chaque paire de critères et de vérifier la cohérence."
    if len(problem.alternatives) < 4:
        return "Entropie", (
            "Avec peu d'alternatives, les corrélations de CRITIC sont fragiles. "
            "L'entropie offre une lecture plus simple de la dispersion."
        )
    return "CRITIC", "CRITIC prend en compte la dispersion et la redondance entre les critères."


def compare_rankings(problem: DecisionProblem, weights: ArrayLike) -> pd.DataFrame:
    wsm, topsis = WSM().compute(problem, weights), TOPSIS().compute(problem, weights)
    return pd.DataFrame({
        "Alternative": problem.alternatives,
        "Rang WSM": wsm.ranks, "Score WSM": wsm.scores,
        "Rang TOPSIS": topsis.ranks, "Score TOPSIS": topsis.scores,
        "Écart de rang": topsis.ranks - wsm.ranks,
    }).sort_values("Rang WSM", kind="stable").reset_index(drop=True)


def sensitivity(
    problem: DecisionProblem, weights: ArrayLike, method: str, criterion: int,
    variations: ArrayLike = (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3),
) -> pd.DataFrame:
    weights = validate_weights(weights, len(problem.criteria))
    if method not in ("WSM", "TOPSIS") or not 0 <= criterion < len(weights):
        raise ValueError("La méthode ou le critère de sensibilité est invalide.")
    variations = np.asarray(variations, dtype=float)
    if variations.ndim != 1 or not np.isfinite(variations).all() or (variations < -1).any():
        raise ValueError("Les variations doivent être finies et supérieures ou égales à −100 %.")
    model = WSM() if method == "WSM" else TOPSIS()
    rows = []
    for variation in variations:
        adjusted = weights.copy()
        adjusted[criterion] *= 1 + variation
        if adjusted.sum() == 0:
            raise ValueError("La variation annule tous les poids.")
        adjusted /= adjusted.sum()
        result = model.compute(problem, adjusted)
        for i, name in enumerate(problem.alternatives):
            rows.append({
                "Variation (%)": float(variation * 100), "Alternative": name,
                "Rang": int(result.ranks[i]), "Score": float(result.scores[i]),
                "Poids du critère": float(adjusted[criterion]),
            })
    return pd.DataFrame(rows)
