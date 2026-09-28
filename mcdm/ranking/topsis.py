import numpy as np
from numpy.typing import ArrayLike

from mcdm.models import DecisionProblem, RankingResult, validate_weights
from mcdm.normalization import vector


class TOPSIS:
    def compute(self, problem: DecisionProblem, weights: ArrayLike) -> RankingResult:
        weights = validate_weights(weights, len(problem.criteria))
        normalized = vector(problem)
        weighted = normalized * weights
        best = np.where(problem.benefits, weighted.max(axis=0), weighted.min(axis=0))
        worst = np.where(problem.benefits, weighted.min(axis=0), weighted.max(axis=0))
        distance_best = np.linalg.norm(weighted - best, axis=1)
        distance_worst = np.linalg.norm(weighted - worst, axis=1)
        total = distance_best + distance_worst
        scores = np.divide(distance_worst, total, out=np.full_like(total, 0.5), where=total > 0)
        return RankingResult(
            "TOPSIS",
            scores,
            normalized,
            weighted,
            {
                "ideal_best": best,
                "ideal_worst": worst,
                "distance_best": distance_best,
                "distance_worst": distance_worst,
            },
        )
