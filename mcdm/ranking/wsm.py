from numpy.typing import ArrayLike

from mcdm.models import DecisionProblem, RankingResult, validate_weights
from mcdm.normalization import minmax


class WSM:
    def compute(self, problem: DecisionProblem, weights: ArrayLike) -> RankingResult:
        weights = validate_weights(weights, len(problem.criteria))
        normalized, _ = minmax(problem)
        weighted = normalized * weights
        return RankingResult("WSM", weighted.sum(axis=1), normalized, weighted)
