import numpy as np

from mcdm.models import DecisionProblem, WeightingResult
from mcdm.normalization import minmax


class Entropy:
    def compute(self, problem: DecisionProblem) -> WeightingResult:
        normalized, active = minmax(problem)
        if not active.any():
            raise ValueError(
                "Tous les critères sont constants : l'entropie ne peut pas les pondérer."
            )
        proportions = normalized / normalized.sum(axis=0)
        logarithms = np.zeros_like(proportions)
        np.log(proportions, out=logarithms, where=proportions > 0)
        entropy = -(proportions * logarithms).sum(axis=0) / np.log(len(problem.alternatives))
        divergence = np.maximum(0, 1 - entropy)
        divergence[~active] = 0
        warnings = []
        if not active.all():
            warnings.append("Les critères constants reçoivent un poids nul.")
        return WeightingResult(
            "Entropie",
            divergence / divergence.sum(),
            {
                "normalized_matrix": normalized,
                "proportions": proportions,
                "entropy": entropy,
                "divergence": divergence,
            },
            warnings,
        )
