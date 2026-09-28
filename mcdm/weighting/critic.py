import numpy as np

from mcdm.models import DecisionProblem, WeightingResult
from mcdm.normalization import minmax


class CRITIC:
    def compute(self, problem: DecisionProblem) -> WeightingResult:
        normalized, active = minmax(problem)
        if not active.any():
            raise ValueError("Tous les critères sont constants : CRITIC ne peut pas les pondérer.")
        std = normalized.std(axis=0, ddof=0)
        correlation = np.eye(len(problem.criteria))
        information = np.zeros(len(problem.criteria))
        if active.sum() == 1:
            information[active] = std[active]
        else:
            active_correlation = np.clip(np.corrcoef(normalized[:, active], rowvar=False), -1, 1)
            correlation[np.ix_(active, active)] = active_correlation
            information[active] = std[active] * (1 - active_correlation).sum(axis=0)
        if information.sum() < 1e-12:
            raise ValueError(
                "Les critères variables sont parfaitement corrélés : CRITIC ne peut pas "
                "les départager. Essayez l'entropie ou une méthode subjective."
            )
        warnings = []
        if not active.all():
            warnings.append("Les critères constants reçoivent un poids nul.")
        return WeightingResult(
            "CRITIC",
            information / information.sum(),
            {
                "normalized_matrix": normalized,
                "standard_deviation": std,
                "correlation": correlation,
                "information": information,
            },
            warnings,
        )
