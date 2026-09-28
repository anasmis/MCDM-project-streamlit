import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linprog

from mcdm.models import FloatArray, WeightingResult


class BWM:
    def validate(
        self,
        best: int,
        worst: int,
        best_to_others: ArrayLike,
        others_to_worst: ArrayLike,
    ) -> tuple[FloatArray, FloatArray]:
        ab = np.asarray(best_to_others, dtype=float)
        aw = np.asarray(others_to_worst, dtype=float)
        if ab.ndim != 1 or len(ab) < 2 or aw.shape != ab.shape:
            raise ValueError("Fournissez deux vecteurs de comparaison de même taille.")
        if not isinstance(best, (int, np.integer)) or not isinstance(worst, (int, np.integer)):
            raise ValueError("Les indices des critères doivent être entiers.")
        if best == worst or not (0 <= best < len(ab) and 0 <= worst < len(ab)):
            raise ValueError(
                "Le meilleur et le moins important des critères doivent être distincts."
            )
        if any(not np.isfinite(v).all() or (v < 1).any() or (v > 9).any() for v in (ab, aw)):
            raise ValueError("Les comparaisons BWM doivent être comprises entre 1 et 9.")
        if not np.isclose(ab[best], 1) or not np.isclose(aw[worst], 1):
            raise ValueError("La comparaison d'un critère avec lui-même doit valoir 1.")
        if not np.isclose(ab[worst], aw[best]):
            raise ValueError("La comparaison meilleur / moins important doit être identique.")
        return ab, aw

    def compute(
        self,
        best: int,
        worst: int,
        best_to_others: ArrayLike,
        others_to_worst: ArrayLike,
    ) -> WeightingResult:
        ab, aw = self.validate(best, worst, best_to_others, others_to_worst)
        size = len(ab)
        constraints = []
        for j in range(size):
            first = np.zeros(size + 1)
            first[best] += 1
            first[j] -= ab[j]
            second = np.zeros(size + 1)
            second[j] += 1
            second[worst] -= aw[j]
            for row in (first, second):
                for sign in (1, -1):
                    bound = row * sign
                    bound[-1] = -1
                    constraints.append(bound)
        solution = linprog(
            c=np.r_[np.zeros(size), 1.0],
            A_ub=np.array(constraints),
            b_ub=np.zeros(len(constraints)),
            A_eq=[np.r_[np.ones(size), 0.0]],
            b_eq=[1.0],
            bounds=[(0, None)] * (size + 1),
            method="highs",
        )
        if not solution.success:
            raise ValueError("Le calcul BWM n'a pas convergé. Revoyez vos comparaisons.")
        weights = solution.x[:-1]
        weights /= weights.sum()
        residual = float(solution.x[-1])
        return WeightingResult(
            "BWM",
            weights,
            {
                "best": best,
                "worst": worst,
                "best_to_others": ab,
                "others_to_worst": aw,
                "residual": residual,
            },
        )
