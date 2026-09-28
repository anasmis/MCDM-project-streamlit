import numpy as np
from numpy.typing import ArrayLike

from mcdm.models import FloatArray, WeightingResult


class AHP:
    RANDOM_INDEX = {
        1: 0.0,
        2: 0.0,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45,
        10: 1.49,
        11: 1.51,
        12: 1.48,
        13: 1.56,
        14: 1.57,
        15: 1.59,
    }

    def validate(self, comparisons: ArrayLike) -> FloatArray:
        matrix = np.asarray(comparisons, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("La matrice AHP doit être carrée.")
        if not 2 <= len(matrix) <= 15:
            raise ValueError("AHP accepte entre 2 et 15 critères.")
        if not np.isfinite(matrix).all() or (matrix < 1 / 9 - 1e-10).any():
            raise ValueError("Les comparaisons AHP doivent être comprises entre 1/9 et 9.")
        if (matrix > 9 + 1e-10).any():
            raise ValueError("Les comparaisons AHP doivent être comprises entre 1/9 et 9.")
        if not np.allclose(np.diag(matrix), 1) or not np.allclose(matrix * matrix.T, 1):
            raise ValueError("La matrice AHP doit être réciproque, avec une diagonale de 1.")
        return matrix

    def consistency_ratio(self, eigenvalue: float, size: int) -> tuple[float, float]:
        index = max(0.0, (eigenvalue - size) / (size - 1))
        random_index = self.RANDOM_INDEX[size]
        return index, index / random_index if random_index else 0.0

    def compute(self, comparisons: ArrayLike) -> WeightingResult:
        matrix = self.validate(comparisons)
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        index = int(np.argmax(eigenvalues.real))
        weights = np.abs(eigenvectors[:, index].real)
        weights /= weights.sum()
        eigenvalue = float(eigenvalues[index].real)
        ci, cr = self.consistency_ratio(eigenvalue, len(matrix))
        warnings = []
        if cr > 0.10:
            warnings.append(
                "Le ratio de cohérence dépasse 10 %. Revoyez les comparaisons "
                "ou confirmez explicitement leur utilisation."
            )
        return WeightingResult(
            "AHP",
            weights,
            {
                "comparisons": matrix,
                "lambda_max": eigenvalue,
                "consistency_index": ci,
                "consistency_ratio": cr,
            },
            warnings,
        )
