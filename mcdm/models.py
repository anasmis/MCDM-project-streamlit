from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.stats import rankdata

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Criterion:
    name: str
    direction: Literal["max", "min"] = "max"
    unit: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Chaque critère doit avoir un nom.")
        object.__setattr__(self, "name", self.name.strip())
        if self.direction not in ("max", "min"):
            raise ValueError("Le sens d'un critère doit être « max » ou « min ».")
        if not isinstance(self.unit, str):
            raise ValueError("L'unité d'un critère doit être un texte.")


@dataclass(frozen=True)
class DecisionProblem:
    name: str
    alternatives: tuple[str, ...]
    criteria: tuple[Criterion, ...]
    matrix: FloatArray
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Donnez un nom au problème.")
        if not isinstance(self.description, str):
            raise ValueError("La description doit être un texte.")
        if not all(isinstance(a, str) and a.strip() for a in self.alternatives):
            raise ValueError("Chaque alternative doit avoir un nom.")
        alternatives = tuple(a.strip() for a in self.alternatives)
        criteria = tuple(self.criteria)
        if len(alternatives) < 2 or len(criteria) < 2:
            raise ValueError("Saisissez au moins deux alternatives et deux critères.")
        if len(alternatives) > 500 or len(criteria) > 15:
            raise ValueError("Un problème peut contenir jusqu'à 500 alternatives et 15 critères.")
        if len(set(alternatives)) != len(alternatives):
            raise ValueError("Les noms des alternatives doivent être uniques.")
        if not all(isinstance(c, Criterion) for c in criteria):
            raise ValueError("La liste des critères est invalide.")
        names = [c.name for c in criteria]
        if len(set(names)) != len(names):
            raise ValueError("Les noms des critères doivent être uniques.")
        if "Alternative" in names:
            raise ValueError("« Alternative » est réservé aux noms des alternatives.")
        try:
            matrix = np.array(self.matrix, dtype=float, copy=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("Toutes les performances doivent être numériques.") from exc
        if matrix.shape != (len(alternatives), len(criteria)):
            raise ValueError("La matrice ne correspond pas aux alternatives et aux critères.")
        if not np.isfinite(matrix).all():
            raise ValueError("Complétez chaque performance avec un nombre fini.")
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "alternatives", alternatives)
        object.__setattr__(self, "criteria", criteria)
        object.__setattr__(self, "name", self.name.strip())

    @property
    def criterion_names(self) -> list[str]:
        return [criterion.name for criterion in self.criteria]

    @property
    def benefits(self) -> NDArray[np.bool_]:
        return np.array([criterion.direction == "max" for criterion in self.criteria])

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.matrix, index=self.alternatives, columns=self.criterion_names)


@dataclass
class WeightingResult:
    method: str
    weights: FloatArray
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class RankingResult:
    method: str
    scores: FloatArray
    normalized_matrix: FloatArray
    weighted_matrix: FloatArray
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def ranks(self) -> NDArray[np.int64]:
        return rankdata(-np.round(self.scores, 12), method="min").astype(np.int64)

    def to_frame(self, alternatives: tuple[str, ...]) -> pd.DataFrame:
        return (
            pd.DataFrame(
                {
                    "Rang": self.ranks,
                    "Alternative": alternatives,
                    "Score": self.scores,
                }
            )
            .sort_values("Rang", kind="stable")
            .reset_index(drop=True)
        )


def validate_weights(weights: ArrayLike, count: int) -> FloatArray:
    values = np.asarray(weights, dtype=float)
    if values.shape != (count,) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Fournissez un poids fini et positif ou nul par critère.")
    if not np.isclose(values.sum(), 1.0, rtol=0, atol=1e-8):
        raise ValueError("La somme des poids doit être égale à 1.")
    return values / values.sum()
