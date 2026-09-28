import numpy as np

from mcdm.models import DecisionProblem, FloatArray


def scaled_matrix(problem: DecisionProblem) -> FloatArray:
    scale = np.abs(problem.matrix).max(axis=0)
    return problem.matrix / np.where(scale == 0, 1, scale)


def minmax(problem: DecisionProblem) -> tuple[FloatArray, np.ndarray]:
    values = scaled_matrix(problem)
    low, high = values.min(axis=0), values.max(axis=0)
    span = high - low
    active = span > 0
    normalized = (values - low) / np.where(active, span, 1)
    normalized[:, ~problem.benefits] = 1 - normalized[:, ~problem.benefits]
    normalized[:, ~active] = 1.0
    return normalized, active


def vector(problem: DecisionProblem) -> FloatArray:
    values = scaled_matrix(problem)
    norms = np.linalg.norm(values, axis=0)
    return values / np.where(norms == 0, 1, norms)
