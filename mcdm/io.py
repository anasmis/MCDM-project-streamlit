import csv
import io
import json
import zipfile
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from mcdm.analysis import compare_rankings
from mcdm.models import Criterion, DecisionProblem, RankingResult, WeightingResult


def example_problem() -> DecisionProblem:
    return DecisionProblem(
        "Choisir un fournisseur",
        ("Atlas", "Boréal", "Cèdre", "Dune", "Estuaire"),
        (
            Criterion("Coût", "min", "k€"),
            Criterion("Qualité", "max", "/100"),
            Criterion("Délai", "min", "jours"),
            Criterion("Responsabilité", "max", "/100"),
        ),
        np.array(
            [
                [42, 86, 12, 75],
                [38, 78, 16, 88],
                [47, 94, 10, 82],
                [35, 72, 20, 65],
                [44, 89, 14, 93],
            ],
            dtype=float,
        ),
        "Sélectionner un partenaire pour le prochain contrat d'approvisionnement.",
    )


def problem_to_json(problem: DecisionProblem) -> str:
    return json.dumps(
        {
            "version": 1,
            "name": problem.name,
            "description": problem.description,
            "alternatives": problem.alternatives,
            "criteria": [asdict(c) for c in problem.criteria],
            "matrix": problem.matrix.tolist(),
        },
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )


def problem_from_json(content: str | bytes) -> DecisionProblem:
    try:
        data = json.loads(content)
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("Version du fichier de problème non reconnue.")
        return DecisionProblem(
            name=data["name"],
            description=data.get("description", ""),
            alternatives=tuple(data["alternatives"]),
            criteria=tuple(Criterion(**c) for c in data["criteria"]),
            matrix=data["matrix"],
        )
    except (KeyError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Le fichier JSON ne contient pas un problème valide.") from exc


def matrix_from_csv(content: bytes) -> pd.DataFrame:
    try:
        text = content.decode("utf-8-sig")
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t")
        headers = next(csv.reader(io.StringIO(text), dialect=dialect))
        headers = [header.strip() for header in headers]
        if len(set(headers)) != len(headers) or not all(headers):
            raise ValueError("Les noms des colonnes du CSV doivent être renseignés et uniques.")
        frame = pd.read_csv(io.StringIO(text), sep=dialect.delimiter, dtype=str)
        if frame.shape[1] < 3 or not 2 <= len(frame) <= 500 or frame.shape[1] > 16:
            raise ValueError("Le CSV doit contenir 2 à 500 alternatives et 2 à 15 critères.")
        frame.columns = ["Alternative", *headers[1:]]
        if "Alternative" in headers[1:]:
            raise ValueError("« Alternative » est réservé à la première colonne.")
        for column in frame.columns[1:]:
            frame[column] = pd.to_numeric(
                frame[column].str.strip().str.replace(",", ".", regex=False),
                errors="raise",
            )
        if not np.isfinite(frame.iloc[:, 1:].to_numpy(dtype=float)).all():
            raise ValueError("Le CSV contient une performance vide ou non finie.")
        if frame["Alternative"].isna().any():
            raise ValueError("Chaque alternative du CSV doit avoir un nom.")
        frame["Alternative"] = frame["Alternative"].str.strip()
        if frame["Alternative"].eq("").any() or frame["Alternative"].duplicated().any():
            raise ValueError("Les noms des alternatives doivent être renseignés et uniques.")
        return frame
    except (UnicodeDecodeError, csv.Error, pd.errors.ParserError) as exc:
        raise ValueError(
            "Utilisez un CSV UTF-8 séparé par des virgules, points-virgules ou tabulations."
        ) from exc
    except ValueError as exc:
        if "Unable to parse" in str(exc):
            raise ValueError("Toutes les performances du CSV doivent être numériques.") from exc
        raise


def csv_bytes(frame: pd.DataFrame) -> bytes:
    safe = frame.copy()
    for column in safe.select_dtypes(include=["object", "string"]).columns:
        safe[column] = safe[column].map(
            lambda value: (
                "'" + value
                if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r"))
                else value
            )
        )
    return safe.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Type non sérialisable : {type(value).__name__}")


def export_bundle(
    problem: DecisionProblem,
    weighting: WeightingResult,
    ranking: RankingResult,
) -> bytes:
    output = io.BytesIO()
    weights = pd.DataFrame({"Critère": problem.criterion_names, "Poids": weighting.weights})
    table = ranking.to_frame(problem.alternatives)
    winners = table.loc[table["Rang"] == 1, "Alternative"].tolist()
    report = (
        f"# {problem.name}\n\n{problem.description}\n\n"
        f"Pondération : {weighting.method}. Classement : {ranking.method}.\n\n"
        f"Premier rang : {', '.join(winners)}.\n\n"
        "Les scores sont relatifs aux alternatives, aux critères et aux conventions choisies.\n"
        "WSM : normalisation min–max orientée, critères constants à 1.\n"
        "TOPSIS : normalisation vectorielle, idéaux selon le sens de chaque critère.\n"
        "Entropie : proportions après normalisation min–max orientée.\n"
        "CRITIC : corrélations entre les seuls critères variables.\n"
        "BWM : formulation linéaire ; le résidu n'est pas le ratio de cohérence AHP.\n\n"
        + "\n".join(f"Attention : {message}" for message in weighting.warnings)
    )
    audit = {"weighting": asdict(weighting), "ranking": asdict(ranking)}
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("probleme.json", problem_to_json(problem))
        archive.writestr(
            "performances.csv",
            csv_bytes(problem.to_frame().rename_axis("Alternative").reset_index()),
        )
        archive.writestr("poids.csv", csv_bytes(weights))
        archive.writestr("classement.csv", csv_bytes(table))
        archive.writestr("comparaison.csv", csv_bytes(compare_rankings(problem, weighting.weights)))
        archive.writestr(
            "calculs.json", json.dumps(audit, default=_json_default, ensure_ascii=False, indent=2)
        )
        archive.writestr("synthese.md", report)
    return output.getvalue()
