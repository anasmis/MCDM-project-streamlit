"""Portable problem definitions and CSV imports, without inventory assumptions."""

import io
import json

import numpy as np
import pandas as pd

from .fuzzy import DEFAULT_SCALE, parse_cell


def example(name="Supplier selection"):
    if name == "Location planning":
        names = ["North", "Central", "South", "West"]
        columns = ["Rent", "Access", "Capacity"]
        values = [[1200, 8, 180], [1700, 9, 220], [900, 6, 140], [1350, 7, 210]]
        directions = ["Cost", "Benefit", "Benefit"]
    elif name == "Fuzzy assessment":
        names = ["Option A", "Option B", "Option C"]
        columns = ["Quality", "Effort", "Resilience"]
        values = [
            ["High", "Medium", "5;7;9"],
            ["Medium", "Low", "High"],
            ["Very high", "High", "Very high"],
        ]
        directions = ["Benefit", "Cost", "Benefit"]
    elif name == "Blank problem":
        names, columns = (
            ["Option A", "Option B", "Option C"],
            ["Criterion 1", "Criterion 2", "Criterion 3"],
        )
        values, directions = [[0, 0, 0] for _ in names], ["Benefit"] * 3
    else:
        names = ["Supplier A", "Supplier B", "Supplier C", "Supplier D"]
        columns = ["Price", "Quality", "Delivery days", "Sustainability"]
        values = [[120, 8, 5, 7], [100, 7, 8, 8], [140, 9, 4, 9], [110, 8, 6, 6]]
        directions = ["Cost", "Benefit", "Cost", "Benefit"]
    return {
        "version": 1,
        "title": name,
        "description": "",
        "mode": "Fuzzy" if name == "Fuzzy assessment" else "Crisp",
        "alternatives": names,
        "criteria": [
            {"name": c, "direction": d, "weight": 1.0, "unit": ""}
            for c, d in zip(columns, directions, strict=False)
        ],
        "values": [[str(v) for v in row] for row in values],
        "scale": {k: list(v) for k, v in DEFAULT_SCALE.items()},
        "settings": {},
    }


def validate_project(p):
    if not isinstance(p, dict) or p.get("version") != 1:
        raise ValueError("Unsupported project format. Use a version 1 Decision Lab JSON file.")
    for key in ["alternatives", "criteria", "values"]:
        if not isinstance(p.get(key), list) or not p[key]:
            raise ValueError(f"Project '{key}' must be a nonempty list.")
    names = p["alternatives"]
    if any(not isinstance(n, str) or not n.strip() for n in names) or len(
        set(n.strip() for n in names)
    ) != len(names):
        raise ValueError("Alternative names must be nonempty and unique.")
    columns = []
    for c in p["criteria"]:
        if not isinstance(c, dict) or not isinstance(c.get("name"), str) or not c["name"].strip():
            raise ValueError("Each criterion needs a nonempty name.")
        columns.append(c["name"].strip())
        if c.get("direction") not in ["Benefit", "Cost"]:
            raise ValueError("Criterion directions must be Benefit or Cost.")
        try:
            weight = float(c.get("weight", 1))
        except (TypeError, ValueError):
            raise ValueError("Criterion weights must be finite and nonnegative.") from None
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("Criterion weights must be finite and nonnegative.")
        c["weight"] = weight
        c.setdefault("unit", "")
        if not isinstance(c["unit"], str):
            raise ValueError("Criterion units must be text.")
    if len(set(columns)) != len(columns) or "Alternative" in columns:
        raise ValueError(
            "Criterion names must be unique; 'Alternative' is reserved for row labels."
        )
    if len(p["values"]) != len(names) or any(
        not isinstance(row, list) or len(row) != len(columns) for row in p["values"]
    ):
        raise ValueError(
            "The decision matrix dimensions do not match the alternatives and criteria."
        )
    if p.get("mode", "Crisp") not in ["Crisp", "Fuzzy"]:
        raise ValueError("Mode must be Crisp or Fuzzy.")
    for key in ["title", "description"]:
        if not isinstance(p.get(key, ""), str):
            raise ValueError(f"Project {key} must be text.")
    scale = p.get("scale", DEFAULT_SCALE)
    if not isinstance(scale, dict) or any(not isinstance(k, str) or not k.strip() for k in scale):
        raise ValueError("The linguistic scale must map nonempty terms to triangles.")
    if len({k.strip().casefold() for k in scale}) != len(scale):
        raise ValueError("Linguistic terms must be unique, ignoring case.")
    for triangle in scale.values():
        parse_cell(triangle, {})
    if not isinstance(p.get("settings", {}), dict):
        raise ValueError("Project settings must be an object.")
    settings = p.get("settings", {})
    for parameter in ["v", "lambda"]:
        value = settings.get(parameter, 0.5)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Setting '{parameter}' must be between 0 and 1.")
    for key in ["pairwise", "fuzzy_pairwise", "fuzzy_weights", "bwm"]:
        if not isinstance(settings.get(key, {}), dict):
            raise ValueError(f"Setting '{key}' must be an object.")
    for key in ["methods_Crisp", "methods_Fuzzy"]:
        if key in settings and (
            not isinstance(settings[key], list)
            or any(not isinstance(m, str) for m in settings[key])
        ):
            raise ValueError("Selected methods must be a list of names.")
    bw = settings.get("bwm", {})
    if "shared" in bw and (
        not isinstance(bw["shared"], (int, float)) or not 1 <= bw["shared"] <= 9
    ):
        raise ValueError("BWM preference must be between 1 and 9.")
    for key in ["a", "c"]:
        if key in bw and (
            not isinstance(bw[key], list)
            or any(
                not isinstance(v, (int, float)) or not np.isfinite(v) or not 1 <= v <= 9
                for v in bw[key]
            )
        ):
            raise ValueError("BWM comparisons must be numeric values from 1 to 9.")
    for row in p["values"]:
        for value in row:
            if not isinstance(value, (str, int, float)):
                raise ValueError("Matrix cells must be numbers or text (for example, '2;3;5').")
    return p


def load_json(data):
    try:
        return validate_project(json.loads(data))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Invalid JSON project file.") from exc


def dump_json(project):
    validate_project(project)
    return json.dumps(project, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def load_csv(data, *, sep=",", decimal="."):
    try:
        frame = pd.read_csv(
            io.BytesIO(data), sep=sep, decimal=decimal, dtype=str, keep_default_na=False
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise ValueError("Could not read the CSV. Check encoding and delimiter.") from exc
    if frame.shape[1] < 2 or not len(frame):
        raise ValueError(
            "CSV needs a label column, at least one criterion column and one alternative row."
        )
    # Catch duplicate headers before pandas renames them.
    import csv

    header = next(csv.reader(io.StringIO(data.decode("utf-8-sig")), delimiter=sep))
    if len(set(s.strip() for s in header)) != len(header):
        raise ValueError("CSV column names must be unique.")
    p = example("Blank problem")
    p["title"] = "Imported decision problem"
    p["alternatives"] = frame.iloc[:, 0].astype(str).str.strip().tolist()
    p["criteria"] = [
        {"name": str(c).strip(), "direction": "Benefit", "weight": 1.0, "unit": ""}
        for c in frame.columns[1:]
    ]
    values = frame.iloc[:, 1:].astype(str)
    if decimal == ",":
        # Semicolons delimit TFN endpoints, so decimal commas remain unambiguous.
        values = values.map(lambda v: v.replace(",", "."))
    p["values"] = values.values.tolist()
    return validate_project(p)
