# Decision Lab

Domain-independent multicriteria decision analysis with a Streamlit UI and a separately testable calculation engine. Define any number of alternatives and criteria, set benefit/cost directions and units, enter numeric or triangular fuzzy performances, and select the methods appropriate to your decision.

## Run

From this directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Python 3.11 or newer is recommended. From the repository root, the same install and run commands work through the root launcher. For Streamlit Community Cloud, use root `app.py` as the entry point and root `requirements.txt` for dependencies.

The inventory workflow, including ABC classification, machine learning and the optional Jev integration, is preserved separately:

```powershell
python -m streamlit run legacy_inventory_app.py
```

## Workflow

1. Load a supplier, location, fuzzy assessment or blank example; alternatively import CSV or project JSON.
2. Add, remove and rename criteria and alternatives in the editors. Set each criterion to **Benefit** (maximize) or **Cost** (minimize). Set optional units and manual weights.
3. Choose **Crisp** or **Fuzzy**. Fuzzy cells accept a number, a linguistic term from the editable scale, or a triangle such as `2;3;5`.
4. Choose a weighting method and supply comparisons if needed.
5. Select one or more ranking methods. Inspect scores, tied ranks and intermediate calculations; compare ranks and Spearman correlations.
6. Run weight sensitivity analysis. Save the project JSON or download inputs and a ZIP containing project, weights, rankings and calculation settings.

JSON preserves the problem context, directions, units, matrix, linguistic scale, weighting judgments and ranking parameters. CSV contains only the matrix; imported directions default to **Benefit** and must be checked. Edits stay in the active Streamlit session; download JSON to retain them across browser sessions.

## Input formats

CSV starts with a label column followed by criteria. Choose comma, semicolon or tab delimiter, and dot or comma decimal separator. Quoted cells can contain triangular values. UTF-8 is supported. Names must be unique and performances must be complete and finite; invalid values are not silently filled or dropped.

```csv
Alternative,Price,Quality,Delivery days
Supplier A,120,8,5
Supplier B,100,7,8
Supplier C,140,9,4
```

After import, set Price and Delivery days to **Cost**. Example JSON files in `examples/` preserve these directions.

Fuzzy input example:

```csv
Alternative,Quality,Effort,Resilience
Option A,High,Medium,5;7;9
Option B,Medium,Low,High
Option C,Very high,High,Very high
```

Switch to Fuzzy and set Effort to Cost. Terms describe performance magnitudes: “High” effort is still undesirable when effort is a cost. Supply a custom linguistic scale to encode qualitative terms specific to your domain. A triangle must satisfy `lower <= modal <= upper`. A number becomes `(x,x,x)`. These are possibility assessments, not probability distributions.

## Ranking methods and conventions

| Method | Calculation and domain |
| --- | --- |
| TOPSIS | Vector normalization, criterion weights, observed benefit/cost ideals, Euclidean distances and relative closeness. Signed values allowed. Coincident ideals yield 0.5. |
| SAW / WSM | Weighted sum of oriented min–max values. Signed values allowed; constant criteria map to 1. |
| WPM | Weighted geometric product, with benefit ratios `x/max(x)` and cost ratios `min(x)/x`. Strictly positive active values required. |
| WASPAS | `lambda * ratio-weighted sum + (1-lambda) * WPM`. Lambda is configurable. Positive active values required. |
| VIKOR | Oriented min–max regret, group utility S, maximum individual regret R and compromise Q with configurable v. **Lower Q is better.** Constant criteria contribute no regret. Acceptable advantage and stability checks are displayed; a failed check means the leading rank is not a unique compromise recommendation. |
| MOORA | Vector normalization; weighted benefit sum minus weighted cost sum. Signed values allowed. |
| COPRAS | Column-sum normalization, benefit sum and reciprocal cost adjustment; relative utility divided by its maximum. Positive active values required. |
| EDAS | Weighted positive/negative distances from the criterion mean and normalized appraisal. Positive active values required. A completely tied matrix yields 0.5. |
| PROMETHEE II | Usual strict-preference function with net outranking flows. No preference threshold is assumed. Signed values allowed; equal performances produce no preference. |
| Fuzzy TOPSIS | Benefit `triangle/max(upper)`; cost `min(lower)/(upper,modal,lower)`. Multiply triangular performances and weights. Per criterion, the positive ideal repeats the maximum weighted upper endpoint; the negative ideal repeats the minimum lower endpoint. Sum vertex distances across criteria and compute relative closeness. No early defuzzification. |
| Fuzzy SAW | Same fuzzy ratio normalization and multiplication; sum triangular weighted performances and rank their centroids. |

Both fuzzy methods require nonnegative triangles and strictly positive lower endpoints for cost criteria. All-zero benefit columns are constant and normalize to `(1,1,1)`. Fuzzy TOPSIS uses an observed-endpoint ideal variant; its degenerate-triangle limit uses summed absolute distances and can differ from Euclidean crisp TOPSIS. The implementation identifies this convention explicitly rather than treating different variants as interchangeable.

Zero-weight criteria are removed before domain checks and calculations. Scores have method-specific scales; compare **ranks**, not raw scores. Scores rounded to 12 decimals share competition ranks (`1,1,3`). All ties are retained in the displayed winners. Mean rank is only an agreement summary, not an independent decision method.

## Weighting

- **Equal / manual:** nonnegative weights with positive total, normalized to sum to one.
- **Entropy:** benefit-oriented min–max normalization, proportions, Shannon entropy and normalized divergence. `0 log(0)=0`. Constant criteria receive zero weight.
- **CRITIC:** benefit-oriented min–max values, population standard deviations and intercriterion correlation conflict. Constant criteria receive zero weight. A single variable criterion receives all weight. No information or perfectly correlated criteria produce a clear message requesting a different weighting method.
- **AHP:** principal eigenvector of a reciprocal pairwise matrix. The UI generates reciprocals from upper-triangle judgments. CR uses Saaty random indices through 10 criteria; above 10, it reports CR as unavailable. CR greater than 0.10 is flagged.
- **BWM:** linear minimax model for best-to-others and others-to-worst judgments on 1–9. Self-comparisons and the shared best/worst comparison are enforced. The displayed optimum is a linear optimization residual, not an AHP CR.
- **Fuzzy manual:** nonnegative triangular weights. A common positive scaling normalizes the modal total while preserving shape; displayed crisp importance uses normalized centroids.
- **Fuzzy AHP:** Buckley row geometric means and triangular inverse-sum normalization, with centroid weights for display. Full triangular weights feed fuzzy ranking. CR is a diagnostic of the modal comparison matrix, not a fuzzy consistency ratio.

In fuzzy mode, entropy and CRITIC use centroids to derive crisp weights; all endpoints remain in the ranking calculation. Sensitivity redistributes other weights proportionally as one weight moves from 0 to 1; for triangular weights it preserves each existing triangle's relative shape. If all other weights were zero, it redistributes equally.

## Scope

The solver covers finite, complete alternative-by-criterion matrices. A target can be encoded as a distance-to-target cost criterion. This is not a universal solver for continuous multiobjective optimization, portfolio constraints, criteria dependencies/ANP, all outranking variants, other fuzzy set types, or rule-based Mamdani/Sugeno inference. Choose a formulation and method appropriate to your problem; the app states assumptions and reports incompatible inputs per method.

## Development and verification

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check app.py decisionlab tests
python -m ruff format --check app.py decisionlab tests
```

If unrelated globally installed pytest plugins fail to load, use a fresh virtual environment or set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. No optional pytest plugin is needed.

Tests cover hand-calculated examples, published VIKOR/EDAS fixtures, direction handling, ties, constant criteria, zero weights, scale invariance, invalid data, AHP/BWM, triangular arithmetic, uncertainty effects, JSON/CSV round trips, all app weighting choices, sensitivity, templates and the repository launcher.

The calculation engine is in `decisionlab/`; `app.py` contains the UI. The legacy app and supplied inventory CSVs are retained for the original workflow.

## References

- [pymcdm primary method documentation and numerical examples](https://pymcdm.readthedocs.io/en/master/pymcdm.methods.html).
- [VIKOR reference implementation and fixture](https://github.com/kotbaton/pymcdm/blob/master/pymcdm/methods/vikor.py).
- [EDAS reference implementation and fixture](https://github.com/kotbaton/pymcdm/blob/master/pymcdm/methods/edas.py).
- [Triangular fuzzy arithmetic and vertex distance](https://www.mdpi.com/2071-1050/13/6/3020), Appendix B.
- Buckley, J. J. (1985), *Fuzzy hierarchical analysis*, Fuzzy Sets and Systems 17(3), 233–247, [DOI](https://doi.org/10.1016/0165-0114(85)90090-9).
