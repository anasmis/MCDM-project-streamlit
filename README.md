# Decision Lab — crisp and fuzzy MCDM

A general Streamlit workspace for comparing alternatives in any domain using numeric or triangular fuzzy performances, benefit/cost criteria, selectable methods, and sensitivity analysis.

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The root entry point launches the current application in [`mcdm_streamlit_app`](mcdm_streamlit_app). See its [full documentation](mcdm_streamlit_app/README.md) for methods, input formats, mathematical conventions, examples, and tests.

- **Nine crisp ranking methods:** TOPSIS, SAW/WSM, WPM, WASPAS, VIKOR, MOORA, COPRAS, EDAS, PROMETHEE II.
- **Fuzzy rankings:** triangular fuzzy TOPSIS and fuzzy SAW.
- **Weighting:** equal, manual, entropy, CRITIC, AHP, linear BWM, fuzzy manual and fuzzy AHP.
- **Workflow:** edit or import a problem, compare methods, inspect calculation details, vary weights, and export a complete project or results archive.

This supports finite decision matrices. Each method has its own assumptions; it does not claim to solve every possible MCDM formulation. The previous inventory-specific application is preserved as `mcdm_streamlit_app/legacy_inventory_app.py`.
