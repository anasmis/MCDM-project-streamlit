"""Repository-level Streamlit entry point for Decision Lab."""
import runpy
import sys
from pathlib import Path

application = Path(__file__).resolve().parent / "mcdm_streamlit_app"
sys.path.insert(0, str(application))
runpy.run_path(str(application / "app.py"), run_name="__main__")
