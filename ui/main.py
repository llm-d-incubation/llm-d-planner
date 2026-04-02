"""
Planner entrypoint — defines page navigation with friendly titles.
Run with: streamlit run ui/main.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st

st.set_page_config(
    page_title="Planner",
    page_icon="docs/planner-logo-32.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

pg = st.navigation(
    [
        st.Page("app.py", title="Planner"),
        st.Page("pages/1_Capacity_Planner.py", title="Capacity Planner"),
        st.Page("pages/2_GPU_Recommender.py", title="GPU Recommender"),
    ]
)
pg.run()
