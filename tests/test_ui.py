import pytest
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).parent.parent
HOME_PATH = str(PROJECT_ROOT / "app" / "Home.py")
PLANNER_PATH = str(PROJECT_ROOT / "app" / "pages" / "1_Planner.py")


def test_home_page_runs():
    """Verify app/Home.py runs without exceptions."""
    at = AppTest.from_file(HOME_PATH).run()
    assert not at.exception


@patch("streamlit_folium.st_folium", return_value=None)
def test_planner_canonical_time(mock_folium):
    """Verify Planner runs at a canonical time (default 13:00 / 780 min)."""
    at = AppTest.from_file(PLANNER_PATH).run()
    assert not at.exception


@patch("streamlit_folium.st_folium", return_value=None)
def test_planner_interpolated_time(mock_folium):
    """Verify Planner runs at 10:00 (600 min, interpolated time)."""
    at = AppTest.from_file(PLANNER_PATH).run()
    at.slider[0].set_value(600).run()
    assert not at.exception


@patch("streamlit_folium.st_folium", return_value=None)
def test_planner_why_panel_selectbox(mock_folium):
    """Verify WHY panel renders after a selectbox choice."""
    at = AppTest.from_file(PLANNER_PATH).run()
    assert not at.exception
    options = at.selectbox[0].options
    if len(options) > 1:
        at.selectbox[0].select(options[1]).run()
        assert not at.exception

