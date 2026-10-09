import pytest
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HOME_PATH = str(PROJECT_ROOT / "app" / "Home.py")
PLANNER_PATH = str(PROJECT_ROOT / "app" / "pages" / "1_Planner.py")


def test_home_page_runs():
    """Verify app/Home.py runs without exceptions."""
    at = AppTest.from_file(HOME_PATH).run(timeout=30)
    assert not at.exception
    # Check for "Config Version"
    assert any("Config Version" in getattr(m, "value", "") for m in at.markdown)


@patch("streamlit_folium.st_folium", return_value={"last_clicked": None, "last_object_clicked": None})
def test_planner_canonical_time(mock_folium):
    """Planner at default canonical time (13:00) shows computed badge, mode label, disclaimer, and no 'Interpolated' badge."""
    at = AppTest.from_file(PLANNER_PATH).run(timeout=30)
    assert not at.exception
    
    # Computed badge
    assert any("Mode: Computed (Canonical)" in getattr(s, "value", "") for s in at.success)
    # Ensure no Interpolated badge
    assert not any("Interpolated" in getattr(i, "value", "") for i in at.info)
    
    # Exposure mode label
    all_md = " ".join(getattr(m, "value", "") for m in at.markdown)
    assert "Exposure Mode:" in all_md
    
    # Disclaimer
    all_cap = " ".join(getattr(c, "value", "") for c in at.caption)
    assert "Disclaimer" in all_cap
    
    assert mock_folium.called


@patch("streamlit_folium.st_folium", return_value={"last_clicked": None, "last_object_clicked": None})
def test_planner_interpolated_time(mock_folium):
    """Planner at 10:00 (600 min) shows interpolated badge and correct slider configuration."""
    at = AppTest.from_file(PLANNER_PATH).run(timeout=30)
    
    # Set to 10:00 (600 mins)
    at.slider[0].set_value(600).run(timeout=30)
    assert not at.exception
    
    # Interpolated badge
    assert any("Mode: Interpolated (estimated)" in getattr(i, "value", "") for i in at.info)
    
    # Slider configuration
    slider = at.slider[0]
    assert getattr(slider, "min", None) == 540
    assert getattr(slider, "max", None) == 1020
    assert getattr(slider, "step", None) == 30
    
    assert mock_folium.called


@patch("streamlit_folium.st_folium", return_value={"last_clicked": None, "last_object_clicked": None})
def test_planner_why_panel_selectbox(mock_folium):
    """WHY panel renders details after a selectbox selection."""
    at = AppTest.from_file(PLANNER_PATH).run(timeout=30)
    assert not at.exception
    
    options = at.selectbox[0].options
    assert len(options) > 1, f"Expected >1 options, got {len(options)}"
    
    # Select second option
    at.selectbox[0].select(options[1]).run(timeout=30)
    assert not at.exception
    
    # Check street detail
    assert any("Street Detail" in getattr(s, "value", "") for s in at.subheader)
    
    # Check risk score metric
    assert any("Risk Score" in getattr(m, "label", "") for m in at.metric)
    
    # Check explanation markdown
    all_md = " ".join(getattr(m, "value", "") for m in at.markdown)
    assert "Explanation" in all_md
    assert "Risk Drivers" in all_md
    
    # Expander
    assert len(at.expander) > 0


@patch("streamlit_folium.st_folium", return_value={"last_clicked": None, "last_object_clicked": None})
def test_planner_why_not_hottest_tab(mock_folium):
    """Why-not-the-hottest renders two street IDs or the 'same street' message."""
    at = AppTest.from_file(PLANNER_PATH).run(timeout=30)
    assert not at.exception
    
    # The comparison text logic is in tab 3, but AppTest flattens elements.
    all_md = " ".join(getattr(m, "value", "") for m in at.markdown)
    all_info = " ".join(getattr(i, "value", "") for i in at.info)
    all_success = " ".join(getattr(s, "value", "") for s in at.success)
    all_text = all_md + " " + all_info + " " + all_success
    
    # Either same street message or different streets
    same_street_msg = "is both the street with the highest heat exposure and the highest overall risk score"
    diff_street_msg = "whereas the street with the highest overall risk is"
    
    assert same_street_msg in all_text or diff_street_msg in all_text, "Comparison sentence not found."

@patch("src.optimizer.run_optimizer", return_value=([{'candidate_id': 'INV-1', 'type': 'shade', 'target_id': 's1', 'prov_status': 'MODELLED'}], {'risk_records': {}, 'cooling_access': None}, {'objective_before': 100, 'objective_after': 90, 'runtime_s': 0.1}))
@patch("src.impact_engine.compute_impact", return_value={'vuln_weighted_exposure': {'before': 100, 'after': 90, 'delta': -10}, 'cooling_coverage': {'before': 0, 'after': 1, 'delta': 1}, 'avg_cooling_dist_m': {'before': 1000, 'after': 900, 'delta': -100}})
@patch("streamlit_folium.st_folium", return_value={"last_clicked": None, "last_object_clicked": None})
def test_planner_optimizer_tab(mock_folium, mock_compute_impact, mock_run_opt):
    """Optimize -> plan renders -> Before/After shows 'Modelled Impact' -> moving slider doesn't re-run."""
    at = AppTest.from_file(PLANNER_PATH).run(timeout=30)
    assert not at.exception
    
    # Click Optimize button
    at.button[0].click().run(timeout=30)
    assert not at.exception
    
    # Optimizer should have been called once
    assert mock_run_opt.call_count == 1
    
    # Check for Modelled Impact label
    all_cap = " ".join(getattr(c, "value", "") for c in at.caption)
    assert "Modelled Impact" in all_cap
    
    # Move slider
    at.slider[0].set_value(600).run(timeout=30)
    assert not at.exception
    
    # Optimizer should NOT be called again
    assert mock_run_opt.call_count == 1
