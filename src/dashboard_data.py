"""
Dashboard Data Layer interface for HeatShield X UI.
Delegates to central data_engine, risk_engine, and explain_engine.
"""
from src.data_engine import (
    load_dashboard_data,
    get_snapshot,
    nearest_canonical_segment,
    top_priority_streets,
    class_counts,
    get_segment_detail,
)
from src.explain_engine import compare_hottest_vs_highest_risk


def get_exposure_mode_label(mode=None):
    """Return human-readable label for exposure computation mode."""
    if mode is None:
        data = load_dashboard_data()
        mode = data.get('mode', 'geometric')
    if mode == 'geometric':
        return "Geometric Shadow Model (3D Building Projections)"
    elif mode == 'fallback' or mode == 'estimated_exposure_mode':
        return "Estimated Exposure Model (Static Building Buffers)"
    return f"Mode: {mode}"
