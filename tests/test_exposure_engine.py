"""
Tests for exposure_engine.py — comprehensive audit per Prompt 4 §0c.
"""
import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, LineString
from pathlib import Path
from src.exposure_engine import compute_exposure_formula, compute_exposure, get_exposure_mode_label


class TestExposureFormulaBounds:
    """exposure_value must be in [0, 1]."""

    def test_bounds_zero_temp(self):
        ev, el = compute_exposure_formula(20.0, 20.0, 40.0, 45.0, 0.5, 0.3, 100.0)
        assert 0 <= ev <= 1

    def test_bounds_max_temp(self):
        ev, el = compute_exposure_formula(40.0, 20.0, 40.0, 90.0, 1.0, 0.3, 100.0)
        assert 0 <= ev <= 1

    def test_bounds_no_exposure(self):
        ev, _ = compute_exposure_formula(30.0, 20.0, 40.0, 45.0, 0.0, 0.3, 100.0)
        assert 0 <= ev <= 1

    def test_bounds_full_exposure(self):
        ev, _ = compute_exposure_formula(30.0, 20.0, 40.0, 45.0, 1.0, 0.3, 100.0)
        assert 0 <= ev <= 1

    def test_bounds_below_temp_min(self):
        ev, _ = compute_exposure_formula(10.0, 20.0, 40.0, 45.0, 0.5, 0.3, 100.0)
        assert ev == 0.0

    def test_bounds_above_temp_max(self):
        ev, _ = compute_exposure_formula(50.0, 20.0, 40.0, 45.0, 0.5, 0.3, 100.0)
        assert 0 <= ev <= 1


class TestExposureMonotonicity:
    """Monotonic in shade, exposure_fraction, and temperature."""

    def test_monotonic_in_shade(self):
        """More shade → lower exposure."""
        ev_low_shade, _ = compute_exposure_formula(30.0, 20.0, 40.0, 45.0, 0.9, 0.3, 100.0)
        ev_high_shade, _ = compute_exposure_formula(30.0, 20.0, 40.0, 45.0, 0.1, 0.3, 100.0)
        assert ev_high_shade < ev_low_shade

    def test_monotonic_in_exposure_fraction(self):
        ev1, _ = compute_exposure_formula(30.0, 20.0, 40.0, 45.0, 0.2, 0.3, 100.0)
        ev2, _ = compute_exposure_formula(30.0, 20.0, 40.0, 45.0, 0.8, 0.3, 100.0)
        assert ev2 > ev1

    def test_monotonic_in_temperature(self):
        ev1, _ = compute_exposure_formula(25.0, 20.0, 40.0, 45.0, 0.5, 0.3, 100.0)
        ev2, _ = compute_exposure_formula(35.0, 20.0, 40.0, 45.0, 0.5, 0.3, 100.0)
        assert ev2 > ev1


class TestExposureLoad:
    """exposure_load = exposure_value × length_m."""

    def test_load_equals_value_times_length(self):
        ev, el = compute_exposure_formula(30.0, 20.0, 40.0, 45.0, 0.5, 0.3, 100.0)
        assert abs(el - ev * 100.0) < 1e-9

    def test_load_zero_length(self):
        ev, el = compute_exposure_formula(30.0, 20.0, 40.0, 45.0, 0.5, 0.3, 0.0)
        assert el == 0.0

    def test_load_with_series(self):
        ev, el = compute_exposure_formula(
            30.0, 20.0, 40.0,
            pd.Series([45.0, 60.0]),
            pd.Series([0.5, 0.8]),
            0.3,
            pd.Series([100.0, 200.0])
        )
        np.testing.assert_allclose(el, ev * pd.Series([100.0, 200.0]))


class TestExposureDeterminism:
    def test_deterministic(self):
        args = (30.0, 20.0, 40.0, 45.0, 0.5, 0.3, 100.0)
        ev1, _ = compute_exposure_formula(*args)
        ev2, _ = compute_exposure_formula(*args)
        assert ev1 == ev2


class TestFallbackModeSelection:
    def test_fallback_triggered_without_shadow_cache(self, tmp_path):
        from src.config_loader import ConfigLoader
        ConfigLoader._config = None

        segments = gpd.GeoDataFrame({
            'segment_id': ['s1'],
            'geometry': [LineString([(0, 0), (10, 0)])],
            'length_m': [10.0]
        }, crs="EPSG:32610")

        buildings = gpd.GeoDataFrame({
            'building_id': ['b1'],
            'footprint': [Polygon([(0, 5), (10, 5), (10, 15), (0, 15)])],
            'height_m': [10.0],
            'estimated_flag': [False]
        }, geometry='footprint', crs="EPSG:32610")

        results, mode = compute_exposure(segments, buildings, "testhash", tmp_path)

        assert mode == "estimated_exposure_mode"
        assert len(results) == 5

        for t, df in results.items():
            # Provenance/status fields
            assert 'exposure_value' in df.columns
            assert 'exposure_load' in df.columns
            assert 'prov_computation_mode' in df.columns
            assert 'prov_status' in df.columns
            assert 'prov_temperature_status' in df.columns
            assert 'prov_assumptions_version' in df.columns

            # Fallback label never "geometric"
            assert (df['prov_computation_mode'] == "estimated_exposure_mode").all()
            assert (df['prov_computation_mode'] != "geometric").all()

            # Values in bounds
            assert (df['exposure_value'] >= 0).all()
            assert (df['exposure_value'] <= 1).all()

    def test_fallback_label_never_geometric(self, tmp_path):
        """Explicit check: fallback mode label is never 'geometric'."""
        assert get_exposure_mode_label("estimated_exposure_mode") == "Estimated Exposure Mode"
        assert get_exposure_mode_label("geometric") == "Geometric Shadow Mode"
        assert get_exposure_mode_label("estimated_exposure_mode") != "Geometric Shadow Mode"

    def test_five_times_exist(self, tmp_path):
        from src.config_loader import ConfigLoader
        ConfigLoader._config = None

        segments = gpd.GeoDataFrame({
            'segment_id': ['s1'],
            'geometry': [LineString([(0, 0), (10, 0)])],
            'length_m': [10.0]
        }, crs="EPSG:32610")

        buildings = gpd.GeoDataFrame({
            'building_id': ['b1'],
            'footprint': [Polygon([(0, 5), (10, 5), (10, 15), (0, 15)])],
            'height_m': [10.0],
            'estimated_flag': [False]
        }, geometry='footprint', crs="EPSG:32610")

        results, _ = compute_exposure(segments, buildings, "testhash", tmp_path)
        assert set(results.keys()) == {"09:00", "11:00", "13:00", "15:00", "17:00"}
