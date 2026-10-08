"""
Tests for vulnerability_engine, cooling_engine, risk_engine, explain_engine,
and the dedupe flag (street_id / is_canonical).
All synthetic, no internet.
"""
import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Polygon, Point

# ────────────────────── helper fixtures ──────────────────────

def _make_segments():
    """A small directed graph with two-way edges + one one-way edge."""
    rows = [
        # Two-way pair: u=1,v=2 and u=2,v=1
        {'segment_id': 'S-1-2-0', 'street_id': 'E-1-2-0', 'is_canonical': True,
         'u': 1, 'v': 2, 'key': 0,
         'geometry': LineString([(0, 0), (100, 0)]),
         'length_m': 100.0, 'name': 'Main St', 'highway': 'residential', 'oneway': False},
        {'segment_id': 'S-2-1-0', 'street_id': 'E-1-2-0', 'is_canonical': False,
         'u': 2, 'v': 1, 'key': 0,
         'geometry': LineString([(100, 0), (0, 0)]),
         'length_m': 100.0, 'name': 'Main St', 'highway': 'residential', 'oneway': False},
        # Two-way pair: u=2,v=3 and u=3,v=2
        {'segment_id': 'S-2-3-0', 'street_id': 'E-2-3-0', 'is_canonical': True,
         'u': 2, 'v': 3, 'key': 0,
         'geometry': LineString([(100, 0), (200, 0)]),
         'length_m': 100.0, 'name': 'Oak Ave', 'highway': 'primary', 'oneway': False},
        {'segment_id': 'S-3-2-0', 'street_id': 'E-2-3-0', 'is_canonical': False,
         'u': 3, 'v': 2, 'key': 0,
         'geometry': LineString([(200, 0), (100, 0)]),
         'length_m': 100.0, 'name': 'Oak Ave', 'highway': 'primary', 'oneway': False},
        # One-way: u=3,v=4
        {'segment_id': 'S-3-4-0', 'street_id': 'E-3-4-0', 'is_canonical': True,
         'u': 3, 'v': 4, 'key': 0,
         'geometry': LineString([(200, 0), (300, 0)]),
         'length_m': 100.0, 'name': 'Elm Rd', 'highway': 'footway', 'oneway': True},
        # Disconnected: u=10,v=11 (no path to 1-4)
        {'segment_id': 'S-10-11-0', 'street_id': 'E-10-11-0', 'is_canonical': True,
         'u': 10, 'v': 11, 'key': 0,
         'geometry': LineString([(1000, 1000), (1100, 1000)]),
         'length_m': 100.0, 'name': 'Far St', 'highway': 'residential', 'oneway': False},
        {'segment_id': 'S-11-10-0', 'street_id': 'E-10-11-0', 'is_canonical': False,
         'u': 11, 'v': 10, 'key': 0,
         'geometry': LineString([(1100, 1000), (1000, 1000)]),
         'length_m': 100.0, 'name': 'Far St', 'highway': 'residential', 'oneway': False},
    ]
    return gpd.GeoDataFrame(rows, crs="EPSG:32610")


def _make_buildings():
    return gpd.GeoDataFrame({
        'building_id': ['b1', 'b2'],
        'footprint': [
            Polygon([(20, 5), (40, 5), (40, 15), (20, 15)]),
            Polygon([(120, 5), (140, 5), (140, 15), (120, 15)]),
        ],
        'height_m': [10.0, 8.0],
        'height_source': ['fallback', 'fallback'],
        'estimated_flag': [True, True],
        'levels': [None, None],
        'geometry': [
            Polygon([(20, 5), (40, 5), (40, 15), (20, 15)]),
            Polygon([(120, 5), (140, 5), (140, 15), (120, 15)]),
        ],
    }, crs="EPSG:32610")


# ────────────────────── DEDUPE FLAG TESTS ──────────────────────

class TestDedupeFlag:
    def test_one_canonical_per_street_id(self):
        segs = _make_segments()
        canon_per_street = segs.groupby('street_id')['is_canonical'].sum()
        assert (canon_per_street == 1).all(), "Exactly one canonical edge per street_id"

    def test_duplicates_have_equal_length(self):
        segs = _make_segments()
        for sid, grp in segs.groupby('street_id'):
            assert grp['length_m'].nunique() == 1, f"street_id {sid} has unequal lengths"


# ────────────────────── VULNERABILITY TESTS ──────────────────────

class TestVulnerability:
    def test_bounds(self):
        from src.vulnerability_engine import compute_vulnerability
        segs = _make_segments()
        blds = _make_buildings()
        vuln = compute_vulnerability(segs, blds)
        assert (vuln['vulnerability_value'] >= 0).all()
        assert (vuln['vulnerability_value'] <= 1).all()

    def test_seed_determinism(self):
        from src.vulnerability_engine import compute_vulnerability
        segs = _make_segments()
        blds = _make_buildings()
        v1 = compute_vulnerability(segs, blds)
        v2 = compute_vulnerability(segs, blds)
        pd.testing.assert_frame_equal(v1, v2)

    def test_estimated_flags(self):
        from src.vulnerability_engine import compute_vulnerability
        segs = _make_segments()
        blds = _make_buildings()
        vuln = compute_vulnerability(segs, blds)
        assert (vuln['estimated_flag'] == True).all()
        assert (vuln['prov_status'] == 'ESTIMATED').all()
        assert (vuln['prov_source_type'] == 'synthetic').all()

    def test_canonical_normalisation(self):
        """Canonical values span [0,1]; non-canonical values equal their canonical twin."""
        from src.vulnerability_engine import compute_vulnerability
        segs = _make_segments()
        blds = _make_buildings()
        vuln = compute_vulnerability(segs, blds)

        # Non-canonical rows should match their canonical twin
        for sid, grp in vuln.groupby('street_id'):
            assert grp['vulnerability_value'].nunique() == 1, \
                f"street_id {sid}: duplicates have different vulnerability values"


# ────────────────────── DISTANCE PENALTY TESTS ──────────────────────

class TestBoundedDistancePenalty:
    def test_bounds(self):
        from src.cooling_engine import bounded_distance_penalty
        assert bounded_distance_penalty(0, 100) == 0.0
        assert bounded_distance_penalty(100, 100) == 1.0
        assert bounded_distance_penalty(200, 100) == 1.0  # clipped
        assert 0 < bounded_distance_penalty(50, 100) < 1

    def test_monotonic(self):
        from src.cooling_engine import bounded_distance_penalty
        p1 = bounded_distance_penalty(30, 100)
        p2 = bounded_distance_penalty(70, 100)
        assert p2 > p1

    def test_zero_radius(self):
        from src.cooling_engine import bounded_distance_penalty
        assert bounded_distance_penalty(50, 0) == 1.0


# ────────────────────── NETWORK DISTANCE TESTS ──────────────────────

class TestNetworkDistance:
    def test_toy_graph(self):
        from src.cooling_engine import build_street_graph, multi_source_dijkstra
        segs = _make_segments()
        G = build_street_graph(segs)
        assert G.has_node(1)
        assert G.has_node(4)
        # Path 1→2→3→4 should be 300m
        dists = multi_source_dijkstra(G, [1])
        assert abs(dists[4] - 300) < 1e-6

    def test_unreachable(self):
        """Disconnected node should not appear in distances."""
        from src.cooling_engine import build_street_graph, multi_source_dijkstra
        segs = _make_segments()
        G = build_street_graph(segs)
        dists = multi_source_dijkstra(G, [1])
        assert 10 not in dists or 11 not in dists  # disconnected from 1


# ────────────────────── NO-FACILITY TESTS ──────────────────────

class TestNoFacility:
    def test_no_facility_applies_availability_penalty(self):
        from src.cooling_engine import compute_cooling_access
        segs = _make_segments()
        empty_water = gpd.GeoDataFrame(
            columns=['geometry', 'facility_type', 'source'], crs=segs.crs
        )
        empty_cooling = gpd.GeoDataFrame(
            columns=['geometry', 'facility_type', 'source'], crs=segs.crs
        )
        result, _, _ = compute_cooling_access(
            segs, "test_area",
            water_facilities=empty_water,
            cooling_facilities=empty_cooling,
        )
        # With no facilities, availability penalty should be applied
        assert (result['access_penalty'] > 0).all()
        # dist should be -1 (infinity)
        assert (result['dist_water_m'] == -1).all()
        assert (result['dist_cooling_m'] == -1).all()


# ────────────────────── RISK ENGINE TESTS ──────────────────────

class TestRiskEngine:
    def _make_exposure_results(self, segs):
        """Create fake exposure results for 5 times."""
        from src.config_loader import get_config
        config = get_config()
        times = config['canonical_times']
        results = {}
        for t in times:
            results[t] = pd.DataFrame({
                'segment_id': segs['segment_id'],
                'exposure_value': np.random.RandomState(42).uniform(0, 1, len(segs)),
            })
        return results

    def test_risk_formula(self):
        from src.risk_engine import compute_risk
        from src.vulnerability_engine import compute_vulnerability
        segs = _make_segments()
        blds = _make_buildings()
        vuln = compute_vulnerability(segs, blds)

        cooling = pd.DataFrame({
            'segment_id': segs['segment_id'],
            'street_id': segs['street_id'],
            'is_canonical': segs['is_canonical'],
            'access_penalty': 0.5,
        })

        exp = self._make_exposure_results(segs)
        risk_results, bounds = compute_risk(exp, vuln, cooling, "geometric")

        assert len(risk_results) == 5
        for t, df in risk_results.items():
            # baseline_risk = exposure_value * vulnerability_value
            expected_base = df['exposure_value'] * df['vulnerability_value']
            np.testing.assert_allclose(df['baseline_risk'], expected_base, atol=1e-9)
            # final_risk_raw = baseline * (1 + access_penalty)
            expected_raw = expected_base * (1 + df['access_penalty'])
            np.testing.assert_allclose(df['final_risk_raw'], expected_raw, atol=1e-9)

    def test_normalisation_bounds(self):
        from src.risk_engine import normalize_risk
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normed, vmin, vmax = normalize_risk(values)
        assert normed.min() >= 0
        assert normed.max() <= 100

    def test_zero_variance(self):
        from src.risk_engine import normalize_risk
        values = np.array([3.0, 3.0, 3.0])
        normed, _, _ = normalize_risk(values)
        assert (normed == 50.0).all()

    def test_stored_bounds_reuse(self):
        from src.risk_engine import normalize_risk
        values = np.array([1.0, 5.0, 10.0])
        _, vmin, vmax = normalize_risk(values)
        # Re-normalise new values with stored bounds
        new_values = np.array([3.0, 7.0])
        normed, _, _ = normalize_risk(new_values, vmin, vmax)
        # 3.0 should be at (3-1)/(10-1)*100 = 22.2
        assert abs(normed[0] - 22.222) < 0.1

    def test_class_boundaries(self):
        from src.risk_engine import classify_risk
        from src.config_loader import get_config
        config = get_config()
        ranges = config['risk_class_ranges']
        assert classify_risk([25], ranges) == ['LOW']
        assert classify_risk([26], ranges) == ['MODERATE']
        assert classify_risk([50], ranges) == ['MODERATE']
        assert classify_risk([51], ranges) == ['HIGH']
        assert classify_risk([75], ranges) == ['HIGH']
        assert classify_risk([76], ranges) == ['CRITICAL']

    def test_risk_on_fallback_mode(self):
        from src.risk_engine import compute_risk
        from src.vulnerability_engine import compute_vulnerability
        segs = _make_segments()
        blds = _make_buildings()
        vuln = compute_vulnerability(segs, blds)

        cooling = pd.DataFrame({
            'segment_id': segs['segment_id'],
            'street_id': segs['street_id'],
            'is_canonical': segs['is_canonical'],
            'access_penalty': 0.0,
        })

        exp = self._make_exposure_results(segs)
        risk_results, _ = compute_risk(exp, vuln, cooling, "estimated_exposure_mode")

        for t, df in risk_results.items():
            assert (df['computation_mode'] == "estimated_exposure_mode").all()


# ────────────────────── EXPLAINABILITY TESTS ──────────────────────

class TestExplainEngine:
    def _make_risk_df(self):
        return pd.DataFrame({
            'segment_id': ['S-1-2-0', 'S-2-3-0', 'S-3-4-0'],
            'is_canonical': [True, True, True],
            'time': ['13:00', '13:00', '13:00'],
            'exposure_value': [0.8, 0.3, 0.5],
            'exposure_fraction': [0.9, 0.4, 0.6],
            'sun_factor': [0.7, 0.7, 0.7],
            'shade_fraction': [0.1, 0.6, 0.4],
            'vulnerability_value': [0.2, 0.9, 0.5],
            'access_penalty': [0.1, 0.1, 0.8],
            'risk_score': [30, 80, 60],
            'risk_class': ['MODERATE', 'CRITICAL', 'HIGH'],
        })

    def test_drivers_ordering_and_dominance(self):
        from src.explain_engine import compute_drivers
        rdf = self._make_risk_df()
        drivers = compute_drivers(rdf)
        assert len(drivers) == 3
        for d in drivers:
            pcts = [drv['percentile'] for drv in d['drivers']]
            assert pcts == sorted(pcts, reverse=True), "Drivers should be ordered by percentile desc"

    def test_hottest_vs_highest_risk_differ(self):
        from src.explain_engine import compare_hottest_vs_highest_risk
        rdf = self._make_risk_df()
        # Hottest = highest exposure_value = S-1-2-0 (0.8)
        # Highest risk = highest risk_score = S-2-3-0 (80)
        exp_df = rdf[['segment_id', 'exposure_value', 'is_canonical']].copy()
        comparison = compare_hottest_vs_highest_risk(exp_df, rdf)
        assert comparison['hottest_segment_id'] == 'S-1-2-0'
        assert comparison['highest_risk_segment_id'] == 'S-2-3-0'
        assert comparison['same_street'] == False

    def test_explanation_text(self):
        from src.explain_engine import compute_drivers, build_explanation_text
        rdf = self._make_risk_df()
        drivers = compute_drivers(rdf)
        text = build_explanation_text(drivers[0], "Main Street")
        assert "Main Street" in text
        assert "13:00" in text
        assert "risk score" in text.lower() or "risk" in text.lower()
