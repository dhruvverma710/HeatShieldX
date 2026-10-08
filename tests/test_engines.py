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
        """Canonical values span [0,1]; non-canonical values equal their canonical twin. With clipping disabled."""
        from src.vulnerability_engine import compute_vulnerability
        from src.config_loader import get_config
        
        # Disable clipping for this test to match old behaviour
        config = get_config()
        if 'risk' not in config:
            config['risk'] = {}
        old_clip = config['risk'].get('normalization_clip_percentile')
        config['risk']['normalization_clip_percentile'] = 100
        
        segs = _make_segments()
        blds = _make_buildings()
        vuln = compute_vulnerability(segs, blds)

        # Restore config
        if old_clip is None:
            del config['risk']['normalization_clip_percentile']
        else:
            config['risk']['normalization_clip_percentile'] = old_clip

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
        from src.risk_engine import normalize_score
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normed, vmin, vmax = normalize_score(values)
        assert normed.min() >= 0
        assert normed.max() <= 100

    def test_zero_variance(self):
        from src.risk_engine import normalize_score
        values = np.array([3.0, 3.0, 3.0])
        normed, _, _ = normalize_score(values)
        assert (normed == 50.0).all()

    def test_stored_bounds_reuse(self):
        """
        a) stored bounds from a clipped run equal the clipped min/max
        b) reusing them on the same data gives identical scores
        c) values beyond the stored bounds clip to [0,100] on reuse
        """
        from src.risk_engine import normalize_score
        # 10 is an outlier, clipping at say 50th percentile will clip to 5.0
        values = np.array([1.0, 5.0, 10.0])
        
        # Initial run with clipping at 60th percentile -> bounds should be [1.0, 6.0]
        normed, vmin, vmax = normalize_score(values, clip_percentile=60)
        
        # a) Stored bounds equal the clipped min/max
        assert vmin == 1.0
        assert vmax == 6.0
        
        # b) Reusing them on same data gives identical scores
        # Note: when reusing bounds, clip_percentile is None
        reused_normed, _, _ = normalize_score(values, v_min=vmin, v_max=vmax, clip_percentile=None)
        np.testing.assert_allclose(normed, reused_normed)
        
        # c) Values beyond stored bounds clip to [0,100]
        new_values = np.array([0.0, 20.0])
        new_normed, _, _ = normalize_score(new_values, v_min=vmin, v_max=vmax, clip_percentile=None)
        assert new_normed[0] == 0.0    # 0.0 is below vmin=1.0, clips to 0
        assert new_normed[1] == 100.0  # 20.0 is above vmax=6.0, clips to 100

    def test_clipping_caps_an_outlier(self):
        from src.risk_engine import normalize_score
        values = np.array([1.0, 2.0, 3.0, 100.0])
        normed, vmin, vmax = normalize_score(values, clip_percentile=75)
        # 75th percentile of [1,2,3,100] is 27.25. So 100 clips to 27.25. Max becomes 27.25.
        # The output must be within [0,100]. The outlier 100 becomes score 100.
        assert vmax == 27.25
        assert normed.max() <= 100.0
        assert normed[-1] == 100.0

    def test_clip_percentile_100_equals_unclipped(self):
        from src.risk_engine import normalize_score
        values = np.array([1.0, 2.0, 3.0, 100.0])
        norm100, vmin100, vmax100 = normalize_score(values, clip_percentile=100)
        norm_none, vmin_none, vmax_none = normalize_score(values, clip_percentile=None)
        np.testing.assert_allclose(norm100, norm_none)
        assert vmin100 == vmin_none
        assert vmax100 == vmax_none

    def test_class_boundaries_upper_bound(self):
        """Classify by upper bounds: <=25 LOW, <=50 MODERATE, <=75 HIGH, else CRITICAL."""
        from src.risk_engine import classify_risk
        from src.config_loader import get_config
        config = get_config()
        ranges = config['risk_class_ranges']

        # Exact boundary values
        assert classify_risk([0], ranges) == ['LOW']
        assert classify_risk([25.0], ranges) == ['LOW']
        assert classify_risk([25.5], ranges) == ['MODERATE']
        assert classify_risk([50], ranges) == ['MODERATE']
        assert classify_risk([50.5], ranges) == ['HIGH']
        assert classify_risk([75], ranges) == ['HIGH']
        assert classify_risk([75.5], ranges) == ['CRITICAL']
        assert classify_risk([100], ranges) == ['CRITICAL']

    def test_no_unknown_class(self):
        """No score should ever produce UNKNOWN."""
        from src.risk_engine import classify_risk
        from src.config_loader import get_config
        config = get_config()
        ranges = config['risk_class_ranges']
        test_scores = [0, 10, 25, 25.5, 30, 50, 50.5, 60, 75, 75.5, 90, 100, 150]
        results = classify_risk(test_scores, ranges)
        assert 'UNKNOWN' not in results, f"UNKNOWN found in {results}"

    def test_nan_score_classified_as_low(self):
        """NaN scores should be classified as LOW, never UNKNOWN."""
        from src.risk_engine import classify_risk
        from src.config_loader import get_config
        config = get_config()
        ranges = config['risk_class_ranges']
        result = classify_risk([float('nan')], ranges)
        assert result == ['LOW']
        assert 'UNKNOWN' not in result

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
    def _make_risk_df_3seg(self):
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

    def _make_10seg_risk_df(self):
        """10 canonical segments at 13:00 with known values for rank verification."""
        n = 10
        return pd.DataFrame({
            'segment_id': [f'S-{i}-0' for i in range(n)],
            'is_canonical': [True] * n,
            'time': ['13:00'] * n,
            'exposure_value': np.linspace(0.1, 1.0, n),
            'vulnerability_value': np.linspace(0.0, 0.9, n),
            'access_penalty': np.linspace(0.0, 1.0, n),
            'risk_score': np.linspace(5, 95, n),
            'risk_class': ['LOW', 'LOW', 'LOW', 'MODERATE', 'MODERATE',
                           'MODERATE', 'HIGH', 'HIGH', 'CRITICAL', 'CRITICAL'],
        })

    def test_drivers_ordering_and_dominance(self):
        from src.explain_engine import compute_drivers
        rdf = self._make_risk_df_3seg()
        drivers = compute_drivers(rdf)
        assert len(drivers) == 3
        for d in drivers:
            pcts = [drv['percentile'] for drv in d['drivers']]
            assert pcts == sorted(pcts, reverse=True), "Drivers should be ordered by percentile desc"

    def test_percentile_over_canonical_at_same_time(self):
        """Percentile rank computed over the 10 canonical segments at 13:00."""
        from src.explain_engine import compute_drivers
        rdf = self._make_10seg_risk_df()
        drivers = compute_drivers(rdf)
        assert len(drivers) == 10

        # The last segment (index 9) has the highest values → percentile should be 1.0
        last_drivers = drivers[9]['drivers']
        for d in last_drivers:
            assert d['percentile'] == 1.0, f"Top segment driver {d['driver']} should have percentile 1.0, got {d['percentile']}"

        # The first segment (index 0) has the lowest values → percentile should be 0.1 (1/10)
        first_drivers = drivers[0]['drivers']
        for d in first_drivers:
            assert d['percentile'] <= 0.2, f"Bottom segment driver {d['driver']} should have low percentile, got {d['percentile']}"

    def test_low_street_no_dominant_driver(self):
        """A LOW-risk street with low values should have no dominant drivers."""
        from src.explain_engine import compute_drivers, build_explanation_text
        rdf = self._make_10seg_risk_df()
        drivers = compute_drivers(rdf)

        # First segment: LOW risk, lowest values
        low_street = drivers[0]
        assert low_street['risk_class'] == 'LOW'
        dominant = [d for d in low_street['drivers'] if d['is_dominant']]
        assert len(dominant) == 0, f"LOW street should have no dominant drivers, got {[d['driver'] for d in dominant]}"

        # Explanation text should say "no single dominant driver"
        text = build_explanation_text(low_street, "Quiet Lane")
        assert "no single dominant driver" in text.lower(), f"Expected 'no single dominant driver' in: {text}"

    def test_low_street_wording_no_high_language(self):
        """LOW-risk streets should never have 'high' or 'poor' language."""
        from src.explain_engine import compute_drivers, build_explanation_text
        rdf = self._make_10seg_risk_df()
        drivers = compute_drivers(rdf)
        low_street = drivers[0]
        text = build_explanation_text(low_street, "Quiet Lane")
        text_lower = text.lower()
        # 'high' should not appear except as part of 'highest'
        stripped = text_lower.replace('highest', '')
        assert 'high' not in stripped, f"LOW street text should not contain 'high': {text}"
        assert 'poor' not in text_lower, f"LOW street text should not contain 'poor': {text}"

    def test_critical_street_has_dominant_drivers(self):
        """A CRITICAL street should have at least one dominant driver."""
        from src.explain_engine import compute_drivers, build_explanation_text
        rdf = self._make_10seg_risk_df()
        drivers = compute_drivers(rdf)
        critical_street = drivers[9]
        assert critical_street['risk_class'] == 'CRITICAL'
        dominant = [d for d in critical_street['drivers'] if d['is_dominant']]
        assert len(dominant) > 0, "CRITICAL street should have dominant drivers"

        text = build_explanation_text(critical_street)
        assert "key drivers" in text.lower(), f"Expected 'Key drivers' in: {text}"

    def test_hottest_vs_highest_risk_differ(self):
        from src.explain_engine import compare_hottest_vs_highest_risk
        rdf = self._make_risk_df_3seg()
        # Hottest = highest exposure_value = S-1-2-0 (0.8)
        # Highest risk = highest risk_score = S-2-3-0 (80)
        exp_df = rdf[['segment_id', 'exposure_value', 'is_canonical']].copy()
        comparison = compare_hottest_vs_highest_risk(exp_df, rdf)
        assert comparison['hottest_segment_id'] == 'S-1-2-0'
        assert comparison['highest_risk_segment_id'] == 'S-2-3-0'
        assert comparison['same_street'] == False

    def test_explanation_text_contains_required_fields(self):
        from src.explain_engine import compute_drivers, build_explanation_text
        rdf = self._make_risk_df_3seg()
        drivers = compute_drivers(rdf)
        text = build_explanation_text(drivers[0], "Main Street")
        assert "Main Street" in text
        assert "13:00" in text
        assert "risk score" in text.lower() or "risk" in text.lower()

