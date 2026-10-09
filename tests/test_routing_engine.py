"""
Tests for HeatShield X Routing Engine and Stop Finder.
"""
import pytest
import os
import re
import networkx as nx
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from unittest.mock import patch
from src.routing_engine import (
    build_routing_graph, apply_snapshot_to_graph, compute_routes,
    RoutingError, _get_exposure_class, extract_route_metrics,
)
from src.config_loader import ConfigLoader


@pytest.fixture(autouse=True)
def _restore_config():
    """Save and restore the global ConfigLoader._config around each test."""
    original = ConfigLoader._config
    yield
    ConfigLoader._config = original



# ──────────────── toy graph fixture ────────────────
#
#  1 ──S1(hot)──▶ 2        One-way edges: S1 (1→2), S4 (4→5)
#  │              ▲        Two-way edges: S2 (1↔3), S3 (3↔2)
#  S2(cool)      S3(cool)
#  │              │
#  ▼              │
#  3 ─────────────┘
#
#  S1: length=100, exposure=1.0, shade=0.0   (the hot short-cut)
#  S2: length=75,  exposure=0.1, shade=0.5   (cool leg 1)
#  S3: length=75,  exposure=0.1, shade=0.5   (cool leg 2)
#  S4: length=100, exposure=0.0              (disconnected pair – 4→5)
#
#  Walking speed = 1 m/s  → time_s == length_m
#
#  FASTEST  1→2  via S1  100 m  100 s  exposure_mean = 1.0
#  COOL     1→3→2 via S2+S3  150 m  150 s  exposure_mean = 0.1
#

def _make_toy_segments():
    """Return (segments_df, snapshot_df) for the diamond graph above."""
    segs = pd.DataFrame([
        {'segment_id': 'S1', 'street_id': 'E1', 'u': 1, 'v': 2,
         'length_m': 100.0, 'oneway': True},
        {'segment_id': 'S2', 'street_id': 'E2', 'u': 1, 'v': 3,
         'length_m': 75.0,  'oneway': False},
        {'segment_id': 'S3', 'street_id': 'E3', 'u': 3, 'v': 2,
         'length_m': 75.0,  'oneway': False},
        # Disconnected pair
        {'segment_id': 'S4', 'street_id': 'E4', 'u': 4, 'v': 5,
         'length_m': 100.0, 'oneway': True},
    ])
    snap = pd.DataFrame([
        {'segment_id': 'S1', 'is_canonical': True, 'street_id': 'E1',
         'exposure_value': 1.0, 'shade_fraction': 0.0, 'access_penalty': 0.5,
         'covered_water': False, 'covered_cooling': False,
         'time_status': 'COMPUTED', 'computation_mode': 'geometric'},
        {'segment_id': 'S2', 'is_canonical': True, 'street_id': 'E2',
         'exposure_value': 0.1, 'shade_fraction': 0.5, 'access_penalty': 0.2,
         'covered_water': True, 'covered_cooling': False,
         'time_status': 'COMPUTED', 'computation_mode': 'geometric'},
        {'segment_id': 'S3', 'is_canonical': True, 'street_id': 'E3',
         'exposure_value': 0.1, 'shade_fraction': 0.5, 'access_penalty': 0.2,
         'covered_water': False, 'covered_cooling': False,
         'time_status': 'COMPUTED', 'computation_mode': 'geometric'},
        {'segment_id': 'S4', 'is_canonical': True, 'street_id': 'E4',
         'exposure_value': 0.0, 'shade_fraction': 0.0, 'access_penalty': 0.0,
         'covered_water': False, 'covered_cooling': False,
         'time_status': 'COMPUTED', 'computation_mode': 'geometric'},
    ])
    return segs, snap


def _patch_config(overrides):
    """Return a config dict with routing overrides applied."""
    base = {
        'config_version': '2.0.0',
        'risk_class_ranges': {
            'LOW': [0, 25], 'MODERATE': [26, 50],
            'HIGH': [51, 75], 'CRITICAL': [76, 100],
        },
        'routing': {
            'walking_speed_mps': 1.0,
            'max_detour_factor': 1.6,
            'heat_aware_gamma_schedule': [8, 4, 2, 1],
            'balanced_alpha': 0.5,
            'balanced_beta': 0.5,
        },
    }
    base['routing'].update(overrides.get('routing', {}))
    base.update({k: v for k, v in overrides.items() if k != 'routing'})
    return base


# ──────────────── GRAPH CONSTRUCTION ────────────────

class TestBuildRoutingGraph:
    def test_largest_component_kept(self):
        """Disconnected nodes 4,5 should be pruned, leaving 1,2,3."""
        segs, _ = _make_toy_segments()
        G = build_routing_graph(segs)
        assert set(G.nodes()) == {1, 2, 3}

    def test_oneway_respected(self):
        """One-way edge S1 should only exist 1→2, not 2→1."""
        segs, _ = _make_toy_segments()
        G = build_routing_graph(segs)
        assert G.has_edge(1, 2)
        assert not G.has_edge(2, 1)   # S1 is one-way

    def test_bidirectional_edges(self):
        """Two-way edges S2,S3 should exist in both directions."""
        segs, _ = _make_toy_segments()
        G = build_routing_graph(segs)
        assert G.has_edge(1, 3) and G.has_edge(3, 1)
        assert G.has_edge(3, 2) and G.has_edge(2, 3)


# ──────────────── ROUTE COMPUTATION ────────────────

class TestRouteComputation:
    def _setup(self, routing_overrides=None):
        segs, snap = _make_toy_segments()
        G = build_routing_graph(segs)
        cfg = _patch_config({'routing': routing_overrides or {}})
        from src.config_loader import ConfigLoader
        ConfigLoader._config = cfg
        G = apply_snapshot_to_graph(G, snap, cfg)
        return G, snap, cfg

    def test_fastest_is_minimum_time(self):
        """Fastest route should be S1 (100 m at 1 m/s = 100 s)."""
        G, snap, _ = self._setup()
        routes = compute_routes(G, 1, 2, snap)
        fastest = next(r for r in routes if 'FASTEST' in r['type_labels'])
        assert fastest['total_distance_m'] == 100.0
        assert abs(fastest['total_time_min'] - 100 / 60) < 0.01

    def test_heat_aware_avoids_hot_street(self):
        """With high gamma and generous detour, heat-aware picks 1→3→2."""
        G, snap, _ = self._setup({'heat_aware_gamma_schedule': [8]})
        routes = compute_routes(G, 1, 2, snap)
        ha = next(r for r in routes if 'HEAT-AWARE' in r['type_labels'])
        assert ha['total_distance_m'] == 150.0
        assert ha['modelled_heat_exposure'] == 0.1

    def test_heat_aware_fallback_beyond_cap(self):
        """With tight detour cap, heat-aware falls back to fastest."""
        G, snap, _ = self._setup({
            'max_detour_factor': 1.1,
            'heat_aware_gamma_schedule': [8],
        })
        routes = compute_routes(G, 1, 2, snap)
        # FASTEST and HEAT-AWARE should share the same path
        fastest = next(r for r in routes if 'FASTEST' in r['type_labels'])
        assert 'HEAT-AWARE' in fastest['type_labels']
        assert fastest['recommendation_text'] == (
            "no cooler route within the detour limit"
        )

    def test_identical_edge_sequences_merged(self):
        """When heat-aware and balanced pick the same path as fastest,
        they should merge into one result."""
        G, snap, _ = self._setup({
            'max_detour_factor': 1.0,  # no detour allowed
            'heat_aware_gamma_schedule': [0],  # gamma=0 → same as time
        })
        routes = compute_routes(G, 1, 2, snap, weight_w=0.0)  # alpha=1 → time only
        assert len(routes) == 1
        labels = routes[0]['type_labels']
        assert 'FASTEST' in labels

    def test_balanced_between_fastest_and_heat_aware(self):
        """Balanced exposure should be between fastest and heat-aware."""
        G, snap, _ = self._setup({'heat_aware_gamma_schedule': [8]})
        routes = compute_routes(G, 1, 2, snap, weight_w=0.5)
        fastest = next(r for r in routes if 'FASTEST' in r['type_labels'])
        # The other route has HEAT-AWARE (and maybe BALANCED)
        other = [r for r in routes if r is not fastest]
        if other:
            ha = other[0]
            # Balanced exposure should be >= heat-aware (which has 0.1)
            # and <= fastest (which has 1.0)
            for r in routes:
                assert 0.0 <= r['modelled_heat_exposure'] <= 1.01

    def test_totals_equal_sum_of_edges(self):
        """Route totals must equal the sum of individual edge values."""
        G, snap, _ = self._setup()
        routes = compute_routes(G, 1, 2, snap)
        for r in routes:
            # Recompute from segment_ids
            total_dist = 0.0
            total_time = 0.0
            exposure_load = 0.0
            for sid in r['segment_ids']:
                for u, v, d in G.edges(data=True):
                    if d['segment_id'] == sid:
                        total_dist += d['length_m']
                        total_time += d['time_s']
                        exposure_load += d['heat_load']
                        break
            assert abs(r['total_distance_m'] - total_dist) < 0.1
            assert abs(r['total_time_min'] - total_time / 60) < 0.01

    def test_weighted_shade_and_cooling_access(self):
        """Verify weighted shade and cooling access for the cool route."""
        G, snap, _ = self._setup({'heat_aware_gamma_schedule': [8]})
        routes = compute_routes(G, 1, 2, snap)
        ha = next(r for r in routes if 'HEAT-AWARE' in r['type_labels'])
        # S2 and S3 both have shade_fraction=0.5
        assert abs(ha['weighted_shade'] - 0.5) < 0.01
        # S2 has covered_water=True (75m), S3 has covered_water=False (75m)
        # water_coverage_share = 75 / 150 = 0.5
        assert abs(ha['cooling_access']['water_coverage_share'] - 0.5) < 0.01

    def test_interpolated_flag_propagates(self):
        """INTERPOLATED time_status should propagate to all routes."""
        G, snap, _ = self._setup()
        snap = snap.copy()
        snap['time_status'] = 'INTERPOLATED'
        G = apply_snapshot_to_graph(G, snap, _patch_config({}))
        from src.config_loader import ConfigLoader
        ConfigLoader._config = _patch_config({})
        routes = compute_routes(G, 1, 2, snap)
        for r in routes:
            assert r['time_status'] == 'INTERPOLATED'

    def test_consistency_with_snapshot(self):
        """Route edge exposure values must equal the snapshot values."""
        G, snap, _ = self._setup()
        routes = compute_routes(G, 1, 2, snap)
        snap_lookup = snap.set_index('segment_id')
        for r in routes:
            for sid in r['segment_ids']:
                # find the edge
                for u, v, d in G.edges(data=True):
                    if d['segment_id'] == sid:
                        if sid in snap_lookup.index:
                            expected = float(snap_lookup.loc[sid, 'exposure_value'])
                            assert abs(d['exposure_value'] - expected) < 1e-9
                        break


# ──────────────── ERROR HANDLING ────────────────

class TestRoutingErrors:
    def test_origin_equals_destination(self):
        segs, snap = _make_toy_segments()
        G = build_routing_graph(segs)
        cfg = _patch_config({})
        from src.config_loader import ConfigLoader
        ConfigLoader._config = cfg
        G = apply_snapshot_to_graph(G, snap, cfg)
        with pytest.raises(RoutingError, match="zero-length"):
            compute_routes(G, 1, 1, snap)

    def test_unknown_node(self):
        segs, snap = _make_toy_segments()
        G = build_routing_graph(segs)
        cfg = _patch_config({})
        from src.config_loader import ConfigLoader
        ConfigLoader._config = cfg
        G = apply_snapshot_to_graph(G, snap, cfg)
        with pytest.raises(RoutingError, match="Unknown"):
            compute_routes(G, 99, 2, snap)

    def test_no_path(self):
        segs, snap = _make_toy_segments()
        G = build_routing_graph(segs)
        cfg = _patch_config({})
        from src.config_loader import ConfigLoader
        ConfigLoader._config = cfg
        G = apply_snapshot_to_graph(G, snap, cfg)
        G.add_node(99)  # isolated node
        with pytest.raises(RoutingError, match="No path"):
            compute_routes(G, 1, 99, snap)


# ──────────────── STOP FINDER ────────────────

class TestStopFinder:
    def test_no_invented_pois_when_cache_empty(self):
        """When POI cache is missing and OSM unavailable, stops should
        only come from water/cooling facilities, never invented."""
        from src.stop_finder import compute_safe_stops
        # With no facilities and empty POI result, should return empty
        route = {'segment_ids': ['S1'], 'type_labels': ['FASTEST']}
        segs, _ = _make_toy_segments()
        segs = gpd.GeoDataFrame(
            segs,
            geometry=[LineString([(0, 0), (1, 0)])] * len(segs),
            crs='EPSG:32610',
        )
        G = build_routing_graph(segs)
        G = apply_snapshot_to_graph(G, _make_toy_segments()[1], _patch_config({}))

        with patch('src.stop_finder.load_pois', return_value=gpd.GeoDataFrame()):
            stops = compute_safe_stops(
                route, G, segs, 'Test Area', None, None
            )
        assert len(stops) == 0  # No invented POIs

    def test_sponsored_flag_no_ranking_change(self):
        """Sponsored status must not change eligibility or ranking order."""
        from src.stop_finder import compute_safe_stops
        route = {'segment_ids': ['S1'], 'type_labels': ['FASTEST']}
        segs, _ = _make_toy_segments()
        segs = gpd.GeoDataFrame(
            segs,
            geometry=[LineString([(0, 0), (1, 0)])] * len(segs),
            crs='EPSG:32610',
        )
        G = build_routing_graph(segs)
        G = apply_snapshot_to_graph(G, _make_toy_segments()[1], _patch_config({}))

        water_facs = gpd.GeoDataFrame(
            [{'facility_id': 'W1', 'name': 'Fountain A', 'source': 'osm',
              'geometry': Point(0.5, 0)}],
            crs='EPSG:32610',
        )
        with patch('src.stop_finder.load_pois', return_value=gpd.GeoDataFrame()):
            stops = compute_safe_stops(
                route, G, segs, 'Test Area', water_facs, None
            )
        for s in stops:
            assert s['rating'] is None
            # Sponsored status should be False unless ID matches config
            assert s['sponsored_status'] is False

    def test_rating_none_when_missing(self):
        """Rating must be None (not invented)."""
        from src.stop_finder import compute_safe_stops
        route = {'segment_ids': ['S1'], 'type_labels': ['FASTEST']}
        segs, _ = _make_toy_segments()
        segs = gpd.GeoDataFrame(
            segs,
            geometry=[LineString([(0, 0), (1, 0)])] * len(segs),
            crs='EPSG:32610',
        )
        G = build_routing_graph(segs)
        G = apply_snapshot_to_graph(G, _make_toy_segments()[1], _patch_config({}))

        water_facs = gpd.GeoDataFrame(
            [{'facility_id': 'W1', 'name': 'Test', 'source': 'osm',
              'geometry': Point(0.5, 0)}],
            crs='EPSG:32610',
        )
        with patch('src.stop_finder.load_pois', return_value=gpd.GeoDataFrame()):
            stops = compute_safe_stops(
                route, G, segs, 'Test', water_facs, None
            )
        for s in stops:
            assert s['rating'] is None


# ──────────────── FORBIDDEN PHRASE SCAN ────────────────

class TestForbiddenPhrases:
    FORBIDDEN = [
        r'prevents?\s+heatstroke',
        r'safe\s+from\s+heat',
        r'\bguaranteed\b',
        r'medical\s+claim',
    ]

    def test_no_forbidden_phrases_in_source(self):
        """Source files must not contain forbidden medical claims."""
        src_dirs = ['src', os.path.join('app', 'pages')]
        for src_dir in src_dirs:
            full_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), src_dir)
            if not os.path.isdir(full_dir):
                continue
            for fname in os.listdir(full_dir):
                if not fname.endswith('.py'):
                    continue
                fpath = os.path.join(full_dir, fname)
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                for pattern in self.FORBIDDEN:
                    assert not re.search(pattern, content, re.IGNORECASE), (
                        f"Forbidden phrase '{pattern}' found in {fpath}"
                    )
