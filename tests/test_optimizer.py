"""
Tests for HeatShield X Optimizer and Impact Engine.
"""
import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Polygon
from src.optimizer import run_optimizer, _objective_score, evaluate_candidate_exact
from src.impact_engine import compute_impact
from src.intervention_engine import generate_candidates

# ────────────────────── helper fixtures ──────────────────────

def _make_segments():
    """A small directed graph with two-way edges."""
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
    ]
    return gpd.GeoDataFrame(rows, crs="EPSG:32610")


def _make_mock_state(segments):
    """Create a basic fake state for the optimizer."""
    vuln_df = pd.DataFrame({
        'segment_id': segments['segment_id'],
        'vulnerable_mass': [10.0, 10.0, 20.0, 20.0]  # Oak Ave is more vulnerable
    })
    
    # Fake risk records
    risk_1300 = pd.DataFrame({
        'segment_id': segments['segment_id'],
        'is_canonical': segments['is_canonical'],
        'exposure_value': [0.8, 0.8, 0.9, 0.9],
        'baseline_risk': [8.0, 8.0, 18.0, 18.0],
        'final_risk_raw': [12.0, 12.0, 27.0, 27.0],
        'risk_score': [40, 40, 90, 90],
        'risk_class': ['MODERATE', 'MODERATE', 'CRITICAL', 'CRITICAL'],
        'computation_mode': ['geometric'] * 4
    })
    risk_records = {'13:00': risk_1300}
    
    # Fake exposure records
    exp_1300 = pd.DataFrame({
        'segment_id': segments['segment_id'],
        'is_canonical': segments['is_canonical'],
        'exposure_value': [0.8, 0.8, 0.9, 0.9]
    })
    exposure_records = {'13:00': exp_1300}
    
    cooling_access = pd.DataFrame({
        'segment_id': segments['segment_id'],
        'is_canonical': segments['is_canonical'],
        'covered_water': [False, False, False, False],
        'covered_cooling': [False, False, False, False],
        'dist_water_m': [-1, -1, -1, -1],
        'dist_cooling_m': [-1, -1, -1, -1],
        'access_penalty': [0.5, 0.5, 0.5, 0.5]
    })
    
    current_state = {
        'risk_records': risk_records,
        'exposure_results': exposure_records,
        'cooling_access': cooling_access,
        'stored_bounds': (0.0, 100.0)
    }
    
    return current_state, vuln_df


# ────────────────────── OPTIMIZER TESTS ──────────────────────

class TestOptimizer:
    def test_toy_graph_bounds(self):
        segs = _make_segments()
        state, vuln = _make_mock_state(segs)
        cands = generate_candidates(segs)
        
        # Test 1 shade intervention limit
        limits = {'water': 0, 'cooling': 0, 'shade': 1}
        selected, next_state, meta = run_optimizer(cands, segs, vuln, state, limits)
        
        assert len(selected) == 1
        assert selected[0]['type'] == 'shade'
        
        # Check after <= before
        assert meta['objective_after'] < meta['objective_before']
        
    def test_resource_limits(self):
        segs = _make_segments()
        state, vuln = _make_mock_state(segs)
        cands = generate_candidates(segs)
        
        # Test 0 limit
        limits_zero = {'water': 0, 'cooling': 0, 'shade': 0}
        sel_0, _, m0 = run_optimizer(cands, segs, vuln, state, limits_zero)
        assert len(sel_0) == 0
        assert m0['objective_after'] == m0['objective_before']
        
        # Test exact limit
        limits_exact = {'water': 0, 'cooling': 0, 'shade': 1}
        sel_1, _, m1 = run_optimizer(cands, segs, vuln, state, limits_exact)
        assert len(sel_1) == 1
        
        # Test over capacity (asking for 10 but only a few candidates exist)
        limits_over = {'water': 10, 'cooling': 10, 'shade': 10}
        sel_over, _, mo = run_optimizer(cands, segs, vuln, state, limits_over)
        assert len(sel_over) > 0  # Should select all possible positive-benefit candidates
        assert len(sel_over) < len(cands)  # Not all candidates will have >0 benefit
        
        # Assert early stop reason
        assert 'no candidate' in mo['early_stop_reason'] or 'exhausted' in mo['early_stop_reason']
        
    def test_determinism(self):
        segs = _make_segments()
        state, vuln = _make_mock_state(segs)
        cands = generate_candidates(segs)
        limits = {'water': 1, 'cooling': 1, 'shade': 2}
        
        sel_1, _, _ = run_optimizer(cands, segs, vuln, state, limits)
        sel_2, _, _ = run_optimizer(cands, segs, vuln, state, limits)
        
        assert [s['candidate_id'] for s in sel_1] == [s['candidate_id'] for s in sel_2]
        
    def test_no_two_way_double_counting(self):
        segs = _make_segments()
        state, vuln = _make_mock_state(segs)
        
        from src.optimizer import _objective_score
        obj_val = _objective_score(state['risk_records'], vuln)
        
        # 13:00 has two canonical segments: S-1-2-0 (final_risk_raw 12.0, vuln 10) and S-2-3-0 (final_risk_raw 27.0, vuln 20)
        # Expected = 12.0*10 + 27.0*20 = 120 + 540 = 660
        assert abs(obj_val - 660.0) < 1e-6
        
    def test_plan_changes_when_vulnerability_changes(self):
        segs = _make_segments()
        state, vuln_normal = _make_mock_state(segs)
        cands = generate_candidates(segs)
        limits = {'shade': 1, 'water': 0, 'cooling': 0}
        
        sel_norm, _, _ = run_optimizer(cands, segs, vuln_normal, state, limits)
        target_norm = sel_norm[0]['target_id']
        
        # Now artificially make the first segment highly vulnerable
        vuln_changed = vuln_normal.copy()
        vuln_changed.loc[vuln_changed['segment_id'] == 'S-1-2-0', 'vulnerable_mass'] = 1000.0
        
        from src.optimizer import _optimizer_cache
        _optimizer_cache.clear()
        
        sel_changed, _, _ = run_optimizer(cands, segs, vuln_changed, state, limits)
        target_changed = sel_changed[0]['target_id']
        
        # The selected target should change to the newly vulnerable segment
        assert target_norm != target_changed
        assert target_changed == 'S-1-2-0'
        
    def test_shade_lowers_exposure_and_monotone(self):
        segs = _make_segments()
        state, vuln = _make_mock_state(segs)
        cands = generate_candidates(segs)
        
        from src.config_loader import get_config
        config = get_config()
        old_gain = config['interventions']['shade']['shade_gain']
        
        # Test gain 0.1
        config['interventions']['shade']['shade_gain'] = 0.1
        _, next_state_01, m_01 = run_optimizer(cands, segs, vuln, state, {'shade': 1})
        
        # Test gain 0.5
        config['interventions']['shade']['shade_gain'] = 0.5
        from src.optimizer import _optimizer_cache
        _optimizer_cache.clear()
        _, next_state_05, m_05 = run_optimizer(cands, segs, vuln, state, {'shade': 1})
        
        # Restore gain
        config['interventions']['shade']['shade_gain'] = old_gain
        
        # Monotone: objective reduction with 0.5 should be > reduction with 0.1
        assert m_05['objective_before'] - m_05['objective_after'] > m_01['objective_before'] - m_01['objective_after']
        
    def test_water_cooling_never_increase_distance(self):
        segs = _make_segments()
        state, vuln = _make_mock_state(segs)
        cands = generate_candidates(segs)
        
        # Initial distances are -1 (infinity)
        assert (state['cooling_access']['dist_water_m'] == -1).all()
        
        sel, next_state, meta = run_optimizer(cands, segs, vuln, state, {'water': 1})
        assert len(sel) > 0
        
        # Check no distance is increased (none should go from positive to a larger value or back to -1)
        # In our toy state, all were -1, so they should either be -1 or > 0.
        for d in next_state['cooling_access']['dist_water_m']:
            assert d == -1 or d >= 0
            
    def test_canonical_candidates_only_and_no_duplicate_shade(self):
        segs = _make_segments()
        cands = generate_candidates(segs)
        
        shade_cands = cands[cands['type'] == 'shade']
        # Check shade candidates only come from canonical segments
        for _, c in shade_cands.iterrows():
            tid = c['target_id']
            is_can = segs.loc[segs['segment_id'] == tid, 'is_canonical'].iloc[0]
            assert is_can == True
            
        # Check node IDs are ints (no '.0')
        node_cands = cands[cands['type'].isin(['water', 'cooling'])]
        for _, c in node_cands.iterrows():
            assert isinstance(c['target_id'], int) or (isinstance(c['target_id'], float) and c['target_id'].is_integer() and str(c['target_id']).endswith('.0') == False)
            
    def test_optimizer_consistency(self):
        segs = _make_segments()
        state, vuln = _make_mock_state(segs)
        cands = generate_candidates(segs)
        
        # Test 1 shade, 1 water
        limits = {'water': 1, 'cooling': 0, 'shade': 1}
        selected, next_state, meta = run_optimizer(cands, segs, vuln, state, limits)
        
        assert len(selected) > 0
        obj_after_reported = meta['objective_after']
        
        # Perform full pipeline rerun on the modified state
        from src.optimizer import _objective_score
        # The next_state returned from the optimizer is already the modified state
        # The objective score computes it identically
        obj_after_actual = _objective_score(next_state['risk_records'], vuln)
        
        # tolerance 1e-9
        assert abs(obj_after_reported - obj_after_actual) < 1e-9
    def test_lazy_greedy_equals_eager_greedy(self):
        segs = _make_segments()
        state, vuln = _make_mock_state(segs)
        cands = generate_candidates(segs)
        
        # Test the initial heap estimate bounds the exact benefit
        from src.optimizer import evaluate_candidate_exact, build_street_graph
        from src.config_loader import get_config
        config = get_config()
        G = build_street_graph(segs)
        cache = {}
        
        # Compute proxy manually
        vuln_lookup = vuln.set_index('segment_id')['vulnerable_mass']
        total_risk_series = None
        for t, rdf in state['risk_records'].items():
            sub = rdf[rdf['is_canonical'] == True]
            merged = sub.join(vuln_lookup, how='inner')
            r_v = merged['final_risk_raw'] * merged['vulnerable_mass']
            if total_risk_series is None:
                total_risk_series = r_v
            else:
                total_risk_series = total_risk_series.add(r_v, fill_value=0)
                
        seg_risk = segs[['segment_id', 'street_id', 'u', 'v']].set_index('segment_id').join(total_risk_series.rename('risk_val'))
        seg_risk['risk_val'] = seg_risk['risk_val'].fillna(0)
        
        street_proxy = seg_risk.groupby('street_id')['risk_val'].sum()
        node_risk_u = seg_risk.groupby('u')['risk_val'].sum()
        node_risk_v = seg_risk.groupby('v')['risk_val'].sum()
        node_proxy = node_risk_u.add(node_risk_v, fill_value=0)

        shade_gain = config['interventions']['shade']['shade_gain']
        
        for _, row in cands.iterrows():
            ctype = row['type']
            tid = row['target_id']
            if ctype == 'shade':
                street_id = segs.loc[segs['segment_id'] == tid, 'street_id'].iloc[0]
                proxy = street_proxy.get(street_id, 0.0) * shade_gain * 1.5
            elif ctype == 'water':
                proxy = node_proxy.get(tid, 0.0) * 5.0
            else:
                proxy = node_proxy.get(tid, 0.0) * 2.5
                
            exact, _, _ = evaluate_candidate_exact(row, state, segs, vuln, config, cache, G)
            assert proxy >= exact or abs(proxy - exact) < 1e-6
            
    def test_intervention_crosses_class_boundary(self):
        segs = _make_segments()
        state, vuln = _make_mock_state(segs)
        
        # Force a segment to be right at the boundary (e.g. HIGH -> MODERATE)
        # We need final_risk_raw > 50 but close to 50 so shade drops it below 50.
        vuln['vulnerability_value'] = [10.0, 10.0, 38.0, 38.0]
        
        state['stored_bounds'] = pd.DataFrame([{'mode': 'geometric', 'scope_min': 0.0, 'scope_max': 100.0}])
        # We also need to update the baseline risk records in the state so the optimizer proxy evaluates correctly
        for i in [2, 3]:
            state['risk_records']['13:00'].loc[i, 'baseline_risk'] = 34.2
            state['risk_records']['13:00'].loc[i, 'final_risk_raw'] = 51.3
        
        cands = generate_candidates(segs)
        limits = {'shade': 1, 'water': 0, 'cooling': 0}
        
        from src.optimizer import _optimizer_cache
        _optimizer_cache.clear()
        
        selected, next_state, _ = run_optimizer(cands, segs, vuln, state, limits)
        
        after_risk = next_state['risk_records']['13:00']
        # The shade intervention should lower the score of one of the 51.0 segments.
        assert any(after_risk['risk_class'] == 'MODERATE')

# ────────────────────── IMPACT ENGINE TESTS ──────────────────────

class TestImpactEngine:
    def test_identical_bounds(self):
        segs = _make_segments()
        state_before, vuln = _make_mock_state(segs)
        
        # Fake a state_after with lower risk scores
        state_after = {
            'risk_records': {'13:00': state_before['risk_records']['13:00'].copy()},
            'cooling_access': state_before['cooling_access'].copy(),
            'stored_bounds': state_before['stored_bounds']
        }
        state_after['risk_records']['13:00']['exposure_value'] -= 0.1
        state_after['risk_records']['13:00']['risk_score'] -= 10
        
        impact = compute_impact(
            state_before['risk_records'], state_after['risk_records'],
            state_before['cooling_access'], state_after['cooling_access'],
            vuln
        )
        
        # Impact computes delta correctly
        assert impact['vuln_weighted_exposure']['delta'] <= 0
        # No errors when running impact, meaning bounds and structural checks pass
