"""
Optimizer Engine for HeatShield X.
Lazy greedy optimizer using SRS-faithful marginal benefit evaluation.
Vectorized exact evaluation for speed.
"""
import time as _time
import pandas as pd
import numpy as np
import heapq
import logging
import pickle
from pathlib import Path
from src.config_loader import get_config
from src.cooling_engine import build_street_graph, multi_source_dijkstra, bounded_distance_penalty
from src.provenance import MODELLED

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / 'data' / 'processed'


def _objective_score(risk_records, vuln_df):
    """
    The SRS objective is the vulnerability-weighted final_risk_raw over all canonical segments and times.
    """
    total = 0.0
    vuln_lookup = vuln_df.set_index('segment_id')['vulnerable_mass']
    for t, rdf in risk_records.items():
        canon = rdf[rdf['is_canonical'] == True]
        if 'segment_id' in canon.columns:
            canon = canon.set_index('segment_id')
        merged = canon.join(vuln_lookup, how='left')
        total += (merged['final_risk_raw'] * merged['vulnerable_mass'].fillna(0)).sum()
    return float(total)


def _get_affected_segments(candidate, segments, G, config):
    ctype = candidate['type']
    tid = candidate['target_id']

    if ctype == 'shade':
        # Affects target segment (and its non-canonical twin)
        street_id = segments.loc[segments['segment_id'] == tid, 'street_id'].iloc[0]
        affected = segments[segments['street_id'] == street_id]['segment_id'].tolist()
        return affected, None
    else:
        import networkx as nx
        radius = config['water_service_radius'] if ctype == 'water' else config['cooling_service_radius']

        dists = nx.single_source_dijkstra_path_length(G, tid, cutoff=radius, weight='weight')
        affected_nodes = set(dists.keys())

        affected = []
        new_dists = {}

        if affected_nodes:
            # Only iterate over segments connecting to affected nodes
            affected_segs = segments[segments['u'].isin(affected_nodes) | segments['v'].isin(affected_nodes)]
            for _, row in affected_segs.iterrows():
                sid = row['segment_id']
                u, v = row['u'], row['v']
                du = dists.get(u, float('inf'))
                dv = dists.get(v, float('inf'))
                dist = min(du, dv) + 0.5 * row['length_m']

                if dist <= radius:
                    affected.append(sid)
                    new_dists[sid] = dist

        return affected, new_dists


class _VectorState:
    """Numpy-backed state for fast exact evaluation."""

    def __init__(self, current_state, segments, vuln_df, config):
        # Build segment_id -> integer index mapping from segments
        all_sids = segments['segment_id'].values
        self.sid_to_idx = {sid: i for i, sid in enumerate(all_sids)}
        n = len(all_sids)
        self.n = n

        # Canonical mask
        self.is_canonical = segments['is_canonical'].values.astype(bool)

        # Vulnerability lookup (aligned by segment index)
        vuln_lookup = vuln_df.set_index('segment_id')['vulnerable_mass']
        self.vuln_mass = np.zeros(n)
        for sid, idx in self.sid_to_idx.items():
            if sid in vuln_lookup.index:
                self.vuln_mass[idx] = vuln_lookup[sid]

        # Street_id per segment (for shade)
        self.street_ids = segments['street_id'].values

        # Risk arrays per time: final_risk_raw, baseline_risk, exposure_value
        self.times = sorted(current_state['risk_records'].keys())
        self.final_risk_raw = {}
        self.baseline_risk = {}
        self.exposure_value = {}

        for t in self.times:
            rdf = current_state['risk_records'][t]
            if 'segment_id' in rdf.columns:
                rdf_idx = rdf.set_index('segment_id')
            else:
                rdf_idx = rdf

            frr = np.zeros(n)
            br = np.zeros(n)
            ev = np.zeros(n)
            for sid, idx in self.sid_to_idx.items():
                if sid in rdf_idx.index:
                    row = rdf_idx.loc[sid]
                    frr[idx] = float(row['final_risk_raw'])
                    br[idx] = float(row['baseline_risk'])
                    ev[idx] = float(row['exposure_value'])
            self.final_risk_raw[t] = frr.copy()
            self.baseline_risk[t] = br.copy()
            self.exposure_value[t] = ev.copy()

        # Cooling access arrays
        cool = current_state['cooling_access']
        if 'segment_id' in cool.columns:
            cool_idx = cool.set_index('segment_id')
        else:
            cool_idx = cool

        self.dist_water = np.full(n, -1.0)
        self.dist_cooling = np.full(n, -1.0)
        self.covered_water = np.zeros(n, dtype=bool)
        self.covered_cooling = np.zeros(n, dtype=bool)
        self.access_penalty = np.zeros(n)

        for sid, idx in self.sid_to_idx.items():
            if sid in cool_idx.index:
                row = cool_idx.loc[sid]
                self.dist_water[idx] = float(row['dist_water_m'])
                self.dist_cooling[idx] = float(row['dist_cooling_m'])
                self.covered_water[idx] = bool(row['covered_water'])
                self.covered_cooling[idx] = bool(row['covered_cooling'])
                if 'access_penalty' in cool_idx.columns:
                    self.access_penalty[idx] = float(row['access_penalty'])

        self.config = config
        self.stored_bounds = current_state['stored_bounds']

    def objective(self):
        """Compute total objective over canonical segments and all times."""
        total = 0.0
        mask = self.is_canonical
        for t in self.times:
            total += (self.final_risk_raw[t][mask] * self.vuln_mass[mask]).sum()
        return float(total)

    def partial_objective(self, indices):
        """Compute objective contribution for a subset of segment indices (canonical only)."""
        total = 0.0
        mask = self.is_canonical[indices]
        for t in self.times:
            total += (self.final_risk_raw[t][indices][mask] * self.vuln_mass[indices][mask]).sum()
        return float(total)

    def evaluate_shade(self, affected_indices, shade_gain):
        """Evaluate shade intervention benefit. Returns benefit and new risk values."""
        base_obj = self.partial_objective(affected_indices)

        new_frr = {}
        new_obj = 0.0
        canon = self.is_canonical[affected_indices]
        vm = self.vuln_mass[affected_indices]

        for t in self.times:
            new_vals = self.final_risk_raw[t][affected_indices] * (1 - shade_gain)
            new_frr[t] = new_vals
            new_obj += (new_vals[canon] * vm[canon]).sum()

        return float(base_obj - new_obj), new_frr

    def evaluate_water_cooling(self, affected_indices, new_dists_by_idx, ctype):
        """Evaluate water/cooling intervention benefit."""
        base_obj = self.partial_objective(affected_indices)

        config = self.config
        radius = config['water_service_radius'] if ctype == 'water' else config['cooling_service_radius']
        penalty_w = config['access_penalty_weights']
        max_penalty = config['max_access_penalty']

        # Copy current distances for affected
        new_dw = self.dist_water[affected_indices].copy()
        new_dc = self.dist_cooling[affected_indices].copy()
        new_cw = self.covered_water[affected_indices].copy()
        new_cc = self.covered_cooling[affected_indices].copy()

        for local_i, global_i in enumerate(affected_indices):
            if global_i in new_dists_by_idx:
                new_d = new_dists_by_idx[global_i]
                if ctype == 'water':
                    old_d = new_dw[local_i]
                    if old_d == -1 or new_d < old_d:
                        new_dw[local_i] = new_d
                        new_cw[local_i] = True
                else:
                    old_d = new_dc[local_i]
                    if old_d == -1 or new_d < old_d:
                        new_dc[local_i] = new_d
                        new_cc[local_i] = True

        # Recompute access penalty for affected segments
        pw = np.where(new_dw != -1, bounded_distance_penalty(new_dw, config['water_service_radius']), 1.0)
        pc = np.where(new_dc != -1, bounded_distance_penalty(new_dc, config['cooling_service_radius']), 1.0)
        avail = np.where((new_dw != -1) | (new_dc != -1), 0.0, 1.0)
        new_ap = np.clip(
            penalty_w['water'] * pw + penalty_w['cooling'] * pc + penalty_w['availability'] * avail,
            0, max_penalty
        )

        # Recompute final_risk_raw for affected segments
        new_frr = {}
        new_obj = 0.0
        canon = self.is_canonical[affected_indices]
        vm = self.vuln_mass[affected_indices]

        for t in self.times:
            br = self.baseline_risk[t][affected_indices]
            new_vals = br * (1 + new_ap)
            new_frr[t] = new_vals
            new_obj += (new_vals[canon] * vm[canon]).sum()

        return float(base_obj - new_obj), new_frr, new_ap, new_dw, new_dc, new_cw, new_cc

    def apply_shade(self, affected_indices, shade_gain):
        """Apply shade effect to state."""
        for t in self.times:
            self.final_risk_raw[t][affected_indices] *= (1 - shade_gain)
            self.baseline_risk[t][affected_indices] *= (1 - shade_gain)
            self.exposure_value[t][affected_indices] *= (1 - shade_gain)

    def apply_water_cooling(self, affected_indices, new_ap, new_dw, new_dc, new_cw, new_cc, new_frr):
        """Apply water/cooling effect to state."""
        self.dist_water[affected_indices] = new_dw
        self.dist_cooling[affected_indices] = new_dc
        self.covered_water[affected_indices] = new_cw
        self.covered_cooling[affected_indices] = new_cc
        self.access_penalty[affected_indices] = new_ap
        for t in self.times:
            self.final_risk_raw[t][affected_indices] = new_frr[t]

    def to_state_dict(self, original_state, segments):
        """Convert back to the dict-of-DataFrames format for downstream use."""
        from src.risk_engine import normalize_score, classify_risk
        config = self.config

        s_bounds = self.stored_bounds
        if isinstance(s_bounds, pd.DataFrame):
            geom = s_bounds[s_bounds['mode'] == 'geometric']
            if not geom.empty:
                s_bounds = (float(geom.iloc[0]['scope_min']), float(geom.iloc[0]['scope_max']))
            else:
                s_bounds = (float(s_bounds.iloc[0]['scope_min']), float(s_bounds.iloc[0]['scope_max']))

        class_ranges = config['risk_class_ranges']
        all_sids = segments['segment_id'].values

        risk_records = {}
        for t in self.times:
            rdf = original_state['risk_records'][t].copy()
            if 'segment_id' not in rdf.columns:
                rdf = rdf.reset_index()
            # Update values from vector state
            sid_series = rdf['segment_id']
            for i, sid in enumerate(sid_series):
                idx = self.sid_to_idx[sid]
                rdf.at[i, 'final_risk_raw'] = self.final_risk_raw[t][idx]
                rdf.at[i, 'baseline_risk'] = self.baseline_risk[t][idx]
                rdf.at[i, 'exposure_value'] = self.exposure_value[t][idx]

            raw_vals = rdf['final_risk_raw'].values
            normed, _, _ = normalize_score(raw_vals, canonical_mask=None,
                                           v_min=s_bounds[0], v_max=s_bounds[1],
                                           clip_percentile=None, scale=100.0)
            rdf['risk_score'] = normed
            rdf['risk_class'] = classify_risk(normed, class_ranges)
            risk_records[t] = rdf

        cool = original_state['cooling_access'].copy()
        if 'segment_id' not in cool.columns:
            cool = cool.reset_index()
        for i, sid in enumerate(cool['segment_id']):
            if sid in self.sid_to_idx:
                idx = self.sid_to_idx[sid]
                cool.at[i, 'dist_water_m'] = self.dist_water[idx]
                cool.at[i, 'dist_cooling_m'] = self.dist_cooling[idx]
                cool.at[i, 'covered_water'] = self.covered_water[idx]
                cool.at[i, 'covered_cooling'] = self.covered_cooling[idx]
                if 'access_penalty' in cool.columns:
                    cool.at[i, 'access_penalty'] = self.access_penalty[idx]

        return {
            'risk_records': risk_records,
            'cooling_access': cool,
            'stored_bounds': self.stored_bounds
        }


def evaluate_candidate_exact(candidate, state, segments, vuln_df, config, affected_cache, G):
    """
    SRS-faithful evaluation.
    Applies the candidate's effect to its affected segments only.
    Recomputes risk/cooling for those segments.
    Returns the delta in objective (positive = reduction in risk).
    """
    cid = candidate['candidate_id']
    ctype = candidate['type']

    if cid not in affected_cache:
        affected_cache[cid] = _get_affected_segments(candidate, segments, G, config)

    affected, new_dists = affected_cache[cid]
    if not affected:
        return 0.0, None, None

    # We only care about canonical affected segments for the objective
    affected_canon = set(segments[segments['segment_id'].isin(affected) & segments['is_canonical']]['segment_id'])
    if not affected_canon:
        return 0.0, None, None

    vuln_subset = vuln_df[vuln_df['segment_id'].isin(affected_canon)]

    base_obj = 0.0
    for t, rdf in state['risk_records'].items():
        sub_rdf = rdf[rdf.index.isin(affected_canon)]
        # We need segment_id as a column for merging with vuln_subset
        merged = sub_rdf.merge(vuln_subset, left_index=True, right_on='segment_id', how='left')
        base_obj += (merged['final_risk_raw'] * merged['vulnerable_mass'].fillna(0)).sum()

    new_obj = 0.0
    bounds = state['stored_bounds']

    new_risk = {}
    new_cool = None

    if ctype == 'shade':
        shade_gain = config['interventions']['shade']['shade_gain']
        for t, rdf in state['risk_records'].items():
            sub_rdf = rdf[rdf.index.isin(affected)].copy()
            sub_rdf['exposure_value'] *= (1 - shade_gain)
            sub_rdf['baseline_risk'] *= (1 - shade_gain)
            sub_rdf['final_risk_raw'] *= (1 - shade_gain)
            new_risk[t] = sub_rdf

            sub_canon = sub_rdf[sub_rdf['is_canonical'] == True]
            merged = sub_canon.merge(vuln_subset, left_index=True, right_on='segment_id', how='left')
            new_obj += (merged['final_risk_raw'] * merged['vulnerable_mass'].fillna(0)).sum()

    elif ctype in ['water', 'cooling']:
        radius = config['water_service_radius'] if ctype == 'water' else config['cooling_service_radius']
        penalty_w = config['access_penalty_weights']
        max_penalty = config['max_access_penalty']

        sub_cool = state['cooling_access'][state['cooling_access'].index.isin(affected)].copy()

        for sid in affected:
            new_d = new_dists[sid]
            if sid not in sub_cool.index:
                continue
            idx = sid

            if ctype == 'water':
                old_d = sub_cool.at[idx, 'dist_water_m']
                if old_d == -1 or new_d < old_d:
                    sub_cool.at[idx, 'dist_water_m'] = new_d
                    sub_cool.at[idx, 'covered_water'] = True
            else:
                old_d = sub_cool.at[idx, 'dist_cooling_m']
                if old_d == -1 or new_d < old_d:
                    sub_cool.at[idx, 'dist_cooling_m'] = new_d
                    sub_cool.at[idx, 'covered_cooling'] = True

            dw = sub_cool.at[idx, 'dist_water_m']
            dc = sub_cool.at[idx, 'dist_cooling_m']

            pw = bounded_distance_penalty(dw, config['water_service_radius']) if dw != -1 else 1.0
            pc = bounded_distance_penalty(dc, config['cooling_service_radius']) if dc != -1 else 1.0
            avail = 0.0 if (dw != -1 or dc != -1) else 1.0

            acc_pen = np.clip(
                penalty_w['water'] * pw + penalty_w['cooling'] * pc + penalty_w['availability'] * avail,
                0, max_penalty
            )
            sub_cool.at[idx, 'access_penalty'] = acc_pen

        new_cool = sub_cool

        for t, rdf in state['risk_records'].items():
            sub_rdf = rdf[rdf.index.isin(affected)].copy()
            sub_rdf['access_penalty'] = sub_cool['access_penalty']
            sub_rdf['final_risk_raw'] = sub_rdf['baseline_risk'] * (1 + sub_rdf['access_penalty'])
            new_risk[t] = sub_rdf

            sub_canon = sub_rdf[sub_rdf['is_canonical'] == True]
            merged = sub_canon.merge(vuln_subset, left_index=True, right_on='segment_id', how='left')
            new_obj += (merged['final_risk_raw'] * merged['vulnerable_mass'].fillna(0)).sum()

    benefit = base_obj - new_obj
    return benefit, new_risk, new_cool


def _apply_effect(state, new_risk, new_cool):
    """Update state with affected subsets."""
    for t, rdf in new_risk.items():
        state['risk_records'][t].update(rdf)
    if new_cool is not None:
        state['cooling_access'].update(new_cool)


_optimizer_cache = {}


def _plan_cache_path(cache_key_str):
    """Path for on-disk optimizer plan cache."""
    return PROCESSED_DIR / f"optimizer_plan_{cache_key_str}.pkl"


def _make_cache_key(resource_limits, mode, config_ver):
    """Deterministic cache key for optimizer plans."""
    return (frozenset(resource_limits.items()), mode, config_ver)


def _cache_key_str(cache_key):
    """String representation of cache key for file naming."""
    limits = dict(cache_key[0])
    parts = [f"{k}{v}" for k, v in sorted(limits.items())]
    return f"{'_'.join(parts)}_{cache_key[1]}_{cache_key[2]}"


def load_precomputed_plan(resource_limits, mode, config_ver):
    """Load a precomputed plan from disk if available."""
    ck = _make_cache_key(resource_limits, mode, config_ver)
    path = _plan_cache_path(_cache_key_str(ck))
    if path.exists():
        try:
            with open(path, 'rb') as f:
                result = pickle.load(f)
            logger.info(f"Loaded precomputed plan from {path.name}")
            return result
        except Exception as e:
            logger.warning(f"Failed to load precomputed plan: {e}")
    return None


def save_precomputed_plan(resource_limits, mode, config_ver, result):
    """Save optimizer plan to disk."""
    ck = _make_cache_key(resource_limits, mode, config_ver)
    path = _plan_cache_path(_cache_key_str(ck))
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(result, f)
    logger.info(f"Saved precomputed plan to {path.name}")


def run_optimizer(candidates_df, segments, vuln_df, current_state, resource_limits):
    """Lazy greedy optimizer with vectorized exact marginal benefit evaluation."""
    t_start = _time.time()
    logger.info(f"Running optimizer with limits: {resource_limits}")
    config = get_config()
    time_budget = config.get('impact', {}).get('optimizer_time_budget_s', 120)

    # Determine mode for cache key
    first_time = next(iter(current_state['risk_records']))
    rdf_first = current_state['risk_records'][first_time]
    if 'computation_mode' in rdf_first.columns:
        mode = rdf_first['computation_mode'].iloc[0]
    elif 'computation_mode' in rdf_first.index.names:
        mode = 'geometric'
    else:
        mode = 'geometric'

    config_ver = config.get('config_version', '2.0')
    cache_key = _make_cache_key(resource_limits, mode, config_ver)

    # Check in-memory cache
    if cache_key in _optimizer_cache:
        logger.info("Returning cached optimizer plan (in-memory).")
        import copy
        return copy.deepcopy(_optimizer_cache[cache_key])

    # Check on-disk precomputed plan
    disk_plan = load_precomputed_plan(resource_limits, mode, config_ver)
    if disk_plan is not None:
        import copy
        _optimizer_cache[cache_key] = disk_plan
        return copy.deepcopy(disk_plan)

    selected = []

    limits = resource_limits.copy()

    available_types = [k for k, v in limits.items() if v > 0]
    if not available_types:
        obj_before = _objective_score(current_state['risk_records'], vuln_df)
        meta = {
            'objective_before': obj_before,
            'objective_after': obj_before,
            'runtime_s': _time.time() - t_start,
            'early_stop_reason': 'all limits zero',
            'n_candidates_evaluated': 0,
        }
        return selected, current_state, meta

    cands = candidates_df[candidates_df['type'].isin(available_types)].copy()

    # ── Top-K Pruning ──
    top_k = config.get('interventions', {}).get('candidate_pruning_top_k', None)

    # Build vectorized state
    vstate = _VectorState(current_state, segments, vuln_df, config)
    obj_before = vstate.objective()

    n_evaluated = 0
    early_stop_reason = 'all candidates exhausted'

    G = build_street_graph(segments)
    affected_cache = {}

    # --- Fast Vectorized Proxy Initialization ---
    logger.info("Starting proxy initialization")
    # 1. Compute total objective (risk * vuln) per canonical segment across all times
    total_risk = np.zeros(vstate.n)
    for t in vstate.times:
        total_risk += vstate.final_risk_raw[t] * vstate.vuln_mass * vstate.is_canonical

    # Join with segments to get street_id, u, v
    seg_risk_by_street = {}
    node_risk = {}
    for i in range(vstate.n):
        rv = total_risk[i]
        sid_street = vstate.street_ids[i]
        seg_risk_by_street[sid_street] = seg_risk_by_street.get(sid_street, 0.0) + rv

        u_val = segments.iloc[i]['u']
        v_val = segments.iloc[i]['v']
        node_risk[u_val] = node_risk.get(u_val, 0.0) + rv
        node_risk[v_val] = node_risk.get(v_val, 0.0) + rv

    shade_gain = config['interventions']['shade']['shade_gain']

    logger.info("Computing proxy estimates")

    # Compute proxy for each candidate and optionally prune
    proxy_list = []
    for _, row in cands.iterrows():
        ctype = row['type']
        tid = row['target_id']
        if ctype == 'shade':
            street_id = segments.loc[segments['segment_id'] == tid, 'street_id'].iloc[0]
            proxy_ben = seg_risk_by_street.get(street_id, 0.0) * shade_gain * 1.5
        elif ctype == 'water':
            proxy_ben = node_risk.get(tid, 0.0) * 5.0
        else:
            proxy_ben = node_risk.get(tid, 0.0) * 2.5

        if proxy_ben > 0:
            proxy_list.append((-proxy_ben, row['candidate_id'], dict(row)))

    # Sort by proxy (descending benefit)
    proxy_list.sort()

    # Top-K pruning per type if configured
    if top_k is not None:
        type_counts = {}
        pruned = []
        for neg_ben, cid, cand in proxy_list:
            ct = cand['type']
            type_counts[ct] = type_counts.get(ct, 0) + 1
            if type_counts[ct] <= top_k:
                pruned.append((neg_ben, cid, cand))
        logger.info(f"Pruned candidates from {len(proxy_list)} to {len(pruned)} (top_k={top_k})")
        proxy_list = pruned

    heap = []
    for neg_ben, cid, cand in proxy_list:
        heapq.heappush(heap, (neg_ben, cid, cand, False))

    logger.info(f"Heap initialized with {len(heap)} candidates. Starting lazy greedy loop.")
    cumulative_benefit = 0.0
    shaded_streets = set()
    pops = 0
    exacts = 0
    total_exact_time = 0.0

    while heap and any(v > 0 for v in limits.values()):
        elapsed = _time.time() - t_start
        if elapsed > time_budget:
            early_stop_reason = f'time budget exceeded ({elapsed:.1f}s > {time_budget}s)'
            logger.info(early_stop_reason)
            break

        neg_ben, cid, cand, is_fresh = heapq.heappop(heap)
        pops += 1

        if pops % 200 == 0:
            logger.info(f"Popped {pops}, exacts {exacts}, elapsed {elapsed:.1f}s")

        ctype = cand['type']

        if limits.get(ctype, 0) <= 0:
            continue

        if ctype == 'shade':
            street_id = segments.loc[segments['segment_id'] == cand['target_id'], 'street_id'].iloc[0]
            if street_id in shaded_streets:
                continue

        if not is_fresh:
            # Evaluate exact using vectorized state
            n_evaluated += 1
            exacts += 1
            t_eval_start = _time.time()

            # Get affected segments
            if cid not in affected_cache:
                affected_cache[cid] = _get_affected_segments(cand, segments, G, config)
            affected, new_dists = affected_cache[cid]

            if not affected:
                total_exact_time += _time.time() - t_eval_start
                continue

            # Convert to indices
            affected_indices = np.array([vstate.sid_to_idx[sid] for sid in affected if sid in vstate.sid_to_idx])
            if len(affected_indices) == 0:
                total_exact_time += _time.time() - t_eval_start
                continue

            # Check if any canonical segments are affected
            if not vstate.is_canonical[affected_indices].any():
                total_exact_time += _time.time() - t_eval_start
                continue

            if ctype == 'shade':
                new_ben, _ = vstate.evaluate_shade(affected_indices, shade_gain)
            else:
                # Convert new_dists from sid -> dist to global_idx -> dist
                new_dists_by_idx = {}
                if new_dists:
                    for sid, d in new_dists.items():
                        if sid in vstate.sid_to_idx:
                            new_dists_by_idx[vstate.sid_to_idx[sid]] = d
                new_ben, _, _, _, _, _, _ = vstate.evaluate_water_cooling(affected_indices, new_dists_by_idx, ctype)

            total_exact_time += _time.time() - t_eval_start

            if new_ben > 0:
                heapq.heappush(heap, (-new_ben, cid, cand, True))
            continue

        # Select this candidate
        ben = -neg_ben
        cumulative_benefit += ben
        cand['marginal_benefit'] = ben
        cand['cumulative_benefit'] = cumulative_benefit

        # Get affected segments for application
        if cid not in affected_cache:
            affected_cache[cid] = _get_affected_segments(cand, segments, G, config)
        affected, new_dists = affected_cache[cid]
        affected_indices = np.array([vstate.sid_to_idx[sid] for sid in affected if sid in vstate.sid_to_idx])

        if ctype == 'shade':
            cand['newly_covered_streets'] = 1
            street_id = segments.loc[segments['segment_id'] == cand['target_id'], 'street_id'].iloc[0]
            shaded_streets.add(street_id)
            vstate.apply_shade(affected_indices, shade_gain)
        else:
            # Count newly covered
            before_cov = vstate.covered_water[affected_indices].sum() if ctype == 'water' else vstate.covered_cooling[affected_indices].sum()

            new_dists_by_idx = {}
            if new_dists:
                for sid, d in new_dists.items():
                    if sid in vstate.sid_to_idx:
                        new_dists_by_idx[vstate.sid_to_idx[sid]] = d

            _, new_frr, new_ap, new_dw, new_dc, new_cw, new_cc = vstate.evaluate_water_cooling(
                affected_indices, new_dists_by_idx, ctype)

            after_cov = new_cw.sum() if ctype == 'water' else new_cc.sum()
            cand['newly_covered_streets'] = int(after_cov - before_cov)

            vstate.apply_water_cooling(affected_indices, new_ap, new_dw, new_dc, new_cw, new_cc, new_frr)

        selected.append(cand)
        limits[ctype] -= 1

        # Mark remaining heap as stale (lazy greedy)
        new_heap = []
        for nb, i_cid, i_cand, i_fresh in heap:
            new_heap.append((nb, i_cid, i_cand, False))
        heapq.heapify(new_heap)
        heap = new_heap

    if not heap and any(v > 0 for v in limits.values()):
        early_stop_reason = 'no candidate with positive benefit'

    runtime = _time.time() - t_start
    obj_after = vstate.objective()

    logger.info(f"Optimizer done: {exacts} exact evaluations in {total_exact_time:.2f}s "
                f"({total_exact_time/max(exacts,1)*1000:.1f}ms/eval), runtime={runtime:.1f}s")

    # Convert vectorized state back to DataFrames
    final_state = vstate.to_state_dict(current_state, segments)

    meta = {
        'objective_before': obj_before,
        'objective_after': obj_after,
        'runtime_s': runtime,
        'early_stop_reason': early_stop_reason,
        'n_candidates_evaluated': n_evaluated,
        'n_exact_evaluations': exacts,
        'total_exact_time_s': total_exact_time,
        'ms_per_exact_eval': total_exact_time / max(exacts, 1) * 1000,
    }

    # Cache the plan
    _optimizer_cache[cache_key] = (selected, final_state, meta)

    return selected, final_state, meta
