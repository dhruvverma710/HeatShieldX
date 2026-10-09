"""
Routing Engine for HeatShield X.
Computes FASTEST, HEAT-AWARE, and BALANCED walking routes using the same
street segments and exposure data as the risk system.
"""
import logging
import networkx as nx
import numpy as np
import pandas as pd
import uuid
from src.config_loader import get_config

logger = logging.getLogger(__name__)


class RoutingError(Exception):
    pass


def _get_exposure_class(exposure_score, config):
    """
    Map the route's mean exposure_value (0-1) to the existing risk class bands.
    Mapping: exposure_score * 100 → risk_class_ranges thresholds.
    """
    score_100 = exposure_score * 100
    ranges = config.get('risk_class_ranges', {
        'LOW': [0, 25], 'MODERATE': [26, 50], 'HIGH': [51, 75], 'CRITICAL': [76, 100]
    })
    for cls in ('LOW', 'MODERATE', 'HIGH', 'CRITICAL'):
        bounds = ranges.get(cls)
        if bounds and len(bounds) == 2 and bounds[0] <= score_100 <= bounds[1]:
            return cls
    if score_100 > 75:
        return 'CRITICAL'
    return 'LOW'


def build_routing_graph(segments):
    """
    Build a directed NetworkX graph from segments for routing.
    Respects one-way streets. Adds reverse edge for bidirectional streets.
    If the street network is disconnected, restricts to the largest
    weakly connected component and reports the share of segments kept.
    """
    G = nx.DiGraph()

    total_edges = 0
    for _, row in segments.iterrows():
        u = row['u']
        v = row['v']
        if pd.isna(u) or pd.isna(v):
            continue
        u = int(u)
        v = int(v)
        length_m = float(row['length_m'])
        sid = row['segment_id']
        oneway = row.get('oneway', False)

        G.add_edge(u, v, segment_id=sid, length_m=length_m)
        total_edges += 1
        if not oneway:
            G.add_edge(v, u, segment_id=sid, length_m=length_m)
            total_edges += 1

    # Restrict to largest weakly connected component
    if not nx.is_weakly_connected(G):
        largest_wcc = max(nx.weakly_connected_components(G), key=len)
        kept_edges = G.subgraph(largest_wcc).number_of_edges()
        share = (kept_edges / total_edges * 100) if total_edges > 0 else 0
        logger.info(
            f"Routing graph disconnected. Keeping largest component: "
            f"{kept_edges}/{total_edges} edges ({share:.1f}%)."
        )
        G = G.subgraph(largest_wcc).copy()
    else:
        logger.info(f"Routing graph is connected. {G.number_of_nodes()} nodes, {total_edges} edges.")

    return G


def apply_snapshot_to_graph(G, snapshot_df, config):
    """
    Attach time_s, exposure_value, shade_fraction, access_penalty, heat_load
    to every edge of G from the snapshot dataframe.
    Duplicates carry identical values since they share the same segment_id.
    """
    speed_mps = config.get('routing', {}).get('walking_speed_mps', 1.3)

    lookup = {}
    for _, row in snapshot_df.iterrows():
        lookup[row['segment_id']] = row

    defaults = {
        'exposure_value': 0.0, 'shade_fraction': 0.0, 'access_penalty': 0.0,
        'covered_water': False, 'covered_cooling': False
    }

    for u, v, data in G.edges(data=True):
        sid = data.get('segment_id')
        row = lookup.get(sid, defaults)

        data['time_s'] = data['length_m'] / speed_mps
        data['exposure_value'] = float(row.get('exposure_value', 0.0))
        data['shade_fraction'] = float(row.get('shade_fraction', 0.0))
        data['access_penalty'] = float(row.get('access_penalty', 0.0))
        data['covered_water'] = bool(row.get('covered_water', False))
        data['covered_cooling'] = bool(row.get('covered_cooling', False))
        data['heat_load'] = data['exposure_value'] * data['length_m']

    return G


def extract_route_metrics(G, path, route_types, time_status, comp_mode,
                          config_version, fastest_metrics=None):
    """Extract all route metrics from a node path through G."""
    if not path or len(path) < 2:
        return None

    total_time_s = 0.0
    total_dist = 0.0
    exposure_load = 0.0
    shade_load = 0.0
    access_load = 0.0
    water_covered_dist = 0.0
    cooling_covered_dist = 0.0
    segment_ids = []

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        data = G[u][v]
        segment_ids.append(data['segment_id'])
        length = data['length_m']
        total_dist += length
        total_time_s += data['time_s']
        exposure_load += data['heat_load']
        shade_load += data['shade_fraction'] * length
        access_load += data['access_penalty'] * length
        if data['covered_water']:
            water_covered_dist += length
        if data['covered_cooling']:
            cooling_covered_dist += length

    total_time_min = total_time_s / 60.0
    mean_exposure = exposure_load / total_dist if total_dist > 0 else 0.0
    mean_shade = shade_load / total_dist if total_dist > 0 else 0.0
    mean_access = access_load / total_dist if total_dist > 0 else 0.0

    config = get_config()
    exp_class = _get_exposure_class(mean_exposure, config)

    route = {
        'route_id': str(uuid.uuid4()),
        'type_labels': list(route_types),
        'segment_ids': segment_ids,
        'geometry': [],
        'total_time_min': round(total_time_min, 2),
        'total_distance_m': round(total_dist, 1),
        'modelled_heat_exposure': round(mean_exposure, 4),
        'exposure_load_total': round(exposure_load, 4),
        'weighted_shade': round(mean_shade, 4),
        'cooling_access': {
            'mean_access_penalty': round(mean_access, 4),
            'water_coverage_share': round(
                water_covered_dist / total_dist if total_dist > 0 else 0.0, 4),
            'cooling_coverage_share': round(
                cooling_covered_dist / total_dist if total_dist > 0 else 0.0, 4),
        },
        'exposure_class': exp_class,
        'time_status': time_status,
        'status': 'MODELLED',
        'config_version': config_version,
        'computation_mode': comp_mode,
    }

    # Delta vs fastest and recommendation text
    if fastest_metrics:
        t_delta = total_time_min - fastest_metrics['total_time_min']
        if fastest_metrics['modelled_heat_exposure'] > 0:
            e_delta = ((mean_exposure - fastest_metrics['modelled_heat_exposure'])
                       / fastest_metrics['modelled_heat_exposure'] * 100)
        else:
            e_delta = 0.0
        route['delta'] = {
            'time_minutes': round(t_delta, 2),
            'exposure_percent': round(e_delta, 1),
        }
        if e_delta < -0.5:
            route['recommendation_text'] = (
                f"Lower modelled heat exposure ({e_delta:.0f}%); "
                f"adds about {t_delta:.1f} min."
            )
        else:
            route['recommendation_text'] = (
                "No lower-exposure route found within the detour limit."
            )
    else:
        route['delta'] = {'time_minutes': 0.0, 'exposure_percent': 0.0}
        route['recommendation_text'] = ""

    return route


def compute_routes(G, origin, destination, snapshot_df, weight_w=0.5):
    """
    Compute FASTEST, HEAT-AWARE, and BALANCED routes.
    If two routes share the identical edge sequence they are merged into one
    result with combined labels (e.g. "FASTEST + HEAT-AWARE").
    """
    if origin not in G or destination not in G:
        raise RoutingError("Unknown origin or destination node.")

    if origin == destination:
        raise RoutingError(
            "Origin and destination are the same (zero-length route)."
        )

    config = get_config()
    version = config.get('config_version', '2.0.0')
    r_cfg = config.get('routing', {})

    max_detour = r_cfg.get('max_detour_factor', 1.5)
    gamma_sched = r_cfg.get('heat_aware_gamma_schedule', [8, 4, 2, 1])

    time_status = 'COMPUTED'
    comp_mode = 'geometric'
    if 'time_status' in snapshot_df.columns and len(snapshot_df) > 0:
        time_status = snapshot_df['time_status'].iloc[0]
    if 'computation_mode' in snapshot_df.columns and len(snapshot_df) > 0:
        comp_mode = snapshot_df['computation_mode'].iloc[0]

    # ── 1. FASTEST ──
    try:
        fastest_path = nx.shortest_path(G, origin, destination, weight='time_s')
    except nx.NetworkXNoPath:
        raise RoutingError(
            "No path found between the origin and destination."
        )

    fastest_route = extract_route_metrics(
        G, fastest_path, ['FASTEST'], time_status, comp_mode, version, None
    )

    # ── 2. HEAT-AWARE ──
    heat_aware_path = None
    heat_aware_route = None
    for gamma in gamma_sched:
        for u, v, d in G.edges(data=True):
            d['ha_weight'] = d['time_s'] * (1 + gamma * d['exposure_value'])
        try:
            path = nx.shortest_path(
                G, origin, destination, weight='ha_weight'
            )
        except nx.NetworkXNoPath:
            continue
        r = extract_route_metrics(
            G, path, ['HEAT-AWARE'], time_status, comp_mode, version,
            fastest_route
        )
        if r['total_time_min'] <= max_detour * fastest_route['total_time_min']:
            heat_aware_path = path
            heat_aware_route = r
            break

    if heat_aware_path is None:
        # Fallback: no cooler route within detour limit
        heat_aware_path = fastest_path
        heat_aware_route = extract_route_metrics(
            G, fastest_path, ['HEAT-AWARE'], time_status, comp_mode, version,
            fastest_route
        )
        heat_aware_route['recommendation_text'] = (
            "no cooler route within the detour limit"
        )

    # ── 3. BALANCED ──
    alpha = 1.0 - weight_w
    beta = weight_w
    t_fastest = fastest_route['total_time_min'] * 60  # seconds
    h_fastest = fastest_route['exposure_load_total']

    for u, v, d in G.edges(data=True):
        if h_fastest > 0:
            d['bal_weight'] = (
                alpha * (d['time_s'] / t_fastest)
                + beta * (d['heat_load'] / h_fastest)
            )
        else:
            d['bal_weight'] = d['time_s']

    try:
        balanced_path = nx.shortest_path(
            G, origin, destination, weight='bal_weight'
        )
        balanced_route = extract_route_metrics(
            G, balanced_path, ['BALANCED'], time_status, comp_mode, version,
            fastest_route
        )
    except nx.NetworkXNoPath:
        balanced_path = fastest_path
        balanced_route = extract_route_metrics(
            G, fastest_path, ['BALANCED'], time_status, comp_mode, version,
            fastest_route
        )

    # ── Merge identical routes ──
    all_results = [
        ('FASTEST', tuple(fastest_path), fastest_route),
        ('HEAT-AWARE', tuple(heat_aware_path), heat_aware_route),
        ('BALANCED', tuple(balanced_path), balanced_route),
    ]

    merged = {}
    for label, path_tup, route_obj in all_results:
        if path_tup not in merged:
            merged[path_tup] = route_obj
            merged[path_tup]['type_labels'] = [label]
        else:
            if label not in merged[path_tup]['type_labels']:
                merged[path_tup]['type_labels'].append(label)
            # Carry over non-empty recommendation text (e.g. heat-aware fallback)
            rec = route_obj.get('recommendation_text', '')
            if rec and not merged[path_tup].get('recommendation_text'):
                merged[path_tup]['recommendation_text'] = rec

    return list(merged.values())
