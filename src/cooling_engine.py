"""
Cooling Access Engine for HeatShield X.
Loads water and cooling facilities from OSM, adds synthetic fallback,
computes network walking distances via multi-source Dijkstra,
and produces CoolingAccessRecords with access_penalty.
"""
import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
import logging
from pathlib import Path
from shapely.geometry import Point
from src.config_loader import get_config
from src.data_engine import _get_area_hash, RAW_DIR
from src.provenance import MODELLED, ESTIMATED

logger = logging.getLogger(__name__)

# ──────────────────────── central penalty function ────────────────────────
def bounded_distance_penalty(distance, radius):
    """clip(d / R, 0, 1) — the ONE central bounded penalty function."""
    if radius <= 0:
        return 1.0
    return float(np.clip(distance / radius, 0, 1))


# ──────────────────────── facility loading ────────────────────────────────
def _load_osm_facilities(area, tags, cache_path):
    """Load POIs from OSM using configured tags; cache result."""
    if cache_path.exists():
        logger.info(f"Loading facilities from cache: {cache_path}")
        return gpd.read_parquet(cache_path)
    logger.info(f"Downloading facilities for {area} from OSM...")
    try:
        import osmnx as ox
        gdf = ox.features_from_place(area, tags=tags)
        # Keep only Points, or compute centroids of polygons
        gdf['geometry'] = gdf.geometry.centroid
        for col in gdf.columns:
            if gdf[col].apply(lambda x: isinstance(x, list)).any():
                gdf[col] = gdf[col].astype(str)
        gdf.to_parquet(cache_path)
        return gdf
    except Exception as e:
        logger.warning(f"OSM facility download failed: {e}")
        if cache_path.exists():
            return gpd.read_parquet(cache_path)
        return gpd.GeoDataFrame(columns=['geometry'], crs="EPSG:4326")


def _add_synthetic_facilities(area_bounds, crs, count, seed, facility_type, segments):
    """Generate deterministic synthetic facilities snapped to street nodes."""
    rng = np.random.RandomState(seed)
    minx, miny, maxx, maxy = area_bounds

    points = []
    # Pick random segment midpoints to snap to
    indices = rng.choice(len(segments), size=min(count, len(segments)), replace=False)
    for idx in indices:
        geom = segments.geometry.iloc[idx]
        pt = geom.interpolate(0.5, normalized=True)
        points.append(pt)

    gdf = gpd.GeoDataFrame({
        'geometry': points,
        'facility_type': facility_type,
        'source': 'synthetic',
    }, crs=crs)
    return gdf


def load_facilities(area, segments):
    """Load water and cooling facilities. Add synthetic if configured."""
    config = get_config()
    cfg_fac = config['facilities']
    area_hash = _get_area_hash(area)

    # Water facilities
    water_cache = RAW_DIR / f"{area_hash}_water_facilities.parquet"
    water_tags = cfg_fac['osm_water_tags']
    water = _load_osm_facilities(area, water_tags, water_cache)
    water['facility_type'] = 'water'
    water['source'] = 'osm'
    water = water[['geometry', 'facility_type', 'source']].copy()

    # Cooling facilities
    cooling_cache = RAW_DIR / f"{area_hash}_cooling_facilities.parquet"
    cooling_tags = cfg_fac['osm_cooling_tags']
    cooling = _load_osm_facilities(area, cooling_tags, cooling_cache)
    cooling['facility_type'] = 'cooling'
    cooling['source'] = 'osm'
    cooling = cooling[['geometry', 'facility_type', 'source']].copy()

    # Project to same CRS as segments
    if len(water) > 0:
        water = water.to_crs(segments.crs)
    else:
        water = gpd.GeoDataFrame(columns=['geometry', 'facility_type', 'source'], crs=segments.crs)
    if len(cooling) > 0:
        cooling = cooling.to_crs(segments.crs)
    else:
        cooling = gpd.GeoDataFrame(columns=['geometry', 'facility_type', 'source'], crs=segments.crs)

    n_water_osm = len(water)
    n_cooling_osm = len(cooling)

    # Add synthetic if enabled and count is low
    if cfg_fac.get('allow_synthetic_facilities', False):
        syn_counts = cfg_fac.get('synthetic_counts', {})
        syn_seed = cfg_fac.get('seed', 123)

        if len(water) < syn_counts.get('water', 0):
            n_add = syn_counts['water'] - len(water)
            syn_water = _add_synthetic_facilities(
                segments.total_bounds, segments.crs, n_add, syn_seed, 'water', segments
            )
            water = pd.concat([water, syn_water], ignore_index=True)

        if len(cooling) < syn_counts.get('cooling', 0):
            n_add = syn_counts['cooling'] - len(cooling)
            syn_cooling = _add_synthetic_facilities(
                segments.total_bounds, segments.crs, n_add, syn_seed + 1, 'cooling', segments
            )
            cooling = pd.concat([cooling, syn_cooling], ignore_index=True)

    n_water_syn = len(water) - n_water_osm
    n_cooling_syn = len(cooling) - n_cooling_osm

    logger.info(f"Water facilities: {n_water_osm} OSM + {n_water_syn} synthetic = {len(water)}")
    logger.info(f"Cooling facilities: {n_cooling_osm} OSM + {n_cooling_syn} synthetic = {len(cooling)}")

    return water, cooling


# ──────────────────────── network distance ────────────────────────────────
def build_street_graph(segments):
    """Build an undirected NetworkX graph from segments (u, v, length_m)."""
    G = nx.Graph()
    for _, row in segments.iterrows():
        u = row['u']
        v = row['v']
        length = row['length_m']
        if u is not None and v is not None:
            # Keep shortest edge between same pair
            if G.has_edge(u, v):
                if G[u][v]['weight'] > length:
                    G[u][v]['weight'] = length
            else:
                G.add_edge(u, v, weight=length)
    return G


def _snap_to_node(point, nodes_gdf, max_snap_dist=500):
    """Find the closest graph node to a point."""
    dists = nodes_gdf.geometry.distance(point)
    min_idx = dists.idxmin()
    if dists[min_idx] <= max_snap_dist:
        return nodes_gdf.loc[min_idx, 'node_id'], dists[min_idx]
    return None, float('inf')


def multi_source_dijkstra(G, source_nodes):
    """Run multi-source Dijkstra from a set of source nodes. Returns dict node -> min_distance."""
    # Use networkx multi-source
    if not source_nodes:
        return {}

    valid_sources = [n for n in source_nodes if G.has_node(n)]
    if not valid_sources:
        return {}

    lengths = nx.multi_source_dijkstra_path_length(G, valid_sources, weight='weight')
    return dict(lengths)


def compute_cooling_access(segments, area, water_facilities=None, cooling_facilities=None):
    """
    Compute CoolingAccessRecords for every segment.
    """
    logger.info("Computing cooling access engine...")
    config = get_config()

    water_radius = config['water_service_radius']
    cooling_radius = config['cooling_service_radius']
    penalty_w = config['access_penalty_weights']
    max_penalty = config['max_access_penalty']

    # Load facilities if not provided
    if water_facilities is None or cooling_facilities is None:
        water_facilities, cooling_facilities = load_facilities(area, segments)

    # Build graph
    G = build_street_graph(segments)
    logger.info(f"Street graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Build nodes GeoDataFrame for snapping
    node_ids = list(G.nodes())
    # Get node positions from segment endpoints
    node_points = {}
    for _, row in segments.iterrows():
        u, v = row['u'], row['v']
        if u is not None and row.geometry is not None:
            coords = list(row.geometry.coords)
            if u not in node_points:
                node_points[u] = Point(coords[0])
            if v not in node_points:
                node_points[v] = Point(coords[-1])

    nodes_gdf = gpd.GeoDataFrame({
        'node_id': list(node_points.keys()),
        'geometry': list(node_points.values()),
    }, crs=segments.crs)

    # Snap facilities to nodes
    def snap_facilities(facilities):
        snapped_nodes = []
        snap_dists = []
        for _, row in facilities.iterrows():
            nid, sdist = _snap_to_node(row.geometry, nodes_gdf)
            if nid is not None:
                snapped_nodes.append(nid)
                snap_dists.append(sdist)
        return snapped_nodes, snap_dists

    water_nodes, water_snap = snap_facilities(water_facilities) if len(water_facilities) > 0 else ([], [])
    cooling_nodes, cooling_snap = snap_facilities(cooling_facilities) if len(cooling_facilities) > 0 else ([], [])

    logger.info(f"Snapped water nodes: {len(water_nodes)}, cooling nodes: {len(cooling_nodes)}")

    # Multi-source Dijkstra
    water_dists = multi_source_dijkstra(G, water_nodes)
    cooling_dists = multi_source_dijkstra(G, cooling_nodes)

    # Compute per-segment distances
    records = []
    for _, row in segments.iterrows():
        sid = row['segment_id']
        u, v = row['u'], row['v']
        length = row['length_m']

        # Water distance
        if water_nodes:
            du = water_dists.get(u, float('inf'))
            dv = water_dists.get(v, float('inf'))
            dist_water = min(du, dv) + 0.5 * length
            water_avail = 0.0
        else:
            dist_water = float('inf')
            water_avail = 1.0  # no facility penalty

        # Cooling distance
        if cooling_nodes:
            du = cooling_dists.get(u, float('inf'))
            dv = cooling_dists.get(v, float('inf'))
            dist_cooling = min(du, dv) + 0.5 * length
            cooling_avail = 0.0
        else:
            dist_cooling = float('inf')
            cooling_avail = 1.0  # no facility penalty

        # Bounded penalties
        water_penalty = bounded_distance_penalty(dist_water, water_radius)
        cooling_penalty = bounded_distance_penalty(dist_cooling, cooling_radius)

        # Availability factor: 1 if no facility of that type exists, else 0
        avail_factor = max(water_avail, cooling_avail)

        # Access penalty
        access_penalty = np.clip(
            penalty_w['water'] * water_penalty +
            penalty_w['cooling'] * cooling_penalty +
            penalty_w['availability'] * avail_factor,
            0, max_penalty
        )

        covered_water = dist_water <= water_radius
        covered_cooling = dist_cooling <= cooling_radius

        records.append({
            'segment_id': sid,
            'street_id': row['street_id'],
            'is_canonical': row['is_canonical'],
            'dist_water_m': dist_water if dist_water != float('inf') else -1,
            'dist_cooling_m': dist_cooling if dist_cooling != float('inf') else -1,
            'covered_water': covered_water,
            'covered_cooling': covered_cooling,
            'access_penalty': float(access_penalty),
            'prov_penalty_status': MODELLED,
            'prov_facility_source': 'mixed',
        })

    result = pd.DataFrame(records)

    # Sync duplicates: ensure non-canonical edges carry canonical values
    canon_map = result.loc[result['is_canonical'], ['street_id', 'access_penalty']].set_index('street_id')['access_penalty']
    result['access_penalty'] = result['street_id'].map(canon_map).fillna(result['access_penalty'])

    logger.info(f"Cooling access computed. Mean penalty={result['access_penalty'].mean():.3f}")
    return result, water_facilities, cooling_facilities
