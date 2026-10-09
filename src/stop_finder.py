import logging
import os
import hashlib
import networkx as nx
import pandas as pd
import geopandas as gpd
from src.config_loader import get_config

logger = logging.getLogger(__name__)

def load_pois(area, config):
    """
    Load named POIs via OSMnx.
    """
    import osmnx as ox
    tags = config.get('routing', {}).get('poi_tags', {})
    
    hash_str = hashlib.md5(area.encode()).hexdigest()
    cache_file = f"data/raw/{hash_str}_pois.parquet"
    
    if os.path.exists(cache_file):
        try:
            pois = gpd.read_parquet(cache_file)
            return pois
        except Exception as e:
            logger.warning(f"Could not load POI cache: {e}")
            
    try:
        pois = ox.features_from_place(area, tags=tags)
        # Filter to only points or centroids
        pois['geometry'] = pois.geometry.centroid
        pois = pois[['geometry'] + [c for c in pois.columns if c != 'geometry']]
        os.makedirs("data/raw", exist_ok=True)
        # Drop columns that can't be saved
        cols_to_keep = [c for c in pois.columns if isinstance(c, str) and type(pois[c].iloc[0]) in (str, int, float, bool, type(None))]
        if 'name' not in cols_to_keep and 'name' in pois.columns:
            cols_to_keep.append('name')
        if 'amenity' not in cols_to_keep and 'amenity' in pois.columns:
            cols_to_keep.append('amenity')
        if 'leisure' not in cols_to_keep and 'leisure' in pois.columns:
            cols_to_keep.append('leisure')
        pois = pois[['geometry'] + [c for c in cols_to_keep if c in pois.columns]]
        pois.to_parquet(cache_file)
        return pois
    except Exception as e:
        logger.warning(f"Could not fetch POIs from OSM: {e}")
        return gpd.GeoDataFrame()

def _get_stop_type(row, water_facs, cooling_facs):
    # This identifies if a row from POIs/facilities is water, cooling, shaded_public, or business
    # actually it's easier to pass them already categorized.
    pass

def compute_safe_stops(route, G, segments, area, water_facs, cooling_facs):
    config = get_config()
    speed_mps = config.get('routing', {}).get('walking_speed_mps', 1.3)
    max_detour = config.get('routing', {}).get('max_stop_detour_m', 300)
    sponsored_ids = config.get('routing', {}).get('sponsored_stop_ids', [])
    priority = config.get('routing', {}).get('stop_type_priority', ["water", "cooling", "shaded_public", "business"])
    
    pois = load_pois(area, config)
    
    stops = []
    
    def add_stop(sid, stype, name, geom, source_type, verified, amenities):
        stops.append({
            'stop_id': str(sid),
            'stop_type': stype,
            'name': name if pd.notna(name) else f"Unnamed {stype}",
            'lat': geom.y,
            'lon': geom.x,
            'geometry': geom,
            'source_type': source_type,
            'verified_status': verified,
            'amenities': amenities,
            'rating': None,
            'sponsored_status': str(sid) in sponsored_ids
        })
        
    # Add water facilities
    if water_facs is not None:
        for idx, row in water_facs.iterrows():
            source = row.get('source', 'osm')
            add_stop(row.get('facility_id', f'W_{idx}'), 'water', row.get('name', None), row.geometry, source, source=='osm', [])
            
    # Add cooling facilities
    if cooling_facs is not None:
        for idx, row in cooling_facs.iterrows():
            source = row.get('source', 'osm')
            add_stop(row.get('facility_id', f'C_{idx}'), 'cooling', row.get('name', None), row.geometry, source, source=='osm', [])
            
    # Add POIs
    if not pois.empty:
        for idx, row in pois.iterrows():
            sid = f"P_{idx}"
            amenity = row.get('amenity', None)
            leisure = row.get('leisure', None)
            name = row.get('name', None)
            
            stype = 'business'
            verified = False
            if leisure in ['park', 'garden'] or amenity == 'shelter':
                stype = 'shaded_public'
                verified = False
            elif amenity in ['drinking_water', 'fountain']:
                stype = 'water'
                verified = True
            
            amenities = []
            if pd.notna(amenity): amenities.append(str(amenity))
            if pd.notna(leisure): amenities.append(str(leisure))
                
            add_stop(sid, stype, name, row.geometry, 'osm', verified, amenities)
            
    if not stops:
        return []
        
    stops_df = pd.DataFrame(stops)
    stops_gdf = gpd.GeoDataFrame(stops_df, geometry='geometry', crs=segments.crs)
    
    # Map route nodes
    route_segs = route['segment_ids']
    route_nodes = set()
    route_nodes_ordered = []
    
    # We need to find the node on the route with highest exposure to boost relevance
    node_exposure = {}
    
    for sid in route_segs:
        edge = segments[segments['segment_id'] == sid]
        if not edge.empty:
            u, v = edge.iloc[0]['u'], edge.iloc[0]['v']
            # if we have G, we can look up exposure
            exp = 0.0
            if G.has_edge(u, v):
                exp = G[u][v].get('exposure_value', 0.0)
            node_exposure[u] = max(node_exposure.get(u, 0.0), exp)
            node_exposure[v] = max(node_exposure.get(v, 0.0), exp)
            
            if u not in route_nodes:
                route_nodes.add(u)
                route_nodes_ordered.append(u)
            if v not in route_nodes:
                route_nodes.add(v)
                route_nodes_ordered.append(v)
                
    if not route_nodes:
        return []
        
    # Get nodes from graph
    node_points = []
    node_ids = []
    for n in route_nodes:
        # crude approx: we can find coordinates from segments or we can just snap to the route geometry.
        pass
        
    # To compute distance to route efficiently:
    # 1. build a GeoSeries of the route nodes
    # 2. distance from each stop to nearest route node
    
    # Let's extract node coords from segments
    node_coords = {}
    for sid in route_segs:
        edge = segments[segments['segment_id'] == sid]
        if not edge.empty:
            u, v = edge.iloc[0]['u'], edge.iloc[0]['v']
            geom = edge.iloc[0]['geometry']
            node_coords[u] = geom.coords[0]
            node_coords[v] = geom.coords[-1]
            
    rn_gdf = gpd.GeoDataFrame({
        'node_id': list(node_coords.keys()),
        'geometry': gpd.points_from_xy([c[0] for c in node_coords.values()], [c[1] for c in node_coords.values()])
    }, crs=segments.crs)
    
    final_stops = []
    for _, stop in stops_gdf.iterrows():
        dists = rn_gdf.geometry.distance(stop.geometry)
        min_idx = dists.idxmin()
        min_dist = dists[min_idx]
        route_distance_m = min_dist
        detour_m = 2 * route_distance_m
        
        if detour_m > max_detour:
            continue
            
        nearest_node = rn_gdf.iloc[min_idx]['node_id']
        detour_min = (detour_m / speed_mps) / 60.0
        
        # Relevance calculation
        # Base relevance by priority
        p_idx = priority.index(stop['stop_type']) if stop['stop_type'] in priority else 99
        rel_priority = 100 - p_idx * 10
        
        # Penalty for detour
        rel_detour = -detour_m / 10.0
        
        # Boost for near high exposure
        local_exp = node_exposure.get(nearest_node, 0.0)
        rel_exp = local_exp * 50
        
        relevance = rel_priority + rel_detour + rel_exp
        
        s = stop.to_dict()
        del s['geometry']
        s['route_distance_m'] = route_distance_m
        s['detour_m'] = detour_m
        s['detour_min'] = detour_min
        s['route_relevance'] = relevance
        if s['stop_type'] == 'shaded_public' and not s['verified_status']:
            # labelled "shade not verified" (can be appended to name or used in UI)
            pass
            
        final_stops.append(s)
        
    # Sort
    final_stops.sort(key=lambda x: (-x['route_relevance'], x['stop_id']))
    
    # Cap max stops
    max_stops = config.get('routing', {}).get('max_stops_shown', 8)
    return final_stops[:max_stops]
