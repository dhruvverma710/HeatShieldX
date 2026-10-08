# pyrefly: ignore [missing-import]
import osmnx as ox
import geopandas as gpd
import pandas as pd
from pathlib import Path
import logging
import hashlib
from src.config_loader import get_config
from src.provenance import attach_provenance, OBSERVED, ESTIMATED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent / 'data' / 'raw'
PROCESSED_DIR = Path(__file__).parent.parent / 'data' / 'processed'

# Ensure directories exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def safe_float(val):
    try:
        if isinstance(val, str):
            val = val.split(' ')[0]
            val = val.replace(',', '.')
        return float(val)
    except (ValueError, TypeError, AttributeError):
        return None

def parse_height(row, h_per_floor, fallback_floors):
    actual = safe_float(row.get('height'))
    if actual is not None and actual > 0:
        return actual, 'actual', False
    
    levels = safe_float(row.get('building:levels'))
    if levels is not None and levels > 0:
        return levels * h_per_floor, 'levels', True
        
    return fallback_floors * h_per_floor, 'fallback', True
    

def _get_area_hash(area):
    if isinstance(area, list):
        return hashlib.md5(str(area).encode()).hexdigest()
    return hashlib.md5(area.encode()).hexdigest()

def _load_osm_streets(area, cache_path):
    if cache_path.exists():
        logger.info(f"Loading streets from cache: {cache_path}")
        return gpd.read_parquet(cache_path)
    logger.info(f"Downloading street graph for {area} from OSM...")
    try:
        G = ox.graph_from_place(area, network_type='all')
        gdf = ox.graph_to_gdfs(G, nodes=False)
        
        # fix types for parquet: convert any list columns to string
        for col in gdf.columns:
            if gdf[col].apply(lambda x: isinstance(x, list)).any():
                gdf[col] = gdf[col].astype(str)
                
        gdf.to_parquet(cache_path)
        return gdf
    except Exception as e:
        logger.error(f"Failed to download street graph: {e}")
        if cache_path.exists():
            return gpd.read_parquet(cache_path)
        raise RuntimeError(f"Could not load street graph from OSM. Error: {e}")

def _load_osm_data(area, tags, cache_path):
    if cache_path.exists():
        logger.info(f"Loading data from cache: {cache_path}")
        return gpd.read_parquet(cache_path)
        
    logger.info(f"Downloading data for {area} from OSM...")
    try:
        # Assuming area is a place name string for simplicity in demo
        gdf = ox.features_from_place(area, tags=tags)
        
        # Keep only Polygons and MultiPolygons for buildings, LineStrings for streets
        if 'building' in tags:
            gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        elif 'highway' in tags:
            gdf = gdf[gdf.geometry.type.isin(['LineString', 'MultiLineString'])]
            
        gdf.to_parquet(cache_path)
        return gdf
    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        if cache_path.exists():
            logger.warning("Using existing cache despite download failure.")
            return gpd.read_parquet(cache_path)
        raise RuntimeError(f"Could not load data from OSM and no cache available. Error: {e}")

def load_street_network(area):
    cache_path = RAW_DIR / f"{_get_area_hash(area)}_streets.parquet"
    gdf = _load_osm_streets(area, cache_path)
    
    # Project to metric CRS
    gdf = gdf.to_crs(gdf.estimate_utm_crs())
    
    # Repair geometry
    gdf.geometry = gdf.geometry.make_valid()
    gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty]
    
    # Segment ID
    def make_segment_id(idx, wkt):
        if isinstance(idx, tuple) and len(idx) >= 3:
            return f"S-{idx[0]}-{idx[1]}-{idx[2]}"
        else:
            return hashlib.md5(f"{idx}_{wkt}".encode()).hexdigest()[:12]
            
    gdf['segment_id'] = [make_segment_id(idx, wkt) 
                         for idx, wkt in zip(gdf.index, gdf.geometry.to_wkt())]
    
    # Extract u, v, key if present in index
    if isinstance(gdf.index, pd.MultiIndex):
        gdf['u'] = gdf.index.get_level_values(0)
        gdf['v'] = gdf.index.get_level_values(1)
        gdf['key'] = gdf.index.get_level_values(2)
    else:
        gdf['u'] = None
        gdf['v'] = None
        gdf['key'] = None
        
    gdf = gdf.reset_index(drop=True)
    
    # Generate street_id and is_canonical
    def make_street_id(row):
        if row['u'] is not None and row['v'] is not None and row['key'] is not None:
            u, v = sorted([row['u'], row['v']])
            return f"E-{u}-{v}-{row['key']}"
        return row['segment_id']
    
    gdf['street_id'] = gdf.apply(make_street_id, axis=1)
    gdf['is_canonical'] = ~gdf.duplicated(subset=['street_id'], keep='first')
    
    # Metadata
    gdf['length_m'] = gdf.geometry.length
    if 'name' not in gdf.columns:
        gdf['name'] = 'Unknown'
    if 'highway' not in gdf.columns:
        gdf['highway'] = 'unknown'
    if 'oneway' not in gdf.columns:
        gdf['oneway'] = False
        
    gdf = gdf[['segment_id', 'street_id', 'is_canonical', 'u', 'v', 'key', 'geometry', 'length_m', 'name', 'highway', 'oneway']]
    
    config = get_config()
    gdf = attach_provenance(
        gdf, 
        source_type="OSM", 
        source_reference=str(area), 
        observed_or_estimated=OBSERVED, 
        assumptions_version=config['config_version'], 
        computation_mode=config.get('computation_mode', 'standard')
    )
    
    logger.info(f"Loaded {len(gdf)} street segments.")
    return gdf

def load_buildings(area):
    cache_path = RAW_DIR / f"{_get_area_hash(area)}_buildings.parquet"
    tags = {'building': True}
    gdf = _load_osm_data(area, tags, cache_path)
    
    # Project
    gdf = gdf.to_crs(gdf.estimate_utm_crs())
    
    # Repair geometry
    gdf.geometry = gdf.geometry.make_valid()
    gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty]
    
    config = get_config()
    h_per_floor = config['building_height_per_floor']
    fallback_floors = config['fallback_building_floors']
    
    # Parsing heights
    heights = gdf.apply(lambda row: parse_height(row, h_per_floor, fallback_floors), axis=1)
    gdf['height_m'] = [h[0] for h in heights]
    gdf['height_source'] = [h[1] for h in heights]
    gdf['estimated_flag'] = [h[2] for h in heights]
    
    if 'building:levels' in gdf.columns:
        gdf['levels'] = gdf['building:levels'].apply(safe_float)
    else:
        gdf['levels'] = None
        
    gdf['building_id'] = [hashlib.md5(f"{idx}_{wkt}".encode()).hexdigest()[:12] 
                          for idx, wkt in zip(gdf.index, gdf.geometry.to_wkt())]
    gdf['footprint'] = gdf.geometry
    
    # Select columns
    gdf = gdf[['building_id', 'footprint', 'height_m', 'height_source', 'estimated_flag', 'levels', 'geometry']]
    
    gdf = attach_provenance(
        gdf, 
        source_type="OSM", 
        source_reference=str(area), 
        observed_or_estimated=ESTIMATED, 
        assumptions_version=config['config_version'], 
        computation_mode=config.get('computation_mode', 'standard')
    )
    
    source_counts = gdf['height_source'].value_counts().to_dict()
    logger.info(f"Loaded {len(gdf)} buildings. Height sources: {source_counts}")
    return gdf

def get_data_engine_summary(area):
    streets = load_street_network(area)
    buildings = load_buildings(area)
    
    return {
        'streets_count': len(streets),
        'buildings_count': len(buildings),
        'height_sources': buildings['height_source'].value_counts().to_dict(),
        'crs_streets': streets.crs.to_string(),
        'crs_buildings': buildings.crs.to_string()
    }
