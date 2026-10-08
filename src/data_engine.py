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


# ──────────────────────── DATA LAYER & CACHING ────────────────────────

try:
    import streamlit as st
    cache_data = st.cache_data
except ImportError:
    def cache_data(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func


@cache_data
def load_dashboard_data(area=None, mode=None):
    """
    Cached data loader for dashboard and data layer.
    Loads precomputed segments, risk records, drivers/exposure inputs, cooling access,
    vulnerability, and scope bounds.
    If any precomputed cache is missing, raises RuntimeError("run python scripts/precompute.py").
    """
    config = get_config()
    if area is None:
        area = config['demo_area']
    area_hash = _get_area_hash(area)
    cfg_version = config['config_version'].replace('.', '_')
    times = config['canonical_times']

    # Load streets
    try:
        streets = load_street_network(area)
    except Exception as e:
        raise RuntimeError(f"Missing streets cache or network data: {e}. Please run python scripts/precompute.py")

    # Paths for vulnerability, cooling, bounds
    vuln_path = PROCESSED_DIR / f"vulnerability_{area_hash}_{cfg_version}.parquet"
    cooling_path = PROCESSED_DIR / f"cooling_{area_hash}_{cfg_version}.parquet"
    bounds_path = PROCESSED_DIR / f"risk_bounds_{area_hash}_{cfg_version}.parquet"

    for p in [vuln_path, cooling_path, bounds_path]:
        if not p.exists():
            raise RuntimeError(f"Missing precomputed file {p.name}. Please run python scripts/precompute.py")

    vulnerability = pd.read_parquet(vuln_path)
    cooling_access = pd.read_parquet(cooling_path)
    scope_bounds = pd.read_parquet(bounds_path)

    # Determine mode
    if mode is None:
        comp_mode = config.get('computation_mode', 'auto')
        if comp_mode in ['geometric', 'fallback']:
            mode = comp_mode
        else:
            geo_09 = PROCESSED_DIR / f"risk_{area_hash}_{cfg_version}_0900_geometric.parquet"
            mode = 'geometric' if geo_09.exists() else 'fallback'

    risk_records = {}
    exposure_records = {}

    for t in times:
        hour, minute = map(int, t.split(':'))
        r_path = PROCESSED_DIR / f"risk_{area_hash}_{cfg_version}_{hour:02d}{minute:02d}_{mode}.parquet"
        e_path = PROCESSED_DIR / f"exposure_{area_hash}_{cfg_version}_{hour:02d}{minute:02d}_{mode}.parquet"

        if not r_path.exists():
            raise RuntimeError(f"Missing precomputed risk file {r_path.name}. Please run python scripts/precompute.py")

        r_df = pd.read_parquet(r_path)
        if 'name' not in r_df.columns and 'name' in streets.columns:
            r_df = r_df.merge(streets[['segment_id', 'name', 'highway']], on='segment_id', how='left')
        risk_records[t] = r_df

        if e_path.exists():
            exposure_records[t] = pd.read_parquet(e_path)

    return {
        'streets': streets,
        'vulnerability': vulnerability,
        'cooling_access': cooling_access,
        'scope_bounds': scope_bounds,
        'risk_records': risk_records,
        'exposure_records': exposure_records,
        'mode': mode,
        'config': config,
    }


def _to_minutes(time_input):
    """Convert string HH:MM or integer minutes to minutes from midnight."""
    if isinstance(time_input, str):
        if ':' in time_input:
            h, m = map(int, time_input.split(':'))
            return h * 60 + m
        return int(time_input)
    return int(time_input)


def get_snapshot(time_input, data=None):
    """
    Get risk snapshot at specified time (HH:MM or minutes from midnight).
    Canonical times return cached record unchanged.
    Interpolates linearly between canonical times for risk_score, exposure_value,
    shade_fraction, exposure_fraction. Recomputes risk_class from interpolated score.
    Times outside 09:00-17:00 (540-1020 minutes) raise ValueError.
    """
    time_minutes = _to_minutes(time_input)
    if time_minutes < 540 or time_minutes > 1020:
        raise ValueError(f"Time {time_minutes} minutes ({time_minutes // 60:02d}:{time_minutes % 60:02d}) is outside valid range 09:00-17:00 (540-1020 minutes).")

    if data is None:
        data = load_dashboard_data()

    canonical_map = {
        "09:00": 540,
        "11:00": 660,
        "13:00": 780,
        "15:00": 900,
        "17:00": 1020,
    }

    # Exact canonical match
    for t_str, t_min in canonical_map.items():
        if time_minutes == t_min:
            return data['risk_records'][t_str].copy()

    # Between two canonical times
    sorted_canon = sorted(canonical_map.items(), key=lambda x: x[1])
    t1_str, t1_min = sorted_canon[0]
    t2_str, t2_min = sorted_canon[-1]

    for i in range(len(sorted_canon) - 1):
        if sorted_canon[i][1] <= time_minutes <= sorted_canon[i+1][1]:
            t1_str, t1_min = sorted_canon[i]
            t2_str, t2_min = sorted_canon[i+1]
            break

    alpha = (time_minutes - t1_min) / float(t2_min - t1_min)

    df1 = data['risk_records'][t1_str].copy()
    df2 = data['risk_records'][t2_str].copy()

    interp_df = df1.copy()
    interp_cols = ['risk_score', 'exposure_value', 'shade_fraction', 'exposure_fraction']

    for col in interp_cols:
        if col in df1.columns and col in df2.columns:
            interp_df[col] = (1.0 - alpha) * df1[col].astype(float) + alpha * df2[col].astype(float)

    # Recompute risk class
    from src.risk_engine import classify_risk
    class_ranges = data['config']['risk_class_ranges']
    interp_df['risk_class'] = classify_risk(interp_df['risk_score'].values, class_ranges)

    # Flags and time label
    h = time_minutes // 60
    m = time_minutes % 60
    interp_df['time'] = f"{h:02d}:{m:02d}"
    interp_df['prov_status'] = 'INTERPOLATED'
    interp_df['prov_obs_est'] = ESTIMATED
    interp_df['estimated_flag'] = True

    return interp_df


def nearest_canonical_segment(lat, lon, data=None):
    """
    Project (lat, lon) to metric CRS, use spatial index, return nearest canonical segment dict/row.
    Returns None if distance > dashboard.click_tolerance_m.
    """
    from shapely.geometry import Point

    if data is None:
        data = load_dashboard_data()

    streets = data['streets']
    canon_segs = streets[streets['is_canonical'] == True].copy()
    click_tol = data['config'].get('dashboard', {}).get('click_tolerance_m', 50.0)

    pt_series = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(canon_segs.crs)
    pt_geom = pt_series.iloc[0]

    dists = canon_segs.geometry.distance(pt_geom)
    min_idx = dists.idxmin()
    min_dist = float(dists[min_idx])

    if min_dist > click_tol:
        return None

    res_row = canon_segs.loc[min_idx].to_dict()
    res_row['distance_m'] = min_dist
    return res_row


def top_priority_streets(time_input, n=20, data=None):
    """Return top N priority canonical streets sorted by risk_score descending."""
    df = get_snapshot(time_input, data=data)
    canon = df[df['is_canonical'] == True].copy()
    if 'risk_valid' in canon.columns:
        canon = canon[canon['risk_valid'] == True]
    canon.sort_values(by='risk_score', ascending=False, inplace=True)
    return canon.head(n).copy()


def class_counts(time_input, data=None):
    """Return dictionary of risk class frequencies for canonical segments at time_input."""
    df = get_snapshot(time_input, data=data)
    canon = df[df['is_canonical'] == True].copy()
    counts = canon['risk_class'].value_counts().to_dict()
    for cls in ['LOW', 'MODERATE', 'HIGH', 'CRITICAL']:
        counts.setdefault(cls, 0)
    return counts


def get_segment_detail(segment_id, time_input, data=None):
    """
    Get detail dictionary for a specific segment at time_input.
    Returns drivers, explanation_text, facility distances, and provenance.
    """
    if data is None:
        data = load_dashboard_data()

    df = get_snapshot(time_input, data=data)
    seg_rows = df[df['segment_id'] == segment_id]
    if len(seg_rows) == 0:
        return None

    seg_record = seg_rows.iloc[0]

    # Compute driver analysis
    from src.explain_engine import compute_drivers, build_explanation_text
    all_drivers = compute_drivers(df, config=data['config'])
    driver_rec = next((d for d in all_drivers if d['segment_id'] == segment_id), None)

    explanation = build_explanation_text(driver_rec, street_name=seg_record.get('name')) if driver_rec else ""

    # Facility access info
    cooling = data['cooling_access']
    cool_rows = cooling[cooling['segment_id'] == segment_id]

    dist_water = float(cool_rows['dist_water_m'].iloc[0]) if len(cool_rows) > 0 and 'dist_water_m' in cool_rows.columns else -1.0
    dist_cooling = float(cool_rows['dist_cooling_m'].iloc[0]) if len(cool_rows) > 0 and 'dist_cooling_m' in cool_rows.columns else -1.0
    covered_water = bool(cool_rows['covered_water'].iloc[0]) if len(cool_rows) > 0 and 'covered_water' in cool_rows.columns else False
    covered_cooling = bool(cool_rows['covered_cooling'].iloc[0]) if len(cool_rows) > 0 and 'covered_cooling' in cool_rows.columns else False

    provenance = {
        'prov_status': seg_record.get('prov_status', 'MODELLED'),
        'prov_obs_est': seg_record.get('prov_obs_est', ESTIMATED),
        'estimated_flag': bool(seg_record.get('estimated_flag', True)),
        'computation_mode': seg_record.get('computation_mode', data.get('mode', 'geometric')),
        'assumptions_version': seg_record.get('prov_assumptions_version', data['config'].get('config_version', '1.3.0')),
    }

    return {
        'segment_id': segment_id,
        'street_name': seg_record.get('name', 'Unknown'),
        'time': seg_record.get('time', str(time_input)),
        'risk_score': float(seg_record.get('risk_score', 0.0)) if pd.notna(seg_record.get('risk_score')) else None,
        'risk_class': seg_record.get('risk_class', 'LOW'),
        'drivers': driver_rec['drivers'] if driver_rec else [],
        'explanation_text': explanation,
        'dist_water_m': dist_water,
        'dist_cooling_m': dist_cooling,
        'covered_water': covered_water,
        'covered_cooling': covered_cooling,
        'provenance': provenance,
    }
