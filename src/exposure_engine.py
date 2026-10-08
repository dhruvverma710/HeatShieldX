import pandas as pd
import geopandas as gpd
import numpy as np
import logging
from pathlib import Path
from src.config_loader import get_config
from src.provenance import MODELLED, ESTIMATED, attach_provenance

logger = logging.getLogger(__name__)

def _clip(val, min_val, max_val):
    if isinstance(val, (pd.Series, np.ndarray)):
        return np.clip(val, min_val, max_val)
    return max(min_val, min(val, max_val))

def compute_exposure_formula(T, temp_min_c, temp_max_c, solar_elevation, exposure_fraction, a, length_m):
    """
    Computes exposure_value and exposure_load based on the exact formula.
    """
    temp_factor = _clip((T - temp_min_c) / (temp_max_c - temp_min_c), 0, 1)
    
    # solar_elevation is in degrees
    sun_factor = _clip(np.sin(np.radians(solar_elevation)), 0, 1)
    
    exposure_value = temp_factor * (a + (1 - a) * exposure_fraction * sun_factor)
    exposure_load = exposure_value * length_m
    
    return exposure_value, exposure_load

def compute_static_shade_factor(segments, buildings, buffer_m, gain, max_shade):
    """
    Compute static shade factor for fallback mode.
    static_shade_factor = clip(static_shade_gain * building_coverage_ratio
                               within static_shade_buffer_m, 0, max_static_shade)
    """
    from shapely.ops import unary_union

    logger.info("Computing fallback static shade factors...")
    buffered_segments = segments[['segment_id', 'geometry']].copy()
    buffered_segments['geometry'] = buffered_segments.geometry.buffer(buffer_m, cap_style=2)

    # Prepare a buildings GeoDataFrame whose active geometry is the footprint,
    # but also keep footprint as a plain column so it survives sjoin.
    bld = gpd.GeoDataFrame({
        'building_id': buildings['building_id'].values,
        'geometry': buildings['footprint'].values,
        'footprint_wkt': buildings['footprint'].apply(lambda g: g.wkt).values,
    }, crs=segments.crs)
    # sjoin will use 'geometry' for the spatial predicate
    joined = gpd.sjoin(buffered_segments, bld, how='inner', predicate='intersects')

    seg_areas = buffered_segments.set_index('segment_id').geometry.area

    static_shade_factors = []
    for segment_id, group in joined.groupby('segment_id'):
        seg_geom = buffered_segments.loc[
            buffered_segments['segment_id'] == segment_id, 'geometry'
        ].iloc[0]

        from shapely import wkt
        footprints = [wkt.loads(w) for w in group['footprint_wkt'].tolist()]
        if footprints:
            union_bld = unary_union(footprints)
            intersection = seg_geom.intersection(union_bld)
            ratio = intersection.area / seg_areas[segment_id]
            factor = _clip(gain * ratio, 0, max_shade)
        else:
            factor = 0.0

        static_shade_factors.append({
            'segment_id': segment_id, 'shade_fraction': factor
        })

    res_df = pd.DataFrame(static_shade_factors)

    final_res = segments[['segment_id', 'length_m']].copy()
    if not res_df.empty:
        final_res = final_res.merge(res_df, on='segment_id', how='left')
    else:
        final_res['shade_fraction'] = 0.0

    final_res['shade_fraction'] = final_res['shade_fraction'].fillna(0.0)
    final_res['exposure_fraction'] = 1.0 - final_res['shade_fraction']

    return final_res

def check_geometric_shadows_available(times, area_hash, cfg_version, data_dir):
    """Check if all 5 shadow snapshots exist"""
    for t in times:
        hour, minute = map(int, t.split(':'))
        path = data_dir / f"shadow_{area_hash}_{cfg_version}_{hour:02d}{minute:02d}.parquet"
        if not path.exists():
            return False
    return True

def get_exposure_mode_label(mode):
    if mode == "geometric":
        return "Geometric Shadow Mode"
    return "Estimated Exposure Mode"

def compute_exposure(segments, buildings, area_hash, data_dir, force_mode=None):
    """
    Computes exposure for all canonical times. 
    Returns dictionary mapping time string (e.g. '09:00') to DataFrame of ExposureRecords.
    """
    config = get_config()
    times = config['canonical_times']
    cfg_version = config['config_version'].replace('.', '_')
    
    mode = config.get('computation_mode', 'auto')
    if force_mode:
        mode = force_mode
        
    if mode == 'auto':
        if check_geometric_shadows_available(times, area_hash, cfg_version, data_dir):
            mode = 'geometric'
        else:
            logger.warning("Geometric shadow snapshots missing. Falling back to estimated_exposure_mode.")
            mode = 'estimated_exposure_mode'
            
    logger.info(f"Using computation mode: {mode}")
    
    a = config['exposure']['ambient_floor']
    temp_min = config['temperature_proxy']['temp_min_c']
    temp_max = config['temperature_proxy']['temp_max_c']
    
    results = {}
    
    # Pre-compute static shade if in fallback
    static_df = None
    if mode != 'geometric':
        buf = config['fallback']['static_shade_buffer_m']
        gain = config['fallback']['static_shade_gain']
        max_sh = config['fallback']['max_static_shade']
        static_df = compute_static_shade_factor(segments, buildings, buf, gain, max_sh)
    
    for t in times:
        T = config['temperature_proxy'][t]
        hour, minute = map(int, t.split(':'))
        
        if mode == 'geometric':
            shadow_path = data_dir / f"shadow_{area_hash}_{cfg_version}_{hour:02d}{minute:02d}.parquet"
            if not shadow_path.exists():
                raise FileNotFoundError(f"Missing geometric shadow cache: {shadow_path}")
            df = pd.read_parquet(shadow_path)
            df = df.merge(segments[['segment_id', 'length_m']], on='segment_id', how='left')
            
            # Sun factor handled correctly
            sol_elev = df['solar_elevation']
            # if sun below threshold, solar_elevation is still provided but sun_factor will be 0 if <0
        else:
            df = static_df.copy()
            df['time'] = t
            # In fallback, sun_factor = 1.0 (equivalent to solar_elevation=90)
            df['solar_elevation'] = 90.0
            df['sun_below_threshold'] = False
            
            # Replicate other cols
            df['solar_azimuth'] = 180.0
            df['estimated_height_share'] = buildings['estimated_flag'].mean()
            
        exposure_val, exposure_load = compute_exposure_formula(
            T, temp_min, temp_max, df['solar_elevation'], df['exposure_fraction'], a, df['length_m']
        )
        
        df['exposure_value'] = exposure_val
        df['exposure_load'] = exposure_load
        df['temperature_proxy_c'] = T
        
        # Derived sun factor for output
        df['sun_factor'] = _clip(np.sin(np.radians(df['solar_elevation'])), 0, 1)
        
        # Provenance
        df['prov_computation_mode'] = mode
        df['prov_status'] = MODELLED
        df['prov_temperature_status'] = ESTIMATED
        df['prov_assumptions_version'] = config['config_version']
        
        # Reorder cols 
        out_cols = [
            'segment_id', 'time', 'exposure_value', 'exposure_load', 
            'shade_fraction', 'exposure_fraction', 'temperature_proxy_c', 
            'sun_factor', 'prov_computation_mode', 'prov_status', 
            'prov_temperature_status', 'estimated_height_share', 'prov_assumptions_version'
        ]
        
        results[t] = df[out_cols]
        
    return results, mode
