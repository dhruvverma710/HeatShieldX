import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import pvlib
import logging
from pathlib import Path
from src.config_loader import get_config
from src.provenance import MODELLED, attach_provenance

logger = logging.getLogger(__name__)

class ShadowEngineError(Exception):
    pass

def solar_position(lat, lon, tz_aware_timestamp):
    """
    Calculate solar azimuth and elevation.
    Returns azimuth_deg, elevation_deg.
    Azimuth is clockwise from north (0=N, 90=E).
    """
    solpos = pvlib.solarposition.get_solarposition(tz_aware_timestamp, lat, lon)
    return solpos['azimuth'].iloc[0], solpos['elevation'].iloc[0]

def building_shadow(footprint, height_m, azimuth_deg, elevation_deg):
    """
    Compute shadow polygon for a single building footprint.
    footprint: shapely Polygon or MultiPolygon
    """
    config = get_config()
    min_elev = config['shadow']['min_sun_elevation_deg']
    max_length = config['shadow']['max_shadow_length_m']
    
    if elevation_deg < min_elev:
        return Polygon()
        
    # Shadow length L = height / tan(elevation)
    # elevation in radians
    elev_rad = np.radians(elevation_deg)
    L = height_m / np.tan(elev_rad)
    L = min(L, max_length)
    
    # Shadow direction = azimuth + 180 degrees
    # North is +Y, East is +X
    # PVlib azimuth is clockwise from North (0). 
    # Mathematical angle (counter-clockwise from X-axis):
    # Math_angle = 90 - (azimuth + 180) = -90 - azimuth = 270 - azimuth
    shadow_dir_deg = (azimuth_deg + 180) % 360
    math_angle_rad = np.radians(90 - shadow_dir_deg)
    
    dx = L * np.cos(math_angle_rad)
    dy = L * np.sin(math_angle_rad)
    
    def project_polygon(poly):
        exterior = list(poly.exterior.coords)
        polygons_to_union = [poly]
        
        # Translated polygon
        translated_coords = [(x + dx, y + dy) for x, y in exterior]
        polygons_to_union.append(Polygon(translated_coords))
        
        # Swept quadrilaterals for each edge
        for i in range(len(exterior) - 1):
            p1 = exterior[i]
            p2 = exterior[i+1]
            p1_d = (p1[0] + dx, p1[1] + dy)
            p2_d = (p2[0] + dx, p2[1] + dy)
            quad = Polygon([p1, p2, p2_d, p1_d])
            if quad.is_valid:
                polygons_to_union.append(quad)
                
        return unary_union(polygons_to_union)
        
    if isinstance(footprint, MultiPolygon):
        shadows = [project_polygon(p) for p in footprint.geoms]
        return unary_union(shadows)
    elif isinstance(footprint, Polygon):
        return project_polygon(footprint)
    else:
        return Polygon()

def compute_shade_fractions(segments, buildings, timestamp):
    config = get_config()
    lat = config['solar']['latitude']
    lon = config['solar']['longitude']
    
    azimuth, elevation = solar_position(lat, lon, timestamp)
    
    min_elev = config['shadow']['min_sun_elevation_deg']
    street_half_width = config['shadow']['street_half_width_m']
    
    if elevation < min_elev:
        logger.info(f"Sun below threshold ({elevation:.2f} < {min_elev}) at {timestamp}")
        # Return empty shade fractions
        result = pd.DataFrame({
            'segment_id': segments['segment_id'],
            'time': [timestamp] * len(segments),
            'shade_fraction': 0.0,
            'exposure_fraction': 1.0,
            'solar_azimuth': azimuth,
            'solar_elevation': elevation,
            'sun_below_threshold': True
        })
        return result
        
    # Compute shadows for all buildings
    logger.info(f"Computing shadows for {len(buildings)} buildings at {timestamp}")
    shadow_polys = []
    
    # Vectorized / looped application
    # Pre-calculate to avoid per-row overhead if possible, but looping is fine for prototype
    for _, row in buildings.iterrows():
        geom = row['footprint']
        h = row['height_m']
        sh_geom = building_shadow(geom, h, azimuth, elevation)
        if sh_geom.is_valid and not sh_geom.is_empty:
            shadow_polys.append(sh_geom)
            
    # Union all shadows
    # For large datasets, unary_union can be slow. A spatial index is better for intersecting streets.
    # We create a GeoDataFrame of shadows
    if shadow_polys:
        shadows_gdf = gpd.GeoDataFrame(geometry=shadow_polys, crs=segments.crs)
        # Spatial index is automatically used in sjoin or overlay, but we can do a localized intersection
        
        # Buffer streets (flat caps per instructions)
        buffered_segments = segments.copy()
        buffered_segments['geometry'] = buffered_segments.geometry.buffer(street_half_width, cap_style=2)
        
        # Calculate intersection
        logger.info("Intersecting shadows with streets...")
        # Since we want fraction of area, we can union shadows that intersect each segment
        # Using sjoin to find candidate shadows for each segment
        joined = gpd.sjoin(buffered_segments, shadows_gdf, how="inner", predicate="intersects")
        
        # Group by segment, union the relevant shadows, and compute area
        shade_results = []
        
        # Original areas
        seg_areas = buffered_segments.set_index('segment_id').geometry.area
        
        # Group joined shadows by segment
        for segment_id, group in joined.groupby('segment_id'):
            seg_geom = buffered_segments.loc[buffered_segments['segment_id'] == segment_id, 'geometry'].iloc[0]
            
            # The indices of shadows in the group
            shadow_indices = group['index_right']
            relevant_shadows = shadows_gdf.loc[shadow_indices, 'geometry'].tolist()
            
            if relevant_shadows:
                union_shadow = unary_union(relevant_shadows)
                intersection = seg_geom.intersection(union_shadow)
                shade_area = intersection.area
                frac = shade_area / seg_areas[segment_id]
                frac = min(max(frac, 0.0), 1.0) # clip to [0,1]
            else:
                frac = 0.0
                
            shade_results.append({'segment_id': segment_id, 'shade_fraction': frac})
            
        res_df = pd.DataFrame(shade_results)
    else:
        res_df = pd.DataFrame(columns=['segment_id', 'shade_fraction'])
        
    # Merge back to segments
    final_res = segments[['segment_id']].copy()
    final_res = final_res.merge(res_df, on='segment_id', how='left')
    final_res['shade_fraction'] = final_res['shade_fraction'].fillna(0.0)
    final_res['exposure_fraction'] = 1.0 - final_res['shade_fraction']
    final_res['time'] = timestamp
    final_res['solar_azimuth'] = azimuth
    final_res['solar_elevation'] = elevation
    final_res['sun_below_threshold'] = False
    
    # Calculate estimated-height share used
    est_share = buildings['estimated_flag'].mean()
    final_res['estimated_height_share'] = est_share
    
    # Add provenance
    config = get_config()
    final_res = attach_provenance(
        final_res,
        source_type="geometric_shadow_engine",
        source_reference="buildings_and_streets",
        observed_or_estimated=MODELLED,
        assumptions_version=config['config_version'],
        computation_mode="geometric"
    )
    
    return final_res
