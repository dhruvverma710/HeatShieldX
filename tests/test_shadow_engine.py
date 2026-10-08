import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, LineString
import numpy as np
from src.shadow_engine import building_shadow, solar_position, compute_shade_fractions
from unittest.mock import patch
from datetime import datetime
import pytz
import math

def setup_module(module):
    from src.config_loader import ConfigLoader
    # Ensure config is loaded
    ConfigLoader.get_config()

def test_t1_one_building():
    # 10x10 m footprint, 10 m height
    footprint = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    h = 10
    
    # sun az=90 (East), elev=45
    # shadow should extend 10m West
    shadow = building_shadow(footprint, h, azimuth_deg=90, elevation_deg=45)
    
    # Original area = 100
    # Shadow length = 10 / tan(45) = 10. Direction = 270 (West)
    # The union of footprint and translated footprint should be x from -10 to 10, y from 0 to 10.
    # Area should be 20x10 = 200
    assert abs(shadow.area - 200) < 1.0
    
    bounds = shadow.bounds
    # minx, miny, maxx, maxy
    assert abs(bounds[0] - (-10)) < 0.1
    assert abs(bounds[2] - 10) < 0.1

def test_t2_known_timestamp():
    # Fixed mid-latitude northern hemisphere
    lat = 40.0
    lon = -100.0
    tz = pytz.timezone('America/Chicago')
    
    times = [
        tz.localize(datetime(2024, 7, 15, 9, 0)),
        tz.localize(datetime(2024, 7, 15, 13, 0)),
        tz.localize(datetime(2024, 7, 15, 17, 0))
    ]
    
    az9, el9 = solar_position(lat, lon, times[0])
    az13, el13 = solar_position(lat, lon, times[1])
    az17, el17 = solar_position(lat, lon, times[2])
    
    assert az9 < 180
    assert az17 > 180
    assert el13 > el9
    assert el13 > el17

def test_t3_direction():
    footprint = Polygon([(0,0), (1,0), (1,1), (0,1)])
    h = 10
    # sun east (90) -> shadow west (minx < 0)
    sh_e = building_shadow(footprint, h, 90, 45)
    assert sh_e.bounds[0] < 0 and sh_e.bounds[2] == 1
    
    # sun south (180) -> shadow north (maxy > 1)
    sh_s = building_shadow(footprint, h, 180, 45)
    assert sh_s.bounds[3] > 1 and sh_s.bounds[1] == 0
    
    # sun west (270) -> shadow east (maxx > 1)
    sh_w = building_shadow(footprint, h, 270, 45)
    assert sh_w.bounds[2] > 1 and sh_w.bounds[0] == 0
    
    # sun north (0) -> shadow south (miny < 0)
    sh_n = building_shadow(footprint, h, 0, 45)
    assert sh_n.bounds[1] < 0 and sh_n.bounds[3] == 1

@patch('src.shadow_engine.solar_position')
def test_t4_street_intersection(mock_sp):
    # Setup
    mock_sp.return_value = (90, 45) # Sun East, shadow extends West
    
    # Building at (0,0) to (10,10)
    buildings = gpd.GeoDataFrame({
        'building_id': ['b1'],
        'footprint': [Polygon([(0,0), (10,0), (10,10), (0,10)])],
        'height_m': [10.0],
        'estimated_flag': [True]
    }, geometry='footprint', crs="EPSG:32610")
    
    # Street 1: in shadow path (-5, 5) to (-5, 15)
    # Street 2: on sun side (15, 5) to (15, 15)
    # Street 3: half covered (-10, 5) to (0, 5)
    segments = gpd.GeoDataFrame({
        'segment_id': ['s1', 's2', 's3'],
        'geometry': [
            LineString([(-5, 5), (-5, 15)]),
            LineString([(15, 5), (15, 15)]),
            LineString([(-10, 5), (0, 5)])
        ]
    }, crs="EPSG:32610")
    
    tz = pytz.UTC
    ts = tz.localize(datetime(2024, 7, 15, 12, 0))
    
    res = compute_shade_fractions(segments, buildings, ts)
    
    s1_frac = res[res['segment_id'] == 's1']['shade_fraction'].iloc[0]
    s2_frac = res[res['segment_id'] == 's2']['shade_fraction'].iloc[0]
    s3_frac = res[res['segment_id'] == 's3']['shade_fraction'].iloc[0]
    
    # street 1 is fully inside the shadow (-10 to 10 in X, 0 to 10 in Y)
    # buffered street 1 extends from -8 to -2 in X (width=3). 
    # wait, street 1 goes up to Y=15, but shadow only goes to Y=10.
    # So street 1 is 50% in shadow (Y=5 to 10 is inside, 10 to 15 is outside).
    assert abs(s1_frac - 0.5) < 0.1, f"Expected ~0.5, got {s1_frac}"
    
    # street 2 is entirely outside shadow
    assert s2_frac == 0.0
    
    # street 3 goes from -10 to 0. The shadow covers -10 to 10.
    # Thus street 3 is 100% in shadow, except for its caps which might extend past -10.
    # Let's check it's > 0.8
    assert s3_frac > 0.8

@patch('src.shadow_engine.solar_position')
def test_t5_multiple_buildings(mock_sp):
    mock_sp.return_value = (90, 45) 
    
    # Two overlapping buildings
    buildings = gpd.GeoDataFrame({
        'building_id': ['b1', 'b2'],
        'footprint': [
            Polygon([(0,0), (10,0), (10,10), (0,10)]),
            Polygon([(5,0), (15,0), (15,10), (5,10)])
        ],
        'height_m': [10.0, 10.0],
        'estimated_flag': [True, True]
    }, geometry='footprint', crs="EPSG:32610")
    
    segments = gpd.GeoDataFrame({
        'segment_id': ['s1'],
        'geometry': [LineString([(-5, 5), (-5, 15)])]
    }, crs="EPSG:32610")
    
    ts = pytz.UTC.localize(datetime(2024, 7, 15, 12, 0))
    res = compute_shade_fractions(segments, buildings, ts)
    
    assert res['shade_fraction'].iloc[0] <= 1.0

def test_elevation_below_threshold():
    footprint = Polygon([(0,0), (10,0), (10,10), (0,10)])
    sh = building_shadow(footprint, 10, 90, 2) # elev=2 < 5
    assert sh.is_empty
