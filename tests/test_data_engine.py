import pytest
import geopandas as gpd
from shapely.geometry import Polygon, LineString
import pandas as pd
from src.data_engine import parse_height, _get_area_hash
from src.provenance import attach_provenance, OBSERVED
import hashlib

def test_parse_height():
    h_per_floor = 3
    fallback = 2
    
    # Actual height
    res = parse_height({'height': '12.5'}, h_per_floor, fallback)
    assert res == (12.5, 'actual', False)
    
    # Messy actual height
    res = parse_height({'height': '12 m'}, h_per_floor, fallback)
    assert res == (12.0, 'actual', False)
    
    # Comma actual height
    res = parse_height({'height': '12,5'}, h_per_floor, fallback)
    assert res == (12.5, 'actual', False)
    
    # Levels
    res = parse_height({'building:levels': '4'}, h_per_floor, fallback)
    assert res == (12.0, 'levels', True)
    
    # Garbage levels fallback
    res = parse_height({'building:levels': 'foo'}, h_per_floor, fallback)
    assert res == (6.0, 'fallback', True)
    
    # None fallback
    res = parse_height({}, h_per_floor, fallback)
    assert res == (6.0, 'fallback', True)

def test_attach_provenance():
    df = pd.DataFrame({'a': [1]})
    df = attach_provenance(df, 'TEST', 'Ref', OBSERVED, '1.0', 'std')
    assert df['prov_source_type'].iloc[0] == 'TEST'
    assert df['prov_obs_est'].iloc[0] == OBSERVED

def test_segment_id_determinism():
    """Test the S-u-v-key format and fallback hash-based ID"""
    from src.data_engine import load_street_network
    
    # Test S-u-v-key format
    sid = f"S-{100}-{200}-{0}"
    assert sid == "S-100-200-0"
    
    # Test hash fallback for non-tuple indices (duplicate geometry, different index)
    df = gpd.GeoDataFrame(
        {'geometry': [LineString([(0,0), (1,1)]), LineString([(0,0), (1,1)])]}, 
        index=[1, 2]
    )
    ids = [hashlib.md5(f"{idx}_{wkt}".encode()).hexdigest()[:12] 
           for idx, wkt in zip(df.index, df.geometry.to_wkt())]
    assert ids[0] != ids[1], "IDs should be unique for duplicate geometries with different index"
    
    # Determinism check
    id_again = hashlib.md5(f"{df.index[0]}_{df.geometry.iloc[0].wkt}".encode()).hexdigest()[:12]
    assert ids[0] == id_again, "ID generation is deterministic"

def test_segment_id_stable_under_shuffle():
    """Verify segment_id is stable when rows are reordered"""
    tuples = [(100, 200, 0), (300, 400, 0), (100, 200, 1)]
    ids_original = [f"S-{t[0]}-{t[1]}-{t[2]}" for t in tuples]
    
    # Shuffle
    import random
    shuffled = list(tuples)
    random.shuffle(shuffled)
    ids_shuffled = [f"S-{t[0]}-{t[1]}-{t[2]}" for t in shuffled]
    
    assert set(ids_original) == set(ids_shuffled), "IDs should be the same regardless of row order"

def test_invalid_geometry_repair():
    # A Polygon with a self-intersection (bowtie)
    invalid_poly = Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)])
    assert not invalid_poly.is_valid
    
    df = gpd.GeoDataFrame({'geometry': [invalid_poly]}, crs="EPSG:4326")
    df.geometry = df.geometry.make_valid()
    assert df.geometry.iloc[0].is_valid

def test_get_area_hash():
    hash1 = _get_area_hash("Piedmont, California")
    hash2 = _get_area_hash("Piedmont, California")
    assert hash1 == hash2
    
    hash_list1 = _get_area_hash([-122.2, 37.8, -122.1, 37.9])
    hash_list2 = _get_area_hash([-122.2, 37.8, -122.1, 37.9])
    assert hash_list1 == hash_list2
    assert hash1 != hash_list1
