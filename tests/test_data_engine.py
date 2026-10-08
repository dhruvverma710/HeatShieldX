import pytest
import numpy as np
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


# ──────────────────────── DATA LAYER TESTS ────────────────────────

def test_canonical_time_unchanged():
    from src.data_engine import get_snapshot, load_dashboard_data
    data = load_dashboard_data()
    rec_str = get_snapshot("13:00", data=data)
    rec_min = get_snapshot(780, data=data)
    pd.testing.assert_frame_equal(rec_str, rec_min)
    pd.testing.assert_frame_equal(rec_str, data['risk_records']['13:00'])


def test_midpoint_equals_average():
    from src.data_engine import get_snapshot, load_dashboard_data
    data = load_dashboard_data()
    snap_09 = get_snapshot("09:00", data=data)
    snap_11 = get_snapshot("11:00", data=data)
    snap_10 = get_snapshot("10:00", data=data)

    expected_score = (snap_09['risk_score'].astype(float) + snap_11['risk_score'].astype(float)) / 2.0
    np.testing.assert_allclose(snap_10['risk_score'].values, expected_score.values, atol=1e-6)


def test_interpolated_flag_set():
    from src.data_engine import get_snapshot, load_dashboard_data
    data = load_dashboard_data()
    snap_10 = get_snapshot("10:00", data=data)
    assert (snap_10['prov_status'] == 'INTERPOLATED').all()


def test_class_recomputed():
    from src.data_engine import get_snapshot, load_dashboard_data
    from src.risk_engine import classify_risk
    data = load_dashboard_data()
    snap_10 = get_snapshot("10:00", data=data)
    expected_classes = classify_risk(snap_10['risk_score'].values, data['config']['risk_class_ranges'])
    assert snap_10['risk_class'].tolist() == expected_classes


def test_out_of_range_rejected():
    from src.data_engine import get_snapshot
    with pytest.raises(ValueError):
        get_snapshot("08:00")
    with pytest.raises(ValueError):
        get_snapshot(480)
    with pytest.raises(ValueError):
        get_snapshot("18:00")
    with pytest.raises(ValueError):
        get_snapshot(1080)


def test_nearest_segment_hit_and_miss():
    from src.data_engine import nearest_canonical_segment, load_dashboard_data
    data = load_dashboard_data()
    streets = data['streets']
    canon = streets[streets['is_canonical'] == True].iloc[0]

    # Convert geometry centroid to WGS84 (lat, lon)
    pt_wgs = gpd.GeoSeries([canon.geometry.centroid], crs=streets.crs).to_crs("EPSG:4326").iloc[0]
    lon, lat = pt_wgs.x, pt_wgs.y

    # Hit test (on/near segment)
    hit = nearest_canonical_segment(lat, lon, data=data)
    assert hit is not None
    assert hit['segment_id'] == canon['segment_id']
    assert hit['distance_m'] <= 5.0

    # Miss test (far away)
    miss = nearest_canonical_segment(0.0, 0.0, data=data)
    assert miss is None


def test_missing_cache_error_message():
    from src.data_engine import load_dashboard_data
    with pytest.raises(RuntimeError) as exc_info:
        load_dashboard_data(area="NonexistentAreaPlaceholderAreaName")
    assert "run python scripts/precompute.py" in str(exc_info.value)

