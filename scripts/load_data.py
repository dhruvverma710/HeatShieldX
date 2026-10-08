import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config_loader import get_config
from src.data_engine import get_data_engine_summary

def main():
    config = get_config()
    area = config['demo_area']
    print(f"Loading data for area: {area}")
    
    try:
        summary = get_data_engine_summary(area)
        
        # We can't directly check streets dataframe here since we only get a summary.
        # Wait, the prompt says "Assert segment_id uniqueness in scripts/load_data.py output on real data."
        # I should load the streets to assert it.
        from src.data_engine import load_street_network
        streets = load_street_network(area)
        unique_segments = streets['segment_id'].nunique()
        assert unique_segments == len(streets), f"Segment IDs not unique! {unique_segments} vs {len(streets)}"
        print(f"Segment IDs uniqueness verified: {unique_segments}/{len(streets)}")
        
        print("\n--- Data Engine Summary ---")
        print(f"Streets Count: {summary['streets_count']}")
        print(f"Buildings Count: {summary['buildings_count']}")
        print(f"CRS Streets: {summary['crs_streets']}")
        print(f"CRS Buildings: {summary['crs_buildings']}")
        print("Building Height Sources:")
        for source, count in summary['height_sources'].items():
            print(f"  {source}: {count}")
        print("---------------------------")
    except Exception as e:
        print(f"Error loading data: {e}")

if __name__ == "__main__":
    main()
