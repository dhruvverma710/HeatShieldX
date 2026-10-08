import sys
from pathlib import Path
import logging
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from src.config_loader import get_config
from src.data_engine import load_street_network, load_buildings, _get_area_hash
from src.exposure_engine import compute_exposure

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / 'data' / 'processed'

def main():
    config = get_config()
    area = config['demo_area']
    area_hash = _get_area_hash(area)
    cfg_version = config['config_version'].replace('.', '_')
    
    streets = load_street_network(area)
    buildings = load_buildings(area)
    
    # Compute geometric if available
    logger.info("Computing geometric exposure...")
    geo_res, geo_mode = compute_exposure(streets, buildings, area_hash, PROCESSED_DIR, force_mode='geometric')
    
    # Compute fallback
    logger.info("Computing fallback exposure...")
    fb_res, fb_mode = compute_exposure(streets, buildings, area_hash, PROCESSED_DIR, force_mode='estimated_exposure_mode')
    
    times = config['canonical_times']
    for t in times:
        hour, minute = map(int, t.split(':'))
        
        # Save geometric
        geo_path = PROCESSED_DIR / f"exposure_{area_hash}_{cfg_version}_{hour:02d}{minute:02d}_geometric.parquet"
        geo_res[t].to_parquet(geo_path)
        
        # Save fallback
        fb_path = PROCESSED_DIR / f"exposure_{area_hash}_{cfg_version}_{hour:02d}{minute:02d}_fallback.parquet"
        fb_res[t].to_parquet(fb_path)
        
        # Print stats
        geo_df = geo_res[t]
        fb_df = fb_res[t]
        
        print(f"--- Exposure at {t} ---")
        print(f"Geometric Mode: Mean={geo_df['exposure_value'].mean():.3f}, Min={geo_df['exposure_value'].min():.3f}, Max={geo_df['exposure_value'].max():.3f}")
        print(f"Fallback Mode:  Mean={fb_df['exposure_value'].mean():.3f}, Min={fb_df['exposure_value'].min():.3f}, Max={fb_df['exposure_value'].max():.3f}")
        print("-" * 25)

if __name__ == "__main__":
    main()
