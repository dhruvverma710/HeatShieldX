import sys
from pathlib import Path
import time
import logging
import pandas as pd
import geopandas as gpd
from datetime import datetime
import pytz

sys.path.append(str(Path(__file__).parent.parent))

from src.config_loader import get_config
from src.data_engine import load_street_network, load_buildings, _get_area_hash
from src.shadow_engine import compute_shade_fractions, ShadowEngineError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / 'data' / 'processed'

def main():
    config = get_config()
    area = config['demo_area']
    area_hash = _get_area_hash(area)
    cfg_version = config['config_version'].replace('.', '_')
    
    logger.info(f"Loading data for area: {area}")
    streets = load_street_network(area)
    buildings = load_buildings(area)
    
    tz = pytz.timezone(config['solar']['timezone'])
    analysis_date = config['solar']['analysis_date']
    
    times = config['canonical_times']
    
    for t in times:
        hour, minute = map(int, t.split(':'))
        dt_naive = datetime.strptime(f"{analysis_date} {t}", "%Y-%m-%d %H:%M")
        ts = tz.localize(dt_naive)
        
        cache_path = PROCESSED_DIR / f"shadow_{area_hash}_{cfg_version}_{hour:02d}{minute:02d}.parquet"
        
        if cache_path.exists():
            logger.info(f"[{t}] Loading cached shadow snapshot from {cache_path}")
            res = pd.read_parquet(cache_path)
        else:
            logger.info(f"[{t}] Computing shadows...")
            start = time.time()
            try:
                res = compute_shade_fractions(streets, buildings, ts)
                res.to_parquet(cache_path)
                duration = time.time() - start
                logger.info(f"[{t}] Computation took {duration:.2f}s")
            except Exception as e:
                logger.error(f"Failed at {t}: {e}")
                raise ShadowEngineError(f"Computation failed at {t}: {e}")
        
        if not res['sun_below_threshold'].iloc[0]:
            mean_sf = res['shade_fraction'].mean()
            min_sf = res['shade_fraction'].min()
            max_sf = res['shade_fraction'].max()
            gt0 = (res['shade_fraction'] > 0).sum()
            az = res['solar_azimuth'].iloc[0]
            el = res['solar_elevation'].iloc[0]
            est_share = res['estimated_height_share'].iloc[0]
            
            print(f"--- Snapshot at {t} ---")
            print(f"  Solar: Azimuth {az:.1f}, Elevation {el:.1f}")
            print(f"  Shade Fraction: Mean={mean_sf:.3f}, Min={min_sf:.3f}, Max={max_sf:.3f}")
            print(f"  Segments shaded (>0): {gt0} / {len(res)}")
            print(f"  Estimated Height Share: {est_share:.2f}")
            print("-" * 25)
        else:
            print(f"--- Snapshot at {t} ---")
            print("  Sun below threshold.")
            print("-" * 25)

if __name__ == "__main__":
    main()
