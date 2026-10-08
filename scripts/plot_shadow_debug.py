import sys
from pathlib import Path
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from datetime import datetime
import pytz

sys.path.append(str(Path(__file__).parent.parent))

from src.config_loader import get_config
from src.data_engine import load_street_network, load_buildings, _get_area_hash
from src.shadow_engine import building_shadow, solar_position

def main():
    config = get_config()
    area = config['demo_area']
    
    streets = load_street_network(area)
    buildings = load_buildings(area)
    
    tz = pytz.timezone(config['solar']['timezone'])
    analysis_date = config['solar']['analysis_date']
    times = config['canonical_times']
    
    out_dir = Path(__file__).parent.parent / 'data' / 'processed'
    area_hash = _get_area_hash(area)
    cfg_version = config['config_version'].replace('.', '_')
    
    for t in times:
        hour, minute = map(int, t.split(':'))
        dt_naive = datetime.strptime(f"{analysis_date} {t}", "%Y-%m-%d %H:%M")
        ts = tz.localize(dt_naive)
        
        cache_path = out_dir / f"shadow_{area_hash}_{cfg_version}_{hour:02d}{minute:02d}.parquet"
        
        if not cache_path.exists():
            print(f"Skipping {t}, cache missing.")
            continue
            
        res = pd.read_parquet(cache_path)
        if res['sun_below_threshold'].iloc[0]:
            print(f"Skipping {t}, sun below threshold.")
            continue
            
        az, el = solar_position(config['solar']['latitude'], config['solar']['longitude'], ts)
        
        # Merge shade fractions to streets
        streets_sh = streets.merge(res[['segment_id', 'shade_fraction']], on='segment_id')
        
        # Create shadow polygons for plotting (subset for speed)
        # Just grab center 1000m x 1000m to make plotting fast and visible
        bounds = streets.total_bounds
        cx, cy = (bounds[0]+bounds[2])/2, (bounds[1]+bounds[3])/2
        bbox = [cx - 500, cy - 500, cx + 500, cy + 500]
        
        bld_sub = buildings.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        str_sub = streets_sh.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        
        shadow_polys = []
        for _, row in bld_sub.iterrows():
            sh = building_shadow(row['footprint'], row['height_m'], az, el)
            if not sh.is_empty:
                shadow_polys.append(sh)
                
        shadow_gdf = gpd.GeoDataFrame(geometry=shadow_polys, crs=buildings.crs)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        # Plot shadows
        if not shadow_gdf.empty:
            shadow_gdf.plot(ax=ax, color='gray', alpha=0.5, edgecolor='none')
        # Plot buildings
        bld_sub.plot(ax=ax, color='salmon', edgecolor='red')
        # Plot streets with color map
        str_sub.plot(ax=ax, column='shade_fraction', cmap='viridis', linewidth=2, legend=True, 
                     vmin=0, vmax=1, label='Shade Fraction')
                     
        ax.set_title(f"Shadows at {t} (Az: {az:.1f}, El: {el:.1f})")
        
        out_png = out_dir / f"debug_shadows_{hour:02d}{minute:02d}.png"
        plt.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {out_png}")

if __name__ == "__main__":
    main()
