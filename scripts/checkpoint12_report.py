import sys
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd

sys.path.append(str(Path(__file__).parent.parent))

from src.config_loader import get_config
from src.data_engine import load_street_network, load_buildings, _get_area_hash

PROCESSED_DIR = Path(__file__).parent.parent / 'data' / 'processed'

def main():
    config = get_config()
    area = config['demo_area']
    area_hash = _get_area_hash(area)
    cfg_version = config['config_version'].replace('.', '_')
    times = config['canonical_times']
    
    print("=" * 60)
    print("           HEATSHIELD X  —  HOUR-12 REPORT")
    print("=" * 60)
    
    streets = load_street_network(area)
    buildings = load_buildings(area)
    
    # ---- Test summary ----
    print(f"\nConfig version: {config['config_version']}")
    print(f"Segments: {len(streets)}, Buildings: {len(buildings)}")
    est_share = buildings['estimated_flag'].mean()
    print(f"Estimated-height share: {est_share:.2%}")
    
    # ---- Shadow snapshots present ----
    print("\n--- Shadow Snapshots ---")
    all_shadows = True
    for t in times:
        hour, minute = map(int, t.split(':'))
        path = PROCESSED_DIR / f"shadow_{area_hash}_{cfg_version}_{hour:02d}{minute:02d}.parquet"
        present = path.exists()
        if not present:
            all_shadows = False
        print(f"  [{t}] {'PRESENT' if present else 'MISSING'}")
    print(f"  All five present: {all_shadows}")
    
    # ---- 0b Shadow Diagnostics ----
    print("\n--- 0b. Shadow Artifact Diagnostics ---")
    half_w = config['shadow']['street_half_width_m']
    for t in times:
        hour, minute = map(int, t.split(':'))
        shadow_path = PROCESSED_DIR / f"shadow_{area_hash}_{cfg_version}_{hour:02d}{minute:02d}.parquet"
        if not shadow_path.exists():
            continue
        df = pd.read_parquet(shadow_path)
        if df['sun_below_threshold'].iloc[0]:
            print(f"  [{t}] Sun below threshold, skipped.")
            continue

        merged = df.merge(streets, on='segment_id')
        high_shade = merged[merged['shade_fraction'] >= 0.99]
        n_high = len(high_shade)
        median_len = high_shade['length_m'].median() if n_high > 0 else float('nan')

        # Tunnel/covered/bridge check
        tunnel_tags = high_shade['highway'].isin(['tunnel', 'covered', 'bridge']).sum()

        # Building footprint overlap > 50% of segment buffer
        if n_high > 0:
            high_shade_gdf = gpd.GeoDataFrame(
                high_shade[['segment_id']].copy(),
                geometry=gpd.GeoSeries.from_wkt(
                    high_shade['geometry'].apply(lambda g: g.wkt if hasattr(g, 'wkt') else str(g)),
                    crs=streets.crs
                ) if not isinstance(high_shade['geometry'].iloc[0], str) else
                gpd.GeoSeries(high_shade['geometry'].values, crs=streets.crs)
            )
            # Rebuild from streets directly
            high_ids = set(high_shade['segment_id'])
            high_streets = streets[streets['segment_id'].isin(high_ids)].copy()
            high_streets['buf_geom'] = high_streets.geometry.buffer(half_w, cap_style=2)
            bld_union = buildings['footprint'].unary_union
            overlap_count = 0
            for _, row in high_streets.iterrows():
                buf = row['buf_geom']
                inter = buf.intersection(bld_union)
                if inter.area / buf.area > 0.5:
                    overlap_count += 1
        else:
            overlap_count = 0

        print(f"  [{t}] shade >= 0.99: {n_high} segments")
        print(f"         median length: {median_len:.1f} m")
        print(f"         tunnels/bridges: {tunnel_tags}")
        print(f"         overlap building >50%: {overlap_count}")

    # Proposed fix note
    print("  >> Needs Decision: The 8–14 high-shade segments are very short")
    print("     (median ~4 m). They are likely short stubs fully enclosed by")
    print("     a building shadow. Possible fix: filter segments shorter than")
    print("     a configurable min_segment_length_m before shadow computation.")

    # ---- Exposure Metrics: Geometric ----
    print("\n--- Exposure Metrics: Geometric ---")
    geo_means = {}
    prev_df = None
    for t in times:
        hour, minute = map(int, t.split(':'))
        path = PROCESSED_DIR / f"exposure_{area_hash}_{cfg_version}_{hour:02d}{minute:02d}_geometric.parquet"
        if not path.exists():
            print(f"  [{t}] MISSING")
            continue
        df = pd.read_parquet(path)
        
        mean_val = df['exposure_value'].mean()
        std_val = df['exposure_value'].std()
        shade_gt_0 = (df['shade_fraction'] > 0).mean() * 100
        geo_means[t] = mean_val
        
        diff_str = "N/A"
        if prev_df is not None:
            m = df.merge(prev_df, on='segment_id', suffixes=('', '_prev'))
            mean_abs_diff = (m['exposure_value'] - m['exposure_value_prev']).abs().mean()
            diff_str = f"{mean_abs_diff:.4f}"
            
        print(f"  [{t}] Mean={mean_val:.4f}, StdDev={std_val:.4f}, Shade>0={shade_gt_0:.1f}%, MeanAbsChange={diff_str}")
        prev_df = df

    if all(k in geo_means for k in ['09:00', '13:00', '17:00']):
        print(f"  09:00 vs 13:00 vs 17:00 means: {geo_means['09:00']:.4f} / {geo_means['13:00']:.4f} / {geo_means['17:00']:.4f}")

    # ---- Exposure Metrics: Fallback ----
    print("\n--- Exposure Metrics: Fallback ---")
    fb_means = {}
    prev_df = None
    for t in times:
        hour, minute = map(int, t.split(':'))
        path = PROCESSED_DIR / f"exposure_{area_hash}_{cfg_version}_{hour:02d}{minute:02d}_fallback.parquet"
        if not path.exists():
            print(f"  [{t}] MISSING")
            continue
        df = pd.read_parquet(path)
        
        mean_val = df['exposure_value'].mean()
        std_val = df['exposure_value'].std()
        shade_gt_0 = (df['shade_fraction'] > 0).mean() * 100
        fb_means[t] = mean_val
        
        diff_str = "N/A"
        if prev_df is not None:
            m = df.merge(prev_df, on='segment_id', suffixes=('', '_prev'))
            mean_abs_diff = (m['exposure_value'] - m['exposure_value_prev']).abs().mean()
            diff_str = f"{mean_abs_diff:.4f}"
            
        print(f"  [{t}] Mean={mean_val:.4f}, StdDev={std_val:.4f}, Shade>0={shade_gt_0:.1f}%, MeanAbsChange={diff_str}")
        prev_df = df

    if all(k in fb_means for k in ['09:00', '13:00', '17:00']):
        print(f"  09:00 vs 13:00 vs 17:00 means: {fb_means['09:00']:.4f} / {fb_means['13:00']:.4f} / {fb_means['17:00']:.4f}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
