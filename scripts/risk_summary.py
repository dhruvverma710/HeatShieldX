"""
Risk Summary script for HeatShield X.
Prints FACTS only: canonical street count; per-time risk mean/std and class counts;
facility counts by source; coverage % and mean distances; vulnerability distribution;
overlap between top-10 exposure and top-10 risk streets; one example drivers output.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import logging
import pandas as pd
import numpy as np
from src.config_loader import get_config
from src.data_engine import load_street_network, load_buildings, _get_area_hash
from src.explain_engine import compute_drivers, build_explanation_text, compare_hottest_vs_highest_risk

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / 'data' / 'processed'


def main():
    config = get_config()
    area = config['demo_area']
    area_hash = _get_area_hash(area)
    cfg_version = config['config_version'].replace('.', '_')
    times = config['canonical_times']

    streets = load_street_network(area)
    canonical_count = streets['is_canonical'].sum()

    print("=" * 60)
    print("         HEATSHIELD X  —  RISK SUMMARY")
    print("=" * 60)
    print(f"\nConfig version: {config['config_version']}")
    print(f"Total edges: {len(streets)}, Canonical streets: {canonical_count}")

    # ── Per-time risk stats ──
    print("\n--- Per-Time Risk Statistics (Canonical Only, Geometric) ---")
    for t in times:
        hour, minute = map(int, t.split(':'))
        path = PROCESSED_DIR / f"risk_{area_hash}_{cfg_version}_{hour:02d}{minute:02d}_geometric.parquet"
        if not path.exists():
            print(f"  [{t}] MISSING")
            continue
        df = pd.read_parquet(path)
        canon = df[df['is_canonical']]
        mean_r = canon['risk_score'].mean()
        std_r = canon['risk_score'].std()
        class_counts = canon['risk_class'].value_counts().to_dict()
        print(f"  [{t}] Mean={mean_r:.1f}, Std={std_r:.1f}  Classes: {class_counts}")

    # ── Facility counts ──
    print("\n--- Facility Counts ---")
    cooling_path = PROCESSED_DIR / f"cooling_{area_hash}_{cfg_version}.parquet"
    if cooling_path.exists():
        cool_df = pd.read_parquet(cooling_path)
        canon_cool = cool_df[cool_df['is_canonical']]

        # Coverage
        water_cov = canon_cool['covered_water'].mean() * 100
        cooling_cov = canon_cool['covered_cooling'].mean() * 100
        water_dist = canon_cool.loc[canon_cool['dist_water_m'] > 0, 'dist_water_m']
        cooling_dist = canon_cool.loc[canon_cool['dist_cooling_m'] > 0, 'dist_cooling_m']

        print(f"  Water coverage: {water_cov:.1f}%")
        if len(water_dist) > 0:
            print(f"  Mean water distance: {water_dist.mean():.0f}m")
        print(f"  Cooling coverage: {cooling_cov:.1f}%")
        if len(cooling_dist) > 0:
            print(f"  Mean cooling distance: {cooling_dist.mean():.0f}m")

    # ── Vulnerability distribution ──
    print("\n--- Vulnerability Distribution (Canonical) ---")
    vuln_path = PROCESSED_DIR / f"vulnerability_{area_hash}_{cfg_version}.parquet"
    if vuln_path.exists():
        vuln_df = pd.read_parquet(vuln_path)
        canon_v = vuln_df[vuln_df['is_canonical']]
        print(f"  Mean: {canon_v['vulnerability_value'].mean():.3f}")
        print(f"  Std:  {canon_v['vulnerability_value'].std():.3f}")
        print(f"  Min:  {canon_v['vulnerability_value'].min():.3f}")
        print(f"  Max:  {canon_v['vulnerability_value'].max():.3f}")
        # Quartiles
        q = canon_v['vulnerability_value'].quantile([0.25, 0.5, 0.75])
        print(f"  Q25={q.iloc[0]:.3f}, Median={q.iloc[1]:.3f}, Q75={q.iloc[2]:.3f}")

    # ── Top-10 overlap ──
    print("\n--- Top-10 Exposure vs Top-10 Risk Overlap (Canonical, per time) ---")
    for t in times:
        hour, minute = map(int, t.split(':'))
        risk_path = PROCESSED_DIR / f"risk_{area_hash}_{cfg_version}_{hour:02d}{minute:02d}_geometric.parquet"
        exp_path = PROCESSED_DIR / f"exposure_{area_hash}_{cfg_version}_{hour:02d}{minute:02d}_geometric.parquet"
        if not risk_path.exists() or not exp_path.exists():
            continue
        risk_df = pd.read_parquet(risk_path)
        exp_df = pd.read_parquet(exp_path)

        canon_risk = risk_df[risk_df['is_canonical']]
        # Need to merge exposure with is_canonical
        exp_merged = exp_df.merge(streets[['segment_id', 'is_canonical', 'street_id']], on='segment_id', how='left')
        canon_exp = exp_merged[exp_merged['is_canonical'] == True]

        top10_exp = set(canon_exp.nlargest(10, 'exposure_value')['segment_id'])
        top10_risk = set(canon_risk.nlargest(10, 'risk_score')['segment_id'])
        overlap = top10_exp & top10_risk
        print(f"  [{t}] Overlap: {len(overlap)} / 10")

    # ── Hottest vs highest-risk ──
    print("\n--- Hottest vs Highest-Risk (13:00, Geometric) ---")
    risk_path = PROCESSED_DIR / f"risk_{area_hash}_{cfg_version}_1300_geometric.parquet"
    exp_path = PROCESSED_DIR / f"exposure_{area_hash}_{cfg_version}_1300_geometric.parquet"
    if risk_path.exists() and exp_path.exists():
        risk_df = pd.read_parquet(risk_path)
        exp_df = pd.read_parquet(exp_path)
        exp_merged = exp_df.merge(streets[['segment_id', 'is_canonical']], on='segment_id', how='left')

        comparison = compare_hottest_vs_highest_risk(exp_merged, risk_df)
        print(f"  Hottest segment: {comparison['hottest_segment_id']}")
        print(f"    Exposure={comparison['hottest_exposure']:.4f}, Risk={comparison['hottest_risk_score']}")
        print(f"  Highest risk:    {comparison['highest_risk_segment_id']}")
        print(f"    Risk={comparison['highest_risk_score']:.1f}, Exposure={comparison['highest_exposure']:.4f}")
        print(f"  Same street? {comparison['same_street']}")

    # ── Example drivers ──
    print("\n--- Example Drivers (first canonical, 13:00) ---")
    if risk_path.exists():
        risk_df = pd.read_parquet(risk_path)
        canon = risk_df[risk_df['is_canonical']].head(1)
        if len(canon) > 0:
            drivers = compute_drivers(canon)
            if drivers:
                d = drivers[0]
                seg_id = d['segment_id']
                name_lookup = streets.set_index('segment_id')['name']
                sname = name_lookup.get(seg_id, None)
                text = build_explanation_text(d, sname)
                print(f"  {text}")
                for drv in d['drivers']:
                    print(f"    {drv['driver']}: value={drv['value']:.3f}, "
                          f"pct={drv['percentile']:.0%}, {drv['level']}, "
                          f"dominant={drv['is_dominant']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
