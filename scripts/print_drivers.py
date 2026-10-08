"""Print driver analysis for top-risk and LOW-risk streets at 13:00."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.config_loader import get_config
from src.data_engine import load_street_network, _get_area_hash
from src.explain_engine import compute_drivers, build_explanation_text

config = get_config()
area = config['demo_area']
area_hash = _get_area_hash(area)
cfg_version = config['config_version'].replace('.', '_')
PROCESSED_DIR = Path(__file__).parent.parent / 'data' / 'processed'

streets = load_street_network(area)

# Load 13:00 geometric risk
risk_path = PROCESSED_DIR / f'risk_{area_hash}_{cfg_version}_1300_geometric.parquet'
risk_df = pd.read_parquet(risk_path)

# NaN analysis
nan_count = risk_df['risk_score'].isna().sum()
print(f'NaN risk_score count at 13:00: {nan_count}')
unknown_count = (risk_df['risk_class'] == 'UNKNOWN').sum()
print(f'UNKNOWN class count at 13:00: {unknown_count}')
print('Class counts at 13:00 (canonical):')
canon = risk_df[risk_df['is_canonical']]
print(canon['risk_class'].value_counts().to_dict())

# Compute drivers for canonical segments at 13:00
drivers = compute_drivers(risk_df)
name_lookup = streets.set_index('segment_id')['name']

# Filter to canonical only
canon_drivers = []
for d in drivers:
    mask = risk_df['segment_id'] == d['segment_id']
    if mask.any() and risk_df.loc[mask, 'is_canonical'].iloc[0]:
        canon_drivers.append(d)

# TOP RISK STREET
top_risk = max(canon_drivers, key=lambda d: d['risk_score'])
sname = name_lookup.get(top_risk['segment_id'], None)
print()
print('=== TOP-RISK STREET AT 13:00 ===')
print(f'Segment: {top_risk["segment_id"]}  Name: {sname}')
print(f'Risk score: {top_risk["risk_score"]:.1f}  Class: {top_risk["risk_class"]}')
for drv in top_risk['drivers']:
    print(f'  {drv["driver"]}: value={drv["value"]:.4f}, '
          f'percentile={drv["percentile"]:.2%}, level={drv["level"]}, '
          f'dominant={drv["is_dominant"]}')
text = build_explanation_text(top_risk, sname)
print(f'Text: {text}')

# LOW-RISK STREET (pick one with the lowest score)
low_candidates = [d for d in canon_drivers if d['risk_class'] == 'LOW']
if low_candidates:
    low_street = min(low_candidates, key=lambda d: d['risk_score'])
    sname_low = name_lookup.get(low_street['segment_id'], None)
    print()
    print('=== LOW-RISK STREET AT 13:00 ===')
    print(f'Segment: {low_street["segment_id"]}  Name: {sname_low}')
    print(f'Risk score: {low_street["risk_score"]:.1f}  Class: {low_street["risk_class"]}')
    for drv in low_street['drivers']:
        print(f'  {drv["driver"]}: value={drv["value"]:.4f}, '
              f'percentile={drv["percentile"]:.2%}, level={drv["level"]}, '
              f'dominant={drv["is_dominant"]}')
    text_low = build_explanation_text(low_street, sname_low)
    print(f'Text: {text_low}')
else:
    print('No LOW-risk canonical streets found at 13:00')
