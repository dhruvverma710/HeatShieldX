"""Reclassify risk parquets in-place using the fixed classify_risk logic."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from src.config_loader import get_config
from src.risk_engine import classify_risk
from src.data_engine import _get_area_hash

config = get_config()
area_hash = _get_area_hash(config['demo_area'])
cfg_version = config['config_version'].replace('.', '_')
PROCESSED_DIR = Path(__file__).parent.parent / 'data' / 'processed'
times = config['canonical_times']

for mode in ['geometric', 'fallback']:
    for t in times:
        hour, minute = map(int, t.split(':'))
        path = PROCESSED_DIR / f'risk_{area_hash}_{cfg_version}_{hour:02d}{minute:02d}_{mode}.parquet'
        if not path.exists():
            print(f'SKIP {path.name} (not found)')
            continue
        df = pd.read_parquet(path)
        old_unknown = (df['risk_class'] == 'UNKNOWN').sum()
        df['risk_class'] = classify_risk(df['risk_score'].values, config['risk_class_ranges'])
        new_unknown = (df['risk_class'] == 'UNKNOWN').sum()
        df.to_parquet(path)
        print(f'{path.name}: old UNKNOWN={old_unknown} -> new UNKNOWN={new_unknown}')
