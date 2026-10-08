"""
Explainability Engine for HeatShield X.
Produces per-segment/time driver analysis and templated explanation text.
Deterministic, no LLM.
"""
import pandas as pd
import numpy as np
import logging
from src.config_loader import get_config

logger = logging.getLogger(__name__)


def _percentile_rank(values, value):
    """Compute the percentile rank of `value` within `values`."""
    return float(np.mean(values <= value))


def _level_from_percentile(percentile, bands):
    """Assign LOW/MODERATE/HIGH from driver_level_bands config."""
    for level, (lo, hi) in bands.items():
        if lo <= percentile <= hi:
            return level
    return "HIGH"


def compute_drivers(risk_df, config=None):
    """
    For each segment/time, compute driver analysis.

    Drivers:
    - solar_exposure = exposure_fraction × sun_factor  (from exposure data)
    - shade_deficit = 1 - shade_fraction  (low shade = risk driver)
    - vulnerability = vulnerability_value
    - cooling_access = access_penalty

    Each driver gets: value, scope percentile (over canonical segments),
    level (LOW/MODERATE/HIGH), is_dominant (percentile >= threshold).
    Ordered by percentile descending.
    """
    if config is None:
        config = get_config()

    threshold = config['risk']['driver_percentile_threshold']
    bands = config['risk']['driver_level_bands']

    canonical = risk_df[risk_df['is_canonical']].copy()

    driver_cols = {
        'solar_exposure': None,     # computed below
        'shade_deficit': None,      # 1 - shade_fraction (if available)
        'vulnerability': 'vulnerability_value',
        'cooling_access': 'access_penalty',
    }

    # Precompute scope arrays for percentile ranking (canonical only)
    scope = {}
    if 'exposure_fraction' in risk_df.columns and 'sun_factor' in risk_df.columns:
        scope['solar_exposure'] = (canonical['exposure_fraction'] * canonical['sun_factor']).values
    else:
        scope['solar_exposure'] = canonical['exposure_value'].values

    if 'shade_fraction' in risk_df.columns:
        scope['shade_deficit'] = (1 - canonical['shade_fraction']).values
    else:
        scope['shade_deficit'] = canonical['exposure_value'].values

    scope['vulnerability'] = canonical['vulnerability_value'].values
    scope['cooling_access'] = canonical['access_penalty'].values

    all_drivers = []
    for _, row in risk_df.iterrows():
        seg_id = row['segment_id']

        # Compute driver values for this segment
        if 'exposure_fraction' in risk_df.columns and 'sun_factor' in risk_df.columns:
            solar_val = row['exposure_fraction'] * row['sun_factor']
        else:
            solar_val = row['exposure_value']

        if 'shade_fraction' in risk_df.columns:
            shade_val = 1 - row['shade_fraction']
        else:
            shade_val = row['exposure_value']

        vuln_val = row['vulnerability_value']
        cool_val = row['access_penalty']

        drivers = {
            'solar_exposure': solar_val,
            'shade_deficit': shade_val,
            'vulnerability': vuln_val,
            'cooling_access': cool_val,
        }

        driver_list = []
        for name, val in drivers.items():
            pct = _percentile_rank(scope[name], val)
            lvl = _level_from_percentile(pct, bands)
            dominant = pct >= threshold
            driver_list.append({
                'driver': name,
                'value': val,
                'percentile': pct,
                'level': lvl,
                'is_dominant': dominant,
            })

        # Sort by percentile descending
        driver_list.sort(key=lambda x: x['percentile'], reverse=True)

        all_drivers.append({
            'segment_id': seg_id,
            'time': row.get('time', ''),
            'risk_score': row.get('risk_score', 0),
            'risk_class': row.get('risk_class', ''),
            'drivers': driver_list,
        })

    return all_drivers


def build_explanation_text(driver_record, street_name=None):
    """
    Build a templated explanation string from computed driver values only.
    """
    seg_id = driver_record['segment_id']
    time = driver_record['time']
    risk_score = driver_record['risk_score']
    risk_class = driver_record['risk_class']
    drivers = driver_record['drivers']

    if street_name and street_name not in ('Unknown', '', None, 'nan'):
        label = street_name
    else:
        label = f"Unnamed street ({seg_id})"

    dominant = [d for d in drivers if d['is_dominant']]
    dominant_names = [d['driver'].replace('_', ' ') for d in dominant]

    text = f"At {time}, {label} has a risk score of {risk_score:.0f} ({risk_class})."

    if dominant_names:
        text += f" Key drivers: {', '.join(dominant_names)}."

    top = drivers[0] if drivers else None
    if top:
        text += f" The strongest factor is {top['driver'].replace('_', ' ')} " \
                f"(percentile {top['percentile']:.0%}, {top['level']})."

    return text


def compare_hottest_vs_highest_risk(exposure_df, risk_df, canonical_only=True):
    """
    Compare the segment with the highest exposure_value vs the highest risk_score.
    Returns a dict with both segments' details and whether they are the same street.
    """
    if canonical_only:
        exp = exposure_df[exposure_df['is_canonical']].copy() if 'is_canonical' in exposure_df.columns else exposure_df.copy()
        rsk = risk_df[risk_df['is_canonical']].copy()
    else:
        exp = exposure_df.copy()
        rsk = risk_df.copy()

    hottest_idx = exp['exposure_value'].idxmax()
    hottest = exp.loc[hottest_idx]

    highest_idx = rsk['risk_score'].idxmax()
    highest = rsk.loc[highest_idx]

    # Look up risk for the hottest segment
    hottest_risk = rsk.loc[rsk['segment_id'] == hottest['segment_id']]
    # Look up exposure for the highest risk segment
    highest_exp = exp.loc[exp['segment_id'] == highest['segment_id']]

    same_street = hottest['segment_id'] == highest['segment_id']

    return {
        'hottest_segment_id': hottest['segment_id'],
        'hottest_exposure': float(hottest['exposure_value']),
        'hottest_risk_score': float(hottest_risk['risk_score'].iloc[0]) if len(hottest_risk) > 0 else None,
        'hottest_vulnerability': float(hottest_risk['vulnerability_value'].iloc[0]) if len(hottest_risk) > 0 else None,
        'hottest_access_penalty': float(hottest_risk['access_penalty'].iloc[0]) if len(hottest_risk) > 0 else None,
        'highest_risk_segment_id': highest['segment_id'],
        'highest_risk_score': float(highest['risk_score']),
        'highest_exposure': float(highest['exposure_value']),
        'highest_vulnerability': float(highest['vulnerability_value']),
        'highest_access_penalty': float(highest['access_penalty']),
        'same_street': same_street,
    }
