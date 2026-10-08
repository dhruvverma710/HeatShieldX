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


def _percentile_rank(scope_values, value):
    """Compute the percentile rank of `value` within `scope_values`."""
    if len(scope_values) == 0:
        return 0.0
    return float(np.mean(scope_values <= value))


def _level_from_percentile(percentile, bands):
    """Assign LOW/MODERATE/HIGH from driver_level_bands config."""
    for level, (lo, hi) in bands.items():
        if lo <= percentile <= hi:
            return level
    return "HIGH"


def _class_consistent_adjective(risk_class, driver_level):
    """
    Return a wording adjective consistent with the risk class.
    LOW-risk streets never get 'high' or 'poor' language.
    """
    if risk_class == 'LOW':
        return {
            'LOW': 'low',
            'MODERATE': 'moderate',
            'HIGH': 'moderate',
        }.get(driver_level, 'moderate')
    elif risk_class == 'MODERATE':
        return {
            'LOW': 'low',
            'MODERATE': 'moderate',
            'HIGH': 'elevated',
        }.get(driver_level, 'moderate')
    else:  # HIGH or CRITICAL
        return {
            'LOW': 'low',
            'MODERATE': 'moderate',
            'HIGH': 'high',
        }.get(driver_level, 'high')


def compute_drivers(risk_df, config=None):
    """
    For each segment/time, compute driver analysis.

    Drivers:
    - solar_exposure = exposure_fraction × sun_factor  (from exposure data)
    - shade_deficit = 1 - shade_fraction  (low shade = risk driver)
    - vulnerability = vulnerability_value
    - cooling_access = access_penalty

    Each driver gets: value, scope percentile (over canonical segments AT THE SAME TIME),
    level (LOW/MODERATE/HIGH), is_dominant (percentile >= threshold).
    Ordered by percentile descending.
    """
    if config is None:
        config = get_config()

    threshold = config['risk']['driver_percentile_threshold']
    bands = config['risk']['driver_level_bands']

    # Group by time so percentiles are computed within same-time canonical segments
    times = risk_df['time'].unique() if 'time' in risk_df.columns else ['']

    all_drivers = []

    for t in times:
        if t != '':
            time_df = risk_df[risk_df['time'] == t].copy()
        else:
            time_df = risk_df.copy()

        canonical = time_df[time_df['is_canonical']].copy()

        # Scope arrays for percentile ranking (canonical at THIS time only)
        scope = {}
        if 'exposure_fraction' in canonical.columns and 'sun_factor' in canonical.columns:
            scope['solar_exposure'] = (canonical['exposure_fraction'] * canonical['sun_factor']).values
        else:
            scope['solar_exposure'] = canonical['exposure_value'].values

        if 'shade_fraction' in canonical.columns:
            scope['shade_deficit'] = (1 - canonical['shade_fraction']).values
        else:
            scope['shade_deficit'] = canonical['exposure_value'].values

        scope['vulnerability'] = canonical['vulnerability_value'].values
        scope['cooling_access'] = canonical['access_penalty'].values

        for _, row in time_df.iterrows():
            seg_id = row['segment_id']

            # Compute driver values for this segment
            if 'exposure_fraction' in time_df.columns and 'sun_factor' in time_df.columns:
                solar_val = row['exposure_fraction'] * row['sun_factor']
            else:
                solar_val = row['exposure_value']

            if 'shade_fraction' in time_df.columns:
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
    Lists ONLY dominant drivers (or says "no single dominant driver").
    Wording is consistent with the risk class.
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
    else:
        text += " No single dominant driver."

    # Add class-consistent detail for the top driver
    if dominant:
        top = dominant[0]
        adj = _class_consistent_adjective(risk_class, top['level'])
        text += f" {top['driver'].replace('_', ' ').capitalize()} is {adj} (percentile {top['percentile']:.0%})."
    elif drivers:
        top = drivers[0]
        adj = _class_consistent_adjective(risk_class, top['level'])
        text += f" The leading factor is {top['driver'].replace('_', ' ')} ({adj}, percentile {top['percentile']:.0%})."

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
