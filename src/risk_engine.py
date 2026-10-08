"""
Risk Engine for HeatShield X.
Computes baseline_risk, final_risk_raw, risk_score (normalised 0-100),
and risk_class for every segment × time.
Normalisation is pooled over canonical segments × ALL five times.
Scope bounds are stored for reuse in post-intervention scoring.
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from src.config_loader import get_config
from src.provenance import MODELLED

logger = logging.getLogger(__name__)

# ──────── central normalisation (reusable with stored bounds) ────────

def normalize_risk(values, v_min=None, v_max=None):
    """
    Min-max normalise to [0, 100].
    If v_min == v_max (zero variance), return 50.
    Clip to [0, 100].
    Returns (normalised_array, v_min, v_max).
    """
    if v_min is None:
        v_min = values.min()
    if v_max is None:
        v_max = values.max()
    if v_max == v_min:
        return np.full_like(values, 50.0, dtype=float), v_min, v_max
    normed = (values - v_min) / (v_max - v_min) * 100
    return np.clip(normed, 0, 100), v_min, v_max


def classify_risk(scores, class_ranges):
    """Assign risk_class from config class ranges."""
    classes = []
    for s in scores:
        assigned = 'UNKNOWN'
        for cls_name, (lo, hi) in class_ranges.items():
            if lo <= s <= hi:
                assigned = cls_name
                break
        classes.append(assigned)
    return classes


def compute_risk(exposure_results: dict, vulnerability: pd.DataFrame,
                 cooling_access: pd.DataFrame, mode: str,
                 stored_bounds=None):
    """
    Compute RiskRecords for each time in exposure_results.

    Parameters
    ----------
    exposure_results : dict[str, pd.DataFrame]
        time -> exposure DataFrame with segment_id, exposure_value
    vulnerability : pd.DataFrame
        segment_id, vulnerability_value
    cooling_access : pd.DataFrame
        segment_id, access_penalty
    mode : str
        "geometric" or "estimated_exposure_mode"
    stored_bounds : tuple (v_min, v_max) or None
        If provided, use these bounds for normalisation (for post-intervention reuse).

    Returns
    -------
    dict[str, pd.DataFrame] : time -> RiskRecords
    tuple : (scope_min, scope_max) for reuse
    """
    logger.info("Computing risk engine...")
    config = get_config()
    class_ranges = config['risk_class_ranges']

    # Merge vulnerability and cooling into a lookup
    vuln_lookup = vulnerability.set_index('segment_id')[['vulnerability_value', 'is_canonical', 'street_id']].copy()
    cool_lookup = cooling_access.set_index('segment_id')[['access_penalty']].copy()

    # Phase 1: compute raw risk for all times, collect canonical values for pooled normalisation
    all_records = {}
    all_canonical_raw = []

    for t, exp_df in exposure_results.items():
        merged = exp_df[['segment_id', 'exposure_value']].copy()
        merged = merged.merge(vuln_lookup.reset_index(), on='segment_id', how='left')
        merged = merged.merge(cool_lookup.reset_index(), on='segment_id', how='left')

        merged['vulnerability_value'] = merged['vulnerability_value'].fillna(0.0)
        merged['access_penalty'] = merged['access_penalty'].fillna(0.0)

        merged['baseline_risk'] = merged['exposure_value'] * merged['vulnerability_value']
        merged['final_risk_raw'] = merged['baseline_risk'] * (1 + merged['access_penalty'])

        merged['time'] = t
        all_records[t] = merged

        # Collect canonical raw values for pooled normalisation
        canon_raw = merged.loc[merged['is_canonical'] == True, 'final_risk_raw'].values
        all_canonical_raw.append(canon_raw)

    # Phase 2: pooled normalisation
    pooled_canonical = np.concatenate(all_canonical_raw)

    if stored_bounds is not None:
        scope_min, scope_max = stored_bounds
    else:
        scope_min = pooled_canonical.min()
        scope_max = pooled_canonical.max()

    # Phase 3: apply normalisation and classify
    results = {}
    for t, merged in all_records.items():
        scores, _, _ = normalize_risk(merged['final_risk_raw'].values, scope_min, scope_max)
        merged['risk_score'] = scores
        merged['risk_class'] = classify_risk(scores, class_ranges)
        merged['computation_mode'] = mode
        merged['prov_status'] = MODELLED
        merged['estimated_flag'] = True
        merged['prov_assumptions_version'] = config['config_version']

        out_cols = [
            'segment_id', 'street_id', 'is_canonical', 'time',
            'exposure_value', 'vulnerability_value', 'access_penalty',
            'baseline_risk', 'final_risk_raw', 'risk_score', 'risk_class',
            'computation_mode', 'prov_status', 'estimated_flag', 'prov_assumptions_version',
        ]
        results[t] = merged[out_cols]

    logger.info(f"Risk computed for {len(results)} times. "
                f"Pooled scope: [{scope_min:.6f}, {scope_max:.6f}]")
    return results, (scope_min, scope_max)
