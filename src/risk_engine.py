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

def normalize_score(values, canonical_mask=None, v_min=None, v_max=None, clip_percentile=None, scale=100.0):
    """
    Central normalisation function with optional clipping.
    If canonical_mask is provided, clipping and min-max bounds are computed
    ONLY over non-NaN canonical values.
    Raw values above the clip_percentile are clipped to that value before min-max.
    Min-max normalises to [0, scale].
    If v_min == v_max (zero variance), return scale / 2.
    Output is always clipped to [0, scale].
    Returns (normalised_array, v_min, v_max).
    """
    canon_vals = values if canonical_mask is None else values[canonical_mask]
    canon_vals = canon_vals[~np.isnan(canon_vals)]

    if len(canon_vals) == 0:
        return np.full_like(values, np.nan, dtype=float), 0.0, 100.0

    if clip_percentile is not None and clip_percentile < 100:
        if v_max is None:
            upper_bound = np.percentile(canon_vals, clip_percentile)
            values = np.where(np.isnan(values), np.nan, np.clip(values, None, upper_bound))
            canon_vals = np.clip(canon_vals, None, upper_bound)
            
    if v_min is None:
        v_min = float(canon_vals.min())
    if v_max is None:
        v_max = float(canon_vals.max())
        
    if v_max == v_min:
        res = np.where(np.isnan(values), np.nan, scale / 2.0)
        return res, v_min, v_max
        
    normed = (values - v_min) / (v_max - v_min) * scale
    return np.where(np.isnan(values), np.nan, np.clip(normed, 0, scale)), v_min, v_max


def classify_risk(scores, class_ranges):
    """
    Assign risk_class by upper-bound thresholds from config.
    <=25 LOW, <=50 MODERATE, <=75 HIGH, else CRITICAL.
    NaN scores are classified as 'LOW' (defensive default, logged upstream).
    No UNKNOWN class may be produced.
    """
    # Build sorted list of (upper_bound, class_name)
    bounds = []
    for cls_name, (lo, hi) in class_ranges.items():
        bounds.append((hi, cls_name))
    bounds.sort(key=lambda x: x[0])  # sort by upper bound ascending

    classes = []
    for s in scores:
        if np.isnan(s):
            classes.append('LOW')  # defensive: NaN → LOW (logged upstream)
            continue
        assigned = None
        for upper, cls_name in bounds:
            if s <= upper:
                assigned = cls_name
                break
        if assigned is None:
            # Score exceeds all upper bounds → last (highest) class
            assigned = bounds[-1][1]
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

        # Check for missing / NaN lookups (do NOT fill access_penalty or vulnerability with 0.0)
        invalid_mask = merged['vulnerability_value'].isna() | merged['access_penalty'].isna() | merged['exposure_value'].isna()
        nan_count = invalid_mask.sum()
        if nan_count > 0:
            logger.warning(f"[{t}] {nan_count} segments have NaN or missing vulnerability/cooling lookups.")
        merged['risk_valid'] = ~invalid_mask

        merged['baseline_risk'] = merged['exposure_value'] * merged['vulnerability_value']
        merged['final_risk_raw'] = merged['baseline_risk'] * (1 + merged['access_penalty'])

        merged['time'] = t
        all_records[t] = merged

        # Collect non-NaN canonical raw values for pooled normalisation
        canon_raw = merged.loc[(merged['is_canonical'] == True) & (merged['risk_valid'] == True), 'final_risk_raw'].dropna().values
        all_canonical_raw.append(canon_raw)

    # Phase 2: pooled normalisation (compute global min/max over all valid canonicals)
    if all_canonical_raw:
        pooled_canonical = np.concatenate(all_canonical_raw)
        pooled_canonical = pooled_canonical[~np.isnan(pooled_canonical)]
    else:
        pooled_canonical = np.array([])

    clip_pct = config.get('risk', {}).get('normalization_clip_percentile', 99)

    if stored_bounds is not None:
        scope_min, scope_max = stored_bounds
    else:
        if len(pooled_canonical) > 0:
            if clip_pct is not None and clip_pct < 100:
                pooled_upper = float(np.percentile(pooled_canonical, clip_pct))
                pooled_canonical = np.clip(pooled_canonical, None, pooled_upper)
            scope_min = float(pooled_canonical.min())
            scope_max = float(pooled_canonical.max())
        else:
            scope_min, scope_max = 0.0, 100.0

    # Phase 3: apply normalisation and classify
    results = {}
    for t, merged in all_records.items():
        scores, _, _ = normalize_score(
            merged['final_risk_raw'].values, 
            canonical_mask=None,  # bounds already computed pooled
            v_min=scope_min, 
            v_max=scope_max, 
            clip_percentile=None,  # already clipped in Phase 2 or stored bounds
            scale=100.0
        )
        merged['risk_score'] = scores
        # Log NaN scores before classification
        nan_count = np.isnan(scores).sum()
        if nan_count > 0:
            logger.warning(f"[{t}] {nan_count} NaN risk_scores found after normalisation. Flagged with risk_valid=False.")
        merged['risk_class'] = classify_risk(scores, class_ranges)
        merged['computation_mode'] = mode
        merged['prov_status'] = MODELLED
        merged['estimated_flag'] = True
        merged['prov_assumptions_version'] = config['config_version']

        out_cols = [
            'segment_id', 'street_id', 'is_canonical', 'time',
            'exposure_value', 'vulnerability_value', 'access_penalty',
            'baseline_risk', 'final_risk_raw', 'risk_score', 'risk_class',
            'risk_valid', 'computation_mode', 'prov_status', 'estimated_flag',
            'prov_assumptions_version',
        ]
        results[t] = merged[out_cols]

    logger.info(f"Risk computed for {len(results)} times. "
                f"Pooled scope: [{scope_min:.6f}, {scope_max:.6f}]")
    return results, (scope_min, scope_max)
