"""
Impact Engine for HeatShield X.
Computes "Modelled Impact" (Before/After/Delta) for pooled and time-specific metrics.
Metrics: vulnerability-weighted exposure, cooling coverage, average distance.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_impact(before_risk, after_risk, before_cooling, after_cooling, vuln_df):
    """
    Computes system-wide impact metrics before and after interventions.
    """
    impact = {}
    
    # Pool all times for exposure/risk
    times = list(before_risk.keys())
    
    # 1. Vulnerability-weighted exposure (pooled over all times, canonical only)
    b_exp_pool = []
    a_exp_pool = []
    
    for t in times:
        b_df = before_risk[t]
        a_df = after_risk[t]
        
        b_canon = b_df[b_df['is_canonical']]
        a_canon = a_df[a_df['is_canonical']]
        
        # Merge vulnerability mass
        b_merged = b_canon.merge(vuln_df[['segment_id', 'vulnerable_mass']], on='segment_id', how='left')
        a_merged = a_canon.merge(vuln_df[['segment_id', 'vulnerable_mass']], on='segment_id', how='left')
        
        # Final Risk Raw * Vulnerable Mass (matches objective function exactly)
        b_exp_mass = (b_merged['final_risk_raw'] * b_merged['vulnerable_mass']).sum()
        a_exp_mass = (a_merged['final_risk_raw'] * a_merged['vulnerable_mass']).sum()
        
        b_exp_pool.append(b_exp_mass)
        a_exp_pool.append(a_exp_mass)
        
    b_total_exp = sum(b_exp_pool)
    a_total_exp = sum(a_exp_pool)
    
    impact['vuln_weighted_exposure'] = {
        'before': float(b_total_exp),
        'after': float(a_total_exp),
        'delta': float(a_total_exp - b_total_exp)
    }
    
    # 2. Cooling coverage (canonical segments with dist <= radius)
    b_cool_canon = before_cooling[before_cooling['is_canonical']]
    a_cool_canon = after_cooling[after_cooling['is_canonical']]
    
    b_cov = int(b_cool_canon['covered_cooling'].sum())
    a_cov = int(a_cool_canon['covered_cooling'].sum())
    
    impact['cooling_coverage'] = {
        'before': b_cov,
        'after': a_cov,
        'delta': a_cov - b_cov
    }
    
    b_water_cov = int(b_cool_canon['covered_water'].sum())
    a_water_cov = int(a_cool_canon['covered_water'].sum())
    
    impact['water_coverage'] = {
        'before': b_water_cov,
        'after': a_water_cov,
        'delta': a_water_cov - b_water_cov
    }
    
    # 3. Average distance (excluding unreachable)
    b_dist = b_cool_canon.loc[b_cool_canon['dist_cooling_m'] >= 0, 'dist_cooling_m'].mean()
    a_dist = a_cool_canon.loc[a_cool_canon['dist_cooling_m'] >= 0, 'dist_cooling_m'].mean()
    
    impact['avg_cooling_dist_m'] = {
        'before': float(b_dist) if not np.isnan(b_dist) else None,
        'after': float(a_dist) if not np.isnan(a_dist) else None,
        'delta': float(a_dist - b_dist) if not np.isnan(a_dist) and not np.isnan(b_dist) else None
    }
    
    return impact
