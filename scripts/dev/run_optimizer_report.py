"""Quick optimizer verification: 2/1/2 plan, 0/0/0 baseline, runtime check."""
import time
import pandas as pd
from src.config_loader import get_config
from src.data_engine import load_dashboard_data
from src.intervention_engine import generate_candidates
from src.optimizer import run_optimizer
from src.impact_engine import compute_impact

def main():
    print("Loading dashboard data...")
    data = load_dashboard_data()
    config = get_config()
    streets = data['streets']
    vuln = data['vulnerability']
    
    current_state = {
        'exposure_results': data.get('exposure_records', {}),
        'cooling_access': data['cooling_access'],
        'risk_records': data['risk_records'],
        'stored_bounds': data['scope_bounds'],
    }
    
    print("Generating candidates...")
    cands_df = generate_candidates(streets)
    
    shade_count = len(cands_df[cands_df['type'] == 'shade'])
    water_count = len(cands_df[cands_df['type'] == 'water'])
    cooling_count = len(cands_df[cands_df['type'] == 'cooling'])
    print(f"Candidates: shade={shade_count}, water={water_count}, cooling={cooling_count}, total={len(cands_df)}")
    
    # ── Run 1: 2/1/2 ──
    print("\n=== Run 1: Water=2, Cooling=1, Shade=2 ===")
    t0 = time.time()
    limits_1 = {'water': 2, 'cooling': 1, 'shade': 2}
    sel_1, state_1, m1 = run_optimizer(cands_df, streets, vuln, current_state, limits_1)
    t1 = time.time()
    
    print(f"Runtime: {t1-t0:.2f}s")
    print(f"Objective before: {m1['objective_before']:.4f}")
    print(f"Objective after:  {m1['objective_after']:.4f}")
    print(f"Early stop: {m1['early_stop_reason']}")
    print(f"Candidates evaluated: {m1['n_candidates_evaluated']}")
    print(f"\nSelected interventions ({len(sel_1)}):")
    print(f"{'Priority':>8} {'Type':>8} {'Candidate ID':>35} {'Street':>25} {'Marginal':>12} {'Cumulative':>12}")
    for i, s in enumerate(sel_1, 1):
        label = streets.loc[streets['segment_id'] == s['target_id'], 'name'].iloc[0] if s['type'] == 'shade' else str(s['target_id'])
        print(f"{i:>8} {s['type']:>8} {s['candidate_id']:>35} {label:>25} {s['marginal_benefit']:>12.4f} {s['cumulative_benefit']:>12.4f}")
    
    # ── Run 2: 0/0/0 ──
    print("\n=== Run 2: Water=0, Cooling=0, Shade=0 ===")
    t0 = time.time()
    limits_0 = {'water': 0, 'cooling': 0, 'shade': 0}
    sel_0, state_0, m0 = run_optimizer(cands_df, streets, vuln, current_state, limits_0)
    t1 = time.time()
    
    print(f"Runtime: {t1-t0:.2f}s")
    print(f"Objective before: {m0['objective_before']:.4f}")
    print(f"Objective after:  {m0['objective_after']:.4f}")
    assert m0['objective_before'] == m0['objective_after'], "ERROR: 0/0/0 should have before == after"
    print("PASS: before == after")
    
    # ── Impact table ──
    if sel_1:
        imp = compute_impact(
            current_state['risk_records'], state_1['risk_records'],
            current_state['cooling_access'], state_1['cooling_access'],
            vuln
        )
        print("\n=== Impact (2/1/2) ===")
        for k, v in imp.items():
            if isinstance(v, dict):
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {v}")
                
        print("\n=== Detailed Metrics (Pooled) ===")
        print(f"  Objective Before: {m1['objective_before']:.4f}")
        print(f"  Objective After:  {m1['objective_after']:.4f}")
        
        # High/Critical Counts Pooled
        high_crit_before = sum(((rdf['risk_class'] == 'HIGH') | (rdf['risk_class'] == 'CRITICAL')).sum() for rdf in current_state['risk_records'].values())
        high_crit_after = sum(((rdf['risk_class'] == 'HIGH') | (rdf['risk_class'] == 'CRITICAL')).sum() for rdf in state_1['risk_records'].values())
        print(f"  HIGH/CRITICAL Segments Pooled: Before={high_crit_before}, After={high_crit_after}")
        
        # 13:00 Metrics
        print("\n=== Detailed Metrics (13:00) ===")
        r13_b = current_state['risk_records']['13:00']
        r13_a = state_1['risk_records']['13:00']
        
        hc_b_13 = ((r13_b['risk_class'] == 'HIGH') | (r13_b['risk_class'] == 'CRITICAL')).sum()
        hc_a_13 = ((r13_a['risk_class'] == 'HIGH') | (r13_a['risk_class'] == 'CRITICAL')).sum()
        print(f"  HIGH/CRITICAL Segments (13:00): Before={hc_b_13}, After={hc_a_13}")
        
        # Objective at 13:00
        canon_b = r13_b[r13_b['is_canonical'] == True]
        merged_b = canon_b.set_index('segment_id').join(vuln.set_index('segment_id')['vulnerable_mass'], how='left')
        obj_b_13 = (merged_b['final_risk_raw'] * merged_b['vulnerable_mass'].fillna(0)).sum()
        
        canon_a = r13_a[r13_a['is_canonical'] == True]
        merged_a = canon_a.set_index('segment_id').join(vuln.set_index('segment_id')['vulnerable_mass'], how='left')
        obj_a_13 = (merged_a['final_risk_raw'] * merged_a['vulnerable_mass'].fillna(0)).sum()
        print(f"  Objective (13:00): Before={obj_b_13:.4f}, After={obj_a_13:.4f}")

if __name__ == '__main__':
    main()
