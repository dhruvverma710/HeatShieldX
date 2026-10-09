import pandas as pd
import time
from src.data_engine import load_dashboard_data
from src.optimizer import run_optimizer
from src.intervention_engine import generate_candidates
from src.config_loader import get_config

def main():
    print("Loading data...")
    data = load_dashboard_data()
    segs = data['streets']
    vuln = data['vulnerability']
    cands = generate_candidates(segs)
    state = {
        'risk_records': data['risk_records'],
        'cooling_access': data['cooling_access'],
        'stored_bounds': data['scope_bounds']
    }
    
    print("\n=== RUNNING 0/0/0 ===")
    limits_0 = {'shade': 0, 'water': 0, 'cooling': 0}
    t0 = time.time()
    sel_0, state_0, meta_0 = run_optimizer(cands, segs, vuln, state, limits_0)
    print(f"Runtime: {meta_0['runtime_s']:.2f} s")
    print(f"Objective Before: {meta_0['objective_before']:.2f} | After: {meta_0['objective_after']:.2f}")
    
    print("\n=== RUNNING 2/1/2 ===")
    # Clear cache to force exact evaluation for timing
    from src.optimizer import _optimizer_cache
    _optimizer_cache.clear()
    
    limits_212 = {'shade': 2, 'water': 1, 'cooling': 2}
    t1 = time.time()
    sel_212, state_212, meta_212 = run_optimizer(cands, segs, vuln, state, limits_212)
    t_end = time.time()
    
    print("\n--- Optimizer Plan Table (2/1/2) ---")
    print("Priority | Type | Candidate ID | Street Label | Marginal Ben | Cumulative Ben | Newly Covered")
    for i, s in enumerate(sel_212):
        nc = s.get('newly_covered_streets', 0)
        print(f"{i+1} | {s['type']} | {s['candidate_id']} | {s['target_id']} | {s.get('marginal_benefit', 0):.2f} | {s.get('cumulative_benefit', 0):.2f} | {nc}")
        
    print(f"\nRuntime: {meta_212['runtime_s']:.2f} s")
    print(f"Objective Before: {meta_212['objective_before']:.2f} | After: {meta_212['objective_after']:.2f}")

    print("\n--- Impact Table (13:00) ---")
    r_before = state['risk_records']['13:00']
    r_after = state_212['risk_records']['13:00']
    
    def count_high_crit(df):
        return len(df[df['risk_class'].isin(['HIGH', 'CRITICAL'])])
        
    print(f"HIGH/CRITICAL counts - Before: {count_high_crit(r_before)}, After: {count_high_crit(r_after)}")
    
    # Pooled counts
    high_crit_before_pooled = sum(count_high_crit(r) for r in state['risk_records'].values())
    high_crit_after_pooled = sum(count_high_crit(r) for r in state_212['risk_records'].values())
    print(f"HIGH/CRITICAL counts (Pooled) - Before: {high_crit_before_pooled}, After: {high_crit_after_pooled}")

if __name__ == '__main__':
    main()
