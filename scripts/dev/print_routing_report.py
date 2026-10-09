import pandas as pd
from src.routing_engine import build_routing_graph, apply_snapshot_to_graph, compute_routes
from src.data_engine import load_dashboard_data
from src.config_loader import get_config

def main():
    data = load_dashboard_data()
    segs = data['streets']
    G = build_routing_graph(segs)
    
    # Pick two real O-D pairs. We can just pick random distinct nodes in the largest connected component.
    import networkx as nx
    lcc = max(nx.weakly_connected_components(G), key=len)
    nodes = list(lcc)
    
    # Let's pick 3 specific nodes so we have two pairs: A->B, and C->D
    # Just picking the first few nodes in the list.
    import random
    random.seed(42)
    sample_nodes = random.sample(nodes, 4)
    pair1 = (sample_nodes[0], sample_nodes[1])
    pair2 = (sample_nodes[2], sample_nodes[3])
    
    times_to_test = [('13:00', pair1), ('13:00', pair2), ('09:00', pair1), ('17:00', pair1)]
    
    for t, (u, v) in times_to_test:
        print(f"\n=== Route {u} -> {v} at {t} ===")
        G_snap = apply_snapshot_to_graph(G, data['risk_records'][t], data['cooling_access'])
        routes = compute_routes(G_snap, u, v, get_config())
        for r in routes:
            print(f"[{', '.join(r['type_labels'])}]")
            print(f"  Time: {r['total_time_min']:.1f} min | Dist: {r['total_distance_m']:.1f} m")
            print(f"  Exposure: {r['modelled_heat_exposure']:.3f} | Shade: {r['weighted_shade_fraction']:.2f}")
            print(f"  Cooling access: {r['cooling_access_score']:.2f}")
            print(f"  Message: {r.get('message', '')}")
            if r.get('delta_vs_fastest'):
                print(f"  Delta: +{r['delta_vs_fastest'].get('time_added_min', 0):.1f} min")

if __name__ == '__main__':
    main()
