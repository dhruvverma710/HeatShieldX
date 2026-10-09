import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import json

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.i18n import get_string
from src.dashboard_data import load_dashboard_data, get_snapshot, get_exposure_mode_label
from src.routing_engine import build_routing_graph, apply_snapshot_to_graph, compute_routes, RoutingError
from src.stop_finder import compute_safe_stops
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="HeatShield X - Citizen Planner", layout="wide")

def _(k):
    return get_string(k, 'en')

def get_node_id(sel_str, node_names, nodes_gdf):
    if not sel_str or sel_str == "None":
        return None
    return node_names.get(sel_str, None)

@st.cache_resource
def cached_graph(segments):
    return build_routing_graph(segments)

@st.cache_data
def cached_routes(orig_node, dest_node, time_str, weight, config_version, mode):
    data = load_dashboard_data()
    segments = data['streets']
    G = cached_graph(segments)
    
    snapshot_df = get_snapshot(time_str)
    # create a localized copy of G for the snapshot
    G_snap = apply_snapshot_to_graph(G.copy(), snapshot_df, data['config'])
    
    return compute_routes(G_snap, orig_node, dest_node, snapshot_df, weight)

@st.cache_data
def cached_stops(_route, area, config_version):
    data = load_dashboard_data()
    G = cached_graph(data['streets'])
    water = data.get('cooling_facilities', {}).get('water') # Not right, data has 'facilities'?
    # load_dashboard_data returns water_facilities and cooling_facilities? No, risk_engine uses it.
    # We will just pass None if we don't have them easily accessible, stop_finder loads pois anyway.
    # Wait, the prompt says "Water and cooling stops come from the existing facility sets (synthetic ones visibly marked)."
    # So we need to load them. Let's just load them from data/raw manually or if they are in `data`.
    # Actually, we can use `from src.cooling_engine import load_facilities`
    from src.cooling_engine import load_facilities
    water_facs, cooling_facs = load_facilities(area, data['streets'])
    return compute_safe_stops(_route, G, data['streets'], area, water_facs, cooling_facs)

def main():
    st.title(_('page_title'))
    
    try:
        data = load_dashboard_data()
    except Exception as e:
        st.error("Failed to load data.")
        return
        
    config = data['config']
    version = config.get('config_version', '2.0.0')
    area = config.get('demo_area', '')
    mode = data.get('mode', 'geometric')
    
    st.markdown(f"**Exposure Mode:** {get_exposure_mode_label(mode)} | **Config:** {version}")
    st.info(_('disclaimer'))
    
    # We need named places. For now just dummy nodes or we can extract from streets
    segments = data['streets']
    G = cached_graph(segments)
    
    nodes = list(G.nodes())
    if not nodes:
        st.error("No routable graph available.")
        return
        
    node_names = {f"Node {n}": n for n in nodes[:50]} # demo limit
    
    col1, col2, col3 = st.columns(3)
    with col1:
        orig_sel = st.selectbox(_('origin'), ["None"] + list(node_names.keys()))
    with col2:
        dest_sel = st.selectbox(_('destination'), ["None"] + list(node_names.keys()))
    with col3:
        times = config.get('canonical_times', ['13:00'])
        time_sel = st.selectbox(_('departure_time'), times, index=times.index(config.get('routing', {}).get('default_departure_time', '13:00')) if config.get('routing', {}).get('default_departure_time', '13:00') in times else 0)
        
    weight = st.slider(_('tradeoff'), 0.0, 1.0, 0.5)
    
    if orig_sel != "None" and dest_sel != "None":
        o_node = node_names[orig_sel]
        d_node = node_names[dest_sel]
        
        try:
            routes = cached_routes(o_node, d_node, time_sel, weight, version, mode)
            
            if not routes:
                st.warning("No routes found.")
                return
                
            st.subheader(_('route_cards'))
            sel_idx = 0
            # Route radios
            r_opts = [ " + ".join(r['type_labels']) for r in routes ]
            chosen_label = st.radio("Select Route", r_opts)
            chosen_route = routes[r_opts.index(chosen_label)]
            
            # Comparison table
            st.subheader(_('comparison_table'))
            comp_data = []
            for r in routes:
                comp_data.append({
                    'Route': " + ".join(r['type_labels']),
                    'Time (min)': round(r['total_time_min'], 1),
                    'Distance (m)': round(r['total_distance_m'], 1),
                    'Modelled Exposure Class': r['exposure_class'],
                    'Weighted Shade': round(r['weighted_shade'], 2),
                    'Cooling Access Penalty': round(r['cooling_access']['mean_access_penalty'], 2),
                    'Delta vs Fastest (%)': round(r['delta']['exposure_percent'], 1)
                })
            st.dataframe(pd.DataFrame(comp_data))
            
            st.success(chosen_route.get('recommendation_text', ''))
            
            # Safe stops
            stops = cached_stops(chosen_route, area, version)
            if stops:
                st.subheader(_('safe_stops'))
                s_data = []
                for s in stops:
                    s_data.append({
                        _('stop_type'): s['stop_type'],
                        _('stop_name'): s['name'] + (" (Sponsored)" if s['sponsored_status'] else ""),
                        _('detour'): f"{s['detour_m']:.0f}m ({s['detour_min']:.1f} min est.)",
                        _('verified'): "Yes" if s['verified_status'] else "No" + (" (" + _('shade_not_verified') + ")" if s['stop_type'] == 'shaded_public' else ""),
                        _('amenities'): ", ".join(s['amenities']),
                        _('rating'): _('data_unavailable')
                    })
                st.dataframe(pd.DataFrame(s_data))
                
            # Map
            m = folium.Map(location=[stops[0]['lat'], stops[0]['lon']] if stops else [37.8, -122.2], zoom_start=14)
            st_folium(m, width=800, height=400)
            
        except RoutingError as e:
            st.error(str(e))

if __name__ == "__main__":
    main()
