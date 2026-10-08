import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import geopandas as gpd
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.dashboard_data import (
    load_dashboard_data,
    get_snapshot,
    nearest_canonical_segment,
    top_priority_streets,
    class_counts,
    get_segment_detail,
    get_exposure_mode_label,
    compare_hottest_vs_highest_risk,
)
from src.cooling_engine import load_facilities

st.set_page_config(page_title="Risk Planner - HeatShield X", page_icon="🗺️", layout="wide")

# ── 1. Load Data & Config ──
try:
    data = load_dashboard_data()
    config = data['config']
    streets = data['streets']
except Exception as e:
    st.error(f"Error loading precomputed data: {e}")
    st.info("Please run `python scripts/precompute.py` first.")
    st.stop()

# ── 2. Sidebar Controls ──
st.sidebar.title("🗺️ Risk Planner")

time_minutes = st.sidebar.slider(
    "Select Time of Day",
    min_value=540,
    max_value=1020,
    value=780,  # Default 13:00
    step=30,
    format="%d",
)
hours = time_minutes // 60
mins = time_minutes % 60
time_str = f"{hours:02d}:{mins:02d}"
st.sidebar.write(f"**Selected Time**: `{time_str}`")

# Time Badge
canonical_mins = {540, 660, 780, 900, 1020}
if time_minutes in canonical_mins:
    st.sidebar.success("Mode: Computed (Canonical)")
else:
    st.sidebar.info("Mode: Interpolated (estimated)")

show_facilities = st.sidebar.checkbox("Show Water & Cooling Facilities", value=True)

# ── 3. Status Strip & Disclaimer ──
mode_label = get_exposure_mode_label(data.get('mode'))
cfg_ver = config['config_version']

st.title("🗺️ Extreme Heat Risk Planner")

status_html = f"""
<div style="background-color: #f8f9fa; padding: 10px 15px; border-radius: 6px; margin-bottom: 15px; border-left: 4px solid #1f77b4;">
    <strong>Exposure Mode:</strong> {mode_label} &nbsp;|&nbsp; 
    <strong>Config Version:</strong> {cfg_ver} &nbsp;|&nbsp; 
    <span style="background-color: #e2e3e5; padding: 2px 6px; border-radius: 4px; font-size: 0.85em;">Modelled</span> &nbsp;
    <span style="background-color: #e2e3e5; padding: 2px 6px; border-radius: 4px; font-size: 0.85em;">Estimated/Synthetic population</span> &nbsp;
    <span style="background-color: #e2e3e5; padding: 2px 6px; border-radius: 4px; font-size: 0.85em;">Synthetic facilities</span>
</div>
"""
st.markdown(status_html, unsafe_allow_html=True)
st.caption("⚠️ **Disclaimer**: HeatShield X provides modelled prioritisation scores for urban planning, not medical thresholds. Population and facility locations include synthetic estimates.")

# ── 4. Snapshot & Priority Streets ──
snapshot_df = get_snapshot(time_minutes, data=data)
canon_df = snapshot_df[snapshot_df['is_canonical'] == True].copy()

top20_df = top_priority_streets(time_minutes, n=20, data=data)

# Street Selection Controls
col_sel1, col_sel2 = st.columns([2, 1])
with col_sel1:
    select_options = ["-- None selected --"] + [f"{r['name']} ({r['segment_id']})" for _, r in top20_df.iterrows()]
    selected_option = st.selectbox("Select Priority Street (Top 20)", options=select_options)

selected_segment_id = None
if selected_option != "-- None selected --":
    selected_segment_id = selected_option.split("(")[-1].rstrip(")")

# ── 5. Folium Map Setup ──
# Attach geometry if needed
if 'geometry' not in canon_df.columns:
    canon_df = canon_df.merge(streets[['segment_id', 'geometry']], on='segment_id', how='left')

canon_gdf = gpd.GeoDataFrame(canon_df, crs=streets.crs).to_crs("EPSG:4326")

bounds = canon_gdf.total_bounds
center_lat = float((bounds[1] + bounds[3]) / 2)
center_lon = float((bounds[0] + bounds[2]) / 2)

m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="OpenStreetMap")

# Color mapping
color_map = {
    'LOW': '#27ae60',
    'MODERATE': '#f39c12',
    'HIGH': '#e67e22',
    'CRITICAL': '#e74c3c',
}

# Add canonical segments
for _, row in canon_gdf.iterrows():
    r_class = row.get('risk_class', 'LOW')
    r_score = row.get('risk_score', 0)
    s_name = row.get('name', 'Unnamed street')
    s_id = row['segment_id']
    color = color_map.get(r_class, '#27ae60')

    is_selected = (s_id == selected_segment_id)
    weight = 6 if is_selected else 3
    opacity = 1.0 if is_selected else 0.75

    folium.GeoJson(
        row.geometry,
        style_function=lambda x, c=color, w=weight, op=opacity: {'color': c, 'weight': w, 'opacity': op},
        tooltip=f"<b>{s_name}</b> ({s_id})<br>Risk Score: {r_score:.0f} ({r_class})",
    ).add_to(m)

# Highlight selected segment
if selected_segment_id:
    sel_row = canon_gdf[canon_gdf['segment_id'] == selected_segment_id]
    if not sel_row.empty:
        folium.GeoJson(
            sel_row.iloc[0].geometry,
            style_function=lambda x: {'color': '#8e44ad', 'weight': 8, 'opacity': 1.0},
            tooltip=f"<b>SELECTED: {sel_row.iloc[0].get('name')}</b>",
        ).add_to(m)

# Facility Markers
if show_facilities:
    try:
        water_fac, cooling_fac = load_facilities(config['demo_area'], streets)
        if len(water_fac) > 0:
            water_wgs = water_fac.to_crs("EPSG:4326")
            for _, w_row in water_wgs.iterrows():
                src = w_row.get('source', 'osm')
                label = f"Water Point ({src.upper()})"
                color = '#00bc8c' if src == 'osm' else '#3498db'
                folium.CircleMarker(
                    location=[w_row.geometry.y, w_row.geometry.x],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.8,
                    popup=label,
                    tooltip=label,
                ).add_to(m)
        if len(cooling_fac) > 0:
            cooling_wgs = cooling_fac.to_crs("EPSG:4326")
            for _, c_row in cooling_wgs.iterrows():
                src = c_row.get('source', 'osm')
                label = f"Cooling Centre ({src.upper()})"
                color = '#9b59b6' if src == 'osm' else '#e74c3c'
                folium.CircleMarker(
                    location=[c_row.geometry.y, c_row.geometry.x],
                    radius=6,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.9,
                    popup=label,
                    tooltip=label,
                ).add_to(m)
    except Exception as e:
        pass

# Render Map
map_output = st_folium(m, width=None, height=450, use_container_width=True, returned_objects=["last_clicked"])

# Map click selection
if map_output and map_output.get("last_clicked"):
    click_lat = map_output["last_clicked"]["lat"]
    click_lon = map_output["last_clicked"]["lng"]
    clicked_seg = nearest_canonical_segment(click_lat, click_lon, data=data)
    if clicked_seg:
        selected_segment_id = clicked_seg['segment_id']

# ── 6. Tabs for Dashboard Panels ──
tab_risk, tab_why, tab_hottest = st.tabs(["📊 Risk Overview", "🔍 WHY Panel (Detail)", "🔥 Why Not The Hottest?"])

# ── Tab 1: Risk Overview ──
with tab_risk:
    st.subheader(f"Risk Distribution at {time_str}")
    counts = class_counts(time_minutes, data=data)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LOW Risk", counts.get('LOW', 0))
    c2.metric("MODERATE Risk", counts.get('MODERATE', 0))
    c3.metric("HIGH Risk", counts.get('HIGH', 0))
    c4.metric("CRITICAL Risk", counts.get('CRITICAL', 0))

    st.subheader("Top 10 Priority Streets")
    top10_df = top_priority_streets(time_minutes, n=10, data=data)
    disp_cols = ['segment_id', 'name', 'risk_score', 'risk_class', 'highway']
    disp_cols = [c for c in disp_cols if c in top10_df.columns]
    st.dataframe(top10_df[disp_cols].reset_index(drop=True), use_container_width=True)

# ── Tab 2: WHY Panel (Detail & Plotly Chart) ──
with tab_why:
    if not selected_segment_id:
        st.info("Click on a street segment on the map or select a street from the dropdown above to view driver details.")
    else:
        detail = get_segment_detail(selected_segment_id, time_minutes, data=data)
        if not detail:
            st.warning(f"No details available for segment `{selected_segment_id}`.")
        else:
            st.subheader(f"Street Detail: {detail['street_name']} (`{detail['segment_id']}`)")
            
            w_col1, w_col2, w_col3 = st.columns([1, 1, 2])
            w_col1.metric("Risk Score", f"{detail['risk_score']:.0f}" if detail['risk_score'] is not None else "N/A")
            w_col2.metric("Risk Class", detail['risk_class'])
            with w_col3:
                st.write("**Nearest Facilities:**")
                st.write(f"- Water Point: **{detail['dist_water_m']:.0f} m** ({'Covered' if detail['covered_water'] else 'Far'})")
                st.write(f"- Cooling Centre: **{detail['dist_cooling_m']:.0f} m** ({'Covered' if detail['covered_cooling'] else 'Far'})")

            st.markdown("#### Explanation")
            st.write(f"> *\"{detail['explanation_text']}\"*")

            st.markdown("#### Risk Drivers")
            if detail['drivers']:
                d_df = pd.DataFrame(detail['drivers'])
                d_df['driver'] = d_df['driver'].apply(lambda x: x.replace('_', ' '))
                d_df['percentile'] = d_df['percentile'].apply(lambda p: f"{p:.0%}")
                d_df['value'] = d_df['value'].apply(lambda v: f"{v:.4f}")
                d_df.rename(columns={
                    'driver': 'Driver',
                    'value': 'Raw Value',
                    'percentile': 'Scope Percentile',
                    'level': 'Level',
                    'is_dominant': 'Dominant Driver?'
                }, inplace=True)
                st.table(d_df)

            # Provenance Expander
            with st.expander("🔍 Provenance Metadata"):
                prov = detail['provenance']
                st.write(f"- **Status**: `{prov['prov_status']}`")
                st.write(f"- **Observation/Estimate**: `{prov['prov_obs_est']}`")
                st.write(f"- **Computation Mode**: `{prov['computation_mode']}`")
                st.write(f"- **Assumptions Version**: `{prov['assumptions_version']}`")

            # Plotly Chart across times
            st.markdown("#### Risk Trend Across Canonical Times")
            canon_times = ["09:00", "11:00", "13:00", "15:00", "17:00"]
            time_scores = []
            for ct in canon_times:
                r_rec = data['risk_records'][ct]
                s_rec = r_rec[r_rec['segment_id'] == selected_segment_id]
                if not s_rec.empty:
                    time_scores.append(s_rec['risk_score'].iloc[0])
                else:
                    time_scores.append(None)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=canon_times,
                y=time_scores,
                mode='lines+markers',
                name='Risk Score',
                line=dict(color='#e74c3c', width=3),
                marker=dict(size=8),
            ))
            if time_str in canon_times:
                fig.add_shape(
                    type="line",
                    x0=time_str,
                    x1=time_str,
                    y0=0,
                    y1=105,
                    line=dict(color="#8e44ad", width=2, dash="dash"),
                )
            fig.update_layout(
                title=f"Risk Score Trend for {detail['street_name']}",
                xaxis_title="Canonical Time",
                yaxis_title="Risk Score (0-100)",
                yaxis=dict(range=[0, 105]),
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

# ── Tab 3: "Why Not The Hottest?" Comparison ──
with tab_hottest:
    st.subheader("🔥 Hottest Exposure vs Highest Risk Comparison")
    
    # Determine nearest canonical time for risk/exposure comparison
    canon_times = ["09:00", "11:00", "13:00", "15:00", "17:00"]
    canonical_mins_list = [540, 660, 780, 900, 1020]
    idx_nearest = int(np.argmin([abs(time_minutes - m) for m in canonical_mins_list]))
    t_nearest = canon_times[idx_nearest]
    
    r_df = data['risk_records'][t_nearest]
    e_df = data['exposure_records'].get(t_nearest, r_df)
    
    comp = compare_hottest_vs_highest_risk(e_df, r_df, canonical_only=True)
    
    h_seg = streets[streets['segment_id'] == comp['hottest_segment_id']]
    r_seg = streets[streets['segment_id'] == comp['highest_risk_segment_id']]
    
    h_name = h_seg['name'].iloc[0] if not h_seg.empty else comp['hottest_segment_id']
    r_name = r_seg['name'].iloc[0] if not r_seg.empty else comp['highest_risk_segment_id']
    
    if comp['same_street']:
        st.success(f"At {t_nearest}, **{h_name}** is both the street with the highest heat exposure and the highest overall risk score ({comp['highest_risk_score']:.0f}).")
    else:
        st.info(
            f"At {t_nearest}, the street with the highest heat exposure is **{h_name}** (exposure {comp['hottest_exposure']:.2f}, risk score {comp['hottest_risk_score']:.0f}), "
            f"whereas the street with the highest overall risk is **{r_name}** (risk score {comp['highest_risk_score']:.0f}, exposure {comp['highest_exposure']:.2f}) "
            f"due to higher vulnerability ({comp['highest_vulnerability']:.2f} vs {comp['hottest_vulnerability']:.2f}) "
            f"and cooling access penalty ({comp['highest_access_penalty']:.2f} vs {comp['hottest_access_penalty']:.2f})."
        )
        
    c_h1, c_h2 = st.columns(2)
    with c_h1:
        st.markdown(f"### 🔥 Hottest Street: {h_name}")
        st.write(f"- **Segment ID**: `{comp['hottest_segment_id']}`")
        st.write(f"- **Exposure Value**: {comp['hottest_exposure']:.4f}")
        st.write(f"- **Risk Score**: {comp['hottest_risk_score']:.0f}" if comp['hottest_risk_score'] is not None else "N/A")
        st.write(f"- **Vulnerability**: {comp['hottest_vulnerability']:.4f}" if comp['hottest_vulnerability'] is not None else "N/A")
        st.write(f"- **Cooling Penalty**: {comp['hottest_access_penalty']:.4f}" if comp['hottest_access_penalty'] is not None else "N/A")

    with c_h2:
        st.markdown(f"### ⚠️ Highest Risk Street: {r_name}")
        st.write(f"- **Segment ID**: `{comp['highest_risk_segment_id']}`")
        st.write(f"- **Risk Score**: {comp['highest_risk_score']:.0f}")
        st.write(f"- **Exposure Value**: {comp['highest_exposure']:.4f}")
        st.write(f"- **Vulnerability**: {comp['highest_vulnerability']:.4f}")
        st.write(f"- **Cooling Penalty**: {comp['highest_access_penalty']:.4f}")
