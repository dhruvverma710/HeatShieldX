import streamlit as st
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.config_loader import get_config
from src.dashboard_data import load_dashboard_data, get_exposure_mode_label

st.set_page_config(
    page_title="HeatShield X - Extreme Heat Risk Platform",
    page_icon="🌡️",
    layout="wide",
)

st.title("🌡️ HeatShield X: Extreme Heat Risk Platform")
st.caption("Hyper-local heat exposure, vulnerability, and cooling access prioritisation system.")

try:
    data = load_dashboard_data()
    config = data['config']
    mode_label = get_exposure_mode_label(data.get('mode'))
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("System Overview")
        st.write(f"**Demo Area**: {config['demo_area']}")
        st.write(f"**Config Version**: {config['config_version']}")
        st.write(f"**Computation Mode**: {mode_label}")
        
        st.info("⚠️ **Persistent Disclaimer**: HeatShield X provides modelled prioritisation scores for urban planning and emergency response, not medical thresholds. Population and facility distributions include synthetic estimates.")
        
        st.page_link("pages/1_Planner.py", label="🚀 Open HeatShield X Risk Planner", icon="🗺️", use_container_width=True)

    with col2:
        st.subheader("Data Summary")
        streets = data['streets']
        buildings = data.get('buildings')
        st.metric("Total Street Segments", len(streets))
        st.metric("Canonical Streets", streets['is_canonical'].sum())
        if 'vulnerability' in data:
            st.metric("Vulnerability Coverage", len(data['vulnerability']))
        if 'cooling_access' in data:
            st.metric("Cooling Access Network Nodes", len(data['cooling_access']))

except Exception as e:
    st.error(f"Error loading system data: {e}")
    st.info("If precomputed files are missing, please run: `python scripts/precompute.py`")
