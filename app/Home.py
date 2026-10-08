import streamlit as st
import folium
from streamlit_folium import st_folium
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config_loader import get_config
from src.data_engine import load_street_network, load_buildings

st.set_page_config(page_title="HeatShield X", layout="wide")

st.title("HeatShield X: Demo Area")

@st.cache_data
def load_data(area):
    streets = load_street_network(area)
    buildings = load_buildings(area)
    
    # Project to EPSG:4326 for folium
    streets_wgs84 = streets.to_crs("EPSG:4326")
    buildings_wgs84 = buildings.to_crs("EPSG:4326")
    
    return streets_wgs84, buildings_wgs84, streets, buildings

try:
    config = get_config()
    area = config['demo_area']

    with st.sidebar:
        st.header("Configuration")
        st.write(f"**Version**: {config['config_version']}")
        st.write(f"**Demo Area**: {area}")
        
    streets_wgs84, buildings_wgs84, streets_proj, buildings_proj = load_data(area)
    
    with st.sidebar:
        st.subheader("Data Summary")
        st.write(f"**Streets**: {len(streets_proj)}")
        st.write(f"**Buildings**: {len(buildings_proj)}")
        
        st.write("**Height Sources:**")
        counts = buildings_proj['height_source'].value_counts()
        for source, count in counts.items():
            st.write(f"- {source}: {count}")

    # Create map
    # Get center
    bounds = buildings_wgs84.total_bounds # [minx, miny, maxx, maxy]
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15)
    
    # Add streets
    folium.GeoJson(
        streets_wgs84,
        style_function=lambda x: {'color': '#333333', 'weight': 2}
    ).add_to(m)
    
    # Add buildings
    folium.GeoJson(
        buildings_wgs84,
        style_function=lambda x: {'fillColor': '#ff9999', 'color': '#ff0000', 'weight': 1, 'fillOpacity': 0.5}
    ).add_to(m)
    
    st_folium(m, width=1200, height=600, returned_objects=[])
    
except Exception as e:
    st.error(f"Error initializing app: {e}")
