"""
Intervention Engine for HeatShield X.
Generates candidates for water, cooling, and shade interventions.
Labelled "MODELLED".
"""
import pandas as pd
import geopandas as gpd
import numpy as np
import logging
from src.config_loader import get_config
from src.provenance import MODELLED

logger = logging.getLogger(__name__)

def generate_candidates(segments: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame = None):
    """
    Generates candidates for water, cooling, and shade.
    
    Water/Cooling (Node-based):
    - Uses graph nodes if provided, else segment endpoints (deduplicated).
    - ID format: INV-W-{node_id} / INV-C-{node_id}
    
    Shade (Segment-based):
    - One candidate per segment.
    - ID format: INV-S-{segment_id}
    
    All have prov_status = MODELLED.
    """
    config = get_config()
    
    candidates = []
    
    # 1. Node-based (Water/Cooling)
    if nodes is not None:
        valid_nodes = nodes
    else:
        # Extract endpoints
        pts = {}
        for _, row in segments.iterrows():
            if row['u'] is not None and row['v'] is not None and row.geometry is not None:
                coords = list(row.geometry.coords)
                pts[int(row['u'])] = coords[0]
                pts[int(row['v'])] = coords[-1]
        
        valid_nodes = pd.DataFrame([
            {'node_id': k, 'x': v[0], 'y': v[1]} for k, v in pts.items()
        ])
    
    for _, row in valid_nodes.iterrows():
        nid = int(row['node_id'])
        candidates.append({
            'candidate_id': f"INV-W-{nid}",
            'type': 'water',
            'target_id': nid,
            'prov_status': MODELLED
        })
        candidates.append({
            'candidate_id': f"INV-C-{nid}",
            'type': 'cooling',
            'target_id': nid,
            'prov_status': MODELLED
        })
        
    # 2. Segment-based (Shade)
    for _, row in segments[segments['is_canonical'] == True].iterrows():
        sid = row['segment_id']
        candidates.append({
            'candidate_id': f"INV-S-{sid}",
            'type': 'shade',
            'target_id': sid,
            'prov_status': MODELLED
        })
        
    cand_df = pd.DataFrame(candidates)
    
    # Prune candidates if configured
    top_k = config.get('interventions', {}).get('candidate_pruning_top_k', None)
    if top_k is not None:
        pruned = []
        for ctype, group in cand_df.groupby('type'):
            pruned.append(group.sort_values('candidate_id').head(top_k))
        cand_df = pd.concat(pruned, ignore_index=True)

    logger.info(f"Generated {len(cand_df)} intervention candidates.")
    return cand_df

def simulate_intervention(candidate: dict, current_state: dict):
    """
    Simulates the effect of ONE candidate on the current state.
    Returns the delta in relevant metrics (e.g., exposure reduction, access penalty reduction).
    This function is a placeholder and will be used by the optimizer.
    The real evaluation happens in the optimizer by re-running the risk engine or a fast proxy.
    """
    pass
