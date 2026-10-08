"""
Precompute script for HeatShield X.
Runs the full pipeline: data → shadows → exposure → vulnerability → cooling → risk.
Uses caches. Logs duration per stage and config version.
"""
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import logging
import pandas as pd
from src.config_loader import get_config
from src.data_engine import load_street_network, load_buildings, _get_area_hash

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / 'data' / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def timed(label):
    """Context manager to time a stage."""
    class Timer:
        def __enter__(self):
            self.t0 = time.time()
            logger.info(f"[STAGE] {label} — starting...")
            return self
        def __exit__(self, *args):
            elapsed = time.time() - self.t0
            logger.info(f"[STAGE] {label} — done in {elapsed:.1f}s")
    return Timer()


def main():
    config = get_config()
    area = config['demo_area']
    area_hash = _get_area_hash(area)
    cfg_version = config['config_version'].replace('.', '_')
    times = config['canonical_times']

    logger.info(f"Config version: {config['config_version']}")
    logger.info(f"Area: {area}")

    # ── Stage 1: Data ──
    with timed("Load streets"):
        streets = load_street_network(area)
    with timed("Load buildings"):
        buildings = load_buildings(area)

    # ── Stage 2: Shadows ──
    with timed("Compute shadows"):
        from scripts.compute_shadows import main as compute_shadows_main
        # Check if shadows already cached
        from src.exposure_engine import check_geometric_shadows_available
        if check_geometric_shadows_available(times, area_hash, cfg_version, PROCESSED_DIR):
            logger.info("Shadow cache already present, skipping recomputation.")
        else:
            compute_shadows_main()

    # ── Stage 3: Exposure ──
    with timed("Compute exposure"):
        from src.exposure_engine import compute_exposure

        # Geometric
        geo_results, geo_mode = compute_exposure(streets, buildings, area_hash, PROCESSED_DIR)
        for t, df in geo_results.items():
            hour, minute = map(int, t.split(':'))
            out_path = PROCESSED_DIR / f"exposure_{area_hash}_{cfg_version}_{hour:02d}{minute:02d}_geometric.parquet"
            df.to_parquet(out_path)

        # Fallback
        fb_results, fb_mode = compute_exposure(streets, buildings, area_hash, PROCESSED_DIR,
                                                force_mode='estimated_exposure_mode')
        for t, df in fb_results.items():
            hour, minute = map(int, t.split(':'))
            out_path = PROCESSED_DIR / f"exposure_{area_hash}_{cfg_version}_{hour:02d}{minute:02d}_fallback.parquet"
            df.to_parquet(out_path)

    # ── Stage 4: Vulnerability ──
    with timed("Compute vulnerability"):
        from src.vulnerability_engine import compute_vulnerability
        vuln = compute_vulnerability(streets, buildings)
        vuln.to_parquet(PROCESSED_DIR / f"vulnerability_{area_hash}_{cfg_version}.parquet")

    # ── Stage 5: Cooling access ──
    with timed("Compute cooling access"):
        from src.cooling_engine import compute_cooling_access
        cooling, water_fac, cooling_fac = compute_cooling_access(streets, area)
        cooling.to_parquet(PROCESSED_DIR / f"cooling_{area_hash}_{cfg_version}.parquet")

    # ── Stage 6: Risk ──
    with timed("Compute risk"):
        from src.risk_engine import compute_risk

        # Geometric risk
        geo_risk, geo_bounds = compute_risk(geo_results, vuln, cooling, geo_mode)
        for t, df in geo_risk.items():
            hour, minute = map(int, t.split(':'))
            out_path = PROCESSED_DIR / f"risk_{area_hash}_{cfg_version}_{hour:02d}{minute:02d}_geometric.parquet"
            df.to_parquet(out_path)

        # Fallback risk
        fb_risk, fb_bounds = compute_risk(fb_results, vuln, cooling, fb_mode)
        for t, df in fb_risk.items():
            hour, minute = map(int, t.split(':'))
            out_path = PROCESSED_DIR / f"risk_{area_hash}_{cfg_version}_{hour:02d}{minute:02d}_fallback.parquet"
            df.to_parquet(out_path)

        # Store bounds
        bounds_df = pd.DataFrame({
            'mode': ['geometric', 'fallback'],
            'scope_min': [geo_bounds[0], fb_bounds[0]],
            'scope_max': [geo_bounds[1], fb_bounds[1]],
        })
        bounds_df.to_parquet(PROCESSED_DIR / f"risk_bounds_{area_hash}_{cfg_version}.parquet")

    logger.info("=" * 50)
    logger.info("PRECOMPUTE COMPLETE")
    logger.info(f"Config version: {config['config_version']}")
    logger.info(f"Segments: {len(streets)}, Canonical: {streets['is_canonical'].sum()}")
    logger.info(f"Buildings: {len(buildings)}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
