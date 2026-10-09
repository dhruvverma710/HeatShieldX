"""Run precompute twice with different clip percentiles and report stats."""
import subprocess
import time
import pandas as pd
from pathlib import Path
import yaml
import sys

def run_precompute(clip_val):
    print(f"--- Running precompute with normalization_clip_percentile = {clip_val} ---")
    
    # 1. Update config.yaml
    config_path = Path("config/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    if "risk" not in config:
        config["risk"] = {}
        
    config["risk"]["normalization_clip_percentile"] = clip_val
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)
        
    # 2. Run precompute
    start_time = time.time()
    result = subprocess.run([sys.executable, "scripts/precompute.py"], capture_output=True, text=True)
    duration = time.time() - start_time
    
    if result.returncode != 0:
        print(f"Precompute failed! (Duration: {duration:.1f}s)")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)
        
    # Extract stage durations from stdout
    durations = []
    for line in result.stdout.splitlines():
        if "finished in" in line.lower() or "duration" in line.lower():
            durations.append(line)
            
    # 3. Read generated risk parquets and print stats
    area = config["demo_area"]
    # Quick hash since we just need to find the files
    import hashlib
    area_hash = hashlib.md5(area.encode()).hexdigest()
    cfg_ver = str(config["config_version"]).replace(".", "_")
    
    stats = []
    for t in ["0900", "1100", "1300", "1500", "1700"]:
        path = Path(f"data/processed/risk_{area_hash}_{cfg_ver}_{t}_geometric.parquet")
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        canon = df[df["is_canonical"]]
        counts = canon["risk_class"].value_counts().to_dict()
        mean_score = canon["risk_score"].mean()
        std_score = canon["risk_score"].std()
        stats.append({
            "Time": t,
            "LOW": counts.get("LOW", 0),
            "MODERATE": counts.get("MODERATE", 0),
            "HIGH": counts.get("HIGH", 0),
            "CRITICAL": counts.get("CRITICAL", 0),
            "Mean": mean_score,
            "Std": std_score
        })
        
    df_stats = pd.DataFrame(stats)
    print(f"\nStats (canonical segments only):")
    print(df_stats.to_string(index=False))
    print(f"\nPrecompute took {duration:.1f}s.")
    if durations:
        print("Stage logs:")
        for d in durations:
            print(f"  {d}")
    print("\n")

if __name__ == "__main__":
    run_precompute(100)
    run_precompute(99)
