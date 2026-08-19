import polars as pl
import json
import numpy as np
from pathlib import Path
import csv

def compute_candidate_pool_sizes():
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    datasets = ["mind", "ebnerd"]
    results = []

    for ds in datasets:
        behaviors_path = _PROJECT_ROOT / "data" / "interim" / ds / "behaviors.parquet"
        if not behaviors_path.exists():
            continue
        df = pl.read_parquet(behaviors_path)
        lens = []
        for row in df.iter_rows(named=True):
            candidates = json.loads(row["candidates"])
            if candidates:
                lens.append(len(candidates))
        
        if lens:
            results.append({
                "dataset": ds,
                "min": int(np.min(lens)),
                "median": float(np.median(lens)),
                "mean": float(np.mean(lens)),
                "p90": float(np.percentile(lens, 90)),
                "max": int(np.max(lens))
            })

    out_path = _PROJECT_ROOT / "results" / "candidate_pool_sizes.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "min", "median", "mean", "p90", "max"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Wrote {out_path}")
    for r in results:
        print(r)

if __name__ == "__main__":
    compute_candidate_pool_sizes()
