"""Sanity-check cached BM25/embed score parquets before tuning.
Usage: .venv/bin/python -m scripts.diagnose_blend_inputs
"""
import polars as pl
from src.common.paths import processed_dir

DATASET_CONFIG = {
    "mind": {"embed_model": "minilm", "bm25_suffix": "title_abstract_sw_nostem"},
    "ebnerd": {"embed_model": "w2v", "bm25_suffix": "title_abstract_sw_nostem"},
}

for dataset, cfg in DATASET_CONFIG.items():
    embed_path = processed_dir(dataset, "large") / f"embed_scores_{dataset}_{cfg['embed_model']}.parquet"
    bm25_path = processed_dir(dataset, "large") / f"bm25_scores_{dataset}_{cfg['bm25_suffix']}.parquet"
    if not embed_path.exists() or not bm25_path.exists():
        print(f"{dataset}: missing file, skip"); continue

    e = pl.read_parquet(embed_path)
    b = pl.read_parquet(bm25_path)
    print(f"\n{dataset}:")
    print(f"  embed rows={len(e):,}  unique impression_id={e['impression_id'].n_unique():,}")
    print(f"  bm25  rows={len(b):,}  unique impression_id={b['impression_id'].n_unique():,}")
    lens = e.select(pl.col("ranked_ids").str.count_matches(",").alias("n") + 1).to_series()
    print(f"  ranked_ids length per row: min={lens.min()} median={lens.median()} max={lens.max()}")

    joined = e.select(["impression_id"]).join(b.select(["impression_id"]), on="impression_id", how="inner")
    print(f"  JOINED rows={len(joined):,}  (should be <= min(embed rows, bm25 rows))")
    if len(joined) > 2 * min(len(e), len(b)):
        print("  ⚠️  JOIN EXPLOSION -- impression_id is not unique in one of these files.")