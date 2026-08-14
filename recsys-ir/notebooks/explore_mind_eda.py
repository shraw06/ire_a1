"""MIND EDA exploration script — runs all analyses and outputs results as JSON.

This script is used by the notebook 00_explore_mind.ipynb and can also be run
standalone.  It performs all analyses described in the EDA requirements:
  - File listing and sizes
  - Row counts
  - Schema / head(5)
  - Timestamp min/max
  - Click-through rate
  - Clicks per user distribution
  - Impressions per article distribution
  - % articles missing abstract/body
  - Entity annotation format inspection
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path
from datetime import datetime

# Paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIND_DIR = PROJECT_ROOT / "data" / "raw" / "mind"
TRAIN_DIR = MIND_DIR / "MINDsmall_train"
DEV_DIR = MIND_DIR / "MINDsmall_dev"

# MIND TSV column names (no headers in files)
NEWS_COLS = ["news_id", "category", "subcategory", "title", "abstract", "url",
             "title_entities", "abstract_entities"]
BEHAVIORS_COLS = ["impression_id", "user_id", "time", "history", "impressions"]


def file_listing(base_dir: Path) -> list[dict]:
    """Return file sizes for all files under base_dir."""
    result = []
    for p in sorted(base_dir.rglob("*")):
        if p.is_file() and not p.name.startswith("."):
            rel = str(p.relative_to(base_dir))
            size_mb = p.stat().st_size / 1024**2
            result.append({"path": rel, "size_mb": round(size_mb, 2)})
    return result


def row_counts(split_dir: Path) -> dict[str, int]:
    """Count lines in each TSV file."""
    counts = {}
    for f in sorted(split_dir.glob("*")):
        if f.is_file():
            with open(f, "r") as fh:
                counts[f.name] = sum(1 for _ in fh)
    return counts


def read_news(split_dir: Path) -> list[dict]:
    """Read news.tsv into a list of dicts."""
    rows = []
    with open(split_dir / "news.tsv", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= len(NEWS_COLS):
                rows.append(dict(zip(NEWS_COLS, parts[:len(NEWS_COLS)])))
    return rows


def read_behaviors(split_dir: Path) -> list[dict]:
    """Read behaviors.tsv into a list of dicts."""
    rows = []
    with open(split_dir / "behaviors.tsv", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= len(BEHAVIORS_COLS):
                rows.append(dict(zip(BEHAVIORS_COLS, parts[:len(BEHAVIORS_COLS)])))
    return rows


def timestamp_range(behaviors: list[dict]) -> dict:
    """Find min/max timestamps from behaviors."""
    times = []
    for b in behaviors:
        try:
            t = datetime.strptime(b["time"], "%m/%d/%Y %I:%M:%S %p")
            times.append(t)
        except (ValueError, KeyError):
            pass
    if not times:
        return {"min": None, "max": None}
    return {
        "min": min(times).isoformat(),
        "max": max(times).isoformat(),
    }


def click_through_rate(behaviors: list[dict]) -> dict:
    """Compute CTR from impression-article pairs."""
    total_pairs = 0
    positive_pairs = 0
    for b in behaviors:
        imps = b.get("impressions", "").split()
        for imp in imps:
            if "-" in imp:
                _, label = imp.rsplit("-", 1)
                total_pairs += 1
                if label == "1":
                    positive_pairs += 1
    ctr = positive_pairs / total_pairs if total_pairs > 0 else 0
    return {
        "total_impression_article_pairs": total_pairs,
        "positive_clicks": positive_pairs,
        "ctr": round(ctr, 4),
    }


def clicks_per_user(behaviors: list[dict]) -> dict:
    """Distribution of clicks per user."""
    user_clicks: Counter = Counter()
    for b in behaviors:
        user_id = b.get("user_id", "")
        imps = b.get("impressions", "").split()
        for imp in imps:
            if "-" in imp:
                _, label = imp.rsplit("-", 1)
                if label == "1":
                    user_clicks[user_id] += 1
    # Compute distribution stats
    click_counts = list(user_clicks.values())
    if not click_counts:
        return {}
    click_counts.sort()
    n = len(click_counts)
    return {
        "num_users_with_clicks": n,
        "min": min(click_counts),
        "max": max(click_counts),
        "mean": round(sum(click_counts) / n, 2),
        "median": click_counts[n // 2],
        "p75": click_counts[int(n * 0.75)],
        "p90": click_counts[int(n * 0.90)],
        "p95": click_counts[int(n * 0.95)],
        "p99": click_counts[int(n * 0.99)],
    }


def impressions_per_article(behaviors: list[dict]) -> dict:
    """Distribution of impressions per article (head vs tail)."""
    article_counts: Counter = Counter()
    for b in behaviors:
        imps = b.get("impressions", "").split()
        for imp in imps:
            if "-" in imp:
                nid = imp.rsplit("-", 1)[0]
                article_counts[nid] += 1
    counts = sorted(article_counts.values(), reverse=True)
    if not counts:
        return {}
    n = len(counts)
    return {
        "num_articles_in_impressions": n,
        "min": min(counts),
        "max": max(counts),
        "mean": round(sum(counts) / n, 2),
        "median": counts[n // 2],
        "p75": counts[int(n * 0.25)],  # top 25% threshold (sorted desc)
        "p90": counts[int(n * 0.10)],
        "p95": counts[int(n * 0.05)],
        "top10_articles": counts[:10],
    }


def missing_text_stats(news: list[dict]) -> dict:
    """% of articles missing abstract or body/url text."""
    n = len(news)
    if n == 0:
        return {}
    missing_abstract = sum(1 for a in news if not a.get("abstract", "").strip())
    missing_title = sum(1 for a in news if not a.get("title", "").strip())
    return {
        "total_articles": n,
        "missing_abstract": missing_abstract,
        "missing_abstract_pct": round(missing_abstract / n * 100, 2),
        "missing_title": missing_title,
        "missing_title_pct": round(missing_title / n * 100, 2),
    }


def entity_annotation_sample(news: list[dict], n: int = 3) -> list[dict]:
    """Return a few entity annotation samples."""
    samples = []
    for a in news:
        te = a.get("title_entities", "")
        ae = a.get("abstract_entities", "")
        if te and te != "[]":
            samples.append({
                "news_id": a["news_id"],
                "title_entities": te[:500],
                "abstract_entities": ae[:500],
            })
            if len(samples) >= n:
                break
    return samples


def main():
    print("=" * 70)
    print("MIND SMALL DATASET - EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    results = {}

    # 1) File listing
    print("\n File listing:")
    files = file_listing(MIND_DIR)
    for f in files:
        print(f"   {f['path']:50s} {f['size_mb']:>8.2f} MB")
    results["file_listing"] = files

    for split_name, split_dir in [("train", TRAIN_DIR), ("dev", DEV_DIR)]:
        print(f"\n{'='*70}")
        print(f"  Split: {split_name}")
        print(f"{'='*70}")

        # 2) Row counts
        print("\n Row counts:")
        rc = row_counts(split_dir)
        for fname, count in rc.items():
            print(f"   {fname}: {count:,}")
        results[f"{split_name}_row_counts"] = rc

        # 3) Schema + head(5)
        print("\n News schema & head(5):")
        news = read_news(split_dir)
        print(f"   Columns: {NEWS_COLS}")
        for i, row in enumerate(news[:5]):
            print(f"   Row {i}: news_id={row['news_id']}, cat={row['category']}, "
                  f"title={row['title'][:60]}…")

        behaviors = read_behaviors(split_dir)
        print(f"\n Behaviors schema & head(5):")
        print(f"   Columns: {BEHAVIORS_COLS}")
        for i, row in enumerate(behaviors[:5]):
            n_hist = len(row.get('history', '').split()) if row.get('history') else 0
            n_imp = len(row.get('impressions', '').split()) if row.get('impressions') else 0
            print(f"   Row {i}: imp_id={row['impression_id']}, user={row['user_id']}, "
                  f"time={row['time']}, hist_len={n_hist}, imp_len={n_imp}")

        # Store head(5) as serializable
        results[f"{split_name}_news_head5"] = news[:5]
        results[f"{split_name}_behaviors_head5"] = behaviors[:5]

        # 4) Timestamp range
        print("\n Timestamp range:")
        ts = timestamp_range(behaviors)
        print(f"   Min: {ts['min']}")
        print(f"   Max: {ts['max']}")
        results[f"{split_name}_timestamp_range"] = ts

        # 5) CTR
        print("\n Click-through rate:")
        ctr = click_through_rate(behaviors)
        print(f"   Total impression-article pairs: {ctr['total_impression_article_pairs']:,}")
        print(f"   Positive clicks: {ctr['positive_clicks']:,}")
        print(f"   CTR: {ctr['ctr']:.4f} ({ctr['ctr']*100:.2f}%)")
        results[f"{split_name}_ctr"] = ctr

        # 6) Clicks per user
        print("\n Clicks per user distribution:")
        cpu = clicks_per_user(behaviors)
        for k, v in cpu.items():
            print(f"   {k}: {v}")
        results[f"{split_name}_clicks_per_user"] = cpu

        # 7) Impressions per article
        print("\n Impressions per article distribution:")
        ipa = impressions_per_article(behaviors)
        for k, v in ipa.items():
            if k != "top10_articles":
                print(f"   {k}: {v}")
        print(f"   top10_articles (impression counts): {ipa.get('top10_articles', [])}")
        results[f"{split_name}_impressions_per_article"] = ipa

        # 8) Missing text
        print("\n Missing text stats:")
        mt = missing_text_stats(news)
        for k, v in mt.items():
            print(f"   {k}: {v}")
        results[f"{split_name}_missing_text"] = mt

        # 9) Entity annotations
        print("\n Entity annotation samples:")
        ea = entity_annotation_sample(news)
        for s in ea:
            print(f"   {s['news_id']}: title_entities={s['title_entities'][:120]}…")
        results[f"{split_name}_entity_samples"] = ea

    # Entity embedding format
    print("\n Entity embedding format:")
    emb_file = TRAIN_DIR / "entity_embedding.vec"
    with open(emb_file) as f:
        first_line = f.readline().strip()
    parts = first_line.split("\t")
    print(f"   WikiData ID: {parts[0]}")
    print(f"   Embedding dim: {len(parts) - 1}")
    print(f"   Sample values: {parts[1:4]}")
    results["entity_embedding_dim"] = len(parts) - 1

    rel_file = TRAIN_DIR / "relation_embedding.vec"
    with open(rel_file) as f:
        first_line = f.readline().strip()
    parts = first_line.split("\t")
    print(f"   Relation embedding dim: {len(parts) - 1}")
    results["relation_embedding_dim"] = len(parts) - 1

    # Save results
    out_path = PROJECT_ROOT / "notebooks" / "mind_eda_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n Results saved to {out_path}")


if __name__ == "__main__":
    main()
