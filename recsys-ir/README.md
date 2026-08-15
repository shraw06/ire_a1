# recsys-ir - Unified News Recommendation Retrieval Pipeline

Lexical (BM25) and semantic (embedding) retrieval + evaluation on **MIND** (English) and **EB-NeRD** (Danish), using a **single unified schema and shared retrieval/evaluation interface**.

## Quick Start - One-Command Reproduce

```bash
# 1. Create environment & install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" #review

# 2. Download data, parse, split, build features, retrieve, evaluate
make data      # download + parse + temporal split + feature stores (~14s on MIND-small + EB-NeRD demo)
make bm25      # hand-built inverted index + BM25 scoring
make embed     # load / compute embeddings + ANN retrieval
make eval       # AUC, MRR, nDCG@{5,10}, diversity, novelty, coverage
make submit    # Codabench prediction files
```

> **Note:** `make data` requires internet access for initial dataset downloads.
> After the first run, all pipeline stages work offline.
> Re-runs are idempotent (each target skips if outputs exist; use `FORCE=1 make data` to rebuild).

### Pipeline timing (`make data` on real data, single run)

| Stage | Wall-clock |
|-------|-----------|
| Download | skipped (cached) |
| Parse | skipped (cached) |
| Split | ~1s |
| Features (MIND-small: 65K articles, 94K users) | ~6s |
| Features (EB-NeRD demo: 12K articles, 2K users) | ~7s |
| **Total** | **~14s** |

## Repository Layout

```
recsys-ir/
├── configs/          # YAML configs - base + per-dataset overrides
├── data/             # gitignored - raw, interim, processed splits
├── src/
│   ├── ingestion/    # download + unzip + checksum
│   ├── parsing/      # dataset-specific → unified schema adapters
│   ├── splitting/    # temporal train/val/test split
│   ├── feature_store/ # article & user feature stores
│   ├── retrieval/    # BM25, ANN, embedding, candidate generation
│   ├── evaluation/   # ranking metrics, beyond-accuracy, slicing, CI
│   ├── submission/   # Codabench-format output
│   └── common/       # schema definitions, logging utilities
├── notebooks/        # EDA notebooks (read-only after exploration)
├── tests/            # pytest suite incl. anti-leakage & smoke tests
├── design_note/      # LaTeX/Markdown source for the ≤4-page PDF
├── logs/             # AI usage log
└── scripts/          # orchestration scripts
```

## Design Principles

1. **One schema, one interface** - dataset-specific adapters at the edges, identical retrieval & evaluation code for both MIND and EB-NeRD.
2. **Reproducibility** - pinned deps, seeds, deterministic splits.
3. **Anti-leakage by design** - temporal splits with assertion tests.
4. **`rank_bm25` is dev-only** - used solely as a reference implementation in tests; the graded BM25 is hand-built from scratch.

## Datasets

#review 

| Dataset | Language | Users | Articles | Format |
|---------|----------|-------|----------|--------|
| MIND (small) | English | ~50K | ~65K | TSV |
| EB-NeRD (demo) | Danish | ~10K | ~30K | Parquet |
| EB-NeRD (small) | Danish | ~2.7M | ~120K+ | Parquet |

## Evaluation Metrics

- **Ranking:** AUC, MRR, nDCG@5, nDCG@10
- **Beyond accuracy:** Diversity, Novelty, Coverage
- **Slicing:** Cold/warm users, head/tail articles
- **Statistical:** Bootstrap confidence intervals

## AI Usage Policy

All AI-assisted steps are logged in [`logs/ai_usage_log.md`](logs/ai_usage_log.md).