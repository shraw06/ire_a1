PYTHON := .venv/bin/python

# Idempotency: each target checks for its output file(s) and skips
# work if already present, unless FORCE=1 is set.
#   make data          → runs only missing stages
#   FORCE=1 make data  → reruns everything
FORCE ?= 0

.PHONY: data download download-mind download-ebnerd parse parse-mind parse-ebnerd split features bm25 embed eval submit clean help

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-18s %s\n", $$1, $$2}'

# ── Download targets ───────────────────────────────────────────────

download: download-mind download-ebnerd ## Download both datasets (MIND + EB-NeRD)

download-mind: ## Download & unzip MINDsmall (train + dev)
	@if [ "$(FORCE)" = "0" ] && [ -d data/raw/mind/MINDsmall_train ] && [ -d data/raw/mind/MINDsmall_dev ]; then \
		echo "  ✓ MIND raw data already present — skipping download (use FORCE=1 to re-download)"; \
	else \
		$(PYTHON) -m src.ingestion.download_mind; \
	fi

download-ebnerd: ## Download & unzip EB-NeRD demo bundle
	@if [ "$(FORCE)" = "0" ] && [ -d data/raw/ebnerd/train ] && [ -d data/raw/ebnerd/validation ]; then \
		echo "  ✓ EB-NeRD raw data already present — skipping download (use FORCE=1 to re-download)"; \
	else \
		$(PYTHON) -m src.ingestion.download_ebnerd; \
	fi

# Parse targets

parse: parse-mind parse-ebnerd ## Parse both datasets into unified schema

parse-mind: ## Parse MIND TSV → interim Parquet
	@if [ "$(FORCE)" = "0" ] && [ -f data/interim/mind/articles.parquet ] && [ -f data/interim/mind/behaviors.parquet ]; then \
		echo "  ✓ MIND interim data already present — skipping parse (use FORCE=1 to re-parse)"; \
	else \
		$(PYTHON) -m src.parsing.parse_mind; \
	fi

parse-ebnerd: ## Parse EB-NeRD Parquet → interim Parquet
	@if [ "$(FORCE)" = "0" ] && [ -f data/interim/ebnerd/articles.parquet ] && [ -f data/interim/ebnerd/behaviors.parquet ]; then \
		echo "  ✓ EB-NeRD interim data already present — skipping parse (use FORCE=1 to re-parse)"; \
	else \
		$(PYTHON) -m src.parsing.parse_ebnerd; \
	fi

# Split target

split: ## Assign temporal train/val/test splits to both datasets
	@if [ "$(FORCE)" = "0" ] && [ -f data/interim/.split_done ]; then \
		echo "  ✓ Split column already present — skipping (use FORCE=1 to re-split)"; \
	else \
		$(PYTHON) -m src.splitting.temporal_split --dataset all && touch data/interim/.split_done; \
	fi

# Feature store target

features: ## Build article + user feature stores (Parquet)
	@if [ "$(FORCE)" = "0" ] \
		&& [ -f data/processed/mind/article_features.parquet ] \
		&& [ -f data/processed/mind/user_features.parquet ] \
		&& [ -f data/processed/ebnerd/article_features.parquet ] \
		&& [ -f data/processed/ebnerd/user_features.parquet ]; then \
		echo "  ✓ Feature stores already present — skipping (use FORCE=1 to rebuild)"; \
	else \
		$(PYTHON) -m src.feature_store.build_features --dataset all; \
	fi

# Full pipeline

data: download parse split features ## Download, parse, split, and build features

# Downstream pipeline (not yet implemented)

bm25: ## Build inverted index and run BM25 retrieval
	@echo "📖 [bm25] not yet implemented - will build hand-rolled inverted index + BM25 scoring"

embed: ## Compute/load embeddings and run ANN retrieval
	@echo "🧠 [embed] not yet implemented - will load/compute article embeddings + FAISS ANN search"

eval: ## Evaluate all retrieval runs
	@echo "📊 [eval] not yet implemented - will compute AUC, MRR, nDCG@{5,10}, diversity, novelty, coverage"

submit: ## Generate Codabench submission files
	@echo "📦 [submit] not yet implemented - will produce prediction files in Codabench format"

clean: ## Remove processed data and caches
	rm -rf data/interim/mind/ data/interim/ebnerd/
	rm -f data/interim/.split_done
	rm -rf data/processed/mind/ data/processed/ebnerd/
	rm -rf __pycache__ src/__pycache__ src/**/__pycache__ tests/__pycache__
	@echo "🧹 Cleaned interim, processed, and cache directories"
