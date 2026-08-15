PYTHON := .venv/bin/python

.PHONY: data download download-mind download-ebnerd parse parse-mind parse-ebnerd split bm25 embed eval submit clean help

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-18s %s\n", $$1, $$2}'

# Download targets (download only - no parsing / splitting)

download: download-mind download-ebnerd ## Download both datasets (MIND + EB-NeRD)

download-mind: ## Download & unzip MINDsmall (train + dev)
	$(PYTHON) -m src.ingestion.download_mind

download-ebnerd: ## Download & unzip EB-NeRD demo bundle
	$(PYTHON) -m src.ingestion.download_ebnerd

# Parse targets (raw -> interim unified schema)

parse: parse-mind parse-ebnerd ## Parse both datasets into unified schema

parse-mind: ## Parse MIND TSV -> interim Parquet
	$(PYTHON) -m src.parsing.parse_mind

parse-ebnerd: ## Parse EB-NeRD Parquet -> interim Parquet
	$(PYTHON) -m src.parsing.parse_ebnerd

# Split target (adds train/val/test column to interim behaviors)

split: ## Assign temporal train/val/test splits to both datasets
	$(PYTHON) -m src.splitting.temporal_split --dataset all

# Full pipeline targets

data: download parse split ## Download, parse, and split both datasets

bm25: ## Build inverted index and run BM25 retrieval
	@echo "📖 [bm25] not yet implemented - will build hand-rolled inverted index + BM25 scoring"

embed: ## Compute/load embeddings and run ANN retrieval
	@echo "🧠 [embed] not yet implemented - will load/compute article embeddings + FAISS ANN search"

eval: ## Evaluate all retrieval runs
	@echo "📊 [eval] not yet implemented - will compute AUC, MRR, nDCG@{5,10}, diversity, novelty, coverage"

submit: ## Generate Codabench submission files
	@echo "📦 [submit] not yet implemented - will produce prediction files in Codabench format"

clean: ## Remove processed data and caches
	@echo "🧹 [clean] not yet implemented - will remove data/interim/, data/processed/, __pycache__/"
