.PHONY: data bm25 embed eval submit clean help

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-12s %s\n", $$1, $$2}'

data: ## Download, parse, and split both datasets
	@echo "🔻 [data] not yet implemented - will download MIND + EB-NeRD, parse to unified schema, temporal-split"

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
