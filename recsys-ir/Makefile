PYTHON := .venv/bin/python
FORCE ?= 0
DATA_SCALE ?= small

ifeq ($(DATA_SCALE),large)
INTERIM_MARKER := data/interim/large/.split_done
else
INTERIM_MARKER := data/interim/.split_done
endif

.PHONY: data download download-mind download-ebnerd download-large download-large-mind download-large-ebnerd \
        parse parse-mind parse-ebnerd split features bm25 embed compare eval \
        submit submit-mind submit-ebnerd validate-mind validate-ebnerd test clean help

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-24s %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------

download: ## Download development datasets (small/demo)
	$(MAKE) download-mind download-ebnerd

download-mind: ## Download & unzip MINDsmall (train + dev)
	@if [ "$(FORCE)" = "0" ] && [ -d data/raw/mind/MINDsmall_train ] && [ -d data/raw/mind/MINDsmall_dev ]; then \
		echo "  ✓ MIND raw data already present — skipping"; \
	else $(PYTHON) -m src.ingestion.download_mind; fi

download-ebnerd: ## Download & unzip EB-NeRD demo bundle
	@if [ "$(FORCE)" = "0" ] && [ -d data/raw/ebnerd/ebnerd_demo ]; then \
		echo "  ✓ EB-NeRD demo already present — skipping"; \
	else $(PYTHON) -m src.ingestion.download_ebnerd; fi

download-large: ## Download large MIND + EB-NeRD train/dev/test artifacts required for final evaluation/submission
	$(PYTHON) -m src.ingestion.download_large --dataset all

download-large-mind: ## Download MIND large train/dev/test
	$(PYTHON) -m src.ingestion.download_large --dataset mind

download-large-ebnerd: ## Download EB-NeRD large train/validation + test + embeddings
	$(PYTHON) -m src.ingestion.download_large --dataset ebnerd

# ---------------------------------------------------------------------------
# Unified one-command pipeline
# ---------------------------------------------------------------------------

data: download-if-needed parse split features ## One-command build of the selected data scale
	@echo ""
	@echo "✓ DATA PIPELINE COMPLETE (scale=$(DATA_SCALE))"

.PHONY: download-if-needed
download-if-needed:
	@if [ "$(DATA_SCALE)" = "large" ]; then \
		$(MAKE) download-large; \
	else \
		$(MAKE) download; \
	fi

parse: parse-mind parse-ebnerd ## Parse both datasets at DATA_SCALE

parse-mind: ## Parse MIND train/dev at DATA_SCALE into scale-specific interim Parquet
	@if [ "$(FORCE)" = "0" ]; then \
		if [ "$(DATA_SCALE)" = "large" ] && [ -f data/interim/large/mind/.parse_complete ]; then \
			echo "  ✓ MIND large interim already complete — skipping"; \
		elif [ "$(DATA_SCALE)" = "small" ] && [ -f data/interim/mind/articles.parquet ] && [ -f data/interim/mind/behaviors.parquet ]; then \
			echo "  ✓ MIND small interim already present — skipping"; \
		else $(PYTHON) -m src.parsing.parse_mind --scale $(DATA_SCALE); fi; \
	else $(PYTHON) -m src.parsing.parse_mind --scale $(DATA_SCALE); fi

parse-ebnerd: ## Parse EB-NeRD train/validation at DATA_SCALE into scale-specific interim Parquet
	@if [ "$(FORCE)" = "0" ]; then \
		if [ "$(DATA_SCALE)" = "large" ] && [ -f data/interim/large/ebnerd/.parse_complete ]; then \
			echo "  ✓ EB-NeRD large interim already complete — skipping"; \
		elif [ "$(DATA_SCALE)" = "small" ] && [ -f data/interim/ebnerd/articles.parquet ] && [ -f data/interim/ebnerd/behaviors.parquet ]; then \
			echo "  ✓ EB-NeRD small interim already present — skipping"; \
		else $(PYTHON) -m src.parsing.parse_ebnerd --scale $(DATA_SCALE); fi; \
	else $(PYTHON) -m src.parsing.parse_ebnerd --scale $(DATA_SCALE); fi

split: ## Assign temporal train/val/test splits at DATA_SCALE
	@if [ "$(FORCE)" = "0" ] && [ -f "$(INTERIM_MARKER)" ]; then \
		echo "  ✓ Split already present for $(DATA_SCALE) — skipping"; \
	else mkdir -p "$$(dirname $(INTERIM_MARKER))"; $(PYTHON) -m src.splitting.temporal_split --dataset all --scale $(DATA_SCALE) && touch "$(INTERIM_MARKER)"; fi

features: ## Build article + user feature stores at DATA_SCALE
	$(PYTHON) -m src.feature_store.build_features --dataset all --scale $(DATA_SCALE)

# ---------------------------------------------------------------------------
# Retrieval + offline evaluation
# ---------------------------------------------------------------------------

bm25: ## Run BM25 retrieval on the labeled val split at DATA_SCALE
	$(PYTHON) -m src.retrieval.run_bm25 --dataset all --scale $(DATA_SCALE)

embed: ## Run semantic retrieval on the labeled val split at DATA_SCALE
	$(PYTHON) -m src.retrieval.run_embeddings --dataset all --scale $(DATA_SCALE)

compare: ## Compare lexical vs semantic retrieval at DATA_SCALE
	$(PYTHON) -m src.evaluation.compare_retrievers --scale $(DATA_SCALE)

eval: ## Run full offline evaluation harness at DATA_SCALE
	$(PYTHON) -m src.evaluation.run_eval --scale $(DATA_SCALE)

# ---------------------------------------------------------------------------
# Codabench submission: unlabeled large test data only
# ---------------------------------------------------------------------------

submit: submit-mind submit-ebnerd ## Generate both Codabench submissions from large test sets

submit-mind: ## Generate MIND prediction.txt + ZIP from MINDlarge_test
	$(PYTHON) -m src.submission.make_submission --dataset mind --batch-size 50000 --history-cap 20

submit-ebnerd: ## Generate EB-NeRD prediction.txt + ZIP from ebnerd_testset
	$(PYTHON) -m src.submission.make_submission --dataset ebnerd --batch-size 100000 --history-cap 20 --ebnerd-model w2v

validate-mind: ## Validate generated MIND prediction file and ZIP
	$(PYTHON) -m src.submission.writers submissions/mind/prediction.txt
	$(PYTHON) -m src.submission.package_submission submissions/mind/prediction.txt /tmp/mind_submission_check.zip
	@unzip -l /tmp/mind_submission_check.zip

validate-ebnerd: ## Validate generated EB-NeRD prediction file and ZIP
	$(PYTHON) -m src.submission.writers submissions/ebnerd/prediction.txt
	$(PYTHON) -m src.submission.package_submission submissions/ebnerd/prediction.txt /tmp/ebnerd_submission_check.zip
	@unzip -l /tmp/ebnerd_submission_check.zip

# ---------------------------------------------------------------------------
# Tests / cleanup
# ---------------------------------------------------------------------------

test: ## Run unit + anti-leakage tests
	$(PYTHON) -m pytest tests/ src/evaluation/leakage_test.py -v

clean: ## Remove generated processed/interim outputs and submission artifacts
	rm -rf data/interim/mind/ data/interim/ebnerd/ data/interim/large/ \
		data/processed/mind/ data/processed/ebnerd/ data/processed/large/ \
		data/processed/submission/ submissions/ __pycache__ src/**/__pycache__ tests/__pycache__ \
		results/large/
	@echo "🧹 Cleaned generated pipeline outputs"
