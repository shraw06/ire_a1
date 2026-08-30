# AI Usage Log

This is a **living document**.

---

### 2026-08-13 15:23 - Repository scaffold generation

- **Tool / Model:** Gemini (Antigravity, Claude Opus 4.6 Thinking)
- **Prompt / Task:** Create the full `recsys-ir/` repository skeleton with directory structure, stub Python files, configs, Makefile, gitignore, README, and this AI usage log.
- **Output Summary:** Generated ~40 files: directory tree with `.gitkeep` placeholders, all Python stubs with module docstrings, `pyproject.toml` with pinned deps, YAML configs, `Makefile` with stub targets, `.gitignore`, `README.md`, notebook stubs, test stubs, and scripts.
- **Disposition:** Accepted as-is.
- **Edits Made:** No edits made.

---

### 2026-08-14 19:53 - Dataset Download & Exploratory Data Analysis (EDA)

- **Tool / Model:** Claude 4.6
- **Prompt / Task:** Download MIND (English, TSV) and EB-NeRD (Danish, Parquet) datasets, perform EDA on both (row counts, schemas, CTR, distributions), create EDA notebooks, and summarize key numbers. Include HF token usage for gated datasets.
- **Output Summary:** Created download scripts (`src/ingestion/download_mind.py`, `download_ebnerd.py`) integrating `python-dotenv` for HF tokens and robust retry logic. Generated Jupyter notebooks (`notebooks/00_explore_mind.ipynb`, `notebooks/00_explore_ebnerd.ipynb`) using programmatic EDA generation via python script. Created comprehensive `EDA_SUMMARY.md` tracking sizes, temporal splits, CTR (MIND ~4%, EB-NeRD ~8.5%), missingness, and verified Danish unicode requirements.
- **Disposition:** Accepted with edits
- **Edits Made:** User prompted to use `.env` file to store HF tokens instead of direct export, prompted to update `Makefile` to include specific `make download` targets, and instructed the model to retry/continue download processes that timed out due to unstable connections.

---

### 2026-08-14 22:42 - Unified Schema Parsing & Checksums Integration

- **Tool / Model:** Claude Opus 4.6 (Thinking) / Claude Sonnet 4.6 (Thinking)
- **Prompt / Task:** Get both MIND and EB-NeRD datasets into ONE shared schema. Create unified Pydantic schema models, parsers for both datasets (`parse_mind.py`, `parse_ebnerd.py`) resolving structural differences explicitly (e.g. MIND body is null, EB-NeRD history has real timestamps). Validate schemas with tests. Additionally, implement checksum verification (`checksums.py`) to avoid re-downloading if archives are valid.
- **Output Summary:** Created `src/common/schema.py`, `src/parsing/parse_mind.py`, and `src/parsing/parse_ebnerd.py` to parse raw data into unified interim parquet files (`articles`, `behaviors`, `users`). Added `configs/checksums.yaml` and `src/ingestion/checksums.py` to handle SHA-256 validation. Updated `download_mind.py` and `download_ebnerd.py` to use `verify_or_populate`. Created `test_schema.py` to validate parity and known differences.
- **Disposition:** Accepted with edits
- **Edits Made:** Instructed model to fix CSV quoting issues (`quote_char=None`) in MIND parser when encountering unescaped quotes. Fixed import errors by adding a root `conftest.py` for testing and modifying `download_*.py` scripts to use relative imports with a `sys.path` fallback for direct execution. Fixed `pyproject.toml` build backend to allow editable installs (`pip install -e .`).

---

### 2026-08-15 16:02 - Temporal Splitting Implementation

- **Tool / Model:** Antigravity (Claude / Gemini)
- **Prompt / Task:** Implement temporal train/val/test splitting for both MIND and EB-NeRD datasets. Use native held-out file as AS-IS TEST set, and carve out last 1 day of native TRAIN as our VAL set. Persist split directly to behaviors DataFrame. Ensure no data leakage (no future impressions in training data).
- **Output Summary:** Created `src/splitting/temporal_split.py` to assign temporal splits, `tests/test_split_no_leakage.py` for comprehensive leakage testing, and updated `src/common/schema.py` to include the `split` column.
- **Disposition:** Accepted with edits
- **Edits Made:** Prompted to update `Makefile` targets to include `parse` and `split` operations and run `make split` to process the datasets.

---

### 2026-08-15 16:25 - Feature Store Implementation

- **Tool / Model:** Antigravity (Gemini / Claude)
- **Prompt / Task:** Build a queryable feature store over the unified schema, backed by Parquet files queried through DuckDB. Implement `store_backend.py`, `article_store.py` (with `cleaned_text` for BM25), and `user_store.py` (with strict `as_of_ts` filtering for EB-NeRD and pass-through for MIND to prevent data leakage). Include a build CLI and idempotent Makefile targets. Test timestamp filtering and end-to-end pipeline execution, and update design documentation.
- **Output Summary:** Created `src/feature_store/store_backend.py`, `article_store.py`, `user_store.py`, and `build_features.py`. Updated `Makefile` with idempotent targets (download, parse, split, features). Added `tests/test_user_history_filtering.py` and `tests/test_pipeline_e2e_smoke.py`. Updated `design.md` with sections 7-11 explaining the feature store backend choices, text scope, embedding strategy, filtering logic, and Makefile idempotency.
- **Edits Made:** Fixed a bug in `test_pipeline_e2e_smoke.py` where synthetic data generation did not initially produce behaviors in the validation split by adjusting the synthetic timestamps to explicitly cross train, val, and test boundaries. Simplified the Makefile's idempotency check for the split target by using a marker file (`.split_done`) to prevent multi-line bash escaping issues. Updated `README.md` to reflect pipeline timings.

---

### 2026-08-17 14:03 - Hand-Built BM25 Retrieval Pipeline

- **Tool / Model:** Gemini 3.1 Pro (High) / Claude Opus 4.6 (Thinking)
- **Prompt / Task:** Build a hand-crafted inverted index and BM25 scoring engine over article TITLE+ABSTRACT for both MIND (English) and EB-NeRD (Danish) datasets. Validate against `rank_bm25` reference, run 4-way ablation (±stopwords × ±stemming), restrict scoring to per-impression candidates, and evaluate recall@K (K=50, 100, 200).
- **Output Summary:** Created `src/retrieval/bm25.py` with custom `InvertedIndex` and `bm25_score_query` optimized for candidate-restricted scoring (O(candidates × query_terms)). Created pipeline runner `src/retrieval/run_bm25.py` performing 4-way ablations and deduplicating query tokens. Added tests `test_bm25_matches_reference.py` and `test_bm25_recall_sane.py`. Results showed stopwords improve MIND recall by ~1.9pp, while stemming is neutral. EB-NeRD maxes out due to small candidate lists. Scored candidates persisted to Parquet.
- **Disposition:** Accepted as-is
- **Edits Made:** No edits made.

---

### 2026-08-18 10:15 - Semantic Retrieval & Lexical vs Semantic Comparison

- **Tool / Model:** Gemini 3.1 Pro (High)
- **Prompt / Task:** Implement an embedding-based retrieval pipeline for MIND (MiniLM) and EB-NeRD (Word2Vec) datasets using an efficient indexing system (FAISS or NumPy fallback). Evaluate recall@K on the validation split and conduct a comparative analysis against the lexical BM25 baseline, slicing performance by user warmth and article popularity.
- **Output Summary:** Created `src/retrieval/run_embeddings.py` to load semantic representations and perform vector similarity search to generate ranked candidates. Created `src/evaluation/compare_retrievers.py` to merge BM25 and embedding results, classify impressions into cold/warm and tail/head slices, and output the results to `results/lexical_vs_semantic.csv` along with a comparison plot.
- **Disposition:** Accepted as-is
- **Edits Made:** No edits made.

---

### 2026-08-19 16:40 - Evaluation Harness Metrics & Reliability Fixes

- **Tool / Model:** Gemini 3.1 Pro (High)
- **Prompt / Task:** Fix evaluation harness output reliability before finalizing the design note. Specifically, diagnose saturated recall@100 metrics, calculate candidate pool sizes, add recall@5/10, correct BM25 ILD calculation to use embeddings, fix Bootstrap CIs being blank for small/degenerate slices, and explicitly document the ILD comparability caveat in the design note.
- **Output Summary:** Created `src/evaluation/candidate_pool_sizes.py` verifying candidate pools were smaller than K=100. Updated `compare_retrievers.py` to calculate and plot highly discriminative `recall@5` and `recall@10` metrics. Modified `run_eval.py` to properly pass embeddings to BM25 evaluation for valid ILD metrics and fixed the degenerate slice CI logic so that slices <1% or >99% return `"insufficient_n"`, while specifically protecting the `"all"` baseline from being incorrectly flagged. Appended Section 13 to `design.md` detailing why absolute ILD magnitudes aren't comparable across different embedding spaces.
- **Disposition:** Accepted with edits
- **Edits Made:** Required a follow-up prompt to fix a bug where the `"all"` slice was being flagged as degenerate because it represented 1.0 (100%) of the population, leading to its CIs being incorrectly skipped. Also explicitly instructed to report `recall@5` alongside `recall@10` due to EB-NeRD's extremely small median candidate pool.

---

### Optimizing the code to adjust for large dataset and submission requirements
- **Tool / Model:** ChatGPT, Claude Sonnet
- **Chat link:** 
https://chatgpt.com/share/6a945343-bb74-83ee-aaa5-ed206f252eda
https://claude.ai/share/2ec0b000-f0b8-49ab-a073-7780700cc376


### 2026-08-29 20:50 - Scaling MIND Recommendation Pipelines & QK-Attention

- **Tool / Model:** Gemini 3.1 Pro (High)
- **Prompt / Task:** Maximize MIND recommendation performance by transitioning from zero-shot embedding retrieval to a supervised attention-based ranking model. Validate model performance, generate full-test-set submissions, and then pivot to candidate-conditioned query-key (QK) attention to bypass temporal overfitting. Finalize the technical design note.
- **Output Summary:** Created high-performance supervised trainer (`train_attention_fast.py`) using pre-extracted memory-mapped numpy arrays to eliminate CPU bottlenecks. Identified temporal overfitting in the trained attention head (val AUC 0.8108, but test AUC 0.5217). Transitioned to a zero-parameter QK-attention approach (`tune_query_key_attention.py`, `generate_qk_submission.py`) which achieved the best test AUC of 0.5235. Updated `design.md` with final leaderboard results, ceiling analysis, and temporal overfitting diagnosis.
- **Disposition:** Accepted as-is
- **Edits Made:** No edits made.

---

Spec MDs of some chats are saved inside the same current directory (/logs).