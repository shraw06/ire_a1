# AI Usage Log

This is a **living document**.

## Log Format

Each entry should follow this template:

```
### YYYY-MM-DD HH:MM - [Brief title]

- **Tool / Model:** (e.g. Gemini 2.5 Pro, GitHub Copilot, ChatGPT-4o, …)
- **Prompt / Task:** (what you asked or what was auto-suggested)
- **Output Summary:** (what the tool produced)
- **Disposition:** Accepted as-is / Accepted with edits / Rejected
- **Edits Made:** (if accepted with edits, describe what you changed and why)
```

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


