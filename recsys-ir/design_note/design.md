# Design Choices: Unified Schema (MIND & EB-NeRD)

This document outlines the core design decisions made when unifying the MIND (TSV) and EB-NeRD (Parquet) datasets into a single, shared schema (`src/common/schema.py`). The goal is to ensure all downstream code (feature engineering, retrieval, evaluation) is entirely dataset-agnostic.

## 1. Article Structure & Body Text

**The Challenge:** EB-NeRD includes full article body text, while MIND explicitly withholds it due to MSN licensing restrictions.
**The Decision:**
- The unified schema includes `body` and `body_source` fields for all articles.
- **EB-NeRD:** Populates `body` natively and sets `body_source = "native"`.
- **MIND:** Sets `body = None` and `body_source = None` by default.
- We deliberately avoid silently dropping the `body` field from EB-NeRD, preserving the possibility of evaluating full-text retrieval on that dataset.
- We avoid aggressive, default scraping for MIND to respect the course scope ("Lexical - titles/abstracts" per instructor clarification) and avoid unhandled failures. Scraping is gated behind a config flag (`include_mind_body`).

## 2. Abstract Extraction

**The Challenge:** EB-NeRD does not have an explicit `abstract` column; it uses `subtitle`.
**The Decision:**
- The `subtitle` column in EB-NeRD is mapped directly to the `abstract` field in the unified schema. EDA showed this has >93% coverage, aligning well with MIND's abstract usage.

## 3. Entity Representation

**The Challenge:** MIND entities are complex JSON strings with Wikidata IDs and confidences, split across title and abstract. EB-NeRD provides simple lists of strings (`ner_clusters`, `entity_groups`).
**The Decision:**
- We standardized on a JSON string representing a `List[dict]` with fields: `label`, `type`, `wikidata_id`, and `confidence`.
- **MIND:** Directly parses the JSON to populate these fields.
- **EB-NeRD:** Synthesizes this structure by pairing `ner_clusters` (as `label`) and `entity_groups` (as `type`), setting `wikidata_id` and `confidence` to `None`.

## 4. Historical Interactions (The Clicked History)

**The Challenge:** This is the most significant structural difference.
- **MIND:** `history` is a space-separated string of Article IDs representing the user's state *exactly at the time of the impression*. It contains no timestamps.
- **EB-NeRD:** History is stored in a separate `history.parquet` file, representing the user's *entire lifetime history* up to the dataset cutoff, complete with exact `impression_time_fixed` timestamps.
**The Decision:**
- We created a unified `clicked_history` field as a JSON list of dictionaries: `{"article_id": "str", "clicked_at": "datetime|null"}`.
- **MIND:** Parsed into this list with `clicked_at = None`. We document the *assumption* that this is a pre-trimmed snapshot (as we cannot filter it later without timestamps).
- **EB-NeRD:** We explicitly join `history.parquet` onto `behaviors.parquet` during ingestion to attach the full history and precise timestamps to every behavior row.
- **Downstream Consequence:** The feature store pipeline *must* implement timestamp-aware filtering for EB-NeRD to prevent data leakage (i.e., filtering out history items that occurred *after* the current impression timestamp), while passing MIND's history through as-is.

## 5. Persistence Format

**The Challenge:** Efficient storage and fast reads for downstream pipelines.
**The Decision:**
- All interim datasets are written as Parquet files using `polars`. This enforces strict types and schema consistency far better than CSV/TSV, speeding up subsequent data loading during training and evaluation.

## 6. Null Values and Missing Data

**The Challenge:** How to handle missing data consistently.
**The Decision:**
- We do not impute missing text fields (e.g., MIND's ~5% missing abstracts) during ingestion. The schema allows `None` (null) for abstracts. Downstream models (like BM25 or embedding models) are responsible for handling empty strings or nulls according to their specific requirements.

## 7. Feature Store Backend: Parquet + DuckDB

**The Challenge:** Choosing a storage and query backend for the feature store that is efficient, serverless, and suitable for a research assignment.

**Alternatives considered:**
- **Pickled pandas DataFrames:** Forces the entire file into memory on every load; loses columnar read efficiency.
- **SQLite:** Row-oriented by design; poor performance for wide text and embedding columns; adds write-lock contention with no benefit.
- **Full feature-store framework (Feast):** Correct shape for production, but requires significant setup and learning cost relative to the assignment deadline. Graders are assessing pipeline design, not infrastructure maturity.

**The Decision:**
- Use **Parquet + DuckDB** as the feature store backend.
- DuckDB registers each Parquet file as an in-memory SQL view (`CREATE VIEW ... AS SELECT * FROM read_parquet(...)`), enabling selective columnar reads (e.g., only `title`/`abstract` for BM25, only `embedding_ref` for ANN) without materializing the full file.
- No server process required; one pure-Python-installable dependency (`pip install duckdb`).
- Scales to EB-NeRD's larger row counts far better than in-memory-pandas, and allows SQL-style time-filtered joins with no ingestion step.

## 8. Cleaned Text Feature Scope

**The Challenge:** Defining what "article text" means for the retrieval pipeline, given that body availability differs across datasets.

**The Decision:**
- The `cleaned_text` feature (used for BM25 indexing) is defined as **`title + " " + abstract`** only — the confirmed required scope.
- Body text is preserved on the underlying interim article record (available for optional ablation on EB-NeRD where it is natively available), but is **not** included in `cleaned_text` by default.
- When `abstract` is null (~5% of MIND articles), `cleaned_text` falls back to `title` only; no imputation is performed.
- This keeps the BM25 feature construction identical across datasets and avoids a meaningless asymmetry (MIND has no body; EB-NeRD does).

## 9. Embedding Reference as a Lazy Pointer

**The Challenge:** Embedding vectors are large (typically 768 or 1024 floats per article). Materializing them twice - once in the article record and once in the embedding index - wastes memory and disk.

**The Decision:**
- The article feature store exposes an `embedding_ref` column (currently `null` at this stage; will point to a row index in a separate embedding store once embeddings are computed).
- The feature store **never** materializes the embedding vector itself, it only stores the pointer.
- The ANN retrieval module will be responsible for resolving the pointer into the actual vector from the embedding Parquet file.
- This matches the demo bundle reality: the EB-NeRD demo zip contains no pre-computed embeddings, so `embedding_ref = null` is correct and expected.

## 10. As-of-Timestamp Filtering Contract

**The Challenge:** Preventing data leakage when serving user history at inference time, given that the two datasets encode history fundamentally differently.

**The Decision:**
- The function signature `get_user_history(user_id, as_of_ts, dataset)` is identical for both datasets. The implementation diverges internally:
  - **EB-NeRD:** Strictly excludes any history entry with `clicked_at >= as_of_ts`. This is the **actual leakage-prevention mechanism** — EB-NeRD stores the user's full lifetime history (from `history.parquet`) with real per-click timestamps, so filtering is necessary.
  - **MIND:** Returns the full snapshot unchanged (pass-through / no-op). MIND's `behaviors.tsv` already contains a pre-trimmed history snapshot as of each impression; there are no per-item timestamps to filter on. This is a **documented assumption**, not a verified guarantee - MIND provides no per-article `published_time` to independently cross-check it. The caveat is flagged in both `parse_mind.py` and `user_store.py`.
- The as-of-timestamp unit test is written primarily against EB-NeRD's structure (where the invariant is load-bearing) and confirmed by a lighter MIND pass-through test.

## 11. Makefile Idempotency

**The Challenge:** A multi-stage pipeline (`download → parse → split → features`) is expensive to re-run from scratch during development. Re-running individual stages must be safe and predictable.

**The Decision:**
- Each Makefile target checks for the presence of its own output files and **skips work if outputs already exist**, unless `FORCE=1` is passed by the user.
- The split stage uses a lightweight marker file (`data/interim/.split_done`) instead of a fragile embedded Python check, which avoids shell-escaping issues in multi-line Make recipes.
- `make clean` removes all intermediate outputs and the marker file, resetting the pipeline to a clean state.
- Idempotent re-runs complete in ~5ms (all skips); a fresh `make data` on real MIND-small + EB-NeRD demo data takes ~14s.

