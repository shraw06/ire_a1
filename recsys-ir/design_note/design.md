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
