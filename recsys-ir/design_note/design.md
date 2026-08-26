# Design Note: Unified News Recommendation Pipeline

## 1. Overview
This project implements a unified news recommendation retrieval pipeline for the MIND (English) and EB-NeRD (Danish) datasets. The original goal was to build a dataset-agnostic shared schema and evaluation harness capable of supporting both lexical and semantic retrieval. The system evolved from an initial Parquet/DuckDB-based architecture validated at small scales into a memory-mapped, dataset-native streaming architecture capable of handling the tens of millions of rows required by the `DATA_SCALE=large` benchmark.

## 2. Initial Architecture
- **Unified Article Schema**: Included `body` and `body_source` fields. EB-NeRD populated these natively, while MIND defaulted to `None`. EB-NeRD's `subtitle` was mapped to MIND's `abstract`.
- **Entity Representation**: Standardized as a JSON string (`List[dict]`). MIND entities were parsed directly; EB-NeRD entities were synthesized from `ner_clusters` and `entity_groups`.
- **Unified Clicked History**: `{"article_id": "str", "clicked_at": "datetime|null"}`, abstracting over MIND's pre-trimmed snapshot lists and EB-NeRD's timestamped lifetime histories.
- **Leakage-Prevention Contract**: A unified `get_user_history(user_id, as_of_ts, dataset)` signature.
- **Cleaned Text**: `title + " " + abstract`, preserving missing fields as null rather than imputing.
- **Feature Store**: Parquet + DuckDB, avoiding pickled-DataFrame memory overhead and full feature-store framework complexity.
- **Lazy Embeddings**: The `embedding_ref` column stored a lazy pointer instead of materializing large vectors twice.

## 3. What Broke at Large Scale
- **Memory Exhaustion**: The initial `UserFeatureStore` eagerly materialized behavior tables into Polars/Python dictionaries; against EB-NeRD's ~24.6M parsed behavior rows this caused OOM crashes.
- **Evaluator Crashes**: `run_eval.py` crashed the local machine loading entire behavior and score tables simultaneously.
- **Legacy Artifacts**: Stale downstream modules assumed a monolithic `user_features.parquet` that could no longer be safely built or consumed.

These were physical engineering/scaling bottlenecks, not failures of the underlying retrieval methodology.

## 4. Final Large-Scale Architecture
- **MIND-large**: Reads the impression-level snapshot directly from the parsed behavior record rather than building a separate lifetime user table.
- **EB-NeRD-large**: Replaced Parquet history joins with a `MemoryMappedHistoryStore` — decoupled numpy arrays (`user_ids.npy`, `offsets.npy`, `article_ids.npy`, `timestamps.npy`) giving indexed, memory-mapped access without materializing full history in RAM, enabling strict `clicked_at < as_of_timestamp` filtering at inference time.
- **DuckDB**: Remains critical for upstream large-data processing and temporal splitting, but is no longer the sole physical implementation of the final user-feature store at inference time.

## 5. Retrieval Architecture
- **Hand-Built BM25**: Pure-Python inverted index and graded BM25 scoring, using candidate-restricted scoring and precomputed term-frequency lookups instead of full postings-list iteration.
- **Semantic Retrieval**: MIND uses MiniLM; EB-NeRD was evaluated with Word2Vec and BERT.
- **Optimization**: All semantic scoring relies on candidate-restricted ranking and cached embeddings via `embedding_ref` pointers.

## 6. Experimental Results

**MIND large:** BM25 Recall@50 = 0.9137, MiniLM Recall@50 = 0.9294, BM25 Recall@10 = 0.5566, MiniLM Recall@10 = 0.6029.
**EB-NeRD large:** BM25 Recall@50 = 0.9997, Word2Vec Recall@50 = 0.9997, BERT Recall@50 = 0.9996.

Recall@100/@200 reached 1.0 across all methods — candidate pools are small enough that these cutoffs stop being discriminative, so tighter K is more informative here.

### 6.1 Reranking Experiments

Three reranking strategies were tuned offline on cached validation scores (never on test/leaderboard data) before paying for the hours-long cost of live test-set inference.

**Experiment A — Popularity hybrid.** Recency-weighted history + train-split popularity, blended with embedding via weight alpha. Popularity monotonically *hurt* both datasets (MIND: AUC 0.6274→0.5776 as alpha 1.0→0.5; EB-NeRD: 0.5061→0.4373). Best alpha=1.0 (pure embedding) for both — no benefit from popularity. Full sweep in `results/large/hybrid_alpha_tuning.csv`.

*Note:* the initial per-row, per-alpha evaluation loop (JSON parsing + `sklearn.roc_auc_score` per impression per weight, in pure Python) does not scale to MIND's 431K / EB-NeRD's 1.68M validation impressions — an equivalent naive loop in Experiment B ran 6+ hours before crashing. Fix: parse JSON once per row and reuse across all weights, replace per-row sklearn calls with a vectorized rank-sum AUC, and checkpoint per dataset. This cut the full sweep to single-digit seconds.

**Experiment B — BM25 + embedding linear blend (no popularity).** Since BM25 underperforms embedding in isolation, tested whether it still carries *independent* signal via `beta*embedding + (1-beta)*bm25`, min-max normalized, swept over beta.

| Dataset | Best beta | Best AUC | Pure-embed AUC | Lift |
|---|---|---|---|---|
| MIND | 0.8 | 0.6298 | 0.6277 | +0.33% relative |
| EB-NeRD | 1.0 | 0.5062 | 0.5062 | none |

MIND's lift was non-trivial enough to justify a live test submission (`mind_bm25embed_submission.zip`); EB-NeRD's null result meant skipping its ~15h live-inference run entirely.

| Submission | Score |
|---|---|
| `mind_submission.zip` (baseline, pure embedding) | 0.5212 |
| `mind_hybrid_submission.zip` (Exp. A, popularity) | 0.5196 |
| `mind_bm25embed_submission.zip` (Exp. B, beta=0.8) | 0.5211 |

The validation-set lift did **not** survive to the test set (0.5211 vs 0.5212, within noise) — three separate linear/heuristic combinations of the same signals all landed within noise of each other on the leaderboard, suggesting the ceiling wasn't in the blend weights but in the blend being linear at all.

**Experiment C — Gradient-boosted nonlinear combiner.** A `HistGradientBoostingClassifier` per dataset, trained on candidate-level examples with five features (normalized embed/BM25 scores, their rank-percentiles within the candidate list, and their difference), evaluated via group-wise held-out AUC (20% impression-grouped split).

| Dataset | Combiner AUC | Best linear blend AUC | Result |
|---|---|---|---|
| MIND | 0.6193 | 0.6298 | Worse — not submitted |
| EB-NeRD | 0.5460 | 0.5062 | **+8% relative** — only genuine EB-NeRD lift found |

MIND: added capacity found nothing the linear blend hadn't already found, so no MIND submission was generated for this model. EB-NeRD: unlike every linear combination tried, the nonlinear combiner found real signal — suggesting BM25 corrects embedding only in specific rank-agreement regimes a fixed weight can't express. Generating its test predictions was estimated at ~15h; since an EB-NeRD submission was not required for this deliverable, it was **deliberately not run** (model saved at `results/large/combiner_ebnerd.joblib`, generation script implemented but unexecuted).

**Summary:** `mind_submission.zip` (0.5212) remains the best submitted MIND score. Every reranking variant matched it within noise or underperformed on the actual test set, despite two of three looking better offline — a reminder that validation lift wasn't a reliable predictor of test-set lift here, and the one real improvement found (Exp. C, EB-NeRD) was in the dataset out of scope for submission.

## 7. Validation vs. Test Separation
Labeled validation is strictly separated from unlabeled test inference: EB-NeRD's ~12.6M validation behavior rows map to ~1.68M labeled impressions; MIND's large test set has ~2.37M impressions. Test sets are never used for offline evaluation since no ground-truth labels exist there — the reranking sweeps and the combiner's held-out split all ran exclusively on validation scores, with live test inference reserved for configurations that showed validation-side improvement worth the cost.

## 8. Comparison Observations
On MIND, MiniLM improves over BM25 at low/moderate K. On EB-NeRD, BM25/Word2Vec/BERT are effectively tied at high K due to saturation — consistent with the reranking results, where BM25 added only marginal, non-durable value on MIND and none linearly on EB-NeRD. Cold/warm and head/tail slice effects exist but are dataset- and retriever-dependent, and tail slices were statistically small, so broad claims (e.g. "warm users always perform better") should be avoided.

## 9. ILD (Intra-List Diversity)
Different embedding spaces have different cosine-similarity geometries, so absolute ILD should not be compared across datasets computed from different embedding models (MIND MiniLM vs. EB-NeRD Word2Vec/BERT). Within-dataset comparisons using the same representation are more interpretable.

## 10. Anti-Gaming / Leakage Audit
A diagnostic feature audit (small/demo scale) compared train-only popularity (safe) against train+validation popularity (intentionally leaked), showing the leaked signal artificially lowered measured novelty — confirming strict temporal isolation is required for accurate serving-time simulation. The reranking experiments in §6.1 respected the same contract: all popularity/rank features came strictly from train/validation-scoped scores, never from test labels.

## 11. Offline Evaluation Limitations
The full evaluator (AUC, MRR, nDCG@5/10, ILD, Novelty, Coverage, slicing, bootstrap CIs) repeatedly caused memory failures at `DATA_SCALE=large`. A streaming redesign was prototyped but deprioritized under the Codabench deadline in favor of prediction submission. A narrower version of the same problem hit the reranker tuning scripts (§6.1) and was fixed locally (single-pass parsing, vectorized AUC, checkpointing) without needing the full evaluator redesign, since tuning only needs AUC/MRR/nDCG over cached scores, not the full metric suite.

## 12. Codabench Submission Pipeline
`Unlabeled Large Test → Streaming Reader → Dataset-Native User History → Article Embeddings → User Representation → Candidate-Restricted Ranking → Prediction File → ZIP → Codabench`

- **MIND**: MiniLM pure-embedding ranking submitted (~rank 58/67, score 0.5212). BM25 blend (0.5211) and popularity hybrid (0.5196) were also submitted for comparison but did not improve on baseline; the gradient-boosted combiner was trained but not submitted (held-out AUC below the blend's).
- **EB-NeRD**: Word2Vec pure-embedding ranking (prediction generation complete; submission package prepared/correction in progress). The combiner showed a genuine +8% relative validation AUC improvement but was not submitted — its ~15h test-inference cost wasn't justified given an EB-NeRD submission wasn't required here; model and script are retained for future use.

## 13. Engineering Lessons / Conclusion
The project evolved from a clean, dataset-agnostic logical architecture into a large-scale pipeline, retaining shared logical interfaces (unified history, leakage contract) while diverging physical implementation per dataset: MIND uses impression-level snapshots, EB-NeRD uses a memory-mapped history index, and large Parquet data is processed in batches.

The reranking experiments reinforced the same lesson at the modeling layer: each of three successive attempts to beat pure-embedding ranking was tuned cheaply offline before committing to expensive live inference, and two were correctly rejected before consuming a submission slot or hours of compute. The one approach that did show genuine improvement (the combiner, on EB-NeRD) was deliberately not pursued to submission given it was out of scope and disproportionately costly relative to remaining time — a scope decision, not a failure to find signal. The main remaining limitation is the full large-scale multi-metric offline evaluator, postponed due to local memory constraints.