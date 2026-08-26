# Design Note: Unified News Recommendation Pipeline

## 1. Overview
This project implements a unified news recommendation retrieval pipeline for the MIND (English) and EB-NeRD (Danish) datasets. The original goal was to build a dataset-agnostic shared schema and evaluation harness capable of supporting both lexical and semantic retrieval. The system evolved from an initial Parquet/DuckDB-based architecture validated at small scales into a memory-mapped, dataset-native streaming architecture capable of handling the tens of millions of rows required by the `DATA_SCALE=large` benchmark.

## 2. Initial Architecture
The initial design prioritized logical unification and reproducibility via a shared schema and idempotent Makefile orchestration:
- **Unified Article Schema**: Included `body` and `body_source` fields. EB-NeRD populated these natively, while MIND defaulted to `None` to avoid unhandled scraping failures and respect the course scope. EB-NeRD's `subtitle` was explicitly mapped to MIND's `abstract`.
- **Entity Representation**: Standardized as a JSON string (`List[dict]`). MIND entities were parsed directly, whereas EB-NeRD entities were synthesized from `ner_clusters` and `entity_groups`.
- **Unified Clicked History**: Modeled as `{"article_id": "str", "clicked_at": "datetime|null"}` to abstract over MIND's pre-trimmed snapshot lists and EB-NeRD's timestamped lifetime histories.
- **Leakage-Prevention Contract**: A unified `get_user_history(user_id, as_of_ts, dataset)` signature was established.
- **Cleaned Text**: Defined strictly as `title + " " + abstract` across both datasets, preserving missing fields as null rather than imputing them.
- **Feature Store**: Parquet + DuckDB was chosen as the initial storage/query backend to avoid the memory overhead of pickled pandas DataFrames while sidestepping the complexity of full feature-store frameworks.
- **Lazy Embeddings**: The `embedding_ref` column stored a lazy pointer instead of materializing large vectors twice.

## 3. What Broke at Large Scale
While the initial logical abstractions were robust, testing against the `DATA_SCALE=large` EB-NeRD split exposed severe physical scaling limitations:
- **Memory Exhaustion**: The initial `UserFeatureStore` eagerly materialized large behavior tables into Polars/Python dictionaries. Against EB-NeRD's ~24.6M parsed behavior rows, this caused massive memory pressure and OOM crashes.
- **Evaluator Crashes**: The full offline evaluator (`run_eval.py`) crashed the local machine because it loaded entire behavior and score tables simultaneously.
- **Legacy Artifacts**: Stale downstream modules continued assuming the existence of a monolithic `user_features.parquet`, which could no longer be built or safely consumed.

These issues were purely physical engineering/scaling bottlenecks, not failures of the underlying retrieval methodology.

## 4. Final Large-Scale Architecture
To scale safely, the unified logical abstraction was retained, but the physical storage and execution strategy was diverged to use dataset-native paradigms:
- **MIND-large**: The logical clicked_history abstraction was retained, but MIND-large reads the impression-level snapshot directly from the parsed behavior record rather than constructing a separate lifetime user table.
- **EB-NeRD-large**: Replaced the Parquet history joins with a `MemoryMappedHistoryStore`. This uses decoupled numpy arrays (`user_ids.npy`, `offsets.npy`, `article_ids.npy`, `timestamps.npy`) to provide indexed, memory-mapped access to lifetime history without materializing the complete history in RAM, enabling strict `clicked_at < as_of_timestamp` filtering at inference time without RAM explosion.
- **Role of DuckDB**: DuckDB remains a critical tool for upstream large-data processing and temporal splitting, but it is no longer the sole physical implementation of the final user-feature store during inference.

## 5. Retrieval Architecture
The system supports both lexical and semantic candidate generation:
- **Hand-Built BM25**: A pure-Python inverted index and graded BM25 scoring engine were implemented from scratch. To make scoring feasible in pure Python, it relies on candidate-restricted scoring and precomputed term-frequency lookups rather than iterating over full postings lists.
- **Semantic Retrieval**: MIND leverages MiniLM, while EB-NeRD was evaluated with Word2Vec and BERT embeddings.
- **Optimization**: All semantic scoring relies on candidate-restricted ranking and the reuse/caching of underlying article embeddings via the `embedding_ref` pointers.

## 6. Experimental Results
Large-scale retrieval experiments yielded the following verified candidate-generation recall numbers:

**MIND large:**
- BM25 Recall@50 = 0.913691
- MiniLM Recall@50 = 0.929447
- BM25 Recall@10 = 0.556643
- MiniLM Recall@10 = 0.602892

**EB-NeRD large:**
- BM25 Recall@50 = 0.999684
- Word2Vec Recall@50 = 0.999735
- BERT Recall@50 = 0.999579

Recall@100 and Recall@200 reached 1.0 across all evaluated methods; because the candidate pools are relatively small, these larger cutoffs become less discriminative. Consequently, tighter K cutoffs are far more discriminative for this dataset.

### Hybrid Reranker Experiments

Tested a recency-weighted history + train-split-popularity hybrid reranker as an attempt to improve leaderboard rank. Offline tuning on the validation split showed popularity weighting monotonically *decreased* AUC/MRR/nDCG@5/nDCG@10 on both datasets (MIND: AUC 0.6274→0.5776 as alpha went 1.0→0.5; EB-NeRD: AUC 0.5061→0.4373). Best alpha was 1.0 (pure embedding similarity) for both, so no hybrid submission was generated — the existing baseline embedding ranking already outperforms any popularity-blended variant on this candidate-restricted setup.

MIND (431,517 validation impressions, 23,291 train-split candidate popularity entries)
  alpha=1.0  AUC=0.6274  MRR=0.3342  nDCG@5=0.3105  nDCG@10=0.3693  (n=431,517)
  alpha=0.9  AUC=0.6246  MRR=0.3298  nDCG@5=0.3067  nDCG@10=0.3658  (n=431,517)
  alpha=0.8  AUC=0.6201  MRR=0.3229  nDCG@5=0.3003  nDCG@10=0.3597  (n=431,517)
  alpha=0.7  AUC=0.6120  MRR=0.3123  nDCG@5=0.2901  nDCG@10=0.3501  (n=431,517)
  alpha=0.6  AUC=0.5997  MRR=0.2972  nDCG@5=0.2751  nDCG@10=0.3359  (n=431,517)
  alpha=0.5  AUC=0.5776  MRR=0.2763  nDCG@5=0.2546  nDCG@10=0.3166  (n=431,517)
```
EBNERD (1,678,989 validation impressions, 9,653 train-split candidate popularity entries)
  alpha=1.0  AUC=0.5061  MRR=0.3390  nDCG@5=0.3706  nDCG@10=0.4552  (n=1,678,989)
  alpha=0.9  AUC=0.4891  MRR=0.3212  nDCG@5=0.3514  nDCG@10=0.4397  (n=1,678,989)
  alpha=0.8  AUC=0.4739  MRR=0.3046  nDCG@5=0.3338  nDCG@10=0.4254  (n=1,678,989)
  alpha=0.7  AUC=0.4596  MRR=0.2919  nDCG@5=0.3195  nDCG@10=0.4142  (n=1,678,989)
  alpha=0.6  AUC=0.4477  MRR=0.2833  nDCG@5=0.3093  nDCG@10=0.4065  (n=1,678,989)
  alpha=0.5  AUC=0.4373  MRR=0.2765  nDCG@5=0.3012  nDCG@10=0.4006  (n=1,678,989)

Saved: /home/shrawani/Desktop/sem5/Information Retrieval and Extraction/a1_again/ire_a1/recsys-ir/results/large/hybrid_alpha_tuning.csv

Best alpha by AUC per dataset:
  mind: alpha=1.0  AUC=0.6274
  ebnerd: alpha=1.0  AUC=0.5061
```

## 7. Validation vs. Test Separation
The pipeline strictly distinguishes between labeled offline validation and unlabeled test-set inference:
- **EB-NeRD**: The validation split contains ~12.6M behavior rows which map to ~1.68M labeled validation impressions (used for offline retrieval evaluation). The test split contains ~13.5M unlabeled impressions used *only* for Codabench prediction generation.
- **MIND**: Follows the same strict isolation, distinguishing labeled validation from the ~2.37M-impression large test set.
- **Principle**: Test sets are never used for offline recall or ranking evaluation because no ground-truth interaction labels are available.

## 8. Comparison Observations
Based on the verified retrieval numbers:
- **Lexical vs. Semantic**: On MIND, MiniLM semantics improve over BM25 at low/moderate K cutoffs. On EB-NeRD, BM25, Word2Vec, and BERT are effectively tied at high K due to saturation.
- **Slice Behavior**: While cold/warm and head/tail behavior variations exist, they are highly dataset- and retriever-dependent. Furthermore, the tail item slices observed in our comparisons were statistically very small, meaning universally strong conclusions (e.g., "warm users always perform better") should be avoided without broader distributional analysis.

## 9. ILD (Intra-List Diversity)
A critical evaluation caveat is that different embedding spaces naturally exhibit different cosine-similarity geometries. Therefore, absolute ILD values should not be directly compared across datasets when they are computed from different embedding spaces/models. (e.g., MIND MiniLM vs. EB-NeRD Word2Vec/BERT). Within-dataset comparisons using the same embedding representation are more interpretable.

## 10. Anti-Gaming / Leakage Audit
To diagnose the impact of future information leakage, we conducted an intentional diagnostic feature audit at the small/demo scale focused on article popularity (Novelty).
- **Train-Only (Safe)**: Popularity derived strictly from training data, representing a safe serving configuration.
- **Train+Validation (Leaked)**: Popularity derived from training and validation data, intentionally leaking future information.

The experiment qualitatively demonstrated that the leaked popularity signal artificially lowered measured novelty. This serves as a feature-audit diagnostic and confirms that strict temporal isolation is required for accurate serving-time simulation.

## 11. Offline Evaluation Limitations
The intended evaluation harness computes AUC, MRR, nDCG@5/10, ILD, Novelty, Coverage, slicing, and bootstrap CIs. However, at `DATA_SCALE=large`, the full offline evaluator repeatedly caused local memory failures. While a streaming/memory-safe redesign was prototyped, the strict Codabench deadline required prioritizing prediction submission. Therefore, full large-scale offline evaluation was postponed, and large retrieval experiments and Codabench test inferences were completed independently.

## 12. Codabench Submission Pipeline
The final test-time pipeline executes as follows:
`Unlabeled Large Test → Streaming Reader → Dataset-Native User History → Article Embeddings → User Representation → Candidate-Restricted Ranking → Prediction File → ZIP → Codabench`

The selected submission models were:
- **MIND**: MiniLM (Successfully uploaded, achieving approx. rank 58/67).
- **EB-NeRD**: Word2Vec (prediction generation complete; submission package prepared/correction in progress).

## 13. Engineering Lessons / Conclusion
The project evolved from a clean, dataset-agnostic logical architecture validated at small scale into a large-scale retrieval and inference pipeline. Stress-testing the system exposed memory and materialization bottlenecks caused by eager Polars/Python processing and by the assumption of a monolithic user_features.parquet representation. Rather than discarding the original abstractions, the final design retained the shared logical interfaces while changing their physical implementation at scale: MIND uses impression-level history snapshots, EB-NeRD uses a timestamped memory-mapped history index, and large Parquet data is processed in batches. This allowed the system to complete large-scale BM25 and semantic retrieval experiments and to generate Codabench prediction artifacts while preserving the intended temporal and leakage-prevention contracts. The main remaining limitation is the full large-scale multi-metric offline evaluator, which was postponed because of local memory constraints.
