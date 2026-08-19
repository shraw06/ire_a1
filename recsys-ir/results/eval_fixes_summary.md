# Evaluation Fixes Summary

This document outlines the fixes applied to the evaluation harness to ensure the reliability of the lexical vs. semantic retrieval comparison.

## 1. Candidate Pool Size Distribution
We diagnosed that the `recall@100` and `recall@200` metrics were saturated because K was often larger than the entire candidate pool size. The candidate pool sizes are as follows:

*   **MIND**: min=2, median=24.0, mean=37.3, p90=92.0, max=299
*   **EB-NeRD**: min=5, median=9.0, mean=11.6, p90=22.0, max=100

For EB-NeRD in particular, a K of 100 or 200 encompasses 100% of the candidates for every single impression (since max=100). Therefore, `recall@100` for both BM25 and embeddings was artificially returning 1.0. 

## 2. Updated Recall@K Metrics
To better discriminate ranking quality, we added `recall@5` and `recall@10` to the evaluation. 
Here is a comparison of `recall@5` and `recall@10` vs the saturated `recall@100` for the `all` slice:

*   **MIND (BM25)**: recall@5 = 0.3217, recall@10 = 0.4859, recall@100 = 0.9793
*   **MIND (Embeddings)**: recall@5 = 0.4336, recall@10 = 0.6019, recall@100 = 0.9873
*   **EB-NeRD (BM25)**: recall@5 = 0.5713, recall@10 = 0.8527, recall@100 = 1.0000
*   **EB-NeRD (Embeddings)**: recall@5 = 0.6023, recall@10 = 0.8619, recall@100 = 1.0000

Embeddings still outperform BM25 across both datasets on the highly discriminatory metrics. For EB-NeRD in particular, its median candidate pool is 9, making K=10 trivially saturated for over half the impressions. `recall@5` (0.6023 vs 0.5713) is below that median and serves as the primary discriminating metric for EB-NeRD.

## 3. Corrected Intra-List Diversity (ILD) for BM25
Previously, ILD was returning 0.0 for BM25 because the evaluation was skipping diversity calculation if the scoring method was not embedding-based. We updated the pipeline to load the embeddings and calculate ILD for the BM25-ranked top-K items as well.

Corrected `all` slice ILD values (with 95% Confidence Intervals):
*   **MIND (BM25)**: 0.9386 (CI: [0.9383, 0.9390]) vs **Embeddings**: 0.8907 (CI: [0.8899, 0.8915])
*   **EB-NeRD (BM25)**: 0.1803 (CI: [0.1792, 0.1814]) vs **Embeddings**: 0.1673 (CI: [0.1661, 0.1685])

The new values are non-zero and directly comparable in magnitude to the embedding-based retriever's output. The CIs confirm that these differences are statistically significant and not noise.

## 4. Bootstrapping CIs on Small Slices
We added an explicit safeguard for degenerate slices (comprising <1% or >99% of the test population). These slices now report `"insufficient_n"` for Confidence Intervals (CI) rather than leaving them silently blank. They are also explicitly marked via a `flagged_small_slice: true` flag.

The following slice was flagged as too small across both BM25 and Embeddings:
*   `cold_fixed` on EB-NeRD (represents ~0.53% of the population, triggering the < 0.01 threshold)

(Note: We also fixed a bug where the `all` baseline was being incorrectly flagged as degenerate because its `frac_population` is exactly 1.0 (100%). It now properly computes a real Bootstrap CI for all metrics).

## Sanity Check Confirmation
After applying all fixes, the performance comparison remained consistent:
*   **MIND AUC**: 0.63 (Embeddings) vs 0.51 (BM25)
*   **EB-NeRD AUC**: 0.51 (Embeddings) vs 0.46 (BM25)

The direction of the conclusion (Embeddings outperforming BM25) was preserved, while the precision of the observations is substantially sharpened.
