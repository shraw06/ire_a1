"""Run evaluation harness to compute all metrics and save results."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.evaluation.ranking_metrics import auc_score, mrr, ndcg_at_k
from src.evaluation.beyond_accuracy import compute_intra_list_diversity, compute_novelty, compute_coverage
from src.evaluation.slicing import get_user_slice, get_article_slice
from src.evaluation.bootstrap import compute_bootstrap_ci
from src.feature_store.user_store import UserFeatureStore

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_popularity(dataset: str, split: str | None = None) -> dict[str, int]:
    """Load article popularity from behaviors."""
    behaviors_path = _PROJECT_ROOT / "data" / "interim" / dataset / "behaviors.parquet"
    df = pl.read_parquet(behaviors_path)
    
    if split:
        df = df.filter(pl.col("split") == split)
        
    pop = {}
    for row in df.iter_rows(named=True):
        cands = json.loads(row["candidates"])
        for cid in cands:
            pop[cid] = pop.get(cid, 0) + 1
    return pop

def load_user_history_lens(dataset: str) -> dict[str, int]:
    user_store = UserFeatureStore(dataset)
    rows = user_store._store.query_sql(
        f"SELECT user_id, history_len FROM {user_store._store.alias}"
    )
    return {r["user_id"]: r["history_len"] for r in rows}

def load_embeddings(dataset: str, model: str) -> dict[str, np.ndarray]:
    """Load embeddings and map to IDs."""
    if dataset == "mind":
        emb_path = _PROJECT_ROOT / "data" / "processed" / "embeddings" / "mind_minilm.npy"
        ids_path = _PROJECT_ROOT / "data" / "processed" / "embeddings" / "mind_minilm_ids.json"
    else:
        if model == "w2v":
            emb_path = _PROJECT_ROOT / "data" / "processed" / "embeddings" / "ebnerd_Ekstra_Bladet_word2vec.npy"
            ids_path = _PROJECT_ROOT / "data" / "processed" / "embeddings" / "ebnerd_Ekstra_Bladet_word2vec_ids.json"
        else:
            return {}
            
    if not emb_path.exists() or not ids_path.exists():
        return {}
        
    embs = np.load(emb_path)
    with open(ids_path, "r") as f:
        ids = json.load(f)
        
    return {str(iid): embs[i] for i, iid in enumerate(ids)}

def evaluate_retriever(
    dataset: str, 
    retriever: str, 
    scores_df: pl.DataFrame,
    train_pop: dict[str, int],
    full_pop: dict[str, int],
    user_hist: dict[str, int],
    embeddings: dict[str, np.ndarray]
) -> tuple[list[dict], list[dict]]:
    """Evaluate and return per-impression records for leaked and unleaked."""
    total_train_items = sum(train_pop.values())
    total_full_items = sum(full_pop.values())
    
    records_leaked = []
    records_unleaked = []
    
    n_rows = len(scores_df)
    for i, row in enumerate(scores_df.iter_rows(named=True)):
        if i % 5000 == 0:
            logger.info(f"    Evaluated {i}/{n_rows} impressions...")
            
        imp_id = row["impression_id"]
        user_id = row["user_id"]
        
        candidates = json.loads(row["candidates"])
        labels = json.loads(row["labels"])
        ranked_ids = json.loads(row["ranked_ids"])
        scores = json.loads(row["scores"])
        
        if not candidates:
            continue
            
        cand_to_label = dict(zip(candidates, labels))
        ranked_labels = [cand_to_label.get(rid, 0) for rid in ranked_ids]
        
        auc = auc_score(ranked_labels, scores)
        mrr_val = mrr(ranked_labels, scores)
        ndcg5 = ndcg_at_k(ranked_labels, scores, 5)
        ndcg10 = ndcg_at_k(ranked_labels, scores, 10)
        
        top_k = ranked_ids[:10]
        
        nov_train = compute_novelty(top_k, train_pop, total_train_items)
        nov_full = compute_novelty(top_k, full_pop, total_full_items)
        
        ild = 0.0
        if embeddings:
            top_embs = [embeddings.get(rid) for rid in top_k if rid in embeddings]
            if len(top_embs) > 1:
                ild = compute_intra_list_diversity(np.vstack(top_embs))
                
        history_len = user_hist.get(user_id, 0)
        gt_ids = [cid for cid, lbl in cand_to_label.items() if lbl == 1]
        avg_pop_train = np.mean([train_pop.get(aid, 0) for aid in gt_ids]) if gt_ids else 0
        
        slice_cold_fixed = get_user_slice(history_len, dataset, "fixed")
        slice_cold_data = get_user_slice(history_len, dataset, "data-driven")
        slice_tail_fixed = get_article_slice(avg_pop_train, dataset, "fixed")
        slice_tail_data = get_article_slice(avg_pop_train, dataset, "data-driven")
        
        base_record = {
            "dataset": dataset,
            "retriever": retriever,
            "AUC": auc,
            "MRR": mrr_val,
            "nDCG@5": ndcg5,
            "nDCG@10": ndcg10,
            "ILD": ild,
            "slice_cold_fixed": slice_cold_fixed,
            "slice_cold_data": slice_cold_data,
            "slice_tail_fixed": slice_tail_fixed,
            "slice_tail_data": slice_tail_data,
            "top_10": top_k
        }
        
        rec_unleak = dict(base_record)
        rec_unleak["Novelty"] = nov_train
        
        rec_leak = dict(base_record)
        rec_leak["Novelty"] = nov_full
        
        records_unleaked.append(rec_unleak)
        records_leaked.append(rec_leak)
        
    return records_unleaked, records_leaked


def aggregate_and_bootstrap_proper(records: list[dict], catalog_sizes: dict[str, int]) -> pl.DataFrame:
    if not records:
        return pl.DataFrame()
        
    summary_rows = []
    
    slices = [
        ("all", "all"),
        ("cold_fixed", "cold"),
        ("cold_data", "cold"),
        ("warm_fixed", "warm"),
        ("warm_data", "warm"),
        ("tail_fixed", "tail"),
        ("tail_data", "tail"),
        ("head_fixed", "head"),
        ("head_data", "head"),
    ]
    
    metrics = ["AUC", "MRR", "nDCG@5", "nDCG@10", "ILD", "Novelty"]
    
    # Group records manually
    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        key = (r["dataset"], r["retriever"])
        groups[key].append(r)
        
    for (dataset, retriever), group_recs in groups.items():
        c_size = catalog_sizes.get(dataset, 1)
        total_n = len(group_recs)
        
        for slice_desc, target_val in slices:
            sub_recs = []
            if slice_desc == "all":
                sub_recs = group_recs
            elif slice_desc == "cold_fixed":
                sub_recs = [r for r in group_recs if r["slice_cold_fixed"] == "cold"]
            elif slice_desc == "cold_data":
                sub_recs = [r for r in group_recs if r["slice_cold_data"] == "cold"]
            elif slice_desc == "warm_fixed":
                sub_recs = [r for r in group_recs if r["slice_cold_fixed"] == "warm"]
            elif slice_desc == "warm_data":
                sub_recs = [r for r in group_recs if r["slice_cold_data"] == "warm"]
            elif slice_desc == "tail_fixed":
                sub_recs = [r for r in group_recs if r["slice_tail_fixed"] == "tail"]
            elif slice_desc == "tail_data":
                sub_recs = [r for r in group_recs if r["slice_tail_data"] == "tail"]
            elif slice_desc == "head_fixed":
                sub_recs = [r for r in group_recs if r["slice_tail_fixed"] == "head"]
            elif slice_desc == "head_data":
                sub_recs = [r for r in group_recs if r["slice_tail_data"] == "head"]
                
            n = len(sub_recs)
            if n == 0:
                continue
                
            all_rec = set()
            for r in sub_recs:
                all_rec.update(r["top_10"])
            coverage = compute_coverage(all_rec, c_size)
            
            frac = n / total_n
            degenerate = ((frac < 0.01) or (frac > 0.99)) and slice_desc != "all"
            
            row_dict = {
                "dataset": dataset,
                "retriever": retriever,
                "slice": slice_desc,
                "n_impressions": n,
                "frac_population": frac,
                "Coverage": coverage,
            }
            
            row_dict["flagged_small_slice"] = degenerate
            for m in metrics:
                vals = np.array([r[m] for r in sub_recs])
                if degenerate or len(vals) < 2:
                    mean_val = float(np.mean(vals)) if len(vals) > 0 else 0.0
                    ci_low, ci_high = "insufficient_n", "insufficient_n"
                else:
                    mean_val, ci_low, ci_high = compute_bootstrap_ci(vals, b=1000)
                    
                row_dict[m] = mean_val
                row_dict[f"{m}_CI_low"] = ci_low
                row_dict[f"{m}_CI_high"] = ci_high
                
            summary_rows.append(row_dict)
            
    return pl.DataFrame(summary_rows)


def main():
    datasets = ["mind", "ebnerd"]
    
    all_unleaked = []
    all_leaked = []
    
    catalog_sizes = {}
    
    for dataset in datasets:
        logger.info(f"Processing {dataset}")
        
        train_pop = load_popularity(dataset, "train")
        full_pop = load_popularity(dataset, None)
        user_hist = load_user_history_lens(dataset)
        
        catalog_sizes[dataset] = len(full_pop)
        
        # Load embeddings once for both retrievers so BM25 can compute ILD
        model = "minilm" if dataset == "mind" else "w2v"
        embs = load_embeddings(dataset, model)
        
        # BM25
        bm25_path = _PROJECT_ROOT / "data" / "processed" / f"bm25_scores_{dataset}_title_abstract.parquet"
        if bm25_path.exists():
            df = pl.read_parquet(bm25_path)
            u, l = evaluate_retriever(dataset, "bm25", df, train_pop, full_pop, user_hist, embs)
            all_unleaked.extend(u)
            all_leaked.extend(l)
            
        # Embeddings
        embed_path = _PROJECT_ROOT / "data" / "processed" / f"embed_scores_{dataset}_{model}.parquet"
        if embed_path.exists():
            df = pl.read_parquet(embed_path)
            u, l = evaluate_retriever(dataset, f"embed_{model}", df, train_pop, full_pop, user_hist, embs)
            all_unleaked.extend(u)
            all_leaked.extend(l)
            
    out_dir = _PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if all_unleaked:
        df_unleaked_summary = aggregate_and_bootstrap_proper(all_unleaked, catalog_sizes)
        df_unleaked_summary.write_csv(out_dir / "eval_summary_stripped.csv")
        
    if all_leaked:
        df_leaked_summary = aggregate_and_bootstrap_proper(all_leaked, catalog_sizes)
        df_leaked_summary.write_csv(out_dir / "eval_summary.csv")
        
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

