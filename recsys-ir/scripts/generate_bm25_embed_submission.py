"""BM25+embedding blended test-set submission. No popularity term.
Usage: .venv/bin/python -m scripts.generate_bm25_embed_submission --dataset mind --beta 0.7
"""
from __future__ import annotations
import argparse, logging, time
from pathlib import Path
from src.feature_store.history_store import MemoryMappedHistoryStore
from src.parsing.submission_readers import (
    find_ebnerd_test_files, find_mind_test_behaviors, iter_ebnerd_test, iter_mind_test,
)
from src.retrieval.bm25 import BM25Engine
from src.retrieval.run_bm25 import _build_query_text, _load_article_corpus
from src.retrieval.user_representation import build_mean_user_vector
from src.submission.make_submission import _history_batches, _load_index
from src.submission.package_submission import package_prediction
from src.submission.writers import write_ranked_impression

logger = logging.getLogger(__name__)
_ROOT = Path(__file__).resolve().parents[1]

def _minmax(v):
    lo, hi = min(v), max(v)
    return [0.0]*len(v) if hi - lo < 1e-12 else [(x-lo)/(hi-lo) for x in v]

def _rank_blended(history, candidates, index, bm25_engine, article_texts, beta, history_cap):
    query_text = _build_query_text(history, article_texts, history_cap)
    bm25_results = dict(bm25_engine.rank(query_text, candidate_ids=candidates, top_k=len(candidates)))
    bm25_vals = [bm25_results.get(c, 0.0) for c in candidates]
    user_vec = build_mean_user_vector(history, index, history_cap=history_cap)
    embed_sims = dict(index.search_restricted(user_vec, candidates, k=len(candidates)))
    embed_vals = [embed_sims.get(c, 0.0) for c in candidates]
    blended = [beta*e + (1-beta)*b for e, b in zip(_minmax(embed_vals), _minmax(bm25_vals))]
    order = sorted(range(len(candidates)), key=lambda i: -blended[i])
    return [candidates[i] for i in order]

def generate_submission(dataset, args):
    started = time.time()
    index = _load_index(dataset, args)
    corpus, article_texts = _load_article_corpus(dataset, scale="large")
    bm25_engine = BM25Engine.from_corpus(corpus, dataset=dataset, use_stopwords=True, use_stemming=False)
    logger.info("BM25 index built: %s", bm25_engine)

    out_dir = _ROOT / "submissions" / f"{dataset}_bm25embed"
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = out_dir / "prediction.txt"
    zip_path = out_dir / f"{dataset}_bm25embed_submission.zip"

    eb_history = None
    if dataset == "ebnerd":
        _, history_path = find_ebnerd_test_files(_ROOT / "data" / "raw" / "ebnerd")
        history_dir = _ROOT / "data" / "processed" / "submission" / "ebnerd_history"
        eb_history = MemoryMappedHistoryStore.build(history_path, history_dir, force=args.rebuild_history_index)
        batches = iter_ebnerd_test(find_ebnerd_test_files(_ROOT / "data" / "raw" / "ebnerd")[0], batch_size=args.batch_size)
    else:
        batches = iter_mind_test(find_mind_test_behaviors(_ROOT / "data" / "raw" / "mind"), batch_size=args.batch_size)

    row_count = 0
    with prediction_path.open("w", encoding="utf-8") as handle:
        for batch in batches:
            histories = _history_batches(dataset, batch, eb_history)
            for item, history in zip(batch, histories):
                ordered = _rank_blended(history, item.candidates, index, bm25_engine, article_texts, args.beta, args.history_cap)
                write_ranked_impression(handle, item.impression_id, item.candidates, ordered)
            row_count += len(batch)

    package_prediction(prediction_path, zip_path)
    return prediction_path, zip_path, row_count, time.time() - started

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    p.add_argument("--batch-size", type=int, default=50_000)
    p.add_argument("--history-cap", type=int, default=20)
    p.add_argument("--ebnerd-model", choices=["w2v", "bert"], default="w2v")
    p.add_argument("--embedding-batch-size", type=int, default=256)
    p.add_argument("--device", default=None)
    p.add_argument("--rebuild-history-index", action="store_true")
    p.add_argument("--beta", type=float, default=0.7, help="embed weight; set from bm25_embed_blend_tuning.csv")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    pred, zp, rows, elapsed = generate_submission(args.dataset, args)
    print(f"{args.dataset} beta={args.beta} rows={rows} elapsed_min={elapsed/60:.1f}\n{pred}\n{zp}")

if __name__ == "__main__":
    main()