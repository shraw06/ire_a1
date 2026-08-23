"""Lexical vs semantic comparison on labeled validation data.

Large scale is streaming and reuses already-generated retrieval score files.
Small scale preserves the original in-memory behavior.
"""
from __future__ import annotations
import argparse, csv, json, logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from src.evaluation.ranking_metrics import recall_at_k

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
COLD_USER_THRESHOLD = 5
TAIL_ARTICLE_THRESHOLD = 10
RECALL_KS = [5, 10, 50, 100, 200]
STREAM_BATCH_SIZE = 20_000
BM25_CONFIG_TAG = "sw_nostem"


def _article_popularity(dataset: str, scale: str) -> dict[str, int]:
    from src.common.paths import interim_dir
    path = interim_dir(dataset, scale) / "behaviors.parquet"
    pop: dict[str, int] = {}
    if scale == "small":
        df = pl.read_parquet(path)
        rows = df.iter_rows(named=True)
        for row in rows:
            for cid in json.loads(row["candidates"]):
                cid = str(cid); pop[cid] = pop.get(cid, 0) + 1
        return pop
    pf = pq.ParquetFile(path)
    for b in pf.iter_batches(batch_size=STREAM_BATCH_SIZE, columns=["candidates"]):
        for raw in b.column(0).to_pylist():
            if raw is None: continue
            for cid in (json.loads(raw) if isinstance(raw, str) else raw):
                cid = str(cid); pop[cid] = pop.get(cid, 0) + 1
    return pop


def _user_history_lengths(dataset: str, scale: str) -> dict[str, int]:
    if scale == "small":
        from src.feature_store.user_store import UserFeatureStore
        store = UserFeatureStore(dataset, scale=scale)
        rows = store._store.query_sql(f"SELECT user_id, history_len FROM {store._store.alias}")
        return {str(r["user_id"]): int(r["history_len"]) for r in rows}
    if dataset == "ebnerd":
        from src.common.paths import processed_dir
        d = processed_dir(dataset, scale) / "history_index_validation"
        users = np.load(d / "user_ids.npy", mmap_mode="r")
        offsets = np.load(d / "offsets.npy", mmap_mode="r")
        return {str(int(u)): int(offsets[i+1]-offsets[i]) for i, u in enumerate(users)}
    from src.common.paths import interim_dir
    path = interim_dir(dataset, scale) / "behaviors.parquet"
    out: dict[str, int] = {}
    pf = pq.ParquetFile(path)
    for b in pf.iter_batches(batch_size=STREAM_BATCH_SIZE, columns=["user_id", "clicked_history"]):
        users, histories = b.column(0).to_pylist(), b.column(1).to_pylist()
        for uid, raw in zip(users, histories):
            n = len(json.loads(raw)) if isinstance(raw, str) and raw else len(raw or [])
            uid = str(uid)
            if n > out.get(uid, 0): out[uid] = n
    return out


def _score_files(dataset: str, scale: str) -> list[tuple[str, Path]]:
    """Find completed retrieval score files for one dataset."""
    from src.common.paths import processed_dir

    root = processed_dir(dataset, scale)

    bm = (
        root
        / f"bm25_scores_{dataset}_title_abstract_{BM25_CONFIG_TAG}.parquet"
    )

    models = ["minilm"] if dataset == "mind" else ["w2v", "bert"]

    out: list[tuple[str, Path]] = []

    if bm.exists():
        out.append(("bm25", bm))
    else:
        logger.warning("BM25 score file not found: %s", bm)

    for model in models:
        path = root / f"embed_scores_{dataset}_{model}.parquet"
        if path.exists():
            out.append((model, path))
        else:
            logger.warning(
                "Embedding score file not found: %s",
                path,
            )

    return out


def _iter_scores(path: Path) -> Iterator[dict[str, Any]]:
    pf = pq.ParquetFile(path)
    cols = ["impression_id", "user_id", "ranked_ids", "ground_truth"]
    for b in pf.iter_batches(batch_size=STREAM_BATCH_SIZE, columns=cols):
        for row in pa.Table.from_batches([b]).to_pylist():
            yield row


def _parse(v: Any) -> Any:
    return json.loads(v) if isinstance(v, str) else v


def _aggregate(path: Path, user_lens: dict[str, int], pop: dict[str, int]) -> dict[str, dict[int, tuple[float, int]]]:
    sums: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for row in _iter_scores(path):
        ranked = [str(x) for x in _parse(row["ranked_ids"])]
        truth = {str(x) for x in _parse(row["ground_truth"])}
        uid = str(row["user_id"])
        user_slice = "cold" if user_lens.get(uid, 0) <= COLD_USER_THRESHOLD else "warm"
        if truth:
            avg = float(np.mean([pop.get(a, 0) for a in truth]))
            article_slice = "tail" if avg <= TAIL_ARTICLE_THRESHOLD else "head"
        else:
            article_slice = "tail"
        slices = {"all", user_slice, article_slice}
        for k in RECALL_KS:
            r = recall_at_k(ranked, truth, k)
            for s in slices:
                sums[s][k] += float(r); counts[s][k] += 1
    return {s: {k: (sums[s][k] / counts[s][k] if counts[s][k] else 0.0, counts[s][k]) for k in RECALL_KS} for s in ["all","cold","warm","tail","head"]}


def build_comparison(datasets: list[str] | None = None, scale: str = "small") -> Path:
    datasets = datasets or ["mind", "ebnerd"]
    rows: list[dict[str, Any]] = []
    for ds in datasets:
        logger.info("Building comparison for %s", ds)
        pop = _article_popularity(ds, scale)
        lens = _user_history_lengths(ds, scale)
        files = _score_files(ds, scale)
        bm_path = next((p for m, p in files if m == "bm25"), None)
        if bm_path is None:
            logger.warning("BM25 scores missing for %s", ds); continue
        bm = _aggregate(bm_path, lens, pop)
        for model, ep in files:
            if model == "bm25": continue
            emb = _aggregate(ep, lens, pop)
            for k in RECALL_KS:
                for sl in ["all","cold","warm","tail","head"]:
                    br, bn = bm[sl][k]; er, en = emb[sl][k]
                    rows.append({"dataset":ds,"K":k,"slice":sl,"bm25_recall":round(br,6),"embed_model":model,"embed_recall":round(er,6),"bm25_n":bn,"embed_n":en,"bm25_wins":int(br>er)})
    from src.common.paths import results_dir
    out = results_dir(scale) / "lexical_vs_semantic.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset","K","slice","bm25_recall","embed_model","embed_recall","bm25_n","embed_n","bm25_wins"]
    with open(out,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    return out


def plot_comparison(csv_path: Path, scale: str) -> Path:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.common.paths import results_dir
    with open(csv_path) as f: rows=list(csv.DictReader(f))
    rows=[r for r in rows if int(r["K"])==10]
    groups=defaultdict(list)
    for r in rows: groups[(r["dataset"],r["embed_model"])].append(r)
    if not groups: return csv_path
    fig, axes = plt.subplots(1,len(groups),figsize=(6*len(groups),5),squeeze=False)
    order=["all","cold","warm","tail","head"]
    for i,((ds,model),gr) in enumerate(sorted(groups.items())):
        ax=axes[0][i]; x=np.arange(len(order)); w=.35
        bm=[float(next((r["bm25_recall"] for r in gr if r["slice"]==s),0)) for s in order]
        em=[float(next((r["embed_recall"] for r in gr if r["slice"]==s),0)) for s in order]
        ax.bar(x-w/2,bm,w,label="BM25"); ax.bar(x+w/2,em,w,label="Embedding")
        ax.set_title(f"{ds.upper()} — {model}"); ax.set_xticks(x); ax.set_xticklabels(["All","Cold","Warm","Tail","Head"]); ax.set_ylim(0,1.05); ax.set_ylabel("Recall@10"); ax.legend()
    fig.tight_layout(); out=results_dir(scale)/"lexical_vs_semantic.png"; fig.savefig(out,dpi=150,bbox_inches="tight"); plt.close(fig); return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    p=argparse.ArgumentParser(); p.add_argument("--scale",choices=["small","large"],default="small"); a=p.parse_args()
    csv_path=build_comparison(scale=a.scale); plot_path=plot_comparison(csv_path,a.scale)
    print("\nLexical vs. Semantic Comparison"); print(f"CSV: {csv_path}"); print(f"Plot: {plot_path}")
    with open(csv_path) as f: rows=list(csv.DictReader(f))
    for k in [10,100]:
        print(f"\nRecall@{k}"); print(f"{'Dataset':<10}{'Model':<10}{'Slice':<12}{'BM25':<10}{'Embed':<10}{'Winner':<10}")
        for r in rows:
            if int(r["K"])!=k: continue
            winner="BM25" if int(r["bm25_wins"]) else "Embed"
            print(f"{r['dataset']:<10}{r['embed_model']:<10}{r['slice']:<12}{float(r['bm25_recall']):<10.4f}{float(r['embed_recall']):<10.4f}{winner:<10}")

if __name__ == "__main__": main()
