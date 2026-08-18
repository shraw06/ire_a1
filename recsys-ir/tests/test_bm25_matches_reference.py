"""Cross-validate hand-built BM25 index against rank_bm25 reference implementation.

Constructs toy corpora (English + Danish), tokenizes identically for both
implementations, and asserts per-document BM25 scores match within
floating-point tolerance.

This is the ONLY place ``rank_bm25`` is imported — it is never used
in the production scoring path.
"""

from __future__ import annotations

import pytest
from rank_bm25 import BM25Okapi

from src.retrieval.bm25 import (
    BM25Engine,
    InvertedIndex,
    bm25_score_query,
    build_index,
    tokenize,
)


# ── Test fixtures ─────────────────────────────────────────────────

ENGLISH_CORPUS = [
    ("D1", "the cat sat on the mat"),
    ("D2", "the dog sat on the log"),
    ("D3", "cats and dogs are friends"),
    ("D4", "the quick brown fox jumps over the lazy dog"),
    ("D5", "a cat and a dog went to the park"),
]

DANISH_CORPUS = [
    ("D1", "katten sad på måtten og spiste fisk"),
    ("D2", "hunden løb i parken med sin ejer"),
    ("D3", "katte og hunde er gode venner i byen"),
    ("D4", "den hurtige brune ræv sprang over den dovne hund"),
    ("D5", "en kat og en hund gik til stranden for at bade"),
]


# ── Helpers ───────────────────────────────────────────────────────

def _scores_to_dict(results: list[tuple[str, float]]) -> dict[str, float]:
    """Convert [(doc_id, score)] to {doc_id: score}."""
    return {doc_id: score for doc_id, score in results}


def _reference_scores(
    tokenized_corpus: list[list[str]],
    query_tokens: list[str],
    doc_ids: list[str],
) -> dict[str, float]:
    """Score with rank_bm25.BM25Okapi and return {doc_id: score}."""
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query_tokens)
    return {did: float(s) for did, s in zip(doc_ids, scores)}


# ── English tests ─────────────────────────────────────────────────

class TestBM25MatchesReferenceEnglish:
    """Validate hand-built BM25 against rank_bm25 on English toy corpus."""

    @pytest.fixture()
    def setup(self):
        """Pre-tokenize corpus and build both indexes."""
        # No stopwords, no stemming — to get identical tokenization
        lang = "en"
        use_sw = False
        use_stem = False

        corpus_pairs = ENGLISH_CORPUS
        doc_ids = [d[0] for d in corpus_pairs]
        texts = [d[1] for d in corpus_pairs]

        tokenized = [tokenize(t, lang, use_sw, use_stem) for t in texts]
        index = build_index(corpus_pairs, lang, use_sw, use_stem)

        return {
            "index": index,
            "tokenized": tokenized,
            "doc_ids": doc_ids,
            "lang": lang,
            "use_sw": use_sw,
            "use_stem": use_stem,
        }

    def test_single_term_query(self, setup):
        """Single-term query 'cat' should produce matching scores."""
        query = "cat"
        qt = tokenize(query, setup["lang"], setup["use_sw"], setup["use_stem"])

        hand = _scores_to_dict(bm25_score_query(qt, setup["index"]))
        ref = _reference_scores(setup["tokenized"], qt, setup["doc_ids"])

        for did in setup["doc_ids"]:
            h = hand.get(did, 0.0)
            r = ref.get(did, 0.0)
            assert abs(h - r) < 1e-4, (
                f"Score mismatch for {did}: hand={h:.6f}, ref={r:.6f}"
            )

    def test_multi_term_query(self, setup):
        """Multi-term query 'cat dog' should produce matching scores."""
        query = "cat dog"
        qt = tokenize(query, setup["lang"], setup["use_sw"], setup["use_stem"])

        hand = _scores_to_dict(bm25_score_query(qt, setup["index"]))
        ref = _reference_scores(setup["tokenized"], qt, setup["doc_ids"])

        for did in setup["doc_ids"]:
            h = hand.get(did, 0.0)
            r = ref.get(did, 0.0)
            assert abs(h - r) < 1e-4, (
                f"Score mismatch for {did}: hand={h:.6f}, ref={r:.6f}"
            )

    def test_no_match_query(self, setup):
        """Query with no matching terms should produce all-zero scores."""
        query = "elephant"
        qt = tokenize(query, setup["lang"], setup["use_sw"], setup["use_stem"])

        hand = _scores_to_dict(bm25_score_query(qt, setup["index"]))
        ref = _reference_scores(setup["tokenized"], qt, setup["doc_ids"])

        for did in setup["doc_ids"]:
            assert hand.get(did, 0.0) == pytest.approx(0.0, abs=1e-6)
            assert ref.get(did, 0.0) == pytest.approx(0.0, abs=1e-6)

    def test_ranking_order_matches(self, setup):
        """The ranking order should be identical between hand-built and reference."""
        query = "the dog sat"
        qt = tokenize(query, setup["lang"], setup["use_sw"], setup["use_stem"])

        hand_results = bm25_score_query(qt, setup["index"])
        ref = _reference_scores(setup["tokenized"], qt, setup["doc_ids"])

        # Get reference ranking order (non-zero scores only)
        ref_ranked = sorted(
            [(did, s) for did, s in ref.items() if s > 0],
            key=lambda x: -x[1],
        )

        hand_ranked = [(did, s) for did, s in hand_results if s > 0]

        assert len(hand_ranked) == len(ref_ranked), (
            f"Different number of non-zero docs: hand={len(hand_ranked)}, ref={len(ref_ranked)}"
        )

        for (h_did, h_s), (r_did, r_s) in zip(hand_ranked, ref_ranked):
            assert h_did == r_did, f"Rank order mismatch: hand={h_did}, ref={r_did}"
            assert abs(h_s - r_s) < 1e-4, (
                f"Score mismatch for {h_did}: hand={h_s:.6f}, ref={r_s:.6f}"
            )


# ── Danish tests ──────────────────────────────────────────────────

class TestBM25MatchesReferenceDanish:
    """Validate hand-built BM25 against rank_bm25 on Danish toy corpus.

    Verifies that æ/ø/å are handled correctly by the Unicode-aware tokenizer.
    """

    @pytest.fixture()
    def setup(self):
        lang = "da"
        use_sw = False
        use_stem = False

        corpus_pairs = DANISH_CORPUS
        doc_ids = [d[0] for d in corpus_pairs]
        texts = [d[1] for d in corpus_pairs]

        tokenized = [tokenize(t, lang, use_sw, use_stem) for t in texts]
        index = build_index(corpus_pairs, lang, use_sw, use_stem)

        return {
            "index": index,
            "tokenized": tokenized,
            "doc_ids": doc_ids,
            "lang": lang,
            "use_sw": use_sw,
            "use_stem": use_stem,
        }

    def test_danish_single_term(self, setup):
        """Danish query 'ræv' should match correctly."""
        query = "ræv"
        qt = tokenize(query, setup["lang"], setup["use_sw"], setup["use_stem"])

        hand = _scores_to_dict(bm25_score_query(qt, setup["index"]))
        ref = _reference_scores(setup["tokenized"], qt, setup["doc_ids"])

        for did in setup["doc_ids"]:
            h = hand.get(did, 0.0)
            r = ref.get(did, 0.0)
            assert abs(h - r) < 1e-4, (
                f"Score mismatch for {did}: hand={h:.6f}, ref={r:.6f}"
            )

    def test_danish_multi_term_with_special_chars(self, setup):
        """Danish query 'katte hunde gik på' — verify æ/ø/å in matches."""
        query = "katte hunde gik på"
        qt = tokenize(query, setup["lang"], setup["use_sw"], setup["use_stem"])

        hand = _scores_to_dict(bm25_score_query(qt, setup["index"]))
        ref = _reference_scores(setup["tokenized"], qt, setup["doc_ids"])

        for did in setup["doc_ids"]:
            h = hand.get(did, 0.0)
            r = ref.get(did, 0.0)
            assert abs(h - r) < 1e-4, (
                f"Score mismatch for {did}: hand={h:.6f}, ref={r:.6f}"
            )

    def test_danish_ranking_order(self, setup):
        """Danish ranking order should match reference."""
        query = "hunden løb i parken"
        qt = tokenize(query, setup["lang"], setup["use_sw"], setup["use_stem"])

        hand_results = bm25_score_query(qt, setup["index"])
        ref = _reference_scores(setup["tokenized"], qt, setup["doc_ids"])

        ref_ranked = sorted(
            [(did, s) for did, s in ref.items() if s > 0],
            key=lambda x: -x[1],
        )
        hand_ranked = [(did, s) for did, s in hand_results if s > 0]

        assert len(hand_ranked) == len(ref_ranked)
        for (h_did, h_s), (r_did, r_s) in zip(hand_ranked, ref_ranked):
            assert h_did == r_did
            assert abs(h_s - r_s) < 1e-4


# ── BM25Engine wrapper test ───────────────────────────────────────

class TestBM25EngineWrapper:
    """Test the BM25Engine convenience class."""

    def test_engine_rank_returns_results(self):
        engine = BM25Engine.from_corpus(
            ENGLISH_CORPUS, dataset="mind", use_stopwords=False, use_stemming=False,
        )
        results = engine.rank("cat dog", top_k=3)
        assert len(results) <= 3
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_engine_candidate_restriction(self):
        engine = BM25Engine.from_corpus(
            ENGLISH_CORPUS, dataset="mind", use_stopwords=False, use_stemming=False,
        )
        results = engine.rank("cat dog", candidate_ids=["D1", "D3"], top_k=10)
        result_ids = {r[0] for r in results}
        assert result_ids == {"D1", "D3"}, f"Expected only D1, D3 but got {result_ids}"

    def test_engine_all_candidates_returned(self):
        """Even zero-scoring candidates should appear in results."""
        engine = BM25Engine.from_corpus(
            ENGLISH_CORPUS, dataset="mind", use_stopwords=False, use_stemming=False,
        )
        results = engine.rank("elephant", candidate_ids=["D1", "D2"], top_k=10)
        result_ids = {r[0] for r in results}
        assert "D1" in result_ids and "D2" in result_ids
