"""Hand-built inverted index + BM25 scoring (Q2) — no external BM25 library used at runtime.

This module implements:
  1. Language-aware tokenization (English / Danish) with toggleable stopword
     removal and stemming.
  2. An inverted index: ``{term: [(doc_idx, tf), ...]}`` with per-term
     document frequency.
  3. BM25 Okapi scoring applied directly from the formula — not a wrapper
     around ``rank_bm25`` or any other library.

``rank_bm25`` is used ONLY in test assertions (``tests/test_bm25_matches_reference.py``)
to validate correctness; it is never imported here.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from typing import Optional, Sequence

from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem.snowball import SnowballStemmer

logger = logging.getLogger(__name__)

# ── Stopword lists (loaded once, cached) ──────────────────────────

_STOPWORDS: dict[str, frozenset[str]] = {}


def _get_stopwords(lang: str) -> frozenset[str]:
    """Return a frozen set of stopwords for the given language.

    Uses NLTK's curated stopword lists.  English and Danish have
    SEPARATE lists — the Danish list is never the English one.
    """
    if lang not in _STOPWORDS:
        nltk_lang = {"en": "english", "da": "danish"}[lang]
        _STOPWORDS[lang] = frozenset(nltk_stopwords.words(nltk_lang))
    return _STOPWORDS[lang]


# ── Stemmers (loaded once, cached) ────────────────────────────────

_STEMMERS: dict[str, SnowballStemmer] = {}


def _get_stemmer(lang: str) -> SnowballStemmer:
    """Return a Snowball stemmer for the given language."""
    if lang not in _STEMMERS:
        nltk_lang = {"en": "english", "da": "danish"}[lang]
        _STEMMERS[lang] = SnowballStemmer(nltk_lang)
    return _STEMMERS[lang]


# ── Tokenizer ─────────────────────────────────────────────────────

# English: simple ASCII-friendly regex
_EN_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Danish: Unicode-aware regex — æ/ø/å are word characters via \w
_DA_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(
    text: str,
    lang: str = "en",
    use_stopwords: bool = True,
    use_stemming: bool = False,
) -> list[str]:
    """Tokenize *text* with language-aware rules.

    Parameters
    ----------
    text : str
        Raw text to tokenize.
    lang : str
        ``"en"`` (English / MIND) or ``"da"`` (Danish / EB-NeRD).
    use_stopwords : bool
        If True, remove stopwords using the language-appropriate list.
    use_stemming : bool
        If True, apply Snowball stemming.

    Returns
    -------
    list[str]
        Lowercased, optionally stopped/stemmed tokens.
    """
    lowered = text.lower()

    if lang == "da":
        tokens = _DA_TOKEN_RE.findall(lowered)
    else:
        tokens = _EN_TOKEN_RE.findall(lowered)

    if use_stopwords:
        sw = _get_stopwords(lang)
        tokens = [t for t in tokens if t not in sw]

    if use_stemming:
        stemmer = _get_stemmer(lang)
        tokens = [stemmer.stem(t) for t in tokens]

    return tokens


def lang_for_dataset(dataset: str) -> str:
    """Map dataset name to language code."""
    return "da" if dataset.startswith("ebnerd") else "en"


# ── Inverted Index ────────────────────────────────────────────────

class InvertedIndex:
    """Hand-built inverted index with BM25 statistics.

    Attributes
    ----------
    postings : dict[str, list[tuple[int, int]]]
        ``{term: [(doc_idx, tf), ...]}`` — postings list sorted by doc_idx.
    df : dict[str, int]
        Document frequency per term.
    doc_lens : list[int]
        Number of tokens in each document.
    avgdl : float
        Average document length across the corpus.
    N : int
        Total number of documents in the corpus.
    doc_ids : list[str]
        Original document IDs, indexed by doc_idx.
    doc_term_tfs : list[dict[str, int]]
        Per-document term→tf mapping for O(1) candidate-restricted scoring.
    idf_table : dict[str, float]
        Precomputed IDF values (cached to avoid recomputation per query).
    """

    __slots__ = ("postings", "df", "doc_lens", "avgdl", "N", "doc_ids",
                 "_doc_id_to_idx", "doc_term_tfs", "idf_table")

    def __init__(
        self,
        postings: dict[str, list[tuple[int, int]]],
        df: dict[str, int],
        doc_lens: list[int],
        doc_ids: list[str],
        doc_term_tfs: list[dict[str, int]] | None = None,
    ) -> None:
        self.postings = postings
        self.df = df
        self.doc_lens = doc_lens
        self.N = len(doc_ids)
        self.avgdl = sum(doc_lens) / self.N if self.N > 0 else 0.0
        self.doc_ids = doc_ids
        self._doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}
        self.doc_term_tfs = doc_term_tfs or []
        # Pre-compute and cache IDF table
        self.idf_table = _compute_idf_table(df, self.N)

    def doc_idx(self, doc_id: str) -> int | None:
        """Return the internal index for *doc_id*, or None."""
        return self._doc_id_to_idx.get(doc_id)


def build_index(
    corpus: Sequence[tuple[str, str]],
    lang: str = "en",
    use_stopwords: bool = True,
    use_stemming: bool = False,
) -> InvertedIndex:
    """Build an inverted index from *(doc_id, text)* pairs.

    Parameters
    ----------
    corpus : sequence of (doc_id, text)
        The documents to index.
    lang, use_stopwords, use_stemming
        Passed through to :func:`tokenize`.

    Returns
    -------
    InvertedIndex
    """
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    df: dict[str, int] = defaultdict(int)
    doc_lens: list[int] = []
    doc_ids: list[str] = []
    doc_term_tfs: list[dict[str, int]] = []

    for doc_idx, (doc_id, text) in enumerate(corpus):
        doc_ids.append(doc_id)
        tokens = tokenize(text or "", lang, use_stopwords, use_stemming)
        doc_lens.append(len(tokens))

        term_freqs = Counter(tokens)
        doc_term_tfs.append(dict(term_freqs))
        for term, tf in term_freqs.items():
            postings[term].append((doc_idx, tf))
            df[term] += 1

    logger.debug(
        "Built index: %d docs, %d unique terms, avgdl=%.1f",
        len(doc_ids), len(postings), sum(doc_lens) / max(len(doc_ids), 1),
    )
    return InvertedIndex(dict(postings), dict(df), doc_lens, doc_ids, doc_term_tfs)


# ── BM25 Scorer ───────────────────────────────────────────────────

def _compute_idf_table(
    df: dict[str, int],
    N: int,
    epsilon: float = 0.25,
) -> dict[str, float]:
    """Precompute IDF values for all terms.

    Uses the ATIRE BM25 variant (same as ``rank_bm25.BM25Okapi``)::

        idf(t) = log((N - df(t) + 0.5) / (df(t) + 0.5))

    Terms with negative IDF (appearing in > N/2 documents) are floored
    to ``epsilon * average_idf`` to prevent them from penalising matches.

    Parameters
    ----------
    df : dict[str, int]
        Document frequency per term.
    N : int
        Total number of documents.
    epsilon : float
        Multiplier for the average-IDF floor applied to negative-IDF terms.
    """
    idf_map: dict[str, float] = {}
    idf_sum = 0.0
    negative_idf_terms: list[str] = []

    for term, freq in df.items():
        val = math.log((N - freq + 0.5) / (freq + 0.5))
        idf_map[term] = val
        idf_sum += val
        if val < 0:
            negative_idf_terms.append(term)

    avg_idf = idf_sum / len(idf_map) if idf_map else 0.0
    floor = epsilon * avg_idf
    for term in negative_idf_terms:
        idf_map[term] = floor

    return idf_map


def bm25_score_query(
    query_tokens: list[str],
    index: InvertedIndex,
    k1: float = 1.5,
    b: float = 0.75,
    candidate_idxs: set[int] | None = None,
) -> list[tuple[str, float]]:
    """Score documents against *query_tokens* using BM25 Okapi.

    Uses the ATIRE BM25 variant (same IDF as ``rank_bm25.BM25Okapi``)::

        IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5))
                 (floored to epsilon * avg_idf for negative values)
        score(t,d) = IDF(t) * tf(t,d) * (k1 + 1)
                      / (tf(t,d) + k1 * (1 - b + b * dl(d) / avgdl))

    Parameters
    ----------
    query_tokens : list[str]
        Pre-tokenized query.
    index : InvertedIndex
        The inverted index to score against.
    k1, b : float
        BM25 tuning parameters.
    candidate_idxs : set[int], optional
        If provided, score ONLY these document indices (for per-impression
        candidate restriction).  If None, score all documents.

    Returns
    -------
    list[tuple[str, float]]
        ``[(doc_id, score)]`` sorted descending by score.
    """
    N = index.N
    avgdl = index.avgdl
    scores: dict[int, float] = defaultdict(float)

    # Use cached IDF table from index
    idf_table = index.idf_table

    if candidate_idxs is not None and index.doc_term_tfs:
        # OPTIMIZED PATH: for candidate-restricted scoring, iterate over
        # candidates × query_terms instead of postings_len.
        # This is O(|candidates| × |query_terms|) instead of
        # O(sum(postings_len for each query_term)).
        k1_plus_1 = k1 + 1.0
        for doc_idx in candidate_idxs:
            dl = index.doc_lens[doc_idx]
            len_norm = k1 * (1.0 - b + b * dl / avgdl) if avgdl > 0 else k1
            doc_tfs = index.doc_term_tfs[doc_idx]
            doc_score = 0.0
            for term in query_tokens:
                tf = doc_tfs.get(term, 0)
                if tf == 0:
                    continue
                idf = idf_table.get(term, 0.0)
                if idf == 0.0:
                    continue
                doc_score += idf * (tf * k1_plus_1) / (tf + len_norm)
            if doc_score > 0.0:
                scores[doc_idx] = doc_score
    else:
        # STANDARD PATH: iterate over postings (no candidate restriction
        # or no doc_term_tfs available)
        for term in query_tokens:
            if term not in index.postings:
                continue
            idf = idf_table.get(term, 0.0)
            if idf == 0.0:
                continue

            for doc_idx, tf in index.postings[term]:
                if candidate_idxs is not None and doc_idx not in candidate_idxs:
                    continue
                dl = index.doc_lens[doc_idx]
                denom = tf + k1 * (1.0 - b + b * dl / avgdl) if avgdl > 0 else tf + k1
                score = idf * (tf * (k1 + 1.0)) / denom
                scores[doc_idx] += score

    # Sort by score descending, then by doc_idx ascending for stability
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return [(index.doc_ids[idx], sc) for idx, sc in ranked]


# ── BM25 Engine (convenience wrapper) ─────────────────────────────

class BM25Engine:
    """Convenience wrapper encapsulating index + scorer.

    Usage::

        engine = BM25Engine.from_corpus(corpus, dataset="mind")
        results = engine.rank("search query text", candidate_ids=["N1", "N2"], top_k=10)
    """

    def __init__(
        self,
        index: InvertedIndex,
        lang: str,
        use_stopwords: bool,
        use_stemming: bool,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.index = index
        self.lang = lang
        self.use_stopwords = use_stopwords
        self.use_stemming = use_stemming
        self.k1 = k1
        self.b = b

    @classmethod
    def from_corpus(
        cls,
        corpus: Sequence[tuple[str, str]],
        dataset: str = "mind",
        use_stopwords: bool = True,
        use_stemming: bool = False,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> BM25Engine:
        """Build an engine from a list of ``(doc_id, text)`` pairs."""
        lang = lang_for_dataset(dataset)
        idx = build_index(corpus, lang, use_stopwords, use_stemming)
        return cls(idx, lang, use_stopwords, use_stemming, k1, b)

    def rank(
        self,
        query_text: str,
        candidate_ids: list[str] | None = None,
        top_k: int = 100,
    ) -> list[tuple[str, float]]:
        """Tokenize *query_text*, score candidates, return top-K.

        Parameters
        ----------
        query_text : str
            Raw query string (will be tokenized using the engine's settings).
        candidate_ids : list[str], optional
            If provided, restrict scoring to these document IDs only.
        top_k : int
            Maximum number of results to return.
        """
        query_tokens = tokenize(
            query_text, self.lang, self.use_stopwords, self.use_stemming
        )
        if not query_tokens:
            # Empty query → return candidates in original order with 0 score
            if candidate_ids:
                return [(cid, 0.0) for cid in candidate_ids[:top_k]]
            return []

        candidate_idxs: set[int] | None = None
        if candidate_ids is not None:
            candidate_idxs = set()
            for cid in candidate_ids:
                idx = self.index.doc_idx(cid)
                if idx is not None:
                    candidate_idxs.add(idx)

        results = bm25_score_query(
            query_tokens, self.index, self.k1, self.b, candidate_idxs
        )

        # If we have candidate_ids, ensure all candidates appear in results
        # (some may have score 0 if they share no terms with the query)
        if candidate_ids is not None:
            scored_ids = {r[0] for r in results}
            for cid in candidate_ids:
                if cid not in scored_ids:
                    results.append((cid, 0.0))

        return results[:top_k]

    @property
    def corpus_size(self) -> int:
        return self.index.N

    def __repr__(self) -> str:
        return (
            f"BM25Engine(N={self.index.N}, lang={self.lang}, "
            f"stopwords={self.use_stopwords}, stemming={self.use_stemming}, "
            f"k1={self.k1}, b={self.b})"
        )
