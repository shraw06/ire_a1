# EDA Summary — MIND & EB-NeRD Datasets

All statistics below were produced by `notebooks/00_explore_mind.ipynb` and
`notebooks/00_explore_ebnerd.ipynb`.  Every split boundary, threshold, or
design-note claim **must trace back to a number in this file**.

---

## 1  Dataset Scale

| Metric | MIND (MINDsmall) | EB-NeRD (demo) |
|--------|----------------:|---------------:|
| **Format** | TSV (no headers) | Parquet |
| **Language** | English | Danish (æ/ø/å confirmed) |
| **Articles (news table)** | train 51,282 / dev 42,416 | 11,777 (shared) |
| **Impression logs** | train 156,965 / dev 73,152 | train 24,724 / val 25,356 |
| **Impression–article pairs** | train 5,843,444 / dev 2,740,998 | train 278,139 / val 304,915 |
| **Unique users** | train 50,000 / dev 50,000 | train 1,590 / val 1,562 |
| **History records** | (inline in behaviors) | train 1,590 / val 1,562 |

---

## 2  Timestamp Ranges

| Split | Min | Max |
|-------|-----|-----|
| MIND train | 2019-11-09 00:00:19 | 2019-11-14 23:59:13 |
| MIND dev | 2019-11-15 00:00:01 | 2019-11-15 23:58:03 |
| EB-NeRD train | 2023-05-18 07:00:03 | 2023-05-25 06:59:52 |
| EB-NeRD validation | 2023-05-25 07:00:15 | 2023-06-01 06:59:33 |

**Temporal split note:**
- MIND train covers 6 days (Nov 9–14), dev is the next day (Nov 15).
- EB-NeRD train covers 7 days (May 18–25), validation is the next 7 days (May 25 – Jun 1).
- Both datasets use a natural temporal split — no future-leakage risk with these boundaries.

---

## 3  Click-Through Rate (CTR)

| Split | Positive Clicks | Total Pairs | CTR |
|-------|----------------:|------------:|----:|
| MIND train | 236,344 | 5,843,444 | **4.04%** |
| MIND dev | 111,383 | 2,740,998 | **4.06%** |
| EB-NeRD train | 24,888 | 278,139 | **8.95%** |
| EB-NeRD validation | 25,505 | 304,915 | **8.36%** |

**Note:** EB-NeRD CTR (~8.5%) is roughly 2× MIND CTR (~4%). This likely
reflects different impression sizes and UI design (Ekstra Bladet vs. MSN).

---

## 4  Clicks per User Distribution

| Stat | MIND train | MIND dev | EB-NeRD train | EB-NeRD val |
|------|----------:|--------:|--------------:|------------:|
| Users with ≥1 click | 50,000 | 50,000 | 1,590 | 1,562 |
| Min | 1 | 1 | 1 | 1 |
| Max | 129 | 33 | 177 | 122 |
| Mean | 4.73 | 2.23 | 15.65 | 16.33 |
| Median | 3 | 1 | 9 | 10 |
| P75 | 6 | 3 | 22 | 24 |
| P90 | 10 | 4 | 39 | 40 |
| P95 | 15 | 6 | 50 | 54 |
| P99 | 28 | 11 | 77 | 78 |

**Long-tail observation:**
- MIND: Median 3 clicks, P99 = 28 → strong long tail. Many cold-start users with 1–3 clicks.
- EB-NeRD: Median 9–10 clicks, P99 = 77–78 → users are more active, but still a tail.
- Cold-start threshold: users with ≤ 5 clicks could be considered "cold" for MIND
  (covers roughly the bottom 50–75% of users).

---

## 5  Impressions per Article Distribution

| Stat | MIND train | MIND dev | EB-NeRD train | EB-NeRD val |
|------|----------:|--------:|--------------:|------------:|
| Unique articles in impressions | 20,288 | 5,369 | 2,478 | 2,738 |
| Min | 1 | 1 | 1 | 1 |
| Max | 23,037 | 47,285 | 3,649 | 2,323 |
| Mean | 288 | 511 | – | – |
| Median | 7 | 11 | 15 | 12 |
| Top-1 article impressions | 23,037 | 47,285 | 3,649 | 2,323 |

**Head-vs-tail observation:**
- Extreme power-law: top-10 articles each receive 15K–47K impressions (MIND) or 1K–3.6K (EB-NeRD),
  while the median article appears only 7–15 times.
- Tail-article threshold: articles with ≤ 10 impressions are in the bottom ~50%.

---

## 6  Missing Text

| Metric | MIND train | MIND dev | EB-NeRD |
|--------|----------:|--------:|--------:|
| Missing title | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| Missing abstract | 2,666 (5.2%) | 2,021 (4.8%) | — |
| Missing subtitle | — | — | 803 (6.8%) |
| Missing body | N/A (body via URL only) | N/A | 933 (7.9%) |

**Notes:**
- MIND provides abstract but not body text in TSV; body is available via article URL.
- EB-NeRD has title, subtitle, and body directly in Parquet.
- ~5–8% missingness in abstract/subtitle/body — not severe but needs handling.

---

## 7  Entity Annotations (MIND only)

- **Format:** JSON array per article, with fields: `Label`, `Type`, `WikidataId`,
  `Confidence`, `OccurrenceOffsets`, `SurfaceForms`.
- **Entity types:** P (Person), C (Concept), G (Geography), O (Organization), etc.
- **Entity embeddings:** 100-dimensional vectors (tab-separated `.vec` files).
  - `entity_embedding.vec`: train 26,904 entities, dev 22,893 entities.
  - `relation_embedding.vec`: 1,091 relations (same in train and dev).
- **Example:** `N55528` → 3 title entities (Prince Philip Q80976, Prince Charles Q43274,
  Elizabeth II Q9682) with confidence scores.

---

## 8  Article Embeddings (EB-NeRD only)

- The **demo bundle does not include embedding files**.
- The full dataset provides pre-computed embeddings in a separate zip:
  - Word2Vec embeddings
  - Multilingual BERT embeddings
- No embedding columns appear in `articles.parquet` for the demo bundle.
- **Action:** Download embedding zips when moving beyond demo scale.

---

## 9  Language & Tokenization Notes

- **MIND:** English text. Standard English tokenizers and stopword lists apply.
- **EB-NeRD:** Danish text confirmed. Characters **æ, ø, å, Æ, Ø, Å** found in titles.
  - Do **NOT** use English tokenizers or stopword lists.
  - Use Unicode-aware tokenization (e.g., spaCy `da` model, or character-class regex).
  - æ, ø, å are distinct letters (not accented variants) and sort after z in Danish.

---

## 10  Schema Summary

### MIND (TSV, no headers)

**news.tsv** — 8 columns:
`news_id | category | subcategory | title | abstract | url | title_entities (JSON) | abstract_entities (JSON)`

**behaviors.tsv** — 5 columns:
`impression_id | user_id | time (M/D/YYYY h:mm:ss AM/PM) | history (space-separated news IDs) | impressions (space-separated NewsID-Label pairs)`

### EB-NeRD (Parquet)

**articles.parquet** — 21 columns:
`article_id, title, subtitle, last_modified_time, premium, body, published_time, image_ids, article_type, url, ner_clusters, entity_groups, topics, category, subcategory, category_str, total_inviews, total_pageviews, total_read_time, sentiment_score, sentiment_label`

**behaviors.parquet** — 17 columns:
`impression_id, article_id, impression_time, read_time, scroll_percentage, device_type, article_ids_inview (List[Int32]), article_ids_clicked (List[Int32]), user_id, is_sso_user, gender, postcode, age, is_subscriber, session_id, next_read_time, next_scroll_percentage`

**history.parquet** — 5 columns:
`user_id, impression_time_fixed (List[Datetime]), scroll_percentage_fixed (List[Float32]), article_id_fixed (List[Int32]), read_time_fixed (List[Float32])`
