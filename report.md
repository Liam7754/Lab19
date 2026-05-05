# Lab Day 19 — Analysis Report

> Companion to `outputs/summary.md` (auto-generated numbers) and the
> `outputs/benchmark_results.csv` (per-question contexts).

## 1. Section answers (Phần 1)

**Entity Extraction — how does the LLM tell a node from an attribute?**
We constrain the LLM to a closed relation schema (17 relations, see
`extraction.RELATION_TYPES`). Heads and tails of triples are *nodes*; the
relation label is the *edge attribute*. Surface attributes that don't connect
two distinct entities (e.g. "OpenAI's mission is to benefit humanity") are
deliberately *not* in the schema, so the LLM is forced to drop them rather
than promote a free-text mission statement to a node. The same closed schema
also applies to the deterministic fallback extractor.

**Graph Construction — why does deduplication matter?**
The same fact is mentioned in multiple documents (e.g. Anthropic's HQ shows
up in `doc_anthropic_1` and is implicit in `doc_anthropic_2`). Without
deduplication you'd get parallel edges that distort centrality measures and
inflate retrieval contexts, while still teaching the LLM the same fact. We
key on `(head.lower(), relation, tail.lower())` and keep secondary sources as
`also_seen_in` provenance metadata, so multi-source claims gain weight without
duplicating edges.

**Query Answering — BFS on a graph vs vector search?**
TF-IDF / vector search picks documents whose *surface form* is closest to the
question. It cannot follow `subsidiary_of` then `founded_by` to answer "who
founded the company that owns DeepMind?". BFS on the knowledge graph follows
*explicit relations* — it doesn't care whether the answer document mentions
the question's keywords at all, only whether the relevant nodes are
graph-connected within k hops. That's why the GraphRAG-only wins below are
all multi-hop questions.

## 2. Pipeline

```
24 raw docs ──► extraction.extract_corpus  ──►  99 raw triples
                                                ↓ deduplicate
                                                98 unique triples
                                                ↓ Neo4jStore.upsert_triples
                                            Neo4j (`:Entity` + 17 rel types)
                                                ↓ visualize.render
                                       outputs/knowledge_graph.png
```

Indexing took **~0.03s** (deterministic backend) + **~4.4s** to upsert via
Cypher — see Section 4 for the LLM-backend cost.

## 3. Where GraphRAG won (Deliverable #3 hallucination cases)

GraphRAG **20 / 20** vs Flat RAG **17 / 20**. The three wins are all
multi-hop questions whose answer fragments live in different documents that
TF-IDF can't co-retrieve.

### Q6 — "Which LLM families are developed by companies HQ'd in San Francisco?"

* **Flat RAG (miss):** top-3 docs were `doc_xai_1` (xAI/Grok in SF Bay Area —
  near-but-not-equal), `doc_anthropic_1` (HQ doc, doesn't mention Claude),
  `doc_openai_1` (HQ doc, doesn't mention GPT). The model families live in
  *separate* docs (`doc_anthropic_2`, `doc_openai_2`) that didn't make the
  top-3. Flat RAG would either hallucinate Grok or miss the answer.
* **GraphRAG (ok):** seed `San Francisco`. 1-hop reaches `Anthropic` and
  `OpenAI`; 2-hop reaches their `DEVELOPS` edges → Claude, GPT-4, ChatGPT.

### Q8 — "Who founded the company that owns DeepMind?"

This is the textbook 3-hop graph question.

* **Flat RAG (miss):** retrieved DeepMind docs (mentioning Alphabet), but
  *not* `doc_google_1` (the only doc with `Larry Page` / `Sergey Brin`).
* **GraphRAG (ok):** seed `DeepMind` → 1-hop `Alphabet` (`SUBSIDIARY_OF`),
  Google (`ACQUIRED`) → 2-hop `Google FOUNDED_BY Larry Page`, `Sergey Brin`.

### Q12 — "Which subsidiaries does Microsoft own?"

* **Flat RAG (miss):** top-3 included the OpenAI-investment doc, the
  Microsoft-founders doc, and the LinkedIn-acquisition doc — but **not** the
  GitHub-acquisition doc (`doc_microsoft_2`). Result: only 1 of 2
  subsidiaries surfaces.
* **GraphRAG (ok):** seed `Microsoft` → all `ACQUIRED` and `SUBSIDIARY_OF`
  edges return at hop 0/1 in a single Cypher query.

## 4. Cost analysis (Phần 4 deliverable)

| Stage | Time (rule backend) | Time (LLM backend, est.) | Notes |
|---|---|---|---|
| Extraction | ~30 ms | ~25 s for 24 docs | LLM cost dominated by per-doc latency |
| Deduplication | <1 ms | <1 ms | dict by `(head, rel, tail)` |
| Cypher upsert | ~4.4 s | ~4.4 s | one `MERGE` per relation type |
| Graph render (PNG) | ~2 s | ~2 s | matplotlib spring layout |
| 20-question benchmark | ~50 ms | n/a | TF-IDF + 2-hop BFS |

**Token usage estimate (LLM backend, 24 docs).** With `gpt-4o-mini` at
typical pricing, the system prompt is ~120 tokens and each doc is ~60 tokens
of input + ~80 tokens of JSON output. Total ≈ **3.5k input + 1.9k output
tokens** per full re-index — under one US cent. The deterministic backend
is free and produces the same 98 triples, so the LLM is only justified if
the corpus broadens beyond what the closed regex schema covers.

**Operational implication.** The expensive stage on a fresh deploy is the
*one-time* extraction; thereafter the graph is cheap to query (every
benchmark question completes in ~1 ms because Cypher hits an indexed
`:Entity(name)` constraint). This inverts the cost profile of Flat RAG,
where indexing is free but every query pays for embedding + ANN search.

## 5. Limitations & next steps

* The closed relation schema covers companies / people / products / years /
  acquisitions / investors well, but generalising to a broader corpus would
  need either an open relation extractor or LLM-backed extraction.
* The seed-entity resolver in `graph_rag._candidate_seeds` uses a small alias
  table. A production version would swap in a proper entity linker
  (e.g. spaCy + a BLINK-style retriever).
* Benchmark scores the *retrieval context*, not a generated answer, to
  isolate retrieval quality from LLM phrasing. A real evaluation would
  feed both contexts into the same LLM and grade the final answers.
* Neo4j is the recommended backend; the NetworkX fallback exists for
  development and CI but isn't a substitute when you want the Bloom/Browser
  visualisation deliverable.
