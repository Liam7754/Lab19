# Lab Day 19 — GraphRAG with a Tech Company Corpus

A complete GraphRAG pipeline (extraction → graph → multi-hop retrieval) over a
small Tech Company corpus, benchmarked against a Flat RAG baseline on 20
questions. Recommended graph backend is **Neo4j** (per the task brief), with a
NetworkX fallback so the lab still runs end-to-end on a machine with no
database.

## What's in here

| File | Purpose |
|---|---|
| `corpus.py` | 24-document Tech Company corpus |
| `extraction.py` | Triple extraction (LLM backend + deterministic fallback) |
| `graph.py` | `Neo4jStore` + `NetworkxStore` (same surface) |
| `graph_rag.py` | Seed entity resolution + 2-hop traversal + textualisation |
| `flat_rag.py` | TF-IDF + cosine similarity baseline |
| `benchmark.py` | 20 benchmark questions with `must_contain` answer fragments |
| `visualize.py` | Renders the graph to `outputs/knowledge_graph.png` |
| `main.py` | End-to-end orchestrator |
| `outputs/` | Generated: triples CSV, benchmark CSV, graph PNG, summary MD |
| `report.md` | Analysis & cost write-up (Deliverable #4) |

## Setup

```powershell
pip install -r requirements.txt
copy .env.example .env   # then edit with your Neo4j / OpenAI creds
```

Start Neo4j (Docker is easiest):

```powershell
docker run --name neo4j-lab19 -p 7474:7474 -p 7687:7687 `
    -e NEO4J_AUTH=neo4j/password neo4j:5
```

The pipeline auto-detects Neo4j and falls back to NetworkX if unreachable, so
you can also run the lab fully offline. The Bloom/Browser UI at
http://localhost:7474 is what gives you the visualisation deliverable.

## Run

```powershell
python main.py
```

Outputs in `outputs/`:

* `triples.csv` — 98 deduplicated triples with provenance
* `knowledge_graph.png` — matplotlib rendering (Deliverable #2)
* `benchmark_results.csv` — per-question Flat RAG vs GraphRAG (Deliverable #3)
* `summary.md` — accuracy table + per-question verdicts

## Configuration

Set in `.env`:

* `USE_LLM=1` to use OpenAI for extraction (requires `OPENAI_API_KEY`).
  The default `USE_LLM=0` uses the bundled deterministic extractor so the
  lab is fully reproducible offline. Both backends produce the same closed
  relation schema, so the rest of the pipeline doesn't change.
* `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` — bolt credentials.

## Headline numbers

(Reproduce by running `python main.py`.)

| System | Correct | Accuracy |
|---|---|---|
| Flat RAG (TF-IDF) | 17/20 | 85% |
| GraphRAG (2-hop) | 20/20 | 100% |

The three GraphRAG-only wins are all multi-hop questions (Q6, Q8, Q12). See
`report.md` for the qualitative analysis.

## Pipeline at a glance

```
corpus  ──► extraction ──► dedup ──► Neo4j (Cypher) ──► visualize
                                          │
                                          ▼
question ──► seed entity ──► 2-hop subgraph ──► textualise ──► LLM answer
```
