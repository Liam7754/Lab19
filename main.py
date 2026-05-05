"""
End-to-end Lab Day 19 pipeline.

Run:  python main.py

Stages:
  1. Load corpus
  2. Extract triples (LLM if USE_LLM=1, otherwise deterministic rules)
  3. Deduplicate triples
  4. Upsert into Neo4j (falls back to NetworkX if the database is unreachable)
  5. Render the graph PNG
  6. Run the 20-question benchmark across Flat RAG and GraphRAG
  7. Write `benchmark_results.csv` and a Markdown summary
"""

from __future__ import annotations

import csv
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from benchmark import QUESTIONS, evaluate
from corpus import DOCUMENTS
from extraction import deduplicate, extract_corpus
from flat_rag import FlatRAG
from graph import open_store
from graph_rag import GraphRAG
from visualize import render

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("lab19")

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def main() -> None:
    load_dotenv()
    use_llm = os.getenv("USE_LLM", "0") == "1"

    openai_client = None
    if use_llm:
        from openai import OpenAI

        openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # ------------------------------------------------------------------
    # 1. + 2. Extract triples
    # ------------------------------------------------------------------
    log.info("Extracting triples (backend=%s) ...", "llm" if use_llm else "rule")
    triples, ext_stats = extract_corpus(
        DOCUMENTS,
        use_llm=use_llm,
        openai_client=openai_client,
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )
    log.info(
        "Extracted %d raw triples in %.2fs", ext_stats.triples, ext_stats.elapsed_s
    )

    # 3. Dedup
    deduped = deduplicate(triples)
    log.info("After deduplication: %d unique triples", len(deduped))

    # ------------------------------------------------------------------
    # 4. Upsert into Neo4j (or NetworkX fallback)
    # ------------------------------------------------------------------
    neo_cfg = {
        "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        # Accept both `NEO4J_USER` and `NEO4J_USERNAME` (Aura convention).
        "user": os.getenv("NEO4J_USER")
        or os.getenv("NEO4J_USERNAME")
        or "neo4j",
        "password": os.getenv("NEO4J_PASSWORD", "password"),
    }
    t0 = time.time()
    store, backend = open_store(prefer_neo4j=True, neo4j_cfg=neo_cfg)
    if backend == "neo4j":
        log.info("Connected to Neo4j at %s", neo_cfg["uri"])
        store.reset()
    else:
        log.info("Using in-memory NetworkX store (Neo4j not reachable)")
    store.upsert_triples(deduped)
    upsert_elapsed = time.time() - t0
    log.info("Upsert finished in %.2fs", upsert_elapsed)

    # ------------------------------------------------------------------
    # 5. Visualisation
    # ------------------------------------------------------------------
    image_path = OUT / "knowledge_graph.png"
    render(deduped, str(image_path))
    log.info("Wrote graph visualisation to %s", image_path)

    # Persist the triples themselves so a grader can inspect the index.
    triples_csv = OUT / "triples.csv"
    with triples_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["head", "relation", "tail", "source_doc"])
        for t in deduped:
            w.writerow([t.head, t.relation, t.tail, t.source_doc])
    log.info("Wrote %d triples to %s", len(deduped), triples_csv)

    # ------------------------------------------------------------------
    # 6. Benchmark
    # ------------------------------------------------------------------
    flat = FlatRAG(DOCUMENTS)
    graphrag = GraphRAG(store, hops=2)

    rows = []
    flat_correct = graph_correct = 0
    flat_total_time = graph_total_time = 0.0

    for q in QUESTIONS:
        t0 = time.time()
        flat_ctx = flat.context(q.text, k=3)
        flat_dt = time.time() - t0

        t0 = time.time()
        graph_ctx = graphrag.context(q.text)
        graph_dt = time.time() - t0

        flat_ok = evaluate(flat_ctx, q)
        graph_ok = evaluate(graph_ctx, q)
        flat_correct += int(flat_ok)
        graph_correct += int(graph_ok)
        flat_total_time += flat_dt
        graph_total_time += graph_dt

        verdict = ""
        if not flat_ok and graph_ok:
            verdict = "GraphRAG wins"
        elif flat_ok and not graph_ok:
            verdict = "Flat RAG wins"
        elif flat_ok and graph_ok:
            verdict = "tie (both correct)"
        else:
            verdict = "tie (both wrong)"

        rows.append(
            {
                "qid": q.qid,
                "question": q.text,
                "hop_difficulty": q.hop_difficulty,
                "flat_ok": flat_ok,
                "graph_ok": graph_ok,
                "verdict": verdict,
                "flat_time_ms": round(flat_dt * 1000, 2),
                "graph_time_ms": round(graph_dt * 1000, 2),
                "flat_context": flat_ctx.replace("\n", " | "),
                "graph_context": graph_ctx.replace("\n", " | "),
            }
        )

    # CSV
    csv_path = OUT / "benchmark_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote benchmark results to %s", csv_path)

    # ------------------------------------------------------------------
    # 7. Summary report
    # ------------------------------------------------------------------
    n = len(QUESTIONS)
    summary_lines = [
        "# Lab Day 19 — Benchmark Summary",
        "",
        f"- Extraction backend: **{ext_stats.backend}**",
        f"- Graph backend: **{backend}**",
        f"- Documents indexed: {len(DOCUMENTS)}",
        f"- Triples (raw / deduped): {ext_stats.triples} / {len(deduped)}",
        f"- Extraction time: {ext_stats.elapsed_s:.2f}s",
        f"- Graph upsert time: {upsert_elapsed:.2f}s",
        "",
        "## Accuracy",
        "",
        f"| System | Correct | Accuracy | Avg latency (ms) |",
        f"|---|---|---|---|",
        f"| Flat RAG (TF-IDF) | {flat_correct}/{n} | {flat_correct/n:.0%} | "
        f"{flat_total_time/n*1000:.2f} |",
        f"| GraphRAG (2-hop)  | {graph_correct}/{n} | {graph_correct/n:.0%} | "
        f"{graph_total_time/n*1000:.2f} |",
        "",
        "## Per-question verdicts",
        "",
        "| # | Hops | Question | Flat | Graph | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        summary_lines.append(
            "| {qid} | {h} | {q} | {f} | {g} | {v} |".format(
                qid=r["qid"],
                h=next(qq.hop_difficulty for qq in QUESTIONS if qq.qid == r["qid"]),
                q=r["question"],
                f="ok" if r["flat_ok"] else "miss",
                g="ok" if r["graph_ok"] else "miss",
                v=r["verdict"],
            )
        )

    if use_llm:
        summary_lines += [
            "",
            "## Estimated extraction cost (LLM backend)",
            "",
            f"- Prompt tokens (est): {ext_stats.prompt_tokens_estimate}",
            f"- Completion tokens (est): {ext_stats.completion_tokens_estimate}",
        ]

    (OUT / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    log.info("Wrote summary to %s", OUT / "summary.md")

    store.close()

    print()
    print(f"Flat RAG : {flat_correct}/{n}  ({flat_correct/n:.0%})")
    print(f"GraphRAG : {graph_correct}/{n}  ({graph_correct/n:.0%})")
    print(f"Outputs in: {OUT}")


if __name__ == "__main__":
    main()
