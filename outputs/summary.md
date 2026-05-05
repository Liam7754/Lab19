# Lab Day 19 — Benchmark Summary

- Extraction backend: **rule**
- Graph backend: **neo4j**
- Documents indexed: 24
- Triples (raw / deduped): 99 / 98
- Extraction time: 0.01s
- Graph upsert time: 5.61s

## Accuracy

| System | Correct | Accuracy | Avg latency (ms) |
|---|---|---|---|
| Flat RAG (TF-IDF) | 17/20 | 85% | 1.60 |
| GraphRAG (2-hop)  | 20/20 | 100% | 284.54 |

## Per-question verdicts

| # | Hops | Question | Flat | Graph | Verdict |
|---|---|---|---|---|---|
| 1 | 1 | Who founded OpenAI? | ok | ok | tie (both correct) |
| 2 | 1 | In what year was OpenAI founded? | ok | ok | tie (both correct) |
| 3 | 1 | Where is Anthropic headquartered? | ok | ok | tie (both correct) |
| 4 | 1 | Who is the CEO of Google? | ok | ok | tie (both correct) |
| 5 | 1 | Which company acquired GitHub? | ok | ok | tie (both correct) |
| 6 | 2 | Which large language model families are developed by companies headquartered in San Francisco? | miss | ok | GraphRAG wins |
| 7 | 2 | Anthropic's founders previously worked at which company? | ok | ok | tie (both correct) |
| 8 | 2 | Who founded the company that owns DeepMind? | miss | ok | GraphRAG wins |
| 9 | 2 | Which company invested over ten billion dollars in OpenAI? | ok | ok | tie (both correct) |
| 10 | 2 | Name two competitors of ChatGPT. | ok | ok | tie (both correct) |
| 11 | 3 | Which AI labs use Nvidia H100 GPUs to train models? | ok | ok | tie (both correct) |
| 12 | 2 | Which subsidiaries does Microsoft own? | miss | ok | GraphRAG wins |
| 13 | 2 | Who founded xAI and what model does it develop? | ok | ok | tie (both correct) |
| 14 | 2 | Elon Musk co-founded which two AI organisations? | ok | ok | tie (both correct) |
| 15 | 2 | Which company developed AlphaFold and who owns that company? | ok | ok | tie (both correct) |
| 16 | 1 | How much did Meta pay to acquire WhatsApp? | ok | ok | tie (both correct) |
| 17 | 1 | How much did Microsoft pay to acquire LinkedIn? | ok | ok | tie (both correct) |
| 18 | 1 | Who founded Mistral AI? | ok | ok | tie (both correct) |
| 19 | 1 | Which products does Apple develop? | ok | ok | tie (both correct) |
| 20 | 1 | List two AI investors in Anthropic. | ok | ok | tie (both correct) |