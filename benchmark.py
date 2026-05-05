"""
Benchmark suite: 20 questions of escalating difficulty.

Each question carries a list of `must_contain` answer fragments. A retrieved
context is scored as "correct" if every required fragment appears in it
(case-insensitive substring). This is a deliberately strict, retrieval-level
metric — the lab brief asks us to show *where Flat RAG hallucinates and
GraphRAG doesn't*, which is fundamentally a retrieval question, so we judge
the context (not a generated answer) so the score isn't muddied by LLM
phrasing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Question:
    qid: int
    text: str
    must_contain: List[str]
    hop_difficulty: int  # 1 = single-doc lookup, 2+ = multi-hop reasoning
    note: str = ""


QUESTIONS: List[Question] = [
    Question(1, "Who founded OpenAI?", ["Sam Altman", "Elon Musk"], 1),
    Question(2, "In what year was OpenAI founded?", ["2015"], 1),
    Question(3, "Where is Anthropic headquartered?", ["San Francisco"], 1),
    Question(4, "Who is the CEO of Google?", ["Sundar Pichai"], 1),
    Question(5, "Which company acquired GitHub?", ["Microsoft"], 1),

    # 2-hop questions: the answer requires combining two facts.
    Question(
        6,
        "Which large language model families are developed by companies "
        "headquartered in San Francisco?",
        ["GPT", "Claude"],
        2,
        note="Needs HQ→company→DEVELOPS chain (OpenAI, Anthropic).",
    ),
    Question(
        7,
        "Anthropic's founders previously worked at which company?",
        ["OpenAI"],
        2,
        note="Cross-document link via founder bios.",
    ),
    Question(
        8,
        "Who founded the company that owns DeepMind?",
        ["Larry Page", "Sergey Brin"],
        2,
        note="DeepMind -> SUBSIDIARY_OF Alphabet -> ... but founders attach to "
             "Google, which Alphabet now owns.",
    ),
    Question(
        9,
        "Which company invested over ten billion dollars in OpenAI?",
        ["Microsoft"],
        2,
    ),
    Question(
        10,
        "Name two competitors of ChatGPT.",
        ["Claude", "Gemini"],
        2,
    ),

    # 3-hop / network questions
    Question(
        11,
        "Which AI labs use Nvidia H100 GPUs to train models?",
        ["OpenAI", "Anthropic", "Meta"],
        3,
    ),
    Question(
        12,
        "Which subsidiaries does Microsoft own?",
        ["GitHub", "LinkedIn"],
        2,
    ),
    Question(
        13,
        "Who founded xAI and what model does it develop?",
        ["Elon Musk", "Grok"],
        2,
    ),
    Question(
        14,
        "Elon Musk co-founded which two AI organisations?",
        ["OpenAI", "xAI"],
        2,
        note="Classic graph question: same person across two FOUNDED_BY edges.",
    ),
    Question(
        15,
        "Which company developed AlphaFold and who owns that company?",
        ["DeepMind", "Alphabet"],
        2,
    ),

    # Adversarial / specific-fact questions where Flat RAG tends to confuse
    # the right snippet with a similar one.
    Question(
        16,
        "How much did Meta pay to acquire WhatsApp?",
        ["nineteen billion"],
        1,
    ),
    Question(
        17,
        "How much did Microsoft pay to acquire LinkedIn?",
        ["twenty six billion"],
        1,
    ),
    Question(
        18,
        "Who founded Mistral AI?",
        ["Arthur Mensch", "Guillaume Lample", "Timothee Lacroix"],
        1,
    ),
    Question(
        19,
        "Which products does Apple develop?",
        ["iPhone", "iPad", "Mac"],
        1,
    ),
    Question(
        20,
        "List two AI investors in Anthropic.",
        ["Google", "Amazon"],
        1,
    ),
]


def evaluate(context: str, q: Question) -> bool:
    blob = context.lower()
    return all(frag.lower() in blob for frag in q.must_contain)
