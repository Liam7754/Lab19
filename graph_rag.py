"""
GraphRAG retriever.

Pipeline matches Step 3 of the lab brief:
  1. Identify the seed entity in the question.
  2. Locate it in the graph (case-insensitive + substring fallback).
  3. Traverse 2 hops (configurable).
  4. Textualise the resulting facts into prose for the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from graph import SubgraphFact


# Hand-curated alias table. In a production system this would come from an
# entity linker; for the closed corpus a small lookup is enough and makes the
# benchmark deterministic.
_ALIASES = {
    "facebook": "Meta",
    "alphabet's": "Alphabet",
    "openai's": "OpenAI",
    "google's": "Google",
    "microsoft's": "Microsoft",
    "apple's": "Apple",
    "anthropic's": "Anthropic",
    "meta's": "Meta",
    "nvidia's": "Nvidia",
}

# Capitalised tokens we should NOT treat as company seeds — they're question
# words, not entities.
_STOPWORDS_TITLE = {
    "Who", "What", "When", "Where", "Why", "How", "Which", "Whose",
    "The", "A", "An", "Is", "Are", "Was", "Were", "Did", "Does", "Do",
    "Tell", "Name", "List", "AI",
}


@dataclass
class GraphRetrieval:
    seed: str
    facts: List[SubgraphFact]
    text_block: str


def _candidate_seeds(question: str) -> List[str]:
    """Pull plausible entity names out of the question.

    Strategy:
      * Match capitalised multi-word phrases (`San Francisco`, `Mistral AI`).
      * Also accept single tokens whose first letter is lowercase but contain
        an uppercase letter, so `xAI`, `iPhone` etc. survive.
      * For multi-word matches, also queue prefixes (longest-first), so
        `Nvidia H100 GPUs` falls back to `Nvidia H100` then `Nvidia`.
      * Strip trailing punctuation that the question text drops on the
        candidate (`Anthropic.` → `Anthropic`).
      * Map possessives via the alias table (`Facebook` → `Meta`).
    """
    import re

    raw = re.findall(
        r"[A-Z][\w&.]*(?:\s+[A-Z][\w&.]*)*"   # Capitalised phrases
        r"|\b[a-z][A-Z]\w*",                   # Mixed-case starting lower
        question,
    )

    seeds: list[str] = []
    seen: set[str] = set()

    def add(s: str) -> None:
        s = s.strip(" .,?!:;\"'")
        if not s or s in seen:
            return
        if s in _STOPWORDS_TITLE:
            return
        seen.add(s)
        seeds.append(s)

    for tok in raw:
        add(tok)
        # progressively shorter prefixes so over-greedy multi-word matches
        # still resolve when the entity is the first 1–2 words.
        words = tok.split()
        for i in range(len(words) - 1, 0, -1):
            add(" ".join(words[:i]))

    for word in re.findall(r"[A-Za-z]+(?:'s)?", question):
        canon = _ALIASES.get(word.lower())
        if canon:
            add(canon)

    seeds.sort(key=len, reverse=True)
    return seeds


def _format_fact(f: SubgraphFact) -> str:
    rel = f.relation.replace("_", " ").lower()
    return f"- {f.head} — {rel} — {f.tail}  (hop {f.hop})"


class GraphRAG:
    def __init__(self, store, hops: int = 2):
        self.store = store
        self.hops = hops

    def retrieve(self, question: str) -> Optional[GraphRetrieval]:
        for seed in _candidate_seeds(question):
            resolved = self.store.find_node(seed)
            if resolved:
                facts = self.store.k_hop_subgraph(resolved, k=self.hops)
                if facts:
                    text_block = self._textualise(resolved, facts)
                    return GraphRetrieval(
                        seed=resolved, facts=facts, text_block=text_block
                    )
        return None

    def _textualise(self, seed: str, facts: List[SubgraphFact]) -> str:
        header = f"Knowledge subgraph centred on {seed} (up to {self.hops} hops):"
        body = "\n".join(_format_fact(f) for f in facts)
        return f"{header}\n{body}"

    def context(self, question: str) -> str:
        result = self.retrieve(question)
        return result.text_block if result else "(no entity match in graph)"
