"""
Flat (vector) RAG baseline.

Uses TF-IDF cosine similarity over the corpus. We deliberately avoid an
embedding API call here so the comparison is reproducible — the goal of the
benchmark in `benchmark.py` is to show *retrieval* differences (graph
traversal vs. surface-form matching), and TF-IDF makes the failure modes of
flat retrieval visible without a network dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Retrieval:
    doc_id: str
    text: str
    score: float


class FlatRAG:
    def __init__(self, documents):
        self.documents = list(documents)
        self.vectoriser = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        self.matrix = self.vectoriser.fit_transform(
            d["text"] for d in self.documents
        )

    def retrieve(self, query: str, k: int = 3) -> List[Retrieval]:
        q_vec = self.vectoriser.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        top = sims.argsort()[::-1][:k]
        return [
            Retrieval(
                doc_id=self.documents[i]["id"],
                text=self.documents[i]["text"],
                score=float(sims[i]),
            )
            for i in top
        ]

    def context(self, query: str, k: int = 3) -> str:
        hits = self.retrieve(query, k=k)
        return "\n\n".join(f"[{h.doc_id}] {h.text}" for h in hits)
