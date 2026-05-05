"""
Render the knowledge graph to a PNG (Deliverable #2).

Always uses NetworkX for layout — even if the live store is Neo4j, the
visualisation is generated from the same triples we just upserted, so the
image faithfully represents what's in the graph database.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx

from extraction import Triple


def render(triples: Iterable[Triple], out_path: str) -> str:
    G = nx.MultiDiGraph()
    for t in triples:
        G.add_edge(t.head, t.tail, label=t.relation)

    plt.figure(figsize=(20, 14))
    pos = nx.spring_layout(G, seed=42, k=0.9, iterations=80)

    # Colour nodes by an inferred role so the picture is readable.
    company_keywords = {
        "OpenAI", "Google", "Microsoft", "Apple", "Meta", "Anthropic",
        "Nvidia", "xAI", "Mistral AI", "DeepMind", "GitHub", "LinkedIn",
        "Instagram", "WhatsApp", "Alphabet",
    }
    products = {
        "GPT-4", "GPT-4o", "ChatGPT", "Claude", "Gemini", "Llama", "Grok",
        "AlphaGo", "AlphaFold", "iPhone", "iPad", "Mac", "Apple Intelligence",
        "H100", "A100", "GitHub Copilot", "Nvidia H100",
    }
    colours = []
    for n in G.nodes:
        if n in company_keywords:
            colours.append("#4F8EF7")
        elif n in products:
            colours.append("#F7B84F")
        elif n.isdigit() and len(n) == 4:
            colours.append("#9AA0A6")
        else:
            colours.append("#7BC47F")

    nx.draw_networkx_nodes(G, pos, node_color=colours, node_size=1400, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")
    nx.draw_networkx_edges(
        G, pos, edge_color="#888", arrows=True, arrowsize=12,
        connectionstyle="arc3,rad=0.08", alpha=0.6,
    )
    edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_size=6, alpha=0.75,
    )

    plt.title("Tech Company Knowledge Graph (GraphRAG Lab Day 19)", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path
