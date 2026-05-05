"""
Graph construction and querying.

Two storage backends:
  * Neo4jStore  — production-style, uses the bolt driver and Cypher queries
                  for k-hop traversal. This is the recommended backend per
                  the lab brief (`*Use neo4j`).
  * NetworkxStore — in-memory MultiDiGraph used both as a fallback when
                  Neo4j is unreachable and as the source for the matplotlib
                  visualisation deliverable.

Both stores expose the same surface (`upsert_triples`, `find_node`,
`k_hop_subgraph`, `close`) so `graph_rag.py` doesn't care which is wired up.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional

import networkx as nx

from extraction import Triple

log = logging.getLogger(__name__)


@dataclass
class SubgraphFact:
    """A flattened triple plus a hop distance from the seed entity, used for
    textualisation in the GraphRAG retriever."""

    head: str
    relation: str
    tail: str
    hop: int
    source_doc: str = ""


# ---------------------------------------------------------------------------
# NetworkX backend
# ---------------------------------------------------------------------------


class NetworkxStore:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def upsert_triples(self, triples: Iterable[Triple]) -> None:
        for t in triples:
            self.graph.add_node(t.head)
            self.graph.add_node(t.tail)
            self.graph.add_edge(
                t.head,
                t.tail,
                key=t.relation,
                relation=t.relation,
                source_doc=t.source_doc,
            )

    def find_node(self, name: str) -> Optional[str]:
        if name in self.graph:
            return name
        target = name.lower()
        for n in self.graph.nodes:
            if n.lower() == target:
                return n
        for n in self.graph.nodes:
            if target in n.lower():
                return n
        return None

    def k_hop_subgraph(self, seed: str, k: int = 2) -> List[SubgraphFact]:
        node = self.find_node(seed)
        if node is None:
            return []
        # Undirected expansion of the neighbourhood — relationships in this
        # corpus are semantically directional but BOTH directions are
        # informative when answering questions (e.g. "who founded X" needs
        # the inbound `FOUNDED_BY` edge).
        undirected = self.graph.to_undirected(as_view=False)
        layers = {0: {node}}
        seen = {node}
        for hop in range(1, k + 1):
            frontier = set()
            for u in layers[hop - 1]:
                for v in undirected.neighbors(u):
                    if v not in seen:
                        frontier.add(v)
                        seen.add(v)
            layers[hop] = frontier

        # Distance lookup so each fact is tagged with its closest hop.
        dist = {}
        for hop, nodes in layers.items():
            for n in nodes:
                dist.setdefault(n, hop)

        facts: List[SubgraphFact] = []
        for u in seen:
            # outgoing
            for v, edges in self.graph[u].items():
                if v not in seen:
                    continue
                for _, attrs in edges.items():
                    facts.append(
                        SubgraphFact(
                            head=u,
                            relation=attrs["relation"],
                            tail=v,
                            hop=min(dist[u], dist[v]),
                            source_doc=attrs.get("source_doc", ""),
                        )
                    )
        # Stable, hop-major ordering so smaller hops appear first in the
        # textualised context (most-relevant facts go to the LLM first).
        facts.sort(key=lambda f: (f.hop, f.head, f.relation, f.tail))
        return facts

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Neo4j backend
# ---------------------------------------------------------------------------


class Neo4jStore:
    def __init__(self, uri: str, user: str, password: str):
        from neo4j import GraphDatabase  # imported lazily so offline runs work

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        # Sanity check the connection up-front instead of failing inside a tx.
        with self.driver.session() as s:
            s.run("RETURN 1").consume()

    def reset(self) -> None:
        with self.driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n").consume()

    def upsert_triples(self, triples: Iterable[Triple]) -> None:
        # Cypher cannot parameterise relationship types or labels, so we
        # group by relation/role and template per group. Each entity gets
        # both the generic :Entity label (for Cypher uniformity and the
        # uniqueness constraint) and a specific role label (:Company,
        # :Person, :Product, :Year, :Place, :Money) so Neo4j Explore can
        # auto-colour by node type.
        from collections import defaultdict

        from entity_types import categorise

        triples = list(triples)
        roles = categorise(triples)

        by_rel: dict[str, list[Triple]] = defaultdict(list)
        for t in triples:
            by_rel[t.relation].append(t)

        with self.driver.session() as s:
            s.run(
                "CREATE CONSTRAINT entity_name IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            ).consume()

            # 1. Upsert nodes with their typed label.
            by_role: dict[str, list[str]] = defaultdict(list)
            for name, role in roles.items():
                by_role[role].append(name)
            for role, names in by_role.items():
                # Adding a label to an already-existing node is idempotent.
                s.run(
                    f"UNWIND $names AS n "
                    f"MERGE (e:Entity {{name: n}}) "
                    f"SET e:{role}",
                    names=names,
                ).consume()

            # 2. Upsert relationships.
            for rel, group in by_rel.items():
                rows = [
                    {
                        "head": t.head,
                        "tail": t.tail,
                        "source_doc": t.source_doc,
                    }
                    for t in group
                ]
                cypher = (
                    "UNWIND $rows AS row "
                    "MERGE (h:Entity {name: row.head}) "
                    "MERGE (t:Entity {name: row.tail}) "
                    f"MERGE (h)-[r:{rel}]->(t) "
                    "SET r.source_doc = row.source_doc"
                )
                s.run(cypher, rows=rows).consume()

    def find_node(self, name: str) -> Optional[str]:
        with self.driver.session() as s:
            row = s.run(
                "MATCH (n:Entity) "
                "WHERE toLower(n.name) = toLower($name) "
                "RETURN n.name AS name LIMIT 1",
                name=name,
            ).single()
            if row:
                return row["name"]
            row = s.run(
                "MATCH (n:Entity) "
                "WHERE toLower(n.name) CONTAINS toLower($name) "
                "RETURN n.name AS name LIMIT 1",
                name=name,
            ).single()
            return row["name"] if row else None

    def k_hop_subgraph(self, seed: str, k: int = 2) -> List[SubgraphFact]:
        node = self.find_node(seed)
        if node is None:
            return []
        cypher = (
            "MATCH path = (seed:Entity {name: $name})-[*1..$k]-(other) "
            "UNWIND relationships(path) AS r "
            "WITH DISTINCT startNode(r) AS h, type(r) AS rel, endNode(r) AS t, "
            "  r.source_doc AS source_doc "
            "RETURN h.name AS head, rel AS relation, t.name AS tail, source_doc"
        )
        with self.driver.session() as s:
            rows = s.run(cypher.replace("$k", str(int(k))), name=node).data()
        # Recompute hop distance with a quick BFS over the returned facts,
        # because Cypher's variable-length match doesn't natively expose the
        # min hop per edge.
        adjacency: dict[str, set[str]] = {}
        for r in rows:
            adjacency.setdefault(r["head"], set()).add(r["tail"])
            adjacency.setdefault(r["tail"], set()).add(r["head"])
        dist = {node: 0}
        frontier = {node}
        for hop in range(1, k + 1):
            nxt = set()
            for u in frontier:
                for v in adjacency.get(u, ()):
                    if v not in dist:
                        dist[v] = hop
                        nxt.add(v)
            frontier = nxt

        facts = [
            SubgraphFact(
                head=r["head"],
                relation=r["relation"],
                tail=r["tail"],
                hop=min(dist.get(r["head"], k), dist.get(r["tail"], k)),
                source_doc=r.get("source_doc") or "",
            )
            for r in rows
        ]
        facts.sort(key=lambda f: (f.hop, f.head, f.relation, f.tail))
        return facts

    def close(self) -> None:
        self.driver.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def open_store(prefer_neo4j: bool, neo4j_cfg: dict) -> tuple[object, str]:
    """Try Neo4j first when requested; fall back to NetworkX so the lab still
    runs end-to-end on a machine with no database."""
    if prefer_neo4j:
        try:
            store = Neo4jStore(
                neo4j_cfg["uri"],
                neo4j_cfg["user"],
                neo4j_cfg["password"],
            )
            return store, "neo4j"
        except Exception as exc:
            log.warning("Neo4j unavailable (%s) — falling back to NetworkX.", exc)
    return NetworkxStore(), "networkx"
