"""
Categorise entity names into types so Neo4j Explore (and matplotlib) can
colour-code them. The category is inferred from:
  * The relation an entity participates in (FOUNDED_BY → tail is a Person,
    DEVELOPS → tail is a Product).
  * A small hand-written gazetteer for the corpus.
  * A few regex heuristics for years and money.

The upshot is each node gets one specific label (:Company / :Person /
:Product / :Year / :Place / :Money) in addition to the generic :Entity, and
Neo4j Explore picks distinct colours per label out of the box.
"""

from __future__ import annotations

import re
from typing import Iterable

from extraction import Triple

# Relation-to-role mapping. Direction matters: for FOUNDED_BY, the *head*
# is a Company and the *tail* is a Person.
_HEAD_ROLE = {
    "FOUNDED_BY": "Company",
    "FOUNDED_IN_YEAR": "Company",
    "FOUNDED_AT": "Company",
    "HEADQUARTERED_IN": "Company",
    "PARENT_OF": "Company",
    "SUBSIDIARY_OF": "Company",
    "ACQUIRED": "Company",
    "ACQUIRED_IN": "Company",
    "ACQUIRED_FOR": "Company",
    "INVESTOR_IN": "Company",
    "INVESTED_AMOUNT": "Company",
    "DEVELOPS": "Company",
    "PRODUCT_OF": "Product",
    "COMPETES_WITH": "Product",
    "USED_BY": "Company",
    "CEO_OF": "Person",
    "WORKED_AT": "Person",
}

_TAIL_ROLE = {
    "FOUNDED_BY": "Person",
    "FOUNDED_IN_YEAR": "Year",
    "FOUNDED_AT": "Place",
    "HEADQUARTERED_IN": "Place",
    "PARENT_OF": "Company",
    "SUBSIDIARY_OF": "Company",
    "ACQUIRED": "Company",
    "ACQUIRED_IN": "Year",
    "ACQUIRED_FOR": "Money",
    "INVESTOR_IN": "Company",
    "INVESTED_AMOUNT": "Money",
    "DEVELOPS": "Product",
    "PRODUCT_OF": "Company",
    "COMPETES_WITH": "Product",
    "USED_BY": "Product",
    "CEO_OF": "Company",
    "WORKED_AT": "Company",
}

# Tie-breaker when an entity appears under conflicting roles. Higher wins.
_PRIORITY = {
    "Person": 5,
    "Company": 4,
    "Product": 3,
    "Money": 2,
    "Place": 1,
    "Year": 0,
}

_YEAR_RE = re.compile(r"^\d{4}$")
_MONEY_RE = re.compile(r"\b(million|billion|dollars?)\b", re.I)


def _seed_role(name: str) -> str | None:
    """Cheap rules for cases the relation roles can't disambiguate."""
    if _YEAR_RE.match(name):
        return "Year"
    if _MONEY_RE.search(name):
        return "Money"
    return None


def categorise(triples: Iterable[Triple]) -> dict[str, str]:
    """Return {entity_name: role_label} for every node in the graph."""
    role: dict[str, str] = {}

    def vote(name: str, candidate: str) -> None:
        existing = role.get(name)
        if existing is None or _PRIORITY[candidate] > _PRIORITY[existing]:
            role[name] = candidate

    for t in triples:
        seed_h = _seed_role(t.head)
        if seed_h:
            vote(t.head, seed_h)
        elif t.relation in _HEAD_ROLE:
            vote(t.head, _HEAD_ROLE[t.relation])

        seed_t = _seed_role(t.tail)
        if seed_t:
            vote(t.tail, seed_t)
        elif t.relation in _TAIL_ROLE:
            vote(t.tail, _TAIL_ROLE[t.relation])

    return role
