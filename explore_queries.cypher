// =====================================================================
// Lab Day 19 — Cypher recipes for Neo4j Explore / Workspace Query
// =====================================================================
//
// Paste any of these into the Workspace Query bar (Ctrl+Enter to run),
// or use them as starting points in Explore (the "Search" pill accepts
// node labels and relationship types directly).


// ---------------------------------------------------------------------
// 0. Sanity checks
// ---------------------------------------------------------------------

// What labels and relationship types exist?
CALL db.labels() YIELD label RETURN label ORDER BY label;
CALL db.relationshipTypes() YIELD relationshipType
RETURN relationshipType ORDER BY relationshipType;

// Node counts by role (after the typed labels are applied)
MATCH (n) UNWIND labels(n) AS l RETURN l, count(*) AS n ORDER BY n DESC;


// ---------------------------------------------------------------------
// 1. The whole graph (small enough to render in one go)
// ---------------------------------------------------------------------

MATCH (n)-[r]->(m) RETURN n, r, m;


// ---------------------------------------------------------------------
// 2. Just the companies and how they connect
// ---------------------------------------------------------------------

MATCH (a:Company)-[r]->(b:Company) RETURN a, r, b;


// ---------------------------------------------------------------------
// 3. Founders of every company (Person -> Company)
// ---------------------------------------------------------------------

MATCH (p:Person)<-[:FOUNDED_BY]-(c:Company) RETURN c, p;


// ---------------------------------------------------------------------
// 4. AI labs and the LLMs they develop
// ---------------------------------------------------------------------

MATCH (c:Company)-[r:DEVELOPS]->(p:Product)
WHERE p.name IN ['GPT-4', 'GPT-4o', 'ChatGPT', 'Claude', 'Gemini',
                 'Llama', 'Grok', 'AlphaGo', 'AlphaFold']
RETURN c, r, p;


// ---------------------------------------------------------------------
// 5. The Q8 multi-hop question — visualised
//    "Who founded the company that owns DeepMind?"
//    DeepMind -> Alphabet -> Google -> founders
// ---------------------------------------------------------------------

MATCH path = (dm:Entity {name: 'DeepMind'})
              -[:SUBSIDIARY_OF|PARENT_OF*0..2]-(owner:Company)
              -[:FOUNDED_BY]->(founder:Person)
RETURN path;


// ---------------------------------------------------------------------
// 6. Investor graph — who funds whom
// ---------------------------------------------------------------------

MATCH (inv)-[r:INVESTOR_IN]->(c:Company) RETURN inv, r, c;


// ---------------------------------------------------------------------
// 7. Companies headquartered in San Francisco (Q6 on the benchmark)
// ---------------------------------------------------------------------

MATCH (c:Company)-[:HEADQUARTERED_IN|FOUNDED_AT]->(:Place {name: 'San Francisco'})
RETURN c;


// ---------------------------------------------------------------------
// 8. A 2-hop neighbourhood centred on any node
//    (replace $name with the entity you want to inspect)
// ---------------------------------------------------------------------

MATCH path = (seed:Entity {name: 'OpenAI'})-[*1..2]-(other)
RETURN path;
