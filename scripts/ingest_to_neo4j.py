"""
Ingest the pre-call dataset into Neo4j.

Graph model:
  (:Intent {name})
  (:CustomerTag {name})
  (:AgentTag {name})
  (:Outcome {name})
  (:State {key, intent, customer_tag, sent_bucket})
    - key = "{intent}|{customer_tag}|{sent_bucket}"
    - sent_bucket in {"neg","neu","pos"} via thresholds (-0.33, 0.33)

  (:State)-[:HAS_INTENT]->(:Intent)
  (:State)-[:HAS_CUSTOMER_TAG]->(:CustomerTag)
  (:State)-[:USED_AGENT_TAG {count}]->(:AgentTag)
  (:State)-[:LED_TO_OUTCOME {count}]->(:Outcome)
  (:State)-[:NEXT {count}]->(:State)        # transition between consecutive chunks in same conversation

This lets the planner answer:
  "Given current (intent, customer_tag, sentiment), which agent_tag(s) most often
   precede a RESOLVED outcome, and which state typically follows?"
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

from neo4j import GraphDatabase
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.core.config import settings  # noqa: E402


def sent_bucket(score: float | None) -> str:
    if score is None:
        return "neu"
    if score < -0.33:
        return "neg"
    if score > 0.33:
        return "pos"
    return "neu"


def chunk_index(span: str) -> int:
    try:
        return int(str(span).split("_")[0])
    except Exception:
        return 0


def state_key(intent: str, customer_tag: str, bucket: str) -> str:
    return f"{intent}|{customer_tag}|{bucket}"


def aggregate(rows: list[dict]):
    """Aggregate rows into:
      - per-conversation ordered chunks
      - state -> agent_tag counts
      - state -> outcome counts
      - state -> next state counts
    """
    convs: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        convs[r["conversation_id"]].append(r)
    for cid in convs:
        convs[cid].sort(key=lambda r: chunk_index(r.get("chunk_span", "0")))

    state_agent_counts: dict[tuple[str, str], int] = Counter()
    state_outcome_counts: dict[tuple[str, str], int] = Counter()
    state_next_counts: dict[tuple[str, str], int] = Counter()
    state_props: dict[str, tuple[str, str, str]] = {}

    intents: set[str] = set()
    cust_tags: set[str] = set()
    agent_tags: set[str] = set()
    outcomes: set[str] = set()

    for cid, chunks in convs.items():
        prev_key: str | None = None
        for c in chunks:
            intent = c.get("intent") or "UNKNOWN"
            ctag = c.get("customer_tags") or "CUSTOMER_OTHER"
            bucket = sent_bucket(c.get("sentiment_score"))
            k = state_key(intent, ctag, bucket)
            state_props[k] = (intent, ctag, bucket)
            intents.add(intent)
            cust_tags.add(ctag)

            for a in c.get("agent_tags") or []:
                if not a:
                    continue
                agent_tags.add(a)
                state_agent_counts[(k, a)] += 1

            for o in c.get("outcome_tags") or []:
                if not o or o == "none":
                    continue
                outcomes.add(o)
                state_outcome_counts[(k, o)] += 1

            if prev_key is not None and prev_key != k:
                state_next_counts[(prev_key, k)] += 1
            prev_key = k

    return {
        "intents": intents,
        "cust_tags": cust_tags,
        "agent_tags": agent_tags,
        "outcomes": outcomes,
        "state_props": state_props,
        "state_agent_counts": state_agent_counts,
        "state_outcome_counts": state_outcome_counts,
        "state_next_counts": state_next_counts,
    }


def write_graph(driver, agg: dict, batch: int = 1000):
    with driver.session(database=settings.neo4j_database) as s:
        s.run("CREATE CONSTRAINT intent_name IF NOT EXISTS FOR (n:Intent) REQUIRE n.name IS UNIQUE")
        s.run("CREATE CONSTRAINT ctag_name IF NOT EXISTS FOR (n:CustomerTag) REQUIRE n.name IS UNIQUE")
        s.run("CREATE CONSTRAINT atag_name IF NOT EXISTS FOR (n:AgentTag) REQUIRE n.name IS UNIQUE")
        s.run("CREATE CONSTRAINT outcome_name IF NOT EXISTS FOR (n:Outcome) REQUIRE n.name IS UNIQUE")
        s.run("CREATE CONSTRAINT state_key IF NOT EXISTS FOR (n:State) REQUIRE n.key IS UNIQUE")

        s.run("UNWIND $names AS n MERGE (:Intent {name: n})", names=list(agg["intents"]))
        s.run("UNWIND $names AS n MERGE (:CustomerTag {name: n})", names=list(agg["cust_tags"]))
        s.run("UNWIND $names AS n MERGE (:AgentTag {name: n})", names=list(agg["agent_tags"]))
        s.run("UNWIND $names AS n MERGE (:Outcome {name: n})", names=list(agg["outcomes"]))

        states = [
            {"key": k, "intent": v[0], "customer_tag": v[1], "sent_bucket": v[2]}
            for k, v in agg["state_props"].items()
        ]
        for i in tqdm(range(0, len(states), batch), desc="states"):
            chunk = states[i : i + batch]
            s.run(
                """
                UNWIND $rows AS r
                MERGE (st:State {key: r.key})
                SET st.intent = r.intent,
                    st.customer_tag = r.customer_tag,
                    st.sent_bucket = r.sent_bucket
                WITH st, r
                MATCH (i:Intent {name: r.intent})
                MERGE (st)-[:HAS_INTENT]->(i)
                WITH st, r
                MATCH (c:CustomerTag {name: r.customer_tag})
                MERGE (st)-[:HAS_CUSTOMER_TAG]->(c)
                """,
                rows=chunk,
            )

        rows = [
            {"k": k, "tag": t, "count": int(n)}
            for (k, t), n in agg["state_agent_counts"].items()
        ]
        for i in tqdm(range(0, len(rows), batch), desc="state-agent"):
            s.run(
                """
                UNWIND $rows AS r
                MATCH (st:State {key: r.k}), (a:AgentTag {name: r.tag})
                MERGE (st)-[rel:USED_AGENT_TAG]->(a)
                SET rel.count = r.count
                """,
                rows=rows[i : i + batch],
            )

        rows = [
            {"k": k, "out": o, "count": int(n)}
            for (k, o), n in agg["state_outcome_counts"].items()
        ]
        for i in tqdm(range(0, len(rows), batch), desc="state-outcome"):
            s.run(
                """
                UNWIND $rows AS r
                MATCH (st:State {key: r.k}), (o:Outcome {name: r.out})
                MERGE (st)-[rel:LED_TO_OUTCOME]->(o)
                SET rel.count = r.count
                """,
                rows=rows[i : i + batch],
            )

        rows = [
            {"a": a, "b": b, "count": int(n)}
            for (a, b), n in agg["state_next_counts"].items()
        ]
        for i in tqdm(range(0, len(rows), batch), desc="state-next"):
            s.run(
                """
                UNWIND $rows AS r
                MATCH (a:State {key: r.a}), (b:State {key: r.b})
                MERGE (a)-[rel:NEXT]->(b)
                SET rel.count = r.count
                """,
                rows=rows[i : i + batch],
            )


def main():
    path = settings.dataset_path
    print(f"Loading {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data["data"] if isinstance(data, dict) and "data" in data else data
    print(f"  {len(rows)} chunks across {len({r['conversation_id'] for r in rows})} conversations")

    print("Aggregating ...")
    agg = aggregate(rows)
    print(
        f"  intents={len(agg['intents'])} cust_tags={len(agg['cust_tags'])} "
        f"agent_tags={len(agg['agent_tags'])} outcomes={len(agg['outcomes'])} "
        f"states={len(agg['state_props'])} transitions={len(agg['state_next_counts'])}"
    )

    driver = GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )
    try:
        print("Writing to Neo4j ...")
        write_graph(driver, agg)
        print("Done.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
