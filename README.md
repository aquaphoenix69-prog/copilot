# Customer-Care AI Co-pilot

Real-time co-pilot for human customer-care agents on live calls.
Pipeline matches the project diagram:

```
Live call → Parakeet STT → Transcript
                          ├─→ DeBERTa Interpreter (intent + customer tag)
                          └─→ CardiffNLP Sentiment
                                          ↓
                                       State
                                          ↓
                              Neo4j Planner Agent
                                          ↓
                            vLLM (local) → Coaching reply
                                          ↑
                            (fallback path: skip planner if cold state)
```

## Stack

- **STT**: `nvidia/parakeet-tdt-0.6b-v2` (NeMo)
- **Interpreter**: `microsoft/deberta-v3-base` (zero-shot via NLI by default; fine-tunable)
- **Sentiment**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Graph DB**: Neo4j 5 (Cypher)
- **LLM**: vLLM (OpenAI-compatible) — default `meta-llama/Llama-3.1-8B-Instruct`
- **API**: FastAPI (REST + WebSocket)

## Setup

```bash
cp .env.example .env
python -m venv .venv && source .venv/Scripts/activate   # on Windows bash
pip install -r requirements.txt

# Start Neo4j + vLLM
docker compose up -d neo4j
docker compose up -d vllm   # requires NVIDIA GPU + HF_TOKEN for gated models

# Ingest the dataset into Neo4j (one-time)
python scripts/ingest_to_neo4j.py

# Run the API
uvicorn app.main:app --host 0.0.0.0 --port 9000 --reload
```

The dataset path defaults to `C:/Users/aquap/Downloads/final_master_dataset_complete_final.json` — override via `DATASET_PATH` in `.env`.

## Usage

### Text turn

```bash
curl -X POST http://localhost:9000/turn \
  -H "Content-Type: application/json" \
  -d '{"conversation_id":"demo-1","text":"hi i have a double booking and my name is wrong on the reservation"}'
```

### Audio turn

```bash
curl -X POST http://localhost:9000/audio \
  -F conversation_id=demo-1 \
  -F file=@sample.wav
```

### WebSocket (streaming text turns per chunk)

`ws://localhost:9000/ws/<conversation_id>` — send `{"text": "..."}` per chunk, receive `CopilotResponse` JSON.

## Why no 400 errors anymore

The 19k-record dataset never travels through any API request. It is loaded **once** by `scripts/ingest_to_neo4j.py` and projected into Neo4j. At runtime the planner asks Cypher for the top-k agent_tags conditioned on the current `(intent, customer_tag, sent_bucket)` state — typically a handful of nodes, kilobytes of context, well under any LLM input limit.

## Graph model

```
(:State {key, intent, customer_tag, sent_bucket})
   -[:HAS_INTENT]-> (:Intent)
   -[:HAS_CUSTOMER_TAG]-> (:CustomerTag)
   -[:USED_AGENT_TAG {count}]-> (:AgentTag)
   -[:LED_TO_OUTCOME {count}]-> (:Outcome)
   -[:NEXT {count}]-> (:State)
```

Sentiment buckets: `neg < -0.33`, `pos > 0.33`, else `neu`.

## Fallback mechanism

If the planner finds no graph match for the current state (cold node), it backs off:
`(intent, customer_tag, sentiment)` → `(intent, customer_tag)` → `(intent)` → no-graph fallback. When the LLM call fails or planner returns nothing, the API still returns a degraded reply so the human agent never gets a blank screen.

## Project layout

```
copilot/
├── app/
│   ├── main.py                  # FastAPI app
│   ├── core/
│   │   ├── config.py            # Settings (env-driven)
│   │   └── schemas.py           # Pydantic models
│   ├── graph/driver.py          # Neo4j driver singleton
│   └── agents/
│       ├── stt.py               # Parakeet
│       ├── interpreter.py       # DeBERTa intent + customer tag
│       ├── sentiment.py         # CardiffNLP RoBERTa
│       ├── state.py             # State aggregator
│       ├── planner.py           # Neo4j-backed planner
│       └── llm.py               # vLLM client
├── scripts/
│   └── ingest_to_neo4j.py       # JSON → graph (one-time)
├── requirements.txt
├── docker-compose.yml
└── .env.example
```
