# TrustQuery — Governed AI Data Analyst (NL → SQL over a semantic layer)

Ask a data warehouse in plain English and get **governed answers you can trust** —
the same correct, reconciled number every time.

Generating SQL with an LLM is easy. Returning a number a CFO will actually trust
is the hard part. TrustQuery solves that with a governed semantic layer, metadata
retrieval, a spec compiler that makes bad joins structurally impossible, and an
eval/reconciliation suite.

**Live demo:** _(deployed on Vercel)_

## How it works

```
question ─▶ metadata-RAG retrieval ─▶ NL2SQL planner ─▶ query SPEC
                                                          │  (validated against
                                                          ▼   the semantic layer)
                              semantic compiler ─▶ SQL ─▶ DuckDB ─▶ answer
                                                          ▲
                              golden evals reconcile every answer against
                              an independent trusted-SQL report
```

### 1. Governed semantic layer (`semantic_layer.yaml`)
Every metric has **exactly one** definition; every dimension maps to one column;
joins are **certified** in one place. `Revenue`, `Occupancy`, and `ADR` mean the
same thing to everyone.

### 2. Metadata-RAG retrieval (`app/retrieval.py`)
Metrics, dimensions, and the business glossary are embedded and retrieved per
question (OpenAI embeddings, cosine). Only the relevant metadata enters the
model's context — irrelevant schema is kept out.

### 3. NL2SQL planner (`app/agent.py`)
The LLM never writes SQL. It emits a small **query spec** (metric + certified
dimensions + filters). Grounded in retrieved metadata, validated before running.

### 4. Semantic compiler + guardrail (`app/semantic.py`)
The compiler turns the spec into SQL using **only** certified metrics and join
paths. The model cannot pick a wrong join and inflate a number (fan-out). Illegal
grains (e.g. occupancy by room type) are rejected with an explanation, not
silently "fixed".

### 5. Eval-driven development (`app/evals.py`, `golden.yaml`)
Each golden question is reconciled against an **independent hand-written trusted
SQL report**. Regression-tested, with plan-accuracy checks. Run `/evals` or the
button in the UI.

## Modes
- **OpenAI mode** — set `OPENAI_API_KEY`. Real `text-embedding-3-small` retrieval
  and `gpt-4o-mini` planning.
- **Offline demo mode** — no key required; a deterministic planner keeps the demo
  and evals fully functional. The UI shows which mode is live.

## Run locally
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pandas
python -c "from app.warehouse import materialize; materialize('data/warehouse.duckdb')"
uvicorn app.main:app --reload
# open http://localhost:8000
```

## Stack
Python · FastAPI · DuckDB · OpenAI (embeddings + gpt-4o-mini) · YAML semantic layer

Built by **Adel Atya** — adelatya.vercel.app
