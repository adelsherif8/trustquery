"""TrustQuery API — governed NL-to-SQL analytics agent."""

import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

import duckdb
from app.warehouse import build as build_warehouse
from app.semantic import SemanticLayer, SpecError
from app.retrieval import MetadataIndex
from app import agent as agent_mod
from app import evals as evals_mod
from app import llm

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(_HERE, "templates"))

# Prefer a prebuilt read-only DuckDB file (fast serverless cold start).
# Fall back to an in-memory build for local dev if the file is absent.
_DB_FILE = os.path.join(_HERE, "data", "warehouse.duckdb")
if os.path.exists(_DB_FILE):
    _con = duckdb.connect(_DB_FILE, read_only=True)
else:
    _con = build_warehouse()
_sl = SemanticLayer()
_idx = MetadataIndex(_sl)


def build_vocab(con) -> dict:
    return {
        "region": [r[0] for r in con.execute(
            "SELECT DISTINCT region FROM analytics.properties ORDER BY 1").fetchall()],
        "city": [r[0] for r in con.execute(
            "SELECT DISTINCT city FROM analytics.properties ORDER BY 1").fetchall()],
        "property_name": [r[0] for r in con.execute(
            "SELECT DISTINCT property_name FROM analytics.properties ORDER BY 1").fetchall()],
        "room_type": [r[0] for r in con.execute(
            "SELECT DISTINCT room_type FROM analytics.bookings ORDER BY 1").fetchall()],
        "month": [r[0] for r in con.execute(
            "SELECT DISTINCT check_in_month FROM analytics.bookings ORDER BY 1").fetchall()],
    }


_vocab = build_vocab(_con)

SAMPLES = [
    "Total revenue by region",
    "Occupancy rate for each city",
    "ADR by room type",
    "How much revenue did we make in Egypt?",
    "Occupancy in Dubai by month",
    "How many bookings per region?",
]

app = FastAPI(title="TrustQuery")


def _fmt(value, fmt: str) -> str:
    if value is None:
        return "—"
    if fmt == "money":
        return f"${value:,.0f}" if value >= 1000 else f"${value:,.2f}"
    if fmt == "percent":
        return f"{value * 100:.1f}%"
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"


def _answer_sentence(spec, mdef, rows) -> str:
    label = mdef.get("label", spec["metric"])
    fmt = mdef.get("format", "number")
    if not rows:
        return "No data matched that question."
    if not spec.get("dimensions"):
        return f"{label}: {_fmt(rows[0][-1], fmt)}"
    top = rows[0]
    return (f"Top {spec['dimensions'][0]} by {label}: "
            f"{top[0]} at {_fmt(top[-1], fmt)}.")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "mode": llm.mode(),
        "metrics": [
            {"name": n, "label": m.get("label", n), "doc": m.get("doc", "")}
            for n, m in _sl.metrics.items()
        ],
        "dimensions": list(_sl.dimensions.keys()),
        "samples": SAMPLES,
    })


@app.post("/ask")
async def ask(request: Request):
    body = await request.json()
    question = (body.get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "empty question"}, status_code=400)

    retrieved = _idx.retrieve(question, k=6)
    try:
        spec = agent_mod.plan(question, _sl, retrieved, _vocab)
    except SpecError as e:
        return JSONResponse({
            "mode": llm.mode(),
            "question": question,
            "retrieved": retrieved,
            "error": f"Guardrail blocked this: {e}",
        }, status_code=200)

    mdef = _sl.metrics[spec["metric"]]
    fmt = mdef.get("format", "number")
    sql = _sl.compile(spec)
    cur = _con.execute(sql)
    columns = [d[0] for d in cur.description]
    rows = cur.fetchall()
    disp_rows = [
        [(_fmt(v, fmt) if i == len(r) - 1 else v) for i, v in enumerate(r)]
        for r in rows[:50]
    ]

    return {
        "mode": llm.mode(),
        "question": question,
        "spec": spec,
        "sql": sql,
        "retrieved": retrieved,
        "context": _idx.context_block(retrieved),
        "columns": columns,
        "rows": disp_rows,
        "row_count": len(rows),
        "answer": _answer_sentence(spec, mdef, rows),
    }


@app.get("/evals")
def run_evals():
    return evals_mod.run(_con, _sl, _idx, _vocab)


@app.get("/health")
def health():
    return {"ok": True, "mode": llm.mode()}
