"""NL2SQL planning agent: natural-language question -> validated query SPEC.

Crucially, the agent NEVER emits SQL. It emits a small spec (metric + certified
dimensions + filters). The semantic compiler turns that into SQL. So the model
cannot invent a join or a metric formula — it can only choose from the certified
vocabulary, and the spec is validated against the semantic layer before execution.

Two modes:
  - openai: gpt-4o-mini with JSON output, grounded in retrieved metadata.
  - mock:   deterministic rule-based planner so the demo/evals run offline.
"""

from app import llm
from app.semantic import SpecError

_DIM_KEYWORDS = {
    "region": ["region", "country", "countries", "regions"],
    "city": ["city", "cities"],
    "property_name": ["property", "properties", "hotel", "hotels"],
    "room_type": ["room type", "room types", "roomtype", "by type"],
    "month": ["month", "monthly", "over time", "trend", "by month", "each month",
              "per month", "seasonal", "season"],
}


def plan(question: str, sl, retrieved: list[dict], vocab: dict) -> dict:
    spec = _plan_llm(question, sl, retrieved, vocab) if llm.live() \
        else _plan_mock(question, sl, vocab)
    # Guardrail: reject anything outside the certified semantic layer.
    sl.validate(spec)
    return spec


# ── OpenAI planner ───────────────────────────────────────────────────────────
def _plan_llm(question: str, sl, retrieved: list[dict], vocab: dict) -> dict:
    from app.retrieval import MetadataIndex  # for type only
    metrics = ", ".join(sl.metrics.keys())
    dims = ", ".join(sl.dimensions.keys())
    vocab_lines = "\n".join(f"  {k}: {v}" for k, v in vocab.items())
    context = "\n".join(f"  - {d['name']} ({d['kind']})" for d in retrieved)

    system = (
        "You are a careful analytics planner for a governed semantic layer. "
        "You translate a business question into a STRICT JSON query spec. "
        "You may ONLY use the certified metrics and dimensions provided. "
        "Never invent metrics, columns, tables, or joins. Return JSON only."
    )
    user = f"""Certified metrics: {metrics}
Certified dimensions: {dims}

Allowed filter values (use exact spelling):
{vocab_lines}

Retrieved as most relevant to this question:
{context}

Return a JSON object with this shape:
{{
  "metric": "<one certified metric>",
  "dimensions": ["<certified dimension>", ...],   // [] if none
  "filters": [{{"field":"<certified dimension>","op":"=","value":"<value>"}}],  // [] if none
  "limit": <int, default 20>
}}

Rules:
- "by X" / "per X" / "for each X" => X is a dimension.
- A specific value (e.g. "in Egypt", "for suites", "in Dubai") => a filter, not a dimension.
- occupancy_rate cannot use the room_type dimension.
- Use op "=" for a single value, "in" for a list, "between" [lo,hi] for a month range.

Question: "{question}"
JSON:"""

    try:
        spec = llm.chat_json(system, user)
    except Exception:
        return _plan_mock(question, sl, vocab)
    # normalise
    spec.setdefault("dimensions", [])
    spec.setdefault("filters", [])
    spec.setdefault("limit", 20)
    return spec


# ── offline rule-based planner ───────────────────────────────────────────────
def _plan_mock(question: str, sl, vocab: dict) -> dict:
    q = question.lower()

    # 1) metric: best synonym/name match, else revenue
    metric = "revenue"
    best = 0
    for name, m in sl.metrics.items():
        cands = [name] + m.get("synonyms", [])
        score = max((len(c) for c in cands if c.lower() in q), default=0)
        if score > best:
            best, metric = score, name

    # 2) filters: any known dimension value mentioned
    filters = []
    filtered_dims = set()
    for dim, values in vocab.items():
        for v in values:
            if _mentions(q, str(v).lower()):
                filters.append({"field": dim, "op": "=", "value": v})
                filtered_dims.add(dim)
                break

    # 3) dimensions: keyword-driven "by X"; skip a dim already pinned by a filter
    dimensions = []
    for dim, kws in _DIM_KEYWORDS.items():
        if dim in filtered_dims:
            continue
        if any(kw in q for kw in kws):
            dimensions.append(dim)

    # NOTE: we deliberately do NOT silently drop an illegal grain here
    # (e.g. occupancy_rate by room_type). We let the semantic layer's guardrail
    # reject it with an explanation — governed systems surface the reason, they
    # don't quietly return a different number than the user asked for.

    # "top N" hint
    limit = 20
    for tok in q.split():
        if tok.isdigit():
            limit = int(tok)
            break

    return {"metric": metric, "dimensions": dimensions,
            "filters": filters, "limit": limit}


def _mentions(haystack: str, needle: str) -> bool:
    # word-ish boundary check (with optional plural) so "uae" doesn't match
    # inside another word, but "suites" still matches the value "Suite".
    import re
    return re.search(
        rf"(?<![a-z]){re.escape(needle)}s?(?![a-z])", haystack
    ) is not None
