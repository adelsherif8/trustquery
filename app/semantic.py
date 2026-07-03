"""Semantic layer loader + SQL compiler.

A query SPEC is a small, validated object:
    {"metric": "revenue", "dimensions": ["region"], "filters": [{"field":"region","op":"=","value":"Egypt"}], "limit": 20}

The compiler turns a spec into SQL using ONLY definitions from semantic_layer.yaml.
Because the LLM can only choose a metric name + certified dimension names + filter
fields (never raw tables or joins), it is structurally impossible for the model to
pick a wrong join path and inflate a number. That is the whole point.
"""

import os
import yaml

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SpecError(ValueError):
    """Raised when a spec references something not in the semantic layer."""


class SemanticLayer:
    def __init__(self, path: str | None = None):
        path = path or os.path.join(_HERE, "semantic_layer.yaml")
        with open(path) as f:
            self.model = yaml.safe_load(f)
        self.metrics = self.model["metrics"]
        self.dimensions = self.model["dimensions"]
        self.glossary = self.model.get("glossary", {})
        self.entities = self.model["entities"]

    # ── column resolution ────────────────────────────────────────────────
    def _dim_sql(self, dim: str) -> tuple[str, bool]:
        """Return (sql_expression, needs_properties_join)."""
        d = self.dimensions[dim]
        needs_join = d["needs_join"] == "properties"
        if needs_join:
            col = d["source"].split(".")[1]
            return f"p.{col}", True
        col = d["source"].split(".")[1]
        return f"b.{col}", False

    def _filter_sql(self, f: dict) -> tuple[str, bool]:
        """Return (sql_condition, needs_properties_join). Values are literal-safe."""
        field = f["field"]
        if field not in self.dimensions:
            raise SpecError(f"Unknown filter field '{field}' (not in semantic layer)")
        expr, needs_join = self._dim_sql(field)
        op = f.get("op", "=").lower()
        val = f["value"]
        if op in ("=", "!=", ">", "<", ">=", "<="):
            return f"{expr} {op} {_lit(val)}", needs_join
        if op == "in":
            vals = val if isinstance(val, list) else [val]
            joined = ", ".join(_lit(v) for v in vals)
            return f"{expr} IN ({joined})", needs_join
        if op == "between":
            lo, hi = val
            return f"{expr} BETWEEN {_lit(lo)} AND {_lit(hi)}", needs_join
        raise SpecError(f"Unsupported operator '{op}'")

    # ── validation ───────────────────────────────────────────────────────
    def validate(self, spec: dict) -> None:
        metric = spec.get("metric")
        if metric not in self.metrics:
            raise SpecError(
                f"Unknown metric '{metric}'. Certified metrics: {list(self.metrics)}"
            )
        mdef = self.metrics[metric]
        allowed = mdef.get("allowed_dimensions")
        for dim in spec.get("dimensions", []):
            if dim not in self.dimensions:
                raise SpecError(f"Unknown dimension '{dim}'")
            if allowed is not None and dim not in allowed:
                raise SpecError(
                    f"Dimension '{dim}' is not allowed for metric '{metric}' "
                    f"(allowed: {allowed}). This prevents an invalid grain."
                )
        for f in spec.get("filters", []):
            self._filter_sql(f)  # raises if bad

    # ── compilation ──────────────────────────────────────────────────────
    def compile(self, spec: dict) -> str:
        self.validate(spec)
        metric = spec["metric"]
        mdef = self.metrics[metric]
        kind = mdef["kind"]
        dims = spec.get("dimensions", [])
        filters = spec.get("filters", [])
        limit = spec.get("limit", 50)

        if kind == "occupancy":
            return self._compile_occupancy(spec, dims, filters, limit)
        return self._compile_standard(spec, mdef, metric, dims, filters, limit)

    def _compile_standard(self, spec, mdef, metric, dims, filters, limit) -> str:
        kind = mdef["kind"]
        if kind == "additive":
            metric_expr = f"{mdef['sql']} AS {metric}"
        elif kind == "ratio":
            metric_expr = (
                f"ROUND({mdef['numerator']} * 1.0 / "
                f"NULLIF({mdef['denominator']}, 0), 2) AS {metric}"
            )
        else:
            raise SpecError(f"Unhandled metric kind '{kind}'")

        dim_exprs, needs_join = [], False
        for d in dims:
            expr, nj = self._dim_sql(d)
            dim_exprs.append(f"{expr} AS {d}")
            needs_join = needs_join or nj

        where = ["b.status = 'confirmed'"]
        for f in filters:
            cond, nj = self._filter_sql(f)
            where.append(cond)
            needs_join = needs_join or nj

        select_cols = dim_exprs + [metric_expr]
        sql = f"SELECT {', '.join(select_cols)}\nFROM analytics.bookings b"
        if needs_join:
            sql += ("\nJOIN analytics.properties p "
                    "ON b.property_id = p.property_id")
        sql += "\nWHERE " + " AND ".join(where)
        if dims:
            group_exprs = [self._dim_sql(d)[0] for d in dims]
            sql += "\nGROUP BY " + ", ".join(group_exprs)
            sql += f"\nORDER BY {metric} DESC"
        sql += f"\nLIMIT {int(limit)};"
        return sql

    def _compile_occupancy(self, spec, dims, filters, limit) -> str:
        # Split filters: bookings-level (month) vs property-level (name/city/region).
        booking_filters, prop_filters = [], []
        for f in filters:
            if f["field"] == "month":
                booking_filters.append(f)
            else:
                prop_filters.append(f)

        sold_where = ["status = 'confirmed'"]
        for f in booking_filters:
            expr = "check_in_month"
            op = f.get("op", "=").lower()
            if op == "between":
                lo, hi = f["value"]
                sold_where.append(f"{expr} BETWEEN {_lit(lo)} AND {_lit(hi)}")
            elif op == "in":
                vals = f["value"] if isinstance(f["value"], list) else [f["value"]]
                sold_where.append(f"{expr} IN ({', '.join(_lit(v) for v in vals)})")
            else:
                sold_where.append(f"{expr} {op} {_lit(f['value'])}")

        joined_where = []
        for f in prop_filters:
            col = self.dimensions[f["field"]]["source"].split(".")[1]
            op = f.get("op", "=").lower()
            joined_where.append(f"p.{col} {op} {_lit(f['value'])}")

        dim_exprs, group_exprs = [], []
        for d in dims:
            col = self.dimensions[d]["source"].split(".")[1]
            src = "j.month" if d == "month" else f"j.{col}"
            dim_exprs.append(f"{src} AS {d}")
            group_exprs.append(src)

        sql = f"""WITH sold AS (
  SELECT property_id, check_in_month AS month, SUM(nights) AS sold_nights
  FROM analytics.bookings
  WHERE {' AND '.join(sold_where)}
  GROUP BY property_id, check_in_month
),
j AS (
  SELECT p.property_name, p.city, p.region, s.month,
         s.sold_nights, i.available_room_nights
  FROM sold s
  JOIN analytics.inventory i
    ON s.property_id = i.property_id AND s.month = i.month
  JOIN analytics.properties p
    ON s.property_id = p.property_id
  {('WHERE ' + ' AND '.join(joined_where)) if joined_where else ''}
)
SELECT {', '.join(dim_exprs + ['ROUND(SUM(sold_nights) * 1.0 / NULLIF(SUM(available_room_nights),0), 4) AS occupancy_rate'])}
FROM j
{('GROUP BY ' + ', '.join(group_exprs)) if group_exprs else ''}
{('ORDER BY occupancy_rate DESC') if dims else ''}
LIMIT {int(limit)};"""
        return sql


def _lit(v) -> str:
    """Render a safe SQL literal. Strings are single-quoted + escaped."""
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v).replace("'", "''")
    return f"'{s}'"
