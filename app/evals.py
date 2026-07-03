"""Eval-driven development harness.

For every golden question we:
  1. run the full agent pipeline (retrieve -> plan -> compile -> execute),
  2. run the INDEPENDENT trusted_sql,
  3. reconcile the two numbers within tolerance,
  4. check the plan matched the expected metric/dimensions.

This is the reconciliation + regression suite the client asks for. When live,
it can also run an LLM-as-judge faithfulness check on a natural-language answer.
"""

import os
import yaml

from app.semantic import SemanticLayer
from app.retrieval import MetadataIndex
from app import agent as agent_mod

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = 0.01  # 1% reconciliation tolerance


def _first_number(rows):
    """Pull the headline number: first metric value of the first row."""
    if not rows:
        return None
    row = rows[0]
    return float(row[-1]) if row[-1] is not None else None


def run(con, sl: SemanticLayer, idx: MetadataIndex, vocab: dict,
        golden_path: str | None = None) -> dict:
    golden_path = golden_path or os.path.join(_HERE, "golden.yaml")
    with open(golden_path) as f:
        cases = yaml.safe_load(f)

    results = []
    passed = 0
    for case in cases:
        q = case["question"]
        row = {"question": q}
        try:
            retrieved = idx.retrieve(q, k=6)
            spec = agent_mod.plan(q, sl, retrieved, vocab)
            sql = sl.compile(spec)
            agent_val = _first_number(con.execute(sql).fetchall())
            trusted_val = _first_number(con.execute(case["trusted_sql"]).fetchall())

            metric_ok = spec["metric"] == case["expect_metric"]
            dims_ok = sorted(spec.get("dimensions", [])) == sorted(
                case.get("expect_dimensions", []))
            recon_ok = _close(agent_val, trusted_val)
            ok = metric_ok and dims_ok and recon_ok

            row.update({
                "metric": spec["metric"],
                "agent_value": agent_val,
                "trusted_value": trusted_val,
                "plan_ok": metric_ok and dims_ok,
                "reconciled": recon_ok,
                "pass": ok,
            })
        except Exception as e:  # a failed case is a failed eval, not a crash
            row.update({"pass": False, "error": str(e)})
        results.append(row)
        passed += 1 if row.get("pass") else 0

    return {
        "total": len(cases),
        "passed": passed,
        "pass_rate": round(passed / len(cases), 3) if cases else 0.0,
        "results": results,
    }


def _close(a, b) -> bool:
    if a is None or b is None:
        return a == b
    if abs(b) < 1e-9:
        return abs(a) < 1e-9
    return abs(a - b) / abs(b) <= TOL


if __name__ == "__main__":
    from app.warehouse import build
    from app.main import build_vocab  # reuse
    con = build()
    sl = SemanticLayer()
    idx = MetadataIndex(sl)
    vocab = build_vocab(con)
    report = run(con, sl, idx, vocab)
    print(f"Eval: {report['passed']}/{report['total']} "
          f"({report['pass_rate']*100:.0f}%)")
    for r in report["results"]:
        flag = "PASS" if r.get("pass") else "FAIL"
        print(f"  [{flag}] {r['question']}")
        if not r.get("pass"):
            print("        ", {k: v for k, v in r.items() if k != "question"})
