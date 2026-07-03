"""Metadata RAG over the semantic layer.

We embed every certified metric, dimension, and glossary term (plus their synonyms
and docs) into a small in-memory index. At question time we retrieve the top-k most
relevant pieces of metadata and feed ONLY those into the agent's context. This is
what keeps irrelevant schema out of the prompt and makes the agent consistently
pick the right metric/dimension — the "retrieval that actually works" requirement.
"""

from app import llm


class MetadataIndex:
    def __init__(self, semantic):
        self.sl = semantic
        self.docs: list[dict] = []
        self._build_docs()
        self._embeddings = llm.embed([d["text"] for d in self.docs])

    def _build_docs(self):
        for name, m in self.sl.metrics.items():
            syn = ", ".join(m.get("synonyms", []))
            self.docs.append({
                "kind": "metric", "name": name,
                "label": m.get("label", name),
                "text": f"metric {name} ({m.get('label','')}): {m.get('doc','')} "
                        f"synonyms: {syn}",
            })
        for name, d in self.sl.dimensions.items():
            self.docs.append({
                "kind": "dimension", "name": name,
                "label": name,
                "text": f"dimension {name}: group or filter by {name}. "
                        f"source {d['source']}",
            })
        for term, definition in self.sl.glossary.items():
            self.docs.append({
                "kind": "glossary", "name": term,
                "label": term,
                "text": f"glossary {term}: {definition}",
            })

    def retrieve(self, question: str, k: int = 8) -> list[dict]:
        qvec = llm.embed([question])[0]
        scored = []
        for doc, emb in zip(self.docs, self._embeddings):
            scored.append((llm.cosine(qvec, emb), doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for score, doc in scored[:k]:
            item = dict(doc)
            item["score"] = round(float(score), 3)
            out.append(item)
        return out

    def context_block(self, retrieved: list[dict]) -> str:
        """Human/agent-readable context assembled from retrieved metadata."""
        metrics = [d for d in retrieved if d["kind"] == "metric"]
        dims = [d for d in retrieved if d["kind"] == "dimension"]
        gloss = [d for d in retrieved if d["kind"] == "glossary"]
        lines = []
        if metrics:
            lines.append("CERTIFIED METRICS:")
            for d in metrics:
                lines.append(f"  - {d['name']}: {d['text']}")
        if dims:
            lines.append("CERTIFIED DIMENSIONS:")
            for d in dims:
                lines.append(f"  - {d['name']}")
        if gloss:
            lines.append("BUSINESS GLOSSARY:")
            for d in gloss:
                lines.append(f"  - {d['text']}")
        return "\n".join(lines)
