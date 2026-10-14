"""Shared, torch-free evaluation helpers for zh-pii-better (E3 CoNLL reader, per-document counts, F1)."""
import numpy as np

def read_conll(path, fmt):
    sents, chars, tags = [], [], []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            if chars: sents.append((chars, tags)); chars, tags = [], []
            continue
        parts = line.split()
        if len(parts) < 2: continue
        ch = parts[0][0] if fmt == "weibo" else parts[0]
        chars.append(ch); tags.append(parts[-1])
    if chars: sents.append((chars, tags))
    out = []
    for chars, tags in sents:
        text = "".join(chars); ents = []; start = None
        for i, t in enumerate(tags + ["O"]):
            is_b, is_i = t.startswith("B-"), t.startswith("I-")
            typ = t[2:] if (is_b or is_i) else None
            want = typ == ("PER.NAM" if fmt == "weibo" else "PER")
            if start is not None and (not is_i or not want): ents.append({"start": start, "end": i, "type": "person"}); start = None
            if is_b and want: start = i
        out.append({"text": text, "entities": ents})
    return out

# ---------------------------------------------------------------- scoring (per-document counts for paired bootstrap)
TYPES = {"E1": ["person", "address"], "E2": ["person", "address", "phone", "email"], "E4": ["person", "email", "phone"]}

def counts(ev_name, docs, preds):
    types = TYPES.get(ev_name, ["person"]); overlap = ev_name == "E4"
    rows = []
    for d, p in zip(docs, preds):
        row = {}
        for t in types:
            g = {(e["start"], e["end"]) for e in d["entities"] if e["type"] == t}
            q = {(s, e) for s, e, tt in p if tt == t}
            if overlap:
                tp_g = sum(1 for (a, b) in g if any(s < b and a < e for s, e in q))
                tp_p = sum(1 for (s, e) in q if any(s < b and a < e for a, b in g))
                row[t] = (tp_g, tp_p, len(q), len(g))
            else:
                k = len(g & q); row[t] = (k, k, len(q), len(g))
        if "sub" in d: row["_sub"] = d["sub"]
        if "src" in d: row["_src"] = d["src"]; row["_mixed"] = d.get("mixed", False)
        rows.append(row)
    return rows

def f1_from(rows, t):
    a = np.array([r[t] for r in rows]).sum(0)
    P = a[1] / a[2] if a[2] else 0; R = a[0] / a[3] if a[3] else 0
    return 2 * P * R / (P + R) if P + R else 0, P, R

