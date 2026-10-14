"""Comparison table from eval_numaddr.py's preds.json: base model (SB), v1.0, v1.1, and SB+F / V10+F (the optional
post-processing filter, postprocess/cue_filter.py, applied to the saved predictions); decoders D1 and D2.
"numaddr_numonly" is zh-pii-numaddr restricted to the 550 messages whose address has no street or city (568 gold
spans); "numaddr_pure" keeps only the 313 of those gold spans that are digits alone (optionally with 号), and
"numaddr_unit" the 255 that carry a unit word (栋, 号楼, 单元, 室, N-N, ...). Only recall is meaningful on the two
span-level subsets.
Usage: python -I compare_numaddr.py <preds.json> <data dir> <out.json>
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "postprocess"))
from cue_filter import NUM_ADDR, filter_spans

preds_p, data_d, out_p = sys.argv[1:4]
P = json.load(open(preds_p))["systems"]
SETS = ["numaddr", "fresh2", "msra_test", "pii_bench", "msra_train_5k", "weibo_train", "cluener_dev", "e2", "fresh"]
docs = {ds: [json.loads(l) for l in open(os.path.join(data_d, ds + ".jsonl"), encoding="utf-8")] for ds in SETS}
POS = [i for i, d in enumerate(docs["numaddr"]) if d["kind"] == "pos"]
docs["numaddr_numonly"] = [docs["numaddr"][i] for i in POS]
docs["numaddr_pure"] = [dict(d, entities=[e for e in d["entities"] if NUM_ADDR.fullmatch(e["text"])]) for d in docs["numaddr_numonly"]]
docs["numaddr_unit"] = [dict(d, entities=[e for e in d["entities"] if not NUM_ADDR.fullmatch(e["text"])]) for d in docs["numaddr_numonly"]]
SUB = ("numaddr_numonly", "numaddr_pure", "numaddr_unit")
SETS = SETS + list(SUB)
SEEDS = (7, 77, 777)

def system(name, ds, dec):
    """Predictions of a system; names ending in +F apply the cue filter to the underlying system."""
    src = "numaddr" if ds in SUB else ds
    pick = (lambda xs: [xs[i] for i in POS]) if ds in SUB else (lambda xs: xs)
    if name.endswith("+F"):
        base = pick(P[name[:-2]][src][dec])
        return [[tuple(x) for x in filter_spans(d["text"], [tuple(s) for s in p])] for d, p in zip(docs[ds], base)]
    return [[tuple(s) for s in p] for p in pick(P[name][src][dec])]

def rows(ds, preds, t, mode):
    out = []
    for d, p in zip(docs[ds], preds):
        g = [(e["start"], e["end"]) for e in d["entities"] if e["type"] == t]; q = [(s, e) for s, e, y in p if y == t]
        if mode == "strict": k = len(set(g) & set(q)); out.append((k, k, len(q), len(g)))
        else: out.append((sum(any(s < b and a < e for s, e in q) for a, b in g), sum(any(s < b and a < e for a, b in g) for s, e in q), len(q), len(g)))
    return np.array(out, float)

def prf(a):
    s = a.sum(0); P_ = s[1] / s[2] if s[2] else 0.0; R = s[0] / s[3] if s[3] else 0.0
    return (2 * P_ * R / (P_ + R) if P_ + R else 0.0), P_, R

def boot(a, b, B=2000):
    rng = np.random.default_rng(0); n = len(a); ds_ = []
    for _ in range(B):
        i = rng.integers(0, n, n); ds_.append(prf(a[i])[0] - prf(b[i])[0])
    lo, hi = np.percentile(ds_, [2.5, 97.5]); return [round(100 * lo, 2), round(100 * hi, 2)]

def bare_fp(ds, preds):
    """Predicted address spans that are number-only and overlap no gold address."""
    n = 0
    for d, p in zip(docs[ds], preds):
        g = [(e["start"], e["end"]) for e in d["entities"] if e["type"] == "address"]
        n += sum(1 for s, e, y in p if y == "address" and NUM_ADDR.fullmatch(d["text"][s:e]) and not any(s < b and a < e for a, b in g))
    return n

def f1(name, ds, t, mode, dec="D2"): return prf(rows(ds, system(name, ds, dec), t, mode))
mean = lambda xs: float(np.mean(xs))
names = {"SB": ["seatbelt"], "SB+F": ["seatbelt+F"], "V10": [f"v10_s{s}" for s in SEEDS], "V10+F": [f"v10_s{s}+F" for s in SEEDS],
         "V11": [f"v11_s{s}" for s in SEEDS]}
res = {"table": {}}
for grp, ns in names.items():
    for ds in SETS:
        for dec in ("D1", "D2"):
            for t, mode in (("address", "overlap"), ("address", "strict"), ("person", "strict")):
                v = [f1(n, ds, t, mode, dec) for n in ns]
                res["table"][f"{grp}|{ds}|{dec}|{t}|{mode}"] = {"f1": [round(x[0], 4) for x in v], "p": [round(x[1], 4) for x in v], "r": [round(x[2], 4) for x in v]}
            res["table"][f"{grp}|{ds}|{dec}|bare_fp"] = [bare_fp(ds, system(n, ds, dec)) for n in ns]

json.dump({"table": res["table"]}, open(out_p, "w"), ensure_ascii=False, indent=1)
