"""Build the unseen evaluation sets (zh-pii-fresh, Weibo NER train, MSRA NER train sample, CLUENER2020 dev) and check them against training data.
Writes <out>/{weibo_train,msra_train_5k,cluener_dev,fresh}.jsonl with {"id","text","entities":[{start,end,type,text}]}
and <out>/overlap.json. Overlap check: training contexts with inserted
entities masked, exact sentence match and char 8-gram Jaccard > 0.5 against any training passage.
Usage: python -I prep_confirm.py <raw dir> <fresh.jsonl> <n_v2.jsonl> <e2.jsonl> <out dir>
"""
import sys, os, json, random, collections, hashlib

RAW, FRESH, NV2, E2, OUT = sys.argv[1:6]
os.makedirs(OUT, exist_ok=True)

def jl(p): return [json.loads(l) for l in open(p, encoding="utf-8")]

def conll(path, person_tag, char_first):
    """BIO CoNLL -> records; only `person_tag` spans are kept, as type person."""
    sents, ch, tg = [], [], []
    for line in open(path, encoding="utf-8"):
        p = line.rstrip("\n").split()
        if not p:
            if ch: sents.append((ch, tg)); ch, tg = [], []
            continue
        if len(p) < 2: continue
        ch.append(p[0][0] if char_first else p[0]); tg.append(p[-1])
    if ch: sents.append((ch, tg))
    out = []
    for ch, tg in sents:
        text = "".join(ch); ents = []; st = None
        for i, x in enumerate(tg + ["O"]):
            b, it = x.startswith("B-"), x.startswith("I-"); ty = x[2:] if (b or it) else None
            if st is not None and (not it or ty != person_tag):
                ents.append({"start": st, "end": i, "type": "person", "text": text[st:i]}); st = None
            if b and ty == person_tag: st = i
        out.append({"text": text, "entities": ents})
    return out

def cluener(path):
    out = []
    for d in jl(path):
        ents = []
        for lab, ty in (("name", "person"), ("address", "address")):
            for s, spans in d["label"].get(lab, {}).items():
                for a, b in spans:
                    assert d["text"][a:b + 1] == s
                    ents.append({"start": a, "end": b + 1, "type": ty, "text": s})
        out.append({"text": d["text"], "entities": sorted(ents, key=lambda e: e["start"])})
    return out

# ---- training passages, inserted entities masked (overlap check)
def masked(d):
    t = d["text"]; out = []; p = 0
    for e in sorted(d["entities"], key=lambda e: e["start"]):
        out.append(t[p:e["start"]]); out.append("\u0000"); p = e["end"]
    out.append(t[p:]); return "".join(out)

def grams(s, n=8): return {s[i:i + n] for i in range(len(s) - n + 1) if "\u0000" not in s[i:i + n] and "\n" not in s[i:i + n]}

N = jl(NV2)[:40000]
ng, idx = [], collections.defaultdict(list)
for i, d in enumerate(N):
    g = grams(masked(d)); ng.append(g)
    for x in g: idx[x].append(i)
big = "\u0002".join(d["text"] for d in N)

def max_jacc(text):
    g = grams(text)
    if not g: return 0.0
    c = collections.Counter()
    for x in g:
        lst = idx.get(x)
        if lst and len(lst) < 2000: c.update(lst)
    return max((k / len(g | ng[i]) for i, k in c.most_common(20)), default=0.0)

def screen(name, recs):
    keep, drop = [], []
    for r in recs:
        verb = len(r["text"]) >= 15 and r["text"] in big
        j = max_jacc(r["text"])
        (drop if verb or j > 0.5 else keep).append(r)
    stats = {"n_in": len(recs), "dropped": len(drop), "verbatim_in_train": sum(len(r["text"]) >= 15 and r["text"] in big for r in drop),
             "examples_dropped": [r["text"][:60] for r in drop[:5]]}
    return keep, stats

ov = {}
weibo, ov["weibo_train"] = screen("weibo", conll(os.path.join(RAW, "weibo.train"), "PER.NAM", True))
msra_all, ov["msra_train_all"] = screen("msra", [r for r in conll(os.path.join(RAW, "msra.train"), "PER", False) if r["text"]])
msra = random.Random(2026).sample(msra_all, 5000)
clue, ov["cluener_dev"] = screen("cluener", cluener(os.path.join(RAW, "cluener", "dev.json")))
fresh, ov["fresh"] = screen("fresh", [{"text": d["text"], "entities": d["entities"], "src": d["source"]} for d in jl(FRESH)])
# fresh vs E2 contexts: exact passage and near-duplicate
e2 = jl(E2); e2t = {d["text"] for d in e2}
e2g = [grams(masked(d)) for d in e2]
ov["fresh_vs_E2_exact"] = sum(r["text"] in e2t for r in fresh)
fr_g = [grams(masked(r)) for r in fresh]
e2idx = collections.defaultdict(set)
for i, g in enumerate(e2g):
    for x in g: e2idx[x].add(i)
mx = []
for g in fr_g:
    c = collections.Counter(i for x in g for i in e2idx.get(x, ()))
    mx.append(max((k / len(g | e2g[i]) for i, k in c.most_common(5)), default=0.0) if g else 0.0)
ov["fresh_vs_E2_maxjacc_max"] = round(max(mx), 3)
ov["fresh_vs_E2_jacc>0.5"] = sum(m > 0.5 for m in mx)
# inserted-entity reuse: fresh names that were inserted training entities
ins = collections.defaultdict(set)
for d in N:
    for e in d["entities"]: ins[e["type"]].add(e["text"])
for ty in ("person", "address"):
    xs = [e["text"] for r in fresh for e in r["entities"] if e["type"] == ty]
    ov[f"fresh_{ty}_mentions"] = len(xs)
    ov[f"fresh_{ty}_as_inserted_train_entity"] = sum(x in ins[ty] for x in xs)
for name, recs in (("weibo_train", weibo), ("msra_train_5k", msra), ("cluener_dev", clue)):
    per = [e["text"] for r in recs for e in r["entities"] if e["type"] == "person"]
    ov[f"{name}_person_mentions"] = len(per)
    ov[f"{name}_person_as_inserted_train_entity"] = sum(p in ins["person"] for p in per)

for name, recs in (("weibo_train", weibo), ("msra_train_5k", msra), ("cluener_dev", clue), ("fresh", fresh)):
    with open(os.path.join(OUT, name + ".jsonl"), "w", encoding="utf-8") as f:
        for k, r in enumerate(recs):
            f.write(json.dumps({"id": f"{name}_{k:05d}", **r}, ensure_ascii=False) + "\n")
    ov[f"{name}_docs"] = len(recs)
    ov[f"{name}_sha256"] = hashlib.sha256(open(os.path.join(OUT, name + ".jsonl"), "rb").read()).hexdigest()
json.dump(ov, open(os.path.join(OUT, "overlap.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(ov, ensure_ascii=False, indent=1))
