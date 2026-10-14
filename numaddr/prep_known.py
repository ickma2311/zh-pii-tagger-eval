"""Convert the development sets to the evaluation format used by eval_numaddr.py / compare_numaddr.py:
{"id", "text", "entities": [{start, end, type, text}]} with person / address entities only.
  e2.jsonl        <- data/e2.jsonl (zh-pii-ctx-heldout), ids renumbered e2_00000...
  pii_bench.jsonl <- wan9yu/pii-bench-zh rev c350b94 pii_bench_zh.jsonl followed by pii_bench_zh_chat.jsonl
  msra_test.jsonl <- MSRA NER test in BIO CoNLL (person = PER)
numaddr.jsonl and fresh2.jsonl are data/numaddr.jsonl and data/fresh2.jsonl unchanged. fresh, weibo_train,
msra_train_5k and cluener_dev come from confirm/prep_confirm.py (fresh from data/fresh.jsonl). compare_numaddr.py
records the sha256 of every input (results/numaddr/compare.json, "inputs").
Usage: python -I prep_known.py <data/e2.jsonl> <pii_bench_zh.jsonl> <pii_bench_zh_chat.jsonl> <msra.test> <out dir>
"""
import sys, os, json
e2_p, pb1, pb2, msra_p, out = sys.argv[1:6]
os.makedirs(out, exist_ok=True)

def conll(path, person_tag, char_first):
    sents, ch, tg = [], [], []
    for line in open(path, encoding="utf-8"):
        p = line.rstrip("\n").split()
        if not p:
            if ch: sents.append((ch, tg)); ch, tg = [], []
            continue
        if len(p) < 2: continue
        ch.append(p[0][0] if char_first else p[0]); tg.append(p[-1])
    if ch: sents.append((ch, tg))
    recs = []
    for ch, tg in sents:
        text = "".join(ch); ents = []; st = None
        for i, x in enumerate(tg + ["O"]):
            b, it = x.startswith("B-"), x.startswith("I-"); ty = x[2:] if (b or it) else None
            if st is not None and (not it or ty != person_tag):
                ents.append({"start": st, "end": i, "type": "person", "text": text[st:i]}); st = None
            if b and ty == person_tag: st = i
        recs.append({"text": text, "entities": ents})
    return recs

def write(name, recs):
    with open(os.path.join(out, name + ".jsonl"), "w", encoding="utf-8") as f:
        for k, r in enumerate(recs):
            f.write(json.dumps({"id": f"{name}_{k:05d}", "text": r["text"],
                                "entities": [e for e in r["entities"] if e["type"] in ("person", "address")]}, ensure_ascii=False) + "\n")

write("e2", [json.loads(l) for l in open(e2_p, encoding="utf-8")])
write("pii_bench", [json.loads(l) for f in (pb1, pb2) for l in open(f, encoding="utf-8")])
write("msra_test", conll(msra_p, "PER", False))
