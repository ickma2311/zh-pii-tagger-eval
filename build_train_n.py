"""Build training set N: FineWeb-2 cmn_Hani contexts (ODC-BY) with exactly labelled private entities
from the TRAIN half of pools.py, using insertion templates written independently of build_e2.py.

Usage: python -I build_train_n.py <fineweb2_cmn_test.parquet> <e2.jsonl> <weibo_msra_conll...> --out <n.jsonl> [--n 60000]
Asserts: no E2 full name and no E2/E3 context sentence appears in N.
"""
import sys, os, json, random, re, logging, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jieba, jieba.posseg as pseg
import pyarrow.parquet as pq
from pools import Gen

jieba.setLogLevel(logging.ERROR)
ap = argparse.ArgumentParser()
ap.add_argument("parquet"); ap.add_argument("e2"); ap.add_argument("conll", nargs="*")
ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=60000)
ap.add_argument("--v2", action="store_true", help="no name filter; record jieba nr spans as loss-ignore; building suffixes")
a = ap.parse_args()
r = random.Random(7 + 100 * a.v2); g = Gen("train", 7 + 100 * a.v2, bldg_p=0.5 if a.v2 else 0.0)

def has_name(s): return any(f.startswith("nr") for _, f in pseg.cut(s))

class P:
    def __init__(self): self.text = ""; self.ents = []
    def add(self, s): self.text += s
    def ent(self, s, t): self.ents.append((len(self.text), len(self.text) + len(s), t)); self.text += s

def person(mixed):
    if mixed and r.random() < 0.6: return g.en_name()
    return g.nickname() if r.random() < 0.15 else g.zh_name()

# Training-only templates (independent wording from build_e2.py)
LEAD = [("回复", "name", "："), ("转给", "name", "看一下："), ("", "name", "在吗？"), ("致", "name", "：")]
TAIL = [("——", "name", ""), ("签名：", "name", ""), ("（经办人：", "name", "）"), ("by ", "name", "")]
BLOCK = [("联系人", "name", "，电话", "phone", "。"), ("收货人：", "name", " ", "phone", " ", "addr", ""),
         ("家住", "addr", "。"), ("办公地点：", "addr", "。"), ("地址在", "addr", "附近，有事打", "phone", "。"),
         ("身份证号", "id", "，开户人", "name", "，卡号", "card", "。"), ("邮箱：", "email", "（", "name", "）"),
         ("我是", "name", "，住", "addr", "。")]
MIXED = [("Hi ", "name", ", my number is ", "phone", "."), ("Thanks, ", "name", " (", "email", ")"),
         ("请把合同发给", "name", "（", "email", "）"), ("Address: ", "addr", ""), ("跟", "name", "确认下 schedule")]

def emit(p, parts, mixed=False):
    for i, x in enumerate(parts):
        if i % 2 == 0: p.add(x); continue
        if x == "name": p.ent(person(mixed), "person")
        elif x == "addr": p.ent(g.address(), "address")
        elif x == "phone": p.ent(g.phone(), "phone")
        elif x == "email": p.ent(g.email(), "email")
        elif x == "id": p.ent(g.national_id(), "national_id")
        elif x == "card": p.ent(g.bank_card(), "account_number")

def build(sents):
    mixed = r.random() < 0.15; p = P()
    if r.random() < 0.10:  # negative passage: context only
        p.add("".join(sents)); return p, mixed
    lines = sents[:]; nl = r.random() < 0.5  # half as chat-like lines
    if r.random() < 0.3: emit(p, list(r.choice(LEAD)), mixed)
    k = r.randint(0, len(lines)); pron = r.random() < 0.4
    for i, s in enumerate(lines):
        if i and nl: p.add("\n")
        if i == k: emit(p, list(r.choice(MIXED if mixed else BLOCK)), mixed)
        m = re.search(r"(?<!其)[他她](?![们])", s) if pron else None
        if m:
            p.add(s[:m.start()]); p.ent(person(False), "person"); p.add(s[m.end():]); pron = False
        else:
            p.add(s)
    if k == len(lines): emit(p, list(r.choice(MIXED if mixed else BLOCK)), mixed)
    if r.random() < 0.25: emit(p, list(r.choice(TAIL)), mixed)
    return p, mixed

def main():
    e2 = [json.loads(l) for l in open(a.e2, encoding="utf-8")]
    e2_names = {e["text"] for d in e2 for e in d["entities"] if e["type"] == "person"}
    banned = set(open(a.e2 + ".contexts.txt", encoding="utf-8").read().split("\n")) if os.path.exists(a.e2 + ".contexts.txt") else set()
    for path in a.conll:  # E3 sentences (weibo / msra): rebuild raw text per sentence
        cur = []
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                if cur: banned.add("".join(cur)); cur = []
                continue
            cur.append(line.split()[0][0])
        if cur: banned.add("".join(cur))
    banned = {b for b in banned if len(b) >= 8}
    texts = pq.read_table(a.parquet, columns=["text"]).column("text").to_pylist()
    r.shuffle(texts)
    out = []; ctx_hits = 0
    for doc in texts:
        if len(out) >= a.n: break
        sents = [s for s in re.findall(r"[^。！？\n]+[。！？]?", doc) if 6 <= len(s.strip()) <= 120]
        if len(sents) < 2: continue
        st = 0
        while st < len(sents) - 1 and len(out) < a.n:  # several non-overlapping windows per document
            w = r.randint(2, 4); sel = [s.strip() for s in sents[st:st + w]]; st += w
            if any(s in banned for s in sel): ctx_hits += 1; continue
            if not a.v2 and any(has_name(s) for s in sel): continue
            p, mixed = build(sel)
            if any(p.text[s_:e_] in e2_names for s_, e_, t in p.ents if t == "person"): continue
            out.append((p, mixed))
    with open(a.out, "w", encoding="utf-8") as f:
        for i, (p, mixed) in enumerate(out):
            ents = [{"start": s, "end": e, "type": t, "text": p.text[s:e]} for s, e, t in p.ents]
            row = {"id": f"n_{i:06d}", "mixed": mixed, "text": p.text, "entities": ents}
            if a.v2:  # jieba person spans in the context that are not inserted entities -> ignore in loss
                ign, pos = [], 0
                for w, fl in pseg.cut(p.text):
                    if fl.startswith("nr") and not any(pos < e and s < pos + len(w) for s, e, _ in p.ents):
                        ign.append([pos, pos + len(w)])
                    pos += len(w)
                row["ignore"] = ign
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    from collections import Counter
    c = Counter(t for p, _ in out for _, _, t in p.ents)
    names = {p.text[s:e] for p, _ in out for s, e, t in p.ents if t == "person"}
    assert not (names & e2_names), "E2 name leaked into N"
    print("N passages", len(out), "mixed", sum(m for _, m in out), "entities", dict(c),
          "unique names", len(names), "contexts dropped as E2/E3 overlap", ctx_hits)

main()
