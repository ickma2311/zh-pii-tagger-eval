"""Build E2 `zh-pii-ctx-heldout`: real Chinese contexts (LCCC dialogue, zh Wikipedia) with exactly
labelled private entities from the EVAL half of pools.py. Eval-only; never used for training.

Usage: python -I build_e2.py <lccc_test.jsonl.gz> <zhwiki_rows.jsonl> <out.jsonl>
"""
import sys, os, json, gzip, random, re, hashlib, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jieba, jieba.posseg as pseg
from pools import Gen

jieba.setLogLevel(logging.ERROR)
SEED = 20261013
r = random.Random(SEED); g = Gen("eval", SEED)

def despace(s):
    toks = s.split(" "); out = toks[0] if toks else ""
    for t in toks[1:]:
        if out and t and out[-1].isascii() and out[-1].isalnum() and t[0].isascii() and t[0].isalnum():
            out += " "
        out += t
    return out

def has_name(s):
    return any(f == "nr" or f.startswith("nr") for _, f in pseg.cut(s))

class P:
    """Passage builder that tracks entity offsets."""
    def __init__(self): self.text = ""; self.ents = []
    def add(self, s): self.text += s
    def ent(self, s, t): self.ents.append((len(self.text), len(self.text) + len(s), t)); self.text += s

def person(mixed=False):
    if mixed and r.random() < 0.7: return g.en_name()
    return g.nickname() if r.random() < 0.15 else g.zh_name()

ADDR_TURNS = [("东西直接寄到", "addr", "吧"), ("我现在住", "addr", "，离公司挺远的"),
              ("新地址：", "addr", ""), ("周末来", "addr", "这边吃饭不"),
              ("快递地址改成", "addr", "，收件人", "name", "，手机", "phone"),
              ("导航到", "addr", "就行，到了打", "phone")]
MIXED_TURNS = [("", "name", " said he'd call you at ", "phone"), ("cc ", "name", " <", "email", ">"),
               ("明天和", "name", "开个 sync，邮箱是", "email"), ("Meeting moved, ping ", "name", " first"),
               ("ship to ", "addr", " pls")]
WIKI_INSERTS = [("（本段资料由", "name", "提供，联系电话", "phone", "。）"),
                ("据当地居民", "name", "介绍，其老宅位于", "addr", "。"),
                ("如需更正，请发邮件至", "email", "，联系人", "name", "。"),
                ("现居住于", "addr", "的", "name", "回忆了这段经历。")]

def emit(p, parts, mixed=False):
    for i, x in enumerate(parts):
        if i % 2 == 0: p.add(x)
        elif x == "name": p.ent(person(mixed), "person")
        elif x == "addr": p.ent(g.address(), "address")
        elif x == "phone": p.ent(g.phone(), "phone")
        elif x == "email": p.ent(g.email(), "email")
        elif x == "id": p.ent(g.national_id(), "national_id")

def from_dialog(turns):
    mixed = r.random() < 0.2
    p = P(); ops = set(r.sample(["pron", "at", "call", "addr"], r.randint(1, 3)))
    pron_done = False
    for i, t in enumerate(turns):
        if i: p.add("\n")
        if "at" in ops and i == 0: p.add("@"); p.ent(person(mixed), "person"); p.add(" ")
        if "call" in ops and i == len(turns) - 1: p.ent(person(mixed), "person"); p.add("，")
        m = None
        if "pron" in ops and not pron_done:
            m = re.search(r"(?<!其)[他她](?![们])", t)
        if m:
            p.add(t[:m.start()]); p.ent(person(False), "person"); p.add(t[m.end():]); pron_done = True
        else:
            p.add(t)
    if "addr" in ops:
        p.add("\n"); emit(p, list(r.choice(ADDR_TURNS)))
    if mixed:
        p.add("\n"); emit(p, list(r.choice(MIXED_TURNS)), mixed=True)
    return p, mixed

def from_wiki(sents):
    p = P(); k = r.randint(0, len(sents)); mixed = False
    for i, s in enumerate(sents):
        if i == k: emit(p, list(r.choice(WIKI_INSERTS)))
        p.add(s)
    if k == len(sents): emit(p, list(r.choice(WIKI_INSERTS)))
    return p, mixed

def main(lccc_path, wiki_path, out_path):
    dialogs = [[despace(t) for t in json.loads(l)] for l in gzip.open(lccc_path, "rt", encoding="utf-8")]
    r.shuffle(dialogs)
    out = []; contexts = []
    for d in dialogs:
        if len(out) >= 1200: break
        if any(has_name(t) for t in d) or sum(map(len, d)) < 8: continue
        p, mixed = from_dialog(d); out.append(("lccc", p, mixed)); contexts.extend(d)
    wiki = [json.loads(l)["text"] for l in open(wiki_path, encoding="utf-8")]
    r.shuffle(wiki)
    for doc in wiki:
        if len(out) >= 2000: break
        sents = [s for s in re.findall(r"[^。！？\n]+[。！？]", doc) if 8 <= len(s) <= 120]
        sents = [s for s in sents if not has_name(s)]
        if len(sents) < 2: continue
        st = r.randint(0, max(0, len(sents) - 3)); sel = sents[st:st + r.randint(2, 4)]
        p, mixed = from_wiki(sel); out.append(("wiki", p, mixed)); contexts.extend(sel)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (src, p, mixed) in enumerate(out):
            ents = [{"start": s, "end": e, "type": t, "text": p.text[s:e]} for s, e, t in p.ents]
            f.write(json.dumps({"id": f"e2_{i:05d}", "src": src, "mixed": mixed, "text": p.text, "entities": ents},
                               ensure_ascii=False) + "\n")
    with open(out_path + ".contexts.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(contexts))
    from collections import Counter
    c = Counter(e["type"] for _, p, _ in out for e in [{"type": t} for _, _, t in p.ents])
    print("passages", len(out), "by src", Counter(s for s, _, _ in out), "mixed", sum(m for _, _, m in out))
    print("entities", dict(c))

if __name__ == "__main__":
    main(*sys.argv[1:4])
