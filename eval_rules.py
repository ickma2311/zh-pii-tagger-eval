"""Non-model baselines on E1 / E2 / E3, same per-document count format as train_eval.py.

  argus_fast   : argus-redact mode="fast" (Apache, pure rules; strongest published pure-rule system on PII Bench ZH)
  jieba_rules  : jieba posseg nr for person + a Chinese address regex + phone/email regex
Usage: python -I eval_rules.py <spike_dir> <pii_bench_dir> <out.json>
"""
import sys, os, json, re, logging, warnings, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
from metrics import read_conll, counts, f1_from

spike, bench, out_path = sys.argv[1:4]
ev = {"E2": [json.loads(l) for l in open(os.path.join(spike, "data", "e2.jsonl"), encoding="utf-8")]}
e1 = []
for f in ["pii_bench_zh.jsonl", "pii_bench_zh_chat.jsonl"]:
    for l in open(os.path.join(bench, f), encoding="utf-8"):
        d = json.loads(l); d["sub"] = "chat" if "chat" in f else "formal"; e1.append(d)
ev["E1"] = e1
for name, fmt in [("weibo.test", "weibo"), ("weibo.dev", "weibo"), ("msra.test", "msra")]:
    ev["E3_" + name] = read_conll(os.path.join(spike, "data", "e3", name), fmt)

from argus_redact import redact
AMAP = {"person": "person", "address": "address", "phone": "phone", "email": "email"}
def argus(text):
    try:
        rep = redact(text, mode="fast", lang="zh", report=True, salt=os.urandom(32))
    except Exception:
        return []
    return [(e["start"], e["end"], AMAP[e["type"]]) for e in rep.entities if e["type"] in AMAP]

import jieba, jieba.posseg as pseg
jieba.setLogLevel(logging.ERROR)
ADDR = re.compile(r"(?:[一-龥]{2,7}(?:省|自治区))?(?:[一-龥]{2,7}市)?(?:[一-龥]{1,7}(?:区|县|镇|乡))?"
                  r"[一-龥]{1,10}(?:村|路|街|大道|巷|小区|花园|家园|苑|公寓|新村|府|城)"
                  r"[一-龥\d]{0,12}?\d+(?:号楼|号|栋|幢|单元|室|组)(?:\d+(?:号楼|号|栋|幢|单元|室|组))*")
PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"); EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
def jieba_rules(text):
    out, pos = [], 0
    for w, f in pseg.cut(text):
        if f == "nr": out.append((pos, pos + len(w), "person"))
        pos += len(w)
    out += [(m.start(), m.end(), "address") for m in ADDR.finditer(text)]
    out += [(m.start(), m.end(), "phone") for m in PHONE.finditer(text)]
    out += [(m.start(), m.end(), "email") for m in EMAIL.finditer(text)]
    return out

res = {}
for sysname, fn in [("argus_fast", argus), ("jieba_rules", jieba_rules)]:
    res[sysname] = {}
    for name, docs in ev.items():
        t0 = time.time(); preds = [fn(d["text"]) for d in docs]
        rows = counts(name.split("_")[0] if name.startswith("E3") else name, docs, preds)
        res[sysname][name] = rows
        print(sysname, name, {t: round(f1_from(rows, t)[0], 4) for t in rows[0] if not t.startswith("_")},
              f"{1000 * (time.time() - t0) / len(docs):.2f} ms/doc", flush=True)
json.dump(res, open(out_path, "w"))
