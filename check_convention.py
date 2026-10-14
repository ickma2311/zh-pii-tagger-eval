"""Zero-GPU check: base-model int8 vs inserted address spans in N, by address format.
Usage: python -I check_convention.py <seatbelt_dir> <n.jsonl> [limit]"""
import sys, os, json, re, collections
import numpy as np, onnxruntime as ort
from tokenizers import Tokenizer
mdir, path = sys.argv[1], sys.argv[2]; lim = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
labels = json.load(open(f"{mdir}/labels.json")); tok = Tokenizer.from_file(f"{mdir}/tokenizer.json")
so = ort.SessionOptions(); so.intra_op_num_threads = 4
sess = ort.InferenceSession(f"{mdir}/model.int8.onnx", so, providers=["CPUExecutionProvider"])
def pred(text):
    out = []
    for w0 in range(0, len(text), 400):
        t = text[w0:w0 + 400]; enc = tok.encode(t); ids = np.array([enc.ids], dtype=np.int64)
        lab = [labels[i] for i in sess.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0][0].argmax(-1)]
        cur = None
        for (s, e), l in zip(enc.offsets, lab):
            if s == e: continue
            if not l.endswith("-address"):
                if cur: out.append(cur); cur = None
                continue
            p = l[0]
            if p in "BS" or cur is None: 
                if cur: out.append(cur)
                cur = [s + w0, e + w0]
            else: cur[1] = e + w0
            if p in "ES": out.append(cur); cur = None
        if cur: out.append(cur)
    m = []
    for s, e in out:
        while s < e and text[s].isspace(): s += 1
        if m and text[m[-1][1]:s].strip() == "": m[-1][1] = e
        else: m.append([s, e])
    return {tuple(x) for x in m}
def fmt(a):
    if a.startswith("Room"): return "en"
    if "交叉口" in a: return "intersection"
    if "村" in a and "组" in a: return "rural" + ("+floor" if a.endswith("楼") else "")
    if "单元" in a and "省" not in a: return "estate"
    if "省" in a: return "prov_road" + ("+bldg" if re.search(r"号.+(栋|号楼)", a) else "")
    return "other"
stat = collections.defaultdict(lambda: [0, 0]); ign = 0; rows = 0
for l in open(path, encoding="utf-8"):
    r = json.loads(l); rows += 1
    if rows > lim: break
    ign += len(r.get("ignore", []))
    gold = [(e["start"], e["end"], e["text"]) for e in r["entities"] if e["type"] == "address"]
    if not gold: continue
    p = pred(r["text"])
    for s, e, t in gold:
        k = fmt(t); stat[k][1] += 1; stat[k][0] += (s, e) in p
for k, (a, n) in sorted(stat.items()):
    print(f"{k:20s} strict agreement {a / n:.3f}  (n={n})")
print("rows", min(rows, lim), "ignore spans per row", round(ign / min(rows, lim), 2))
