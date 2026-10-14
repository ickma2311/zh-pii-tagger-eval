"""Local 7–8B baseline: Qwen3-8B (Apache), prompt only, greedy, thinking off.
The model lists private persons' names and addresses as JSON; every exact occurrence in the text becomes a span.
Writes per-document counts (metrics.counts format) to $OUT_DIR/counts_qwen3_8b.json.
  python qwen_baseline.py [--limit-e1 1000]
"""
import os, sys, json, time, re, argparse, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from metrics import read_conll, counts, f1_from

OUT = os.environ.get("OUT_DIR", "out"); os.makedirs(OUT, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT = ("从下面的文本中找出所有指向个人的姓名（含小名、英文名）和个人住址。"
          "只输出 JSON：{\"person\": [...], \"address\": [...]}，每项必须是原文中逐字出现的片段；没有就输出空列表。\n\n文本：\n")

def log(*a): print("[progress]", time.strftime("%H:%M:%S"), *a, flush=True)

def spans_from(text, out):
    m = re.search(r"\{.*\}", out, re.S); res = []
    if not m: return res
    try: d = json.loads(m.group(0))
    except Exception: return res
    for t in ("person", "address"):
        for s in d.get(t, []) or []:
            if not isinstance(s, str) or not s.strip(): continue
            for mm in re.finditer(re.escape(s.strip()), text): res.append((mm.start(), mm.end(), t))
    return res

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit-e1", type=int, default=1000); ap.add_argument("--bs", type=int, default=24)
    a = ap.parse_args()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import hf_hub_download
    name = "Qwen/Qwen3-8B"
    tok = AutoTokenizer.from_pretrained(name); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="auto").eval()
    ev = {"E2": [json.loads(l) for l in open(os.path.join(HERE, "data", "e2.jsonl"), encoding="utf-8")]}
    e1 = []
    for f in ["pii_bench_zh.jsonl", "pii_bench_zh_chat.jsonl"]:
        p = hf_hub_download("wan9yu/pii-bench-zh", f"data/{f}", repo_type="dataset", revision="c350b94897af668517ff5de237d89f2ce2eaa6f0")
        for l in open(p, encoding="utf-8"):
            d = json.loads(l); d["sub"] = "chat" if "chat" in f else "formal"; e1.append(d)
    random.Random(1).shuffle(e1); ev["E1_sample"] = e1[:a.limit_e1]
    for n, fmt in [("weibo.test", "weibo"), ("msra.test", "msra")]:
        docs = read_conll(os.path.join(HERE, "data", "e3", n), fmt)
        ev["E3_" + n] = [d for d in docs if len(d["text"]) <= 400] if fmt == "msra" else docs
    res, lat = {}, {}
    for evn, docs in ev.items():
        preds = []; t0 = time.time()
        order = sorted(range(len(docs)), key=lambda i: len(docs[i]["text"]))
        outs = [None] * len(docs)
        for b in range(0, len(order), a.bs):
            idx = order[b:b + a.bs]
            msgs = [tok.apply_chat_template([{"role": "user", "content": PROMPT + docs[i]["text"]}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False) for i in idx]
            enc = tok(msgs, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=160, do_sample=False)
            for j, i in enumerate(idx):
                outs[i] = tok.decode(g[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            if b // a.bs % 10 == 0: log(evn, b, "/", len(docs))
        preds = [spans_from(d["text"], o) for d, o in zip(docs, outs)]
        lat[evn] = (time.time() - t0) / len(docs)
        key = "E3" if evn.startswith("E3") else ("E1" if evn.startswith("E1") else evn)
        rows = counts(key, docs, preds); res[evn] = rows
        log(evn, {t: round(f1_from(rows, t)[0], 4) for t in rows[0] if not t.startswith("_")}, f"{lat[evn]:.3f}s/doc batched")
    # single-request latency on 50 E2 docs (batch 1)
    t0 = time.time()
    for d in ev["E2"][:50]:
        m = tok.apply_chat_template([{"role": "user", "content": PROMPT + d["text"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tok([m], return_tensors="pt").to(model.device)
        with torch.no_grad(): model.generate(**enc, max_new_tokens=160, do_sample=False)
    lat["E2_single_request_s"] = (time.time() - t0) / 50
    json.dump(res, open(os.path.join(OUT, "counts_qwen3_8b.json"), "w"))
    json.dump(lat, open(os.path.join(OUT, "qwen_latency.json"), "w"), indent=1)
    log("done", lat)

if __name__ == "__main__":
    main()
