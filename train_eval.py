"""Replay self-labelling, training (configs M / A / F2), evaluation on E1–E4.

Kaggle entry. Writes results to $OUT_DIR. Progress lines start with "[progress]".
  python train_eval.py --mode smoke
  python train_eval.py --mode main --recipe v2 --trim-punct --seeds 7,77,777   # base model + M x seeds + A x seeds
  python train_eval.py --mode f2   --recipe v2 --trim-punct --seeds 7,77,777   # hfl/rbt3 family
Never reads the evaluation sets (E1–E4) for training or checkpoint choice.
"""
import os, sys, json, time, math, random, argparse, gzip, io, re
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import read_conll, counts, f1_from, TYPES

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT_DIR", os.path.join(HERE, "out")); os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SB = "cjhwong/seatbelt-pii-tagger"; SB_REV = "main"

def log(*a):
    print("[progress]", time.strftime("%H:%M:%S"), *a, flush=True)

# ---------------------------------------------------------------- tokenisers / models
def load_seatbelt():
    from huggingface_hub import hf_hub_download
    from transformers import BertForTokenClassification, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(hf_hub_download(SB, "tokenizer.json")),
                                  bos_token="<s>", cls_token="<s>", eos_token="</s>", sep_token="</s>",
                                  pad_token="<pad>", unk_token="<unk>", mask_token="<mask>")
    labels = json.load(open(hf_hub_download(SB, "labels.json")))
    model = BertForTokenClassification.from_pretrained(SB)
    return tok, model, labels

def load_rbt3(labels):
    from transformers import AutoTokenizer, BertForTokenClassification
    tok = AutoTokenizer.from_pretrained("hfl/rbt3")
    model = BertForTokenClassification.from_pretrained(
        "hfl/rbt3", num_labels=len(labels), id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)})
    return tok, model

def encode(tok, texts, max_len=256):
    return tok(texts, truncation=True, max_length=max_len, return_offsets_mapping=True, padding=False)

# ---------------------------------------------------------------- decoding (shared by every system)
TRIM = False  # --trim-punct: strip leading/trailing punctuation from every decoded span (both taggers alike)
PUNCT = set("，。、；：！？（）《》“”‘’【】—…·,.;:!?()[]<>\"'")

def decode(text, offsets, tags):
    spans, cur = [], None
    for (s, e), t in zip(offsets, tags):
        if s == e: continue
        if t == "O":
            if cur: spans.append(cur); cur = None
            continue
        p, typ = t.split("-", 1)
        if p in ("B", "S") or cur is None or cur[2] != typ:
            if cur: spans.append(cur)
            cur = [s, e, typ]
        else:
            cur[1] = e
        if p in ("E", "S"): spans.append(cur); cur = None
    if cur: spans.append(cur)
    out = []
    for s, e, t in spans:
        while s < e and (text[s].isspace() or (TRIM and text[s] in PUNCT)): s += 1
        while e > s and (text[e - 1].isspace() or (TRIM and text[e - 1] in PUNCT)): e -= 1
        if s >= e: continue
        if out and out[-1][2] == t and text[out[-1][1]:s].strip() == "":
            out[-1] = (out[-1][0], e, t)
        else:
            out.append((s, e, t))
    return out

@torch.no_grad()
def predict(model, tok, labels, texts, bs=64, window=400):
    """Character spans for each text; long texts are split into 400-char windows."""
    model.eval().to(DEV)
    pieces = [(i, w0, t[w0:w0 + window]) for i, t in enumerate(texts) for w0 in range(0, max(1, len(t)), window)]
    res = [[] for _ in texts]
    order = sorted(range(len(pieces)), key=lambda k: len(pieces[k][2]))
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]; chunk = [pieces[k][2] for k in idx]
        enc = tok(chunk, truncation=True, max_length=512, return_offsets_mapping=True, padding=True, return_tensors="pt")
        offs = enc.pop("offset_mapping").tolist()
        enc.pop("token_type_ids", None)
        with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
            logits = model(**{k: v.to(DEV) for k, v in enc.items()}).logits
        am = enc["attention_mask"].tolist()
        for j, k in enumerate(idx):
            i, w0, txt = pieces[k]
            n = sum(am[j]); ids = logits[j, :n].argmax(-1).tolist()
            for s, e, t in decode(txt, offs[j][:n], [labels[x] for x in ids]):
                res[i].append((s + w0, e + w0, t))
    return res

# ---------------------------------------------------------------- label alignment
def bioes_labels(enc_offsets, spans, label2id):
    lab = [label2id["O"]] * len(enc_offsets)
    for s, e, t in spans:
        toks = [k for k, (a, b) in enumerate(enc_offsets) if a < b and a < e and s < b]
        if not toks or f"B-{t}" not in label2id: continue
        if len(toks) == 1: lab[toks[0]] = label2id[f"S-{t}"]; continue
        lab[toks[0]] = label2id[f"B-{t}"]; lab[toks[-1]] = label2id[f"E-{t}"]
        for k in toks[1:-1]: lab[k] = label2id[f"I-{t}"]
    for k, (a, b) in enumerate(enc_offsets):
        if a == b: lab[k] = -100  # special tokens
    return lab

def make_dataset(tok, rows, label2id, max_len=256):
    enc = encode(tok, [r["text"] for r in rows], max_len)
    data = []
    for i, r in enumerate(rows):
        offs = enc["offset_mapping"][i]
        spans = [(e["start"], e["end"], e["type"]) for e in r["entities"]]
        lab = bioes_labels(offs, spans, label2id)
        for s_, e_ in r.get("ignore", []):  # uncertain context names -> no loss
            for k, (a_, b_) in enumerate(offs):
                if a_ < b_ and a_ < e_ and s_ < b_ and lab[k] == label2id["O"]: lab[k] = -100
        data.append({"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i], "labels": lab})
    return data

def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    pad = lambda k, v: [b[k] + [v] * (n - len(b[k])) for b in batch]
    return {"input_ids": torch.tensor(pad("input_ids", 0)), "attention_mask": torch.tensor(pad("attention_mask", 0)),
            "labels": torch.tensor(pad("labels", -100))}

# ---------------------------------------------------------------- data
NUM_ADDR = re.compile(r"[\d\s.,，．%°]+号?")  # --addr-clean: a base-model "address" that is only a number is not an address

def clean_number_addresses(text, spans):
    """Drop base-model address spans whose text is only digits / spaces / .,% (optionally one trailing 号)."""
    return [(s, e, t) for s, e, t in spans if not (t == "address" and NUM_ADDR.fullmatch(text[s:e]))]

def cjk_ratio(s): return sum("一" <= c <= "鿿" for c in s) / max(1, len(s))

def stream_jsonl(url, max_bytes):
    import requests
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status(); buf = b""; got = 0
        for chunk in r.iter_content(1 << 20):
            got += len(chunk); buf += chunk
            *lines, buf = buf.split(b"\n")
            for l in lines:
                if l.strip(): yield json.loads(l)
            if got > max_bytes: return

def replay_texts(n_zh_openpii=20000, n_en_openpii=10000, n_nym=15000, n_nemo=15000, seed=0):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    rng = random.Random(seed); zh, en, out = [], [], []
    for row in stream_jsonl("https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1.5m/resolve/main/data/train.jsonl", 900 << 20):
        t = row.get("source_text", "")[:400]
        if cjk_ratio(t) > 0.3 and len(zh) < n_zh_openpii: zh.append(t)
        elif t.isascii() and len(en) < n_en_openpii: en.append(t)
        if len(zh) >= n_zh_openpii and len(en) >= n_en_openpii: break
    out += [("openpii_zh", t) for t in zh] + [("openpii_en", t) for t in en]
    log("replay openpii", len(zh), len(en))
    nym = [json.loads(l)["text"][:400] for l in open(hf_hub_download("Wismut/nym-pii-multilingual-data", "train.jsonl", repo_type="dataset"), encoding="utf-8")]
    nym = [t for t in nym if cjk_ratio(t) > 0.3 or t.isascii()]; rng.shuffle(nym)
    out += [("nym", t) for t in nym[:n_nym]]
    nemo = pq.read_table(hf_hub_download("nvidia/Nemotron-PII", "data/train-00000-of-00001.parquet", repo_type="dataset"), columns=["text"]).column("text").to_pylist()
    rng.shuffle(nemo); out += [("nemotron", t[:400]) for t in nemo[:n_nemo]]
    log("replay total", len(out))
    return out

def load_evals(data_dir):
    from huggingface_hub import hf_hub_download
    ev = {}
    ev["E2"] = [json.loads(l) for l in open(os.path.join(data_dir, "e2.jsonl"), encoding="utf-8")]
    e1 = []
    for f in ["pii_bench_zh.jsonl", "pii_bench_zh_chat.jsonl"]:
        p = hf_hub_download("wan9yu/pii-bench-zh", f"data/{f}", repo_type="dataset", revision="c350b94897af668517ff5de237d89f2ce2eaa6f0")
        for l in open(p, encoding="utf-8"):
            d = json.loads(l); d["sub"] = "chat" if "chat" in f else "formal"; e1.append(d)
    ev["E1"] = e1
    for name, fmt in [("weibo.test", "weibo"), ("weibo.dev", "weibo"), ("msra.test", "msra")]:
        ev["E3_" + name] = read_conll(os.path.join(data_dir, "e3", name), fmt)
    import pyarrow.parquet as pq
    t = pq.read_table(hf_hub_download("nvidia/Nemotron-PII", "data/test-00000-of-00001.parquet", repo_type="dataset"), columns=["text", "spans"]).to_pylist()
    random.Random(4).shuffle(t); e4 = []
    m = {"first_name": "person", "last_name": "person", "name": "person", "email": "email", "phone_number": "phone"}
    import ast
    for r in t[:1000]:
        sp = r["spans"]; sp = ast.literal_eval(sp) if isinstance(sp, str) else sp
        e4.append({"text": r["text"], "entities": [{"start": s["start"], "end": s["end"], "type": m[s["label"]]} for s in sp if s["label"] in m]})
    ev["E4"] = e4
    return ev

def evaluate(tag, model, tok, labels, ev):
    res = {}
    for name, docs in ev.items():
        preds = predict(model, tok, labels, [d["text"] for d in docs])
        rows = counts(name.split("_")[0] if name.startswith("E3") else name, docs, preds)
        res[name] = rows
        summ = {t: round(f1_from(rows, t)[0], 4) for t in rows[0] if not t.startswith("_")}
        log(tag, name, summ)
    json.dump(res, open(os.path.join(OUT, f"counts_{tag}.json"), "w"))
    return res

# ---------------------------------------------------------------- training
def train(model, tok, labels, train_rows, val_rows, seed, steps, lr, bs=32, tag=""):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    l2i = {l: i for i, l in enumerate(labels)}
    tr = make_dataset(tok, train_rows, l2i); va = make_dataset(tok, val_rows, l2i)
    g = torch.Generator(); g.manual_seed(seed)
    dl = DataLoader(tr, batch_size=bs, shuffle=True, collate_fn=collate, generator=g, drop_last=True)
    model.to(DEV).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * max(0.0, (steps - s) / max(1, steps - warm)))
    scaler = torch.cuda.amp.GradScaler(enabled=DEV == "cuda")
    step, best, best_state, it = 0, 1e9, None, iter(dl)
    t0 = time.time()
    while step < steps:
        try: b = next(it)
        except StopIteration: it = iter(dl); b = next(it)
        b = {k: v.to(DEV) for k, v in b.items()}
        with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
            loss = model(**b).loss
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step(); step += 1
        if step % max(1, steps // 4) == 0 or step == steps:
            vl = val_loss(model, va); model.train()
            log(tag, f"step {step}/{steps} loss {loss.item():.4f} val {vl:.4f} {time.time() - t0:.0f}s")
            if vl < best: best = vl; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best

@torch.no_grad()
def val_loss(model, va, bs=64):
    model.eval(); tot, n = 0.0, 0
    for i in range(0, len(va), bs):
        b = {k: v.to(DEV) for k, v in collate(va[i:i + bs]).items()}
        with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
            tot += model(**b).loss.item() * len(va[i:i + bs]); n += len(va[i:i + bs])
    return tot / max(1, n)

# ---------------------------------------------------------------- main
def build_rows(mode_cfg, n_rows, replay_rows):
    if mode_cfg == "A": return list(replay_rows)
    if mode_cfg == "F2":  # Chinese part of N + R
        return list(n_rows) + [r for r in replay_rows if cjk_ratio(r["text"]) > 0.3]
    return list(n_rows) + list(replay_rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke"); ap.add_argument("--seeds", default="13,42,2026")
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--recipe", default="v1", choices=["v1", "v2"])
    ap.add_argument("--trim-punct", action="store_true", help="strip leading/trailing punctuation from decoded spans (released setting)")
    ap.add_argument("--addr-clean", action="store_true", help="v1.1: base-model number-only addresses become O")
    ap.add_argument("--cfgs", default="", help="comma list of configs to train (default by mode)")
    a = ap.parse_args(); seeds = [int(s) for s in a.seeds.split(",")]
    t_start = time.time()
    log("device", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "")
    sb_tok, sb_model, labels = load_seatbelt()
    ev = load_evals(a.data)
    if a.mode == "smoke":
        ev = {k: v[:150] for k, v in ev.items()}
    log("evals", {k: len(v) for k, v in ev.items()})

    # N rows (gold inserted person/address/...) + seatbelt self-labels for OTHER types on N contexts
    v2 = a.recipe == "v2"
    global TRIM; TRIM = a.trim_punct
    n_rows = [json.loads(l) for l in open(os.path.join(a.data, "n_v2.jsonl" if v2 else "n.jsonl"), encoding="utf-8")]
    if v2: n_rows = n_rows[:40000]
    sizes = (24000, 12000, 12000, 12000) if v2 else (20000, 10000, 15000, 15000)
    rtexts = replay_texts(*((300, 200, 200, 200) if a.mode == "smoke" else sizes))
    if a.mode == "smoke": n_rows = n_rows[:800]
    sl_n = predict(sb_model, sb_tok, labels, [r["text"] for r in n_rows])
    n_clean = 0
    if a.addr_clean:
        before = sum(len(p) for p in sl_n)
        sl_n = [clean_number_addresses(r["text"], p) for r, p in zip(n_rows, sl_n)]; n_clean += before - sum(len(p) for p in sl_n)
    for r, p in zip(n_rows, sl_n):
        gold = [(e["start"], e["end"]) for e in r["entities"]]
        for s, e, t in p:
            if t in ("person", "address") and not v2: continue  # v1: context person/address left as O
            if any(s < b and a_ < e for a_, b in gold): continue
            r["entities"].append({"start": s, "end": e, "type": t})
        if v2:  # an ignore span that seatbelt labelled (or that overlaps gold) is no longer uncertain
            ents = [(e["start"], e["end"]) for e in r["entities"]]
            r["ignore"] = [[s, e] for s, e in r.get("ignore", []) if not any(s < b and a_ < e for a_, b in ents)]
    sl_r = predict(sb_model, sb_tok, labels, [t for _, t in rtexts])
    if a.addr_clean:
        before = sum(len(p) for p in sl_r)
        sl_r = [clean_number_addresses(t, p) for (_, t), p in zip(rtexts, sl_r)]; n_clean += before - sum(len(p) for p in sl_r)
        log("addr-clean: number-only base-model addresses dropped", n_clean)
    replay_rows = [{"text": t, "entities": [{"start": s, "end": e, "type": ty} for s, e, ty in p], "src": src}
                   for (src, t), p in zip(rtexts, sl_r)]
    log("selflabel done", len(n_rows), len(replay_rows))

    rng = random.Random(0)
    def split(rows):
        rows = rows[:]; rng.shuffle(rows); k = max(20, len(rows) // 50); return rows[k:], rows[:k]
    n_tr, n_va = split(n_rows); r_tr, r_va = split(replay_rows)
    meta = {"n_train": len(n_tr), "replay_train": len(r_tr), "seeds": seeds, "mode": a.mode, "recipe": a.recipe,
            "ignore_spans": sum(len(r.get("ignore", [])) for r in n_rows),
            "ctx_selflabel_person": sum(1 for r in n_rows for e in r["entities"] if "text" not in e and e["type"] == "person"),
            "addr_clean": a.addr_clean, "addr_clean_dropped": n_clean}
    log("meta", meta)

    if a.mode in ("smoke", "main"):
        evaluate("seatbelt", sb_model, sb_tok, labels, ev)  # with TRIM on, seatbelt gets the same decoder
        if a.mode == "main":
            try:
                ref_tok, ref_model = load_uer()
                evaluate_uer(ref_model, ref_tok, {k: v for k, v in ev.items() if k.startswith(("E2", "E3", "E1"))})
            except Exception as e:
                log("uer reference failed", repr(e))
    bs = 32
    steps_M = (len(n_tr) + len(r_tr)) // bs  # one epoch of N+R
    if a.mode == "smoke": steps_M = 200
    cfgs = ["M", "A"] if a.mode in ("smoke", "main") else ["F2"]
    if a.cfgs: cfgs = a.cfgs.split(",")
    if a.mode == "smoke": seeds = seeds[:1]
    for cfg in cfgs:
        for seed in seeds:
            tag = f"{cfg}{'3' if TRIM else ('2' if v2 else '')}{'a' if a.addr_clean else ''}_s{seed}"
            if cfg == "F2":
                tok, model = load_rbt3(labels); lr = 5e-5
            else:
                tok = sb_tok; model = load_seatbelt()[1]; lr = 3e-5
            tr = build_rows(cfg, n_tr, r_tr); va = build_rows(cfg, n_va, r_va)
            model, bv = train(model, tok, labels, tr, va, seed, steps_M, lr, bs, tag)
            evaluate(tag, model, tok, labels, ev)
            meta[tag] = {"best_val": bv, "train_rows": len(tr), "steps": steps_M}
            if cfg == "M" and a.mode == "main":
                model.save_pretrained(os.path.join(OUT, f"model_{tag}"))
            del model; torch.cuda.empty_cache()
    meta["wall_h"] = (time.time() - t_start) / 3600
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    log("done", meta["wall_h"])

# uer CLUENER reference (no licence; reported for reference only)
def load_uer():
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    m = "uer/roberta-base-finetuned-cluener2020-chinese"
    return AutoTokenizer.from_pretrained(m), AutoModelForTokenClassification.from_pretrained(m)

def evaluate_uer(model, tok, ev):
    lab = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    mp = {"name": "person", "address": "address"}
    lab2 = []
    for l in lab:
        if l == "O" or "-" not in l: lab2.append("O"); continue
        p, t = l.split("-", 1); t = mp.get(t)
        lab2.append("O" if t is None else ("B" if p in ("B", "S") else "I") + "-" + t)
    evaluate("uer_cluener_ref", model, tok, lab2, ev)

if __name__ == "__main__":
    main()
