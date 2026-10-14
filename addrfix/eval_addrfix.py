"""v1.1 check: seatbelt vs the released model (M3) vs the address-fixed model (M3a),
fp32, seeds 7 / 77 / 777, decoders D1 (project) and D2 (seatbelt's own Viterbi + punctuation trim). No extra rules.
Also counts number-only spans predicted as address (the error being fixed).
Usage: python eval_addrfix.py --data data --ckpt ckpt --new new   (writes $OUT_DIR/results.json, preds.json)
"""
import os, sys, json, time, hashlib, argparse
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--data"); ap.add_argument("--ckpt"); ap.add_argument("--smoke", type=int, default=0)
ap.add_argument("--new", help="dir with model_M3a_s{7,77,777}/ (config.json + weights.st)")
A = ap.parse_args()
OUT = os.environ.get("OUT_DIR", "out"); os.makedirs(OUT, exist_ok=True)
def prog(*a): print("[progress]", *a, flush=True)

SB_REPO, SB_REV = "cjhwong/seatbelt-pii-tagger", "c2ca0cd7d1c2f93092dfa6bac8e4c74ad82aad18"
OURS_REPO, OURS_REV = "ickma2311/zh-pii-tagger-35m", "c206a27b5b028c23b0e1d9c3faefbd8cb502274c"
SHA = {"sb_int8": "0a4696ac95ec8c1cb8fb560a88c2dfcf347fc9628fd5cbfadbbd34f8d429d9f4",
       "ours_int8": "ea20aa416406a82393c55aa046f5ee4f4415fd6f2537c35aaa087f33fe93d942",
       "ours_s7_fp32": "b620a28ef853f12698561376daded0f6404f1a46c91398b2287b41d32109782a"}
PUNCT = set("，。、；：！？（）《》“”‘’【】—…·,.;:!?()[]<>\"'")
SENTENCE_ENDS = "。！？\n"
TYPES = {"fresh2": ["person", "address"], "weibo_train": ["person", "address"], "msra_train_5k": ["person", "address"],
         "cluener_dev": ["person", "address"], "pii_bench": ["person", "address"], "fresh": ["person", "address"],
         "e2": ["person", "address"], "msra_test": ["person", "address"]}
import re
NUM_ADDR = re.compile(r"[\d\s.,，．%°]+号?")

def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
from huggingface_hub import hf_hub_download
def hub(repo, rev, f): return hf_hub_download(repo, f, revision=rev)
tok_path = hub(SB_REPO, SB_REV, "tokenizer.json"); labels = json.load(open(hub(SB_REPO, SB_REV, "labels.json")))
assert labels == json.load(open(hub(OURS_REPO, OURS_REV, "labels.json"))), "label sets differ"
assert sha(tok_path) == sha(hub(OURS_REPO, OURS_REV, "tokenizer.json")), "tokenizers differ"
from tokenizers import Tokenizer
TOK = Tokenizer.from_file(tok_path); TOK.no_padding(); TOK.enable_truncation(max_length=512)
OUTSIDE = labels.index("O")

# ---------------------------------------------------------------- forward passes
import onnxruntime as ort
class Onnx:
    def __init__(self, p):
        so = ort.SessionOptions(); so.intra_op_num_threads = os.cpu_count() or 4
        self.s = ort.InferenceSession(p, so, providers=["CPUExecutionProvider"]); self.n = [i.name for i in self.s.get_inputs()]
    def __call__(self, ids):
        x = np.array([ids], dtype=np.int64); feed = {"input_ids": x, "attention_mask": np.ones_like(x)}
        if "token_type_ids" in self.n: feed["token_type_ids"] = np.zeros_like(x)
        return self.s.run(None, feed)[0][0]
class Torch:
    def __init__(self, d):
        import torch; from transformers import BertForTokenClassification
        torch.set_num_threads(os.cpu_count() or 4); self.torch = torch
        if not os.path.exists(os.path.join(d, "model.safetensors")) and os.path.exists(os.path.join(d, "weights.st")):
            # our cloud-run bundler drops *.safetensors, so bundled checkpoints travel as weights.st (packaging workaround)
            import shutil, tempfile
            t = tempfile.mkdtemp(); shutil.copy(os.path.join(d, "config.json"), t)
            shutil.copy(os.path.join(d, "weights.st"), os.path.join(t, "model.safetensors")); d = t
        self.m = BertForTokenClassification.from_pretrained(d).eval()
    def __call__(self, ids):
        with self.torch.no_grad():
            x = self.torch.tensor([ids]); return self.m(input_ids=x, attention_mask=self.torch.ones_like(x)).logits[0].numpy()

# ---------------------------------------------------------------- decoders
def d1_window(text, enc, lg):  # train_eval.decode with TRIM
    spans, cur = [], None
    for (s, e), i in zip(enc.offsets, lg.argmax(-1)):
        t = labels[i]
        if s == e: continue
        if t == "O":
            if cur: spans.append(cur); cur = None
            continue
        p, typ = t.split("-", 1)
        if p in ("B", "S") or cur is None or cur[2] != typ:
            if cur: spans.append(cur)
            cur = [s, e, typ]
        else: cur[1] = e
        if p in ("E", "S"): spans.append(cur); cur = None
    if cur: spans.append(cur)
    out = []
    for s, e, t in spans:
        while s < e and (text[s].isspace() or text[s] in PUNCT): s += 1
        while e > s and (text[e - 1].isspace() or text[e - 1] in PUNCT): e -= 1
        if s >= e: continue
        if out and out[-1][2] == t and text[out[-1][1]:s].strip() == "": out[-1] = (out[-1][0], e, t)
        else: out.append((s, e, t))
    return out

def split_tag(t): return ("O", None) if t == "O" else tuple(t.split("-", 1))
def allowed(a, b):
    pa, ta = split_tag(a); pb, tb = split_tag(b)
    if pa in ("B", "I"): return pb in ("I", "E") and tb == ta
    return pb in ("O", "B", "S")
TR = np.where(np.array([[allowed(a, b) for b in labels] for a in labels]), 0.0, -np.inf)
CS = np.array([split_tag(l)[0] in ("O", "B", "S") for l in labels]); CE = np.array([split_tag(l)[0] in ("O", "E", "S") for l in labels])
def viterbi(em):
    g = em.argmax(1)
    if CS[g[0]] and CE[g[-1]] and not np.isinf(TR[g[:-1], g[1:]]).any(): return g.tolist()
    sc = np.where(CS, em[0], -np.inf); back = []
    for row in em[1:]:
        c = sc[:, None] + TR; b = c.argmax(0); sc = c[b, np.arange(len(CS))] + row; back.append(b)
    path = [int(np.where(CE, sc, -np.inf).argmax())]
    for b in reversed(back): path.append(int(b[path[-1]]))
    return path[::-1]
def d2_window(text, enc, lg, trim):
    offs = enc.offsets
    real = [i for i, (tk, (s, e)) in enumerate(zip(enc.tokens, offs)) if e > s and tk != "▁"]
    if not real or (lg.argmax(1) == OUTSIDE).all(): return []
    path = viterbi(lg[real]); found = []; start = None
    for k, pi in enumerate(path):
        p, t = split_tag(labels[pi]); s, e = offs[real[k]]
        if p == "S": found.append((s, e, t))
        elif p == "B": start = s
        elif p == "E" and start is not None: found.append((start, e, t)); start = None
    if not trim: return found
    out = []
    for s, e, t in found:
        while s < e and (text[s].isspace() or text[s] in PUNCT): s += 1
        while e > s and (text[e - 1].isspace() or text[e - 1] in PUNCT): e -= 1
        if s < e: out.append((s, e, t))
    return out

def chunk_text(text, max_chars=400):  # agent-seatbelt tagger.chunk_text
    sents, start = [], 0
    for i, c in enumerate(text):
        if c in SENTENCE_ENDS: sents.append((start, i + 1)); start = i + 1
    if start < len(text): sents.append((start, len(text)))
    pieces = [(s, min(s + max_chars, e)) for s0, e in sents for s in range(s0, e, max_chars)]
    chunks = []
    for ps, pe in pieces:
        if chunks and pe - chunks[-1][0] <= max_chars: chunks[-1] = (chunks[-1][0], pe)
        else: chunks.append((ps, pe))
    return [(cs, text[cs:ce]) for cs, ce in chunks]

def predict_all(fwd, texts):
    """{decoder: [spans per text]} with one forward per distinct window."""
    cache = {}
    def run(chunk):
        if chunk not in cache:
            enc = TOK.encode(chunk); cache[chunk] = (enc, fwd(enc.ids) if enc.ids else None)
        return cache[chunk]
    res = {"D1": [], "D2": [], "D2raw": []}
    for t in texts:
        r1, r2, r2r = [], [], []
        for w0 in range(0, max(1, len(t)), 400):
            ch = t[w0:w0 + 400]; enc, lg = run(ch)
            if lg is not None: r1 += [(s + w0, e + w0, y) for s, e, y in d1_window(ch, enc, lg)]
        for off, ch in chunk_text(t):
            enc, lg = run(ch)
            if lg is None: continue
            r2 += [(s + off, e + off, y) for s, e, y in d2_window(ch, enc, lg, True)]
            r2r += [(s + off, e + off, y) for s, e, y in d2_window(ch, enc, lg, False)]
        res["D1"].append(r1); res["D2"].append(r2); res["D2raw"].append(r2r)
    return res

# ---------------------------------------------------------------- metrics
def rows(docs, preds, t, mode):
    out = []
    for d, p in zip(docs, preds):
        g = [(e["start"], e["end"]) for e in d["entities"] if e["type"] == t]; q = [(s, e) for s, e, y in p if y == t]
        if mode == "strict": k = len(set(g) & set(q)); out.append((k, k, len(q), len(g)))
        else: out.append((sum(any(s < b and a < e for s, e in q) for a, b in g), sum(any(s < b and a < e for a, b in g) for s, e in q), len(q), len(g)))
    return np.array(out, float)
def prf(a):
    s = a.sum(0); P = s[1] / s[2] if s[2] else 0.0; R = s[0] / s[3] if s[3] else 0.0
    return (2 * P * R / (P + R) if P + R else 0.0), P, R
def boot(a, b, B=2000):
    rng = np.random.default_rng(0); n = len(a); ds = []
    for _ in range(B):
        i = rng.integers(0, n, n); ds.append(prf(a[i])[0] - prf(b[i])[0])
    lo, hi = np.percentile(ds, [2.5, 97.5]); return round(100 * lo, 2), round(100 * hi, 2)

def compare(docs, ours, base, ds, dec, label):
    out = []
    for t in TYPES[ds]:
        for mode in ("strict", "overlap"):
            a, b = rows(docs, ours[dec], t, mode), rows(docs, base[dec], t, mode)
            fa, pa, ra = prf(a); fb, pb, rb = prf(b); lo, hi = boot(a, b)
            out.append({"pair": label, "set": ds, "decoder": dec, "type": t, "metric": mode, "gold": int(a[:, 3].sum()),
                        "base_f1": round(fb, 4), "ours_f1": round(fa, 4), "base_p": round(pb, 4), "ours_p": round(pa, 4),
                        "base_r": round(rb, 4), "ours_r": round(ra, 4), "delta": round(100 * (fa - fb), 2), "ci95": [lo, hi]})
            prog(f"{label:10s} {ds:14s} {dec:5s} {t:7s} {mode:7s} base {fb:.4f} ours {fa:.4f} Δ {100*(fa-fb):+.2f} [{lo:+.2f},{hi:+.2f}]")
    return out

# ---------------------------------------------------------------- run
sets = {}
for ds in TYPES:
    sets[ds] = [json.loads(l) for l in open(os.path.join(A.data, ds + ".jsonl"), encoding="utf-8")]
    if A.smoke: sets[ds] = sets[ds][:A.smoke]
    prog("loaded", ds, len(sets[ds]), "sha256", sha(os.path.join(A.data, ds + ".jsonl"))[:16])
sb_dir = os.path.dirname(hub(SB_REPO, SB_REV, "model.safetensors")); hub(SB_REPO, SB_REV, "config.json")
s7 = hub(OURS_REPO, OURS_REV, "model.safetensors"); hub(OURS_REPO, OURS_REV, "config.json")
assert sha(s7) == SHA["ours_s7_fp32"], "s7 fp32 hash mismatch"
systems = [("seatbelt", lambda: Torch(sb_dir)), ("rel_s7", lambda: Torch(os.path.dirname(s7)))]
systems += [(f"rel_s{s}", (lambda s=s: Torch(os.path.join(A.ckpt, f"model_M3_s{s}")))) for s in (77, 777)]
systems += [(f"new_s{s}", (lambda s=s: Torch(os.path.join(A.new, f"model_M3a_s{s}")))) for s in (7, 77, 777)]
P = {}
for name, mk in systems:
    m = mk(); t0 = time.time()
    P[name] = {ds: predict_all(m, [d["text"] for d in docs]) for ds, docs in sets.items()}
    prog(name, "predicted in", round(time.time() - t0), "s"); del m
json.dump(P, open(os.path.join(OUT, "preds.json"), "w"))
res = {"numaddr": {}, "rows": []}
for name in P:
    for ds, docs in sets.items():
        for dec in ("D1", "D2"):
            n = sum(1 for d, p in zip(docs, P[name][ds][dec]) for s, e, t in p if t == "address" and NUM_ADDR.fullmatch(d["text"][s:e]))
            a = sum(1 for p in P[name][ds][dec] for s, e, t in p if t == "address")
            res["numaddr"][f"{name}|{ds}|{dec}"] = [n, a]
pairs = [(f"new_s{s}", "seatbelt") for s in (7, 77, 777)] + [(f"new_s{s}", f"rel_s{s}") for s in (7, 77, 777)] + [(f"rel_s{s}", "seatbelt") for s in (7, 77, 777)]
for ours, base in pairs:
    for ds, docs in sets.items():
        for dec in ("D1", "D2"):
            for t in TYPES[ds]:
                for mode in ("strict", "overlap"):
                    a, b = rows(docs, P[ours][ds][dec], t, mode), rows(docs, P[base][ds][dec], t, mode)
                    fa, pa, ra = prf(a); fb, pb, rb = prf(b)
                    ci = boot(a, b) if dec == "D2" and ((t == "person" and mode == "strict") or (t == "address" and mode == "overlap")) else None
                    res["rows"].append({"ours": ours, "base": base, "set": ds, "decoder": dec, "type": t, "metric": mode, "gold": int(a[:, 3].sum()),
                                        "base_f1": round(fb, 4), "ours_f1": round(fa, 4), "base_p": round(pb, 4), "ours_p": round(pa, 4),
                                        "base_r": round(rb, 4), "ours_r": round(ra, 4), "delta": round(100 * (fa - fb), 2), "ci95": ci})
        prog("scored", ours, "vs", base, ds)
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), ensure_ascii=False, indent=1)
prog("done", len(res["rows"]), "rows")
