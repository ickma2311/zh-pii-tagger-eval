"""Export a trained checkpoint to int8 ONNX (same recipe for any BERT token tagger) and compare CPU latency +
strict F1 (punctuation-trim decoding) against seatbelt's shipped int8 ONNX on E1 / E2.
Usage: python -I export_bench.py <checkpoint_dir> <seatbelt_dir> <out_dir> <pii_bench_dir>"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from tokenizers import Tokenizer
from transformers import BertForTokenClassification
import train_eval as te
from metrics import counts, f1_from
te.TRIM = True
ck, sbdir, out, bench = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
m = BertForTokenClassification.from_pretrained(ck, attn_implementation="eager").eval()  # eager attention so ORT can fuse it
dummy = torch.ones(1, 16, dtype=torch.long)
fp32 = os.path.join(out, "model.onnx"); int8 = os.path.join(out, "model.int8.onnx")
torch.onnx.export(m, (dummy, dummy), fp32, input_names=["input_ids", "attention_mask"], output_names=["logits"],
                  dynamic_axes={"input_ids": {0: "b", 1: "t"}, "attention_mask": {0: "b", 1: "t"}, "logits": {0: "b", 1: "t"}},
                  opset_version=17, dynamo=False)
from onnxruntime.transformers.optimizer import optimize_model  # same fused-graph recipe as seatbelt's shipped int8
opt = optimize_model(fp32, model_type="bert", num_heads=m.config.num_attention_heads, hidden_size=m.config.hidden_size)
pre = os.path.join(out, "model.opt.onnx"); opt.save_model_to_file(pre)
import onnx as _onnx
quantize_dynamic(pre, int8, weight_type=QuantType.QInt8, extra_options={"DefaultTensorType": _onnx.TensorProto.FLOAT})
labels = json.load(open(os.path.join(sbdir, "labels.json"))); tok = Tokenizer.from_file(os.path.join(sbdir, "tokenizer.json"))
def runner(path):
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"]); names = [i.name for i in s.get_inputs()]
    def pred(text):
        res = []
        for w0 in range(0, max(1, len(text)), 400):
            t = text[w0:w0 + 400]; enc = tok.encode(t); ids = np.array([enc.ids], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
            if "token_type_ids" in names: feed["token_type_ids"] = np.zeros_like(ids)
            tags = [labels[i] for i in s.run(None, feed)[0][0].argmax(-1)]
            res += [(a + w0, b + w0, ty) for a, b, ty in te.decode(t, enc.offsets, tags)]
        return res
    return pred
e2 = [json.loads(l) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "e2.jsonl"), encoding="utf-8")]
e1 = [json.loads(l) for f in ["pii_bench_zh.jsonl", "pii_bench_zh_chat.jsonl"] for l in open(os.path.join(bench, f), encoding="utf-8")]
rep = {}
# latency: interleaved, 3 rounds, E2 only, median ms/doc
lat = {"seatbelt_int8": [], "ours_int8": []}
runs = {"seatbelt_int8": runner(os.path.join(sbdir, "model.int8.onnx")), "ours_int8": runner(int8)}
for k in runs: runs[k](e2[0]["text"])
for rnd in range(3):
    for k in (list(runs) if rnd % 2 == 0 else list(runs)[::-1]):
        t0 = time.time(); [runs[k](d["text"]) for d in e2]; lat[k].append(1000 * (time.time() - t0) / len(e2))
print("latency ms/doc E2 (3 interleaved rounds)", {k: [round(x, 3) for x in v] for k, v in lat.items()}, flush=True)
for name, path in [("seatbelt_int8", os.path.join(sbdir, "model.int8.onnx")), ("ours_int8", int8)]:
    pred = runner(path); rep[name] = {"size_mb": round(os.path.getsize(path) / 1e6, 1)}
    for evn, docs, types in [("E2", e2, ["person", "address"]), ("E1", e1, ["person", "address"])]:
        pred(docs[0]["text"]); t0 = time.time(); ps = [pred(d["text"]) for d in docs]; dt = time.time() - t0
        rows = counts(evn, docs, ps)
        rep[name][evn] = {t: round(f1_from(rows, t)[0], 4) for t in types}; rep[name][evn + "_ms_per_doc"] = round(1000 * dt / len(docs), 3)
    print(name, rep[name], flush=True)
rep["latency_rounds_ms_per_doc_E2"] = lat
json.dump(rep, open(os.path.join(out, "bench.json"), "w"), indent=1)
