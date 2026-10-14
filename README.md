# zh-pii-tagger-eval

Training and evaluation code for [`ickma2311/zh-pii-tagger-35m`](https://huggingface.co/ickma2311/zh-pii-tagger-35m),
a 35M Chinese / English PII token tagger fine-tuned from
[`cjhwong/seatbelt-pii-tagger`](https://huggingface.co/cjhwong/seatbelt-pii-tagger). The repository also includes
**zh-pii-ctx-heldout** (`data/e2.jsonl`), a held-out set of 1,875 real Chinese passages with inserted synthetic
private names and addresses.

The model card has the evaluation protocol, result tables and limits; int8 latency and accuracy are in
`results/bench_int8.json`. In short: on zh-pii-ctx-heldout (real contexts,
synthetic names), person strict span F1 goes from 0.698 for the base model to 0.731–0.736, at the same model size
and about 2 ms per short passage on CPU. This is not a state-of-the-art or across-the-board improvement.

## Files

| file | what it does |
|---|---|
| `pools.py` | entity pools (names, addresses, ...), split by a stable hash into a training half and an evaluation half |
| `build_e2.py` | builds the held-out set from LCCC and zh-Wikipedia contexts with the evaluation half |
| `build_train_n.py` | builds training passages from FineWeb-2 `cmn_Hani` contexts with the training half; checks that no held-out name or context leaks; `--v2` is the released recipe |
| `train_eval.py` | replay self-labelling with the base model, training (configs M / A / F2), evaluation on E1–E4 (Kaggle T4) |
| `eval_rules.py` | non-model baselines (jieba nr + regex, argus-redact fast) |
| `qwen_baseline.py` | Qwen3-8B prompted baseline |
| `export_bench.py` | int8 ONNX export (eager attention, ORT BERT fusion, dynamic int8) and CPU latency vs the base model |
| `check_convention.py` | address-span check against the base model |
| `metrics.py` | shared CoNLL reader and per-document counts |

## Reproduce the released model

```bash
python build_e2.py <lccc_base_test.jsonl.gz> <zhwiki_rows.jsonl> data/e2.jsonl   # optional; sha256 must match data/e2.sha256
python build_train_n.py <fineweb2 cmn_Hani test 000_00000.parquet> data/e2.jsonl <weibo/msra conll files> --out data/n_v2.jsonl --n 40000 --v2
python train_eval.py --mode main --recipe v2 --trim-punct --seeds 7,77,777       # one T4, about 0.5 h
python train_eval.py --mode f2   --recipe v2 --trim-punct --seeds 7,77,777
```

Weibo NER (`hltcoe/golden-horse`, `weiboNER_2nd_conll.{test,dev}`) and MSRA NER test go in `data/e3/`; they are
not redistributed.

## Notes

- This work was done with AI assistance (Claude Code): the code, the experiments and this README.
- Numbers are measured by us. The held-out set has real contexts but synthetic entities, so it does not measure
  accuracy on real personal data. English was evaluated only for person, email and phone.
- The two taggers share one decoder that strips leading and trailing punctuation from spans. The Qwen3-8B
  baseline maps JSON strings back to spans, and the rule baselines use their own spans.

## Licence

Code: Apache-2.0 (`LICENSE`). `data/e2.jsonl`: CC BY-SA 3.0 (`DATA_LICENSE.md`).
