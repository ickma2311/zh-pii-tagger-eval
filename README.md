# zh-pii-tagger-eval

Training and evaluation code for [`ickma2311/zh-pii-tagger-35m`](https://huggingface.co/ickma2311/zh-pii-tagger-35m),
a 35M Chinese / English PII token tagger fine-tuned from
[`cjhwong/seatbelt-pii-tagger`](https://huggingface.co/cjhwong/seatbelt-pii-tagger). The repository also includes
two Chinese test sets with real contexts and inserted synthetic entities:
- **zh-pii-ctx-heldout** (`data/e2.jsonl`): 1,875 passages. It was used during development. It is built with the
  same generator code as the training data, but with disjoint name pools and different contexts and templates.
- **zh-pii-fresh** (`data/fresh.jsonl`): 2,000 passages, built after release from new contexts. The names and
  addresses come from Faker, inserted by templates written independently of the training generator.

The model card has the result tables and limits. In short, compared with the base model:
- this model misses fewer Chinese person names: recall is about 3 to 10 points higher on test sets never used
  during development, while precision moves between −3.5 and +3.5 points;
- person F1 gains are modest: +1.0 to +2.1 on zh-pii-fresh and +3.3 to +6.7 on MSRA news; on Weibo and CLUENER
  they are significant only under the project decoder;
- on PII Bench ZH person it is slightly below the base model under the base model's own decoder (−0.4 to −0.65);
- address detection is not improved;
- int8 scores depend on the CPU.

This is not a state-of-the-art or across-the-board improvement.

## Files

| file | what it does |
|---|---|
| `pools.py` | entity pools (names, addresses, ...), split by a stable hash into a training half and an evaluation half |
| `build_e2.py` | builds zh-pii-ctx-heldout from LCCC test and zh-Wikipedia contexts, using the evaluation half |
| `build_train_n.py` | builds training passages from FineWeb-2 `cmn_Hani` contexts, using the training half; checks that no held-out name or context leaks; use `--v2` to build the released model's data |
| `train_eval.py` | replay self-labelling with the base model, training (the released model, a matched-step replay-only ablation, and a second model family as a check), evaluation (Kaggle T4) |
| `eval_rules.py` | non-model baselines (jieba nr + regex, argus-redact fast) |
| `qwen_baseline.py` | Qwen3-8B prompted baseline |
| `export_bench.py` | int8 ONNX export (eager attention, ORT BERT fusion, dynamic int8) and CPU latency vs the base model |
| `check_convention.py` | address-span check against the base model |
| `metrics.py` | shared CoNLL reader and per-document counts |
| `confirm/gen_fresh.py` | builds zh-pii-fresh from LCCC valid and new zh-Wikipedia articles, with Faker 40.43.0 entities |
| `confirm/prep_confirm.py` | converts Weibo NER train, an MSRA NER train sample and CLUENER2020 dev to one format, and checks them for overlap with the training passages |
| `confirm/eval_confirm.py`, `confirm/eval_known.py` | score both models with the project decoder and with the base model's own Viterbi decoder; paired bootstrap |
| `results/confirm/*.json` | raw outputs of the two scripts above |
| `confirm/requirements.txt` | pinned versions used for zh-pii-fresh and these evaluations |

## Reproduce the released model

```bash
python build_e2.py <lccc_base_test.jsonl.gz> <zhwiki_rows.jsonl> data/e2.jsonl   # optional; sha256 must match data/e2.sha256
python build_train_n.py <fineweb2 cmn_Hani test 000_00000.parquet> data/e2.jsonl <weibo/msra conll files> --out data/n_v2.jsonl --n 40000 --v2
python train_eval.py --mode main --recipe v2 --trim-punct --seeds 7,77,777       # one T4, about 0.5 h
```

Weibo NER (`hltcoe/golden-horse`, `weiboNER_2nd_conll.*`), MSRA NER and CLUENER2020 are evaluation inputs only and
are not redistributed.

## Notes

- Numbers are measured by us. The synthetic test sets have real contexts but synthetic entities, so they do not
  measure accuracy on real personal data. English was evaluated only for person, email and phone.
- The two taggers are always scored with the same decoder. The Qwen3-8B baseline maps JSON strings back to
  spans, and the rule baselines use their own spans.
- Revised on 2026-10-13 after a re-evaluation. It added zh-pii-fresh, the unused-split evaluations and
  the base model's own decoder, and corrected an overstated person gain.

## Licence

Code: Apache-2.0 (`LICENSE`). `data/e2.jsonl` and `data/fresh.jsonl`: CC BY-SA 4.0 (`DATA_LICENSE.md`).
