# zh-pii-tagger-eval

Training and evaluation code for [`ickma2311/zh-pii-tagger-35m`](https://huggingface.co/ickma2311/zh-pii-tagger-35m),
a 35M Chinese / English PII token tagger fine-tuned from
[`cjhwong/seatbelt-pii-tagger`](https://huggingface.co/cjhwong/seatbelt-pii-tagger). The repository also includes
four Chinese test sets:
- **zh-pii-ctx-heldout** (`data/e2.jsonl`): 1,875 passages. It was used during development. It is built with the
  same generator code as the training data, but with disjoint name pools and different contexts and templates.
- **zh-pii-fresh** (`data/fresh.jsonl`) and **zh-pii-fresh-2** (`data/fresh2.jsonl`): 2,000 passages each, built
  after release from new contexts. They share no Wikipedia article; 13 of the 1,300 LCCC dialogue contexts in each
  set are short contexts that appear in both. The names and addresses come from Faker, inserted by templates
  written independently of the training generator.
- **zh-pii-numaddr** (`data/numaddr.jsonl`): 1,200 short messages from an independent generator, for addresses
  that are only a number (room, door, building or unit numbers) and for numbers that are not addresses.

The model card has the result tables and limits. The main model on Hugging Face is v1.0 (back on main since
2026-10-14); v1.1 is kept under the tag `v1.1` and is not recommended because it misses addresses that are only a
number. In short, compared with the base model:
- this model misses fewer Chinese person names: recall is about 2 to 10 points higher on test sets never used
  during development, while precision moves between −3.5 and +3.5 points;
- person F1 gains are modest: +0.6 to +2.1 on zh-pii-fresh and zh-pii-fresh-2, and +3.3 to +6.7 on MSRA news; on
  Weibo and CLUENER they are not significant under the base model's own decoder;
- on PII Bench ZH person it is slightly below the base model under the base model's own decoder (−0.4 to −0.6;
  −1.0 with the released int8 on an x86 CPU);
- addresses: no consistent improvement over the base model. The optional `postprocess/cue_filter.py` (off by
  default) removes most plain numbers tagged as addresses, at a cost of about 18 points of recall on addresses
  that are only a number;
- int8 scores depend on the CPU.

This is not a state-of-the-art or across-the-board improvement.

## Files

| file | what it does |
|---|---|
| `pools.py` | entity pools (names, addresses, ...), split by a stable hash into a training half and an evaluation half |
| `build_e2.py` | builds zh-pii-ctx-heldout from LCCC test and zh-Wikipedia contexts, using the evaluation half |
| `build_train_n.py` | builds training passages from FineWeb-2 `cmn_Hani` contexts, using the training half; checks that no held-out name or context leaks; use `--v2` to build the released model's data |
| `train_eval.py` | replay self-labelling with the base model, training (the released model, a matched-step replay-only ablation, and a second model family as a check), evaluation (Kaggle T4); `--addr-clean` relabels the base model's number-only "addresses" as not-an-address (v1.1) |
| `eval_rules.py` | non-model baselines (jieba nr + regex, argus-redact fast) |
| `qwen_baseline.py` | Qwen3-8B prompted baseline |
| `export_bench.py` | int8 ONNX export (eager attention, ORT BERT fusion, dynamic int8) and CPU latency vs the base model |
| `check_convention.py` | address-span check against the base model |
| `metrics.py` | shared CoNLL reader and per-document counts |
| `confirm/gen_fresh.py` | builds zh-pii-fresh from LCCC valid and new zh-Wikipedia articles, with Faker 40.43.0 entities; `FRESH_SEED=20261015` with `FRESH_EXCLUDE=<zh-pii-fresh>` builds zh-pii-fresh-2 |
| `confirm/prep_confirm.py` | converts Weibo NER train, an MSRA NER train sample and CLUENER2020 dev to one format, and checks them for overlap with the training passages |
| `confirm/eval_confirm.py`, `confirm/eval_known.py` | score both models with the project decoder and with the base model's own Viterbi decoder; paired bootstrap |
| `results/confirm/*.json` | raw outputs of the two scripts above |
| `confirm/requirements.txt` | pinned versions used for zh-pii-fresh and these evaluations |
| `addrfix/eval_addrfix.py` | v1.1 check: base model vs v1.0 vs v1.1 on all sets, two decoders, plus a count of number-only spans tagged as address |
| `addrfix/eval_int8.py` | the same comparison for the int8 files on the current CPU |
| `results/addrfix/*.json` | raw outputs of the two scripts above, and `bench_int8_v11.json` (int8 latency and size) |
| `postprocess/cue_filter.py` | optional post-processing (off by default): drop "address" spans that are only a number unless an address word is next to them |
| `numaddr/gen_numaddr.py` | builds zh-pii-numaddr (independent generator, standard library only, seed 20261016) |
| `numaddr/eval_numaddr.py`, `numaddr/compare_numaddr.py` | predictions of the base model, v1.0 and v1.1 on all test sets, then the comparison table with and without the filter |
| `results/numaddr/compare.json` | raw output of `compare_numaddr.py` |

## Reproduce the released model

```bash
python build_e2.py <lccc_base_test.jsonl.gz> <zhwiki_rows.jsonl> data/e2.jsonl   # optional; sha256 must match data/e2.sha256
python build_train_n.py <fineweb2 cmn_Hani test 000_00000.parquet> data/e2.jsonl <weibo/msra conll files> --out data/n_v2.jsonl --n 40000 --v2
python train_eval.py --mode main --recipe v2 --trim-punct --seeds 7,77,777                        # v1.0 (main); one T4, about 0.5 h
# variant: --addr-clean --cfgs M builds v1.1 (tag v1.1); M = the released configuration only
# SB_REV=c2ca0cd7d1c2f93092dfa6bac8e4c74ad82aad18 pins the base model used as the teacher
```

Weibo NER (`hltcoe/golden-horse`, `weiboNER_2nd_conll.*`), MSRA NER and CLUENER2020 are evaluation inputs only and
are not redistributed.

## Notes

- Numbers are measured by us. The synthetic test sets have real contexts but synthetic entities, so they do not
  measure accuracy on real personal data. English was evaluated only for person, email and phone.
- The two taggers are always scored with the same decoder. The Qwen3-8B baseline maps JSON strings back to
  spans, and the rule baselines use their own spans.
- 2026-10-13: revised after a re-evaluation. It added zh-pii-fresh, the unused-split evaluations and the base
  model's own decoder, and corrected an overstated person gain.
- 2026-10-14: v1.1. It added `--addr-clean`, zh-pii-fresh-2 and the v1.1 evaluations.
- 2026-10-14: Hugging Face main is v1.0 again, with an optional filter; v1.1 is kept under the tag `v1.1`. Added
  zh-pii-numaddr and its comparison.

## Licence

Code: Apache-2.0 (`LICENSE`). `data/e2.jsonl`, `data/fresh.jsonl` and `data/fresh2.jsonl`: CC BY-SA 4.0;
`data/numaddr.jsonl`: CC BY 4.0 (`DATA_LICENSE.md`).
