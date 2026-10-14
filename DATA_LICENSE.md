# Licence of `data/e2.jsonl`

`data/e2.jsonl` (zh-pii-ctx-heldout) is released under **CC BY-SA 3.0**, because part of its text comes from
Wikipedia.

- Contexts:
  - LCCC base test split (thu-coai, via `silver/lccc` on Hugging Face), MIT License;
  - Chinese Wikipedia (`wikimedia/wikipedia`, config 20231101.zh), CC BY-SA 3.0 / GFDL. Attribution: Wikipedia
    contributors.
- Inserted entities (names, addresses, phone numbers, emails) are synthetic, made by `pools.py` and
  `build_e2.py`. No real personal data was added on purpose. The original contexts were filtered with jieba to
  drop sentences that look like they contain names, but a few real given names in the original text may remain.
  They are unlabelled.

The code in this repository is Apache-2.0 (`LICENSE`). Weibo NER and MSRA NER are evaluation inputs only and are
not redistributed here.
