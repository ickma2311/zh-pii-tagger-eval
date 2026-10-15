# Licence of the test sets

`data/e2.jsonl` (zh-pii-ctx-heldout), `data/fresh.jsonl` (zh-pii-fresh) and `data/fresh2.jsonl` (zh-pii-fresh-2)
are released under **CC BY-SA 4.0**,
because part of their text comes from Wikipedia (the 20231101 dump; Wikipedia text has been licensed under
CC BY-SA 4.0 since June 2023).

## Contexts

- **LCCC**: the base *test* split for e2, the base *valid* split for fresh and fresh2 (thu-coai, via `silver/lccc` on
  Hugging Face). MIT License; see the notice below.
- **Chinese Wikipedia**: `wikimedia/wikipedia`, config 20231101.zh, licensed under
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Attribution: Wikipedia contributors of the
  articles listed in `data/wiki_sources.jsonl`, which gives the article id, title and URL (and a `?curid=` link,
  where the page history lists the authors) for every Wikipedia-derived record:
  - fresh and fresh2: from the `wiki_id` field of each record;
  - e2: its records keep no article id, so the source was recovered by matching the record's sentences against
    the dump rows it was built from. 671 of the 675 records match one article; 4 have text that is identical in
    the same 5 articles, and all 5 are listed.
  fresh and fresh2 use articles that e2 did not use, and fresh2 none that fresh used. See also the
  [Wikimedia terms of use on re-use](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content).

## Inserted entities

- All inserted entities are synthetic. No real personal data was added on purpose.
  - e2: names, addresses, phone numbers and emails made by `pools.py` and `build_e2.py`.
  - fresh and fresh2: names and addresses made by Faker 40.43.0 (MIT), through `confirm/gen_fresh.py`.
- The original contexts were filtered with jieba to drop text that looks like it contains names, but a few real
  names in the original text may remain. They are unlabelled.

## zh-pii-numaddr

`data/numaddr.jsonl` is generated entirely by `numaddr/gen_numaddr.py` from templates and word lists written in the
script (no third-party text, no Faker). It is released under **CC BY 4.0**. Numbers are invented; street and community names are generic words combined at
random and may coincide with real ones; city and district names are real. No real person is attached to any of them.

The code in this repository is Apache-2.0 (`LICENSE`). Weibo NER, MSRA NER and CLUENER2020 are evaluation inputs
only and are not redistributed here.

## LCCC notice

```
MIT License

Copyright (c) 2020 lemon234071

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
