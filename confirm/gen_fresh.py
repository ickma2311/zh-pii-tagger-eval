"""Independent Chinese PII (person / address) evaluation-set generator.

Contexts: LCCC dialogues and Chinese Wikipedia body sentences, filtered so that
they contain no (jieba-detected) real person names. Synthetic names and
addresses come only from Faker (zh_CN, plus en_US for a small share of
English names) and are inserted through hand-written patterns with exact gold
character offsets.

Usage: python -I gen_fresh.py <lccc_valid.jsonl.gz> <wiki_fresh.jsonl> <out.jsonl>
"""

import gzip
import json
import logging
import random
import re
import statistics
import sys

import jieba
import jieba.posseg as pseg
from faker import Faker
from faker.providers.person.zh_CN import Provider as ZhPersonProvider

SEED = 20261014
N_LCCC = 1300
N_WIKI = 700
MAX_LEN = 400
MAX_PER_SIG = 6           # max wiki passages sharing a 5-char opening signature
CTX_BUDGET = 270          # max context characters before insertion
P_ADDRESS = 0.40          # share of passages that also get one address
P_ENGLISH = 0.08          # share of person identities with an English name
P_REUSE = 0.15            # chance a later pattern reuses a person already in the passage
P_GIVEN_SWAP = 0.25       # chance a plain {P} slot is rendered as the given name (zh, 2-char given)
N_PERSON_WEIGHTS = [(1, 0.45), (2, 0.35), (3, 0.20)]

jieba.setLogLevel(logging.CRITICAL)

# ---------------------------------------------------------------------------
# Patterns. Slots: {P1}/{P2} = person (full name), {P1f}/{P2f} = familiar form
# (given name only when it has >= 2 characters, else full name), {A} = address.
# Kinds: dialogue "turn" (new line), "prefix"/"suffix" (attached to an existing
# turn); wiki "sent" (new sentence), "prefix" (attached to the start of an
# existing sentence). Every pattern id is unique.
# ---------------------------------------------------------------------------
DIALOGUE_PATTERNS = [
    ("d01", "turn", "刚才{P1}给我打电话了，说晚点到"),
    ("d02", "turn", "{P1f}，你明天有空吗？"),
    ("d03", "turn", "转发：{P1}说今晚的聚餐改到七点，别迟到"),
    ("d04", "turn", "我昨天在超市碰到{P1}了，听说{P2}也回来了"),
    ("d05", "turn", "这事你问{P1}吧，{P1f}比我清楚多了"),
    ("d06", "turn", "收件人：{P1}，地址：{A}"),
    ("d07", "turn", "快递帮我放{A}门口就行，收件人写{P1}"),
    ("d08", "turn", "我搬家啦，新地址是{A}，有空来玩"),
    ("d09", "turn", "【转】{P1}：大家周末记得把表交上来"),
    ("d10", "turn", "哈哈哈{P1f}你太逗了"),
    ("d11", "turn", "{P1}和{P2}在一起了你知道吗"),
    ("d12", "turn", "报名表上联系人填{P1}就行，钱我帮你交"),
    ("d13", "turn", "麻烦寄到{A}，电话我私信你"),
    ("d14", "turn", "等会儿{P1}来接我，你们先吃"),
    ("d15", "turn", "听{P1}说你最近在减肥？"),
    ("d16", "turn", "替{P1}谢谢大家的关心"),
    ("d17", "turn", "我妈让我问问{P1f}，周末回不回家吃饭"),
    ("d18", "turn", "上次那家店就在{A}附近，{P1}带我去过"),
    ("d19", "turn", "{P1}的生日是下周三，礼物准备好了没"),
    ("d20", "turn", "外卖地址写的{A}，到了打{P1}的电话"),
    ("d21", "turn", "群里刚有人说{P1}要结婚了，真的假的"),
    ("d22", "turn", "签到表上我替{P1}和{P2}都签了"),
    ("d23", "prefix", "{P1f}，"),
    ("d24", "suffix", "，对吧{P1f}？"),
    ("d25", "suffix", "，不信你问{P1}"),
    ("d26", "prefix", "跟{P1}说一声，"),
]

WIKI_PATTERNS = [
    ("w01", "sent", "该条目部分资料由{P1}整理提供。"),
    ("w02", "prefix", "据{P1}回忆，"),
    ("w03", "sent", "{P1}在其著作中对此有过详细论述。"),
    ("w04", "sent", "相关档案现存于{A}的一处私人收藏馆，由{P1}负责保管。"),
    ("w05", "sent", "{P1}与{P2}曾于同年合作完成一项实地调查，两人的报告后来被多次引用。"),
    ("w06", "sent", "当地居民{P1}表示，这一变化对周边生活的影响不大。"),
    ("w07", "sent", "其旧址位于{A}。"),
    ("w08", "sent", "{P1}是最早关注这一问题的研究者之一，后来{P1}又将相关成果整理出版。"),
    ("w09", "prefix", "根据{P1}提供的资料，"),
    ("w10", "sent", "这一观点最早由{P1}提出。"),
    ("w11", "sent", "{P1}晚年长期居住在{A}，直至去世。"),
    ("w12", "sent", "编者注：本段内容经{P1}审校。"),
    ("w13", "sent", "该协会的办公地址为{A}，现任负责人为{P1}。"),
    ("w14", "sent", "{P1}、{P2}等人参与了后续的修复工作。"),
    ("w15", "sent", "该处的门牌地址为{A}。"),
    ("w16", "prefix", "按照{P1}的说法，"),
    ("w17", "sent", "地方志编纂者{P1}曾实地走访，并留下了详细的记录。"),
    ("w18", "sent", "纪念馆设在{A}，由{P1}的后人捐资修建。"),
]

SLOT_RE = re.compile(r"\{(P1f|P2f|P1|P2|A)\}")


def pattern_counts(tmpl):
    slots = SLOT_RE.findall(tmpl)
    return sum(1 for s in slots if s != "A"), sum(1 for s in slots if s == "A")


# ---------------------------------------------------------------------------
# Context loading and filtering
# ---------------------------------------------------------------------------
CJK_SPACE_RE = re.compile(r" ")
ADDR_LIKE_RE = re.compile(r"(路|街|巷|道|村)\d+号|号楼|\d+单元|\d+室")
LATIN_NAME_RE = re.compile(r"[A-Z][a-z]+\s+[A-Z][a-z]+")
# Surname inventory is Faker's own zh_CN list (used only to *reject* contexts).
_SURNAMES = "".join(sorted(k for k in ZhPersonProvider.last_names if len(k) == 1))
TITLE_RE = re.compile(
    "[" + _SURNAMES + "](先生|女士|小姐|老师|教授|医生|大夫|经理|主任|老板|师傅|"
    "叔叔|阿姨|叔|姨|哥|姐|总统|总理|主席|部长|局长|书记|同学|老总|总监)")
NICK_RE = re.compile("(老|小|阿)[" + _SURNAMES + "]")


def clean_lccc_turn(turn):
    # LCCC is tokenised per character (ASCII included: "O K", "3 1"), so no word
    # boundaries survive; removing every single space restores the original text.
    return CJK_SPACE_RE.sub("", turn).strip()


def has_person_tag(text):
    for w in pseg.cut(text):
        if w.flag.startswith("nr"):   # nr, nrfg, nrt, nrf
            return True
    return False


def context_ok(text):
    if "@" in text or "＠" in text:
        return False
    if "·" in text or "•" in text:           # transliterated-name separator
        return False
    if LATIN_NAME_RE.search(text):
        return False
    if ADDR_LIKE_RE.search(text):              # pre-existing street addresses
        return False
    if TITLE_RE.search(text):                  # real "surname + title" mentions
        return False
    for m in NICK_RE.finditer(text):           # "老王 / 小李 / 阿黄" style nicknames,
        if jieba.dt.FREQ.get(m.group(0), 0) < 100:   # unless a common word (小时, 老师)
            return False
    return not has_person_tag(text)


def load_lccc(path, rng, n):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        dialogs = [json.loads(line) for line in f if line.strip()]
    order = list(range(len(dialogs)))
    rng.shuffle(order)
    out, seen = [], 0
    for i in order:
        seen += 1
        turns = [clean_lccc_turn(t) for t in dialogs[i]]
        turns = [t for t in turns if t]
        kept, total = [], 0
        for t in turns:
            if total + len(t) + 1 > CTX_BUDGET:
                break
            kept.append(t)
            total += len(t) + 1
        if not kept or total < 8:
            continue
        if not context_ok("\n".join(kept)):
            continue
        out.append({"idx": i, "units": kept})
        if len(out) == n:
            break
    return out, seen


SENT_RE = re.compile(r"[^。！？]+[。！？]")
LIST_START_RE = re.compile(r"^\s*([*#\-•·|=]|\d+[.、)）])")


def wiki_sentences(text):
    """Return a list of paragraphs, each a list of usable body sentences."""
    paras = []
    for line in text.split("\n"):
        line = line.strip()
        if len(line) < 20 or "。" not in line or LIST_START_RE.match(line):
            continue                          # headings, lists, very short lines
        sents = []
        for s in SENT_RE.findall(line):
            s = s.strip()
            bad = (len(s) < 8 or len(s) > 200 or "（，" in s or "（）" in s
                   or "，）" in s or "（；" in s or "()" in s)
            sents.append(None if bad else s)
        paras.append(sents)
    return paras


def load_wiki(path, rng, n):
    with open(path, encoding="utf-8") as f:
        arts = [json.loads(line) for line in f if line.strip()]
    order = list(range(len(arts)))
    rng.shuffle(order)
    out, cache = [], {}

    def ok(s):
        if s not in cache:
            cache[s] = context_ok(s)
        return cache[s]

    # Pass 1: one passage per article; pass 2 (only if short): a second window.
    used, sig_count = {}, {}
    for pass_no in (1, 2):
        for ai in order:
            if len(out) == n:
                return out
            art = arts[ai]
            paras = wiki_sentences(art["text"])
            windows = []
            for pi, sents in enumerate(paras):
                for si in range(len(sents)):
                    for k in (1, 2, 3, 4):
                        w = sents[si:si + k]
                        if len(w) < k or any(s is None for s in w):
                            break
                        if sum(len(s) for s in w) > CTX_BUDGET:
                            break
                        windows.append((pi, si, k))
            if pass_no == 2:
                prev = used.get(ai)
                if prev is None:
                    continue
                # keep windows disjoint from the one already used
                windows = [w for w in windows if w[0] != prev[0]
                           or w[1] + w[2] <= prev[1] or w[1] >= prev[1] + prev[2]]
            elif ai in used:
                continue
            rng.shuffle(windows)
            for (pi, si, k) in windows[:25]:
                w = paras[pi][si:si + k]
                # cap near-duplicate stub articles (township / census statistics)
                sig = re.sub(r"\d+", "0", w[0])[:5]
                if sig_count.get(sig, 0) >= MAX_PER_SIG:
                    continue
                if all(ok(s) for s in w):
                    sig_count[sig] = sig_count.get(sig, 0) + 1
                    out.append({"idx": ai, "wiki_id": art["id"], "units": list(w)})
                    used.setdefault(ai, (pi, si, k))
                    break
    return out


# ---------------------------------------------------------------------------
# Synthetic entities (Faker only)
# ---------------------------------------------------------------------------
class EntityFactory:
    def __init__(self, rng):
        self.rng = rng
        self.zh = Faker("zh_CN")
        self.en = Faker("en_US")

    def person(self):
        if self.rng.random() < P_ENGLISH:
            # first + last avoids en_US prefixes/suffixes such as "Dr." or "MD"
            full = f"{self.en.first_name()} {self.en.last_name()}"
            return {"full": full, "given": None, "lang": "en"}
        # Same composition as zh_CN name() ("{{last_name}}{{first_name}}"),
        # done in two calls so the given-name part is known exactly.
        last, first = self.zh.last_name(), self.zh.first_name()
        return {"full": last + first, "given": first if len(first) >= 2 else None, "lang": "zh"}

    def _building(self):
        # Faker zh_CN building_number() yields e.g. "s座"; upper-case the letter.
        return self.zh.building_number().upper()

    def address(self):
        r = self.rng.random()
        if r < 0.5:
            a = self.zh.address()
            a = re.sub(r"\s*\d{6}$", "", a).strip()      # strip trailing postcode
            return a[:-2] + a[-2:].upper()
        parts = []
        if self.rng.random() < 0.6:
            parts.append(self.zh.province())
        parts.append(self.zh.city())
        d = self.zh.district()
        parts.append(d if d.endswith("区") else d + "区")   # list contains "吉区"
        parts.append(self.zh.street_name())
        parts.append(self.zh.numerify("%#号") if self.rng.random() < 0.7
                     else self.zh.numerify("%##号"))
        tail = self.rng.random()
        if tail < 0.35:
            parts.append(self.zh.numerify("%号楼%单元%0%室"))
        elif tail < 0.6:
            parts.append(self.zh.numerify("%栋%0%室"))
        elif tail < 0.75:
            parts.append(self._building())
        return "".join(parts)


# ---------------------------------------------------------------------------
# Passage construction
# ---------------------------------------------------------------------------
def weighted_choice(rng, pairs):
    x, acc = rng.random(), 0.0
    for v, w in pairs:
        acc += w
        if x < acc:
            return v
    return pairs[-1][0]


def choose_patterns(rng, patterns, n_person, want_addr):
    """Pick distinct patterns whose slots sum to n_person persons and want_addr addresses."""
    info = [(p, *pattern_counts(p[2])) for p in patterns]
    chosen, need_p = [], n_person
    if want_addr:
        cands = [x for x in info if x[2] == 1 and x[1] <= need_p]
        pick = rng.choice(cands)
        chosen.append(pick[0])
        need_p -= pick[1]
    while need_p > 0:
        has_attach = any(c[1] in ("prefix", "suffix") for c in chosen)
        cands = [x for x in info if x[2] == 0 and 1 <= x[1] <= need_p
                 and x[0] not in chosen
                 and not (has_attach and x[0][1] in ("prefix", "suffix"))]
        pick = rng.choice(cands)
        chosen.append(pick[0])
        need_p -= pick[1]
    return chosen


def render(tmpl, persons, address, rng, allow_swap):
    """Return a list of (text, type|None) pieces for one pattern instance."""
    pieces, pos = [], 0
    for m in SLOT_RE.finditer(tmpl):
        if m.start() > pos:
            pieces.append((tmpl[pos:m.start()], None, None))
        slot = m.group(1)
        if slot == "A":
            pieces.append((address, "address", "address"))
        else:
            p = persons[int(slot[1]) - 1]
            familiar = slot.endswith("f") or (allow_swap and rng.random() < P_GIVEN_SWAP)
            if familiar and p["given"]:
                pieces.append((p["given"], "person", "given"))
            else:
                pieces.append((p["full"], "person", "full_" + p["lang"]))
        pos = m.end()
    if pos < len(tmpl):
        pieces.append((tmpl[pos:], None, None))
    return pieces


TRAIL_PUNCT_RE = re.compile(r"[。！？!?~～，,、…\.]+$")


def build_passage(ctx, register, rng, factory, usage):
    patterns = DIALOGUE_PATTERNS if register == "lccc" else WIKI_PATTERNS
    n_person = weighted_choice(rng, N_PERSON_WEIGHTS)
    want_addr = rng.random() < P_ADDRESS
    chosen = choose_patterns(rng, patterns, n_person, want_addr)
    rng.shuffle(chosen)

    # each unit: {"pieces": [(text, type, form)], "ctx": bool, "attached": bool}
    units = [{"pieces": [(u, None, None)], "ctx": True, "attached": False}
             for u in ctx["units"]]
    people = []
    for pid, kind, tmpl in chosen:
        n_a = pattern_counts(tmpl)[1]
        n_distinct = 2 if "{P2" in tmpl else (1 if "{P1" in tmpl else 0)
        persons = []
        for _ in range(n_distinct):
            cand = [p for p in people if p not in persons]
            if cand and rng.random() < P_REUSE:
                persons.append(rng.choice(cand))
                continue
            p = factory.person()
            people.append(p)
            persons.append(p)
        address = factory.address() if n_a else None
        pieces = render(tmpl, persons, address, rng, allow_swap=(register == "lccc"))
        usage[pid] = usage.get(pid, 0) + 1
        if kind in ("turn", "sent"):
            units.insert(rng.randint(0, len(units)),
                         {"pieces": pieces, "ctx": False, "attached": False})
            continue
        # prefix / suffix: attach to a random context unit (at most one per passage)
        u = rng.choice([u for u in units if u["ctx"] and not u["attached"]])
        base = u["pieces"][0][0]
        if kind == "prefix":
            u["pieces"] = pieces + [(base, None, None)]
        else:                                          # suffix: drop trailing punctuation
            u["pieces"] = [(TRAIL_PUNCT_RE.sub("", base) or base, None, None)] + pieces
        u["attached"] = True

    sep = "\n" if register == "lccc" else ""
    text, ents = "", []
    for ui, unit in enumerate(units):
        if ui:
            text += sep
        for s, typ, form in unit["pieces"]:
            if typ:
                ents.append({"start": len(text), "end": len(text) + len(s),
                             "type": typ, "text": s, "_form": form})
            text += s
    return text, ents


def validate(rec):
    text = rec["text"]
    assert len(text) <= MAX_LEN, rec["id"]
    last_end = -1
    for e in sorted(rec["entities"], key=lambda e: e["start"]):
        assert text[e["start"]:e["end"]] == e["text"], (rec["id"], e)
        assert e["start"] >= last_end, ("overlap", rec["id"])
        assert e["end"] > e["start"]
        last_end = e["end"]
    n_p = sum(1 for e in rec["entities"] if e["type"] == "person")
    n_a = sum(1 for e in rec["entities"] if e["type"] == "address")
    assert 1 <= n_p <= 3 and n_a <= 1, rec["id"]


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    lccc_path, wiki_path, out_path = sys.argv[1:]
    jieba.initialize()
    rng = random.Random(SEED)
    Faker.seed(SEED)
    factory = EntityFactory(rng)

    lccc_ctx, lccc_seen = load_lccc(lccc_path, rng, N_LCCC)
    wiki_ctx = load_wiki(wiki_path, rng, N_WIKI)

    usage, forms, records = {}, {}, []
    jobs = [("lccc", c) for c in lccc_ctx] + [("wiki", c) for c in wiki_ctx]
    rng.shuffle(jobs)
    for src, ctx in jobs:
        for _attempt in range(10):
            trial_usage = {}
            text, ents = build_passage(ctx, src, rng, factory, trial_usage)
            if len(text) <= MAX_LEN:
                break
            ctx = dict(ctx, units=ctx["units"][:-1] or ctx["units"])
        else:
            continue
        for k, v in trial_usage.items():
            usage[k] = usage.get(k, 0) + v
        for e in ents:
            forms[e["_form"]] = forms.get(e["_form"], 0) + 1
            del e["_form"]
        rec = {"id": f"fresh_{len(records) + 1:05d}", "source": src,
               "wiki_id": ctx.get("wiki_id") if src == "wiki" else None,
               "text": text, "entities": sorted(ents, key=lambda e: e["start"])}
        validate(rec)
        records.append(rec)

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    persons = [e for r in records for e in r["entities"] if e["type"] == "person"]
    addrs = [e for r in records for e in r["entities"] if e["type"] == "address"]
    english = [e for e in persons if re.search(r"[A-Za-z]", e["text"])]
    lens = [len(r["text"]) for r in records]
    thirds = [0, 0, 0]
    for r in records:
        for e in r["entities"]:
            thirds[min(2, int(3 * e["start"] / max(1, len(r["text"]))))] += 1
    repeated = sum(1 for r in records
                   if len({e["text"] for e in r["entities"] if e["type"] == "person"})
                   < sum(1 for e in r["entities"] if e["type"] == "person"))
    stats = {
        "seed": SEED,
        "passages": len(records),
        "passages_by_source": {s: sum(1 for r in records if r["source"] == s)
                               for s in ("lccc", "wiki")},
        "lccc_dialogues_scanned": lccc_seen,
        "person_mentions": len(persons),
        "address_mentions": len(addrs),
        "passages_with_address": sum(1 for r in records
                                     if any(e["type"] == "address" for e in r["entities"])),
        "english_name_mentions": len(english),
        "english_name_share": round(len(english) / max(1, len(persons)), 4),
        "person_mention_forms": dict(sorted((k, v) for k, v in forms.items()
                                            if k != "address")),
        "passages_with_repeated_name": repeated,
        "person_mentions_per_passage": {str(k): sum(
            1 for r in records
            if sum(1 for e in r["entities"] if e["type"] == "person") == k) for k in (1, 2, 3)},
        "entity_position_thirds": {"start": thirds[0], "middle": thirds[1], "end": thirds[2]},
        "length_mean": round(statistics.mean(lens), 1),
        "length_median": statistics.median(lens),
        "length_max": max(lens),
        "distinct_patterns_used": len(usage),
        "pattern_usage": dict(sorted(usage.items())),
    }
    with open(out_path + ".stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
