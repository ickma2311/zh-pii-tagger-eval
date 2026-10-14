"""Optional post-processing for zh-pii-tagger-35m (and for the base model): drop "address" spans that are only a
number unless an address word is next to them.

A span typed address whose text is only digits / spaces / .,% and at most one trailing 号 is DROPPED unless
  - one of PRE_CUES occurs in the PRE_WINDOW characters before it (same sentence), or
  - the text right after it (POST_WINDOW characters, same sentence) starts with one of POST_CUES.
So "人口3065人" or "24号回家" lose their address tag, while "我住502", "门牌66号" or "5号楼" keep it.
Trade-off measured on zh-pii-numaddr (see the model card): about 18 points less recall on addresses that are
only a number, far fewer plain numbers masked as addresses. Off by default.

Usage:
    from cue_filter import filter_spans
    spans = filter_spans(text, spans)   # spans: list of (start, end, type) character spans
"""
import re

NUM_ADDR = re.compile(r"[\d\s.,，．%°]+号?")
PRE_CUES = ("住", "地址", "门牌", "房间", "房号", "室号", "寝室", "宿舍", "公寓", "搬到", "搬去", "搬进", "寄到", "寄至",
            "送到", "送至", "收货", "收件", "楼号", "单元", "栋", "幢", "号楼", "小区", "村", "路", "街", "巷", "弄", "胡同")
POST_CUES = ("号楼", "号院", "号门", "号房", "楼", "栋", "幢", "座", "单元", "室", "弄", "房")
PRE_WINDOW, POST_WINDOW = 8, 3
SENT_END = "。！？!?\n"

def _clip_pre(text, s):
    lo = max(0, s - PRE_WINDOW); seg = text[lo:s]
    for i in range(len(seg) - 1, -1, -1):
        if seg[i] in SENT_END: return seg[i + 1:]
    return seg

def _clip_post(text, e):
    seg = text[e:e + POST_WINDOW]
    for i, c in enumerate(seg):
        if c in SENT_END: return seg[:i]
    return seg

def bare_number(text, s, e):
    """True if text[s:e] is a number-only span with no address cue nearby (i.e. should not be an address)."""
    if not NUM_ADDR.fullmatch(text[s:e]): return False
    pre, post = _clip_pre(text, s), _clip_post(text, e).lstrip()
    if any(c in pre for c in PRE_CUES): return False
    if any(post.startswith(c) for c in POST_CUES): return False
    return True

def filter_spans(text, spans):
    """Drop address spans that are bare numbers; spans are (start, end, type)."""
    return [(s, e, t) for s, e, t in spans if not (t == "address" and bare_number(text, s, e))]
