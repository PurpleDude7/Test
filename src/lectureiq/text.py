"""Small text helpers shared by retrieval and the offline engine."""

from __future__ import annotations

import re
from collections import Counter

_WORD = re.compile(r"[A-Za-z][A-Za-z\-']+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")

_STOPWORD_TEXT = """
a about above after again against all also am an and any are aren't as at be because been
before being below between both but by can can't cannot could did do does doing don't down
during each few for from further get gets got had has have having he her here hers herself
him himself his how i if in into is isn't it it's its itself just let's like me more most my
myself no nor not now of off on once one only or other ought our ours ourselves out over own
really right same she should so some such than that that's the their theirs them themselves
then there there's these they this those through to too under until up us very was we were
what when where which while who whom why will with would you your yours yourself yourselves
okay ok um uh yeah well going gonna kind sort thing things lot lots way today see actually
basically maybe mean know think say said make makes made want use used using able need
"""
STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def words(text: str) -> list[str]:
    """Lower-cased word tokens."""
    return [w.lower().strip("'-") for w in _WORD.findall(text)]


def content_words(text: str) -> list[str]:
    return [w for w in words(text) if w not in STOPWORDS and len(w) > 2]


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if len(s.split()) >= 4]


def keywords(text: str, n: int = 10) -> list[str]:
    return [w for w, _ in Counter(content_words(text)).most_common(n)]
