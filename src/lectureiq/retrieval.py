"""BM25 ranking over lecture chunks: dependency-free and plenty for one lecture."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from lectureiq.text import content_words
from lectureiq.transcript import Chunk


class BM25Index:
    """Okapi BM25 with the usual k1/b defaults."""

    def __init__(self, chunks: Sequence[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self._tf = [Counter(content_words(c.text)) for c in self.chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = (sum(self._len) / len(self._len)) if self._len else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self.chunks)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: str) -> list[float]:
        terms = content_words(query)
        scores = []
        for tf, length in zip(self._tf, self._len, strict=True):
            total = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    norm = self.k1 * (1 - self.b + self.b * length / (self._avg or 1))
                    total += self._idf[t] * f * (self.k1 + 1) / (f + norm)
            scores.append(total)
        return scores

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        """The ``k`` best chunks with a positive score, best first."""
        ranked = sorted(
            zip(self.chunks, self.score(query), strict=True), key=lambda p: p[1], reverse=True
        )
        return [(c, s) for c, s in ranked[:k] if s > 0]
