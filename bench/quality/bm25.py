"""Okapi BM25, implemented directly.

Fifty lines against a new dependency is an easy trade, and at this corpus
size (a few hundred documents per haystack) there is nothing for a clever
implementation to win. Standard parameters k1=1.5, b=0.75; the idf is the
Robertson-Sparck-Jones form with the +1 inside the log, which keeps terms
appearing in over half the corpus from going negative.

No stopword list and no stemming, deliberately. Both help or hurt depending
on language and domain; the eval compares *arms*, and a vanilla BM25 is the
honest lexical baseline people mean when they say "just use BM25".
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(d) for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 1.0
        self.tf: list[Counter] = [Counter(t) for t in self.doc_tokens]
        df: Counter = Counter()
        for t in self.doc_tokens:
            df.update(set(t))
        n = len(docs)
        self.idf = {
            term: math.log((n - d + 0.5) / (d + 0.5) + 1.0) for term, d in df.items()
        }

    def scores(self, query: str) -> list[float]:
        q_terms = tokenize(query)
        out = [0.0] * len(self.doc_tokens)
        for term in q_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(term)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                out[i] += idf * f * (self.k1 + 1) / denom
        return out
