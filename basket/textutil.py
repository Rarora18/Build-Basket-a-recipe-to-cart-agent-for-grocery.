"""Tokenizer, Okapi BM25, and hashed character-trigram embeddings."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import numpy as np

from basket.config import EMBED_DIM
from basket.lexicon import PHRASES, TOKEN_MAP

_KEEP = {"couscous", "asparagus", "hummus"}
_TOKEN = re.compile(r"[a-z0-9]+")


def stem(token: str) -> str:
    if token in _KEEP or len(token) <= 3:
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("oes"):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    raw = text.lower().replace("%", " percent ")
    raw = raw.replace("-", " ").replace("_", " ")
    return [stem(tok) for tok in _TOKEN.findall(raw)]


def embed_tokens(text: str) -> list[str]:
    """Tokens plus lexicon expansions. Used by the embedding channel only."""
    raw = text.lower()
    extras: list[str] = []
    for phrase, adds in PHRASES:
        if phrase in raw:
            extras.extend(adds)
    out: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        if not token or token in seen:
            return
        seen.add(token)
        out.append(token)
        for extra in TOKEN_MAP.get(token, ()):
            for piece in tokenize(extra) or [stem(extra)]:
                if piece not in seen:
                    seen.add(piece)
                    out.append(piece)

    for tok in tokenize(text):
        add(tok)
    for extra in extras:
        for tok in tokenize(extra) or [stem(extra)]:
            add(tok)
    return out


def _bucket(key: str, dim: int) -> int:
    digest = hashlib.md5(key.encode()).digest()
    return int.from_bytes(digest[:4], "little") % dim


def embed(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in embed_tokens(text):
        vec[_bucket("w:" + tok, dim)] += 1.0
        if len(tok) < 4:
            continue
        padded = f"#{tok}#"
        for i in range(len(padded) - 2):
            vec[_bucket("g:" + padded[i : i + 3], dim)] += 0.45
    norm = float(np.linalg.norm(vec))
    if norm:
        vec /= norm
    return vec


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(doc) for doc in docs]
        self.avgdl = sum(len(doc) for doc in self.docs) / max(len(self.docs), 1)
        df: dict[str, int] = {}
        for doc in self.docs:
            for tok in set(doc):
                df[tok] = df.get(tok, 0) + 1
        n = len(self.docs)
        self.idf = {
            tok: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for tok, freq in df.items()
        }
        self.tf = [Counter(doc) for doc in self.docs]

    def scores(self, query: str) -> np.ndarray:
        tokens = tokenize(query)
        out = np.zeros(len(self.docs), dtype=np.float64)
        if not tokens:
            return out
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i]) or 1
            score = 0.0
            for tok in tokens:
                freq = tf.get(tok)
                if not freq:
                    continue
                idf = self.idf.get(tok, 0.0)
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += idf * (freq * (self.k1 + 1)) / denom
            out[i] = score
        return out
