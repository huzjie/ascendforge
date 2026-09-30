"""Deterministic, high-entropy RNG helpers (md5-seeded).

These power the mock backend: the same key always yields the same values,
which makes training monotone and benchmarks reproducible.
"""
import hashlib
import random


def _seed(key):
    return int.from_bytes(hashlib.md5(str(key).encode("utf-8")).digest()[:8], "big")


def stable_float(key, lo=0.0, hi=1.0):
    r = random.Random(_seed(key))
    return r.uniform(lo, hi)


def stable_ints(key, n, lo=0, hi=1000):
    r = random.Random(_seed(key))
    return [r.randint(lo, hi) for _ in range(n)]


def stable_vector(key, n, lo=-1.0, hi=1.0):
    r = random.Random(_seed(key))
    return [r.uniform(lo, hi) for _ in range(n)]


def stable_choice(key, items):
    if not items:
        return None
    r = random.Random(_seed(key))
    return r.choice(items)
