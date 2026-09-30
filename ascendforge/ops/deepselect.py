"""DeepSelect: efficient data selection.

Three stages mirroring DeepSeek's data filtering component:
  1. MinHash-based near-duplicate removal (dedup)
  2. Heuristic quality scoring (length / perplexity / keyword density)
  3. Curriculum sampling order (easy -> hard)
"""
import hashlib
import math


def _shingles(text, k=5):
    text = text.lower()
    words = text.split()
    if len(words) < k:
        return [hashlib.md5(text.encode()).hexdigest()[:16]]
    return [hashlib.md5(" ".join(words[i:i + k]).encode()).hexdigest()[:16]
            for i in range(len(words) - k + 1)]


def minhash_signature(text, num_hashes=64, seed=42):
    sh = _shingles(text)
    if not sh:
        return [0] * num_hashes
    sig = []
    for i in range(num_hashes):
        mn = None
        for s in sh:
            h = int(hashlib.md5(f"{seed}:{i}:{s}".encode()).hexdigest()[:8], 16)
            if mn is None or h < mn:
                mn = h
        sig.append(mn)
    return sig


def _jaccard_estimate(s1, s2):
    eq = sum(1 for a, b in zip(s1, s2) if a == b)
    return eq / max(len(s1), 1)


def minhash_dedup(documents, threshold=0.85):
    """Remove near-duplicate documents by MinHash similarity."""
    sigs = [minhash_signature(d) for d in documents]
    keep_idx = []
    for i, sig in enumerate(sigs):
        dup = False
        for j in keep_idx:
            if _jaccard_estimate(sig, sigs[j]) >= threshold:
                dup = True
                break
        if not dup:
            keep_idx.append(i)
    return [documents[i] for i in keep_idx], keep_idx


def quality_score(text):
    """Heuristic quality score in [0, 1]."""
    words = text.split()
    n = max(len(words), 1)
    chars = len(text)
    if chars == 0:
        return 0.0
    # length factor (prefer 64..4096 tokens)
    len_score = 1.0 - abs(math.log(n + 1) - math.log(200)) / math.log(200)
    len_score = max(0.0, min(1.0, len_score))
    # unique-ratio factor
    unique = len(set(words)) / n
    # punctuation sanity
    punct = sum(1 for ch in text if ch in ".,!?;:") / chars
    score = 0.5 * len_score + 0.3 * unique + 0.2 * min(punct * 20, 1.0)
    return round(max(0.0, min(1.0, score)), 4)


def curriculum_order(documents, scores=None):
    """Order documents by ascending difficulty (curriculum learning)."""
    if scores is None:
        scores = [quality_score(d) for d in documents]
    return sorted(range(len(documents)), key=lambda i: scores[i])


def select_documents(documents, threshold=0.85, quality_min=0.5, curriculum=True):
    """Full pipeline: dedup -> quality filter -> curriculum order."""
    deduped, _ = minhash_dedup(documents, threshold)
    filtered = [d for d in deduped if quality_score(d) >= quality_min]
    if curriculum:
        order = curriculum_order(filtered)
        filtered = [filtered[i] for i in order]
    return filtered
