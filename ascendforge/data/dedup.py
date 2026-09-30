"""MinHash near-duplicate deduplication."""
from ..ops.deepselect import minhash_dedup


class MinHashDeduper:
    def __init__(self, threshold=0.85, num_hashes=64):
        self.threshold = threshold
        self.num_hashes = num_hashes

    def dedup(self, documents):
        return minhash_dedup(documents, self.threshold)

    def stats(self, before, after):
        return {"before": before, "after": after, "removed": before - after}
