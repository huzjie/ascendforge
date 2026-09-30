"""Batch loader with curriculum-friendly ordering."""
from ..core.tensor import Tensor
from ..utils.stable import stable_vector


class BatchLoader:
    def __init__(self, documents, batch_size=8, seq_len=128, key="loader"):
        self.documents = documents
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.key = key

    def __len__(self):
        return max(1, (len(self.documents) + self.batch_size - 1) // self.batch_size)

    def iter_batches(self):
        for b in range(len(self)):
            batch = self.documents[b * self.batch_size:(b + 1) * self.batch_size]
            yield self._encode(batch)

    def _encode(self, batch):
        """Deterministic tokenization (hash-based)."""
        tokens = []
        for doc in batch:
            h = abs(hash(doc)) % 100000
            ids = stable_vector(f"{self.key}:{h}", self.seq_len, 0, 50000)
            tokens.append([int(abs(x)) for x in ids])
        # return as (B, seq_len) tensor
        flat = []
        for row in tokens:
            flat.extend(row)
        return Tensor(flat, (len(batch), self.seq_len), device="npu")
