"""Data pipeline: synthetic corpus, DeepSelect, batch loading."""
from .synth import SyntheticCorpus
from .select import DeepSelectPipeline
from .dedup import MinHashDeduper
from .loader import BatchLoader

__all__ = ["SyntheticCorpus", "DeepSelectPipeline", "MinHashDeduper", "BatchLoader"]
