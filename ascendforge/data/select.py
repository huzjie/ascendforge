"""DeepSelect pipeline wrapper."""
from ..ops.deepselect import select_documents, quality_score, curriculum_order


class DeepSelectPipeline:
    def __init__(self, threshold=0.85, quality_min=0.5, curriculum=True):
        self.threshold = threshold
        self.quality_min = quality_min
        self.curriculum = curriculum

    def run(self, documents):
        return select_documents(documents, self.threshold, self.quality_min, self.curriculum)

    def score_all(self, documents):
        return [quality_score(d) for d in documents]
