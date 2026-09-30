"""Deterministic synthetic corpus generator."""
from ..utils.stable import stable_choice, stable_ints


_TOPICS = ["算子融合", "分布式通信", "稀疏注意力", "低秩压缩", "专家并行",
           "数据去重", "矩阵乘法", "张量分块", "内存访存", "负载均衡"]
_VERBS = ["优化", "加速", "重构", "分析", "调度", "压缩", "缓存", "对齐"]


class SyntheticCorpus:
    def __init__(self, num_docs=1000, seed=42):
        self.num_docs = num_docs
        self.seed = seed

    def generate(self):
        docs = []
        for i in range(self.num_docs):
            n_sent = stable_ints(f"{self.seed}:len:{i}", 1, 3, 8)[0]
            topic = stable_choice(f"{self.seed}:topic:{i}", _TOPICS)
            verb = stable_choice(f"{self.seed}:verb:{i}", _VERBS)
            sent = f"{topic} 的 {verb} 方案 第{i}号 样本，包含若干与 {topic} 相关的技术描述。"
            doc = " ".join([sent] * n_sent)
            docs.append(doc)
        return docs

    def labeled(self):
        """Return (doc, quality_label) pairs where label is the true quality bucket."""
        docs = self.generate()
        return [(d, i % 3) for i, d in enumerate(docs)]
