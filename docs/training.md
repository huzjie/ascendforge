# 训练

## mock 后端训练闭环

```python
from ascendforge.backends import make_backend
from ascendforge.train.trainer import Trainer

backend = make_backend("mock")
trainer = Trainer(backend)
history = trainer.train(steps=500)
# skill: 0.5 -> ~1.0, loss 单调下降
```

## 为什么 skill 单调趋 1

训练 delta 恒为正：`delta = 0.05*(1-skill) + 0.001`，skill 单调爬向 1.0。
评分 `skill * correctness + noise*(0.5-correctness)` 让 skill 直接驱动正确性，
训练「算子栈」这个动作真的提升了预测准确率。

## 优化器 / 调度器

- AdamW（含 weight decay）
- Warmup + Cosine 学习率调度
- 流水线并行（1F1B 调度）与气泡率估算
