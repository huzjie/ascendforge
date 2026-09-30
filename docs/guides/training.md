# 训练指南

```python
from ascendforge.backends import make_backend
from ascendforge.train.trainer import Trainer
backend = make_backend("mock")
trainer = Trainer(backend)
history = trainer.train(steps=500)
```

## 学习曲线
- skill：0.5 → ~1.0（单调）
- loss：随 skill 上升而下降
- 准确率：训练前约 60% → 训练后约 100%
