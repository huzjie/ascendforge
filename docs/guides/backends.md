# 后端指南

| 后端 | 何时用 |
|---|---|
| mock | 无硬件时开发、演示、训练闭环 |
| cpu | 本地真实计算推理 |
| ascend | 规划真实昇腾部署（性能模型） |
| openai | 对接已有 OpenAI 兼容推理服务 |

```python
from ascendforge.backends import make_backend, list_backends
print(list_backends())
backend = make_backend("mock")
```
