# 服务指南

```bash
python -m ascendforge serve
curl http://127.0.0.1:8000/v1/models
curl -X POST http://127.0.0.1:8000/v1/completions -d '{"prompt":"hello"}'
```

OpenAI 兼容：`/v1/models`、`/v1/completions`、`/health`。
