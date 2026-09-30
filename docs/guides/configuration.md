# 配置

`config.yaml` 支持 YAML（无 PyYAML 自动回退内置解析器）与 JSON。

## 关键字段
- `backend`：mock / cpu / ascend / openai
- `tilelang.*`：tile 大小 / 数据类型 / 流水级数
- `gemm.*`：分块参数
- `mla.*`：注意力超参
- `moe.*`：专家数 / top-k / 负载均衡
- `comm.*`：rank 数 / 拓扑
- `train.*`：学习率 / 调度
- `data.*`：语料 / 去重 / 课程
