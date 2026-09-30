# 算子库

## DeepGEMM（分组 GEMM）

MoE 中每个专家对输入做 (M×K)@(K×N) GEMM，权重各不相同，分组批量执行。

## FlashMLA（低秩注意力）

- 压缩：`c = concat(k,v) @ w_kv_down`（维度 hidden → kv_lora_rank）
- 解压：`k = c @ w_k`，`v = c @ w_v`
- 注意力：`softmax(Q Kᵀ) V`，可选稀疏/滑窗掩码

KV cache 从 `2×T×hidden` 降到 `T×kv_lora_rank`，压缩比约 `2×hidden/kv_lora_rank`。

## DeepSelect（数据选择）

1. MinHash 签名 + Jaccard 近似 → 去重
2. 质量评分（长度/独特率/标点密度）→ 过滤
3. 课程排序（易→难）→ 采样
