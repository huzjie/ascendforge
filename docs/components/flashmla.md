# 组件：FlashMLA

## 定位
Multi-head Latent Attention。DeepSeek 的 MLA 通过低秩投影压缩 KV cache。

## 核心思想
- 压缩：`c = concat(k, v) @ w_kv_down`（hidden → kv_lora_rank）
- 解压：`k = c @ w_k`，`v = c @ w_v`
- 注意力：`softmax(Q Kᵀ) V`，可加稀疏/滑窗掩码

## 收益
KV cache 从 `2×T×hidden` 降到 `T×kv_lora_rank`，压缩比约 `2×hidden/kv_lora_rank`（如 512/64 → 16×）。
