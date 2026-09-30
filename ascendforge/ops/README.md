# ascendforge.ops

DeepSeek 昇腾算子库的零依赖参考实现：

| 模块 | 组件 | 说明 |
|---|---|---|
| `deepgemm` | DeepGEMM | 通用 / 分组（MoE）矩阵乘法，tile 化 |
| `tilekernels` | TileKernels | RMSNorm / LayerNorm / Softmax / RoPE / SwiGLU 融合算子 |
| `flashmla` | FlashMLA | 低秩 KV cache + 稀疏/滑窗注意力 |
| `deepselect` | DeepSelect | MinHash 去重 + 质量评分 + 课程采样 |
