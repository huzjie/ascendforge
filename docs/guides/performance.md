# 性能调优

- GEMM 分块：增大 block_m/n 提升局部性，减小 block_k 降低中间量
- MoE：capacity_factor 控制每个专家的 token 上限
- FlashMLA：kv_lora_rank 越小 KV cache 越省，但精度越低
- 通信：分层拓扑比 ring 在超节点场景跳数更少
