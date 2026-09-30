# API：moe

- `TopKRouter(num_experts, top_k)`：`logits / route`
- `SwiGLUExpert(hidden, expert_hidden)`：`forward`
- `MoELayer(...)`：`forward / load_counts`
- `aux_loss / load_balance_loss`
