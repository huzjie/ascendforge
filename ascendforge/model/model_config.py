"""Model hyperparameters."""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    hidden_size: int = 512
    num_heads: int = 8
    kv_lora_rank: int = 64
    qk_rope_dim: int = 32
    num_layers: int = 12
    num_experts: int = 8
    top_k: int = 2
    expert_hidden: int = 1024
    vocab_size: int = 50257
    max_seq_len: int = 2048
    sliding_window: int = 256
    sparse_attention: bool = True
