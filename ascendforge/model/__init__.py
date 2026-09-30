"""Model architecture: MLA attention, MoE decoder block, transformer."""
from .mla import MultiHeadLatentAttention
from .block import MoEDecoderBlock
from .transformer import AscendTransformer
from .model_config import ModelConfig

__all__ = ["MultiHeadLatentAttention", "MoEDecoderBlock", "AscendTransformer", "ModelConfig"]
