"""ascendforge: DeepSeek Ascend NPU high-performance operator infrastructure & training framework.

A zero-dependency, fully runnable reference implementation inspired by DeepSeek's
open-sourced Ascend infrastructure components (2026-09-30):

- TileLang  -- tile-based high-level DSL compiler targeting Ascend C
- DeepGEMM  -- general / grouped matrix multiplication kernels
- TileKernels -- fused vector & memory-access operators (RMSNorm/Softmax/RoPE/act)
- FlashMLA  -- Multi-head Latent Attention with low-rank KV cache (sparse / sliding)
- DeepSelect -- efficient data selection (MinHash dedup + quality + curriculum)
- DeepEP    -- large-scale cross-device communication (MoE expert parallelism)

Every component runs on a deterministic, trainable mock backend with no external
dependencies, plus real CPU kernels, an Ascend backend stub, an OpenAI-compatible
HTTP server, CLI, Docker/K8s/Helm, CI and LangChain/MCP integrations.
"""
from .version import __version__
from .config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
