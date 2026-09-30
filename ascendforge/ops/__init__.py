"""High-performance operator libraries (DeepSeek Ascend infrastructure components)."""
from .deepgemm import gemm, grouped_gemm, GemmConfig
from .tilekernels import rmsnorm, layernorm, softmax, rope, swiglu, fused_mlp
from .flashmla import mla_forward, MlaContext
from .deepselect import select_documents, minhash_dedup, quality_score, curriculum_order

__all__ = [
    "gemm", "grouped_gemm", "GemmConfig",
    "rmsnorm", "layernorm", "softmax", "rope", "swiglu", "fused_mlp",
    "mla_forward", "MlaContext",
    "select_documents", "minhash_dedup", "quality_score", "curriculum_order",
]

# register the public operators as callable kernels (for `doctor` / registry)
from ..core.registry import register_kernel

for _name in ("gemm", "grouped_gemm", "rmsnorm", "softmax", "layernorm",
              "rope", "swiglu", "fused_mlp", "mla_forward", "minhash_dedup",
              "quality_score", "select_documents"):
    try:
        register_kernel(_name)(globals()[_name])
    except KeyError:
        pass
''
