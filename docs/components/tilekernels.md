# 组件：TileKernels

## 定位
常规向量计算与访存算子：RMSNorm、LayerNorm、Softmax、RoPE、SwiGLU 融合 MLP。

## API
```python
from ascendforge.ops.tilekernels import rmsnorm, layernorm, softmax, rope, swiglu, fused_mlp
```

## 融合
`fused_mlp(x, w_gate, w_up, w_down)` = `down(silu(x@w_gate) * (x@w_up))`，把 SwiGLU 三步融合为一次调用。
