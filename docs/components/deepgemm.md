# 组件：DeepGEMM

## 定位
通用 / 分组矩阵乘法 kernel，tile 分块。分组 GEMM 是 MoE 的工作负载——每个专家一组权重，批量执行。

## API
```python
from ascendforge.ops.deepgemm import gemm, grouped_gemm, GemmConfig
c = gemm(a, b)                    # 普通 GEMM
outs = grouped_gemm(inputs, weights)  # 分组 GEMM（MoE）
```

## 分块
`GemmConfig(block_m=128, block_n=128, block_k=32)`，三重循环累加。
