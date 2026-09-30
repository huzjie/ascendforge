# API：core

- `Tensor(data, shape, dtype, device)` — 零依赖张量
- `zeros / ones / randn` — 构造
- `matmul / add / mul / relu / gelu / silu` — 算子
- `softmax / layernorm / rmsnorm / rope` — 归一化与位置编码
- `Device / Tile` — 设备与 tile 抽象
- `register_kernel / list_kernels` — 算子注册
