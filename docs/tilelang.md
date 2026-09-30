# TileLang

TileLang 把 Ascend C 底层指令封装成 tile 化的高级编程模型，不损失硬件性能。

## 编译管线

```
源程序(.tl) -> parser -> TileProgram(IR) -> scheduler(调度/融合/流水) -> codegen(Ascend C + CPU) 
```

## 支持的 IR 节点

| OpKind | 说明 |
|---|---|
| LOAD / STORE | GM↔L1 数据搬运 |
| GEMM | tile 化矩阵乘法 |
| ELEMENTWISE / REDUCE | 逐元素 / 归约算子 |
| ATTENTION | 注意力算子 |
| FUSED | 融合算子 |

## 调度 pass

- `pipeline`：软件流水标注（prefetch）
- `fuse_elementwise`：相邻逐元素算子融合成 FUSED
- `split_loop`：循环切分

## 示例

见 `examples/tilelang_gemm.tl`，编译：

```bash
python -m ascendforge.tilelang examples/tilelang_gemm.tl
```

输出 Ascend C 伪代码（`__aicore__` kernel）。
