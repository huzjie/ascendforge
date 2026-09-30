# 组件：TileLang

## 定位
tile 化高级 DSL 编译器，把 Ascend C 底层指令封装成高级编程模型，不损失硬件性能。

## 编程模型
以 `Tile` 为计算单位：`tile <name> : <dtype>[M, K] @ <memory>`，内存层级 GM / L1 / L0C / UB。

## 编译流程
`源程序(.tl) -> parser -> TileProgram(IR) -> scheduler -> codegen(Ascend C + CPU) -> Simulator`

## 关键文件
- `ir.py`：Tile / Op / Loop / TileProgram 数据结构
- `parser.py`：迷你 tile DSL 语法解析
- `scheduler.py`：流水、融合、循环切分
- `codegen.py`：Ascend C 伪代码与 CPU 转译
- `simulator.py`：确定性 CPU 模拟执行
- `compiler.py`：编译入口

## 示例
见 `examples/tilelang_gemm.tl` 与 `examples/08_tilelang_compile.py`。
