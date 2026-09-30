# API：tilelang

- `compile_source(src)` → `{program, ascend_c, cpu}`
- `compile_program(program)` — 从 IR 编译
- `Simulator().feed(name, tensor).run(program)` — 模拟执行
- `TileProgram / Tile / Op / Loop / OpKind` — IR
