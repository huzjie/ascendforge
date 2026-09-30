# 排障

- **ImportError: beyond top-level package** → 相对导入层级错误，检查 `from .` vs `from ..`
- **KeyError: unknown backend** → 确认 `backends/__init__.py` 显式 import 各后端模块
- **No module named yaml** → 正常，已回退内置 yamlish 解析器
- **shape mismatch** → 检查 GEMM 维度 `(M,K)@(K,N)`
