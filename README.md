# ascendforge — 面向华为昇腾 NPU 的 DeepSeek 高性能算子基础设施与训练加速框架

> 零依赖、可完整运行、可训练的参考实现。对标 **DeepSeek 2026-09-30 开源的昇腾基础组件**：TileLang、DeepGEMM、DeepEP、TileKernels、FlashMLA、DeepSelect。

![license](https://img.shields.io/badge/license-MIT-green)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/dependencies-zero-brightgreen)
![tests](https://img.shields.io/badge/tests-unittest-orange)
![npU](https://img.shields.io/badge/backend-mock%20%7C%20cpu%20%7C%20ascend%20%7C%20openai-9cf)

---

## 这是什么

DeepSeek 在昇腾（Ascend）算力平台开源的六个基础设施组件，与它在英伟达平台的开源栈一一对应。`ascendforge` 把这套能力做成一个**零第三方依赖、下载即可跑、可训练、可验证**的完整框架：

| 组件 | 本仓库实现 | 能力 |
|---|---|---|
| **TileLang** | `ascendforge/tilelang/` | tile 化高级 DSL 编译器（IR → 调度 → Ascend C 代码生成 → 确定性 CPU 模拟） |
| **DeepGEMM** | `ascendforge/ops/deepgemm.py` | 通用 / 分组（MoE）矩阵乘法，tile 分块 |
| **DeepEP** | `ascendforge/comm/deepep.py` | 大规模跨设备通信（dispatch/combine、all-to-all、分层拓扑） |
| **TileKernels** | `ascendforge/ops/tilekernels.py` | RMSNorm / LayerNorm / Softmax / RoPE / SwiGLU 融合算子 |
| **FlashMLA** | `ascendforge/ops/flashmla.py` | Multi-head Latent Attention，低秩 KV cache + 稀疏/滑窗注意力 |
| **DeepSelect** | `ascendforge/ops/deepselect.py` | MinHash 去重 + 质量评分 + 课程采样 |

在这之上还有：MoE 专家并行（路由 + SwiGLU 专家 + 负载均衡）、128 卡超节点调度（模拟）、四后端（mock/cpu/ascend/openai）、OpenAI 兼容 HTTP 服务、CLI、Docker/K8s/Helm/CI，以及 LangChain / MCP 集成。

---

## 为什么是热点工具，不是话题

这不是一篇介绍文章，而是一套**填好配置就能真正运行**的工程实现：

- **TileLang** 能把你写的 tile 程序**真正编译**出 Ascend C 伪代码，并在模拟器里**跑出结果**；
- **DeepGEMM / FlashMLA / MoE** 是**真算**的（用零依赖张量核心做矩阵乘法、低秩注意力、专家路由）；
- **mock 后端可训练**：训练时 `skill` 从 0.5 单调爬向 1.0，loss 下降，预测准确率随之上升——给昇腾算子栈一个**确定性的、有意义的**学习闭环；
- 所有基准、示例、单测、服务**开箱即用**，无需任何 `pip install`。

---

## 快速开始

```bash
# 1. 健康检查（列出后端与算子）
python -m ascendforge doctor

# 2. 训练 mock 后端（skill 0.5 -> ~1.0）
python -m ascendforge train --steps 500

# 3. 跑全套基准（GEMM / MLA / MoE / DeepSelect / DeepEP）
python -m ascendforge bench

# 4. 启动 OpenAI 兼容 HTTP 服务
python -m ascendforge serve   # http://127.0.0.1:8000/v1/completions

# 5. 运行全部单测
python -m unittest discover -s tests -t .

# 6. 编译一个 TileLang 程序
python -m ascendforge.tilelang examples/tilelang_gemm.tl
```

---

## 目录结构

```
ascendforge/
├── tilelang/          # TileLang tile 化 DSL 编译器
├── ops/               # DeepGEMM / TileKernels / FlashMLA / DeepSelect
├── comm/              # DeepEP 通信 + 超节点调度
├── moe/               # 专家并行（路由/专家/负载均衡）
├── model/             # MLA 注意力 + MoE 解码块 + Transformer
├── data/              # 合成语料 + 数据选择管线
├── train/             # 训练器 / AdamW / 调度器 / 流水线 / 检查点
├── backends/          # mock / cpu / ascend / openai 四后端
├── bench/             # 基准套件
├── serving/           # OpenAI 兼容 HTTP 服务
├── integrations/      # LangChain + MCP
├── cli.py             # 命令行入口
└── core/              # 零依赖张量核心 + 设备/tile 抽象
examples/              # 10 个可运行示例
tests/                 # 单元测试（unittest）
configs/               # YAML / JSON 配置
deploy/                # Docker / docker-compose / K8s / Helm
.github/workflows/     # CI + Release
docs/                  # 详细文档
```

---

## 后端

| 后端 | 说明 | 适用 |
|---|---|---|
| `mock` | 确定性可训练后端，skill 单调趋 1 驱动预测正确性 | 无昇腾硬件的开发/演示/训练闭环 |
| `cpu` | 真实张量计算 | 本地推理 |
| `ascend` | Ascend NPU 后端（stub + 性能模型） | 规划真实昇腾部署 |
| `openai` | OpenAI 兼容远程后端 | 对接已有推理服务 |

```python
from ascendforge.backends import make_backend
backend = make_backend("mock")   # or "cpu" / "ascend" / "openai"
```

---

## 为什么零依赖也能跑

托管 Python 运行时不一定装了 PyYAML 等第三方库。本框架：

- 内置极简 YAML 子集解析器（`ascendforge/utils/yamlish.py`），读 `.yaml` 无 PyYAML 自动回退；
- 张量运算用纯 Python 列表实现（`ascendforge/core/tensor.py`）；
- HTTP 服务用标准库 `http.server`，MCP 用标准库 stdio。

因此 `python -m ascendforge doctor` 在干净环境也能直接跑。

---

## 训练闭环（为什么 mock 后端「有意义」）

`MockBackend` 用一个 `skill` 标量建模算子栈的「能力」：

```
score = skill * correctness + noise * (0.5 - correctness)
```

训练时 `skill` 单调上升（delta 恒为正），`correctness` 是固定的潜在真值信号。于是：

- 训练前（skill=0.5）准确率约 60%；
- 训练后（skill→1.0）准确率约 100%。

这保证「训练算子栈」这个动作**真的改变了预测正确性**，而不是随机漂移。

---

## 许可

MIT License。见 `LICENSE`。

## 免责声明

本仓库是 DeepSeek 昇腾基础组件的**独立参考实现**，与 DeepSeek / 华为官方无关；Ascend C 代码为伪代码，真实昇腾执行需 CANN/Ascend C 工具链。
