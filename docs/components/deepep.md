# 组件：DeepEP

## 定位
大规模跨设备通信。MoE 专家并行下，token 需被路由到不同专家（dispatch），专家输出需合并回原位（combine）。

## API
```python
from ascendforge.comm.deepep import DeepEP
ep = DeepEP(num_ranks=8, num_experts=8)
routed = ep.dispatch(tokens)      # {expert_id: [tokens]}
combined = ep.combine(expert_outs)
```

## 拓扑
`ring` / `mesh` / `hierarchical`（128 卡超节点分层）。
