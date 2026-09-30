# 组件：DeepSelect

## 定位
高效数据筛选。

## 三阶段
1. **MinHash 去重**：MinHash 签名 + Jaccard 近似，阈值去重
2. **质量评分**：长度因子 + 独特率 + 标点密度
3. **课程采样**：按难度从易到难排序

## API
```python
from ascendforge.ops.deepselect import select_documents, minhash_dedup, quality_score, curriculum_order
```
