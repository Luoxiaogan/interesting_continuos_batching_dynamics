# 第二阶段：跟踪 G = max - min

## 1. 背景

在 `single_discrete.tex` 的 "Proof of the collapse" 部分（第639行），引入了一个类似能量泛函的 metric：

$$G_i = \max_{t}\{x_i^{(t)}/w_t\} - \min_{t}\{x_i^{(t)}/w_t\}$$

其中 $w_t = l_0 + t + 1$ 是权重。

该 metric 用于证明 single-type 系统的 collapse 行为：G 单调不减，直到触发更深层 eviction。

## 2. 本阶段目标

在 2-types coprime 实验中追踪一个简化版的 G metric，观察 Theory 和 Simulation 的收敛行为。

## 3. G 的定义（简化版，不除权重）

$$G = \max_{(t, type) \in \text{valid}} x^{(t)}_{type} - \min_{(t, type) \in \text{valid}} x^{(t)}_{type}$$

其中：
- Type A 有效 stages: 0, 1, ..., l_A - 1
- Type B 有效 stages: 0, 1, ..., l_B - 1
- 直接取请求数的 max - min，**不除以权重**

## 4. 实现

### 4.1 新增文件：`metrics.py`

```python
def compute_G(state, l0, l_A, l_B, state_format='stage'):
    """
    计算 G = max(state values) - min(state values)

    Args:
        state: 状态字典
        state_format: 'stage' (Theory) 或 'length' (Simulation)
    """
    values = []
    # 遍历所有有效的 (stage, type) 组合
    # 收集请求数到 values 列表
    return max(values) - min(values)
```

### 4.2 修改：`run_coupling_experiment.py`

- 从 Theory 和 Simulation 的历史中提取每一步的 state
- 调用 `compute_G()` 计算 `theory_G` 和 `sim_G`
- 添加到 trajectory 数据中

### 4.3 修改：`visualize_coupling.py`

新增 `plot_G_comparison()` 函数，绘制两条线的对比图。

## 5. 实验结果

参数：l0=3, l_A=2, l_B=3, B=60, steps=50

| Batch | Theory G | Simulation G |
|-------|----------|--------------|
| 0     | 7.50     | 7.50         |
| 10    | 2.90     | 1.94         |
| 30    | 0.12     | 0.08         |
| 49    | 0.007    | 0.004        |

**观察**：
1. 初始 G = 7.5（只有 stage 0 有请求，其他 stages 为 0）
2. 两条线都单调递减，收敛到接近 0
3. Simulation 的 G 比 Theory 更快趋近 0（因为 eviction 更激进地"平衡"了状态）

## 6. 输出文件

```
outputs/YYYYMMDD_HHMMSS/
├── G_comparison.png     # G 对比图
├── trajectory.csv       # 包含 theory_G, sim_G 列
└── ...
```

## 7. 后续可能的改进

1. **加权版本**：使用 $G_w = \max(x/w) - \min(x/w)$，更接近 tex 的定义
2. **Deviation from Equilibrium**：计算 $\|X - X_{eq}\|$，在 equilibrium 时严格为 0
3. **分 type 计算**：分别追踪 $G_A$ 和 $G_B$

---

**文档版本**：v1.0
**创建日期**：2026-01-04
