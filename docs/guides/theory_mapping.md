# 理论工作集成 - 模拟与论文的对应关系

> 建立数学理论与代码实现的精确对应

## 论文结构与模拟代码映射

**论文主文件**: `/Users/ruicheng/Library/CloudStorage/Dropbox-MIT/Ao Ruicheng/应用/Overleaf/LLM_serving/main.tex`

### 核心.tex文件与模拟器的对应

| LaTeX文件 | 理论内容 | 对应模拟器 | 验证方式 |
|----------|---------|-----------|---------|
| `single_discrete.tex` | Theorem 1: Greedy策略不稳定性 | `multi_type_simulator.py` (单类型) | 观察limit cycle收敛 |
| `multiple_discrete.tex` | Theorem 2: GCD稳定性条件 | `multi_type_simulator.py` (多类型) | 对比互质/非互质情况 |
| `admission_control.tex` | Rate-limited准入策略 | `llm_scheduler_simulator_real.py` | 阈值扫描实验 |
| `Discrete_time.tex` | 离散时间系统定义 | 所有模拟器的基础框架 | - |

## 关键数学概念与代码变量对照表

### 状态变量

| 数学符号 | LaTeX定义 | Python变量 | 文件位置 |
|---------|----------|-----------|---------|
| $X_n$ | 时间n的系统状态向量 | `self.state` | `multi_type_simulator.py` |
| $x_n^{(i)}$ | 阶段i的请求数 | `self.state[length][type]` | |
| $Q_n$ | 队列长度 | `self.queue_length` | `admission_control/` |
| $B$ | GPU内存容量（tokens） | `self.B` | 所有模拟器 |
| $l_0$ | 初始prompt长度 | `request_type[0]` | |
| $l_1$ | decoding长度 | `request_type[1]` | |
| $w$ | 内存权重向量 $[l_0+1, ..., l_0+l_1]$ | `weight` (计算中) | |

### 参数与速率

| 数学符号 | LaTeX定义 | Python变量 | 备注 |
|---------|----------|-----------|-----|
| $\lambda$ | 到达率 | `self.arrival_rates` | 列表（多类型） |
| $\lambda_A, \lambda_B$ | 类型A, B到达率 | `self.arrival_rates[0], [1]` | |
| $p$ | 流量混合比例 $\lambda_A/(\lambda_A+\lambda_B)$ | 代码中动态计算 | |
| $b_0, b_1$ | 服务时间参数 $s(n) = b_0 + b_1 Z(n)$ | `self.b0, self.b1` | |
| $\Delta T$ | 批次迭代时长 | `service_time` | 每批次计算 |

### 平衡点与稳定性

| 数学符号 | LaTeX定义 | 验证方法 | 代码位置 |
|---------|----------|---------|---------|
| $x^*$ | No-eviction平衡点 | 观察长期状态收敛值 | CSV分析 |
| $c_i$ | Level-i limit cycle常数 | 观察周期轨道幅度 | `state_differences` |
| $g = \gcd(l_A, l_B)$ | 最大公约数 | 对比实验设计 | 参数配置 |
| $P^{(i)}$ | Poincaré映射 | - | 仅理论分析 |

## 定理验证实验设计

### Theorem 1: Greedy Instability

**理论陈述** (`single_discrete.tex`, lines 88-91):

> 假设到达率 $\lambda$ 足够大使得系统运行在内存边界。系统具有 $l_1$ 个不同的平衡点（1个固定点 + $l_1-1$ 个limit cycles）。前 $l_1-1$ 个平衡点不稳定，只有最后一个平衡点（level-$(l_1-1)$）渐近稳定。

**模拟验证策略**:

```python
# 实验配置
request_types = [(2, 5)]  # 单类型: l0=2, l1=5
B = 50
arrival_rates = [20.0]  # 高负载，确保overloaded

# 预期结果
# - 初始状态: 任意X0
# - 最终状态: 收敛到level-4 limit cycle (最差平衡点)
# - 观察指标: state_differences图应显示最终plateau（非指数衰减到0）
```

**检查方法**:

1. 运行模拟1000批次
2. 查看 `state_differences_from_0_jump_1.png`:
   - 前期: 差异下降（接近某个平衡点）
   - 后期: 差异稳定在非零值（limit cycle的周期性）
3. 计算最终吞吐量，应等于 $c_{l_1-1} = B/(l_1(l_0+l_1))$

### Theorem 2: GCD Stability Condition

**理论陈述** (`multiple_discrete.tex`, lines 120-122):

> 假设到达率足够大使得系统运行在内存边界。令 $g = \gcd(l_A, l_B)$。
> - 若 $g > 1$: no-eviction平衡点不稳定
> - 若 $g = 1$: no-eviction平衡点渐近稳定

**模拟验证策略**:

#### 实验组1: 互质对（g=1）

```python
# 配置1: gcd(2, 3) = 1
request_types = [(5, 2), (5, 3)]
# 配置2: gcd(3, 5) = 1
request_types = [(5, 3), (5, 5)]

# 预期结果: 收敛到no-eviction equilibrium
# state_differences应指数衰减到0
```

#### 对照组: 非互质对（g>1）

```python
# 配置3: gcd(2, 4) = 2
request_types = [(5, 2), (5, 4)]
# 配置4: gcd(3, 6) = 3
request_types = [(5, 3), (5, 6)]

# 预期结果: 不收敛或收敛到limit cycle
# state_differences应稳定在非零值或振荡
```

**统计分析**:

```python
# 加载最后100个批次的状态差异
final_diffs = pd.read_csv('...')
final_diffs_filtered = final_diffs[final_diffs['batch'] > 900]

# 检验收敛性
mean_diff = final_diffs_filtered.groupby('length')['difference'].mean()
convergence = (mean_diff < 1e-6).all()  # 阈值根据精度调整

# 分组对比
print(f"gcd=1组: 收敛率 = {convergence_rate_g1}")
print(f"gcd>1组: 收敛率 = {convergence_rate_g_gt_1}")
# 预期: convergence_rate_g1 ≈ 100%, convergence_rate_g_gt_1 ≈ 0%
```

## 数学符号在代码中的实现细节

### Decode Priority计算

**数学定义**:
$$\text{priority}(x_n^{(j)}) = j \quad \text{(当前decode次数)}$$

**代码实现** (`multi_type_simulator.py`):

```python
# 对于处于长度length的请求
current_length = length
decode_count = current_length - l0  # 这就是priority

# 排序: 按decode_count从小到大
# 驱逐时优先驱逐decode_count最小的
sorted_stages = sorted(state.keys(), key=lambda l: l - l0)
```

### n-proportional Eviction

**数学定义**:
$$\text{evict}_{\text{type } k} = \frac{\lambda_k}{\sum_i \lambda_i} \times \text{total\_eviction}$$

**代码实现**:

```python
# 同一decode层级的不同类型，按到达率比例驱逐
total_at_stage = sum(state[length])
proportion_type_k = arrival_rates[k] / sum(arrival_rates)
evict_amount_k = proportion_type_k * total_eviction_needed
```

### Memory Constraint

**数学定义**:
$$\sum_{j=0}^{l_1-1} \sum_{k} (l_0 + j) \cdot x_n^{(j)}_k \leq B$$

**代码实现**:

```python
def compute_memory_usage(state, request_types):
    total_memory = 0
    for length, counts_by_type in state.items():
        memory_weight = length  # 当前长度即为内存权重
        total_memory += memory_weight * sum(counts_by_type)
    return total_memory

# 驱逐决策
if compute_memory_usage(state_after_admission) > self.B:
    # 需要驱逐...
```

## 相空间可视化与理论对应

**`simultaion_of_the_root/` 模块**:

- **`stable_condition.py`**: 绘制稳定流形 (stable manifold)
  - 对应: `single_discrete.tex` 中矩阵 $B_1, B_2^{(i)}$ 的特征向量
  - 可视化: 3D相空间中的吸引子和排斥子

- **`3d_draw.py`**: 绘制状态轨迹在相空间中的演化
  - 对应: 系统状态 $X_n$ 的时间演化轨迹
  - 展示: 从不同初始条件出发的收敛行为

- **`different_init.py`**: 测试多个初始条件
  - 验证: 吸引域 (basin of attraction) 的形状
  - 对应理论: 平衡点的局部/全局稳定性

## 实验结果与论文图表对应

**建议的实验-图表映射** (在 `experiments/README.md` 中维护):

```markdown
# 实验与论文图表对应关系

## Figure 1: Single-Class Limit Cycle Convergence
- 实验配置: experiments/exp_fig1_single_class_config.json
- 数据位置: experiments/archive/fig1_single_class/
- 生成脚本: visualization.py --plot-type state_evolution

## Figure 2: GCD Stability Comparison
- 实验配置:
  - 互质组: experiments/exp_fig2_coprime_config.json
  - 非互质组: experiments/exp_fig2_non_coprime_config.json
- 数据位置: experiments/archive/fig2_gcd_comparison/
- 生成脚本: custom_analysis/plot_convergence_comparison.py

## Table 1: Throughput and Fairness Metrics
- 实验配置: experiments/exp_table1_metrics_config.json
- 分析脚本: custom_analysis/compute_metrics.py
```

## 快速查找表

| 数学符号 | 代码变量 | 文件 |
|---------|---------|------|
| $X_n$ | `self.state` | `multi_type_simulator.py` |
| $x_n^{(i)}$ | `self.state[length][type]` | |
| $B$ | `self.B` | 所有模拟器 |
| $l_0, l_1$ | `request_type[0], [1]` | |
| $\lambda$ | `self.arrival_rates` | |
| $s(n) = b_0 + b_1 Z(n)$ | `self.b0 + self.b1 * batch_size` | |
| $x^*$ | 观察CSV最终收敛值 | 分析结果 |
| $\gcd(l_A, l_B)$ | `math.gcd(l_A, l_B)` | 参数设计 |

---

**相关文档**:
- [实验可重复性](experiment_reproducibility.md)
- [实验工作流](experiment_workflow.md)
- [编程规范](coding_standards.md)
