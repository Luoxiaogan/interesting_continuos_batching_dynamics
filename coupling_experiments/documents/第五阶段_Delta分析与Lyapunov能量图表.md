问题 1：线性系统的标准势能方法

1.1 标准方法：Discrete Lyapunov Equation

是的，存在标准方法。 对于稳定的线性系统 $M^T(t+1) = A \cdot M^T(t)$，当所有特征值 $|\lambda_i| < 1$ 时：

Lyapunov 方程：对任意正定矩阵 $Q > 0$，存在唯一正定矩阵 $P > 0$ 满足：
$$A^\top P A - P = -Q$$

势能函数：
$$V(x) = (x - x^)^\top P (x - x^)$$

能量衰减：
$$V(M^T(t+1)) - V(M^T(t)) = -M^T(t) - x^*_Q^2 < 0$$

1.2 能否用于 Simulation？

可以，但有限制。文档已数值验证：

| 性质                                             |   结果    |
|--------------------------------------------------|-----------|
| $V^T(t+1) < V^T(t)$（Theory 能量递减）           | 199/199 ✓ |
| $V^S(t+1) < V^S(t)$（Simulation 能量也递减）     | 199/199 ✓ |
| $V^T(t) \geq V^S(t)$（Theory 能量 ≥ Simulation） | 200/200 ✓ |

1.3 Eviction 的耗散作用

直觉正确：Eviction 是一种"耗散投影"。

从 知识文档.md 的框架：
- Theory = 线性系统 $x \to Ax$（能量按特征值衰减）
- Simulation = $x \to \mathcal{P}(Ax)$（投影算子额外耗散能量）

数学解释：当 Eviction 发生时，一个"大负值"被分散：
- Stage 0: $a \to 0$（截断）
- Stage k: 减少一定量（eviction）

二次型的性质：将偏差分散到多个方向，通常减少总能量。

但问题是：$V^T \geq V^S$ 不能直接推出 $G^T \geq G^S$，因为 $V$ 是 $L_2$ 范数，$G$ 是 $L_\infty$ 范数差。

---
问题 2：Lyapunov 矩阵与特征方程的关系

2.1 特征方程（from multiple_discrete.tex）

tex 文件中的特征方程是：
$$F(\lambda) = (l_0+1)\lambda^{l_B-1} + \sum_{m=1}^{l_A-1}(l_0+m+1)\lambda^{l_B-1-m} + (1-p)\sum_{m=l_A}^{l_B-1}(l_0+m+1)\lambda^{l_B-1-m} = 0$$

这描述的是 admission 序列 $D_t$ 的递推关系。

2.2 关系图

特征方程 F(λ) = 0
        ↓
    特征值 λ_1, ..., λ_{l_B-1}
        ↓
    状态转移矩阵 A 的特征值
        ↓
    Lyapunov 方程: A^T P A - P = -Q
        ↓
    Lyapunov 矩阵 P
        ↓
    能量函数 V(x) = x^T P x

2.3 具体关系

特征值决定 $P$ 的性质：
- 若 $\rho(A) = \max|\lambda_i| < 1$（coprime 条件保证），则 Lyapunov 方程有正定解 $P$
- $P$ 的特征值与 $A$ 的特征值相关，但不是简单对应

从 Lyapunov_energy_analysis.md：
- 对于 $l_0=50, l_A=4, l_B=7$，$|\lambda_{max}| = 0.988 < 1$
- $P$ 的特征值：$1.0, 3.7, 4.8, 6.0, 14.8, 51.0, 166.6$（均正）

---
问题 3：是否有助于证明 G Dominance？

3.1 直接用 Lyapunov 能量：不够

文档已指出：
无法直接从 $V^T \geq V^S$ 推出 $G^T \geq G^S$

因为：
- $V$ 是加权二次型（$L_2$ 范数相关）
- $G$ 是 $L_\infty$ 范数差
- 数值显示 $G^2/V$ 的比值在 Theory 和 Simulation 中相近，无法建立不等式

3.2 但提供了重要启示

启示 1：Eviction 确实是耗散机制
- $V^T \geq V^S$ 成立 → Eviction 减少了系统的"总能量"
- 虽然不能直接推出 $G$ 的关系，但暗示 Simulation 更"平滑"

启示 2：可能的证明方向

从 知识文档.md 的第 4 节（凸序理论）：

$G(x) = \max(x) - \min(x)$ 是一个凸函数
截断操作通常会减小变量在凸序上的大小

如果能证明 Simulation 状态是 Theory 状态的某种"截断/条件期望"，则由 Jensen 不等式：
$$G(\mathbb{E}[X]) \leq \mathbb{E}[G(X)]$$

或凸序性质：
$$X \geq_{cx} Y \implies \mathbb{E}[\phi(X)] \geq \mathbb{E}[\phi(Y)] \text{ 对凸函数 } \phi$$

3.3 可能的突破方向

方向 A：构造基于 $G$ 的 Lyapunov-like 函数

定义 $U(M) = G(M)^2$，需证明：
1. $U$ 对 Theory 递减
2. Eviction 不增加 $U$

方向 B：利用投影的非扩张性

若 Eviction 操作 $\mathcal{P}$ 在某种范数下是非扩张的：
$$\mathcal{P}(x) - \mathcal{P}(y) \leq x - y$$

则可能推出 spread 的压缩。

方向 C：直接分析 $\Delta$ 向量的演化

利用 Token Balance 约束 $\sum W_s \Delta_s = 0$ 来限制 $\Delta$ 的分布。

---
问题 4：可以做哪些实验验证？

基于现有 codebase，我建议以下实验：

4.1 能量相关实验

| 实验                                | 目的                                 | 代码修改                       |
|-------------------------------------|--------------------------------------|--------------------------------|
| 计算并追踪 Lyapunov 能量 $V^T, V^S$ | 验证 $V^T \geq V^S$ 在更多参数下成立 | 在 metrics.py 添加能量计算函数 |
| 计算能量衰减率                      | 比较 Theory 和 Simulation 的衰减速度 | 拟合 $V(t) \sim e^{-\alpha t}$ |
| 尝试不同的 $Q$ 矩阵                 | 找到使 $V$ 与 $G$ 关系更紧密的 $Q$   | 参数化 $Q$ 并扫描              |

4.2 $\Delta$ 向量相关实验

| 实验                                        | 目的                     | 代码修改                             |
|---------------------------------------------|--------------------------|--------------------------------------|
| 追踪 $\Delta_s(t)$ 的完整演化               | 理解 $\Delta$ 向量的结构 | 在 state_vectors.csv 中添加 $\Delta$ |
| 验证 $\Delta_{m^S} \geq 0$ 在更多参数下成立 | 增强数值证据             | 扫描参数空间                         |
| 分析 $\sum W_s \Delta_s = 0$ 的约束         | 理解正负 $\Delta$ 的分布 | 可视化 $\Delta$ 向量                 |

4.3 Eviction 相关实验

| 实验                             | 目的                           | 代码修改                               |
|----------------------------------|--------------------------------|----------------------------------------|
| 统计 Eviction 位置 vs $m^S$ 位置 | 验证"$m^S$ 受保护"的观察       | 在 trajectory.csv 添加 $m^S, n^S$      |
| 计算 Eviction 前后的 $G$ 变化    | 量化 Eviction 对 spread 的影响 | 在 Simulation 中记录 eviction 前后状态 |
| 分析不同 Eviction 策略           | 理解优先级策略的作用           | 实现替代策略（如随机 eviction）        |

4.4 范数关系实验

| 实验                    | 目的                            | 代码修改               |
|-------------------------|---------------------------------|------------------------|
| 计算 $G^2/V$ 比值的分布 | 理解 $L_2$ 和 $L_\infty$ 的关系 | 在 metrics.py 计算比值 |
| 构造 $G$-based 能量函数 | 测试 $U(M) = G(M)^2$ 是否递减   | 添加新的能量函数       |
| 验证凸序关系            | 测试 $M^T \geq_{cx} M^S$？      | 计算各种凸函数的期望   |

4.5 参数敏感性实验

| 实验                   | 目的                                |
|------------------------|-------------------------------------|
| 扫描 $l_0$             | G Dominance 是否在 $l_0$ 小时失效？ |
| 扫描 $p$（不等到达率） | 非对称情况的表现？                  |
| 测试非 coprime 情况    | 不稳定系统的 $G$ 关系？             |

---
我的建议：优先实验

最有价值的实验（按优先级）：

1. 验证 $\Delta_{m^S} \geq 0$ 在广泛参数下成立
- 这是证明的核心引理
- 如果找到反例，需要修改猜想
2. 追踪 $\Delta$ 向量演化 + 可视化
- 理解 $\Delta$ 的结构可能启发证明方向
- 观察 $\Delta$ 是否有"单峰性"或其他模式
3. Eviction 对 $G$ 的瞬时影响
- 在 eviction 发生的那一步，$G$ 是增加还是减少？
- 如果总是减少，可能直接证明 Eviction 压缩 spread

---

# 可视化扩展计划：Delta 分析与 Lyapunov 能量图表

## 概述

为 coupling_experiments 添加 7 个新的可视化函数，用于展示 4.1-4.4 实验的分析结果。

## 修改文件

1. **`coupling_experiments/visualize_coupling.py`** - 添加新的绘图函数
2. **`coupling_experiments/run_coupling_experiment.py`** - 调用新的可视化函数

---

## 新增可视化函数设计

### 1. `plot_lyapunov_energy()` - Lyapunov 能量分析图

**文件**: `visualize_coupling.py`

**布局**: 2 行 1 列子图
- 上图：V^T(t) 和 V^S(t) 对数尺度演化曲线
- 下图：能量比值 V^T(t) / V^S(t) 随时间变化

**参数**:
```python
def plot_lyapunov_energy(
    lyapunov_data: List[Dict],  # 包含 V_theory, V_sim
    output_path: Optional[Path] = None,
    title: str = "Lyapunov Energy Analysis"
) -> None:
```

**关键设计**:
- 上图使用 `ax.semilogy()` 对数尺度
- 颜色：Theory 蓝色、Simulation 绿色
- 下图添加 y=1 参考线（虚线）
- figsize: `(12, 8)`

**输出文件**: `lyapunov_energy.png`

---

### 2. `plot_delta_heatmap()` - Delta 向量热力图

**布局**: 单个热力图

**参数**:
```python
def plot_delta_heatmap(
    delta_data: List[Dict],  # 包含 delta_vec
    output_path: Optional[Path] = None,
    title: str = "Delta Vector Heatmap"
) -> None:
```

**关键设计**:
- 横轴：时间步 t (0~steps)
- 纵轴：stage s (0~max_stage)
- 颜色映射：`RdBu_r`（红=正，蓝=负，白=0）
- 使用 `plt.imshow()` 或 `ax.pcolormesh()`
- 添加 colorbar
- figsize: `(14, 6)`

**数据处理**:
- 解析 `delta_vec` 字符串为 numpy 数组
- 构建 2D 矩阵 (stages × time_steps)

**输出文件**: `delta_heatmap.png`

---

### 3. `plot_delta_extrema()` - Delta 极值追踪图

**布局**: 2 行 1 列子图
- 上图：Δ_{m^S}(t) 和 Δ_{n^S}(t) 的值
- 下图：m^S(t) 和 n^S(t) 的位置

**参数**:
```python
def plot_delta_extrema(
    delta_data: List[Dict],  # 包含 delta_at_m_S, delta_at_n_S, m_S, n_S
    output_path: Optional[Path] = None,
    title: str = "Delta at Extrema Tracking"
) -> None:
```

**关键设计**:
- 上图：
  - Δ_{m^S} 蓝色实线
  - Δ_{n^S} 红色实线
  - y=0 参考线（黑色虚线，关键！）
  - 高亮 Δ_{m^S} < 0 的区域（如果有）
- 下图：
  - m^S 蓝色散点/阶梯图
  - n^S 红色散点/阶梯图
- figsize: `(12, 8)`

**输出文件**: `delta_extrema.png`

---

### 4. `plot_case_distribution()` - Case 分布图

**布局**: 1 行 2 列子图
- 左图：饼图显示 A/B/C/D 分布
- 右图：堆叠面积图显示随时间的 case 累积

**参数**:
```python
def plot_case_distribution(
    delta_data: List[Dict],  # 包含 case
    output_path: Optional[Path] = None,
    title: str = "Case Distribution Analysis"
) -> None:
```

**关键设计**:
- 颜色映射：
  - Case A: 绿色 (理想情况)
  - Case B: 黄色
  - Case C: 橙色
  - Case D: 红色 (最差情况)
- 饼图显示百分比
- 右图使用 `ax.stackplot()` 或简单的滚动计数
- figsize: `(14, 6)`

**输出文件**: `case_distribution.png`

---

### 5. `plot_G_vs_V_relation()` - G 与 V 的关系图

**布局**: 单个散点图

**参数**:
```python
def plot_G_vs_V_relation(
    lyapunov_data: List[Dict],  # 包含 V_theory, V_sim
    delta_data: List[Dict],      # 包含 theory_G_merge, sim_G_merge
    output_path: Optional[Path] = None,
    title: str = "G² vs V Relationship"
) -> None:
```

**关键设计**:
- x 轴：V (能量)
- y 轴：G² (Spread 平方)
- Theory: 蓝色圆点
- Simulation: 绿色方点
- 可选：拟合线显示 G² ∝ V 关系
- 双对数尺度可能更清晰
- figsize: `(10, 8)`

**输出文件**: `G_vs_V_relation.png`

---

### 6. `plot_analysis_dashboard()` - 综合仪表板

**布局**: 2 行 3 列子图网格

| (0,0) G^T vs G^S | (0,1) V^T vs V^S | (0,2) Δ_{m^S} 时间序列 |
| (1,0) Case 饼图 | (1,1) 能量比值 V^T/V^S | (1,2) G²/V 比值 |

**参数**:
```python
def plot_analysis_dashboard(
    trajectory: List[Dict],
    lyapunov_data: List[Dict],
    delta_data: List[Dict],
    output_path: Optional[Path] = None,
    title: str = "Analysis Dashboard"
) -> None:
```

**关键设计**:
- figsize: `(18, 12)`
- 每个子图精简版，只展示关键信息
- 统一的标题格式
- `plt.tight_layout()` 确保布局美观

**输出文件**: `analysis_dashboard.png`

---

### 7. `plot_eviction_delta_relation()` - Eviction 与 Delta 的关系图

**布局**: 2 行 1 列子图
- 上图：eviction 发生的时间步标记 + Δ_{m^S} 曲线
- 下图：eviction 位置 vs m^S 位置的对比

**参数**:
```python
def plot_eviction_delta_relation(
    trajectory: List[Dict],      # 包含 eviction 信息
    delta_data: List[Dict],      # 包含 m_S, delta_at_m_S
    output_path: Optional[Path] = None,
    title: str = "Eviction vs Delta Relation"
) -> None:
```

**关键设计**:
- 上图：
  - Δ_{m^S} 曲线
  - 垂直虚线标记 eviction 发生的时间步
  - y=0 参考线
- 下图：
  - eviction 的 stage 分布（哪个 stage 被 evict）
  - m^S 位置对比
- figsize: `(12, 8)`

**输出文件**: `eviction_delta_relation.png`

---

## run_coupling_experiment.py 修改

在 `main()` 函数的可视化部分添加：

```python
# 新增可视化
from visualize_coupling import (
    plot_lyapunov_energy,
    plot_delta_heatmap,
    plot_delta_extrema,
    plot_case_distribution,
    plot_G_vs_V_relation,
    plot_analysis_dashboard,
    plot_eviction_delta_relation
)

# 1. Lyapunov 能量分析（如果有能量数据）
if results.get('lyapunov_analysis'):
    plot_lyapunov_energy(
        results['lyapunov_analysis'],
        output_path=output_dir / 'lyapunov_energy.png',
        title=f"Lyapunov Energy: l0={config['l0']}, l_A={config['l_A']}, l_B={config['l_B']}"
    )

# 2. Delta 热力图
plot_delta_heatmap(
    results['delta_analysis'],
    output_path=output_dir / 'delta_heatmap.png',
    title=f"Delta Vector: l0={config['l0']}, l_A={config['l_A']}, l_B={config['l_B']}"
)

# 3. Delta 极值追踪
plot_delta_extrema(
    results['delta_analysis'],
    output_path=output_dir / 'delta_extrema.png',
    title=f"Delta at Extrema: l0={config['l0']}, l_A={config['l_A']}, l_B={config['l_B']}"
)

# 4. Case 分布
plot_case_distribution(
    results['delta_analysis'],
    output_path=output_dir / 'case_distribution.png',
    title=f"Case Distribution: l0={config['l0']}, l_A={config['l_A']}, l_B={config['l_B']}"
)

# 5. G vs V 关系（如果有能量数据）
if results.get('lyapunov_analysis'):
    plot_G_vs_V_relation(
        results['lyapunov_analysis'],
        results['delta_analysis'],
        output_path=output_dir / 'G_vs_V_relation.png',
        title=f"G² vs V: l0={config['l0']}, l_A={config['l_A']}, l_B={config['l_B']}"
    )

# 6. 综合仪表板
plot_analysis_dashboard(
    trajectory,
    results.get('lyapunov_analysis', []),
    results['delta_analysis'],
    output_path=output_dir / 'analysis_dashboard.png',
    title=f"Dashboard: l0={config['l0']}, l_A={config['l_A']}, l_B={config['l_B']}"
)

# 7. Eviction 与 Delta 关系
plot_eviction_delta_relation(
    trajectory,
    results['delta_analysis'],
    output_path=output_dir / 'eviction_delta_relation.png',
    title=f"Eviction-Delta: l0={config['l0']}, l_A={config['l_A']}, l_B={config['l_B']}"
)
```

---

## 实现顺序

1. **第一批（核心）**:
   - `plot_delta_extrema()` - 最重要，验证 Δ_{m^S} ≥ 0
   - `plot_delta_heatmap()` - 直观展示 Δ 结构

2. **第二批（能量相关）**:
   - `plot_lyapunov_energy()` - 能量分析
   - `plot_G_vs_V_relation()` - 范数关系

3. **第三批（汇总）**:
   - `plot_case_distribution()` - Case 统计
   - `plot_eviction_delta_relation()` - Eviction 分析
   - `plot_analysis_dashboard()` - 综合仪表板

---

## 样式规范（遵循现有代码）

- **颜色**: Theory=蓝色, Simulation=绿色, 负值=红色
- **figsize**: 标准 `(12, 6-8)`, 复杂图 `(14-18, 8-12)`
- **dpi**: 150
- **字体**: 标签 12pt, 标题 14pt
- **线宽**: 2
- **网格**: `alpha=0.3`

---

## 预期输出文件

| 文件名 | 描述 |
|--------|------|
| `lyapunov_energy.png` | V^T vs V^S 能量演化 |
| `delta_heatmap.png` | Δ 向量时空热力图 |
| `delta_extrema.png` | Δ_{m^S}, Δ_{n^S} 追踪 |
| `case_distribution.png` | A/B/C/D 分布饼图 |
| `G_vs_V_relation.png` | G² vs V 散点图 |
| `analysis_dashboard.png` | 2×3 综合仪表板 |
| `eviction_delta_relation.png` | Eviction 与 Δ 关系 |
