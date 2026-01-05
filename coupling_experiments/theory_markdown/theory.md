# G Dominance 猜想：Theory vs Simulation 的 Spread 比较

---

## 目录

1. [引言与动机](#1-引言与动机)
2. [系统模型](#2-系统模型)
3. [演化规则](#3-演化规则)
4. [主猜想：G Dominance](#4-主猜想g-dominance)
5. [数值观察](#5-数值观察)
6. [证明尝试与困难](#6-证明尝试与困难)
7. [潜在的证明方向](#7-潜在的证明方向)
8. [开放问题](#8-开放问题)
9. [附录：代码对应](#9-附录代码对应)

---

## 1. 引言与动机

### 1.1 研究背景

我们研究 LLM serving 系统中的连续批处理（continuous batching）动态。系统有两种类型的请求（Type A 和 Type B），它们有不同的服务长度。系统受到 GPU 内存容量 $B$ 的约束。

核心问题：**当系统过载（overloaded）时，如何理解其动态行为？**

### 1.2 两种模型

我们比较两种演化模型：

| 模型 | 特点 | 物理意义 |
|:-----|:-----|:---------|
| **Theory** | 允许负数 admission，无非负约束 | 数学理想化模型 |
| **Simulation** | admission ≥ 0，有 eviction 机制 | 实际物理系统 |

### 1.3 研究目标

**主要目标**：证明 Theory 系统的状态"spread"总是大于等于 Simulation 系统的 spread。

**动机**：
- 若能证明此结论，结合 Theory 系统的收敛性（coprime 条件下），可推出 Simulation 也收敛
- 这为实际系统的稳定性提供理论保证

---

## 2. 系统模型

### 2.1 基本参数

| 符号 | 含义 | 约束 |
|:-----|:-----|:-----|
| $l_0$ | 基础长度（prefill length） | $l_0 \geq 1$ |
| $l_A$ | Type A 的服务长度（decode steps） | $l_A \geq 1$ |
| $l_B$ | Type B 的服务长度（decode steps） | $l_B > l_A$（不失一般性） |
| $B$ | Token 容量约束（GPU memory） | $B > 0$ |
| $\lambda_A, \lambda_B$ | 到达率 | $\lambda_A, \lambda_B > 0$ |
| $p$ | Type A 的到达比例 | $p = \frac{\lambda_A}{\lambda_A + \lambda_B}$ |
| $q$ | Type B 的到达比例 | $q = 1 - p = \frac{\lambda_B}{\lambda_A + \lambda_B}$ |

### 2.2 Stage 与 Length 的关系

请求在系统中经历多个 **stage**（阶段）：

- **Stage $s$**：请求已完成 $s$ 次 decode
- **Length**：请求当前的 token 长度 = $l_0 + s$

```
Stage 0 → Stage 1 → ... → Stage (l-1) → 完成
```

- Type A：经历 stage $0, 1, \ldots, l_A - 1$，在 stage $l_A - 1$ 后完成
- Type B：经历 stage $0, 1, \ldots, l_B - 1$，在 stage $l_B - 1$ 后完成

### 2.3 状态向量

**原始状态**：$X_s^A, X_s^B$ 表示 stage $s$ 的 Type A/B 请求数

**合并状态向量**：定义 $M \in \mathbb{R}^{l_B}$，其中

$$M_s = \begin{cases}
X_s^A + X_s^B & \text{if } s < l_A \\[6pt]
X_s^B \cdot \dfrac{\lambda_A + \lambda_B}{\lambda_B} = \dfrac{X_s^B}{q} & \text{if } l_A \leq s < l_B
\end{cases}$$

**补偿因子的含义**：当 $s \geq l_A$ 时，只有 Type B 存在。为保持"流量平衡"的可比性，我们将 Type B 的数量乘以 $1/q$ 进行补偿。

### 2.4 权重向量

定义 stage $s$ 的 **token 权重**：

$$W_s = \begin{cases}
l_0 + s + 1 & \text{if } s < l_A \\[6pt]
(l_0 + s + 1) \cdot q & \text{if } l_A \leq s < l_B
\end{cases}$$

**解释**：
- 每个请求在 stage $s$ 处理时占用 $(l_0 + s + 1)$ tokens
- 对于合并状态 $M_s$，有效权重需要考虑补偿因子

### 2.5 Token Balance 约束

两系统在所有时刻都满足 **Token Balance**：

$$\sum_{s=0}^{l_B-1} W_s \cdot M_s(t) = B$$

这是因为 overloaded 系统总是填满 GPU 容量。

### 2.6 核心度量：Spread $G$

定义状态向量的 **spread**（跨度）：

$$G(t) = \max_{0 \leq s < l_B} M_s(t) - \min_{0 \leq s < l_B} M_s(t)$$

**物理意义**：$G$ 衡量系统状态的"不均匀程度"。当系统达到平衡时，所有 stage 的（补偿后）请求数应该相等，此时 $G = 0$。

---

## 3. 演化规则

### 3.1 共同的 Advance 步骤

两系统共享相同的 **advance**（推进）规则：

$$M_s(t) \leftarrow M_{s-1}(t-1) \quad \text{for } s = 1, \ldots, l_B - 1$$

- Stage $l_B - 1$ 的请求完成并离开系统
- 其他请求推进到下一个 stage
- Stage 0 需要新的 admission 填充

### 3.2 Theory 系统的 Admission

**Step 1**：计算 advance 后已占用的 tokens

$$E^T(t) = \sum_{s=1}^{l_B-1} W_s \cdot M_s^T(t)$$

**Step 2**：计算 admission（可为负）

$$a^T(t) = \frac{B - E^T(t)}{W_0}$$

**Step 3**：更新 stage 0

$$M_0^T(t) = a^T(t)$$

**关键特性**：
- 当 $E^T > B$ 时，$a^T < 0$（"负 admission"）
- 状态 $M_s^T$ 可以为负数
- 这是一个 **线性动力系统**

### 3.3 Simulation 系统的 Admission + Eviction

Simulation 系统有 **非负约束**：$M_s^S(t) \geq 0$ 对所有 $s, t$。

**Step 1**：计算 advance 后已占用的 tokens

$$E^S(t) = \sum_{s=1}^{l_B-1} W_s \cdot M_s^S(t)$$

**Step 2**：计算期望 admission

$$\tilde{a}(t) = \frac{B - E^S(t)}{W_0}$$

**Step 3**：根据 $\tilde{a}$ 的符号分情况处理

---

#### 情况 A：$\tilde{a} \geq 0$（正常 admission）

$$a^S(t) = \tilde{a}(t), \quad M_0^S(t) = a^S(t)$$

无需 eviction。

---

#### 情况 B：$\tilde{a} < 0$（需要 eviction）

此时 $E^S > B$，需要驱逐部分请求以满足容量约束。

**Eviction 机制（关键！与代码一致）**：

```
按 decode 次数从高到低处理（高 decode = 高优先级 = 优先保留）：

for decode_num in [l_B-1, l_B-2, ..., 1, 0]:  # 从高到低
    该层的请求优先填入 batch
    if 累计 tokens 超过 B:
        该层按比例 evict，使总 tokens = B
        更低层的请求全部 evict
        break
```

**结果**：
- $a^S(t) = 0$（无新 admission）
- 高 stage（接近完成）的请求被保留
- 低 stage（刚进入）的请求被 evict

**物理直觉**：接近完成的请求"投资"已多，优先保留；刚进入的请求可以重新排队。

---

### 3.4 两系统的关键差异总结

| 方面 | Theory | Simulation |
|:-----|:-------|:-----------|
| Admission | 可负 | $\geq 0$ |
| 状态约束 | 无 | $M_s \geq 0$ |
| 处理超载 | 负 admission 在 stage 0 | Eviction 从低 stage 开始 |
| 动力学类型 | 线性 | 非线性（有截断） |
| 守恒量 | Token Balance | Token Balance |

---

## 4. 主猜想：G Dominance

### 4.1 猜想陈述

**猜想（G Dominance）**：设两系统从相同初始状态出发，即 $M^T(0) = M^S(0)$。则对所有 $t \geq 0$：

$$G^T(t) \geq G^S(t)$$

即 **Theory 的 spread 总是大于等于 Simulation 的 spread**。

### 4.2 等价形式

定义：
- $A(t) = \max_s M_s^T(t) - \max_s M_s^S(t)$
- $B(t) = \min_s M_s^S(t) - \min_s M_s^T(t)$

则：
$$G^T - G^S = A + B$$

**猜想等价于**：$A(t) + B(t) \geq 0$ 对所有 $t$。

### 4.3 推论：Simulation 的收敛性

**已知结论**（Theory 收敛性）：当 $\gcd(l_A, l_B) = 1$ 且 $l_0$ 充分大时，Theory 系统指数收敛到平衡态，即：

$$G^T(t) \to 0 \quad \text{as } t \to \infty$$

**推论**：若 G Dominance 成立，则由夹逼定理：

$$0 \leq G^S(t) \leq G^T(t) \to 0$$

因此 **Simulation 也收敛到平衡态**。

---

## 5. 数值观察

### 5.1 实验设置

| 数据集 | $l_0$ | $l_A$ | $l_B$ | $B$ | $\lambda_A$ | $\lambda_B$ | 步数 |
|:------:|:-----:|:-----:|:-----:|:---:|:-----------:|:-----------:|:----:|
| 1 | 10 | 2 | 5 | 60 | 1.0 | 1.0 | 50 |
| 2 | 50 | 4 | 7 | 1000 | 1.0 | 1.0 | 200 |

初始条件：$M(0) = [N, 0, 0, \ldots, 0]$，其中 $N = B / W_0$。

### 5.2 主要数值结果

| 性质 | 数据集 1 | 数据集 2 | 状态 |
|:-----|:--------:|:--------:|:----:|
| $G^T(t) \geq G^S(t)$ | 50/50 ✓ | 200/200 ✓ | **待证明** |
| $\max M^T \geq \max M^S$ | 50/50 ✓ | 200/200 ✓ | **待证明** |
| $\min M^T \leq \min M^S$ | 50/50 ✓ | 200/200 ✓ | **部分证明** |

### 5.3 辅助观察

以下是可能有助于证明的观察：

| 观察 | 结果 | 理论状态 |
|:-----|:-----|:--------:|
| 逐分量：$M_s^T \geq M_s^S$ 对所有 $s$ | **不成立**（约 50% 违反）| N/A |
| $\Delta_{m^S} \geq 0$（见 5.4 节定义） | 100% 成立 | ⚠️ **仅数值观察** |
| $\Delta_{m^S} \geq \Delta_{n^S}$ | 100% 成立 | ⚠️ **仅数值观察** |
| $a^S(t) > a^T(t)$ 的时刻比例 | 约 50%（103/200）| N/A |

### 5.4 关键记号

定义差向量：
$$\Delta_s(t) = M_s^T(t) - M_s^S(t)$$

定义 Simulation 的极值位置：
- $m^S(t) = \arg\max_s M_s^S(t)$（最大值位置）
- $n^S(t) = \arg\min_s M_s^S(t)$（最小值位置）

**关键观察（⚠️ 仅数值验证，无理论证明）**：

$$\Delta_{m^S(t)}(t) \geq 0 \quad \text{对所有 } t$$

即：**在 Simulation 最大值的位置，Theory 的值 ≥ Simulation 的值**。

---

## 6. 证明尝试与困难

### 6.1 部分成功：$B(t) \geq 0$ 的证明

**命题**：当 $\min M^T(t) < 0$ 时，$B(t) \geq 0$。

**证明**：
$$B = \min M^S - \min M^T \geq 0 - \min M^T > 0$$

因为 $M^S \geq 0$（非负约束）而 $\min M^T < 0$。 $\square$

**困难**：当 $\min M^T \geq 0$ 时，需要额外论证 $\min M^S \geq \min M^T$。

### 6.2 核心困难：$A(t) \geq 0$ 的证明

**目标**：证明 $\max M^T \geq \max M^S$。

**可能的路径**：若能证明 $\Delta_{m^S} \geq 0$，则：
$$\max M^T \geq M_{m^S}^T = M_{m^S}^S + \Delta_{m^S} \geq M_{m^S}^S = \max M^S$$

**但**：$\Delta_{m^S} \geq 0$ 目前仅有数值验证，缺乏理论证明。

### 6.3 为什么逐分量比较不成立？

数值显示 $M_s^T \geq M_s^S$ 对约 50% 的 $(t, s)$ 不成立。

**原因分析**：
- 当 $a^T < 0$ 时，$M_0^T < 0 < M_0^S = 0$
- 这个负值传播：$M_1^T < M_1^S, M_2^T < M_2^S, \ldots$
- 但当负值"完成"并退出系统后，Theory 需要大的正 admission 补偿
- 此时 $M_0^T > M_0^S$ 可能成立

关键点：**不同 stage 的 $\Delta_s$ 符号可以不同**。

### 6.4 情况分类分析

对于证明 $\Delta_{m^S} \geq \Delta_{n^S}$（等价于 $G^T \geq G^S$），可分四种情况：

| 情况 | 条件 | 频率 | 分析 |
|:----:|:-----|:----:|:-----|
| A | $\Delta_{m^S} \geq 0$ 且 $\Delta_{n^S} \leq 0$ | ~78% | **显然成立** |
| B | $\Delta_{m^S} \geq 0$ 且 $\Delta_{n^S} > 0$ | ~4% | 需证明 $\Delta_{m^S} \geq \Delta_{n^S}$ |
| C | $\Delta_{m^S} < 0$ 且 $\Delta_{n^S} \leq 0$ | ~18% | 需证明 $|\Delta_{m^S}| \leq |\Delta_{n^S}|$ |
| D | $\Delta_{m^S} < 0$ 且 $\Delta_{n^S} > 0$ | 0% | 数值上不发生 |

**观察**：情况 D 从未发生。若能证明情况 D 不可能，则只需处理情况 B 和 C。

---

## 7. 潜在的证明方向

### 7.1 方向一：Token Balance 约束

**核心方程**：
$$\sum_{s=0}^{l_B-1} W_s \Delta_s(t) = 0$$

这是因为两系统都满足 Token Balance：$\sum W_s M_s^T = \sum W_s M_s^S = B$。

**推论**：正的 $\Delta_s$ 和负的 $\Delta_s$ 的加权和为零。

**可能的应用**：
- 若 $\Delta_{m^S} < 0$（Theory 在最大位置反而更小），则必须存在其他位置 $\Delta_s > 0$ 来补偿
- 这些正 $\Delta$ 位置的 $M^T$ 值更大
- 可能推出矛盾？

### 7.2 方向二：Eviction 的"压缩"效应

**直觉**：Eviction 从低 stage 移除请求，这有"压缩" spread 的效果。

**形式化尝试**：

当 Simulation 发生 eviction 时：
- 高 stage（大值区域）被保留
- 低 stage（可能包含小值或新 admission 的大值）被削减
- 结果：spread 减小

**困难**：这个直觉在某些情况下可能不准确，因为低 stage 也可能有大值（刚发生大 admission 时）。

### 7.3 方向三：耦合论证（Coupling Argument）

**思路**：构造两系统在同一概率空间上的耦合。

**可能的耦合方式**：
- 设 $\delta_s(t) = M_s^T(t) - M_s^S(t)$
- 追踪 $\delta$ 向量的演化
- 证明某种 Lyapunov 函数单调

**困难**：两系统的演化在 eviction 发生时显著不同，难以直接耦合。

### 7.4 方向四：归纳法 + 结构性质

**归纳假设**：不仅假设 $G^T(t) \geq G^S(t)$，还假设 $\Delta$ 向量满足某种结构性质。

**可能的结构性质**：
1. $\Delta$ 向量的"单峰性"
2. $\Delta_{m^S} \geq 0$ 恒成立
3. 某种关于 $\Delta$ 的不变量

**困难**：需要识别正确的归纳假设。

### 7.5 方向五：分析 $m^S$ 位置的特殊性

**问题**：为什么 $\Delta_{m^S} \geq 0$ 总是成立？

**可能的解释**：
- $m^S$ 是 Simulation 最大值位置
- 这个位置通常是某个"累积峰"——连续正 admission 的结果
- 在形成这个峰的时期，Theory 也有正 admission（且可能更大，因为 Theory 有更大的"回弹"）
- 所以 Theory 在这个位置的值 ≥ Simulation

**需要形式化**：如何追踪 $m^S$ 位置的历史？

### 7.6 方向六：特征值分析

Theory 系统是线性的，可以写成：
$$M^T(t+1) = A \cdot M^T(t) + b$$

其中 $A$ 是状态转移矩阵。

**已知**：当 $\gcd(l_A, l_B) = 1$ 时，$A$ 的所有特征值 $|\lambda| < 1$。

**问题**：Simulation 的非线性如何影响收敛？

**直觉**：Eviction 相当于额外的"阻尼"，应该加速收敛而非减缓。

---

## 8. 开放问题

### 8.1 核心问题

1. **证明或证伪 $\Delta_{m^S(t)}(t) \geq 0$**
   - 这是整个证明的关键
   - 目前仅有数值验证

2. **证明当 $\min M^T \geq 0$ 时 $\min M^S \geq \min M^T$**
   - 此时不能用非负约束直接推出

3. **为什么情况 D 不发生？**
   - 即：为什么不会同时有 $\Delta_{m^S} < 0$ 和 $\Delta_{n^S} > 0$？

### 8.2 可能有帮助的子问题

1. **Eviction 位置与 $m^S$ 的关系**
   - $m^S$ 位置是否总是"受保护"的（不被 evict）？
   - 数值显示 $m^S$ 位置累积 eviction = 0

2. **$\Delta$ 向量的演化规律**
   - 能否建立 $\Delta(t+1)$ 和 $\Delta(t)$ 的显式关系？

3. **最大值位置 $m^S$ 的动态**
   - $m^S(t)$ 如何随时间变化？
   - 它与 admission 历史有什么关系？

### 8.3 更弱的结论

如果 G Dominance 太难证明，可以考虑：

1. **渐近版本**：$\limsup_{t \to \infty} G^S(t) \leq \limsup_{t \to \infty} G^T(t)$

2. **概率版本**：$\mathbb{P}(G^T(t) \geq G^S(t)) \to 1$ as $l_0 \to \infty$

3. **时间平均版本**：$\frac{1}{T} \sum_{t=1}^{T} G^S(t) \leq \frac{1}{T} \sum_{t=1}^{T} G^T(t)$

---

## 9. 附录：代码对应

### 9.1 文件结构

```
coupling_experiments/
├── theory_simulator.py      # Theory 系统实现
├── metrics.py               # G 计算函数
├── run_coupling_experiment.py  # 主实验脚本
└── visualize_coupling.py    # 可视化
```

### 9.2 关键代码片段

**Theory Admission**（`theory_simulator.py`）：

```python
# 计算 available tokens
available_tokens = completion_tokens - increment_tokens

# Admission（可负）
total_admission = available_tokens / (l0 + 1)
admission_A = total_admission * p
admission_B = total_admission * q
```

**Simulation Eviction**（`multi_type_simulator.py`）：

```python
# 按 decode 次数从高到低处理（高 decode = 高优先级 = 优先保留）
for decode_num in sorted(self.decode_to_requests.keys(), reverse=True):
    # 该层请求优先填入 batch
    if token_used + needed_tokens <= self.B:
        # 完全容纳
        X_prime[length][type_idx] = current_requests
    else:
        # 部分容纳，剩余部分 evict
        evicted = current_requests - can_admit
        batch_evictions[type_idx].append((length, evicted))
```

**G 计算**（`metrics.py`）：

```python
def compute_G_merge(state, l0, l_A, l_B, lambda_A, lambda_B, state_format):
    """计算合并状态的 G = max - min，带补偿"""
    merged_values = []
    for stage in range(max(l_A, l_B)):
        if stage < min(l_A, l_B):
            merged = state[stage][0] + state[stage][1]  # A + B
        else:
            # 只有一种类型，补偿
            merged = state[stage][1] * (lambda_A + lambda_B) / lambda_B
        merged_values.append(merged)
    return max(merged_values) - min(merged_values)
```

### 9.3 符号对照表

| 数学符号 | 代码变量 |
|:---------|:---------|
| $M_s^T$ | `theory_state[stage]` |
| $M_s^S$ | `sim_state[length]`，`length = l0 + stage` |
| $G^T, G^S$ | `theory_G_merge`, `sim_G_merge` |
| $a^T, a^S$ | `theory_admission`, `sim_admission` |
| $l_0, l_A, l_B$ | `l0`, `l_A`, `l_B` |
| $p, q$ | `p = lambda_A / (lambda_A + lambda_B)`, `q = 1 - p` |

---

## 参考文献

1. **理论论文**：`tex_docs/multiple_discrete.tex`（多类型离散时间分析）
2. **单类型分析**：`tex_docs/single_discrete.tex`（G 函数的原始定义）
3. **实验框架**：`coupling_experiments/documents/实现细节.md`

---

**文档版本**：v2.0
**创建日期**：2026-01-05
**最后更新**：2026-01-05
**状态**：主猜想待证明
