# 能量方法证明 V-Dominance：$V^T \geq V^S$

> **作者**: Claude (理论分析) + 实验验证
> **日期**: 2026-01-05
> **状态**: 理论框架完成，严格证明待完善

---

## 目录

1. [问题背景与目标](#1-问题背景与目标)
2. [符号定义 (Notation)](#2-符号定义-notation)
3. [核心定理陈述](#3-核心定理陈述)
4. [转移矩阵与 Lyapunov 矩阵](#4-转移矩阵与-lyapunov-矩阵)
5. [能量差分解公式](#5-能量差分解公式)
6. [证明框架](#6-证明框架)
7. [关键引理与分析](#7-关键引理与分析)
8. [实验验证](#8-实验验证)
9. [结论与未解决问题](#9-结论与未解决问题)

---

## 1. 问题背景与目标

### 1.1 研究背景

我们研究 LLM serving 系统中的连续批处理动态。系统有两种类型的请求（Type A 和 Type B），它们共享有限的内存资源（budget $B$）。

- **Theory 系统**：允许 "负 admission"（理论上的无约束系统）
- **Simulation 系统**：实际系统，admission 非负，超出容量时需要 eviction

### 1.2 核心问题

**G-Dominance 猜想**：$G^T(t) \geq G^S(t)$ 对所有 $t \geq 0$ 成立。

其中 $G = \max(M) - \min(M)$ 是状态向量的 "spread"（展开度）。

### 1.3 本文目标

通过 **能量方法 (Lyapunov Energy Method)** 证明一个更强的结论：

$$V^T(t) \geq V^S(t) \quad \forall t \geq 0$$

其中 $V$ 是 Lyapunov 能量函数。这个结论蕴含 G-Dominance（在适当的范数等价性下）。

---

## 2. 符号定义 (Notation)

### 2.1 系统参数

| 符号 | 含义 | 备注 |
|------|------|------|
| $l_0$ | Prefill 长度 | 初始 token 数 |
| $l_A$ | Type A 的 decode 长度 | $l_A < l_B$ |
| $l_B$ | Type B 的 decode 长度 | 较长类型 |
| $B$ | 总内存 budget | Token 总数上限 |
| $\lambda_A, \lambda_B$ | 到达率 | 通常设为 1.0 |
| $p = \frac{\lambda_A}{\lambda_A + \lambda_B}$ | Type A 的比例 | |
| $q = 1 - p$ | Type B 的比例 | |

### 2.2 状态向量

**Merged Compensated Vector** $M \in \mathbb{R}^{l_B}$：

$$M_s = \begin{cases}
A_s + B_s & \text{if } s < l_A \text{ (两种类型都存在)} \\
B_s \cdot \frac{\lambda_A + \lambda_B}{\lambda_B} & \text{if } s \geq l_A \text{ (只有 Type B)}
\end{cases}$$

其中 $A_s, B_s$ 分别是 stage $s$ 处 Type A 和 Type B 的请求数。

### 2.3 权重向量

**Token Balance 权重** $W \in \mathbb{R}^{l_B}$：

$$W_s = \begin{cases}
l_0 + s + 1 & \text{if } s < l_A \\
(l_0 + s + 1) \cdot q & \text{if } s \geq l_A
\end{cases}$$

**Token Balance 约束**：$\sum_{s=0}^{l_B-1} W_s M_s = B$

### 2.4 平衡点

**平衡状态** $M^* = [N^*, N^*, \ldots, N^*]$，其中：

$$N^* = \frac{B}{\sum_{s=0}^{l_B-1} W_s}$$

### 2.5 偏差向量

| 符号 | 定义 | 含义 |
|------|------|------|
| $x^T = M^T - M^*$ | Theory 偏差 | |
| $x^S = M^S - M^*$ | Simulation 偏差 | |
| $\Delta = M^T - M^S = x^T - x^S$ | 状态差 | |

### 2.6 约束空间

**Token Balance 超平面**：
$$H = \{v \in \mathbb{R}^{l_B} : W^T v = 0\}$$

重要性质：$x^T, x^S, \Delta \in H$（都满足 Token Balance 约束）。

---

## 3. 核心定理陈述

### 3.1 主定理

**定理 (V-Dominance)**：对于 coprime $(l_A, l_B)$ 系统，有

$$V^T(t) \geq V^S(t) \quad \forall t \geq 0$$

其中 Lyapunov 能量定义为：
$$V(M) = (M - M^*)^T P (M - M^*) = \|x\|_P^2$$

$P$ 是正定对称矩阵，满足离散 Lyapunov 方程。

### 3.2 关键引理

**引理 (内积非负)**：
$$\langle x^S, \Delta \rangle_P \geq 0 \quad \forall t \geq 0$$

其中 P-内积定义为 $\langle u, v \rangle_P = u^T P v$。

### 3.3 定理与引理的关系

由恒等式：
$$V^T - V^S = \|x^T\|_P^2 - \|x^S\|_P^2 = \|x^S + \Delta\|_P^2 - \|x^S\|_P^2 = 2\langle x^S, \Delta \rangle_P + \|\Delta\|_P^2$$

因此：
- $\|\Delta\|_P^2 \geq 0$ 总是成立
- 若 $\langle x^S, \Delta \rangle_P \geq 0$，则 $V^T - V^S \geq 0$

**引理 $\Rightarrow$ 定理**。

---

## 4. 转移矩阵与 Lyapunov 矩阵

本节详细推导转移矩阵 $A$ 和 Lyapunov 矩阵 $P$ 的构造方法，并给出具体的数值例子。

### 4.1 系统动态概述

考虑离散时间系统，每个 batch（时间步）的操作顺序：

1. **Shift**：所有 stage 的请求前进一步
   - Stage $s$ 的请求变成 stage $s+1$
   - Stage $l_B - 1$ 的请求完成并离开系统

2. **Admission**：新请求进入 stage 0
   - Theory 系统：admission 可以为负（理论无约束）
   - Simulation 系统：admission $\geq 0$，超出容量时 eviction

### 4.2 状态转移方程推导

**设定**：状态向量 $M(t) = [M_0(t), M_1(t), \ldots, M_{l_B-1}(t)]^T$

**Shift 操作**：
$$M_s(t+1) = M_{s-1}(t) \quad \text{for } s = 1, 2, \ldots, l_B - 1$$

**Admission 操作**：$M_0(t+1) = a(t)$，其中 $a(t)$ 是新 admission。

**Token Balance 约束**：系统在每个时刻满足
$$\sum_{s=0}^{l_B-1} W_s \cdot M_s(t) = B$$

**推导 $M_0(t+1)$**：

在 $t+1$ 时刻应用 Token Balance：
$$\sum_{s=0}^{l_B-1} W_s \cdot M_s(t+1) = B$$

代入 shift 关系：
$$W_0 \cdot M_0(t+1) + \sum_{s=1}^{l_B-1} W_s \cdot M_{s-1}(t) = B$$

变换求和指标（令 $s' = s - 1$）：
$$W_0 \cdot M_0(t+1) + \sum_{s'=0}^{l_B-2} W_{s'+1} \cdot M_{s'}(t) = B$$

解出 $M_0(t+1)$：
$$M_0(t+1) = \frac{1}{W_0} \left( B - \sum_{s=0}^{l_B-2} W_{s+1} \cdot M_s(t) \right)$$

### 4.3 偏差形式的动态方程

定义偏差 $\delta(t) = M(t) - M^*$，其中 $M^* = [N^*, N^*, \ldots, N^*]^T$ 是平衡点。

由于 $\sum_s W_s M^*_s = B$（平衡点也满足 Token Balance），代入得：

$$\delta_0(t+1) + N^* = \frac{1}{W_0} \left( \sum_s W_s N^* - \sum_{s=0}^{l_B-2} W_{s+1} \cdot (\delta_s(t) + N^*) \right)$$

简化：
$$\delta_0(t+1) = -\frac{1}{W_0} \sum_{s=0}^{l_B-2} W_{s+1} \cdot \delta_s(t)$$

**关键观察**：$\delta_{l_B-1}(t)$ **不出现**在方程中！

这是因为 stage $l_B - 1$ 的请求在当前 batch 完成离开系统，不参与下一步的 Token Balance。

### 4.4 转移矩阵 $A$ 的显式形式

综合以上推导，偏差的演化为 $\delta(t+1) = A \cdot \delta(t)$，其中：

$$A = \begin{pmatrix}
-\frac{W_1}{W_0} & -\frac{W_2}{W_0} & -\frac{W_3}{W_0} & \cdots & -\frac{W_{l_B-1}}{W_0} & 0 \\
1 & 0 & 0 & \cdots & 0 & 0 \\
0 & 1 & 0 & \cdots & 0 & 0 \\
0 & 0 & 1 & \cdots & 0 & 0 \\
\vdots & & & \ddots & & \vdots \\
0 & 0 & 0 & \cdots & 1 & 0
\end{pmatrix}_{l_B \times l_B}$$

**矩阵结构说明**：
- **第一行**：admission 方程的系数 $A_{0,s} = -W_{s+1}/W_0$ for $s = 0, \ldots, l_B-2$，$A_{0, l_B-1} = 0$
- **下对角线**：shift 操作 $A_{s, s-1} = 1$ for $s = 1, \ldots, l_B-1$
- **最后一列全为 0**：$\delta_{l_B-1}$ 完成离开系统

### 4.5 具体数值例子

**参数**：$l_0 = 10$, $l_A = 2$, $l_B = 3$, $\lambda_A = \lambda_B = 1.0$

**计算权重向量**：
- $p = 0.5$, $q = 0.5$
- $W_0 = l_0 + 1 = 11$
- $W_1 = l_0 + 2 = 12$（$s=1 < l_A=2$，两种类型都存在）
- $W_2 = (l_0 + 3) \cdot q = 13 \times 0.5 = 6.5$（$s=2 \geq l_A$，只有 Type B）

$$W = [11, 12, 6.5]$$

**计算转移矩阵**：
$$A = \begin{pmatrix}
-\frac{12}{11} & -\frac{6.5}{11} & 0 \\
1 & 0 & 0 \\
0 & 1 & 0
\end{pmatrix} = \begin{pmatrix}
-1.0909 & -0.5909 & 0 \\
1 & 0 & 0 \\
0 & 1 & 0
\end{pmatrix}$$

**计算特征值**：
$$\det(\lambda I - A) = 0$$
$$\lambda^3 + 1.0909 \lambda^2 + 0.5909 \lambda = 0$$
$$\lambda (\lambda^2 + 1.0909 \lambda + 0.5909) = 0$$

特征值：$\lambda_1 = 0$，$\lambda_{2,3} = \frac{-1.0909 \pm \sqrt{1.19 - 2.36}}{2}$（共轭复数）

数值计算：$|\lambda_{2,3}| \approx 0.769 < 1$（系统稳定）

### 4.6 Lyapunov 矩阵 $P$ 的计算

**离散 Lyapunov 方程**：
$$A^T P A - P = -Q$$

其中 $Q$ 通常取单位矩阵 $I$。

**求解方法**：使用 `scipy.linalg.solve_discrete_lyapunov`

```python
from scipy import linalg

# scipy 求解的方程形式：A X A^T - X + Q = 0
# 我们需要：A^T P A - P + Q = 0
# 所以传入 A^T 作为参数
P = linalg.solve_discrete_lyapunov(A.T, Q)
```

**存在性条件**：当且仅当 $A$ 的所有特征值 $|\lambda| < 1$ 时，$P$ 存在且唯一。

对于 coprime $(l_A, l_B)$，系统稳定，$P$ 存在。

### 4.7 $l_B = 3$ 的 $P$ 矩阵数值例子

继续上面的例子（$l_0 = 10$, $l_A = 2$, $l_B = 3$），计算得：

$$P = \begin{pmatrix}
8.70 & 3.53 & 0 \\
3.53 & 5.04 & 0 \\
0 & 0 & 1
\end{pmatrix}$$

**验证正定性**：特征值为 $\{1.0, 4.8, 8.9\}$，全为正，$P$ 正定。

**结构观察**：
- $P_{2,0} = P_{2,1} = P_{0,2} = P_{1,2} = 0$（与最后一列/行相关的元素为 0）
- 这是因为 $A$ 的最后一列为 0

### 4.8 $l_0 = 50$, $l_A = 4$, $l_B = 7$ 的完整例子

**权重向量**：
$$W = [51, 52, 53, 54, 27.5, 28, 28.5]$$

（前 4 个是完整权重 $l_0 + s + 1$，后 3 个乘以 $q = 0.5$）

**转移矩阵** $A$（$7 \times 7$）：

$$A = \begin{pmatrix}
-1.020 & -1.039 & -1.059 & -0.539 & -0.549 & -0.559 & 0 \\
1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0
\end{pmatrix}$$

**特征值**：$\{0, 0.988e^{\pm i\theta_1}, 0.905e^{\pm i\theta_2}, 0.836e^{\pm i\theta_3}\}$

最大 $|\lambda| = 0.988 < 1$（稳定）

**Lyapunov 矩阵** $P$（$7 \times 7$，数值结果）：

$$P \approx \begin{pmatrix}
59.0 & 43.8 & 7.6 & 21.5 & 29.7 & 9.2 & 0 \\
43.8 & 85.9 & 34.5 & 12.1 & 42.9 & 30.0 & 0 \\
7.6 & 34.5 & 37.0 & -0.1 & 14.0 & 22.9 & 0 \\
21.5 & 12.1 & -0.1 & 15.3 & 9.2 & 1.2 & 0 \\
29.7 & 42.9 & 14.0 & 9.2 & 29.2 & 13.0 & 0 \\
9.2 & 30.0 & 22.9 & 1.2 & 13.0 & 20.4 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1
\end{pmatrix}$$

**$P$ 的特征值**：$\{1.0, 3.7, 4.8, 6.0, 14.8, 51.0, 166.6\}$（全为正，正定）

### 4.9 $P$ 矩阵的关键性质

1. **正定对称**：$P = P^T$，所有特征值 $> 0$

2. **最后一行/列结构**：$P_{l_B-1, j} = P_{j, l_B-1} = 0$ for $j < l_B - 1$，$P_{l_B-1, l_B-1} = 1$
   - 这是因为 $A$ 的最后一列为 0，导致 $\delta_{l_B-1}$ "解耦"

3. **早期 stage 权重较大**：$P_{0,0}, P_{1,1}$ 等对角元素较大
   - 这编码了系统在早期 stage 的"敏感性"

4. **非对角元素**：反映不同 stage 之间的耦合
   - $P_{0,1}$ 较大说明 stage 0 和 stage 1 高度相关

### 4.10 与特征方程的关系

**特征方程**（从理论推导）：
$$F(\lambda) = W_0 \lambda^{l_B-1} + W_1 \lambda^{l_B-2} + \ldots + W_{l_B-1} = 0$$

**转移矩阵特征值**：$A$ 的非零特征值应与 $F(\lambda) = 0$ 的根一致。

**验证**（$l_0 = 50$, $l_A = 4$, $l_B = 7$）：
- 特征方程根：$\max |root| = 0.9885$
- 转移矩阵非零特征值：$\max |\lambda| = 0.9885$

**完全一致！** 这验证了我们的矩阵构造是正确的。

### 4.11 转移矩阵的修正历史

**原始实现的错误**：

原代码中，第一行最后一个元素被设为：
$$A_{0, l_B-1} = \frac{(l_0 + l_B) \cdot q}{W_0} > 0$$

这是将 "completion 释放 tokens" 的效果错误地加入了 admission 方程。

**错误的物理解释**：原实现认为 completion 释放 tokens 会增加 admission。

**正确的理解**：
- Completion 发生在当前 batch 内，请求离开系统
- Token Balance 是在 shift 和 admission 后计算的
- $\delta_{l_B-1}(t)$ 在下一步已经不存在，不应出现在方程中

**修正**：$A_{0, l_B-1} = 0$

**修正前后对比**（$l_0 = 50$, $l_A = 4$, $l_B = 7$）：

| 指标 | 修正前 | 修正后 |
|------|--------|--------|
| $A$ 的 $\max |\lambda|$ | 1.10 | 0.988 |
| 与特征方程一致 | ❌ | ✅ |
| 系统稳定性 | 不稳定 | 稳定 |
| $P$ 可计算 | ❌ | ✅ |

---

## 5. 能量差分解公式

### 5.1 能量定义

$$V^T = \|M^T - M^*\|_P^2 = (x^T)^T P x^T$$
$$V^S = \|M^S - M^*\|_P^2 = (x^S)^T P x^S$$

### 5.2 分解公式推导

由 $x^T = x^S + \Delta$：
$$V^T = \|x^S + \Delta\|_P^2 = (x^S + \Delta)^T P (x^S + \Delta)$$
$$= (x^S)^T P x^S + 2 (x^S)^T P \Delta + \Delta^T P \Delta$$
$$= V^S + 2\langle x^S, \Delta \rangle_P + \|\Delta\|_P^2$$

因此：
$$\boxed{V^T - V^S = 2\langle x^S, \Delta \rangle_P + \|\Delta\|_P^2}$$

### 5.3 分解公式的意义

- **第一项** $2\langle x^S, \Delta \rangle_P$：Simulation 偏差与状态差的 P-内积
- **第二项** $\|\Delta\|_P^2 \geq 0$：状态差的 P-范数平方（总是非负）

**关键**：若 $\langle x^S, \Delta \rangle_P \geq 0$，则 $V^T \geq V^S$。

---

## 6. 证明框架

### 6.1 证明策略

令 $f(t) = \langle x^S(t), \Delta(t) \rangle_P$。

**目标**：证明 $f(t) \geq 0$ 对所有 $t \geq 0$ 成立。

### 6.2 步骤 1：初始条件

$t = 0$ 时，$M^T(0) = M^S(0)$（两系统初始状态相同）。

因此 $\Delta(0) = 0$，$f(0) = \langle x^S(0), 0 \rangle_P = 0 \geq 0$ ✓

### 6.3 步骤 2：无 eviction 时的演化

当 Theory 的 admission $\tilde{a}(t) \geq 0$ 时，两系统相同：
- $\Delta(t)$ 保持不变（或 $= 0$）
- 如果 $\Delta(t) \neq 0$，则 $\Delta(t+1) = A \cdot \Delta(t)$

**无 eviction 时的 f 演化**：
$$f(t+1) = \langle A x^S(t), A \Delta(t) \rangle_P = (x^S)^T A^T P A \Delta$$

由 Lyapunov 方程 $A^T P A = P - Q$：
$$f(t+1) = (x^S)^T (P - Q) \Delta = f(t) - \langle x^S, \Delta \rangle_Q$$

若 $Q = I$：
$$f(t+1) = f(t) - \langle x^S, \Delta \rangle_{L^2}$$

### 6.4 步骤 3：eviction 瞬间（关键！）

当 $\tilde{a}(t) < 0$ 时，eviction 发生。

**eviction 前后的状态变化**：
- Theory：$M^T_0 = \tilde{a} < 0$（允许负 admission）
- Simulation：$M^S_0 = 0$（截断为 0），然后 evict 超出的请求

**$\Delta$ 的结构**：
$$\Delta = [\tilde{a}, e_1, e_2, \ldots, e_{l_B-1}]$$

其中：
- $\tilde{a} < 0$（Theory 的负 admission）
- $e_s \geq 0$（stage $s$ 被 evict 的量）
- Token Balance：$\sum_s W_s \Delta_s = 0$，即 $W_0 \tilde{a} + \sum_{s \geq 1} W_s e_s = 0$

因此：$\sum_{s \geq 1} W_s e_s = -W_0 \tilde{a} = |W_0 \tilde{a}| > 0$

### 6.5 步骤 4：$\Delta$ 的分解

定义：
- $e_0 = [1, 0, 0, \ldots, 0]$（第一个基向量）
- $e = [0, e_1, e_2, \ldots, e_{l_B-1}]$（eviction 向量）

则：
$$\Delta = \tilde{a} \cdot e_0 + e$$

内积分解：
$$\langle x^S, \Delta \rangle_P = \tilde{a} \cdot \langle x^S, e_0 \rangle_P + \langle x^S, e \rangle_P$$

### 6.6 步骤 5：需要证明的条件

由于 $\tilde{a} < 0$，要使 $\langle x^S, \Delta \rangle_P \geq 0$，需要：
$$\langle x^S, e \rangle_P \geq -\tilde{a} \cdot \langle x^S, e_0 \rangle_P = |\tilde{a}| \cdot \langle x^S, e_0 \rangle_P$$

即 **eviction 向量与 $x^S$ 的 P-内积** 需要足够大。

---

## 7. 关键引理与分析

### 7.1 eviction 发生的条件

**引理 7.1**：eviction 发生当且仅当系统 "过满"。

设 shift 后（admission 前）的状态为 $M'^S$：
- $M'^S_0 = 0$（还没 admission）
- $M'^S_s = M^S_{s-1}(t-1)$ for $s \geq 1$

eviction 发生 $\Leftrightarrow$ $\tilde{a} < 0$ $\Leftrightarrow$ $\sum_{s \geq 1} W_s M'^S_s > B$

即 shift 后的 tokens 超过 budget。

### 7.2 $x^S$ 的结构（eviction 发生时）

**引理 7.2**：eviction 发生时，$x^S$ 在某些早期 stage 有 "堆积"。

设 $x'^S = M'^S - M^*$（shift 后、eviction 前的偏差）。

由于 $M'^S_0 = 0$：$x'^S_0 = -N^* < 0$

eviction 发生的条件 $\sum_{s \geq 1} W_s M'^S_s > B$ 等价于：
$$\sum_{s \geq 1} W_s x'^S_s > W_0 N^*$$

即 stage $s \geq 1$ 处的加权偏差为正，存在 "过满"。

### 7.3 eviction 向量 $e$ 的结构

**引理 7.3**：eviction 向量 $e = [0, e_1, e_2, \ldots]$ 满足：
1. $e_s \geq 0$ for all $s \geq 1$
2. $\sum_{s \geq 1} W_s e_s = -W_0 \tilde{a} > 0$（Token Balance）
3. eviction 优先从低 stage 开始（保留高 stage 的请求）

**几何意义**：$e$ 的 "支撑"（非零位置）通常是较低的 stage。

### 7.4 几何协调性

**核心观察**（非正式）：

1. **$x^S$ 的高峰位置**：由于 arrival 在 stage 0，堆积通常发生在早期 stage
2. **$e$ 的支撑位置**：eviction 从低 stage 开始，所以 $e_s > 0$ 在早期 stage
3. **协调性**：$x^S$ 的高峰与 $e$ 的支撑位置对齐

因此 $\langle x^S, e \rangle_P$ 倾向于较大的正值。

### 7.5 $P$ 矩阵的结构

**引理 7.4**：$P$ 矩阵有以下性质：
1. 正定对称
2. 来自 Lyapunov 方程 $A^T P A - P = -Q$
3. 编码了系统的 "内在几何"

数值观察：$P$ 的结构使得早期 stage 的权重较大，放大了几何协调性。

### 7.6 证明的缺口

完整的严格证明需要：

1. **证明** $\langle x^S, e_0 \rangle_P$ **的符号**
   - 数值验证：通常为正
   - 理论证明：需要分析 $P$ 的结构

2. **证明条件** $\langle x^S, e \rangle_P \geq |\tilde{a}| \cdot \langle x^S, e_0 \rangle_P$
   - 需要利用 eviction 策略和系统状态的耦合
   - 可能需要归纳论证

3. **证明 $f(t)$ 在无 eviction 期间保持非负**
   - 需要分析 $\langle x^S, \Delta \rangle_Q$ 的符号
   - 与 L2 内积和 P 内积的关系

---

## 8. 实验验证

### 8.1 实验设置

| 参数 | 值 |
|------|-----|
| $l_0$ | 200 |
| $l_A$ | 4 |
| $l_B$ | 7 |
| $B$ | 1000 |
| $\lambda_A, \lambda_B$ | 1.0 |
| 步数 | 200 |

### 8.2 实验 1：eviction 是否 P-非扩张

**命题**：eviction 减少到平衡点的 P-范数距离。

即：$\|M^S_{after} - M^*\|_P \leq \|M^S_{before} - M^*\|_P$

**结果**：

| 指标 | 值 |
|------|-----|
| eviction 减少距离的次数 | 5/9 (55.6%) |
| 平均距离变化 | -0.79 |

**结论**：**失败**。eviction 不是严格的 P-非扩张映射。有时 eviction 反而增加距离。

### 8.3 实验 2：$\langle x^S, \Delta \rangle_P \geq 0$

**命题**：P-内积总是非负。

**结果**：

| 指标 | 值 |
|------|-----|
| 非负次数 | **200/200 (100%)** |
| 平均值 | 0.9424 |
| 最小值 | $\approx 0$ |

**结论**：**成功**。内积 100% 非负。

### 8.4 推论验证：$V^T \geq V^S$

由分解公式，$\langle x^S, \Delta \rangle_P \geq 0$ 直接蕴含 $V^T \geq V^S$。

**结果**：

| 指标 | 值 |
|------|-----|
| $V^T \geq V^S$ 次数 | **200/200 (100%)** |
| $V^T / V^S$ 平均比值 | 1.102 |

### 8.5 分解公式验证

**验证**：$V^T - V^S = 2\langle x^S, \Delta \rangle_P + \|\Delta\|_P^2$

| 指标 | 值 |
|------|-----|
| 分解误差 | $\sim 10^{-14}$（机器精度） |

公式精确成立。

### 8.6 可视化结果

实验生成了 `p_norm_experiments.png`，包含四个子图：

1. **左上**：eviction 前后的 P-范数距离比较
2. **右上**：每次 eviction 的距离变化（绿=减少，红=增加）
3. **左下**：$\langle x^S, \Delta \rangle_P$ 随时间的变化（总是非负）
4. **右下**：$V^T - V^S$ 的分解验证

---

## 9. 结论与未解决问题

### 9.1 主要结论

1. **转移矩阵修正**：原实现有误，修正后特征值与特征方程根一致

2. **实验验证**：
   - 路径 A（eviction P-非扩张）：**失败**（55.6%）
   - 路径 B（内积非负）：**成功**（100%）

3. **定理成立**：$V^T \geq V^S$ 在所有测试中成立（100%）

4. **证明框架**：
   - 分解公式：$V^T - V^S = 2\langle x^S, \Delta \rangle_P + \|\Delta\|_P^2$
   - 关键条件：$\langle x^S, e \rangle_P \geq |\tilde{a}| \cdot \langle x^S, e_0 \rangle_P$

### 9.2 物理解释

**为什么 $V^T \geq V^S$？**

1. **eviction 不是简单的 "能量耗散"**：单次 eviction 可能增加 P-范数距离

2. **真正的机制**：$x^S$ 与 $\Delta$ 的几何协调性
   - eviction 发生时系统 "过满"
   - 过满位置（$x^S$ 大）与 eviction 位置（$\Delta$ 非零）对齐
   - $P$ 矩阵放大这种协调性

3. **结果**：虽然 eviction 可能暂时增加局部距离，但整体上 Theory 的能量总是大于等于 Simulation

### 9.3 未解决问题

1. **严格证明** $\langle x^S, \Delta \rangle_P \geq 0$
   - 需要分析 $P$ 矩阵的显式结构
   - 需要利用 eviction 策略的几何性质

2. **$P$ 矩阵的解析形式**
   - 对于 companion-like 矩阵 $A$，$P$ 可能有特殊结构
   - 可能存在闭式解或近似解

3. **归纳证明**
   - 证明每次 eviction 后 $f(t) \geq 0$ 保持
   - 证明 $f(t)$ 在无 eviction 期间不变负

4. **从 $V^T \geq V^S$ 推导 $G^T \geq G^S$**
   - 需要建立 P-范数与 L∞ 范数（spread）的关系
   - 可能需要额外的范数等价性分析

### 9.4 研究方向建议

1. **$P$ 矩阵结构分析**
   - 数值计算不同参数下的 $P$
   - 寻找规律和显式公式

2. **简化模型**
   - 分析 $l_B = 2$ 或 $l_B = 3$ 的情况
   - 可能得到完整的解析证明

3. **概率论方法**
   - 将 eviction 视为随机过程
   - 利用 coupling 或 martingale 技术

4. **控制论视角**
   - 将 eviction 视为反馈控制
   - 利用 Lyapunov 稳定性理论的推广

---

## 附录 A：代码实现

### A.1 转移矩阵计算

```python
def compute_transition_matrix(l0, l_A, l_B, lambda_A, lambda_B):
    """
    计算正确的转移矩阵 A。

    关键：最后一列为 0（完成的请求离开系统）。
    """
    n = l_B
    p = lambda_A / (lambda_A + lambda_B)
    q = 1 - p

    # 权重向量
    W = np.zeros(n)
    for s in range(n):
        if s < l_A:
            W[s] = l0 + s + 1
        else:
            W[s] = (l0 + s + 1) * q

    A = np.zeros((n, n))

    # 第一行：admission 方程
    for s in range(n - 1):
        A[0, s] = -W[s + 1] / W[0]
    A[0, n - 1] = 0.0  # 关键：最后一列为 0

    # 下对角线：shift 操作
    for s in range(1, n):
        A[s, s - 1] = 1.0

    return A
```

### A.2 P-内积计算

```python
def compute_P_inner_product(x, y, P):
    """计算 <x, y>_P = x^T P y"""
    return float(x @ P @ y)

def compute_inner_product_analysis(M_theory, M_sim, P, M_star):
    """实验 2：分析 <x^S, Δ>_P 的符号"""
    x_S = M_sim - M_star
    Delta = M_theory - M_sim

    inner_prod_P = compute_P_inner_product(x_S, Delta, P)
    Delta_P_norm_sq = compute_P_inner_product(Delta, Delta, P)

    return {
        'inner_prod_P': inner_prod_P,
        'Delta_P_norm_sq': Delta_P_norm_sq,
        'inner_prod_geq_0': inner_prod_P >= -1e-9,
        'V_diff_decomposition': 2 * inner_prod_P + Delta_P_norm_sq,
    }
```

---

## 附录 B：实验数据摘要

### B.1 P-Norm 实验结果

| Batch | has_eviction | dist_change | inner_prod_P | inner_prod_geq_0 |
|-------|--------------|-------------|--------------|------------------|
| 1 | True | -5.12 | 1.95 | True |
| 2 | True | -2.49 | 4.22 | True |
| 3 | True | -0.85 | 5.08 | True |
| 5 | True | -3.10 | 5.82 | True |
| 6 | True | 0.31 | 6.76 | True |
| ... | ... | ... | ... | ... |

### B.2 分解公式验证

所有 200 个时间步的分解误差均在 $10^{-14}$ 量级，验证了公式的正确性。

---

## 小结
1. Markdown Document (能量方法证明V_dominance.md)

Section 4 has been significantly expanded with:
- Detailed state transition equation derivation
- Explicit $A$ matrix formula with structure explanation
- Numerical examples for $l_B=3$ and $l_B=7$ cases
- $P$ matrix computation method using scipy.linalg.solve_discrete_lyapunov
- Lyapunov equation explanation
- Error history and fix explanation
- Relationship to characteristic equation

2. Visualization Functions (All 8 Implemented)

| Function                       | Purpose                             | Output                      |
|--------------------------------|-------------------------------------|-----------------------------|
| plot_delta_extrema()           | Key: verifies $\Delta_{m^S} \geq 0$ | delta_extrema.png           |
| plot_delta_heatmap()           | Delta vector time-space heatmap     | delta_heatmap.png           |
| plot_lyapunov_energy()         | $V^T$ vs $V^S$ energy evolution     | lyapunov_energy.png         |
| plot_case_distribution()       | A/B/C/D case distribution           | case_distribution.png       |
| plot_G_vs_V_relation()         | $G^2$ vs $V$ scatter plot           | G_vs_V_relation.png         |
| plot_eviction_delta_relation() | Eviction-Delta correlation          | eviction_delta_relation.png |
| plot_analysis_dashboard()      | 2×3 comprehensive dashboard         | analysis_dashboard.png      |
| plot_p_norm_experiments()      | Exp 1 & 2 results                   | p_norm_experiments.png      |

3. Integration in run_coupling_experiment.py

All visualization functions are imported and called in the main() function.

Key Results from Previous Experiments

- Experiment 1 (Eviction P-nonexpansive): 55.6% - FAILED
- Experiment 2 ($\langle x^S, \Delta \rangle_P \geq 0$): 100% - SUCCESS
- $V^T \geq V^S$: 100% verified

## 参考文献

1. 理论论文：`/Users/ruicheng/.../LLM_serving/`
2. 实验代码：`coupling_experiments/run_coupling_experiment.py`
3. 可视化：`coupling_experiments/visualize_coupling.py`
4. 相关理论文档：
   - `theory_markdown/Eviction导致额外能量耗散.md`
   - `theory_markdown/G_dominance_analysis.md`
