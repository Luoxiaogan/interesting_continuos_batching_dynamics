---
深度理论分析：为什么 Eviction 导致额外能量耗散？

1. 重新理解问题结构

你的观察非常关键：能量不是单调递减的，因为系统有外部输入：

$$M(t+1) = A \cdot M(t) + b(t)$$

其中 $b(t)$ 是 arrival（外部扰动），会注入能量。

核心问题重述：
- 两系统有相同的初始状态 $M^T(0) = M^S(0)$
- 两系统有相同的外部输入 $b(t)$ (overloaded系统, 一定会填满)
- Simulation 多了 eviction 操作
- 为什么 $V^T(t) \geq V^S(t)$？

---
1. 差系统分析

定义差向量 $\Delta(t) = M^T(t) - M^S(t)$

无 eviction 时（$\tilde{a} \geq 0$）：
$$\Delta(t+1) = A \cdot \Delta(t)$$

有 eviction 时（$\tilde{a} < 0$）：非线性耦合

Token Balance 约束：
$$\sum_{s=0}^{l_B-1} W_s \Delta_s(t) = 0 \quad \forall t$$

这是因为两系统都满足 $\sum W_s M_s = B$。

---
3. 能量差的精确分解

设 $x^T = M^T - M^$，$x^S = M^S - M^$，则 $x^T - x^S = \Delta$

$$V^T - V^S = x^T_P^2 - x^S_P^2$$

展开：
$$V^T - V^S = x^S + \Delta_P^2 - x^S_P^2 = 2\langle x^S, \Delta \rangle_P + \Delta_P^2$$

其中 $\langle u, v \rangle_P = u^T P v$。

关键：
- $\Delta_P^2 \geq 0$ 总是成立
- 如果 $\langle x^S, \Delta \rangle_P \geq 0$，则 $V^T \geq V^S$ ✓

但：$\langle x^S, \Delta \rangle_P$ 的符号不确定...

---
4. 核心困难：Eviction 不是正交投影

根据 https://tisp.indigits.com/cvxopt/projection 和 https://www.researchgate.net/publication/266942561_Firmly_nonexpansive_mappings：

标准结果：正交投影 $P_C$ 到凸集 $C$ 是 firmly nonexpansive：
$$\langle P_C(x) - P_C(y), x - y \rangle \geq P_C(x) - P_C(y)^2$$

推论：对于凸集内的点 $y^* \in C$：
$$P_C(x) - y^_2 \leq x - y^_2$$

这会直接给出 $V^S \leq V^T$（在 $L_2$ 范数下）。

但问题是：Simulation 的 eviction 操作是一种 https://en.wikipedia.org/wiki/Projection_(linear_algebra)（斜投影）：
- 优先保留高 stage
- 从低 stage 开始 evict

这种投影不一定在欧几里得范数下非扩张！

---
5. 新视角：Coupling 论证

根据 https://people.engr.tamu.edu/andreas-klappenecker/csce658-s18/coupling.pdf 和 https://people.math.wisc.edu/~roch/mdp/roch-mdp-chap4.pdf：

思路：构造 Theory 和 Simulation 的"耦合"，使得某种序关系保持。

Strassen 定理：如果两个马尔可夫核 $P \preceq Q$（$Q$ 随机支配 $P$），则存在耦合使得 $X_t \leq Y_t$ a.s.

应用到我们的问题：
- 能否证明 Theory 的状态在某种序上"支配" Simulation？
- 凸序（Convex Order）：如果 $M^T \geq_{cx} M^S$，则对所有凸函数 $\phi$，$E[\phi(M^T)] \geq E[\phi(M^S)]$

---
6. 可能的证明路径

路径 A：证明 Eviction 在 $P$-范数下是非扩张的

猜想：存在某种加权范数 $\cdot_Q$（可能与 $P$ 相关），使得：
$$\mathcal{E}(M) - M^_Q \leq M - M^_Q$$

其中 $\mathcal{E}$ 是 eviction 操作。

直觉：
- $P$ 矩阵来自 Lyapunov 方程 $A^T P A - P = -Q$
- 它编码了系统的"内在几何"
- Eviction 从低 stage 移除（与信息流同向），可能与这个几何"协调"

实验验证：在每次 eviction 发生时，计算 $M^S_{after} - M^P$ vs $M^S{before} - M^_P$

路径 B：直接分析 $\Delta$ 的结构

观察：Token Balance 约束 $\sum W_s \Delta_s = 0$ 意味着：
- 正 $\Delta$ 和负 $\Delta$ 在加权意义下平衡
- Eviction 后，$\Delta_0 = a^T - 0 < 0$（因为 $a^T < 0$）
- 其他位置的 $\Delta$ 必须补偿

问题：$P$-加权下的内积 $\langle x^S, \Delta \rangle_P$ 是否总是非负？

路径 C：Lyapunov-like 函数直接构造

根据 https://link.springer.com/article/10.1007/BF00276493 和 https://arxiv.org/html/2510.08259：

思路：不用标准的 $V = x^T P x$，而是构造一个考虑约束的能量函数：

$$\tilde{V}(M) = \min_{y \in \Omega} M - y_P^2$$

其中 $\Omega = M \geq 0, \sum W_s M_s = B$。

性质：
- 对于 $M \in \Omega$，$\tilde{V}(M) = 0$
- 对于 $M \notin \Omega$，$\tilde{V}(M) > 0$
- Eviction 将 $M$ 映射回 $\Omega$，所以 $\tilde{V}$ 必然减少

---
7. 最关键的洞察

让我提出一个核心论证框架：

命题：设 $\tilde{M}$ 是 eviction 前的状态（可能 $\tilde{M}_0 < 0$ 或违反容量约束），$M^S$ 是 eviction 后的状态。则：

$$M^S - M^_P \leq \tilde{M} - M^_P$$

为什么可能成立？

1. $M^*$ 在可行域内：$M^* \geq 0$ 且 $\sum W_s M^*_s = B$
2. Eviction 将状态"拉回"可行域：
- 如果 $\tilde{M}_0 < 0$，截断为 0 使其更接近 $M^*_0 > 0$
- 如果总 tokens 超过 $B$，移除使其满足约束
3. 关键问题：这种"拉回"是否在 $P$-范数下减少距离？

这取决于 $P$ 的结构和 eviction 策略的协调性！

---
8. 具体实验建议

让我用代码验证一些关键猜想：

# 实验 1: 验证 eviction 在 P-范数下的效果
for t in range(steps):
    if eviction_happens(t):
        M_before = state_before_eviction  # M̃
        M_after = state_after_eviction    # M^S

        dist_before = (M_before - M_star).T @ P @ (M_before - M_star)
        dist_after = (M_after - M_star).T @ P @ (M_after - M_star)

        print(f"t={t}: dist_before={dist_before:.4f}, dist_after={dist_after:.4f}")
        # 如果总是 dist_after <= dist_before，则 eviction 是 P-非扩张的

# 实验 2: 验证 <x^S, Δ>_P 的符号
for t in range(steps):
    x_S = M_S[t] - M_star
    Delta = M_T[t] - M_S[t]
    inner_prod = x_S.T @ P @ Delta
    print(f"t={t}: <x^S, Δ>_P = {inner_prod:.4f}")
    # 如果总是 >= 0，则 V^T >= V^S 成立

---
9. 总结：证明的本质困难与突破口

| 困难                  | 描述                         | 可能的突破                            |
|-----------------------|------------------------------|---------------------------------------|
| Eviction 不是正交投影 | 标准的非扩张性结果不直接适用 | 寻找 eviction 在 $P$-范数下的特殊性质 |
| $\Delta$ 的符号不确定 | 无法直接比较逐分量           | 利用 Token Balance 约束的结构         |
| 能量不单调            | 有外部输入 $b(t)$            | 分析 eviction 步骤的瞬时效果          |

最有希望的方向：
1. 证明 eviction 是 $P$-contractive（实验验证优先）
2. 利用 coupling 论证：构造两系统的耦合，使得能量序保持

---
要我用 codebase 实现这些实验吗？特别是验证 eviction 前后 $P$-范数的变化？

Sources:
- https://tisp.indigits.com/cvxopt/projection
- https://www.researchgate.net/publication/266942561_Firmly_nonexpansive_mappings
- https://en.wikipedia.org/wiki/Projection_(linear_algebra)
- https://people.engr.tamu.edu/andreas-klappenecker/csce658-s18/coupling.pdf
- https://link.springer.com/article/10.1007/BF00276493
- https://arxiv.org/html/2510.08259
- https://link.springer.com/article/10.1007/BF02192301