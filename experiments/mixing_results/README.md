# Multi-Replica Mixing Experiment - 可视化结果文档

> 实验日期: 2026-01-03
> 实验脚本: `experiments/scripts/run_mixing_experiment.py`
> 分析脚本: `experiments/scripts/analyze_stage_oscillation.py`

## 📋 实验配置

### Request Types设计

本实验使用**4种request types**，分为**2组**，每组包含**non-coprime的GCD条件**：

| Type | Prompt (l0) | Decode (l1) | Group | GCD条件 |
|------|-------------|-------------|-------|---------|
| Type 1 | 4 tokens | 8 tokens | Group 1 | gcd(8,16) = 8 |
| Type 2 | 4 tokens | 16 tokens | Group 1 | (non-coprime) |
| Type 3 | 3 tokens | 5 tokens | Group 2 | gcd(5,15) = 5 |
| Type 4 | 3 tokens | 15 tokens | Group 2 | (non-coprime) |

**设计理由**:
- **Non-coprime GCD**: 每组内的两个types的l1不互质，根据理论应该收敛到limit cycle
- **小型requests**: 最大request (3+15=18 tokens) 相对GPU容量很小，确保系统有足够空间展示mixing优势
- **Capacity ratio**: B/max_request_size = 500/18 ≈ 27.8 >> 25，满足mixing有效的容量条件

### 系统参数

**GPU和模拟参数**:
- **B (GPU容量)**: 500 tokens
  - 说明: 单个replica的最大KV cache容量
  - 可同时容纳: ~34-69个requests（取决于type）
- **b0 (Prompt处理成本)**: 0.1 time/token
- **b1 (Decode处理成本)**: 0.01 time/token
- **模拟步数**: 1000 batches
- **Warmup**: 前200 batches用于系统收敛到稳态

**Arrival Rates**:
- **所有types**: λ = 1.0 requests/time
- **总到达率**: 4.0 requests/time (4种types)
- **系统负载**: Underloaded (到达率 < 处理能力)
  - 理论最大throughput ≈ B/avg_request_size ≈ 500/11 ≈ 45 req/time
  - 实际到达率 = 4.0 req/time
  - 负载率 ≈ 9% (轻载)

### 对比场景

**1. Segregated Routing (隔离路由)**:
- **Group 1** (Type 1,2) → **Replica 0**
- **Group 2** (Type 3,4) → **Replica 1**
- **特点**: 每个replica只处理一组requests，内部形成limit cycle
- **预期**: 各replica被困在2个active stages的震荡中

**2. Mixed Routing (混合路由)**:
- **All Types** (Type 1,2,3,4) → **均匀分配到Replica 0和1**
- **分配策略**: Round-robin，每个到达的request轮流分配
- **特点**: 每个replica同时处理所有4种types
- **预期**: 打破limit cycle，扩展到更多active stages

### 实验目标

1. **验证Limit Cycle**: Segregated场景下各replica应收敛到2个active stages
2. **验证Mixing效果**: Mixed场景应打破limit cycle，增加active stages数量
3. **性能对比**: 测量throughput和latency的改善
4. **负载均衡**: 验证Mixed是否实现更好的跨replica负载均衡

---

## 📖 指标术语表

本节详细解释所有可视化图表中使用的性能和分析指标。

### 性能指标

#### Throughput (吞吐量)
- **定义**: 系统单位时间内完成的请求数量
- **单位**: requests/time
- **计算**: `total_completions / simulation_time`
- **解释**: 越高表示系统处理能力越强
- **示例**: 1.51 req/time 表示每个时间单位完成1.51个请求

#### Latency (延迟)
- **定义**: 单个请求从到达到完成的平均时间
- **单位**: time/request
- **计算**: `sum(completion_time - arrival_time) / total_completions`
- **解释**: 越低表示用户等待时间越短
- **示例**: 0.6615 time 表示平均每个请求耗时0.6615个时间单位

### 分布分析指标

#### Active Stages (活跃stage数量) ⭐ 核心指标
- **定义**: 某时刻同时有requests的不同decode stages的数量
- **取值范围**: 1 到 max_length
- **计算**: `count({length | request_count[length] > 0})`
- **核心含义**:
  - **少 (2个)** = 高震荡，被困在limit cycle
  - **多 (4-6个)** = 低震荡，requests分布广泛
- **示例**:
  - Segregated: 2.0 stages → 只在2个stages间反复震荡 (limit cycle)
  - Mixed: 4.3 stages → requests分散在多个stages (打破limit cycle)

#### Gini Coefficient (基尼系数)
- **定义**: 衡量stage分布不平等程度的经济学指标
- **取值范围**: 0 到 1
- **计算**: `(2 * Σ(i * sorted_counts[i]) - (n+1) * total) / (n * total)`
- **解释**:
  - **0 = 完全均匀**: 所有stages的request数量相同 (如 6,6,6,6)
  - **1 = 完全集中**: 所有requests都在一个stage (如 24,0,0,0)
  - **0.3 左右**: 有一定集中度但不极端
- **示例**:
  - Segregated: 0.261 → 分布相对均匀，但只有2个active stages
  - Mixed: 0.313 → 稍高，但分散在更多stages

#### Max Stage Concentration (最大stage占比)
- **定义**: 请求数量最多的那个stage占总请求的百分比
- **取值范围**: 0% 到 100%
- **计算**: `max(counts) / sum(counts)`
- **解释**:
  - **高 (>70%)**: requests高度集中在某一个stage
  - **低 (<50%)**: requests分布比较均衡
- **示例**:
  - Segregated: 76.5% → 四分之三的requests集中在一个stage
  - Mixed: 44.1% → 最大的stage只占不到一半，更均衡

#### Shannon Entropy (香农熵)
- **定义**: 信息论中衡量分布多样性的指标
- **取值范围**: 0 到 log₂(n)，归一化后为 0 到 1
- **计算**: `H = -Σ(p_i * log₂(p_i))` 其中 `p_i = count_i / total`
- **解释**:
  - **高 (接近1)**: 分布均匀，多样性高
  - **低 (接近0)**: 集中在少数几个类别
- **应用场景**:
  - Type Diversity: 衡量不同request types的混合程度
  - Stage Diversity: 衡量requests在不同stages的分布
- **示例**: Mixed在各length上的熵值更高 → type混合更好

#### Coefficient of Variation (CV, 变异系数)
- **定义**: 标准差与均值的比值，衡量分布的相对波动
- **取值范围**: 0 到 ∞
- **计算**: `CV = std / mean`
- **解释**:
  - **低 (接近0)**: 数据波动小，分布均匀
  - **高 (>1)**: 数据波动大，分布不均
- **应用场景**:
  - Type Distribution: 衡量各type数量的均衡性
  - Stage Distribution: 衡量各stage request count的均匀性
- **优势**: 无量纲，可跨不同scale比较

### 稳定性指标

#### Standard Deviation (标准差)
- **定义**: 衡量数据偏离平均值的程度
- **计算**: `std = sqrt(Σ(x_i - mean)² / n)`
- **应用**:
  - Active Stages的std: 衡量震荡的稳定性
  - Gini的std: 衡量集中度的波动
- **示例**:
  - Segregated Active Stages: std=0.09 → 极稳定地困在2个stages
  - Mixed Active Stages: std=1.13 → 有波动，在多个stages间切换

### 指标间关系

**检测Limit Cycle的指标组合**:
1. **Active Stages = 2.0** + **低std** → 被困在limit cycle（高震荡）
2. **Max Concentration > 75%** → 高度集中
3. **Gini适中但Active Stages少** → Gini不能单独判断limit cycle

**Mixed打破Limit Cycle的证据**:
1. **Active Stages = 4.3** (是Segregated的2倍+) → 分布广泛
2. **Max Concentration = 44%** (比Segregated低32%) → 更均衡
3. **Throughput +7.68%** → 性能提升

---

## 📊 可视化图表说明

### 1. 性能对比图 (Performance Comparison)

**文件**: `performance_comparison.png`

**内容**: Segregated vs Mixed的throughput和latency对比

**使用指标**:
- **Throughput** (见[指标术语表](#throughput-吞吐量)): 系统单位时间完成的请求数
- **Latency** (见[指标术语表](#latency-延迟)): 请求从到达到完成的平均时间

**子图**:
- **左**: Total Throughput (requests/time)
  - Segregated: 1.40 req/time
  - Mixed: 1.51 req/time
  - **Improvement: +7.68%** ✅
  - 说明: Mixed每个时间单位多处理0.11个请求

- **右**: Average Latency (time/request)
  - Segregated: 0.7123 time
  - Mixed: 0.6615 time
  - **Reduction: -7.13%** ✅
  - 说明: Mixed使用户平均等待时间减少约0.05时间单位

**关键发现**: Mixed routing在GPU容量充足时显著提升性能

---

### 2. Batch组成对比图 (Batch Composition Comparison)

**文件**: `batch_composition_comparison.png`

**内容**: 分析不同request types在各个length上的分布

**使用指标**:
- **Shannon Entropy** (见[指标术语表](#shannon-entropy-香农熵)): 衡量type分布的多样性
- **Coefficient of Variation** (见[指标术语表](#coefficient-of-variation-cv-变异系数)): 衡量type数量的均衡性

**子图**:
- **左上**: Segregated - Type Distribution by Length
  - 堆叠条形图，展示各type在不同lengths的分布
  - 可见某些lengths只有特定types

- **右上**: Mixed - Type Distribution by Length
  - 相比segregated，type分布更均匀
  - 各lengths都有多种types混合

- **左下**: Type Diversity at Each Length (Shannon熵)
  - **指标解释**: 熵值高 = types混合度好
  - Mixed场景在各length上的熵值更高
  - 证明Mixed实现了更好的type多样性

- **右下**: Overall Type Distribution (CV对比)
  - **指标解释**: CV低 = 各type数量更均衡
  - Segregated: Type分布不均（某些type缺失，CV高）
  - Mixed: 所有4种types都有合理分布（CV低）

**关键发现**: Mixed routing实现了更好的type多样性和均衡性

---

### 3. Stage分布对比图 (Stage Distribution Comparison)

**文件**: `stage_distribution_comparison.png`

**内容**: 单个replica内不同decode stages的分布快照

**使用指标**:
- **Shannon Entropy** (见[指标术语表](#shannon-entropy-香农熵)): 衡量stage分布的多样性
- **Coefficient of Variation** (见[指标术语表](#coefficient-of-variation-cv-变异系数)): 衡量stage分布的均匀性

**子图** (2行3列):
- **第1列**: Segregated Replica 0和Replica 1的stage分布
  - 展示稳态下各length的平均request count
  - 包含Entropy和CV指标数值

- **第2列**: Mixed Replica 0和Replica 1的stage分布
  - 对比segregated，stage分布更广泛
  - 各replica的Entropy和CV值

- **第3列**: 指标对比
  - **上**: Stage Diversity Comparison (Shannon熵)
    - 熵值越高，stage分布越多样化
  - **下**: Stage Distribution Uniformity (CV)
    - CV越低，各stage的request count越均匀

**关键发现**:
- Segregated和Mixed的单个replica内stage分布形状相似
- 都呈现指数衰减分布（prompt stage多，completion stage少）
- Mixed的优势在于跨replicas的负载均衡（两个replicas处理相同types）

---

### 4. Stage分布稳定性分析 (Stage Stability Over Time)

**文件**: `stage_stability_over_time.png`

**内容**: Stage分布随时间的稳定性对比

**使用指标**:
- **Shannon Entropy** (见[指标术语表](#shannon-entropy-香农熵)): 衡量stage多样性的时间演化
- **Coefficient of Variation** (见[指标术语表](#coefficient-of-variation-cv-变异系数)): 衡量stage分布均匀性的时间演化
- **Standard Deviation** (见[指标术语表](#standard-deviation-标准差)): 衡量指标的波动程度

**子图** (4行2列):

- **第1行**: Heatmap - Stage分布随时间演化
  - **X轴**: Batch number (时间，200-1000，warmup后）
  - **Y轴**: Length (decode stage)
  - **颜色**: Request count (深色=多，浅色=少)
  - **观察**: Segregated vs Mixed对比
    - 水平条纹 = 特定stages长期有requests
    - 证明系统在稳态下的stage分布模式

- **第2行**: Shannon Entropy over time
  - **指标解释**: 熵值的时间序列，衡量stage多样性
  - **观察**: Mixed和Segregated的熵值都相对稳定
  - **含义**: 两种场景的stage分布模式都已收敛

- **第3行**: CV over time
  - **指标解释**: CV的时间序列，衡量stage分布的不均匀程度
  - **观察**: 展示stage分布均匀性随时间的变化
  - **波动**: 反映系统动态但稳定的特性

- **第4行**: 震荡幅度统计
  - **左**: Entropy标准差 (std)
    - 衡量Shannon熵的波动大小
    - 越小说明stage多样性越稳定
  - **右**: CV标准差 (std)
    - 衡量分布均匀性的波动大小
    - 越小说明分布模式越稳定

**关键发现**:
- 两种场景的stage分布都达到**稳态** (熵和CV波动小)
- Heatmap显示清晰的**水平条纹模式**，证明特定stages长期活跃
- 这个图展示**时间稳定性**，但不直接反映limit cycle (需看Active Stages数量)

---

### 5. Stage震荡分析 (Stage Oscillation Analysis) ⭐ 最关键图表

**文件**: `stage_oscillation_analysis.png`

**内容**: 时间序列分析，直接展示stage分布的震荡行为，是检测limit cycle的核心证据

**使用指标**:
- **Active Stages** (见[指标术语表](#active-stages-活跃stage数量-核心指标)): 同时有requests的不同stages数量
- **Gini Coefficient** (见[指标术语表](#gini-coefficient-基尼系数)): stage分布的不平等程度
- **Max Stage Concentration** (见[指标术语表](#max-stage-concentration-最大stage占比)): 最大stage占总requests的比例
- **Standard Deviation** (见[指标术语表](#standard-deviation-标准差)): 衡量时间序列的波动程度

**子图** (3行1列):

#### 5.1 Active Stages数量 ⭐⭐⭐ 震荡核心指标
- **X轴**: Batch number (200-1000, warmup后)
- **Y轴**: Number of active stages
- **指标含义** (见术语表): Active stages少 = 高震荡 (limit cycle), 多 = 低震荡 (分布广泛)

**结果**:
- **Segregated (红线)**:
  - **均值**: 2.0 stages
  - **标准差**: 0.09 (极稳定，几乎不波动)
  - **含义**: 被困在limit cycle，系统**持续在2个固定stages间震荡**
  - **证据强度**: ⭐⭐⭐ 这是limit cycle的直接证据！

- **Mixed (青线)**:
  - **均值**: 4.3 stages
  - **标准差**: 1.13 (有波动，在2-6之间)
  - **含义**: requests分散在多个stages，**成功打破limit cycle**
  - **改善**: 相比Segregated增加了2.3个active stages (+115%)

**关键理解**:
- Active stages = 2 → 系统只在两个decode lengths间来回震荡
- Active stages = 4-5 → 系统在多个lengths间分布，避免了震荡

#### 5.2 Gini系数 (集中度时间序列)
- **X轴**: Batch number
- **Y轴**: Gini Coefficient (0-1)
- **指标含义** (见术语表): 0=完全均匀，1=完全集中

**结果**:
- **Segregated (红线)**:
  - **均值**: 0.261
  - **特点**: 非常稳定，几乎无波动
  - **解释**: 虽然Gini不高，但结合Active Stages=2，说明requests均匀分布在**仅2个stages**上

- **Mixed (青线)**:
  - **均值**: 0.313 (稍高)
  - **特点**: 有一定波动
  - **解释**: Gini稍高是因为有时某个stage会临时集中，但整体分布在更多stages上

**重要**: Gini系数不能单独判断limit cycle，必须结合Active Stages数量！

#### 5.3 最大Stage占比 (Max Concentration时间序列)
- **X轴**: Batch number
- **Y轴**: Max Stage Concentration (0-100%)
- **指标含义** (见术语表): 请求数最多的stage占总数的百分比

**结果**:
- **Segregated (红线)**:
  - **稳定在**: 75-80%
  - **均值**: 76.5%
  - **含义**: 四分之三的requests集中在一个stage，**高度集中**
  - **波动**: 极小（因为困在limit cycle）

- **Mixed (青线)**:
  - **范围**: 30-80%（大幅波动）
  - **均值**: 44.1%
  - **含义**: 最大的stage只占不到一半，**分布更均衡**
  - **波动**: 大（说明系统动态调整，不同时刻不同stage占主导）

**对比**: Mixed的max concentration比Segregated低32个百分点

---

**统计总结**:
```
Active Stages (核心):
  Segregated: 2.0 ± 0.09  → HIGH oscillation (limit cycle证据)
  Mixed:      4.3 ± 1.13  → LOW oscillation (打破limit cycle)

Gini Coefficient:
  Segregated: 0.261 ± 0.004  → 稳定但只有2个stages
  Mixed:      0.313 ± 0.091  → 稍高但分散在多个stages

最大Stage占比:
  Segregated: 76.5% ± 1.2%   → 高度集中
  Mixed:      44.1% ± 10.8%  → 更均衡
```

**关键发现**:
- ✅ **Limit Cycle证据**: Segregated的2个active stages + 极低std(0.09) = 被困在固定震荡模式
- ✅ **打破Limit Cycle**: Mixed的4.3个active stages = 成功扩展到更多stages
- ✅ **性能改善**: 打破limit cycle直接导致throughput提升7.68%
- ✅ **理论验证**: 证明了non-coprime GCD条件下的limit cycle理论

---

### 6. GPU State Evolution - Segregated

**目录**: `segregated/`

**文件**:
- `replica_0_gpu_state.png`: Replica 0的GPU state随时间演化
- `replica_1_gpu_state.png`: Replica 1的GPU state随时间演化
- `all_replicas_segregated_comparison.png`: 两个replicas并排对比

**图表内容**:
- **X轴**: Batch number (时间步)
- **Y轴**: Request count (该(length, type)组合的请求数量)
- **曲线**: 每条曲线代表一个(length, type)组合
  - 曲线颜色区分不同的组合
  - 曲线上下波动反映该组合的请求数量变化

**观察要点**:

1. **Replica 0** (处理Group 1: Type 1,2):
   - 只有与l1=8和l1=16相关的曲线
   - 曲线呈现**周期性震荡**模式
   - **Limit Cycle特征**: 特定lengths的曲线在固定值间来回跳动
   - 例如: length=8的曲线和length=16的曲线交替出现峰值

2. **Replica 1** (处理Group 2: Type 3,4):
   - 只有与l1=5和l1=15相关的曲线
   - 同样呈现周期性震荡
   - 曲线模式与Replica 0不同（因为处理不同types）

3. **两个Replicas对比**:
   - 两个replicas的曲线完全不同（处理不同types）
   - **负载不均**: 由于不同types的l0+l1总和不同，两个replicas的总负载可能不均衡
   - 这是Segregated routing的固有缺陷

**Limit Cycle的视觉证据**:
- 曲线的周期性重复模式
- 特定lengths的曲线在两个固定值间震荡
- 例如: 曲线从高→低→高→低的固定模式

---

### 7. GPU State Evolution - Mixed

**目录**: `mixed/`

**文件**:
- `replica_0_gpu_state.png`: Replica 0的GPU state随时间演化
- `replica_1_gpu_state.png`: Replica 1的GPU state随时间演化
- `all_replicas_mixed_comparison.png`: 两个replicas并排对比

**图表内容**: 与Segregated相同的格式

**观察要点**:

1. **Replica 0和Replica 1** (均处理所有4种Types):
   - 两个replicas的曲线**几乎完全相同**
   - 同时包含l1=8, 16, 5, 15相关的曲线
   - 证明请求被均匀分配

2. **曲线特征**:
   - **更多曲线**: 每个replica有4种types的曲线（vs Segregated的2种）
   - **曲线叠加**: 多种types的曲线相互叠加
   - **波动模式**: 不再是简单的周期性震荡，而是更复杂的动态

3. **打破Limit Cycle的证据**:
   - 曲线不再呈现简单的周期性重复
   - 多种types的混合导致更复杂、更分散的分布
   - 没有明显的"高→低→高→低"固定模式

4. **负载均衡**:
   - 两个replicas的总负载几乎相同
   - 曲线形状几乎完全一致
   - **std ≈ 0**: 完美的跨replica负载均衡

**Segregated vs Mixed对比总结**:

| 特征 | Segregated | Mixed |
|------|------------|-------|
| 每个Replica的Types | 2种 | 4种 |
| 曲线数量 | 少 | 多 |
| 震荡模式 | 周期性 (limit cycle) | 复杂动态 |
| Replica间差异 | 大 (完全不同) | 小 (几乎相同) |
| 负载均衡 | 差 | 完美 |

---

## 🎯 核心结论

### 性能提升
- **Throughput**: +7.68%
- **Latency**: -7.13%
- **Load Balance**: 完美 (std=0.0 vs 1046)

### Limit Cycle现象
- **Segregated**: 被困在2个active stages的limit cycle
- **Mixed**: 打破limit cycle，扩展到4.3个active stages
- **证据**: Active stages数量是震荡程度的直接指标

### 适用场景
- ✅ GPU容量充足 (B/max_request_size > 25): Mixed显著更优
- ⚠️  GPU容量受限 (B/max_request_size < 3): Segregated可能更好
- ✅ 需要公平性: Mixed确保所有types都被处理

---

## 📁 文件列表

```
mixing_results/
├── README.md                                    # 本文档
├── performance_comparison.png                   # 性能对比
├── batch_composition_comparison.png             # Batch组成分析
├── stage_distribution_comparison.png            # Stage分布快照
├── stage_stability_over_time.png                # Stage稳定性
├── stage_oscillation_analysis.png               # Stage震荡分析 ⭐
├── segregated/
│   ├── replica_0_gpu_state.png
│   ├── replica_1_gpu_state.png
│   └── all_replicas_segregated_comparison.png
└── mixed/
    ├── replica_0_gpu_state.png
    ├── replica_1_gpu_state.png
    └── all_replicas_mixed_comparison.png
```

---

## 🔬 理论映射

**对应论文章节**: Multiple request types, GCD stability condition

**关键理论**:
- Non-coprime GCD → Limit cycle (Segregated场景)
- Mixed routing → 打破limit cycle，改善性能
- Active stages数量是limit cycle的直接指标

**实验验证**:
- ✅ Segregated收敛到2个stages的limit cycle
- ✅ Mixed扩展到4.3个active stages
- ✅ Mixed实现+7.68% throughput提升

---

**生成日期**: 2026-01-03
**实验配置**: `experiments/multi_replica_mixing_results.json`
**Git Commit**: e4d0e42 (初始实验) → 本次文档更新见最新commit
