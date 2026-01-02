# Multi-Replica Mixing Experiment - 可视化结果文档

> 实验日期: 2026-01-03
> 实验脚本: `experiments/scripts/run_mixing_experiment.py`
> 分析脚本: `experiments/scripts/analyze_stage_oscillation.py`

## 📋 实验配置

**Request Types**:
- Group 1: (l0=4, l1=8), (l0=4, l1=16) - gcd(8,16) = 8
- Group 2: (l0=3, l1=5), (l0=3, l1=15) - gcd(5,15) = 5

**系统参数**:
- GPU容量: B = 500 tokens
- Replicas数量: 2
- 模拟步数: 1000 batches
- Arrival rates: [1.0, 1.0, 1.0, 1.0]

**对比场景**:
1. **Segregated**: Group 1 → Replica 0, Group 2 → Replica 1
2. **Mixed**: 所有4种types均匀分配到2个replicas

---

## 📊 可视化图表说明

### 1. 性能对比图 (Performance Comparison)

**文件**: `performance_comparison.png`

**内容**: Segregated vs Mixed的throughput和latency对比

**子图**:
- **左**: Total Throughput (requests/time)
  - Segregated: 1.40 req/time
  - Mixed: 1.51 req/time
  - **Improvement: +7.68%** ✅

- **右**: Average Latency (time/request)
  - Segregated: 0.7123 time
  - Mixed: 0.6615 time
  - **Reduction: +7.13%** ✅

**关键发现**: Mixed routing在GPU容量充足时显著提升性能

---

### 2. Batch组成对比图 (Batch Composition Comparison)

**文件**: `batch_composition_comparison.png`

**内容**: 分析不同request types在各个length上的分布

**子图**:
- **左上**: Segregated - Type Distribution by Length
  - 堆叠条形图，展示各type在不同lengths的分布

- **右上**: Mixed - Type Distribution by Length
  - 相比segregated，type分布更均匀

- **左下**: Type Diversity at Each Length (Shannon熵)
  - Mixed场景在各length上的熵值更高
  - 表示type混合度更好

- **右下**: Overall Type Distribution
  - Segregated: Type分布不均（某些type缺失）
  - Mixed: 所有4种types都有合理分布
  - Coefficient of Variation对比

**关键发现**: Mixed routing实现了更好的type多样性

---

### 3. Stage分布对比图 (Stage Distribution Comparison)

**文件**: `stage_distribution_comparison.png`

**内容**: 单个replica内不同decode stages的分布快照

**子图** (2行3列):
- **第1列**: Segregated Replica 0和Replica 1的stage分布
  - 展示稳态下各length的平均request count
  - 包含Entropy和CV指标

- **第2列**: Mixed Replica 0和Replica 1的stage分布
  - 对比segregated，stage分布更广泛

- **第3列**: 指标对比
  - 上: Stage Diversity Comparison (Shannon熵)
  - 下: Stage Distribution Uniformity (CV)

**关键发现**:
- Segregated和Mixed的单个replica内stage分布形状相似
- 都呈现指数衰减分布（prompt stage多，completion stage少）
- Mixed的优势在于跨replicas的负载均衡

---

### 4. Stage分布稳定性分析 (Stage Stability Over Time)

**文件**: `stage_stability_over_time.png`

**内容**: Stage分布随时间的稳定性对比

**子图** (4行2列):
- **第1行**: Heatmap - Stage分布随时间演化
  - X轴: Batch number (时间)
  - Y轴: Length (decode stage)
  - 颜色: Request count
  - Segregated vs Mixed对比

- **第2行**: Entropy over time
  - 展示stage多样性随时间的变化
  - Mixed和Segregated的熵值都相对稳定

- **第3行**: CV over time
  - 展示stage分布均匀性随时间的变化

- **第4行**: 震荡幅度统计
  - 左: Entropy标准差
  - 右: CV标准差

**关键发现**:
- 两种场景的stage分布都相对稳定
- Heatmap显示清晰的水平条纹模式

---

### 5. Stage震荡分析 (Stage Oscillation Analysis) ⭐

**文件**: `stage_oscillation_analysis.png`

**内容**: 时间序列分析，直接展示stage分布的震荡行为

**子图** (3行1列):

#### 5.1 Active Stages数量 (震荡核心指标)
- **Segregated (红线)**: 稳定在2.0个stages
  - 标准差: 0.09 (极稳定)
  - **含义**: 被困在limit cycle，只在2个stages间震荡
  - 这是**HIGH OSCILLATION**的证据！

- **Mixed (青线)**: 平均4.3个stages，波动在2-6之间
  - 标准差: 1.13 (有波动)
  - **含义**: requests分散在多个stages，打破limit cycle
  - 这是**LOW OSCILLATION**（分布广泛）

**关键理解**: Active stages少 = 震荡大（limit cycle）

#### 5.2 Gini系数 (集中度)
- Segregated: 0.261 (稳定)
- Mixed: 0.313 (稍高，但有波动)

#### 5.3 最大Stage占比
- **Segregated**: 稳定在75-80%
  - 说明一个stage始终占主导
  - 高度集中的分布

- **Mixed**: 波动在30-80%之间
  - 平均44.1%
  - 有时非常balanced

**统计总结**:
```
Active Stages:
  Segregated: 2.0 ± 0.09  → HIGH oscillation (limit cycle)
  Mixed:      4.3 ± 1.13  → LOW oscillation (distributed)

最大Stage占比:
  Segregated: 76.5% (高度集中)
  Mixed:      44.1% (更均衡)
```

**关键发现**:
- ✅ Segregated的2个active stages证明其被困在limit cycle
- ✅ Mixed的4.3个active stages说明其打破了limit cycle
- ✅ 这解释了Mixed为何throughput提升7.68%

---

### 6. GPU State Evolution - Segregated

**目录**: `segregated/`

**文件**:
- `replica_0_gpu_state.png`: Replica 0的GPU state随时间演化
- `replica_1_gpu_state.png`: Replica 1的GPU state随时间演化
- `all_replicas_segregated_comparison.png`: 两个replicas并排对比

**内容**:
- X轴: Batch number
- Y轴: Request count
- 多条曲线: 不同(length, type)组合的request count演化

**特征**:
- 显示limit cycle的周期性震荡模式
- Replica 0和Replica 1处理不同types，负载不均

---

### 7. GPU State Evolution - Mixed

**目录**: `mixed/`

**文件**:
- `replica_0_gpu_state.png`: Replica 0的GPU state随时间演化
- `replica_1_gpu_state.png`: Replica 1的GPU state随时间演化
- `all_replicas_mixed_comparison.png`: 两个replicas并排对比

**内容**: 与segregated相同的格式

**特征**:
- 两个replicas的负载完全平衡
- 所有4种types在两个replicas上都有分布
- 更多曲线叠加，反映type多样性

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
**Git Commit**: (待填写)
