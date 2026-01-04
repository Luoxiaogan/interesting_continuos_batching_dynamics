# 第三阶段：G_A, G_B, G_merge 分解图

## 目标

在现有 `G_comparison.png` 基础上，新增第三张图 `G_decomposed.png`，包含三个子图：
- **G_A**: Type A 的 max - min
- **G_B**: Type B 的 max - min
- **G_merge**: 合并后（带补偿）的 max - min

每个子图对比 Theory 和 Simulation 两条线。

---

## G Metrics 定义

### 例子：l₀=2, l_A=2, l_B=3

状态结构：
```
        Type A    Type B
Stage 0:  X[0][A]   X[0][B]    ← 两种 type 都有
Stage 1:  X[1][A]   X[1][B]    ← A 最后一步
Stage 2:    -       X[2][B]    ← 只有 B（A 已完成）
```

假设某时刻：
```
Stage 0: [A: 3.0, B: 4.0]
Stage 1: [A: 2.5, B: 3.5]
Stage 2: [A: N/A, B: 2.0]
```

### G_A（只看 Type A）

```
values_A = [3.0, 2.5]  # stage 0, 1
G_A = max(values_A) - min(values_A) = 3.0 - 2.5 = 0.5
```

### G_B（只看 Type B）

```
values_B = [4.0, 3.5, 2.0]  # stage 0, 1, 2
G_B = max(values_B) - min(values_B) = 4.0 - 2.0 = 2.0
```

### G_merge（合并 + 补偿）

**Step 1: 按 stage 合并**
```
merged[0] = 3.0 + 4.0 = 7.0
merged[1] = 2.5 + 3.5 = 6.0
merged[2] = 0 + 2.0 = 2.0  ← 只有 B，需要补偿
```

**Step 2: 补偿缺失的 type**

对于 stage ≥ l_A（只剩 B）：
```
补偿系数 = (λ_A + λ_B) / λ_B
merged[2] = 2.0 × (λ_A + λ_B) / λ_B
```

如果 λ_A = λ_B = 1.0：
```
merged[2] = 2.0 × 2.0 = 4.0
```

**Step 3: 计算 G_merge**
```
merged = [7.0, 6.0, 4.0]
G_merge = max(merged) - min(merged) = 7.0 - 4.0 = 3.0
```

### 补偿逻辑通用化

```python
if l_A < l_B:
    # Stage l_A 到 l_B-1 只有 B
    补偿系数 = (λ_A + λ_B) / λ_B

elif l_A > l_B:
    # Stage l_B 到 l_A-1 只有 A
    补偿系数 = (λ_A + λ_B) / λ_A

else:  # l_A == l_B
    # 不需要补偿
```

---

## 实现步骤

### Step 1: 扩展 `metrics.py`

新增三个函数：

```python
def compute_G_A(state, l0, l_A, l_B, state_format='stage') -> float:
    """计算 Type A 的 G = max - min"""
    # 只取 Type A 的有效 stages (0 到 l_A-1)

def compute_G_B(state, l0, l_A, l_B, state_format='stage') -> float:
    """计算 Type B 的 G = max - min"""
    # 只取 Type B 的有效 stages (0 到 l_B-1)

def compute_G_merge(state, l0, l_A, l_B, lambda_A, lambda_B,
                    state_format='stage') -> float:
    """计算合并后（带补偿）的 G = max - min"""
    # 1. 按 stage 合并 A + B
    # 2. 对只有一种 type 的 stages 进行补偿
    # 3. 计算 max - min
```

### Step 2: 修改 `run_coupling_experiment.py`

在 trajectory 构建循环中，新增计算：

```python
# 现有
theory_G = compute_G(theory_state, l0, l_A, l_B, state_format='stage')
sim_G = compute_G(sim_state, l0, l_A, l_B, state_format='length')

# 新增
theory_G_A = compute_G_A(theory_state, l0, l_A, l_B, state_format='stage')
theory_G_B = compute_G_B(theory_state, l0, l_A, l_B, state_format='stage')
theory_G_merge = compute_G_merge(theory_state, l0, l_A, l_B, lambda_A, lambda_B, state_format='stage')

sim_G_A = compute_G_A(sim_state, l0, l_A, l_B, state_format='length')
sim_G_B = compute_G_B(sim_state, l0, l_A, l_B, state_format='length')
sim_G_merge = compute_G_merge(sim_state, l0, l_A, l_B, lambda_A, lambda_B, state_format='length')

row = {
    ...,
    'theory_G_A': theory_G_A,
    'theory_G_B': theory_G_B,
    'theory_G_merge': theory_G_merge,
    'sim_G_A': sim_G_A,
    'sim_G_B': sim_G_B,
    'sim_G_merge': sim_G_merge,
}
```

### Step 3: 新增可视化函数 `visualize_coupling.py`

```python
def plot_G_decomposed(trajectory: List[Dict],
                      output_path: Optional[Path] = None,
                      title: str = "G Decomposition"):
    """
    绘制三个子图：G_A, G_B, G_merge
    每个子图包含 Theory（蓝线）和 Simulation（绿线）
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    batches = [row['batch'] for row in trajectory]

    # Subplot 1: G_A
    ax1 = axes[0]
    ax1.plot(batches, [row['theory_G_A'] for row in trajectory], 'b-', label='Theory')
    ax1.plot(batches, [row['sim_G_A'] for row in trajectory], 'g-', label='Simulation')
    ax1.set_title('G_A (Type A only)')
    ax1.legend()

    # Subplot 2: G_B
    ax2 = axes[1]
    ax2.plot(batches, [row['theory_G_B'] for row in trajectory], 'b-', label='Theory')
    ax2.plot(batches, [row['sim_G_B'] for row in trajectory], 'g-', label='Simulation')
    ax2.set_title('G_B (Type B only)')
    ax2.legend()

    # Subplot 3: G_merge
    ax3 = axes[2]
    ax3.plot(batches, [row['theory_G_merge'] for row in trajectory], 'b-', label='Theory')
    ax3.plot(batches, [row['sim_G_merge'] for row in trajectory], 'g-', label='Simulation')
    ax3.set_title('G_merge (compensated)')
    ax3.legend()

    plt.tight_layout()
    # Save...
```

### Step 4: 在主脚本调用新可视化

```python
plot_G_decomposed(
    trajectory,
    output_path=output_dir / 'G_decomposed.png',
    title=f"G Decomposition: l0={config['l0']}, l_A={config['l_A']}, l_B={config['l_B']}"
)
```

---

## 输出文件

```
outputs/YYYYMMDD_HHMMSS/
├── comparison.png        # 图1: Admission/Eviction
├── eviction_detail.png   # 图2: Eviction by Stage
├── G_comparison.png      # 图3: G (combined) - 保持不变
├── G_decomposed.png      # 图4: G_A, G_B, G_merge 三子图 [新增]
├── trajectory.csv        # 新增 6 列
└── config.json
```

### trajectory.csv 新增列

```csv
batch,...,theory_G,sim_G,theory_G_A,theory_G_B,theory_G_merge,sim_G_A,sim_G_B,sim_G_merge
```

---

## 关键文件修改清单

| 文件 | 修改内容 |
|------|---------|
| `metrics.py` | 新增 `compute_G_A`, `compute_G_B`, `compute_G_merge` |
| `run_coupling_experiment.py` | 计算并记录 6 个新指标 |
| `visualize_coupling.py` | 新增 `plot_G_decomposed` 函数 |

---

## 预期结果

- **G_A** 和 **G_B**：分别跟踪单一 type 的收敛，预期都趋近于 0
- **G_merge**：合并后（补偿缺失 type）的收敛，预期趋近于 0
- Theory 和 Simulation 的曲线应该接近但不完全重合（因为 eviction 机制差异）

---

## 注意事项

1. **补偿系数需要 λ_A, λ_B**：`compute_G_merge` 函数签名需要这两个参数
2. **state_format 统一**：Theory 用 'stage'，Simulation 用 'length'
3. **边界情况**：l_A = l_B 时不需要补偿
4. **保持 G_comparison.png 不变**：这是原有功能，不修改
