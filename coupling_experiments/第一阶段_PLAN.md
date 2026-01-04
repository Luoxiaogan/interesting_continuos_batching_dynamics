# Coupling Experiments: Theory vs Simulation 对比实验实现计划

## 目标

对比 2-types coprime overloaded 系统的两条 trajectory：
- **Theory**: 无 eviction，允许负数 admission
- **Simulation**: 有 eviction，强制非负约束

展示 "diverge then converge" 行为。

---

## 文件结构

```
coupling_experiments/
├── tex_docs/                    # 已存在
├── 实现的思考.md                 # 已存在
├── theory_simulator.py          # 新建：无约束版本模拟器
├── eigenvalue_solver.py         # 新建：特征值计算（可选）
├── run_coupling_experiment.py   # 新建：主实验脚本
├── visualize_coupling.py        # 新建：可视化函数
└── outputs/                     # 新建：实验输出目录
    └── YYYYMMDD_HHMMSS/
        ├── config.json
        ├── trajectory.csv
        ├── comparison.png
        └── eviction_detail.png
```

---

## 实现步骤

### Step 1: Theory 模拟器 (`theory_simulator.py`)

核心逻辑（与 Simulation 对齐，只是移除约束）：

```python
class TheorySimulator:
    def __init__(self, l0, l_A, l_B, B, lambda_A, lambda_B):
        # 状态: X[stage][type], 可以是负数

    def update(self):
        # 1. 计算 completion_tokens（stage l_A-1 的 A 完成，stage l_B-1 的 B 完成）
        # 2. 计算 increment_tokens（所有请求 stage+1，每个增加 1 token）
        # 3. available_tokens = completion - increment（可能为负）
        # 4. admission = available_tokens / (l0+1)，按 λ 比例分配（可能为负）
        # 5. 状态推进，允许负数
        # 6. 记录 admission
```

关键点：
- 状态用 `X[stage][type]`，stage = 0, 1, ..., max(l_A, l_B)-1
- 每个 type 只在有效的 stage 范围内存在
- 允许负数

### Step 2: 运行实验脚本 (`run_coupling_experiment.py`)

```python
def run_experiment(config):
    # 1. 初始化 Theory 和 Simulation 模拟器
    # 2. 设置相同的初始条件
    # 3. 运行指定步数，记录每步的：
    #    - theory_admission (总数，可正可负)
    #    - sim_admission (总数，≥0)
    #    - sim_eviction_total (总数)
    #    - sim_eviction_by_stage (按 stage 分解)
    # 4. 保存 config.json 和 trajectory.csv
    # 5. 调用可视化函数
```

### Step 3: 可视化 (`visualize_coupling.py`)

**图1: Admission/Eviction 对比** (`comparison.png`)
- 横轴: batch index
- 纵轴: 请求数
- 蓝线: Theory admission (可正可负)
- 绿线: Simulation admission (≥0)
- 红色负值条形: Simulation eviction

**图2: Eviction Stage 分布** (`eviction_detail.png`)
- 横轴: batch index
- 堆叠条形图或热力图，显示各 stage 的 eviction 数量

### Step 4: 特征值计算器 (`eigenvalue_solver.py`) [可选]

验证参数选择是否合理：
- 计算 $A(\lambda) = \frac{-\lambda^{l_B} + p\lambda^{l_B-l_A} + q}{1-\lambda}$ 的根
- 确认 gcd(l_A, l_B)=1 时所有根 |λ| < 1
- 输出 max|λ| 作为收敛速率参考

---

## 参数配置

```python
config = {
    "l0": 3,
    "l_A": 2,
    "l_B": 3,
    "B": 60,
    "lambda_A": 1.0,
    "lambda_B": 1.0,
    "steps": 50,
}
```

初始条件：
```python
p = 0.5
N = B / (l0 + 1) = 15
X_init[stage=0][A] = 7.5
X_init[stage=0][B] = 7.5
```

---

## 关键文件依赖

- **现有模拟器**: `new_project_for_multi_type/multi_type_simulator.py`
  - 直接 import 使用，不修改

---

## 输出格式

### trajectory.csv
```csv
batch,theory_admission,sim_admission,sim_eviction_total,sim_eviction_stage0,sim_eviction_stage1,sim_eviction_stage2
0,0.0,0.0,0.0,0.0,0.0,0.0
1,-7.5,0.0,7.5,0.0,7.5,0.0
...
```

### config.json
```json
{
    "l0": 3,
    "l_A": 2,
    "l_B": 3,
    "B": 60,
    "lambda_A": 1.0,
    "lambda_B": 1.0,
    "steps": 50,
    "p": 0.5,
    "N": 15.0,
    "timestamp": "20260104_170000",
    "git_commit": "..."
}
```

---

## 实现顺序

1. **theory_simulator.py** - 核心：无约束模拟器
2. **run_coupling_experiment.py** - 主脚本：运行对比实验
3. **visualize_coupling.py** - 可视化：生成对比图
4. **eigenvalue_solver.py** - 可选：验证参数

---

## 注意事项

1. **时序对齐**: Batch 0 记录的是初始状态处理结果（admission=0, eviction=0）
2. **Stage 转换**: 代码中 `length - l0 = stage`
3. **Eviction 记录**: 需要从 Simulation 的历史中提取 `(length, amount)` 并转换为 stage
4. **比例恒定**: 初始条件按比例，保证整个过程比例恒定
