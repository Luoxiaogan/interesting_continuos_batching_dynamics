# Experiments Directory - 实验配置管理

## 📋 目录说明

此目录用于存储**可重复实验的配置文件**，确保研究结果的可追溯性和可重现性。

## 🎯 目录结构

```
experiments/
├── README.md                    # 本文档
├── exp_*.json                   # 命名实验配置文件
├── archive/                     # 已完成实验的归档
│   ├── theorem1_verification/   # 示例：Theorem 1验证
│   │   ├── config.json
│   │   ├── sim_YYYYMMDD_HHMMSS/  # 模拟输出（从output/复制）
│   │   └── README.md            # 实验说明和结论
│   └── gcd_stability/
└── scripts/                     # 实验辅助脚本（可选）
    ├── parameter_sweep.py       # 参数扫描脚本
    └── batch_analysis.py        # 批量分析脚本
```

## ⭐ 配置文件命名规范

**格式**: `exp_<简短描述>_config.json`

**示例**:
- `exp_theorem1_greedy_instability_config.json` - Theorem 1验证
- `exp_theorem2_gcd_coprime_2_3_config.json` - Theorem 2互质情况
- `exp_admission_threshold_sweep_config.json` - 准入阈值扫描

**禁止**:
- ❌ `experiment1.json` (无语义)
- ❌ `test_config.json` (太通用)
- ❌ `config_v2.json` (版本号混乱)

## 📝 配置文件标准格式

```json
{
  "experiment_name": "Theorem 2 GCD Stability Verification - Coprime Case",
  "experiment_purpose": "验证gcd(l_A, l_B) = 1时系统收敛到no-eviction equilibrium",
  "theoretical_basis": "multiple_discrete.tex lines 120-122",
  "related_paper_section": "Section 4.2 - Multi-Class Stability",

  "request_types": [[5, 2], [5, 3]],
  "B": 50,
  "arrival_rates": [8.0, 4.0],
  "b0": 0.1,
  "b1": 0.01,
  "initial_state": {},
  "steps": 1000,
  "precision": 10,

  "expected_result": "状态差异应指数衰减到0，收敛到no-eviction equilibrium",
  "verification_metric": "final_state_variance < 1e-6",

  "notes": "对比实验: exp_theorem2_gcd_non_coprime_2_4_config.json (gcd=2)"
}
```

### 必需字段

| 字段 | 说明 | 示例 |
|-----|------|------|
| `experiment_name` | 实验名称 | "Theorem 1 Verification" |
| `experiment_purpose` | 实验目的（简短） | "验证Greedy策略不稳定性" |
| `theoretical_basis` | 对应的理论文件 | "single_discrete.tex lines 88-91" |
| `request_types` | 请求类型 `[[l0, l1], ...]` | `[[2, 5]]` |
| `B` | GPU容量 | `50` |
| `arrival_rates` | 到达率列表 | `[8.0, 4.0]` |
| `steps` | 模拟步数 | `1000` |

### 可选但推荐的字段

| 字段 | 说明 | 示例 |
|-----|------|------|
| `expected_result` | 理论预期结果 | "收敛到limit cycle" |
| `verification_metric` | 验证指标 | "throughput ≈ B/(l1*(l0+l1))" |
| `notes` | 额外说明 | "对比实验: exp_xxx.json" |
| `random_seed` | 随机种子（如使用） | `42` |

## 🔄 实验工作流

### 1. 创建配置文件

```bash
# 复制模板
cp experiments/TEMPLATE_config.json experiments/exp_my_experiment_config.json

# 编辑配置
vim experiments/exp_my_experiment_config.json
```

### 2. 运行实验

```bash
cd new_project_for_multi_type

# 使用配置文件运行
python run_simulation.py --config ../experiments/exp_my_experiment_config.json
```

### 3. 检查结果

```bash
# 查看最新的输出
cd new_project_for_multi_type/output/sim_YYYYMMDD_HHMMSS/

# 查看摘要
cat summary.txt

# 查看图表
open state_evolution_from_0.png
open state_differences_from_0_jump_1.png
```

### 4. 归档重要实验

```bash
# 创建归档目录
mkdir -p experiments/archive/my_experiment/

# 复制配置文件
cp experiments/exp_my_experiment_config.json \
   experiments/archive/my_experiment/config.json

# 复制模拟输出
cp -r new_project_for_multi_type/output/sim_YYYYMMDD_HHMMSS/ \
      experiments/archive/my_experiment/

# 创建实验说明
cat > experiments/archive/my_experiment/README.md <<EOF
# 实验: XXX

## 实验目的
...

## 理论依据
...

## 主要结果
...

## 结论
...

EOF
```

## 📊 实验与论文图表的对应关系

**维护此映射表，确保论文图表可重现**:

| 论文图/表 | 实验配置文件 | 归档位置 | 生成脚本 | 备注 |
|----------|------------|---------|---------|------|
| Figure 1 | `exp_fig1_single_class_limit_cycle.json` | `archive/fig1_single_class/` | `visualization.py --plot state_evolution` | Theorem 1演示 |
| Figure 2 | `exp_fig2_gcd_comparison.json` | `archive/fig2_gcd_comparison/` | `custom_analysis/plot_convergence_comparison.py` | Theorem 2对比 |
| Table 1 | `exp_table1_metrics.json` | `archive/table1_metrics/` | `custom_analysis/compute_metrics.py` | 吞吐量和公平性 |
| Multi-Replica Mixing | `exp_multi_replica_mixing_config.json` | `mixing_results/` | `scripts/run_mixing_experiment.py` | Segregated vs Mixed routing性能对比 |
| Stage Oscillation Analysis | N/A | `mixing_results/` | `scripts/analyze_stage_oscillation.py` | Stage分布震荡行为分析 (limit cycle检测) |

**更新规则**:
- 每次为论文生成图表时，更新此表
- 配置文件必须保存在 `experiments/`
- 原始数据必须归档在 `experiments/archive/`

## 🧪 实验配置模板

### 模板1: 单类型实验

```json
{
  "experiment_name": "Single-Type Limit Cycle Verification",
  "experiment_purpose": "观察单类型系统收敛到limit cycle",
  "theoretical_basis": "single_discrete.tex",

  "request_types": [[2, 5]],
  "B": 50,
  "arrival_rates": [20.0],
  "b0": 0.1,
  "b1": 0.01,
  "initial_state": {},
  "steps": 1000,
  "precision": 10,

  "expected_result": "收敛到level-4 limit cycle（最差平衡点）"
}
```

### 模板2: 多类型GCD稳定性

```json
{
  "experiment_name": "Multi-Type GCD Stability - Coprime Case",
  "experiment_purpose": "验证gcd(l_A, l_B) = 1的稳定性",
  "theoretical_basis": "multiple_discrete.tex lines 120-122",

  "request_types": [[5, 2], [5, 3]],
  "B": 50,
  "arrival_rates": [8.0, 4.0],
  "b0": 0.1,
  "b1": 0.01,
  "initial_state": {},
  "steps": 1000,
  "precision": 10,

  "expected_result": "状态差异指数衰减，收敛到no-eviction equilibrium",
  "verification_metric": "final_variance < 1e-6"
}
```

### 模板3: 参数扫描

```json
{
  "experiment_name": "B Capacity Sweep",
  "experiment_purpose": "扫描不同GPU容量下的吞吐量",
  "theoretical_basis": "系统性能分析",

  "parameter_sweep": {
    "B_range": [30, 40, 50, 60, 70],
    "fixed_params": {
      "request_types": [[2, 5]],
      "arrival_rates": [10.0],
      "b0": 0.1,
      "b1": 0.01,
      "steps": 500
    }
  },

  "expected_result": "吞吐量随B线性增长"
}
```

### 模板4: Multi-Replica Mixing实验

```json
{
  "experiment_name": "Multi-Replica Request Mixing - 4 Types",
  "experiment_purpose": "验证heterogeneous request mixing对multi-replica系统的性能影响",
  "theoretical_basis": "Load balancing in multi-replica LLM serving systems",

  "num_replicas": 2,
  "request_groups": {
    "group1": [[4, 8], [4, 16]],
    "group2": [[3, 5], [3, 15]]
  },
  "gcd_properties": {
    "group1_gcd": 8,
    "group2_gcd": 5
  },
  "B": 500,
  "arrival_rates": [1.0, 1.0, 1.0, 1.0],
  "b0": 0.1,
  "b1": 0.01,
  "steps": 1000,

  "scenarios": ["segregated", "mixed"],
  "expected_result": "Mixed routing应实现更好的负载均衡，在GPU容量充足时提升throughput",
  "verification_metrics": [
    "total_throughput (requests/time)",
    "avg_latency (time/request)",
    "load_balance_std",
    "per_replica_convergence"
  ],

  "output_files": [
    "experiments/multi_replica_mixing_results.json",
    "experiments/mixing_results/performance_comparison.png",
    "experiments/mixing_results/{segregated,mixed}/replica_*_gpu_state.png"
  ],

  "notes": "Request size必须适配B容量，max_request_size/B比例建议 < 5%"
}
```

**运行方法**:
```bash
python experiments/scripts/run_mixing_experiment.py
```

**关键发现**:
- GPU容量充足时 (B/max_request_size > 25): Mixed routing显著提升throughput (+7.68%)
- GPU容量受限时 (B/max_request_size < 3): Segregated routing可能更优
- Mixed routing始终实现完美负载均衡 (std=0.0)

### 模板5: Stage震荡分析 (Limit Cycle检测)

**运行方法**:
```bash
python experiments/scripts/analyze_stage_oscillation.py
```

**分析目标**:
- 检测segregated routing中的limit cycle现象
- 量化stage分布的震荡程度
- 验证mixed routing是否能打破limit cycle

**核心指标**:
1. **Active Stages数量** (震荡核心指标)
   - 少(2个) = 高震荡，被困在limit cycle
   - 多(4-6个) = 低震荡，分布广泛

2. **Gini系数** (集中度)
   - 0 = 完全均匀分布
   - 1 = 完全集中在一个stage

3. **最大Stage占比**
   - 最大的stage占总requests的百分比

**输出文件**:
- `experiments/mixing_results/stage_oscillation_analysis.png`

**实验结果示例**:
```
Active Stages数量:
  Segregated: 2.0 ± 0.09  (HIGH oscillation - limit cycle)
  Mixed:      4.3 ± 1.13  (LOW oscillation - distributed)

结论:
  ✅ Segregated的2个active stages证明其被困在limit cycle
  ✅ Mixed的4.3个active stages说明其打破了limit cycle
```

**理论依据**:
- Segregated routing在non-coprime GCD条件下收敛到limit cycle
- Limit cycle表现为requests在少数几个stages间反复震荡
- Mixed routing通过type多样化打破limit cycle，实现更广泛的stage分布

## 📌 最佳实践

### ✅ 推荐做法

1. **实验前先设计**:
   - 明确实验目的和理论依据
   - 设计参数组合
   - 预测理论结果

2. **配置文件参数化**:
   - 避免硬编码
   - 所有参数在config.json中明确
   - 便于批量运行和对比

3. **及时归档**:
   - 重要实验立即归档
   - 保留完整的配置和数据
   - 编写简短的README说明

4. **版本追踪**:
   - 配置文件提交到git
   - 重要实验打tag
   - 关联git commit hash

### ❌ 避免做法

1. **不要直接修改代码参数**:
   - ❌ 在Python代码中硬编码参数
   - ✅ 使用配置文件或命令行参数

2. **不要丢失原始数据**:
   - ❌ 删除output/目录中的原始CSV
   - ✅ 归档到experiments/archive/

3. **不要使用无语义命名**:
   - ❌ `exp1.json`, `test.json`
   - ✅ `exp_theorem2_gcd_stability.json`

## 🔧 辅助工具

### 参数扫描脚本示例

创建 `experiments/scripts/parameter_sweep.py`:

```python
#!/usr/bin/env python3
"""
参数扫描工具

Usage:
    python experiments/scripts/parameter_sweep.py \
        --base-config experiments/exp_base.json \
        --sweep-param B \
        --sweep-values 30 40 50 60
"""

import json
import subprocess
import argparse
from pathlib import Path

def run_sweep(base_config, param, values):
    for value in values:
        # 加载基础配置
        with open(base_config) as f:
            config = json.load(f)

        # 修改扫描参数
        config[param] = value
        config['experiment_name'] += f" - {param}={value}"

        # 保存临时配置
        temp_config = f"experiments/temp_{param}_{value}.json"
        with open(temp_config, 'w') as f:
            json.dump(config, f, indent=2)

        # 运行实验
        print(f"Running: {param}={value}")
        subprocess.run([
            "python", "new_project_for_multi_type/run_simulation.py",
            "--config", temp_config
        ])

        # 清理临时文件
        Path(temp_config).unlink()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--sweep-param", required=True)
    parser.add_argument("--sweep-values", nargs='+', type=float, required=True)
    args = parser.parse_args()

    run_sweep(args.base_config, args.sweep_param, args.sweep_values)
```

### 批量分析脚本示例

创建 `experiments/scripts/batch_analysis.py`:

```python
#!/usr/bin/env python3
"""
批量分析实验结果

Usage:
    python experiments/scripts/batch_analysis.py \
        experiments/archive/my_experiment_group/
"""

import pandas as pd
import glob
import json
from pathlib import Path

def analyze_experiments(archive_dir):
    results = []

    # 遍历所有归档实验
    for exp_dir in Path(archive_dir).iterdir():
        if not exp_dir.is_dir():
            continue

        # 加载配置
        config_file = exp_dir / "config.json"
        if not config_file.exists():
            continue

        with open(config_file) as f:
            config = json.load(f)

        # 查找模拟输出目录
        sim_dirs = list(exp_dir.glob("sim_*"))
        if not sim_dirs:
            continue

        sim_dir = sim_dirs[0]

        # 分析结果
        completions = pd.read_csv(sim_dir / "completions.csv")
        total_completions = completions['completed'].sum()

        states = pd.read_csv(sim_dir / "x_prime_states.csv")
        final_states = states[states['batch'] > 900]
        variance = final_states.groupby('length')['count'].std().mean()

        results.append({
            'experiment': exp_dir.name,
            'B': config['B'],
            'request_types': str(config['request_types']),
            'total_completions': total_completions,
            'final_variance': variance,
            'converged': variance < 1e-3
        })

    # 汇总结果
    df = pd.DataFrame(results)
    df.to_csv(f"{archive_dir}/analysis_summary.csv", index=False)
    print(df)
    print(f"\n收敛率: {df['converged'].mean():.2%}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python batch_analysis.py <archive_directory>")
        sys.exit(1)

    analyze_experiments(sys.argv[1])
```

---

**文档版本**: v1.0
**最后更新**: 2026-01-02
**维护者**: @ruicheng
