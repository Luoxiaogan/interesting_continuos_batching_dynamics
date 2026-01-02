# 实验可重复性规范

> 研究项目的生命线：确保所有实验完全可重复

## 核心原则

**研究项目的生命线**：实验结果必须完全可重复，配置必须完整保存

## 强制保存的元数据

### 每个实验运行必须保存

#### 1. 完整的配置参数 (config.json)

```json
{
  "request_types": [[2, 5], [5, 2]],
  "B": 50,
  "arrival_rates": [8.0, 4.0],
  "b0": 0.1,
  "b1": 0.01,
  "initial_state": {
    "2": [5.0, 0.0],
    "5": [0.5, 2.0]
  },
  "steps": 1000,
  "precision": 10
}
```

#### 2. 时间戳和版本信息

- 运行时间: `YYYYMMDD_HHMMSS`
- Git commit hash (推荐)
- Python版本
- 关键依赖版本 (numpy, etc.)

#### 3. 随机种子 (如适用)

```python
np.random.seed(42)  # 必须在config中记录
```

#### 4. 实验目的和假设 (summary.txt)

```
实验目的: 验证Theorem 2 - GCD稳定性条件
理论预测: gcd(5, 2) = 1 → 系统应收敛到no-eviction equilibrium
参数设计: l_A=2, l_B=5 (互质), B=50, 高负载
预期结果: 状态差异应随时间指数衰减
```

## 实验命名规范

### 临时实验 (探索性运行)

```
new_project_for_multi_type/output/sim_20260102_214500/
```

- 格式: `sim_YYYYMMDD_HHMMSS/`
- 用途: 快速测试、参数调优
- 保留时间: 根据结果决定是否归档

### 命名实验 (重要/可重复实验)

```
experiments/exp_gcd_stability_coprime_config.json
experiments/exp_greedy_instability_l1_5_config.json
experiments/exp_admission_control_threshold_sweep_config.json
```

- 格式: `exp_<简短描述>_config.json`
- 用途: 论文图表、重要结论验证
- 必须: 配置文件 + README说明实验目的

## 结果目录标准结构

```
output/sim_20260102_214500/
├── config.json                              # 完整配置参数
├── summary.txt                              # 人类可读摘要
├── git_info.txt                             # Git commit hash (可选)
│
├── x_prime_states.csv                       # 核心状态数据
├── admissions.csv                           # 准入记录
├── evictions.csv                            # 驱逐记录
├── completions.csv                          # 完成记录
├── batch_info.csv                           # 批次元信息
│
├── state_evolution_from_0.png               # 自动生成可视化
├── state_differences_from_0_jump_1.png
└── length_total_differences_from_0_jump_1.png
```

## CSV数据格式标准

### x_prime_states.csv (核心状态数据)

```csv
batch,time,length,type,count
0,0.0000000000,2,0,5.0000000000
0,0.0000000000,2,1,0.0000000000
0,0.0000000000,5,0,0.5000000000
0,0.0000000000,5,1,2.0000000000
1,0.3500000000,2,0,7.5555555556
...
```

- **精度控制**: 默认10位小数，可配置 (PRECISION参数)
- **索引**: batch (批次号), time (累计时间)
- **状态**: length (请求长度), type (请求类型), count (请求数量)

### admissions.csv (准入记录)

```csv
batch,time,type,admitted
1,0.3500000000,0,2.5555555556
1,0.3500000000,1,1.2777777778
...
```

### evictions.csv (驱逐记录)

```csv
batch,time,type,length,evicted
5,1.7500000000,0,2,1.2000000000
5,1.7500000000,1,5,0.6000000000
...
```

### batch_info.csv (批次元信息)

```csv
batch,time,service_time,batch_size
0,0.0000000000,0.0000000000,0.0000000000
1,0.3500000000,0.3500000000,50.0000000000
...
```

## 数据追踪最佳实践

### 实验日志记录

```python
# 在 run_simulation.py 中添加
import subprocess
import sys

def save_git_info(output_dir):
    """保存git commit信息，确保实验可追溯"""
    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).decode('utf-8').strip()

        with open(f"{output_dir}/git_info.txt", 'w') as f:
            f.write(f"Git Commit: {commit_hash}\n")
            f.write(f"Python Version: {sys.version}\n")
            f.write(f"NumPy Version: {np.__version__}\n")
    except:
        pass  # Git不可用时跳过
```

### 参数验证

```python
def validate_config(config):
    """验证配置参数的合法性和一致性"""
    # 检查request_types格式
    assert all(len(rt) == 2 for rt in config['request_types'])

    # 检查arrival_rates长度匹配
    assert len(config['arrival_rates']) == len(config['request_types'])

    # 检查初始状态与request_types一致
    # ...
```

## 实验管理工作流

### 1. 设计实验

```bash
# 在 experiments/ 中创建配置文件
cat > experiments/exp_theorem2_verification_config.json <<EOF
{
  "experiment_name": "Theorem 2 GCD Stability Verification",
  "hypothesis": "gcd(l_A, l_B) = 1 ensures convergence to no-eviction equilibrium",
  "request_types": [[2, 3], [2, 5], [3, 5]],  # 三组互质对
  "B": 50,
  ...
}
EOF
```

### 2. 运行实验

```bash
# 方法1: 使用run_sim.sh（单次运行）
cd new_project_for_multi_type
./run_sim.sh

# 方法2: 使用配置文件（批量运行）
python run_simulation.py --config ../experiments/exp_theorem2_verification_config.json
```

### 3. 结果分析

```python
import pandas as pd

# 加载状态数据
df = pd.read_csv('output/sim_20260102_214500/x_prime_states.csv')

# 检查收敛性
final_batches = df[df['batch'] > 900]
convergence_check = final_batches.groupby(['batch', 'length', 'type'])['count'].std()

# 计算吞吐量
completions = pd.read_csv('output/sim_20260102_214500/completions.csv')
throughput = completions.groupby('type')['completed'].sum()
```

### 4. 归档重要实验

```bash
# 移动到experiments/archive/
mkdir -p experiments/archive/theorem2_verification/
cp -r output/sim_20260102_214500/ experiments/archive/theorem2_verification/
cp experiments/exp_theorem2_verification_config.json experiments/archive/theorem2_verification/
```

## 可重复性检查清单

### 实验提交前必须确认

- [ ] config.json包含所有参数（无硬编码）
- [ ] summary.txt说明实验目的和理论依据
- [ ] CSV数据格式标准、精度一致
- [ ] 如使用随机过程，记录了random seed
- [ ] Git commit信息已保存（或手动记录版本）
- [ ] 可视化图表自动生成且保存
- [ ] 实验结果与理论预测对比已记录

### 论文图表专用实验

- [ ] 配置文件在 `experiments/` 中永久保存
- [ ] 实验名称清晰，与论文图表编号对应
- [ ] README中说明: 哪个实验对应论文哪个图/表
- [ ] 结果归档在 `experiments/archive/`

---

**相关文档**:
- [实验工作流](experiment_workflow.md)
- [理论映射](theory_mapping.md)
- [文件组织规范](file_organization.md)
