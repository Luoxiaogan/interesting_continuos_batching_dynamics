# 使用指南

## 快速开始

### 1. 运行模拟

```bash
# 添加执行权限（首次使用）
chmod +x run_sim.sh

# 运行模拟
./run_sim.sh
```

### 2. 修改参数

编辑 `run_sim.sh` 文件中的参数：

```bash
# ===== 请求类型配置 =====
L0_0=2      # Type 0的初始prompt长度
L1_0=5      # Type 0需要生成的token数
L0_1=5      # Type 1的初始prompt长度
L1_1=2      # Type 1需要生成的token数

# ===== 到达率配置 =====
LAMBDA_0=8.0    # Type 0的到达率
LAMBDA_1=4.0    # Type 1的到达率

# ===== 系统参数 =====
B=50            # GPU容量限制

# ===== 模拟参数 =====
STEPS=100       # 模拟步数

# ===== 初始状态配置 =====
X0='{
  "2": [5.0, 0.0],
  "5": [0.5, 2.0]
}'

# ===== 可视化配置 =====
START_INDEX=0       # 可视化开始的批次索引（默认0）
JUMP=1             # 差异计算的步长（默认1）
NO_PLOT=""         # 设置为"--no_plot"禁用自动生成图表

# ===== 精度配置 =====
PRECISION=10       # 数值精度（小数位数，默认10）
```

### 3. 输出结构

每次运行会在 `output/` 目录下创建一个带时间戳的文件夹：

```
output/
└── sim_20251206_192540/
    ├── config.json                              # 模拟配置
    ├── x_prime_states.csv                       # 核心状态数据（每批次、每长度、每类型）
    ├── admissions.csv                           # 准入记录
    ├── evictions.csv                            # 驱逐记录
    ├── completions.csv                          # 完成记录
    ├── batch_info.csv                           # 批次元信息（时间、服务时间、批次大小）
    ├── summary.txt                                      # 文本总结
    ├── state_evolution_from_0.png                       # 状态演变图（自动生成）
    ├── state_differences_from_0_jump_1.png              # 状态差异图-按type（自动生成）
    └── length_total_differences_from_0_jump_1.png       # 长度总量差异图（自动生成）
```

## 数据格式说明

### 1. x_prime_states.csv（核心数据）

这是最重要的文件，包含每个批次After Admission/Eviction的状态：

```csv
batch,time,length,type,count
0,0.6,2,0,4.35
0,0.6,2,1,0.0
0,0.6,5,0,0.5
0,0.6,5,1,1.74
...
```

**字段说明：**
- `batch`: 批次号（0, 1, 2, ...）
- `time`: 当前模拟时间
- `length`: 请求的当前长度
- `type`: 请求类型（0或1）
- `count`: 该状态下的请求数量

### 2. admissions.csv

记录每个批次新准入的请求：

```csv
batch,time,type,admitted_count
0,0.6,0,0.0
1,1.2,0,0.15
...
```

### 3. evictions.csv

记录每个批次被驱逐的请求：

```csv
batch,time,type,length,evicted_count
0,0.6,0,2,0.65
3,2.4,1,6,0.07
...
```

**字段说明：**
- `length`: 被驱逐时的长度（高优先级的请求先保留，低优先级的被驱逐）

### 4. completions.csv

记录每个批次完成的请求：

```csv
batch,time,type,completed_count
0,0.6,0,0.5
0,0.6,1,1.0
...
```

### 5. batch_info.csv

记录每个批次的元信息：

```csv
batch,time,service_time,batch_size
0,0.6,0.6,50.0
1,1.2,0.6,50.0
...
```

**字段说明：**
- `service_time`: 该批次的服务时间 s(n)
- `batch_size`: 该批次的大小 Z(n)（在overloaded情况下应该等于B）

## 初始状态配置说明

### 从空状态开始

```bash
X0='{}'
```

### 从指定状态开始

```bash
X0='{
  "2": [5.0, 0.0],   # 长度2: Type 0有5个请求，Type 1有0个
  "3": [2.0, 0.0],   # 长度3: Type 0有2个请求
  "5": [0.5, 2.0],   # 长度5: Type 0有0.5个，Type 1有2个
  "6": [0.5, 1.0]    # 长度6: Type 0有0.5个，Type 1有1个
}'
```

**注意：**
- 长度必须在对应type的有效范围内
- Type 0的有效长度：[l0_0, l0_0+l1_0-1]
- Type 1的有效长度：[l0_1, l0_1+l1_1-1]

## 常见配置示例

### 示例1：长prompt vs 短prompt

```bash
# 长prompt, 短生成
L0_0=10
L1_0=2
LAMBDA_0=6.0

# 短prompt, 长生成
L0_1=2
L1_1=10
LAMBDA_1=6.0

B=100
STEPS=200
X0='{}'
```

### 示例2：不同到达率

```bash
# 高频请求
L0_0=5
L1_0=5
LAMBDA_0=10.0

# 低频请求
L0_1=5
L1_1=5
LAMBDA_1=2.0

B=80
STEPS=500
X0='{}'
```

### 示例3：测试不同GPU容量

```bash
L0_0=3
L1_0=4
L0_1=4
L1_1=3
LAMBDA_0=8.0
LAMBDA_1=4.0

# 小容量
B=30
# 中容量
# B=60
# 大容量
# B=100

STEPS=300
X0='{}'
```

## 可视化功能

### 自动生成图表

运行模拟时会自动生成三类图表：

#### 1. 状态演变图 (state_evolution_from_X.png)

显示每个type的每个stage（length）的请求数随批次的变化，包含：
- **总览图**：所有类型和长度的演变
- **Type 0单独图**：仅Type 0的各长度演变
- **Type 1单独图**：仅Type 1的各长度演变
- **堆叠面积图**：显示总体请求分布

#### 2. 状态差异图 - 按type分类 (state_differences_from_X_jump_Y.png)

显示相邻批次（或jump步）状态差的绝对值，用于分析收敛性（按type分类）：
- **Type 0差异图**：对数坐标显示Type 0各长度的变化幅度
- **Type 1差异图**：对数坐标显示Type 1各长度的变化幅度
- **总体差异图**：所有类型和长度的差异叠加

#### 3. 长度总量差异图 - 不区分type (length_total_differences_from_X_jump_Y.png)

显示每个长度的总请求数（所有type求和）的相邻step差的绝对值：
- **各长度差异图**：对数坐标显示每个长度的总请求数变化幅度
- **平均差异图**：显示所有长度的平均差异和最大差异
- 用于分析整体系统的收敛性，关注长度分布而非类型分布

### 可视化参数配置

在 `run_sim.sh` 中配置：

```bash
# 从批次20开始绘图（跳过前20个批次）
START_INDEX=20

# 计算每5个批次的差异（而不是相邻批次）
JUMP=5

# 禁用自动生成图表
# NO_PLOT="--no_plot"
```

### 独立运行可视化

如果已经有模拟结果，可以单独运行可视化：

```bash
# 基本用法
python3 visualization.py output/sim_20251206_192540

# 指定start_index和jump
python3 visualization.py output/sim_20251206_192540 --start_index 50 --jump 10
```

这样可以生成不同参数的多个图表版本，无需重新运行模拟。

## 精度配置

### 设置数值精度

在 `run_sim.sh` 中配置：

```bash
# 低精度（6位小数）- 适合快速查看
PRECISION=6

# 中等精度（10位小数）- 默认值，适合大多数情况
PRECISION=10

# 高精度（15位小数）- 适合需要高精度计算的场景
PRECISION=15
```

### 精度对比示例

| 精度 | 示例值1 | 示例值2 | CSV文件大小 |
|------|---------|---------|------------|
| 6位  | 5.555556 | 0.888889 | 较小 |
| 10位 | 5.5555555556 | 0.8888888889 | 中等（默认） |
| 15位 | 5.555555555555555 | 0.888888888888889 | 较大 |

### 精度应用范围

精度配置影响所有CSV输出文件的浮点数：
- `x_prime_states.csv`: 状态值（count字段）
- `admissions.csv`: 准入数量
- `evictions.csv`: 驱逐数量
- `completions.csv`: 完成数量
- `batch_info.csv`: 时间和批次大小

**注意：**
- 精度仅影响CSV输出格式，不影响内部计算精度
- 内部计算始终使用Python的float（双精度浮点数）
- 更高的精度会增加CSV文件大小

### 使用pandas进行自定义分析

```python
import pandas as pd

# 读取数据
df = pd.read_csv("output/sim_20251206_192540/x_prime_states.csv")

# 按类型和批次聚合
type_totals = df.groupby(['batch', 'type'])['count'].sum().reset_index()

# 计算平均状态
avg_by_length = df.groupby(['length', 'type'])['count'].mean().reset_index()

# 分析稳态（最后20个批次）
steady_state = df[df['batch'] >= 80].groupby(['length', 'type'])['count'].mean()
```

## 启用详细日志

修改 `run_sim.sh`：

```bash
VERBOSE="--verbose"  # 启用详细日志
```

这会打印每个批次的详细状态信息，有助于调试和理解模拟过程。

## 批量运行实验

创建 `batch_run.sh`：

```bash
#!/bin/bash

# 测试不同的GPU容量
for B in 30 50 70 100; do
    echo "Running with B=$B"
    # 修改run_sim.sh中的B值或直接调用Python脚本
    python3 run_simulation.py \
        --l0_0 2 --l1_0 5 \
        --l0_1 5 --l1_1 2 \
        --lambda_0 8.0 --lambda_1 4.0 \
        --B $B \
        --steps 200 \
        --x0 '{}' \
        --output_base "output_batch"
done
```

## 故障排查

### 问题1：权限错误

```bash
chmod +x run_sim.sh
```

### 问题2：找不到Python模块

确保pandas已安装：
```bash
pip install pandas
```

### 问题3：JSON格式错误

确保X0的JSON格式正确，使用单引号包围整个JSON字符串。

### 问题4：数据验证

检查batch_size是否等于B：
```bash
# 应该输出所有行都是True
awk -F',' 'NR>1 {print ($4==50)}' output/sim_*/batch_info.csv | sort -u
```

## 下一步

1. **分析稳态行为**：查看最后N个批次的状态分布
2. **公平性分析**：比较实际吞吐量与期望吞吐量
3. **可视化演变**：绘制状态随时间的变化
4. **参数敏感性**：测试不同参数组合的影响
5. **策略对比**：实现不同的调度策略并比较性能
