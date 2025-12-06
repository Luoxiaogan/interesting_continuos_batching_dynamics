# 快速参考

## 一分钟上手

```bash
# 运行模拟
./run_sim.sh

# 查看结果
cat output/sim_*/summary.txt
```

## 修改参数（编辑 run_sim.sh）

```bash
# 请求类型
L0_0=2      # Type 0 初始长度
L1_0=5      # Type 0 生成长度
L0_1=5      # Type 1 初始长度
L1_1=2      # Type 1 生成长度

# 到达率
LAMBDA_0=8.0
LAMBDA_1=4.0

# GPU容量
B=50

# 模拟步数
STEPS=100

# 初始状态（可选）
X0='{}'  # 空状态
# 或
X0='{"2": [5.0, 0.0], "5": [0.5, 2.0]}'  # 指定状态

# 可视化配置
START_INDEX=0   # 从第几个批次开始画图
JUMP=1          # 差异计算步长
```

## 输出文件

| 文件 | 内容 | 用途 |
|------|------|------|
| `x_prime_states.csv` | 每批次状态 | **核心数据**，用于所有分析 |
| `admissions.csv` | 准入记录 | 分析admission策略 |
| `evictions.csv` | 驱逐记录 | 分析eviction行为 |
| `completions.csv` | 完成记录 | 计算吞吐量 |
| `batch_info.csv` | 批次元信息 | 时间、服务时间分析 |
| `config.json` | 配置参数 | 实验追溯 |
| `summary.txt` | 文本总结 | 快速查看结果 |
| `state_evolution_from_X.png` | **状态演变图** | 可视化各type各stage演变 |
| `state_differences_from_X_jump_Y.png` | **状态差异图-按type** | 分析收敛性（对数坐标） |
| `length_total_differences_from_X_jump_Y.png` | **长度总量差异图** | 不区分type的收敛性分析 |

## 数据格式示例

### x_prime_states.csv（最重要）
```csv
batch,time,length,type,count
0,0.6,2,0,4.35
0,0.6,5,1,1.74
...
```

- `batch`: 批次号
- `time`: 模拟时间
- `length`: 请求当前长度
- `type`: 请求类型（0或1）
- `count`: 请求数量

## 常见场景

### 场景1：测试不同GPU容量

修改 `run_sim.sh`：
```bash
B=30   # 小容量
# B=60   # 中容量
# B=100  # 大容量
```

### 场景2：不对称请求类型

```bash
# 长prompt，短生成
L0_0=10
L1_0=2
LAMBDA_0=8.0

# 短prompt，长生成
L0_1=2
L1_1=10
LAMBDA_1=4.0
```

### 场景3：从特定状态开始

```bash
X0='{
  "5": [3.0, 2.0],   # 长度5：Type0有3个，Type1有2个
  "6": [1.0, 1.5]    # 长度6：Type0有1个，Type1有1.5个
}'
```

## 分析数据（Python）

```python
import pandas as pd

# 读取核心数据
df = pd.read_csv("output/sim_XXXXXX/x_prime_states.csv")

# 查看Type 0在长度3的演变
type0_len3 = df[(df['type']==0) & (df['length']==3)]
print(type0_len3[['batch', 'time', 'count']])

# 计算平均状态
avg_state = df.groupby(['type', 'length'])['count'].mean()
print(avg_state)

# 稳态分析（最后20批次）
steady = df[df['batch'] >= 80].groupby(['type', 'length'])['count'].mean()
print(steady)
```

## 可视化图表

### 自动生成

运行 `./run_sim.sh` 会自动生成三类图表：

1. **状态演变图** - 显示每个type每个stage的请求数随批次变化
2. **状态差异图（按type）** - 显示相邻批次差异，按type分类分析收敛性
3. **长度总量差异图** - 显示每个长度总请求数（不区分type）的相邻批次差异

### 独立生成

```bash
# 对已有结果生成不同参数的图表
python3 visualization.py output/sim_XXXXXX --start_index 50 --jump 10
```

### 控制可视化参数

编辑 `run_sim.sh`：

```bash
START_INDEX=20  # 从批次20开始画图（跳过前20批次）
JUMP=5          # 每5个批次计算一次差异
# NO_PLOT="--no_plot"  # 取消注释以禁用自动生成图表
```

## 故障排查

```bash
# 权限问题
chmod +x run_sim.sh

# 依赖问题
pip install pandas

# 检查输出
ls -lh output/sim_*/

# 验证数据
head output/sim_*/x_prime_states.csv
```

## 更多信息

- 详细使用指南：`USAGE.md`
- 技术文档：`README.md`
- 代码注释：`multi_type_simulator.py`
