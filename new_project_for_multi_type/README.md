# Multi-Type LLM Simulator - 多类型LLM调度模拟器

## 概述

这是一个完整的多类型LLM请求调度模拟系统，包括：
- **核心模拟器**：实现decode优先级调度逻辑
- **运行脚本**：便捷的参数配置和批量运行
- **数据持久化**：完整的CSV数据输出
- **可扩展架构**：易于添加可视化和分析功能

## 项目结构

```
new_project_for_multi_type/
├── multi_type_simulator.py    # 核心模拟器类
├── run_simulation.py           # 模拟运行脚本（数据保存）
├── run_sim.sh                  # Bash启动脚本（参数配置）
├── visualization.py            # 可视化函数库（绘图）
├── README.md                   # 技术文档
├── USAGE.md                    # 使用指南
├── QUICKSTART.md               # 快速参考
└── output/                     # 输出目录（自动创建）
    └── sim_YYYYMMDD_HHMMSS/   # 带时间戳的结果目录
        ├── config.json
        ├── x_prime_states.csv
        ├── admissions.csv
        ├── evictions.csv
        ├── completions.csv
        ├── batch_info.csv
        ├── summary.txt
        ├── state_evolution_from_0.png                      # 状态演变图
        ├── state_differences_from_0_jump_1.png             # 状态差异图（按type）
        └── length_total_differences_from_0_jump_1.png      # 长度总量差异图（不区分type）
```

## 快速开始

```bash
# 1. 添加执行权限（首次使用）
chmod +x run_sim.sh

# 2. 运行模拟（使用默认参数）
./run_sim.sh

# 3. 查看结果
ls output/sim_*/
cat output/sim_*/summary.txt
```

详细使用说明请查看 [USAGE.md](USAGE.md)。

## 核心功能

### 1. 模拟逻辑
- **Decode优先级调度**：按照decode次数（当前长度 - l0）进行优先级排序
- **Admission/Eviction策略**：
  - 高decode次数的请求优先保留
  - 同一decode层级的多类型请求按n-proportional比例驱逐
  - 剩余容量按到达率比例准入新请求
- **Overloaded系统假设**：GPU容量始终被充分利用（Z = B）

### 2. 状态记录与精度控制

模拟器记录每个批次的完整状态信息，存储在CSV文件中，便于后续分析。

**精度配置：**
- 支持可配置的数值精度（小数位数）
- 默认精度：10位小数
- 可在`run_sim.sh`中设置`PRECISION`参数（如6、10、15等）
- 所有CSV文件的浮点数都按配置的精度保存

**精度示例：**
```
PRECISION=6  → 5.555556, 2.777778
PRECISION=10 → 5.5555555556, 2.7777777778
PRECISION=15 → 5.555555555555555, 2.777777777777778
```

### 3. 可视化功能

**自动生成的图表：**

#### 状态演变图 (state_evolution_from_X.png)
- 记录每个type的每个stage（length）的请求数随batch count的变化
- 包含：总览图、各type单独图、堆叠面积图
- 支持`start_index`参数：从指定批次开始绘图（默认0）

#### 状态差异图 (state_differences_from_X_jump_Y.png)
- 记录相邻位置（或jump步）差的绝对值，用于分析收敛性（按type分类）
- 使用对数坐标显示差异变化
- 支持`jump`参数：计算每隔几个批次的差异（默认1）

#### 长度总量差异图 (length_total_differences_from_X_jump_Y.png)
- 记录每个长度的总请求数（所有type求和）的相邻step差的绝对值
- 包含两个子图：
  - 每个长度的差异曲线（对数坐标）
  - 所有长度的平均差异和最大差异（对数坐标）
- 用于分析整体系统的收敛性，不区分请求类型

**独立运行可视化：**
```bash
python3 visualization.py output/sim_XXXXXX --start_index 20 --jump 5
```

### 4. 历史数据记录

模拟器记录每个批次的完整状态信息，存储在`history`字典中：

```python
history = {
    'X_prime': [        # After Admission/Eviction状态
        {
            'batch': int,           # 批次号
            'time': float,          # 当前时间T
            'state': {              # 状态矩阵
                length: [type0_count, type1_count, ...]
            }
        },
        ...
    ],
    'admissions': [     # 每批次的新准入
        {
            'batch': int,
            'time': float,
            'admissions': {type_idx: count}
        },
        ...
    ],
    'evictions': [      # 每批次的驱逐
        {
            'batch': int,
            'time': float,
            'evictions': {
                type_idx: [(length, amount), ...]
            }
        },
        ...
    ],
    'completions': [    # 每批次的完成数
        {
            'batch': int,
            'time': float,
            'completions': [type0_count, type1_count, ...]
        },
        ...
    ],
    'batch_info': [     # 批次元信息
        {
            'batch': int,
            'time': float,
            'service_time': float,
            'batch_size': float
        },
        ...
    ]
}
```

## 使用方法

### 基本使用

```python
from multi_type_simulator import MultiTypeLLMSimulator

# 定义请求类型
request_types = [
    (2, 5),  # Type 0: l0=2, l1=5
    (5, 2),  # Type 1: l0=5, l1=2
]

# 初始状态
X0 = {
    2: [5.0, 0.0],
    5: [0.5, 2.0],
}

# 创建模拟器
sim = MultiTypeLLMSimulator(
    request_type_list=request_types,
    B=50,                           # GPU容量
    X0=X0,                          # 初始状态
    arrival_rates=[8.0, 4.0],       # 到达率
    b0=0.1,                         # 服务时间基础
    b1=0.01,                        # 服务时间系数
    verbose=True                    # 是否打印详细日志
)

# 运行模拟
sim.run(100)

# 获取历史数据
history = sim.get_history()

# 获取特定批次的状态
state_at_batch_10 = sim.get_state_at_batch(10)
```

### 主要参数说明

- `request_type_list`: 请求类型列表，每个元素为 `(l0, l1)` 元组
  - `l0`: 初始prompt长度
  - `l1`: 需要生成的token数
  - 请求的有效长度范围：`[l0, l0+l1-1]`
  - 完成长度：`l0+l1`

- `B`: GPU容量限制（token数）

- `X0`: 初始状态字典
  - 键：长度（length）
  - 值：各类型在该长度的请求数列表

- `arrival_rates`: 各类型的到达率 λ

- `b0, b1`: 服务时间参数，`s(n) = b0 + b1 * Z(n)`

- `verbose`: 是否打印详细的模拟过程（默认False）

## 设计特点

### 1. 关注点分离
- **只负责模拟**：不包含任何可视化代码
- **纯数据输出**：通过`history`提供结构化数据
- **易于扩展**：可视化工具可以独立开发

### 2. 完整的状态记录
- 记录每个批次的`X_prime`状态（After Admission/Eviction）
- 记录admission和eviction的详细信息
- 记录每个批次的完成情况
- 所有后续分析和可视化都基于这些记录

### 3. 灵活的日志控制
- `verbose=True`: 详细打印每个批次的状态变化
- `verbose=False`: 只在最后打印汇总信息
- 适合不同的使用场景（调试 vs 大规模模拟）

## 与原始文件的对比

### 从 `multi_type_simulator_real_overloaded_fix_backup.py` 继承：
- ✅ 完整的decode优先级调度逻辑
- ✅ Admission/Eviction算法
- ✅ n-proportional驱逐策略
- ✅ 容量约束验证

### 从 `draw_multi.py` 继承：
- ✅ 完整的历史状态记录（`X_prime_history`）
- ✅ Admission/Eviction记录
- ✅ 结构化的数据格式

### 新增改进：
- ✅ 更清晰的代码结构和注释
- ✅ 独立的模拟器类（不依赖可视化）
- ✅ 灵活的日志控制（verbose参数）
- ✅ 完整的文档字符串
- ✅ 更完善的历史数据记录（增加了`batch_info`）

## 数据接口

### 获取历史数据
```python
history = sim.get_history()

# 访问X_prime状态
for record in history['X_prime']:
    batch = record['batch']
    time = record['time']
    state = record['state']  # {length: [type0, type1, ...]}
```

### 获取特定批次状态
```python
# 获取第10个批次的状态
state = sim.get_state_at_batch(10)
if state:
    print(f"Length 5, Type 0: {state[5][0]}")
```

## 后续使用

这个模拟器生成的历史数据可以用于：
1. **可视化分析**：状态演变图、差异图、堆叠图等
2. **稳态分析**：收敛性、公平性、吞吐量分析
3. **策略对比**：不同调度策略的性能对比
4. **参数调优**：找到最优的B、到达率等参数

所有可视化和分析代码应该独立于这个模拟器，只需要读取`history`数据即可。

## 示例输出

运行 `python multi_type_simulator.py` 可以看到两个示例：
1. 从非空初始状态开始的两类型模拟（verbose模式）
2. 从空状态开始的三类型模拟（非verbose模式）
