# 文件组织规范

> 本文档定义项目的标准目录结构和文件组织原则

## 核心原则

- **简洁性优先**: 避免过度嵌套和冗余目录
- **功能导向**: 按功能模块组织，不按技术栈分类
- **模块化设计**: 核心代码、脚本、配置、数据分离
- **版本控制友好**: 结构稳定，方便git跟踪和协作

## 标准目录结构（重构目标）

```
interesting_continuos_batching_dynamics/
├── src/                                     # ⭐ 核心源代码（模块化设计）
│   ├── simulators/                          # 模拟器核心模块
│   │   ├── __init__.py
│   │   ├── multi_type_simulator.py          # 多类型请求模拟器
│   │   ├── admission_control_simulator.py   # 准入控制模拟器
│   │   └── base_simulator.py               # 模拟器基类
│   │
│   ├── analysis/                            # 数学分析模块
│   │   ├── __init__.py
│   │   ├── stability_analysis.py            # 稳定性分析
│   │   ├── phase_space.py                   # 相空间可视化
│   │   └── equilibrium_solver.py            # 平衡点求解
│   │
│   ├── visualization/                       # 可视化模块
│   │   ├── __init__.py
│   │   ├── state_plots.py                   # 状态演化图
│   │   ├── convergence_plots.py             # 收敛性分析图
│   │   └── phase_space_plots.py             # 3D相空间图
│   │
│   ├── metrics/                             # 性能指标计算
│   │   ├── __init__.py
│   │   ├── throughput.py                    # 吞吐量计算
│   │   ├── fairness.py                      # 公平性指标
│   │   └── stability_detector.py            # 稳定性检测
│   │
│   └── utils/                               # 工具函数
│       ├── __init__.py
│       ├── data_export.py                   # CSV导出工具
│       ├── config_loader.py                 # 配置加载器
│       └── math_helpers.py                  # 数学辅助函数
│
├── scripts/                                 # ⭐ 运行脚本（调用src模块）
│   ├── run_simulation.py                    # 主模拟运行脚本
│   ├── run_batch_experiments.py             # 批量实验脚本
│   ├── analyze_results.py                   # 结果分析脚本
│   ├── generate_figures.py                  # 论文图表生成
│   └── parameter_sweep.py                   # 参数扫描实验
│
├── configs/                                 # ⭐ 配置文件
│   ├── default_config.json                  # 默认配置
│   ├── single_type_config.json              # 单类型系统配置
│   ├── multi_type_config.json               # 多类型系统配置
│   └── admission_control_config.json        # 准入控制配置
│
├── experiments/                             # ⭐ 实验配置与归档
│   ├── README.md                            # 实验索引和说明
│   ├── theorem1_greedy_instability/         # Theorem 1验证实验
│   │   ├── config.json                      # 实验配置
│   │   ├── README.md                        # 实验说明
│   │   └── results/                         # 实验结果（可选归档）
│   ├── theorem2_gcd_stability/              # Theorem 2验证实验
│   │   ├── coprime_config.json
│   │   ├── non_coprime_config.json
│   │   └── README.md
│   └── paper_figures/                       # 论文图表实验
│       ├── figure1_config.json
│       ├── figure2_config.json
│       └── README.md
│
├── outputs/                                 # ⭐ 输出结果（git忽略）
│   ├── simulations/                         # 模拟输出
│   │   └── sim_YYYYMMDD_HHMMSS/
│   │       ├── config.json
│   │       ├── summary.txt
│   │       ├── git_info.txt
│   │       ├── data/                        # CSV数据
│   │       │   ├── x_prime_states.csv
│   │       │   ├── admissions.csv
│   │       │   ├── evictions.csv
│   │       │   └── completions.csv
│   │       └── figures/                     # 生成的图表
│   │           ├── state_evolution.png
│   │           └── convergence.png
│   ├── analyses/                            # 分析结果
│   └── figures/                             # 最终图表（论文用）
│
├── logs/                                    # ⭐ 日志文件（git忽略）
│   ├── simulation_YYYYMMDD.log
│   └── experiment_YYYYMMDD.log
│
├── data/                                    # ⭐ 数据存储
│   ├── benchmarks/                          # 基准数据
│   └── validation/                          # 验证数据
│
├── tests/                                   # ⭐ 测试脚本
│   ├── test_simulators.py                   # 模拟器单元测试
│   ├── test_analysis.py                     # 分析模块测试
│   └── test_integration.py                  # 集成测试
│
├── notebooks/                               # ⭐ Jupyter Notebooks
│   ├── exploratory_analysis.ipynb           # 探索性分析
│   ├── result_visualization.ipynb           # 结果可视化
│   └── theorem_verification.ipynb           # 定理验证分析
│
├── demo/                                    # ⭐ 示例代码
│   ├── quick_start.py                       # 快速开始示例
│   ├── basic_simulation.py                  # 基础模拟示例
│   └── advanced_analysis.py                 # 高级分析示例
│
├── docs/                                    # ⭐ 项目文档
│   ├── PROJECT_TODO.md                      # 项目任务追踪
│   ├── guides/                              # 指南文档
│   │   ├── file_organization.md             # 本文档
│   │   ├── coding_standards.md              # 编程规范
│   │   ├── experiment_reproducibility.md    # 实验可重复性
│   │   ├── theory_mapping.md                # 理论映射
│   │   ├── experiment_workflow.md           # 实验工作流
│   │   └── daily_reference.md               # 日常参考
│   ├── modules/                             # 模块文档
│   │   ├── simulators.md
│   │   ├── analysis.md
│   │   └── visualization.md
│   └── experiment_notes/                    # 实验笔记
│       └── YYYYMMDD_experiment_name.md
│
├── tmp/                                     # ⭐ 临时测试文件（git忽略）
│   └── README.md
│
├── deprecated/                              # 🗑️ 废弃代码归档
│   ├── legacy_simulators/
│   └── old_scripts/
│
├── .gitignore                               # Git忽略规则
├── .claude/                                 # Claude Code配置
│   └── CLAUDE.md                            # 主配置文档
├── requirements.txt                         # Python依赖
└── README.md                                # 项目主README
```

## 当前目录结构（过渡期）

```
interesting_continuos_batching_dynamics/
├── new_project_for_multi_type/              # 当前主要代码（待迁移到src/）
│   ├── multi_type_simulator.py              # → src/simulators/multi_type_simulator.py
│   ├── run_simulation.py                    # → scripts/run_simulation.py
│   ├── visualization.py                     # → src/visualization/
│   ├── run_sim.sh                           # → scripts/ (或废弃)
│   └── output/                              # → outputs/simulations/
│
├── simulation_admission_control/            # 准入控制模块（待整合）
│   ├── llm_scheduler_simulator_real.py      # → src/simulators/admission_control_simulator.py
│   ├── experiment_runner.py                 # → scripts/run_batch_experiments.py
│   ├── stability_detector.py                # → src/metrics/stability_detector.py
│   └── config.json                          # → configs/admission_control_config.json
│
├── simultaion_of_the_root/                  # 数学分析模块（待整合）
│   ├── stable_condition.py                  # → src/analysis/stability_analysis.py
│   ├── different_init.py                    # → src/analysis/phase_space.py
│   └── 3d_draw.py                           # → src/visualization/phase_space_plots.py
│
├── experiments/                             # 保持（需完善结构）
├── docs/                                    # 保持（需添加模块文档）
├── tmp/                                     # 保持
│
└── [根目录遗留脚本]                         # → deprecated/ 或删除
    ├── llm_scheduler_simulator.py
    ├── multi_type_simulator_real_overloaded_fix.py
    ├── solution*.py
    └── draw_multi*.py
```

## 目录功能说明

### 核心模块目录（标准结构）

- **`src/`**: 所有核心代码，模块化设计，便于复用和测试
  - `simulators/`: 模拟器实现（多类型、准入控制等）
  - `analysis/`: 数学分析工具（稳定性、相空间、平衡点）
  - `visualization/`: 可视化函数库
  - `metrics/`: 性能指标计算
  - `utils/`: 通用工具函数

- **`scripts/`**: 运行脚本，仅做流程控制和参数传递
  - 调用 `src/` 中的模块
  - 不包含复杂业务逻辑（≤50行）
  - 支持命令行参数

- **`configs/`**: 所有配置文件，支持json格式
  - 默认配置、实验配置
  - 参数化所有硬编码值
  - 版本控制友好

### 数据与输出目录

- **`outputs/`**: 所有输出结果（git忽略）
  - `simulations/`: 模拟运行输出
  - `analyses/`: 分析结果
  - `figures/`: 最终图表（论文用）
  - 结构化存储：时间戳目录 + config + data + figures

- **`logs/`**: 日志文件（git忽略）
  - 按日期和模块组织
  - 便于调试和追踪

- **`data/`**: 数据存储
  - 基准数据、验证数据
  - 需要版本控制的数据

### 开发支持目录

- **`tests/`**: 单元测试和集成测试
  - pytest框架
  - 覆盖核心模块

- **`notebooks/`**: Jupyter分析报告
  - 探索性分析
  - 结果可视化
  - 定理验证

- **`demo/`**: 示例代码和使用说明
  - 快速开始
  - 最佳实践

- **`docs/`**: 项目文档
  - 模块文档
  - 实验笔记
  - 理论映射

### 临时与归档目录

- **`tmp/`**: 临时测试文件（≤1天，git忽略）
- **`deprecated/`**: 废弃代码归档（保留历史）

## 迁移策略

### 阶段1: 基础结构搭建（优先级：P1）

```bash
# 1. 创建标准目录结构
mkdir -p src/{simulators,analysis,visualization,metrics,utils}
mkdir -p scripts configs outputs/{simulations,analyses,figures} logs data tests notebooks demo

# 2. 添加__init__.py文件
touch src/__init__.py
touch src/{simulators,analysis,visualization,metrics,utils}/__init__.py

# 3. 创建基础配置文件
cp new_project_for_multi_type/run_simulation.py configs/default_config.json

# 4. 更新.gitignore
echo "outputs/" >> .gitignore
echo "logs/" >> .gitignore
```

### 阶段2: 核心代码迁移（优先级：P1）

```bash
# 1. 迁移模拟器核心代码
cp new_project_for_multi_type/multi_type_simulator.py \
   src/simulators/multi_type_simulator.py

# 2. 迁移可视化模块
cp new_project_for_multi_type/visualization.py \
   src/visualization/state_plots.py

# 3. 迁移分析工具
cp simultaion_of_the_root/stable_condition.py \
   src/analysis/stability_analysis.py

# 4. 迁移运行脚本
cp new_project_for_multi_type/run_simulation.py \
   scripts/run_simulation.py
```

### 阶段3: 更新导入路径（优先级：P1）

```python
# 旧的导入方式
from multi_type_simulator import MultiTypeLLMSimulator

# 新的导入方式
from src.simulators.multi_type_simulator import MultiTypeLLMSimulator

# 或使用相对导入
from ..simulators.multi_type_simulator import MultiTypeLLMSimulator
```

### 阶段4: 实验配置标准化（优先级：P2）

```bash
# 1. 整理实验配置
mkdir -p experiments/theorem1_greedy_instability
mkdir -p experiments/theorem2_gcd_stability
mkdir -p experiments/paper_figures

# 2. 移动配置文件
# 按实验目的组织，而非散乱在根目录
```

### 阶段5: 归档遗留代码（优先级：P2）

```bash
# 1. 创建deprecated目录
mkdir -p deprecated/legacy_simulators
mkdir -p deprecated/old_scripts

# 2. 移动过时代码
mv llm_scheduler_simulator.py deprecated/legacy_simulators/
mv solution*.py deprecated/old_scripts/
mv draw_multi*.py deprecated/old_scripts/

# 3. 添加README说明废弃原因
```

### 阶段6: 文档完善（优先级：P2）

```bash
# 1. 创建模块文档
mkdir -p docs/modules
touch docs/modules/{simulators.md,analysis.md,visualization.md}

# 2. 创建实验笔记模板
touch docs/experiment_notes/TEMPLATE.md

# 3. 更新主README
# 反映新的目录结构
```

## 渐进式迁移原则

1. **保持兼容性**: 迁移过程中保持旧代码可运行
2. **优先核心模块**: 先迁移常用的核心代码
3. **逐步废弃**: 旧代码移至deprecated，不直接删除
4. **文档先行**: 迁移前更新文档，说明新旧位置对应关系
5. **测试验证**: 迁移后运行测试，确保功能一致

## 检查清单（提交前）

**目录结构**:
- [ ] 新增代码放在正确的 `src/` 子目录
- [ ] 脚本放在 `scripts/`，配置放在 `configs/`
- [ ] 输出结果自动写入 `outputs/`，而非项目根目录
- [ ] 临时文件仅在 `tmp/`，用后即删

**代码组织**:
- [ ] 核心逻辑在 `src/` 模块，脚本仅做调用
- [ ] 配置参数化，无硬编码路径
- [ ] 模块间依赖清晰，避免循环导入

**文档同步**:
- [ ] 新增模块有对应的文档
- [ ] README反映最新目录结构
- [ ] 实验配置有说明文件

---

**相关文档**:
- [编程规范](coding_standards.md)
- [实验可重复性](experiment_reproducibility.md)
- [日常开发参考](daily_reference.md)
