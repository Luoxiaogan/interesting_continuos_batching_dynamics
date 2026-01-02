# 核心编程规范

> 严格执行的代码质量标准和最佳实践

## 八荣八耻编程基本原则

1. **以暗猜接口为耻，以认真查阅为荣** - 禁止臆测API行为，必须查阅文档和代码确认
2. **以模糊执行为耻，以寻求确认为荣** - 不确定的实现必须先向用户确认，避免模糊操作
3. **以默认忽略为耻，以主动报告为荣** - 遇到异常、警告、错误必须主动报告，不得静默忽略
4. **以隐式假设为耻，以显式验证为荣** - 所有假设必须通过代码验证，禁止隐式依赖
5. **以随意修改为耻，以谨慎调试为荣** - 修改前必须理解原理，禁止试错式编程
6. **以表面应付为耻，以深入理解为荣** - 解决问题必须找到根本原因，禁止表面修补
7. **以复制粘贴为耻，以原创思考为荣** - 理解每行代码含义，禁止盲目复制
8. **以孤立开发为耻，以协同沟通为荣** - 主动汇报进度和问题，寻求指导和反馈

## 文件命名规范 - 严格禁止

### 禁用前缀后缀列表

- ❌ `enhanced_*` / `*_enhanced` - 禁止enhanced前缀后缀
- ❌ `integrated_*` / `*_integrated` - 禁止integrated前缀后缀
- ❌ `cleaned_*` / `*_cleaned` / `*_clean` - 禁止clean相关命名
- ❌ `improved_*` / `*_improved` - 禁止improved前缀后缀
- ❌ `optimized_*` / `*_optimized` - 禁止optimized前缀后缀
- ❌ `advanced_*` / `*_advanced` - 禁止advanced前缀后缀
- ❌ `*_v2` / `*_new` / `*_old` / `*_temp` - 禁止版本和临时标识符

### 正确命名原则

- ✅ **功能导向命名** - 直接描述文件功能：`multi_type_simulator.py`、`experiment_runner.py`
- ✅ **模块化命名** - 按模块组织：`visualization.py`、`stability_detector.py`
- ✅ **简洁明确** - 避免冗余形容词，直接表达核心功能
- ✅ **统一风格** - 使用下划线分隔，全小写字母

### 命名示例对比

```bash
# ❌ 错误命名
enhanced_multi_type_simulator.py  →  # ✅ multi_type_simulator.py
experiment_runner_integrated.py    →  # ✅ experiment_runner.py
visualize_clean.py                 →  # ✅ visualization.py
stable_condition_v2.py             →  # ✅ stable_condition.py
```

## 错误处理强制规范

```python
# ❌ 严格禁止的fallback模式
try:
    result = complex_operation()
except Exception:
    result = fallback_operation()  # 禁止！

# ❌ 严格禁止的属性检查fallback
if hasattr(obj, 'attribute'):
    return obj.attribute
else:
    return default_value  # 禁止！

# ✅ 正确的错误处理方式
result = complex_operation()  # 让错误自然抛出
required_attribute = obj.attribute  # 直接访问，缺失时报错
```

### 核心要求

- 🔥 **禁止使用try except** - 碰见错误直接显示traceback并退出终止运行程序
- 🔥 **禁止采用fallback方案** - 如缺少属性直接报错返回，不允许降级处理
- ✅ **让错误自然抛出** - 便于从本质上解决问题，而非掩盖问题

## 模块文档引用规范 - 强制执行

### 核心原则

每个 Python 模块必须在文件开头明确指向对应的文档

### 模块头部模板

```python
"""
[模块简短描述 - 一句话说明模块功能]

Documentation:
    Interface: docs/modules/[module_name].md
    Theoretical Basis: /path/to/Overleaf/LLM_serving/[related_tex_file].tex
    Related Paper Section: [Section X.X - Theorem/Lemma name]

Key Features:
    - Feature 1: Brief description
    - Feature 2: Brief description
    - Feature 3: Brief description

Dependencies:
    - numpy: Numerical computations
    - module_a: Purpose

Mathematical Correspondence:
    - State X_n: Corresponds to self.state in code
    - Arrival rate λ: self.arrival_rates parameter
    - Memory capacity B: self.B parameter

Example:
    >>> from multi_type_simulator import MultiTypeLLMSimulator
    >>> sim = MultiTypeLLMSimulator(request_type_list=[(2,5), (5,2)], B=50, ...)
    >>> sim.run(100)
"""

import numpy as np
# ... rest of imports
```

### 针对本项目的特定要求

#### 1. 模拟器模块 (如 `multi_type_simulator.py`)

```python
"""
Multi-Type LLM Request Scheduling Simulator with Decode Priority

Documentation:
    Implementation: new_project_for_multi_type/README.md
    Theoretical Basis: /Users/ruicheng/Library/.../LLM_serving/multiple_discrete.tex
    Related Theorem: Theorem 2 (GCD Stability Condition)

Key Features:
    - Decode priority scheduling (priority = current_length - l0)
    - n-proportional eviction by type
    - Overloaded system assumption (Z = B)
    - Complete history recording

Mathematical Correspondence:
    - X[length][type]: State variable from multiple_discrete.tex
    - l0, l1: Prompt and decoding lengths
    - B: Memory capacity
    - λ (lambda): Arrival rates
"""
```

#### 2. 实验运行器 (如 `experiment_runner.py`)

```python
"""
Batch Experiment Runner for Admission Control Policy Testing

Documentation:
    Implementation: simulation_admission_control/README.md (if exists)
    Theoretical Basis: admission_control.tex
    Related Policy: Rate-limited admission control

Key Features:
    - Multiprocessing for parallel experiments
    - Parameter sweeping (admission threshold range)
    - Stability detection
    - Automatic result aggregation
"""
```

#### 3. 数学分析工具 (如 `stable_condition.py`)

```python
"""
Stability Manifold Analysis and Phase Space Visualization

Documentation:
    Theoretical Basis: single_discrete.tex (lines 52-140)
    Related: Eigenvalue analysis of Poincaré map P^(i)

Key Features:
    - Root finding for characteristic polynomial
    - Eigen-decomposition for stable manifold
    - 3D phase space trajectory plotting

Mathematical Correspondence:
    - Matrix B_1, B_2^(i): No-eviction and eviction operators
    - P^(i): Full cycle Poincaré map
    - Eigenvalues: Stability analysis
"""
```

### 检查清单（提交代码前）

- [ ] 模块开头包含 docstring
- [ ] Documentation 部分列出相关文档和理论文件
- [ ] 如涉及理论验证，明确指出对应的tex文件和定理
- [ ] Key Features 列出核心功能（3-5条）
- [ ] Mathematical Correspondence 列出代码变量与数学符号的对应
- [ ] Example 提供基本用法示例

## 脚本组织和模块化规范

### 脚本复杂度控制

- ✅ **简单脚本**: 直接在scripts/中实现，最多50行
- ✅ **复杂逻辑**: 必须分离到独立模块中，脚本仅做调用
- ❌ **禁止内嵌**: 严禁在脚本中写大段Python代码或函数
- ❌ **禁止重复**: 相同逻辑不得在多个脚本中重复实现

### 模块化分离原则

```bash
# ❌ 错误做法：在脚本中内嵌复杂逻辑
run_experiment.sh:
    python -c "
    import complex_logic
    # 50行复杂代码...
    "

# ✅ 正确做法：分离到模块
multi_type_simulator.py:        # 复杂逻辑在独立模块
    class MultiTypeLLMSimulator: ...

run_simulation.py:              # Python脚本调用模块
    from multi_type_simulator import MultiTypeLLMSimulator
    sim = MultiTypeLLMSimulator(...)
    sim.run(steps)

run_sim.sh:                     # Bash脚本仅传参
    python run_simulation.py --B $B --steps $STEPS
```

### 脚本职责边界

- **scripts/** (如 `run_sim.sh`): 参数传递、流程控制、状态检查
- **模块** (如 `multi_type_simulator.py`): 核心算法、数据处理、复杂逻辑
- **configs/**: 参数配置、超参数设定

## 临时文件管理规范

### 核心原则

临时测试文件必须在 `tmp/` 目录，使用后立即删除

### 适用场景

- ✅ 快速验证某个功能
- ✅ 调试代码片段
- ✅ 临时性能测试
- ✅ 一次性数据探索

### 禁止场景

- ❌ 正式的单元测试 → 应放在 `tests/`
- ❌ 示例代码 → 应放在 `demo/` 或文档中
- ❌ 实验脚本 → 应放在 `experiments/`

### 使用规范

```bash
# 1. 创建临时测试文件
tmp/test_stability.py
tmp/debug_eviction.py

# 2. 测试完成后立即删除
rm tmp/test_stability.py

# 3. git已配置忽略tmp/目录
```

### 强制要求

- 🚫 **禁止提交到git**: tmp/目录必须在.gitignore中
- 🚫 **禁止长期保留**: 文件生命周期 ≤ 1天
- 🚫 **禁止依赖关系**: 任何正式代码不得import tmp/中的文件
- ✅ **鼓励即测即删**: 测试完成后立即清理

---

**相关文档**:
- [文件组织规范](file_organization.md)
- [实验可重复性](experiment_reproducibility.md)
- [日常开发参考](daily_reference.md)
