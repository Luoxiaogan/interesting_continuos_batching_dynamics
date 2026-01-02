# 项目 TODO - LLM连续批处理动态模拟系统

## 🔥 P0 - 紧急任务

*当前无紧急任务*

## 🎯 P1 - 重要任务

- [ ] **[实验]**: 完成Theorem 1验证实验（单类型Greedy不稳定性）
  - 理论依据: `/Users/ruicheng/Library/.../LLM_serving/single_discrete.tex` (lines 88-91)
  - 配置文件: `experiments/exp_theorem1_greedy_instability.json` (待创建)
  - 预期结果: 系统收敛到level-(l1-1) limit cycle
  - 截止: 2026-01-15

- [ ] **[实验]**: 完成Theorem 2验证实验（GCD稳定性条件）
  - 理论依据: `multiple_discrete.tex` (lines 120-122)
  - 配置文件:
    - `experiments/exp_theorem2_gcd_1_coprime.json` (待创建)
    - `experiments/exp_theorem2_gcd_gt_1_non_coprime.json` (待创建)
  - 预期结果: gcd=1收敛，gcd>1不收敛
  - 截止: 2026-01-15

## 📋 P2 - 常规任务

- [ ] **[代码清理]**: 整理根目录遗留脚本
  - 识别仍然有用的脚本: `draw_multi.py`, `solution*.py`
  - 迁移有效功能到对应目录
  - 归档或删除过时代码
  - 参考: `.claude/CLAUDE.md` 文件组织规范

- [ ] **[文档]**: 创建实验笔记模板
  - 位置: `docs/experiment_notes/TEMPLATE.md`
  - 内容: 实验目的、理论依据、参数设计、预期结果、实际结果

- [ ] **[实验配置]**: 创建常用实验配置文件
  - 单类型limit cycle: `experiments/exp_single_type_limit_cycle.json`
  - GCD稳定性对比: `experiments/exp_gcd_comparison.json`
  - 准入控制阈值扫描: `experiments/exp_admission_threshold_sweep.json`

- [ ] **[测试]**: 添加基础验证测试
  - 位置: `tests/` (新建目录)
  - 内容: 配置验证、内存约束检查、吞吐量计算验证

## ✅ 已完成 (本周 2026-01-02)

- [x] **[基础设施]**: 创建标准目录结构 (完成时间: 2026-01-02)
  - 创建: `.claude/`, `docs/`, `experiments/`, `tmp/`

- [x] **[文档]**: 完成 `.claude/CLAUDE.md` 编写 (完成时间: 2026-01-02)
  - 包含: 项目概述、编程规范、实验可重复性、理论联系、日常开发指南
  - 参考: Vidur_toymodel和LLM_serving的CLAUDE.md

- [x] **[文档]**: 创建 `docs/PROJECT_TODO.md` (完成时间: 2026-01-02)

- [x] **[文档]**: 创建 `experiments/README.md` (完成时间: 2026-01-02)

- [x] **[文档]**: 创建 `tmp/README.md` (完成时间: 2026-01-02)

- [x] **[配置]**: 更新 `.gitignore` (完成时间: 2026-01-02)
  - 添加: `tmp/`, `.claude/plans/`, `experiments/output/`

---

## 📝 使用说明

### 优先级定义

- **P0 (紧急)**: 阻塞开发或严重bug，立即处理（通常24小时内）
- **P1 (重要)**: 论文截止前的关键实验、重要功能，本周内完成
- **P2 (常规)**: 代码清理、文档补充、优化改进，本月内完成

### 更新规范

1. **新增任务**: 直接添加到对应优先级部分
2. **完成任务**:
   - 勾选复选框 `- [x]`
   - 移动到"已完成"部分
   - 添加完成时间和commit hash（如适用）
3. **调整优先级**: 根据实际情况上调或下调任务优先级

### 每周维护

建议每周一查看并更新此文件：
- 回顾已完成任务，归档到底部
- 调整任务优先级
- 添加新发现的待办事项
- 删除已过时或不再需要的任务

---

**文档版本**: v1.0
**最后更新**: 2026-01-02
**维护者**: @ruicheng
