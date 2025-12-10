#!/bin/bash

################################################################################
# 多类型LLM调度模拟器运行脚本
#
# 使用方法:
#   chmod +x run_sim.sh  # 首次使用需要添加执行权限
#   ./run_sim.sh
################################################################################

# ===== 请求类型配置 =====
# Type 0: (l0_0, l1_0)
L0_0=6     # Type 0的初始prompt长度
L1_0=2      # Type 0需要生成的token数

# Type 1: (l0_1, l1_1)
L0_1=6      # Type 1的初始prompt长度
L1_1=7      # Type 1需要生成的token数


# ===== 到达率配置 =====
LAMBDA_0=8.0    # Type 0的到达率
LAMBDA_1=4.0    # Type 1的到达率


# ===== 系统参数 =====
B=50            # GPU容量限制（token数）
B0=0.1          # 服务时间基础参数
B1=0.01         # 服务时间系数参数（s(n) = b0 + b1 * Z(n)）


# ===== 模拟参数 =====
STEPS=10000       # 模拟步数


# ===== 初始状态配置 =====
# JSON格式: {"length": [type0_count, type1_count]}
# 示例1: 从非空状态开始
X0='{
  "6": [2.0, 1.0],
  "7": [2.0, 1.0],
  "8": [0.0, 1.0],
  "9": [0.0, 1.0],
  "10": [0.0, 1.0],
  "11": [0.0, 1.0],
  "12": [0.0, 1.0]
}'

# 示例2: 从空状态开始（取消注释以使用）
# X0='{}'


# ===== 输出配置 =====
OUTPUT_BASE="output"    # 输出基础目录
VERBOSE=""              # 添加 "--verbose" 以启用详细日志


# ===== 可视化配置 =====
START_INDEX=0           # 可视化开始的批次索引（默认0表示从头开始）
JUMP=1                  # 差异计算的步长（默认1表示相邻批次）
NO_PLOT=""              # 添加 "--no_plot" 以禁用自动生成图表


# ===== 精度配置 =====
PRECISION=40            # 数值精度（小数位数，默认10位）


################################################################################
# 以下部分通常不需要修改
################################################################################

echo "=============================================================================="
echo "运行多类型LLM调度模拟"
echo "=============================================================================="
echo ""
echo "配置参数:"
echo "  Type 0: l0=$L0_0, l1=$L1_0, λ=$LAMBDA_0"
echo "  Type 1: l0=$L0_1, l1=$L1_1, λ=$LAMBDA_1"
echo "  GPU容量: B=$B"
echo "  服务时间: s(n) = $B0 + $B1 * Z(n)"
echo "  模拟步数: $STEPS"
echo "  输出目录: $OUTPUT_BASE/"
echo ""
echo "可视化配置:"
echo "  开始批次索引: $START_INDEX"
echo "  差异计算步长: $JUMP"
echo ""
echo "精度配置:"
echo "  数值精度: $PRECISION 位小数"
echo "=============================================================================="
echo ""

# 检查Python脚本是否存在
if [ ! -f "run_simulation.py" ]; then
    echo "错误: 找不到 run_simulation.py"
    echo "请确保在 new_project_for_multi_type/ 目录下运行此脚本"
    exit 1
fi

# 检查multi_type_simulator.py是否存在
if [ ! -f "multi_type_simulator.py" ]; then
    echo "错误: 找不到 multi_type_simulator.py"
    echo "请确保在 new_project_for_multi_type/ 目录下运行此脚本"
    exit 1
fi

# 检查visualization.py是否存在（如果启用了绘图）
if [ -z "$NO_PLOT" ] && [ ! -f "visualization.py" ]; then
    echo "警告: 找不到 visualization.py，将跳过图表生成"
fi

# 创建output目录（如果不存在）
mkdir -p "$OUTPUT_BASE"

# 运行Python脚本
python3 run_simulation.py \
    --l0_0 $L0_0 \
    --l1_0 $L1_0 \
    --l0_1 $L0_1 \
    --l1_1 $L1_1 \
    --lambda_0 $LAMBDA_0 \
    --lambda_1 $LAMBDA_1 \
    --B $B \
    --b0 $B0 \
    --b1 $B1 \
    --steps $STEPS \
    --x0 "$X0" \
    --output_base "$OUTPUT_BASE" \
    --start_index $START_INDEX \
    --jump $JUMP \
    --precision $PRECISION \
    $VERBOSE \
    $NO_PLOT

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 模拟成功完成！"
else
    echo ""
    echo "✗ 模拟执行失败，请检查错误信息"
    exit 1
fi
