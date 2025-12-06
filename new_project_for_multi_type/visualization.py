"""
多类型LLM调度器可视化库
从CSV文件读取数据并生成可视化图表
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Tuple


def load_simulation_data(output_dir: str) -> Dict:
    """
    从输出目录加载模拟数据

    参数:
        output_dir: 输出目录路径

    返回:
        包含所有数据的字典
    """
    data = {}

    # 加载配置
    config_file = os.path.join(output_dir, "config.json")
    with open(config_file, 'r', encoding='utf-8') as f:
        data['config'] = json.load(f)

    # 加载CSV数据
    data['x_prime'] = pd.read_csv(os.path.join(output_dir, "x_prime_states.csv"))
    data['admissions'] = pd.read_csv(os.path.join(output_dir, "admissions.csv"))
    data['evictions'] = pd.read_csv(os.path.join(output_dir, "evictions.csv"))
    data['completions'] = pd.read_csv(os.path.join(output_dir, "completions.csv"))
    data['batch_info'] = pd.read_csv(os.path.join(output_dir, "batch_info.csv"))

    return data


def prepare_data_series(df_x_prime: pd.DataFrame, num_types: int, start_index: int = 0) -> Tuple[Dict, List[int], Dict]:
    """
    准备绘图用的数据序列

    参数:
        df_x_prime: X_prime状态的DataFrame
        num_types: 类型数量
        start_index: 开始的批次索引

    返回:
        (data_series, batches, type_valid_lengths)
    """
    # 过滤到start_index之后的数据
    df_filtered = df_x_prime[df_x_prime['batch'] >= start_index].copy()

    # 获取批次列表
    batches = sorted(df_filtered['batch'].unique())

    # 为每个类型找出有效的长度
    type_valid_lengths = {}
    for type_idx in range(num_types):
        type_data = df_filtered[df_filtered['type'] == type_idx]
        # 找出有非零值的长度
        lengths_with_data = type_data.groupby('length')['count'].sum()
        type_valid_lengths[type_idx] = sorted(lengths_with_data[lengths_with_data > 0].index.tolist())

        # 如果没有非零值，至少要包含所有出现过的长度
        if not type_valid_lengths[type_idx]:
            type_valid_lengths[type_idx] = sorted(type_data['length'].unique().tolist())

    # 创建数据序列
    data_series = {}
    for type_idx in range(num_types):
        for length in type_valid_lengths[type_idx]:
            key = f"Type{type_idx}_L{length}"

            # 获取该类型和长度的时间序列
            mask = (df_filtered['type'] == type_idx) & (df_filtered['length'] == length)
            series_data = df_filtered[mask].sort_values('batch')

            data_series[key] = {
                'type': type_idx,
                'length': length,
                'values': series_data['count'].tolist(),
                'batches': series_data['batch'].tolist()
            }

    return data_series, batches, type_valid_lengths


def plot_state_evolution(output_dir: str, start_index: int = 0, save: bool = True) -> str:
    """
    绘制状态演变图：每个type的每个stage的数目随batch count的变化

    参数:
        output_dir: 输出目录路径
        start_index: 开始的批次索引（默认0）
        save: 是否保存图表（默认True）

    返回:
        保存的文件路径
    """
    print(f"\n生成状态演变图（从批次 {start_index} 开始）...")

    # 加载数据
    data = load_simulation_data(output_dir)
    config = data['config']
    df_x_prime = data['x_prime']

    # 获取参数
    request_types = config['request_types']
    num_types = len(request_types)
    B = config['B']
    arrival_rates = config['arrival_rates']

    # 准备数据序列
    data_series, batches, type_valid_lengths = prepare_data_series(df_x_prime, num_types, start_index)

    # 动态选择颜色
    colors_type0 = plt.cm.Blues(np.linspace(0.4, 0.9, max(len(type_valid_lengths[0]), 1)))
    colors_type1 = plt.cm.Reds(np.linspace(0.3, 0.9, max(len(type_valid_lengths[1]), 1)))

    # 创建图表
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, :])

    # 1. 总体演变图（所有类型和长度）
    ax1.set_title('All Types and Lengths Evolution (After Admission/Eviction)',
                  fontsize=14, fontweight='bold')

    # Type 0
    for i, length in enumerate(type_valid_lengths[0]):
        key = f"Type0_L{length}"
        if key in data_series:
            ax1.plot(data_series[key]['batches'], data_series[key]['values'],
                    label=f"Type 0, Length {length}",
                    color=colors_type0[i], linewidth=2, marker='o', markersize=3)

    # Type 1
    for i, length in enumerate(type_valid_lengths[1]):
        key = f"Type1_L{length}"
        if key in data_series:
            ax1.plot(data_series[key]['batches'], data_series[key]['values'],
                    label=f"Type 1, Length {length}",
                    color=colors_type1[i], linewidth=2, marker='s', markersize=3)

    ax1.set_xlabel('Batch Number')
    ax1.set_ylabel('Number of Requests')
    ax1.legend(loc='upper right', ncol=2, fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. Type 0单独图
    ax2.set_title(f'Type 0 (l0={request_types[0][0]}, l1={request_types[0][1]})',
                  fontweight='bold')
    for i, length in enumerate(type_valid_lengths[0]):
        key = f"Type0_L{length}"
        if key in data_series:
            ax2.plot(data_series[key]['batches'], data_series[key]['values'],
                    label=f"Length {length}",
                    color=colors_type0[i], linewidth=2, marker='o')
    ax2.set_xlabel('Batch Number')
    ax2.set_ylabel('Number of Requests')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Type 1单独图
    ax3.set_title(f'Type 1 (l0={request_types[1][0]}, l1={request_types[1][1]})',
                  fontweight='bold')
    for i, length in enumerate(type_valid_lengths[1]):
        key = f"Type1_L{length}"
        if key in data_series:
            ax3.plot(data_series[key]['batches'], data_series[key]['values'],
                    label=f"Length {length}",
                    color=colors_type1[i], linewidth=2, marker='s')
    ax3.set_xlabel('Batch Number')
    ax3.set_ylabel('Number of Requests')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 堆叠面积图
    ax4.set_title('Stacked Area Chart - Request Distribution Over Time', fontweight='bold')

    # 准备堆叠数据（需要确保所有批次都有数据）
    batch_to_idx = {batch: idx for idx, batch in enumerate(batches)}

    # Type 0堆叠
    bottom_type0 = np.zeros(len(batches))
    for i, length in enumerate(type_valid_lengths[0]):
        key = f"Type0_L{length}"
        if key in data_series:
            # 创建对齐的值数组
            values = np.zeros(len(batches))
            for batch, val in zip(data_series[key]['batches'], data_series[key]['values']):
                if batch in batch_to_idx:
                    values[batch_to_idx[batch]] = val

            ax4.fill_between(batches, bottom_type0, bottom_type0 + values,
                           alpha=0.7, color=colors_type0[i],
                           label=f"Type 0, L{length}")
            bottom_type0 += values

    # Type 1堆叠
    bottom_type1 = np.zeros(len(batches))
    for i, length in enumerate(type_valid_lengths[1]):
        key = f"Type1_L{length}"
        if key in data_series:
            values = np.zeros(len(batches))
            for batch, val in zip(data_series[key]['batches'], data_series[key]['values']):
                if batch in batch_to_idx:
                    values[batch_to_idx[batch]] = val

            ax4.fill_between(batches, bottom_type1, bottom_type1 + values,
                           alpha=0.7, color=colors_type1[i],
                           label=f"Type 1, L{length}")
            bottom_type1 += values

    ax4.set_xlabel('Batch Number')
    ax4.set_ylabel('Number of Requests')
    ax4.legend(loc='upper right', ncol=3, fontsize=9)
    ax4.grid(True, alpha=0.3)

    # 创建类型信息字符串
    type_info = ", ".join([f"Type{i}(l0={l0},l1={l1})" for i, (l0, l1) in enumerate(request_types)])

    plt.suptitle(f'Multi-Type LLM Scheduler State Evolution\n{type_info}\nB={B}, λ={arrival_rates}, Start Index={start_index}',
                fontsize=16, fontweight='bold')

    # 保存图表
    if save:
        output_file = os.path.join(output_dir, f"state_evolution_from_{start_index}.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ 状态演变图已保存: {output_file}")
        return output_file

    return None


def plot_state_differences(output_dir: str, start_index: int = 0, jump: int = 1, save: bool = True) -> str:
    """
    绘制状态差异图：相邻位置（或jump步）差的绝对值

    参数:
        output_dir: 输出目录路径
        start_index: 开始的批次索引（默认0）
        jump: 步长跳跃（默认1表示相邻步）
        save: 是否保存图表（默认True）

    返回:
        保存的文件路径
    """
    print(f"\n生成状态差异图（从批次 {start_index} 开始，jump={jump}）...")

    # 加载数据
    data = load_simulation_data(output_dir)
    config = data['config']
    df_x_prime = data['x_prime']

    # 获取参数
    request_types = config['request_types']
    num_types = len(request_types)
    B = config['B']
    arrival_rates = config['arrival_rates']

    # 准备数据序列
    data_series, batches, type_valid_lengths = prepare_data_series(df_x_prime, num_types, start_index)

    # 检查数据是否足够
    if len(batches) <= jump:
        print(f"  ⚠ 警告: 数据点不足以计算jump={jump}的差异")
        return None

    # 动态选择颜色
    colors_type0 = plt.cm.Blues(np.linspace(0.4, 0.9, max(len(type_valid_lengths[0]), 1)))
    colors_type1 = plt.cm.Reds(np.linspace(0.3, 0.9, max(len(type_valid_lengths[1]), 1)))

    # 创建图表
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Type 0的差异图
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title(f'Type 0 State Differences (jump={jump})', fontweight='bold')
    ax1.set_yscale('log')

    for i, length in enumerate(type_valid_lengths[0]):
        key = f"Type0_L{length}"
        if key in data_series:
            values = data_series[key]['values']

            # 计算差异
            differences = []
            diff_batches = []
            for j in range(jump, len(values)):
                diff = abs(values[j] - values[j-jump])
                differences.append(diff + 1e-10)  # 添加小值避免log(0)
                diff_batches.append(data_series[key]['batches'][j])

            if differences:
                ax1.plot(diff_batches, differences,
                        label=f"Length {length}",
                        color=colors_type0[i], linewidth=2, marker='o', markersize=3,
                        markevery=max(1, len(differences)//20))

    ax1.set_xlabel('Batch Number')
    ax1.set_ylabel('|X(t+jump) - X(t)| (log scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which="both")

    # 2. Type 1的差异图
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title(f'Type 1 State Differences (jump={jump})', fontweight='bold')
    ax2.set_yscale('log')

    for i, length in enumerate(type_valid_lengths[1]):
        key = f"Type1_L{length}"
        if key in data_series:
            values = data_series[key]['values']

            differences = []
            diff_batches = []
            for j in range(jump, len(values)):
                diff = abs(values[j] - values[j-jump])
                differences.append(diff + 1e-10)
                diff_batches.append(data_series[key]['batches'][j])

            if differences:
                ax2.plot(diff_batches, differences,
                        label=f"Length {length}",
                        color=colors_type1[i], linewidth=2, marker='s', markersize=3,
                        markevery=max(1, len(differences)//20))

    ax2.set_xlabel('Batch Number')
    ax2.set_ylabel('|X(t+jump) - X(t)| (log scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which="both")

    # 3. 所有类型和长度的差异叠加图
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_title(f'All Types and Lengths - State Differences (jump={jump})', fontweight='bold')
    ax3.set_yscale('log')

    # Type 0
    for i, length in enumerate(type_valid_lengths[0]):
        key = f"Type0_L{length}"
        if key in data_series:
            values = data_series[key]['values']

            differences = []
            diff_batches = []
            for j in range(jump, len(values)):
                diff = abs(values[j] - values[j-jump])
                differences.append(diff + 1e-10)
                diff_batches.append(data_series[key]['batches'][j])

            if differences:
                ax3.plot(diff_batches, differences,
                        label=f"Type 0, L{length}",
                        color=colors_type0[i], linewidth=2, marker='o', markersize=2,
                        markevery=max(1, len(differences)//20), alpha=0.7)

    # Type 1
    for i, length in enumerate(type_valid_lengths[1]):
        key = f"Type1_L{length}"
        if key in data_series:
            values = data_series[key]['values']

            differences = []
            diff_batches = []
            for j in range(jump, len(values)):
                diff = abs(values[j] - values[j-jump])
                differences.append(diff + 1e-10)
                diff_batches.append(data_series[key]['batches'][j])

            if differences:
                ax3.plot(diff_batches, differences,
                        label=f"Type 1, L{length}",
                        color=colors_type1[i], linewidth=2, marker='s', markersize=2,
                        markevery=max(1, len(differences)//20), alpha=0.7)

    ax3.set_xlabel('Batch Number')
    ax3.set_ylabel('|X(t+jump) - X(t)| (log scale)')
    ax3.legend(loc='upper right', ncol=2, fontsize=9)
    ax3.grid(True, alpha=0.3, which="both")

    # 创建类型信息字符串
    type_info = ", ".join([f"Type{i}(l0={l0},l1={l1})" for i, (l0, l1) in enumerate(request_types)])

    plt.suptitle(f'State Differences Analysis - Multi-Type LLM Scheduler\n{type_info}\nB={B}, λ={arrival_rates}, jump={jump}, Start Index={start_index}',
                fontsize=14, fontweight='bold')

    # 保存图表
    if save:
        output_file = os.path.join(output_dir, f"state_differences_from_{start_index}_jump_{jump}.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ 状态差异图已保存: {output_file}")
        return output_file

    return None


def plot_length_total_differences(output_dir: str, start_index: int = 0, jump: int = 1, save: bool = True) -> str:
    """
    绘制每个长度的总请求数差异图（不区分type，求和）

    参数:
        output_dir: 输出目录路径
        start_index: 开始的批次索引（默认0）
        jump: 步长跳跃（默认1表示相邻步）
        save: 是否保存图表（默认True）

    返回:
        保存的文件路径
    """
    print(f"\n生成长度总量差异图（从批次 {start_index} 开始，jump={jump}）...")

    # 加载数据
    data = load_simulation_data(output_dir)
    config = data['config']
    df_x_prime = data['x_prime']

    # 获取参数
    request_types = config['request_types']
    num_types = len(request_types)
    B = config['B']
    arrival_rates = config['arrival_rates']

    # 过滤到start_index之后的数据
    df_filtered = df_x_prime[df_x_prime['batch'] >= start_index].copy()

    # 获取批次列表
    batches = sorted(df_filtered['batch'].unique())

    # 检查数据是否足够
    if len(batches) <= jump:
        print(f"  ⚠ 警告: 数据点不足以计算jump={jump}的差异")
        return None

    # 对每个长度，计算所有type的总和
    # 按(batch, length)分组求和
    length_totals = df_filtered.groupby(['batch', 'length'])['count'].sum().reset_index()
    length_totals.columns = ['batch', 'length', 'total_count']

    # 获取所有长度
    all_lengths = sorted(length_totals['length'].unique())

    # 为每个长度创建时间序列
    length_series = {}
    for length in all_lengths:
        length_data = length_totals[length_totals['length'] == length].sort_values('batch')
        length_series[length] = {
            'batches': length_data['batch'].tolist(),
            'values': length_data['total_count'].tolist()
        }

    # 创建颜色映射（使用viridis色系）
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(all_lengths)))

    # 创建图表
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 1, figure=fig, hspace=0.3)

    # 1. 每个长度单独的差异曲线
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title(f'Total Requests by Length - State Differences (jump={jump})',
                  fontweight='bold', fontsize=14)
    ax1.set_yscale('log')

    for i, length in enumerate(all_lengths):
        if length in length_series:
            values = length_series[length]['values']
            batches_for_length = length_series[length]['batches']

            # 计算差异
            differences = []
            diff_batches = []
            for j in range(jump, len(values)):
                diff = abs(values[j] - values[j-jump])
                differences.append(diff + 1e-10)  # 添加小值避免log(0)
                diff_batches.append(batches_for_length[j])

            if differences:
                ax1.plot(diff_batches, differences,
                        label=f"Length {length}",
                        color=colors[i], linewidth=2, marker='o', markersize=3,
                        markevery=max(1, len(differences)//20))

    ax1.set_xlabel('Batch Number', fontsize=12)
    ax1.set_ylabel('|Total(t+jump) - Total(t)| (log scale)', fontsize=12)
    ax1.legend(loc='upper right', ncol=2, fontsize=9)
    ax1.grid(True, alpha=0.3, which="both")

    # 2. 所有长度的平均差异
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title('Average State Difference Across All Lengths',
                  fontweight='bold', fontsize=14)
    ax2.set_yscale('log')

    # 计算每个批次所有长度的平均差异
    all_diffs_by_batch = {}
    for length in all_lengths:
        if length in length_series:
            values = length_series[length]['values']
            batches_for_length = length_series[length]['batches']

            for j in range(jump, len(values)):
                batch = batches_for_length[j]
                diff = abs(values[j] - values[j-jump])

                if batch not in all_diffs_by_batch:
                    all_diffs_by_batch[batch] = []
                all_diffs_by_batch[batch].append(diff)

    # 计算平均值
    avg_batches = sorted(all_diffs_by_batch.keys())
    avg_diffs = [np.mean(all_diffs_by_batch[b]) + 1e-10 for b in avg_batches]
    max_diffs = [np.max(all_diffs_by_batch[b]) + 1e-10 for b in avg_batches]

    ax2.plot(avg_batches, avg_diffs,
            label='Average Difference',
            color='blue', linewidth=3, marker='o', markersize=4)
    ax2.plot(avg_batches, max_diffs,
            label='Max Difference',
            color='red', linewidth=2, marker='s', markersize=3, alpha=0.7)

    ax2.set_xlabel('Batch Number', fontsize=12)
    ax2.set_ylabel('Difference (log scale)', fontsize=12)
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3, which="both")

    # 创建类型信息字符串
    type_info = ", ".join([f"Type{i}(l0={l0},l1={l1})" for i, (l0, l1) in enumerate(request_types)])

    plt.suptitle(f'Length Total Differences Analysis (All Types Summed)\n{type_info}\nB={B}, λ={arrival_rates}, jump={jump}, Start Index={start_index}',
                fontsize=14, fontweight='bold')

    # 保存图表
    if save:
        output_file = os.path.join(output_dir, f"length_total_differences_from_{start_index}_jump_{jump}.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ 长度总量差异图已保存: {output_file}")
        return output_file

    return None


def generate_all_plots(output_dir: str, start_index: int = 0, jump: int = 1):
    """
    生成所有可视化图表

    参数:
        output_dir: 输出目录路径
        start_index: 开始的批次索引（默认0）
        jump: 差异计算的步长（默认1）
    """
    print("\n" + "=" * 80)
    print("生成可视化图表")
    print("=" * 80)

    # 1. 状态演变图
    plot_state_evolution(output_dir, start_index=start_index, save=True)

    # 2. 状态差异图（按type分类）
    plot_state_differences(output_dir, start_index=start_index, jump=jump, save=True)

    # 3. 长度总量差异图（不区分type）
    plot_length_total_differences(output_dir, start_index=start_index, jump=jump, save=True)

    print("\n" + "=" * 80)
    print("可视化完成！")
    print("=" * 80)


# 命令行接口
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='生成多类型LLM调度器可视化图表')
    parser.add_argument('output_dir', type=str, help='模拟输出目录路径')
    parser.add_argument('--start_index', type=int, default=0,
                       help='开始的批次索引（默认0）')
    parser.add_argument('--jump', type=int, default=1,
                       help='差异计算的步长（默认1）')

    args = parser.parse_args()

    # 生成所有图表
    generate_all_plots(args.output_dir, start_index=args.start_index, jump=args.jump)
