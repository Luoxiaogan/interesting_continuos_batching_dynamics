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


def plot_multi_replica_gpu_states(replicas: List, scenario: str, output_dir: str, start_index: int = 0):
    """
    绘制多replica的GPU state演变图(改进版 - 修复legend问题)

    参数:
        replicas: MultiTypeLLMSimulator实例列表
        scenario: 'segregated' 或 'mixed'
        output_dir: 输出目录
        start_index: 开始批次

    生成:
        - replica_{i}_gpu_state.png (每个replica)
        - all_replicas_{scenario}_comparison.png (并排对比)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    num_replicas = len(replicas)

    # 为每个replica生成独立的state evolution图
    for idx, replica in enumerate(replicas):
        hist = replica.get_history()
        states = hist['X_prime']

        # 转换为DataFrame格式
        rows = []
        for state_record in states:
            batch = state_record['batch']
            state_dict = state_record['state']
            for length, type_counts in state_dict.items():
                for type_idx, count in enumerate(type_counts):
                    rows.append({
                        'batch': batch,
                        'length': length,
                        'type': type_idx,
                        'count': count
                    })

        df = pd.DataFrame(rows)

        # 绘制该replica的state evolution
        fig, ax = plt.subplots(figsize=(14, 7))  # 加大图表尺寸

        # 为每个(length, type)绘制曲线
        colors = plt.cm.tab10(range(10))
        line_idx = 0

        for length in sorted(df['length'].unique()):
            for type_idx in sorted(df['type'].unique()):
                mask = (df['length'] == length) & (df['type'] == type_idx)
                data = df[mask].sort_values('batch')

                if len(data) > 0 and data['count'].max() > 0.1:  # 只绘制有意义的曲线
                    ax.plot(data['batch'], data['count'],
                           label=f'Type{type_idx} (L={length})',
                           linewidth=2, alpha=0.8,
                           color=colors[line_idx % 10])
                    line_idx += 1

        ax.set_xlabel('Batch', fontsize=12)
        ax.set_ylabel('Request Count', fontsize=12)
        ax.set_title(f'GPU State Evolution - Replica {idx} ({scenario.capitalize()})',
                    fontsize=14, fontweight='bold')

        # Legend已移除以避免过度拥挤
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/replica_{idx}_gpu_state.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 并排对比图
    fig, axes = plt.subplots(1, num_replicas, figsize=(14, 6))
    if num_replicas == 1:
        axes = [axes]

    for idx, replica in enumerate(replicas):
        hist = replica.get_history()
        states = hist['X_prime']

        rows = []
        for state_record in states:
            batch = state_record['batch']
            state_dict = state_record['state']
            for length, type_counts in state_dict.items():
                for type_idx, count in enumerate(type_counts):
                    rows.append({'batch': batch, 'length': length, 'type': type_idx, 'count': count})

        df = pd.DataFrame(rows)
        ax = axes[idx]

        colors = plt.cm.tab10(range(10))
        line_idx = 0

        for length in sorted(df['length'].unique()):
            for type_idx in sorted(df['type'].unique()):
                mask = (df['length'] == length) & (df['type'] == type_idx)
                data = df[mask].sort_values('batch')

                if len(data) > 0 and data['count'].max() > 0.1:
                    ax.plot(data['batch'], data['count'],
                           label=f'T{type_idx}_L{length}',
                           linewidth=1.5, alpha=0.8,
                           color=colors[line_idx % 10])
                    line_idx += 1

        ax.set_xlabel('Batch', fontsize=11)
        ax.set_ylabel('Request Count', fontsize=11)
        ax.set_title(f'Replica {idx}', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)

    plt.suptitle(f'Multi-Replica GPU States - {scenario.capitalize()}',
                fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为suptitle留空间
    plt.savefig(f'{output_dir}/all_replicas_{scenario}_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ GPU state可视化已保存到 {output_dir}/")


def plot_performance_comparison(results_dict: dict, output_path: str):
    """
    绘制Segregated vs Mixed的性能对比图

    参数:
        results_dict: 包含'segregated'和'mixed'结果的字典
        output_path: 输出图片路径
    """
    import matplotlib.pyplot as plt
    import numpy as np

    seg = results_dict['segregated']
    mix = results_dict['mixed']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Throughput对比
    scenarios = ['Segregated', 'Mixed']
    throughputs = [seg['total_throughput'], mix['total_throughput']]
    colors = ['#FF6B6B', '#4ECDC4']

    bars1 = ax1.bar(scenarios, throughputs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Total Throughput (requests/time)', fontsize=13, fontweight='bold')
    ax1.set_title('Throughput Comparison', fontsize=15, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # 在柱子上标注数值
    for bar, val in zip(bars1, throughputs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 标注improvement
    improvement = (mix['total_throughput'] - seg['total_throughput']) / seg['total_throughput'] * 100
    ax1.text(0.5, max(throughputs) * 1.05, f'Improvement: {improvement:+.2f}%',
            ha='center', fontsize=13, fontweight='bold',
            color='green' if improvement > 0 else 'red',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Latency对比
    latencies = [seg['avg_latency'], mix['avg_latency']]
    bars2 = ax2.bar(scenarios, latencies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Average Latency (time)', fontsize=13, fontweight='bold')
    ax2.set_title('Latency Comparison', fontsize=15, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # 在柱子上标注数值
    for bar, val in zip(bars2, latencies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 标注improvement (latency降低是好事，所以反向)
    lat_improvement = (seg['avg_latency'] - mix['avg_latency']) / seg['avg_latency'] * 100
    ax2.text(0.5, max(latencies) * 1.05, f'Reduction: {lat_improvement:+.2f}%',
            ha='center', fontsize=13, fontweight='bold',
            color='green' if lat_improvement > 0 else 'red',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Segregated vs Mixed Routing - Performance Comparison',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 性能对比图已保存到 {output_path}")


def plot_batch_composition_comparison(segregated_replicas, mixed_replicas, output_path: str):
    """
    对比Segregated vs Mixed的batch组成均匀性

    展示每个length上不同types的分布，评估mixing效果

    参数:
        segregated_replicas: Segregated场景的replica列表
        mixed_replicas: Mixed场景的replica列表
        output_path: 输出图片路径
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 分析函数：计算稳态下的type分布
    def analyze_composition(replicas, scenario_name):
        # 合并所有replicas的最后50个batches的平均状态
        all_data = []

        for replica_id, replica in enumerate(replicas):
            hist = replica.get_history()
            states = hist['X_prime'][-50:]  # 最后50个batches

            for state_record in states:
                state_dict = state_record['state']
                for length, type_counts in state_dict.items():
                    for type_idx, count in enumerate(type_counts):
                        if count > 0.01:
                            all_data.append({
                                'replica': replica_id,
                                'length': length,
                                'type': type_idx,
                                'count': count
                            })

        df = pd.DataFrame(all_data)

        # 按length分组，计算每个type的平均count
        composition = df.groupby(['length', 'type'])['count'].mean().unstack(fill_value=0)

        # 计算每个length的type多样性（Shannon熵）
        diversity = []
        for length in composition.index:
            counts = composition.loc[length].values
            total = counts.sum()
            if total > 0:
                probs = counts / total
                probs = probs[probs > 0]  # 去除0概率
                entropy = -np.sum(probs * np.log2(probs + 1e-10))
                diversity.append({'length': length, 'entropy': entropy, 'num_types': len(probs)})

        return composition, pd.DataFrame(diversity)

    # 分析两种场景
    seg_comp, seg_div = analyze_composition(segregated_replicas, "Segregated")
    mix_comp, mix_div = analyze_composition(mixed_replicas, "Mixed")

    # 1. Segregated - 堆叠条形图展示各length的type分布
    ax1 = axes[0, 0]
    seg_comp.T.plot(kind='bar', stacked=True, ax=ax1,
                    colormap='tab10', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_title('Segregated: Type Distribution by Length', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Request Type', fontsize=11)
    ax1.set_ylabel('Average Count', fontsize=11)
    ax1.legend(title='Length', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(axis='y', alpha=0.3)

    # 2. Mixed - 堆叠条形图展示各length的type分布
    ax2 = axes[0, 1]
    mix_comp.T.plot(kind='bar', stacked=True, ax=ax2,
                    colormap='tab10', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_title('Mixed: Type Distribution by Length', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Request Type', fontsize=11)
    ax2.set_ylabel('Average Count', fontsize=11)
    ax2.legend(title='Length', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(axis='y', alpha=0.3)

    # 3. Type多样性对比（Shannon熵）
    ax3 = axes[1, 0]
    if not seg_div.empty and not mix_div.empty:
        x = np.arange(len(seg_div))
        width = 0.35

        ax3.bar(x - width/2, seg_div['entropy'], width,
                label='Segregated', color='#FF6B6B', alpha=0.8, edgecolor='black')
        ax3.bar(x + width/2, mix_div['entropy'], width,
                label='Mixed', color='#4ECDC4', alpha=0.8, edgecolor='black')

        ax3.set_xlabel('Length', fontsize=11)
        ax3.set_ylabel('Shannon Entropy (bits)', fontsize=11)
        ax3.set_title('Type Diversity at Each Length', fontsize=13, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'L{int(l)}' for l in seg_div['length']], rotation=45)
        ax3.legend(fontsize=10)
        ax3.grid(axis='y', alpha=0.3)
        ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='H=1.0 (moderate mixing)')

    # 4. 总体type分布对比
    ax4 = axes[1, 1]

    # 计算总的type分布
    seg_total = seg_comp.sum(axis=0)
    mix_total = mix_comp.sum(axis=0)

    # 确保两者有相同的type数量（用0填充缺失的types）
    all_types = set(seg_total.index) | set(mix_total.index)
    seg_total = seg_total.reindex(sorted(all_types), fill_value=0)
    mix_total = mix_total.reindex(sorted(all_types), fill_value=0)

    x = np.arange(len(seg_total))
    width = 0.35

    ax4.bar(x - width/2, seg_total.values, width,
            label='Segregated', color='#FF6B6B', alpha=0.8, edgecolor='black')
    ax4.bar(x + width/2, mix_total.values, width,
            label='Mixed', color='#4ECDC4', alpha=0.8, edgecolor='black')

    ax4.set_xlabel('Request Type', fontsize=11)
    ax4.set_ylabel('Total Average Count (across all replicas)', fontsize=11)
    ax4.set_title('Overall Type Distribution', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'Type {i}' for i in seg_total.index])
    ax4.legend(fontsize=10)
    ax4.grid(axis='y', alpha=0.3)

    # 计算并显示变异系数（只用非零值）
    seg_nonzero = seg_total[seg_total > 0]
    mix_nonzero = mix_total[mix_total > 0]
    seg_cv = seg_nonzero.std() / seg_nonzero.mean() if len(seg_nonzero) > 0 and seg_nonzero.mean() > 0 else 0
    mix_cv = mix_nonzero.std() / mix_nonzero.mean() if len(mix_nonzero) > 0 and mix_nonzero.mean() > 0 else 0

    ax4.text(0.5, 0.95, f'Coefficient of Variation\nSeg: {seg_cv:.3f}, Mix: {mix_cv:.3f}',
             transform=ax4.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10, fontweight='bold')

    plt.suptitle('Batch Composition Analysis: Segregated vs Mixed',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Batch组成对比图已保存到 {output_path}")


def plot_stage_distribution_comparison(segregated_replicas, mixed_replicas, output_path: str):
    """
    对比Segregated vs Mixed的decode stage分布均匀性

    展示单个replica内不同length (decode stage)的分布是否均衡

    参数:
        segregated_replicas: Segregated场景的replica列表
        mixed_replicas: Mixed场景的replica列表
        output_path: 输出图片路径
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 分析单个replica的stage分布
    def analyze_stage_distribution(replica):
        hist = replica.get_history()
        states = hist['X_prime'][-50:]  # 最后50个batches

        # 收集所有length的count
        length_counts = []
        for state_record in states:
            state_dict = state_record['state']
            batch_lengths = {}
            for length, type_counts in state_dict.items():
                total_count = sum(type_counts)
                if total_count > 0.01:
                    batch_lengths[length] = total_count
            length_counts.append(batch_lengths)

        # 计算平均length分布
        all_lengths = set()
        for lc in length_counts:
            all_lengths.update(lc.keys())

        avg_dist = {}
        for length in all_lengths:
            counts = [lc.get(length, 0) for lc in length_counts]
            avg_dist[length] = np.mean(counts)

        # 计算stage多样性指标
        if avg_dist:
            lengths = sorted(avg_dist.keys())
            counts = [avg_dist[l] for l in lengths]
            total = sum(counts)

            if total > 0:
                # Shannon熵
                probs = np.array(counts) / total
                probs = probs[probs > 0]
                entropy = -np.sum(probs * np.log2(probs + 1e-10))

                # 变异系数
                cv = np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else 0

                # Length范围
                length_range = max(lengths) - min(lengths) if lengths else 0

                return {
                    'lengths': lengths,
                    'counts': counts,
                    'entropy': entropy,
                    'cv': cv,
                    'length_range': length_range,
                    'num_stages': len(lengths)
                }

        return None

    # 分析所有replicas
    seg_analyses = [analyze_stage_distribution(r) for r in segregated_replicas]
    mix_analyses = [analyze_stage_distribution(r) for r in mixed_replicas]

    # 为每个replica绘制stage分布
    for idx in range(min(2, len(seg_analyses))):
        # Segregated
        ax = axes[idx, 0]
        if seg_analyses[idx]:
            data = seg_analyses[idx]
            ax.bar(range(len(data['lengths'])), data['counts'],
                   color='#FF6B6B', alpha=0.8, edgecolor='black')
            ax.set_xlabel('Length (Decode Stage)', fontsize=10)
            ax.set_ylabel('Avg Request Count', fontsize=10)
            ax.set_title(f'Segregated Replica {idx}\nEntropy={data["entropy"]:.2f}, CV={data["cv"]:.2f}',
                        fontsize=11, fontweight='bold')
            ax.set_xticks(range(len(data['lengths'])))
            ax.set_xticklabels([f'{int(l)}' for l in data['lengths']], rotation=45, fontsize=8)
            ax.grid(axis='y', alpha=0.3)

        # Mixed
        ax = axes[idx, 1]
        if idx < len(mix_analyses) and mix_analyses[idx]:
            data = mix_analyses[idx]
            ax.bar(range(len(data['lengths'])), data['counts'],
                   color='#4ECDC4', alpha=0.8, edgecolor='black')
            ax.set_xlabel('Length (Decode Stage)', fontsize=10)
            ax.set_ylabel('Avg Request Count', fontsize=10)
            ax.set_title(f'Mixed Replica {idx}\nEntropy={data["entropy"]:.2f}, CV={data["cv"]:.2f}',
                        fontsize=11, fontweight='bold')
            ax.set_xticks(range(len(data['lengths'])))
            ax.set_xticklabels([f'{int(l)}' for l in data['lengths']], rotation=45, fontsize=8)
            ax.grid(axis='y', alpha=0.3)

    # 对比指标
    ax_metrics = axes[0, 2]

    seg_entropies = [a['entropy'] for a in seg_analyses if a]
    mix_entropies = [a['entropy'] for a in mix_analyses if a]

    x = np.arange(len(seg_entropies))
    width = 0.35

    ax_metrics.bar(x - width/2, seg_entropies, width,
                   label='Segregated', color='#FF6B6B', alpha=0.8, edgecolor='black')
    ax_metrics.bar(x + width/2, mix_entropies[:len(x)], width,
                   label='Mixed', color='#4ECDC4', alpha=0.8, edgecolor='black')

    ax_metrics.set_xlabel('Replica ID', fontsize=10)
    ax_metrics.set_ylabel('Stage Diversity (Shannon Entropy)', fontsize=10)
    ax_metrics.set_title('Stage Diversity Comparison', fontsize=11, fontweight='bold')
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels([f'R{i}' for i in range(len(x))])
    ax_metrics.legend(fontsize=9)
    ax_metrics.grid(axis='y', alpha=0.3)

    # CV对比
    ax_cv = axes[1, 2]

    seg_cvs = [a['cv'] for a in seg_analyses if a]
    mix_cvs = [a['cv'] for a in mix_analyses if a]

    ax_cv.bar(x - width/2, seg_cvs, width,
              label='Segregated', color='#FF6B6B', alpha=0.8, edgecolor='black')
    ax_cv.bar(x + width/2, mix_cvs[:len(x)], width,
              label='Mixed', color='#4ECDC4', alpha=0.8, edgecolor='black')

    ax_cv.set_xlabel('Replica ID', fontsize=10)
    ax_cv.set_ylabel('Coefficient of Variation', fontsize=10)
    ax_cv.set_title('Stage Distribution Uniformity\n(Lower CV = More Uniform)',
                    fontsize=11, fontweight='bold')
    ax_cv.set_xticks(x)
    ax_cv.set_xticklabels([f'R{i}' for i in range(len(x))])
    ax_cv.legend(fontsize=9)
    ax_cv.grid(axis='y', alpha=0.3)

    plt.suptitle('Decode Stage Distribution Analysis: Segregated vs Mixed',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Stage分布对比图已保存到 {output_path}")


def plot_stage_distribution_stability(segregated_replicas, mixed_replicas, output_path: str):
    """
    对比Segregated vs Mixed的stage分布随时间的稳定性

    展示stage分布是否震荡，以及mixing是否能减少震荡

    参数:
        segregated_replicas: Segregated场景的replica列表
        mixed_replicas: Mixed场景的replica列表
        output_path: 输出图片路径
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.25)

    # 分析单个replica的stage分布随时间变化
    def analyze_stage_evolution(replica, start_batch=0):
        hist = replica.get_history()
        states = hist['X_prime'][start_batch:]

        # 收集每个batch的stage分布
        batch_data = []
        entropies = []
        cvs = []

        for state_record in states:
            batch_num = state_record['batch']
            state_dict = state_record['state']

            # 统计每个length的总count
            length_dist = {}
            for length, type_counts in state_dict.items():
                total = sum(type_counts)
                if total > 0.01:
                    length_dist[length] = total

            # 计算该batch的熵和CV
            if length_dist:
                lengths = sorted(length_dist.keys())
                counts = [length_dist[l] for l in lengths]
                total = sum(counts)

                if total > 0:
                    # Shannon熵
                    probs = np.array(counts) / total
                    probs = probs[probs > 0]
                    entropy = -np.sum(probs * np.log2(probs + 1e-10))
                    entropies.append(entropy)

                    # CV
                    cv = np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else 0
                    cvs.append(cv)
                else:
                    entropies.append(0)
                    cvs.append(0)

                batch_data.append({
                    'batch': batch_num,
                    'length_dist': length_dist,
                    'entropy': entropy if length_dist else 0,
                    'cv': cv if length_dist else 0
                })

        return batch_data, entropies, cvs

    # 分析所有replicas
    seg_data = [analyze_stage_evolution(r, start_batch=0) for r in segregated_replicas]
    mix_data = [analyze_stage_evolution(r, start_batch=0) for r in mixed_replicas]

    # 1. Heatmap: Stage分布随时间变化 (Segregated Replica 0)
    ax1 = fig.add_subplot(gs[0, 0])
    if seg_data[0][0]:
        # 构建heatmap数据
        batches = [d['batch'] for d in seg_data[0][0]]
        all_lengths = set()
        for d in seg_data[0][0]:
            all_lengths.update(d['length_dist'].keys())
        all_lengths = sorted(all_lengths)

        heatmap_data = np.zeros((len(all_lengths), len(batches)))
        for i, d in enumerate(seg_data[0][0]):
            for j, length in enumerate(all_lengths):
                heatmap_data[j, i] = d['length_dist'].get(length, 0)

        im1 = ax1.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax1.set_xlabel('Batch Number', fontsize=11)
        ax1.set_ylabel('Length (Stage)', fontsize=11)
        ax1.set_title('Segregated Replica 0: Stage Distribution Over Time', fontsize=12, fontweight='bold')
        ax1.set_yticks(range(len(all_lengths)))
        ax1.set_yticklabels([f'{int(l)}' for l in all_lengths], fontsize=8)
        plt.colorbar(im1, ax=ax1, label='Request Count')

    # 2. Heatmap: Mixed Replica 0
    ax2 = fig.add_subplot(gs[0, 1])
    if mix_data[0][0]:
        batches = [d['batch'] for d in mix_data[0][0]]
        all_lengths = set()
        for d in mix_data[0][0]:
            all_lengths.update(d['length_dist'].keys())
        all_lengths = sorted(all_lengths)

        heatmap_data = np.zeros((len(all_lengths), len(batches)))
        for i, d in enumerate(mix_data[0][0]):
            for j, length in enumerate(all_lengths):
                heatmap_data[j, i] = d['length_dist'].get(length, 0)

        im2 = ax2.imshow(heatmap_data, aspect='auto', cmap='viridis', interpolation='nearest')
        ax2.set_xlabel('Batch Number', fontsize=11)
        ax2.set_ylabel('Length (Stage)', fontsize=11)
        ax2.set_title('Mixed Replica 0: Stage Distribution Over Time', fontsize=12, fontweight='bold')
        ax2.set_yticks(range(len(all_lengths)))
        ax2.set_yticklabels([f'{int(l)}' for l in all_lengths], fontsize=8)
        plt.colorbar(im2, ax=ax2, label='Request Count')

    # 3. Entropy随时间变化
    ax3 = fig.add_subplot(gs[1, :])
    colors_seg = ['#FF6B6B', '#FF8E8E']
    colors_mix = ['#4ECDC4', '#70D9D1']

    for idx, (data, entropies, cvs) in enumerate(seg_data):
        if data:
            batches = [d['batch'] for d in data]
            ax3.plot(batches, entropies, color=colors_seg[idx], alpha=0.7, linewidth=1.5,
                    label=f'Segregated R{idx}')

    for idx, (data, entropies, cvs) in enumerate(mix_data):
        if data:
            batches = [d['batch'] for d in data]
            ax3.plot(batches, entropies, color=colors_mix[idx], alpha=0.7, linewidth=1.5,
                    label=f'Mixed R{idx}')

    ax3.set_xlabel('Batch Number', fontsize=11)
    ax3.set_ylabel('Shannon Entropy (Stage Diversity)', fontsize=11)
    ax3.set_title('Stage Distribution Entropy Over Time', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9, loc='best')
    ax3.grid(alpha=0.3)

    # 4. CV随时间变化 (震荡指标)
    ax4 = fig.add_subplot(gs[2, :])

    for idx, (data, entropies, cvs) in enumerate(seg_data):
        if data:
            batches = [d['batch'] for d in data]
            ax4.plot(batches, cvs, color=colors_seg[idx], alpha=0.7, linewidth=1.5,
                    label=f'Segregated R{idx}')

    for idx, (data, entropies, cvs) in enumerate(mix_data):
        if data:
            batches = [d['batch'] for d in data]
            ax4.plot(batches, cvs, color=colors_mix[idx], alpha=0.7, linewidth=1.5,
                    label=f'Mixed R{idx}')

    ax4.set_xlabel('Batch Number', fontsize=11)
    ax4.set_ylabel('Coefficient of Variation', fontsize=11)
    ax4.set_title('Stage Distribution CV Over Time (Lower = More Uniform)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9, loc='best')
    ax4.grid(alpha=0.3)

    # 5. 震荡幅度统计
    ax5 = fig.add_subplot(gs[3, 0])

    seg_entropy_stds = [np.std(entropies) for data, entropies, cvs in seg_data if entropies]
    mix_entropy_stds = [np.std(entropies) for data, entropies, cvs in mix_data if entropies]

    x = np.arange(len(seg_entropy_stds))
    width = 0.35

    ax5.bar(x - width/2, seg_entropy_stds, width, label='Segregated',
            color='#FF6B6B', alpha=0.8, edgecolor='black')
    ax5.bar(x + width/2, mix_entropy_stds[:len(x)], width, label='Mixed',
            color='#4ECDC4', alpha=0.8, edgecolor='black')

    ax5.set_xlabel('Replica ID', fontsize=11)
    ax5.set_ylabel('Std Dev of Entropy', fontsize=11)
    ax5.set_title('Entropy Oscillation Amplitude\n(Lower = More Stable)', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels([f'R{i}' for i in range(len(x))])
    ax5.legend(fontsize=9)
    ax5.grid(axis='y', alpha=0.3)

    # 6. CV震荡幅度统计
    ax6 = fig.add_subplot(gs[3, 1])

    seg_cv_stds = [np.std(cvs) for data, entropies, cvs in seg_data if cvs]
    mix_cv_stds = [np.std(cvs) for data, entropies, cvs in mix_data if cvs]

    ax6.bar(x - width/2, seg_cv_stds, width, label='Segregated',
            color='#FF6B6B', alpha=0.8, edgecolor='black')
    ax6.bar(x + width/2, mix_cv_stds[:len(x)], width, label='Mixed',
            color='#4ECDC4', alpha=0.8, edgecolor='black')

    ax6.set_xlabel('Replica ID', fontsize=11)
    ax6.set_ylabel('Std Dev of CV', fontsize=11)
    ax6.set_title('CV Oscillation Amplitude\n(Lower = More Stable)', fontsize=12, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels([f'R{i}' for i in range(len(x))])
    ax6.legend(fontsize=9)
    ax6.grid(axis='y', alpha=0.3)

    plt.suptitle('Stage Distribution Stability Analysis: Segregated vs Mixed',
                fontsize=16, fontweight='bold')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Stage分布稳定性分析图已保存到 {output_path}")


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
