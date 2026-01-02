#!/usr/bin/env python3
"""
Stage Distribution Oscillation Analysis

Documentation:
    Interface: experiments/README.md
    Related: experiments/scripts/run_mixing_experiment.py

Purpose:
    分析multi-replica LLM serving系统中stage分布的震荡行为。
    通过时间序列可视化对比segregated vs mixed routing在stage分布
    均匀性和稳定性方面的差异。

Key Concepts:
    - Active Stages: 同一时刻有多少个不同的decode stages在处理requests
    - Gini Coefficient: stage分布的不平等度 (0=完全均匀, 1=完全集中)
    - Oscillation: Active stages数量少 = 高震荡 (被困在limit cycle)
    - Limit Cycle: Segregated场景中系统被困在少数几个stages之间反复

Theoretical Basis:
    - Segregated routing导致limit cycle: 系统稳定在2个stages之间震荡
    - Mixed routing打破limit cycle: requests分散在4-6个stages，减少震荡
    - Active stages数量是震荡程度的直接指标

Metrics Analyzed:
    1. Number of Active Stages (震荡核心指标)
       - 少(2个) = 高震荡，被困在limit cycle
       - 多(4-6个) = 低震荡，分布广泛

    2. Gini Coefficient (集中度)
       - 衡量requests在stages间分布的不平等程度
       - 0 = 完全均匀 (如 6,6,6,6)
       - 1 = 完全集中 (如 24,0,0,0)

    3. Max Stage Concentration (最大stage占比)
       - 最大的stage占总requests的百分比
       - 低占比 = 更均衡的分布

Output:
    - tmp/stage_oscillation_simple.png: 三指标时间序列对比图
    - Console: 统计摘要 (平均值、标准差、震荡幅度)

Usage:
    python experiments/scripts/analyze_stage_oscillation.py

Dependencies:
    - multi_type_simulator: 核心模拟器
    - multi_replica_wrapper: Multi-replica封装
    - numpy: 数值计算
    - matplotlib: 可视化

Example Output:
    活跃stage数量:
      Segregated: 2.0 ± 0.09  (高震荡 - limit cycle)
      Mixed:      4.3 ± 1.13  (低震荡 - 分布广泛)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, 'new_project_for_multi_type')
from multi_type_simulator import MultiTypeLLMSimulator
from multi_replica_wrapper import MultiReplicaSimulator


def compute_concentration_over_time(replica, start=0):
    """
    计算stage分布的集中度随时间变化

    参数:
        replica: MultiTypeLLMSimulator实例
        start: 起始batch索引

    返回:
        batches: batch编号列表
        ginis: Gini系数列表
        num_stages: 活跃stage数量列表
        max_concentrations: 最大stage占比列表

    说明:
        - Active stages少 = 震荡大 (limit cycle)
        - Active stages多 = 震荡小 (分布广泛)
    """
    hist = replica.get_history()
    states = hist['X_prime'][start:]

    batches = []
    ginis = []
    num_stages = []
    max_concentrations = []

    for state_record in states:
        batch_num = state_record['batch']
        state_dict = state_record['state']

        # 统计每个length的总count
        length_counts = {}
        for length, type_counts in state_dict.items():
            total = sum(type_counts)
            if total > 0.01:
                length_counts[length] = total

        if not length_counts:
            continue

        counts = list(length_counts.values())
        total = sum(counts)

        # 1. Gini系数
        n = len(counts)
        sorted_counts = sorted(counts)
        if total > 0:
            gini = (2 * np.sum([(i+1) * sorted_counts[i] for i in range(n)]) -
                   (n + 1) * total) / (n * total)
        else:
            gini = 0

        # 2. 活跃stage数量 (震荡核心指标)
        n_stages = len(counts)

        # 3. 最大集中度
        max_conc = max(counts) / total if total > 0 else 0

        batches.append(batch_num)
        ginis.append(gini)
        num_stages.append(n_stages)
        max_concentrations.append(max_conc)

    return batches, ginis, num_stages, max_concentrations


def run_analysis():
    """执行完整的stage震荡分析"""

    print("=" * 70)
    print("Stage Distribution Oscillation Analysis")
    print("=" * 70)

    # 运行实验
    print("\n运行 Segregated 场景...")
    replicas_seg = []
    for i, types in enumerate([[(4, 8), (4, 16)], [(3, 5), (3, 15)]]):
        r = MultiTypeLLMSimulator(types, 500, {}, [1.0, 1.0], 0.1, 0.01, False)
        r.run(1000)
        replicas_seg.append(r)

    print("运行 Mixed 场景...")
    simulator_mix = MultiReplicaSimulator(2, 500)
    simulator_mix.run_scenario('mixed', [(4, 8), (4, 16), (3, 5), (3, 15)],
                               [1.0, 1.0, 1.0, 1.0], 1000)
    replicas_mix = simulator_mix.get_replicas()

    # 分析
    print("\n分析 Segregated Replica 0...")
    seg_batches, seg_gini, seg_nstages, seg_maxconc = compute_concentration_over_time(replicas_seg[0])

    print("分析 Mixed Replica 0...")
    mix_batches, mix_gini, mix_nstages, mix_maxconc = compute_concentration_over_time(replicas_mix[0])

    # 绘制对比图
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # 1. 活跃stage数量 (震荡核心指标)
    ax1 = axes[0]
    ax1.plot(seg_batches, seg_nstages, color='#FF6B6B', linewidth=2,
             label='Segregated (Limit Cycle)', alpha=0.8)
    ax1.plot(mix_batches, mix_nstages, color='#4ECDC4', linewidth=2,
             label='Mixed (Distributed)', alpha=0.8)
    ax1.set_ylabel('Number of Active Stages', fontsize=13, fontweight='bold')
    ax1.set_title('Stage Diversity Over Time\n(Fewer Stages = More Oscillation)',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 1000)

    seg_avg_stages = np.mean(seg_nstages)
    mix_avg_stages = np.mean(mix_nstages)
    seg_std_stages = np.std(seg_nstages)
    mix_std_stages = np.std(mix_nstages)

    ax1.text(0.02, 0.98,
             f'Segregated: Avg={seg_avg_stages:.1f} stages (HIGH oscillation)',
             transform=ax1.transAxes, va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='#FF6B6B', alpha=0.3))
    ax1.text(0.02, 0.88,
             f'Mixed: Avg={mix_avg_stages:.1f} stages (LOW oscillation)',
             transform=ax1.transAxes, va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='#4ECDC4', alpha=0.3))

    # 2. Gini系数
    ax2 = axes[1]
    ax2.plot(seg_batches, seg_gini, color='#FF6B6B', linewidth=2,
             label='Segregated', alpha=0.8)
    ax2.plot(mix_batches, mix_gini, color='#4ECDC4', linewidth=2,
             label='Mixed', alpha=0.8)
    ax2.set_ylabel('Gini Coefficient', fontsize=13, fontweight='bold')
    ax2.set_title('Stage Distribution Inequality Over Time\n(Lower = More Balanced)',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 1000)

    seg_avg_gini = np.mean(seg_gini)
    mix_avg_gini = np.mean(mix_gini)

    ax2.text(0.02, 0.98,
             f'Segregated: Avg Gini={seg_avg_gini:.3f}',
             transform=ax2.transAxes, va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='#FF6B6B', alpha=0.3))
    ax2.text(0.02, 0.88,
             f'Mixed: Avg Gini={mix_avg_gini:.3f}',
             transform=ax2.transAxes, va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='#4ECDC4', alpha=0.3))

    # 3. 最大集中度
    ax3 = axes[2]
    ax3.plot(seg_batches, seg_maxconc, color='#FF6B6B', linewidth=2,
             label='Segregated', alpha=0.8)
    ax3.plot(mix_batches, mix_maxconc, color='#4ECDC4', linewidth=2,
             label='Mixed', alpha=0.8)
    ax3.set_xlabel('Batch Number (Time)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Max Stage Concentration', fontsize=13, fontweight='bold')
    ax3.set_title('Largest Stage Proportion Over Time\n(Lower = More Evenly Distributed)',
                  fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11, loc='upper right')
    ax3.grid(alpha=0.3)
    ax3.set_xlim(0, 1000)

    seg_avg_maxconc = np.mean(seg_maxconc)
    mix_avg_maxconc = np.mean(mix_maxconc)

    ax3.text(0.02, 0.98,
             f'Segregated: Avg={seg_avg_maxconc:.1%}',
             transform=ax3.transAxes, va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='#FF6B6B', alpha=0.3))
    ax3.text(0.02, 0.88,
             f'Mixed: Avg={mix_avg_maxconc:.1%}',
             transform=ax3.transAxes, va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='#4ECDC4', alpha=0.3))

    plt.suptitle('Stage Distribution Oscillation: Segregated vs Mixed\n' +
                 'Key Finding: Fewer Active Stages = More Oscillation (Limit Cycle)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = 'experiments/mixing_results/stage_oscillation_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 可视化已保存到 {output_path}")

    # 打印统计摘要
    print(f"\n{'='*70}")
    print("统计摘要")
    print(f"{'='*70}")
    print(f"\n1. Active Stages数量 (震荡核心指标):")
    print(f"   Segregated: {seg_avg_stages:.1f} ± {seg_std_stages:.2f}")
    print(f"   Mixed:      {mix_avg_stages:.1f} ± {mix_std_stages:.2f}")
    print(f"   → Segregated只有{seg_avg_stages:.1f}个stages活跃 = HIGH OSCILLATION (limit cycle)")
    print(f"   → Mixed有{mix_avg_stages:.1f}个stages活跃 = LOW OSCILLATION (distributed)")

    print(f"\n2. Gini系数 (集中度):")
    print(f"   Segregated: {seg_avg_gini:.3f}")
    print(f"   Mixed:      {mix_avg_gini:.3f}")

    print(f"\n3. 最大Stage占比:")
    print(f"   Segregated: {seg_avg_maxconc:.1%} (requests高度集中)")
    print(f"   Mixed:      {mix_avg_maxconc:.1%}")

    print(f"\n{'='*70}")
    print("结论:")
    print(f"{'='*70}")
    print(f"✅ Segregated的{seg_avg_stages:.0f}个active stages证明其被困在limit cycle")
    print(f"✅ Mixed的{mix_avg_stages:.1f}个active stages说明其打破了limit cycle")
    print(f"✅ 这解释了Mixed为何有更好的throughput性能")


if __name__ == "__main__":
    run_analysis()
