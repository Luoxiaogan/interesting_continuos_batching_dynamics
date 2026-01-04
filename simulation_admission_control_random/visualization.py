"""
可视化模块
生成 2D 和 3D 图表
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def generate_2d_plot(output_dir):
    """
    生成 2D 图：横轴 s，纵轴 throughput / latency

    参数:
        output_dir: 包含 results.csv 的目录
    """
    csv_path = os.path.join(output_dir, "results.csv")
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：Throughput vs s
    ax1 = axes[0]
    ax1.plot(df['admission_threshold'], df['throughput'], 'b-o', linewidth=2, markersize=4)
    ax1.set_xlabel('Admission Threshold (s)', fontsize=12)
    ax1.set_ylabel('Throughput (tokens/sec)', fontsize=12)
    ax1.set_title('Throughput vs Admission Threshold', fontsize=14)
    ax1.grid(True, alpha=0.3)
    # 禁用科学计数法，使用固定小数位
    ax1.ticklabel_format(style='plain', axis='y', useOffset=False)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.4f}'))

    # 标注最优点
    max_idx = df['throughput'].idxmax()
    ax1.scatter(df.loc[max_idx, 'admission_threshold'], df.loc[max_idx, 'throughput'],
                color='red', s=100, zorder=10, label=f"Max at s={df.loc[max_idx, 'admission_threshold']}")
    ax1.legend()

    # 右图：Latency vs s
    ax2 = axes[1]
    ax2.plot(df['admission_threshold'], df['latency'], 'r-o', linewidth=2, markersize=4)
    ax2.set_xlabel('Admission Threshold (s)', fontsize=12)
    ax2.set_ylabel('Average Latency (sec)', fontsize=12)
    ax2.set_title('Latency vs Admission Threshold', fontsize=14)
    ax2.grid(True, alpha=0.3)
    # 禁用科学计数法，使用固定小数位
    ax2.ticklabel_format(style='plain', axis='y', useOffset=False)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.4f}'))

    # 标注最小点
    min_idx = df['latency'].idxmin()
    ax2.scatter(df.loc[min_idx, 'admission_threshold'], df.loc[min_idx, 'latency'],
                color='green', s=100, zorder=10, label=f"Min at s={df.loc[min_idx, 'admission_threshold']}")
    ax2.legend()

    plt.tight_layout()

    fig_path = os.path.join(output_dir, "throughput_latency_2d.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"2D plot saved to: {fig_path}")

    # 生成 Throughput-Latency Tradeoff 图
    fig2, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df['latency'], df['throughput'], c=df['admission_threshold'],
                         cmap='viridis', s=50, alpha=0.8)
    ax.set_xlabel('Average Latency (sec)', fontsize=12)
    ax.set_ylabel('Throughput (tokens/sec)', fontsize=12)
    ax.set_title('Throughput-Latency Tradeoff', fontsize=14)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Admission Threshold (s)', fontsize=10)

    ax.grid(True, alpha=0.3)

    tradeoff_path = os.path.join(output_dir, "throughput_latency_tradeoff.png")
    plt.savefig(tradeoff_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Tradeoff plot saved to: {tradeoff_path}")


def generate_timeseries_plot(output_dir, results, n_sample=5, admission_upper_bound=None):
    """
    生成时序图：随机采样 5 个 s 值
    - 图4: 横轴 batch_idx，纵轴 throughput
    - 图5: 横轴 batch_idx，纵轴 latency
    - 图6: 横轴 batch_idx，纵轴 cumulative_eviction 和 admission

    参数:
        output_dir: 输出目录
        results: 实验结果列表
        n_sample: 采样数目
        admission_upper_bound: 大S上界，用于标注
    """
    import random

    # 过滤有 batch_history 的结果
    valid_results = [r for r in results if r.get('batch_history')]
    if not valid_results:
        print("Warning: No batch_history found. Skipping timeseries plots.")
        return

    # 随机采样 n_sample 个 s 值
    if len(valid_results) <= n_sample:
        sampled = valid_results
    else:
        sampled = random.sample(valid_results, n_sample)

    # 按 s 排序
    sampled.sort(key=lambda x: x['admission_threshold'])

    # 图4 & 图5: Throughput 和 Latency 时序（横轴用时间 T）
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax2 = axes[1]

    for r in sampled:
        s = r['admission_threshold']
        batch_history = r['batch_history']
        time_T = [b['T'] for b in batch_history]
        throughput = [b['throughput'] for b in batch_history]
        latency = [b['latency'] for b in batch_history]

        ax1.plot(time_T, throughput, label=f's={s:.2g}', alpha=0.8)
        ax2.plot(time_T, latency, label=f's={s:.2g}', alpha=0.8)

    ax1.set_xlabel('Time (sec)', fontsize=12)
    ax1.set_ylabel('Throughput (tokens/sec)', fontsize=12)
    ax1.set_title('Throughput vs Time', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Time (sec)', fontsize=12)
    ax2.set_ylabel('Average Latency (sec)', fontsize=12)
    ax2.set_title('Latency vs Time', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "timeseries_throughput_latency.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Timeseries plot saved to: {fig_path}")

    # 图6: Cumulative Eviction 和 Admission 时序 (1x2 布局，横轴用时间 T)
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    ax_eviction = axes2[0]
    ax_admission = axes2[1]

    # 左图: Cumulative Eviction
    for r in sampled:
        s = r['admission_threshold']
        batch_history = r['batch_history']
        time_T = [b['T'] for b in batch_history]
        cum_eviction = [b['cumulative_eviction'] for b in batch_history]

        ax_eviction.plot(time_T, cum_eviction, label=f's={s:.2g}', alpha=0.8)

    ax_eviction.set_xlabel('Time (sec)', fontsize=12)
    ax_eviction.set_ylabel('Cumulative Eviction', fontsize=12)
    ax_eviction.set_title('Cumulative Eviction vs Time', fontsize=14)
    ax_eviction.legend()
    ax_eviction.grid(True, alpha=0.3)

    # 右图: Admission per Batch
    for r in sampled:
        s = r['admission_threshold']
        batch_history = r['batch_history']
        time_T = [b['T'] for b in batch_history]
        admission = [b.get('admission', 0) for b in batch_history]

        ax_admission.plot(time_T, admission, label=f's={s:.2g}', alpha=0.8)

    # 标注大S上界
    if admission_upper_bound is not None:
        ax_admission.axhline(y=admission_upper_bound, color='red', linestyle='--',
                            linewidth=2, label=f'S={admission_upper_bound:.2g} (upper bound)')

    ax_admission.set_xlabel('Time (sec)', fontsize=12)
    ax_admission.set_ylabel('Admission per Batch', fontsize=12)
    ax_admission.set_title('Admission vs Time', fontsize=14)
    ax_admission.legend()
    ax_admission.grid(True, alpha=0.3)

    plt.tight_layout()
    eviction_path = os.path.join(output_dir, "timeseries_eviction_admission.png")
    plt.savefig(eviction_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Eviction & Admission plot saved to: {eviction_path}")


def generate_3d_plot(output_dir):
    """
    生成 3D 图：变化 arrival rate，得到三维图

    参数:
        output_dir: 包含 results_3d.csv 的目录
    """
    csv_path = os.path.join(output_dir, "results_3d.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Skipping 3D plot generation.")
        return

    df = pd.read_csv(csv_path)

    lambda_rates = sorted(df['lambda_rate'].unique())
    s_values = sorted(df['admission_threshold'].unique())

    # 创建网格
    Lambda, S = np.meshgrid(lambda_rates, s_values)
    Throughput = np.full_like(Lambda, np.nan, dtype=float)
    Latency = np.full_like(Lambda, np.nan, dtype=float)

    for i, lam in enumerate(lambda_rates):
        for j, s in enumerate(s_values):
            mask = (df['lambda_rate'] == lam) & (df['admission_threshold'] == s)
            if mask.any():
                Throughput[j, i] = df.loc[mask, 'throughput'].values[0]
                Latency[j, i] = df.loc[mask, 'latency'].values[0]

    # 创建 3D 图
    fig = plt.figure(figsize=(16, 6))

    # 3D Throughput Surface
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    # 处理 NaN 值（用于绘图）
    Throughput_plot = np.ma.masked_invalid(Throughput)

    surf1 = ax1.plot_surface(Lambda, S, Throughput_plot, cmap=cm.viridis,
                              alpha=0.8, linewidth=0, antialiased=True)
    ax1.set_xlabel(r'Arrival Rate ($\lambda$)', fontsize=11)
    ax1.set_ylabel('Admission Threshold (s)', fontsize=11)
    ax1.set_zlabel('Throughput', fontsize=11)
    ax1.set_title('Throughput Surface', fontsize=14)
    fig.colorbar(surf1, ax=ax1, shrink=0.6, label='Throughput')

    # 3D Latency Surface
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    Latency_plot = np.ma.masked_invalid(Latency)

    surf2 = ax2.plot_surface(Lambda, S, Latency_plot, cmap=cm.plasma,
                              alpha=0.8, linewidth=0, antialiased=True)
    ax2.set_xlabel(r'Arrival Rate ($\lambda$)', fontsize=11)
    ax2.set_ylabel('Admission Threshold (s)', fontsize=11)
    ax2.set_zlabel('Latency', fontsize=11)
    ax2.set_title('Latency Surface', fontsize=14)
    fig.colorbar(surf2, ax=ax2, shrink=0.6, label='Latency')

    plt.tight_layout()

    fig_path = os.path.join(output_dir, "throughput_latency_3d.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"3D plot saved to: {fig_path}")

    # 生成等高线图（俯视图）
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    # 使用 pcolormesh 代替 contourf 来处理 NaN
    cs1 = ax1.pcolormesh(Lambda, S, Throughput_plot, cmap='viridis', shading='auto')
    ax1.set_xlabel(r'Arrival Rate ($\lambda$)', fontsize=12)
    ax1.set_ylabel('Admission Threshold (s)', fontsize=12)
    ax1.set_title('Throughput Heatmap', fontsize=14)
    plt.colorbar(cs1, ax=ax1, label='Throughput')

    ax2 = axes[1]
    cs2 = ax2.pcolormesh(Lambda, S, Latency_plot, cmap='plasma', shading='auto')
    ax2.set_xlabel(r'Arrival Rate ($\lambda$)', fontsize=12)
    ax2.set_ylabel('Admission Threshold (s)', fontsize=12)
    ax2.set_title('Latency Heatmap', fontsize=14)
    plt.colorbar(cs2, ax=ax2, label='Latency')

    plt.tight_layout()

    contour_path = os.path.join(output_dir, "throughput_latency_heatmap.png")
    plt.savefig(contour_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Heatmap plot saved to: {contour_path}")

    # 生成多条线图（每条线对应一个 lambda）
    fig3, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    for lam in lambda_rates:
        subset = df[df['lambda_rate'] == lam].sort_values('admission_threshold')
        ax1.plot(subset['admission_threshold'], subset['throughput'],
                 marker='o', markersize=3, label=f'$\\lambda$={lam}')
    ax1.set_xlabel('Admission Threshold (s)', fontsize=12)
    ax1.set_ylabel('Throughput (tokens/sec)', fontsize=12)
    ax1.set_title('Throughput vs s (by arrival rate)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    for lam in lambda_rates:
        subset = df[df['lambda_rate'] == lam].sort_values('admission_threshold')
        ax2.plot(subset['admission_threshold'], subset['latency'],
                 marker='o', markersize=3, label=f'$\\lambda$={lam}')
    ax2.set_xlabel('Admission Threshold (s)', fontsize=12)
    ax2.set_ylabel('Average Latency (sec)', fontsize=12)
    ax2.set_title('Latency vs s (by arrival rate)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    lines_path = os.path.join(output_dir, "throughput_latency_by_lambda.png")
    plt.savefig(lines_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Lines plot saved to: {lines_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
        generate_2d_plot(output_dir)
        if os.path.exists(os.path.join(output_dir, "results_3d.csv")):
            generate_3d_plot(output_dir)
    else:
        print("Usage: python visualization.py <output_dir>")
