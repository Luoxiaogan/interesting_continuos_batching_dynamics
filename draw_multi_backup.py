"""
多类型LLM调度器可视化分析
追踪After Admission/Eviction状态的演变
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd

# 添加上传文件的路径
import numpy as np
from typing import List, Tuple, Dict
from multi_type_simulator_real_overloaded_fix_backup import MultiTypeLLMScheduler

class VisualizedScheduler(MultiTypeLLMScheduler):
    """扩展调度器以记录历史状态"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = []
        self.X_prime_history = []  # 记录After Admission/Eviction的状态
        
    def update(self):
        """重写update方法以记录X_prime状态"""
        
        print("\n" + "-" * 60)
        print(f"Batch {self.n}, T={self.T:.3f}")
        
        # 1. Admission/Eviction phase
        X_prime = {}
        for length in self.length_range:
            X_prime[length] = [0.0] * self.num_types
        
        token_used = 0.0
        
        # 按长度从高到低处理（高长度=高优先级）
        for length in sorted(self.length_range, reverse=True):
            for type_idx in range(self.num_types):
                if length not in self.type_valid_lengths[type_idx]:
                    continue
                
                current_requests = self.X[length][type_idx]
                if current_requests > 0:
                    needed_tokens = (length + 1) * current_requests
                    
                    if token_used + needed_tokens <= self.B:
                        X_prime[length][type_idx] = current_requests
                        token_used += needed_tokens
                    else:
                        available_tokens = self.B - token_used
                        if available_tokens >= 0:
                            can_admit = available_tokens / (length + 1)
                            X_prime[length][type_idx] = can_admit
                            token_used = self.B
                            assert(current_requests >= can_admit)
                        break
            if token_used >= self.B + 0.0001:
                break
        
        assert (self.B - token_used >= 0)
        
        # 2. 如果还有剩余容量，按到达率比例准入新请求
        if token_used < self.B:
            available_tokens = self.B - token_used
            
            denominator = 0.0
            for type_idx in range(self.num_types):
                l0, l1 = self.request_types[type_idx]
                denominator += self.arrival_rates[type_idx] * (l0 + 1)
            
            if denominator > 0:
                x = available_tokens / denominator
                
                for type_idx in range(self.num_types):
                    l0, l1 = self.request_types[type_idx]
                    admission = (available_tokens / denominator) * self.arrival_rates[type_idx]
                    X_prime[l0][type_idx] += admission
                    token_used += admission * (l0 + 1)
        
        # 记录X_prime状态
        self.X_prime_history.append({
            'batch': self.n,
            'time': self.T,
            'state': {length: X_prime[length].copy() for length in self.length_range}
        })
        
        # 继续原有的update逻辑
        Z = self.compute_batch_size(X_prime)
        print(f"Z={Z}")
        assert (abs(Z-self.B)<=0.000001)
        assert (abs(token_used-self.B)<=0.000001)
        s = self.compute_service_time(Z)
        
        # 打印状态
        print(f"\nAfter Admission/Eviction:")
        total_requests = 0
        for length in self.length_range:
            length_sum = sum(X_prime[length])
            print(f"  Length {length}: {[f'{x:.2f}' for x in X_prime[length]]} (sum={length_sum:.2f})")
            total_requests += length_sum
        
        # 4. 更新时间
        self.T += s
        
        # 5. 处理完成和推进阶段
        X_new = {}
        for length in self.length_range:
            X_new[length] = [0.0] * self.num_types
        
        completions_this_batch = [0.0] * self.num_types
        
        for length in self.length_range:
            for type_idx in range(self.num_types):
                if X_prime[length][type_idx] > 0:
                    next_length = length + 1
                    
                    if next_length == self.type_completion_length[type_idx]:
                        completions_this_batch[type_idx] += X_prime[length][type_idx]
                        self.completions[type_idx] += X_prime[length][type_idx]
                        self.total_completions += X_prime[length][type_idx]
                    elif next_length in self.type_valid_lengths[type_idx]:
                        X_new[next_length][type_idx] = X_prime[length][type_idx]
        
        self.X = X_new
        self.n += 1
        
        # 打印完成状态
        print(f"\nAfter Processing:")
        for length in self.length_range:
            length_sum = sum(self.X[length])
            print(f"  Length {length}: {[f'{x:.2f}' for x in self.X[length]]} (sum={length_sum:.2f})")

def plot_state_differences(sim, data_series, jump=1, start_step=10):
    """绘制状态差异的对数图

    参数:
        sim: 调度器对象
        data_series: 数据序列字典
        jump: 步长跳跃（默认1表示相邻步）
        start_step: 开始计算差异的步数
    """
    # 创建差异图
    fig_diff = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig_diff, hspace=0.3, wspace=0.3)

    # 准备数据
    filtered_history = [h for h in sim.X_prime_history if h['batch'] >= start_step]

    if len(filtered_history) <= jump:
        print(f"Warning: Not enough data points for jump={jump}")
        return fig_diff

    batches = [h['batch'] for h in filtered_history[jump:]]

    # 1. Type 0的差异图
    ax1 = fig_diff.add_subplot(gs[0, 0])
    ax1.set_title(f'Type 0 State Differences (jump={jump})', fontweight='bold')
    ax1.set_yscale('log')

    colors_type0 = plt.cm.Blues(np.linspace(0.4, 0.9, len(sim.type_valid_lengths[0])))

    for i, length in enumerate(sorted(sim.type_valid_lengths[0])):
        key = f"Type0_L{length}"
        values = data_series[key]['values']

        # 计算差异
        differences = []
        for j in range(jump, len(values)):
            diff = abs(values[j] - values[j-jump])
            # 添加小值以避免log(0)
            differences.append(diff + 1e-10)

        if differences:
            ax1.plot(batches[:len(differences)], differences,
                    label=f"Length {length}",
                    color=colors_type0[i], linewidth=2, marker='o', markersize=3,
                    markevery=max(1, len(differences)//20))

    ax1.set_xlabel('Batch Number')
    ax1.set_ylabel('|X(t+jump) - X(t)| (log scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which="both")

    # 2. Type 1的差异图
    ax2 = fig_diff.add_subplot(gs[0, 1])
    ax2.set_title(f'Type 1 State Differences (jump={jump})', fontweight='bold')
    ax2.set_yscale('log')

    colors_type1 = plt.cm.Reds(np.linspace(0.3, 0.9, len(sim.type_valid_lengths[1])))

    for i, length in enumerate(sorted(sim.type_valid_lengths[1])):
        key = f"Type1_L{length}"
        values = data_series[key]['values']

        # 计算差异
        differences = []
        for j in range(jump, len(values)):
            diff = abs(values[j] - values[j-jump])
            differences.append(diff + 1e-10)

        if differences:
            ax2.plot(batches[:len(differences)], differences,
                    label=f"Length {length}",
                    color=colors_type1[i], linewidth=2, marker='s', markersize=3,
                    markevery=max(1, len(differences)//20))

    ax2.set_xlabel('Batch Number')
    ax2.set_ylabel('|X(t+jump) - X(t)| (log scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which="both")

    # 3. 所有类型和长度的差异叠加图
    ax3 = fig_diff.add_subplot(gs[1, :])
    ax3.set_title(f'All Types and Lengths - State Differences (jump={jump})', fontweight='bold')
    ax3.set_yscale('log')

    # Type 0
    for i, length in enumerate(sorted(sim.type_valid_lengths[0])):
        key = f"Type0_L{length}"
        values = data_series[key]['values']

        differences = []
        for j in range(jump, len(values)):
            diff = abs(values[j] - values[j-jump])
            differences.append(diff + 1e-10)

        if differences:
            ax3.plot(batches[:len(differences)], differences,
                    label=f"Type 0, L{length}",
                    color=colors_type0[i], linewidth=2, marker='o', markersize=2,
                    markevery=max(1, len(differences)//20), alpha=0.7)

    # Type 1
    for i, length in enumerate(sorted(sim.type_valid_lengths[1])):
        key = f"Type1_L{length}"
        values = data_series[key]['values']

        differences = []
        for j in range(jump, len(values)):
            diff = abs(values[j] - values[j-jump])
            differences.append(diff + 1e-10)

        if differences:
            ax3.plot(batches[:len(differences)], differences,
                    label=f"Type 1, L{length}",
                    color=colors_type1[i], linewidth=2, marker='s', markersize=2,
                    markevery=max(1, len(differences)//20), alpha=0.7)

    ax3.set_xlabel('Batch Number')
    ax3.set_ylabel('|X(t+jump) - X(t)| (log scale)')
    ax3.legend(loc='upper right', ncol=2, fontsize=9)
    ax3.grid(True, alpha=0.3, which="both")

    plt.suptitle(f'State Differences Analysis - Multi-Type LLM Scheduler\nB={sim.B}, λ={sim.arrival_rates}, jump={jump}',
                fontsize=14, fontweight='bold')

    return fig_diff


def plot_state_evolution(sim, num_steps, start_step=10, compute_differences=True, jump=1):
    """绘制状态演变图

    参数:
        sim: 调度器对象
        num_steps: 运行步数
        start_step: 开始绘图的步数（默认10）
        compute_differences: 是否计算差异图（默认True）
        jump: 差异计算的步长跳跃（默认1）
    """

    # 运行模拟
    sim.run(num_steps)

    # 准备数据并过滤到start_step之后
    filtered_history = [h for h in sim.X_prime_history if h['batch'] >= start_step]
    batches = [h['batch'] for h in filtered_history]
    times = [h['time'] for h in filtered_history]
    
    # 为每个类型和长度组合创建时间序列
    data_series = {}
    for type_idx in range(sim.num_types):
        l0, l1 = sim.request_types[type_idx]
        for length in sim.type_valid_lengths[type_idx]:
            key = f"Type{type_idx}_L{length}"
            data_series[key] = {
                'type': type_idx,
                'length': length,
                'values': [],
                'l0': l0,
                'l1': l1
            }
    
    # 填充数据（使用过滤后的历史）
    for h in filtered_history:
        for type_idx in range(sim.num_types):
            for length in sim.type_valid_lengths[type_idx]:
                key = f"Type{type_idx}_L{length}"
                data_series[key]['values'].append(h['state'][length][type_idx])
    
    # 创建图表
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. 按类型分组的子图
    colors_type0 = plt.cm.Blues(np.linspace(0.4, 0.9, 2))  # Type 0有2个长度
    colors_type1 = plt.cm.Reds(np.linspace(0.3, 0.9, 5))   # Type 1有5个长度
    
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    
    # 总体演变图
    ax1.set_title('All Types and Lengths Evolution (After Admission/Eviction)', fontsize=14, fontweight='bold')
    
    # Type 0的线
    for i, length in enumerate(sorted(sim.type_valid_lengths[0])):
        key = f"Type0_L{length}"
        ax1.plot(batches, data_series[key]['values'], 
                label=f"Type 0, Length {length}", 
                color=colors_type0[i], linewidth=2, marker='o', markersize=3)
    
    # Type 1的线
    for i, length in enumerate(sorted(sim.type_valid_lengths[1])):
        key = f"Type1_L{length}"
        ax1.plot(batches, data_series[key]['values'], 
                label=f"Type 1, Length {length}", 
                color=colors_type1[i], linewidth=2, marker='s', markersize=3)
    
    ax1.set_xlabel('Batch Number')
    ax1.set_ylabel('Number of Requests')
    ax1.legend(loc='upper right', ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Type 0单独图
    ax2.set_title(f'Type 0 (l0={sim.request_types[0][0]}, l1={sim.request_types[0][1]})', fontweight='bold')
    for i, length in enumerate(sorted(sim.type_valid_lengths[0])):
        key = f"Type0_L{length}"
        ax2.plot(batches, data_series[key]['values'], 
                label=f"Length {length}", 
                color=colors_type0[i], linewidth=2, marker='o')
    ax2.set_xlabel('Batch Number')
    ax2.set_ylabel('Number of Requests')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Type 1单独图
    ax3.set_title(f'Type 1 (l0={sim.request_types[1][0]}, l1={sim.request_types[1][1]})', fontweight='bold')
    for i, length in enumerate(sorted(sim.type_valid_lengths[1])):
        key = f"Type1_L{length}"
        ax3.plot(batches, data_series[key]['values'], 
                label=f"Length {length}", 
                color=colors_type1[i], linewidth=2, marker='s')
    ax3.set_xlabel('Batch Number')
    ax3.set_ylabel('Number of Requests')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 堆叠面积图显示总体分布
    ax4 = fig.add_subplot(gs[2, :])
    
    # 准备堆叠数据
    bottom_type0 = np.zeros(len(batches))
    bottom_type1 = np.zeros(len(batches))
    
    # Type 0堆叠
    for i, length in enumerate(sorted(sim.type_valid_lengths[0])):
        key = f"Type0_L{length}"
        values = np.array(data_series[key]['values'])
        ax4.fill_between(batches, bottom_type0, bottom_type0 + values,
                        alpha=0.7, color=colors_type0[i], 
                        label=f"Type 0, L{length}")
        bottom_type0 += values
    
    # Type 1堆叠
    for i, length in enumerate(sorted(sim.type_valid_lengths[1])):
        key = f"Type1_L{length}"
        values = np.array(data_series[key]['values'])
        ax4.fill_between(batches, bottom_type1, bottom_type1 + values,
                        alpha=0.7, color=colors_type1[i],
                        label=f"Type 1, L{length}")
        bottom_type1 += values
    
    ax4.set_title('Stacked Area Chart - Request Distribution Over Time', fontweight='bold')
    ax4.set_xlabel('Batch Number')
    ax4.set_ylabel('Number of Requests')
    ax4.legend(loc='upper right', ncol=3)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Multi-Type LLM Scheduler State Evolution\nB={sim.B}, λ={sim.arrival_rates}',
                fontsize=16, fontweight='bold')

    # 计算差异图（如果需要）
    fig_diff = None
    if compute_differences:
        fig_diff = plot_state_differences(sim, data_series, jump=jump, start_step=start_step)

    return fig, data_series, fig_diff

def analyze_steady_state(sim, data_series, last_n_batches=20):
    """分析稳态特性"""
    
    print("\n" + "="*60)
    print("STEADY STATE ANALYSIS")
    print("="*60)
    
    # 计算最后n个批次的平均值和标准差
    print(f"\nAverage state in last {last_n_batches} batches:")
    print("-"*40)
    
    for type_idx in range(sim.num_types):
        l0, l1 = sim.request_types[type_idx]
        print(f"\nType {type_idx} (l0={l0}, l1={l1}):")
        
        total_avg = 0
        for length in sorted(sim.type_valid_lengths[type_idx]):
            key = f"Type{type_idx}_L{length}"
            values = data_series[key]['values'][-last_n_batches:]
            avg = np.mean(values)
            std = np.std(values)
            total_avg += avg
            print(f"  Length {length}: {avg:.3f} ± {std:.3f}")
        
        print(f"  Total average: {total_avg:.3f}")
    
    # 计算类型间的比例
    type0_total = sum(np.mean(data_series[f"Type0_L{l}"]['values'][-last_n_batches:]) 
                     for l in sim.type_valid_lengths[0])
    type1_total = sum(np.mean(data_series[f"Type1_L{l}"]['values'][-last_n_batches:]) 
                     for l in sim.type_valid_lengths[1])
    
    print(f"\nType distribution in steady state:")
    print(f"  Type 0: {type0_total:.3f} ({type0_total/(type0_total+type1_total)*100:.1f}%)")
    print(f"  Type 1: {type1_total:.3f} ({type1_total/(type0_total+type1_total)*100:.1f}%)")
    
    print(f"\nExpected distribution based on arrival rates:")
    total_rate = sum(sim.arrival_rates)
    for i, rate in enumerate(sim.arrival_rates):
        print(f"  Type {i}: {rate/total_rate*100:.1f}%")

# 主程序
if __name__ == "__main__":
    # # 设置参数
    # request_types = [
    #     (1, 2),  # Type 0
    #     (2, 5),  # Type 1
    # ]
    
    # # 初始状态
    # X0 = {
    #     1: [3.0, 0.0],
    #     2: [2.0, 4.0],
    #     3: [0.0, 3.0],
    #     4: [0.0, 2.0],
    #     5: [0.0, 1.0],
    #     6: [0.0, 0.5],
    # }

    a = 6
    request_types = [
        (a, 2),  # Type 0
        (a, 5),  # Type 1
    ]
    
    # 初始状态
    X0 = {
        a: [1.0, 2.0],
        a+1: [1.0, 5.0],
        a+2: [0.0, 1.0],
        a+3: [0.0, 1.0],
        a+4: [0.0, 1.0],
    }
    
    # 创建可视化调度器
    sim = VisualizedScheduler(
        request_type_list=request_types,
        B= 50,
        X0=X0,
        arrival_rates=[8.0, 4.0],
        b0=0.1,
        b1=0.01
    )
    
    # 运行并可视化
    num_steps = 600
    jump = 1  # 可以调整jump参数来看不同步长的差异
    fig, data_series, fig_diff = plot_state_evolution(sim, num_steps, start_step=0,
                                                      compute_differences=True, jump=jump)

    # 保存图表
    plt.figure(fig.number)
    plt.savefig('scheduler_state_evolution.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: scheduler_state_evolution.png")

    # 保存差异图为PDF
    if fig_diff is not None:
        plt.figure(fig_diff.number)
        plt.savefig('difference.png', dpi=150, bbox_inches='tight')
        print(f"Difference plot saved to: difference.pdf (jump={jump})")

    # 分析稳态
    analyze_steady_state(sim, data_series, last_n_batches=20)

    # 显示图表
    plt.show()