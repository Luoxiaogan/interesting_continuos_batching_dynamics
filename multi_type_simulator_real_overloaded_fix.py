"""
LLM推理调度问题模拟器 - 多类型请求版本（按decode次数优先级）
使用decode次数作为优先级：decode次数 = 当前长度 - l0
包含可视化功能：追踪状态演变和稳态分析
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from typing import List, Tuple, Dict

class MultiTypeLLMScheduler:
    def __init__(self, 
                 request_type_list: List[Tuple[int, int]],  # [(l0_1, l1_1), (l0_2, l1_2), ...]
                 B: int,  # GPU容量限制
                 X0: Dict[int, List[float]],  # 初始状态 X0[length][type_idx]
                 arrival_rates: List[float],  # 各类型的到达率 [lambda_1, lambda_2, ...]
                 b0: float, 
                 b1: float):
        """
        初始化多类型模拟器
        
        参数:
        request_type_list: 请求类型列表，每个元素是(l0, l1)元组
        B: GPU容量限制
        X0: 初始状态，X0[length][type_idx]表示长度为length的类型type_idx的请求数
        arrival_rates: 各类型的到达率列表
        b0, b1: 批处理时间参数 s(n) = b0 + b1*Z(n)
        """
        self.request_types = request_type_list
        self.num_types = len(request_type_list)
        self.B = B
        self.arrival_rates = arrival_rates
        self.b0 = b0
        self.b1 = b1
        self.T = 0.0
        self.n = 0  # 批次计数
        
        # 计算长度范围
        self.min_length = min(l0 for l0, l1 in request_type_list)
        self.max_length = max(l0 + l1 - 1 for l0, l1 in request_type_list)
        
        # 计算最大decode次数
        self.max_decode = max(l1 for l0, l1 in request_type_list) - 1
        
        # 为每种类型构建有效长度集合和decode次数映射
        self.type_valid_lengths = {}
        self.type_completion_length = {}
        self.type_decode_mapping = {}  # (type_idx, length) -> decode_num
        self.decode_to_requests = {}   # decode_num -> [(type_idx, length), ...]
        
        for type_idx, (l0, l1) in enumerate(request_type_list):
            self.type_valid_lengths[type_idx] = set()
            self.type_decode_mapping[type_idx] = {}
            
            # 长度从l0到l0+l1-1（在l0+l1时完成）
            for length in range(l0, l0 + l1):
                self.type_valid_lengths[type_idx].add(length)
                decode_num = length - l0  # decode次数
                self.type_decode_mapping[type_idx][length] = decode_num
                
                # 添加到decode_to_requests映射
                if decode_num not in self.decode_to_requests:
                    self.decode_to_requests[decode_num] = []
                self.decode_to_requests[decode_num].append((type_idx, length))
            
            self.type_completion_length[type_idx] = l0 + l1
        
        # 初始化状态矩阵 X[length][type_idx]
        self.length_range = range(self.min_length, self.max_length + 1)
        self.X = {}
        for length in self.length_range:
            self.X[length] = [0.0] * self.num_types
        
        # 从X0初始化状态
        if X0:
            for length in X0:
                if length in self.X:
                    self.X[length] = X0[length].copy()
        
        # 记录完成的请求数
        self.completions = [0.0] * self.num_types
        self.total_completions = 0.0
        
        self._print_init()
    
    def _print_init(self):
        """打印初始化信息"""
        print("=" * 80)
        print("Multi-Type LLM Scheduler (Decode Priority Version)")
        print("=" * 80)
        print(f"Configuration:")
        print(f"  Request types: {self.request_types}")
        print(f"  Arrival rates λ: {self.arrival_rates}")
        print(f"  GPU capacity B: {self.B}")
        print(f"  Service time: s(n) = {self.b0} + {self.b1} * Z(n)")
        
        print(f"\nType decode mapping:")
        for type_idx in range(self.num_types):
            l0, l1 = self.request_types[type_idx]
            print(f"  Type {type_idx} (l0={l0}, l1={l1}):")
            decode_map = []
            for length in sorted(self.type_valid_lengths[type_idx]):
                decode_num = self.type_decode_mapping[type_idx][length]
                decode_map.append(f"L{length}→D{decode_num}")
            print(f"    {', '.join(decode_map)}, completes at L{self.type_completion_length[type_idx]}")
        
        print(f"\nInitial state X[length][type]:")
        print("  Length", end="")
        for type_idx in range(self.num_types):
            print(f"    Type{type_idx}", end="")
        print()
        
        has_initial_state = False
        for length in self.length_range:
            if any(self.X[length][t] > 0 for t in range(self.num_types)):
                has_initial_state = True
                print(f"    {length:2d}  ", end="")
                for type_idx in range(self.num_types):
                    if length in self.type_valid_lengths[type_idx]:
                        print(f"   {self.X[length][type_idx]:6.2f}", end="")
                    else:
                        print(f"        -", end="")
                print()
        
        if not has_initial_state:
            print("  (Empty - starting from zero state)")
        print("=" * 80)
    
    def compute_batch_size(self, X_state=None):
        """计算当前批次大小Z(n)"""
        if X_state is None:
            X_state = self.X
        
        Z = 0.0
        for length in self.length_range:
            for type_idx in range(self.num_types):
                if X_state[length][type_idx] > 0:
                    # 批处理时已有length个token，要生成第length+1个
                    Z += (length + 1) * X_state[length][type_idx]
        return Z
    
    def compute_service_time(self, Z):
        """计算服务时间s(n)"""
        return self.b0 + self.b1 * Z
    
    def update(self):
        """执行一次更新（处理一个批次）"""
        
        print("\n" + "=" * 80)
        print(f"Batch {self.n}, T={self.T:.3f}")
        print("=" * 80)
        
        # 显示Eviction优先级（帮助理解）
        print("\nEviction Priority (by decode steps, high to low):")
        for decode_num in sorted(self.decode_to_requests.keys(), reverse=True):
            requests_at_decode = self.decode_to_requests[decode_num]
            req_strs = []
            for type_idx, length in requests_at_decode:
                if self.X[length][type_idx] > 0:
                    req_strs.append(f"T{type_idx}@L{length}({self.X[length][type_idx]:.2f})")
            if req_strs:
                if len(set(type_idx for type_idx, _ in requests_at_decode)) > 1:
                    print(f"  Decode {decode_num}: {', '.join(req_strs)} [n-proportional]")
                else:
                    print(f"  Decode {decode_num}: {', '.join(req_strs)}")
        
        # 1. Admission/Eviction phase - 按decode次数优先级
        X_prime = {}
        for length in self.length_range:
            X_prime[length] = [0.0] * self.num_types
        
        token_used = 0.0
        eviction_details = []
        
        # 按decode次数从高到低处理
        for decode_num in sorted(self.decode_to_requests.keys(), reverse=True):
            if decode_num not in self.decode_to_requests:
                continue
                
            requests_at_decode = self.decode_to_requests[decode_num]
            
            # 如果这个decode层级只有一种类型的请求，直接处理
            types_at_decode = list(set(type_idx for type_idx, _ in requests_at_decode))
            
            if len(types_at_decode) == 1:
                # 只有一种类型，直接填充
                for type_idx, length in requests_at_decode:
                    current_requests = self.X[length][type_idx]
                    if current_requests > 0:
                        needed_tokens = (length + 1) * current_requests
                        
                        if token_used + needed_tokens <= self.B:
                            # 可以完全容纳
                            X_prime[length][type_idx] = current_requests
                            token_used += needed_tokens
                        else:
                            # 部分容纳
                            available_tokens = self.B - token_used
                            if available_tokens > 0:
                                can_admit = available_tokens / (length + 1)
                                X_prime[length][type_idx] = can_admit
                                evicted = current_requests - can_admit
                                eviction_details.append(f"  Evicted {evicted:.2f} from T{type_idx}@L{length}")
                                token_used = self.B
                            else:
                                evicted = current_requests
                                eviction_details.append(f"  Evicted {evicted:.2f} from T{type_idx}@L{length}")
                            break
            else:
                # 多种类型在同一decode层级，按当前请求数量比例处理
                # 先计算这一层级的总需求
                total_needed = 0.0
                type_needs = {}
                request_counts = {}
                
                for type_idx, length in requests_at_decode:
                    current_requests = self.X[length][type_idx]
                    if current_requests > 0:
                        needed = (length + 1) * current_requests
                        type_needs[(type_idx, length)] = needed
                        request_counts[(type_idx, length)] = current_requests
                        total_needed += needed
                
                if total_needed > 0:
                    available_tokens = self.B - token_used
                    
                    if available_tokens >= total_needed:
                        # 可以完全容纳这一层
                        for (type_idx, length), needed in type_needs.items():
                            current_requests = self.X[length][type_idx]
                            X_prime[length][type_idx] = current_requests
                            token_used += needed
                    else:
                        # 需要按比例驱逐（按当前请求数量比例）
                        if available_tokens > 0:
                            # 计算eviction_ratio (统一的驱逐比例)
                            # available_tokens = total_needed * (1 - eviction_ratio)
                            # eviction_ratio = 1 - (available_tokens / total_needed)
                            eviction_ratio = 1 - (available_tokens / total_needed)
                            
                            # 每种类型保留 (1 - eviction_ratio) 的请求
                            for (type_idx, length), current_requests in request_counts.items():
                                # 保留的请求数 = 原请求数 * (1 - eviction_ratio)
                                retained_requests = current_requests * (1 - eviction_ratio)
                                X_prime[length][type_idx] = retained_requests
                                
                                evicted = current_requests - retained_requests
                                if evicted > 0.001:  # 避免浮点误差
                                    eviction_details.append(f"  Evicted {evicted:.2f} from T{type_idx}@L{length} (n-proportional, ratio={eviction_ratio:.3f})")
                                
                                token_used += retained_requests * (length + 1)
                        else:
                            # available_tokens = 0，全部驱逐
                            for (type_idx, length), current_requests in request_counts.items():
                                if current_requests > 0:
                                    eviction_details.append(f"  Evicted {current_requests:.2f} from T{type_idx}@L{length}")
                        
                        break  # 达到容量限制
            
            if token_used >= self.B - 0.0001:  # 浮点数误差容忍
                break
        
        # 打印eviction信息
        if eviction_details:
            print("\nEvictions occurred:")
            for detail in eviction_details:
                print(detail)
        
        # 2. 如果还有剩余容量，按到达率比例准入新请求
        new_admissions = {}
        if token_used < self.B:
            available_tokens = self.B - token_used
            
            # 计算按到达率比例分配的系数
            denominator = 0.0
            for type_idx in range(self.num_types):
                l0, l1 = self.request_types[type_idx]
                denominator += self.arrival_rates[type_idx] * (l0 + 1)
            
            if denominator > 0:
                # 按比例准入新请求
                for type_idx in range(self.num_types):
                    l0, l1 = self.request_types[type_idx]
                    admission = (available_tokens / denominator) * self.arrival_rates[type_idx]
                    X_prime[l0][type_idx] += admission
                    new_admissions[type_idx] = admission
                    token_used += admission * (l0 + 1)
        
        # 3. 执行批处理
        Z = self.compute_batch_size(X_prime)
        s = self.compute_service_time(Z)
        
        # 打印状态矩阵 - After Admission/Eviction
        print(f"\n1. State X' after Admission/Eviction (Z={Z:.2f}, s={s:.3f}):")
        print("   Length", end="")
        for type_idx in range(self.num_types):
            print(f"    Type{type_idx}", end="")
        print()
        for length in self.length_range:
            print(f"     {length:2d}  ", end="")
            for type_idx in range(self.num_types):
                if length in self.type_valid_lengths[type_idx]:
                    print(f"   {X_prime[length][type_idx]:6.2f}", end="")
                else:
                    print(f"        -", end="")
            print()
        
        # 打印新准入信息
        if new_admissions:
            print(f"\n   New admissions (available tokens={self.B - token_used + sum(new_admissions[t] * (self.request_types[t][0] + 1) for t in new_admissions):.2f}):")
            for type_idx, admission in new_admissions.items():
                if admission > 0:
                    l0 = self.request_types[type_idx][0]
                    print(f"     Type {type_idx}: {admission:.3f} at length {l0}")
        
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
                    # 处理后，请求长度增加1
                    next_length = length + 1
                    
                    # 检查是否完成
                    if next_length == self.type_completion_length[type_idx]:
                        # 这些请求完成了
                        completions_this_batch[type_idx] += X_prime[length][type_idx]
                        self.completions[type_idx] += X_prime[length][type_idx]
                        self.total_completions += X_prime[length][type_idx]
                    elif next_length in self.type_valid_lengths[type_idx]:
                        # 推进到下一个长度
                        X_new[next_length][type_idx] = X_prime[length][type_idx]
        
        # 6. 处理新到达（在overloaded情况下自动处理）
        arrivals_during_service = []
        for type_idx in range(self.num_types):
            arrivals = self.arrival_rates[type_idx] * s
            arrivals_during_service.append(arrivals)
        
        self.X = X_new
        self.n += 1
        
        # 打印状态矩阵 - After Execution
        print(f"\n2. State X after Execution (length++, completions processed):")
        print("   Length", end="")
        for type_idx in range(self.num_types):
            print(f"    Type{type_idx}", end="")
        print()
        for length in self.length_range:
            print(f"     {length:2d}  ", end="")
            for type_idx in range(self.num_types):
                if length in self.type_valid_lengths[type_idx]:
                    print(f"   {self.X[length][type_idx]:6.2f}", end="")
                else:
                    print(f"        -", end="")
            print()
        
        # 打印完成信息
        if any(c > 0 for c in completions_this_batch):
            print(f"\n   Completions this batch:", end="")
            for type_idx in range(self.num_types):
                if completions_this_batch[type_idx] > 0:
                    print(f" T{type_idx}={completions_this_batch[type_idx]:.2f}", end="")
            print()
        
        print(f"\n   Cumulative: Completions={self.total_completions:.2f}, Throughput={self.total_completions/self.T:.3f}")
    
    def run(self, steps):
        """运行模拟指定步数"""
        for _ in range(steps):
            self.update()
        
        print("\n" + "=" * 80)
        print("SIMULATION SUMMARY")
        print("=" * 80)
        print(f"Total time: {self.T:.3f}")
        print(f"Total batches: {self.n}")
        print(f"Total completions: {self.total_completions:.2f}")
        print(f"Overall throughput: {self.total_completions/self.T:.3f}")
        
        print(f"\nCompletions by type:")
        for type_idx in range(self.num_types):
            l0, l1 = self.request_types[type_idx]
            throughput = self.completions[type_idx] / self.T
            print(f"  Type {type_idx} (l0={l0}, l1={l1}): {self.completions[type_idx]:.2f} completions, throughput={throughput:.3f}")
        
        print(f"\nFairness analysis:")
        print(f"  Arrival rates: λ = {self.arrival_rates}, sum={sum(self.arrival_rates):.3f}")
        if self.total_completions > 0:
            print(f"  Throughput ratios:")
            for type_idx in range(self.num_types):
                actual_ratio = self.completions[type_idx] / self.total_completions
                expected_ratio = self.arrival_rates[type_idx] / sum(self.arrival_rates)
                print(f"    Type {type_idx}: actual={actual_ratio:.3f}, expected={expected_ratio:.3f}, deviation={(actual_ratio-expected_ratio)*100:.1f}%")


class VisualizedScheduler(MultiTypeLLMScheduler):
    """扩展调度器以记录历史状态（decode priority版本）"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = []
        self.X_prime_history = []  # 记录After Admission/Eviction的状态

    def update(self):
        """重写update方法以记录X_prime状态"""

        # 静默原始打印输出
        original_print = self._print_init
        self._print_init = lambda: None  # 临时禁用打印

        # 执行admission/eviction阶段的逻辑
        X_prime = {}
        for length in self.length_range:
            X_prime[length] = [0.0] * self.num_types

        token_used = 0.0

        # 按decode次数从高到低处理（与父类逻辑相同）
        for decode_num in sorted(self.decode_to_requests.keys(), reverse=True):
            if decode_num not in self.decode_to_requests:
                continue

            requests_at_decode = self.decode_to_requests[decode_num]
            types_at_decode = list(set(type_idx for type_idx, _ in requests_at_decode))

            if len(types_at_decode) == 1:
                for type_idx, length in requests_at_decode:
                    current_requests = self.X[length][type_idx]
                    if current_requests > 0:
                        needed_tokens = (length + 1) * current_requests

                        if token_used + needed_tokens <= self.B:
                            X_prime[length][type_idx] = current_requests
                            token_used += needed_tokens
                        else:
                            available_tokens = self.B - token_used
                            if available_tokens > 0:
                                can_admit = available_tokens / (length + 1)
                                X_prime[length][type_idx] = can_admit
                                token_used = self.B
                            break
            else:
                # 多种类型在同一decode层级，按比例处理
                total_needed = 0.0
                type_needs = {}
                request_counts = {}

                for type_idx, length in requests_at_decode:
                    current_requests = self.X[length][type_idx]
                    if current_requests > 0:
                        needed = (length + 1) * current_requests
                        type_needs[(type_idx, length)] = needed
                        request_counts[(type_idx, length)] = current_requests
                        total_needed += needed

                if total_needed > 0:
                    available_tokens = self.B - token_used

                    if available_tokens >= total_needed:
                        for (type_idx, length), needed in type_needs.items():
                            current_requests = self.X[length][type_idx]
                            X_prime[length][type_idx] = current_requests
                            token_used += needed
                    else:
                        if available_tokens > 0:
                            eviction_ratio = 1 - (available_tokens / total_needed)
                            for (type_idx, length), current_requests in request_counts.items():
                                retained_requests = current_requests * (1 - eviction_ratio)
                                X_prime[length][type_idx] = retained_requests
                                token_used += retained_requests * (length + 1)
                        break

            if token_used >= self.B - 0.0001:
                break

        # 如果还有剩余容量，按到达率比例准入新请求
        if token_used < self.B:
            available_tokens = self.B - token_used
            denominator = 0.0
            for type_idx in range(self.num_types):
                l0, l1 = self.request_types[type_idx]
                denominator += self.arrival_rates[type_idx] * (l0 + 1)

            if denominator > 0:
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

        # 计算批次大小和服务时间
        Z = self.compute_batch_size(X_prime)
        s = self.compute_service_time(Z)

        # 更新时间
        self.T += s

        # 处理完成和推进阶段
        X_new = {}
        for length in self.length_range:
            X_new[length] = [0.0] * self.num_types

        for length in self.length_range:
            for type_idx in range(self.num_types):
                if X_prime[length][type_idx] > 0:
                    next_length = length + 1

                    if next_length == self.type_completion_length[type_idx]:
                        self.completions[type_idx] += X_prime[length][type_idx]
                        self.total_completions += X_prime[length][type_idx]
                    elif next_length in self.type_valid_lengths[type_idx]:
                        X_new[next_length][type_idx] = X_prime[length][type_idx]

        self.X = X_new
        self.n += 1


def plot_state_evolution(sim, num_steps, start_step=10):
    """绘制状态演变图（decode priority版本）

    参数:
        sim: VisualizedScheduler对象
        num_steps: 运行步数
        start_step: 开始绘图的步数（默认10）
    """

    # 运行模拟（静默模式）
    print(f"Running simulation for {num_steps} steps...")
    for _ in range(num_steps):
        sim.update()
    print(f"Simulation completed. Total time: {sim.T:.3f}")

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
                'l1': l1,
                'decode': sim.type_decode_mapping[type_idx][length]  # 添加decode信息
            }

    # 填充数据
    for h in filtered_history:
        for type_idx in range(sim.num_types):
            for length in sim.type_valid_lengths[type_idx]:
                key = f"Type{type_idx}_L{length}"
                data_series[key]['values'].append(h['state'][length][type_idx])

    # 创建图表
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)

    # 设置颜色方案
    colors = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.Oranges]

    # 1. 总体演变图
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title('All Types and Lengths Evolution (After Admission/Eviction)\nGrouped by Decode Priority',
                  fontsize=14, fontweight='bold')

    # 为每个类型绘制线条
    for type_idx in range(sim.num_types):
        color_map = colors[type_idx % len(colors)]
        lengths = sorted(sim.type_valid_lengths[type_idx])
        n_lengths = len(lengths)
        type_colors = color_map(np.linspace(0.3, 0.9, n_lengths))

        for i, length in enumerate(lengths):
            key = f"Type{type_idx}_L{length}"
            decode = data_series[key]['decode']
            ax1.plot(batches, data_series[key]['values'],
                    label=f"T{type_idx} L{length} (D{decode})",
                    color=type_colors[i], linewidth=2,
                    marker=['o', 's', '^', 'D'][type_idx % 4],
                    markersize=3, markevery=max(1, len(batches)//20))

    ax1.set_xlabel('Batch Number')
    ax1.set_ylabel('Number of Requests')
    ax1.legend(loc='upper right', ncol=3, fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. 按类型分组的子图
    for type_idx in range(min(sim.num_types, 2)):  # 最多显示前两个类型
        ax = fig.add_subplot(gs[1, type_idx])
        l0, l1 = sim.request_types[type_idx]
        ax.set_title(f'Type {type_idx} (l0={l0}, l1={l1})', fontweight='bold')

        color_map = colors[type_idx % len(colors)]
        lengths = sorted(sim.type_valid_lengths[type_idx])
        n_lengths = len(lengths)
        type_colors = color_map(np.linspace(0.3, 0.9, n_lengths))

        for i, length in enumerate(lengths):
            key = f"Type{type_idx}_L{length}"
            decode = data_series[key]['decode']
            ax.plot(batches, data_series[key]['values'],
                   label=f"L{length} (Decode={decode})",
                   color=type_colors[i], linewidth=2,
                   marker=['o', 's', '^', 'D'][type_idx % 4])

        ax.set_xlabel('Batch Number')
        ax.set_ylabel('Number of Requests')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 3. 按decode优先级分组的图
    ax3 = fig.add_subplot(gs[2, :])
    ax3.set_title('Request Distribution by Decode Priority Level', fontweight='bold')

    # 组织按decode级别的数据
    decode_levels = {}
    for key, series in data_series.items():
        decode = series['decode']
        if decode not in decode_levels:
            decode_levels[decode] = []
        decode_levels[decode].append(key)

    # 绘制每个decode级别
    for decode in sorted(decode_levels.keys(), reverse=True):
        total_at_decode = np.zeros(len(batches))
        for key in decode_levels[decode]:
            total_at_decode += np.array(data_series[key]['values'])

        ax3.plot(batches, total_at_decode,
                label=f"Decode {decode}",
                linewidth=2.5, marker='o', markersize=4,
                markevery=max(1, len(batches)//20))

    ax3.set_xlabel('Batch Number')
    ax3.set_ylabel('Total Requests')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 堆叠面积图
    ax4 = fig.add_subplot(gs[3, :])
    ax4.set_title('Stacked Area Chart - Request Distribution Over Time', fontweight='bold')

    # 准备堆叠数据
    bottom = np.zeros(len(batches))

    # 按类型和长度堆叠
    for type_idx in range(sim.num_types):
        color_map = colors[type_idx % len(colors)]
        lengths = sorted(sim.type_valid_lengths[type_idx])
        n_lengths = len(lengths)
        type_colors = color_map(np.linspace(0.3, 0.9, n_lengths))

        for i, length in enumerate(lengths):
            key = f"Type{type_idx}_L{length}"
            values = np.array(data_series[key]['values'])
            decode = data_series[key]['decode']
            ax4.fill_between(batches, bottom, bottom + values,
                           alpha=0.6, color=type_colors[i],
                           label=f"T{type_idx} L{length} (D{decode})")
            bottom += values

    ax4.set_xlabel('Batch Number')
    ax4.set_ylabel('Number of Requests')
    ax4.legend(loc='upper right', ncol=4, fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'Multi-Type LLM Scheduler with Decode Priority\nB={sim.B}, λ={sim.arrival_rates}',
                fontsize=16, fontweight='bold')

    return fig, data_series


def analyze_steady_state(sim, data_series, last_n_batches=20):
    """分析稳态特性（decode priority版本）"""

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
            decode = sim.type_decode_mapping[type_idx][length]
            total_avg += avg
            print(f"  Length {length} (Decode {decode}): {avg:.3f} ± {std:.3f}")

        print(f"  Total average: {total_avg:.3f}")

    # 计算decode级别的分布
    print(f"\n\nDistribution by Decode Priority Level:")
    print("-"*40)
    decode_totals = {}
    for key, series in data_series.items():
        decode = series['decode']
        values = series['values'][-last_n_batches:]
        avg = np.mean(values)
        if decode not in decode_totals:
            decode_totals[decode] = 0
        decode_totals[decode] += avg

    total_requests = sum(decode_totals.values())
    for decode in sorted(decode_totals.keys(), reverse=True):
        percentage = (decode_totals[decode] / total_requests * 100) if total_requests > 0 else 0
        print(f"  Decode {decode}: {decode_totals[decode]:.3f} requests ({percentage:.1f}%)")

    # 计算类型间的比例
    type_totals = [0] * sim.num_types
    for type_idx in range(sim.num_types):
        for length in sim.type_valid_lengths[type_idx]:
            key = f"Type{type_idx}_L{length}"
            values = data_series[key]['values'][-last_n_batches:]
            type_totals[type_idx] += np.mean(values)

    total = sum(type_totals)
    print(f"\n\nType distribution in steady state:")
    print("-"*40)
    for type_idx in range(sim.num_types):
        percentage = (type_totals[type_idx] / total * 100) if total > 0 else 0
        print(f"  Type {type_idx}: {type_totals[type_idx]:.3f} ({percentage:.1f}%)")

    print(f"\nExpected distribution based on arrival rates:")
    total_rate = sum(sim.arrival_rates)
    for i, rate in enumerate(sim.arrival_rates):
        print(f"  Type {i}: {rate/total_rate*100:.1f}%")

    # 计算吞吐量比率
    print(f"\n\nThroughput Analysis:")
    print("-"*40)
    print(f"Total completions: {sim.total_completions:.2f}")
    print(f"Total time: {sim.T:.3f}")
    print(f"Overall throughput: {sim.total_completions/sim.T:.3f}")

    if sim.total_completions > 0:
        print(f"\nThroughput ratios:")
        for type_idx in range(sim.num_types):
            actual_ratio = sim.completions[type_idx] / sim.total_completions
            expected_ratio = sim.arrival_rates[type_idx] / sum(sim.arrival_rates)
            deviation = (actual_ratio - expected_ratio) * 100
            print(f"  Type {type_idx}: actual={actual_ratio:.3f}, expected={expected_ratio:.3f}, deviation={deviation:+.1f}%")


# 示例使用
if __name__ == "__main__":
    # 选择运行模式
    mode = "visualization"  # 可选: "basic" 或 "visualization"

    if mode == "basic":
        # 基础模式 - 仅运行模拟并打印详细输出
        print("Example: Decode Priority Scheduling (Basic Mode)")
        print("=" * 80)

        request_types = [
            (2, 5),  # Type 0: short initial, long generation
            (5, 2),  # Type 1: long initial, short generation
        ]

        X0 = {
            2: [3.0, 0.0],   # 3个type 0在长度2 (decode=0)
            3: [2.0, 0.0],   # 2个type 0在长度3 (decode=1)
            4: [1.0, 0.0],   # 1个type 0在长度4 (decode=2)
            5: [0.5, 2.0],   # 0.5个type 0在长度5 (decode=3), 2个type 1在长度5 (decode=0)
            6: [0.5, 1.0],   # 0.5个type 0在长度6 (decode=4), 1个type 1在长度6 (decode=1)
        }

        sim = MultiTypeLLMScheduler(
            request_type_list=request_types,
            B=50,
            X0=X0,
            arrival_rates=[8.0, 4.0],
            b0=0.1,
            b1=0.01
        )

        sim.run(5)

    else:  # visualization mode
        print("Multi-Type LLM Scheduler with Decode Priority - Visualization Mode")
        print("=" * 80)

        # 配置1: 两种类型，不同decode特性
        request_types = [
            (2, 5),  # Type 0: short initial, long generation
            (5, 2),  # Type 1: long initial, short generation
        ]

        X0 = {
            2: [1.0, 0.0],
            3: [1.0, 0.0],
            4: [1.0, 0.0],
            5: [1.0, 1.0],
            6: [1.0, 1.0],
        }

        # 创建可视化调度器
        sim = VisualizedScheduler(
            request_type_list=request_types,
            B=50,
            X0=X0,
            arrival_rates=[8.0, 4.0],
            b0=0.1,
            b1=0.01
        )

        # 运行模拟并绘图
        num_steps = 500  # 运行更多步数以观察稳态行为
        fig, data_series = plot_state_evolution(sim, num_steps, start_step=num_steps-50)

        # 保存图表
        output_file = 'scheduler_decode_priority_evolution.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_file}")

        # 分析稳态
        analyze_steady_state(sim, data_series, last_n_batches=30)

        # 显示图表
        plt.show()

        # 配置2: 三种类型的示例
        print("\n\n" + "=" * 80)
        print("Example 2: Three Request Types")
        print("=" * 80)

        request_types = [
            (1, 3),  # Type 0: very short initial, medium generation
            (2, 2),  # Type 1: medium initial, short generation
            (3, 4),  # Type 2: long initial, long generation
        ]

        # 创建新的可视化调度器
        sim2 = VisualizedScheduler(
            request_type_list=request_types,
            B=80,
            X0={},  # 从空状态开始
            arrival_rates=[6.0, 4.0, 2.0],
            b0=0.1,
            b1=0.01
        )

        # 运行并可视化
        fig2, data_series2 = plot_state_evolution(sim2, 500, start_step=450)

        # 保存第二个图表
        output_file2 = 'scheduler_three_types_evolution.png'
        plt.savefig(output_file2, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_file2}")

        # 分析第二个配置的稳态
        analyze_steady_state(sim2, data_series2, last_n_batches=30)

        plt.show()