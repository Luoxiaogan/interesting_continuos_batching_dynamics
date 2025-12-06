"""
LLM推理调度问题模拟器 - 多类型请求版本（最终正确版）
使用统一的长度维度来跟踪所有请求
"""

import numpy as np
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
        # 最小长度是min(l0)，最大长度是max(l0+l1)-1（不包括完成时的长度）
        self.min_length = min(l0 for l0, l1 in request_type_list)
        self.max_length = max(l0 + l1 - 1 for l0, l1 in request_type_list)
        
        # 为每种类型构建有效长度集合
        self.type_valid_lengths = {}
        self.type_completion_length = {}  # 记录每种类型的完成长度
        
        for type_idx, (l0, l1) in enumerate(request_type_list):
            self.type_valid_lengths[type_idx] = set()
            # 长度从l0到l0+l1-1（在l0+l1时完成）
            for length in range(l0, l0 + l1):
                self.type_valid_lengths[type_idx].add(length)
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
        print("*" * 60)
        print(f"INIT: Multi-Type LLM Scheduler (Final Version)")
        print(f"Request types: {self.request_types}")
        print(f"Arrival rates: {self.arrival_rates}")
        print(f"GPU capacity B: {self.B}")
        print(f"Length range: {self.min_length} to {self.max_length}")
        print(f"\nType configurations:")
        for type_idx in range(self.num_types):
            l0, l1 = self.request_types[type_idx]
            valid_lengths = sorted(self.type_valid_lengths[type_idx])
            print(f"  Type {type_idx} (l0={l0}, l1={l1}):")
            print(f"    Valid lengths: {valid_lengths}")
            print(f"    Completion at length: {self.type_completion_length[type_idx]}")
        print(f"\nInitial state X[length][type]:")
        for length in self.length_range:
            if any(self.X[length][t] > 0 for t in range(self.num_types)):
                print(f"  Length {length}: {self.X[length]}")
        print("*" * 60)
    
    def compute_batch_size(self, X_state=None):
        """计算当前批次大小Z(n), 这个是active batch size, 先不管batch处理时间这些啦"""
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
                # 这里需要保证l0长度按照num_types递增. 这个的目的是, 在eviction的时候, 相同的长度, l0越小, 说明已经做过了更多次计算, 更不应该evicted.
                # 检查这个长度对这个类型是否有效
                if length not in self.type_valid_lengths[type_idx]:
                    continue
                
                current_requests = self.X[length][type_idx]
                if current_requests > 0:
                    # 每个请求需要 length + 1 个token的空间, 因为计算的是, 如果执行之后的长度
                    needed_tokens = (length + 1) * current_requests
                    # print(f"length={length}, type_idx={type_idx}, current_requests={current_requests}, needed_tokens={needed_tokens}, token_used={token_used}")

                    if token_used + needed_tokens <= self.B:
                        # 可以完全容纳
                        X_prime[length][type_idx] = current_requests
                        token_used += needed_tokens
                    else:
                        # 部分容纳
                        available_tokens = self.B - token_used
                        if available_tokens >= 0:
                            can_admit = available_tokens / (length + 1)
                            X_prime[length][type_idx] = can_admit
                            token_used = self.B
                            assert(current_requests >= can_admit)
                            # 后面没有填入, 其实等价于自动被evicted掉了!
                        # 达到容量限制，停止
                        # print(f"STOP! token_used={token_used}, can_admit={can_admit}")
                        break
            if token_used >= self.B + 0.0001:  # 浮点数误差容忍
                # print(f"token_used={token_used}")
                break
        assert (self.B - token_used >=0)
        
        # 2. 如果还有剩余容量，按到达率比例准入新请求
        if token_used < self.B:
            available_tokens = self.B - token_used
            # print(f"available_tokens={available_tokens}")
            
            # 计算按到达率比例分配的系数x
            # 每种类型的新请求进入其初始长度l0
            denominator = 0.0
            for type_idx in range(self.num_types):
                l0, l1 = self.request_types[type_idx]
                # 新请求的初始长度是l0, 但是被处理之后是l0+1, 我们需要被处理之后还装得下(虽然会有completion, 但是峰值是l0+1)
                denominator += self.arrival_rates[type_idx] * (l0 + 1)
            
            if denominator > 0:
                x = available_tokens / denominator
                
                # 按比例准入新请求
                admission_info = []
                for type_idx in range(self.num_types):
                    l0, l1 = self.request_types[type_idx]
                    admission = (available_tokens / denominator) * self.arrival_rates[type_idx]
                    X_prime[l0][type_idx] += admission
                    token_used += admission * (l0 + 1)
                    # if admission > 0:
                    #     admission_info.append(f"Type {type_idx}: {admission:.3f} at length {l0}, token_used={token_used}")
                
                if admission_info:
                    print(f"Admissions (factor x={(available_tokens / denominator):.3f}):")
                    for info in admission_info:
                        print(f"  {info}")
        
        # 3. 执行批处理
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
        
        # actual_token_usage = sum(length * sum(X_prime[length]) for length in self.length_range)
        # print(f"\nToken usage: {actual_token_usage:.2f}/{self.B} ({actual_token_usage/self.B*100:.1f}%)")
        # print(f"Total requests in GPU: {total_requests:.2f}")
        # print(f"Batch size Z={Z:.2f}, Service time s={s:.3f}")
        
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
        
        # 打印完成状态
        print(f"\nAfter Processing:")
        for length in self.length_range:
            length_sum = sum(self.X[length])
            print(f"  Length {length}: {[f'{x:.2f}' for x in self.X[length]]} (sum={length_sum:.2f})")
        
        if any(c > 0 for c in completions_this_batch):
            # print(f"\nCompletions this batch:")
            for type_idx in range(self.num_types):
                if completions_this_batch[type_idx] > 0:
                    # print(f"  Type {type_idx}: {completions_this_batch[type_idx]:.3f}")
                    pass
        
        # print(f"\nMetrics:")
        # print(f"  Total completions: {self.total_completions:.2f}")
        # print(f"  Throughput: {self.total_completions/self.T:.3f}")
        # print(f"  Arrival rate sum: {sum(self.arrival_rates):.3f}")
        # print(f"  Arrivals during service: {[f'{a:.2f}' for a in arrivals_during_service]}")
    
    def run(self, steps):
        """运行模拟指定步数"""
        for _ in range(steps):
            self.update()
        
        print("\n" + "=" * 60)
        print("SIMULATION SUMMARY")
        print("=" * 60)
        print(f"Total time: {self.T:.3f}")
        print(f"Total batches: {self.n}")
        print(f"Total completions: {self.total_completions:.2f}")
        print(f"Overall throughput: {self.total_completions/self.T:.3f}")
        print(f"\nCompletions by type:")
        for type_idx in range(self.num_types):
            l0, l1 = self.request_types[type_idx]
            print(f"  Type {type_idx} (l0={l0}, l1={l1}): {self.completions[type_idx]:.2f} ({self.completions[type_idx]/self.T:.3f} per time unit)")
        print(f"\nArrival rates: {self.arrival_rates} (sum={sum(self.arrival_rates):.3f})")
        
        # 计算吞吐量比例
        if self.total_completions > 0:
            print(f"\nThroughput ratios:")
            for type_idx in range(self.num_types):
                actual_ratio = self.completions[type_idx] / self.total_completions
                expected_ratio = self.arrival_rates[type_idx] / sum(self.arrival_rates)
                print(f"  Type {type_idx}: actual={actual_ratio:.3f}, expected={expected_ratio:.3f}")


# 示例使用
if __name__ == "__main__":
    print("Example 1: Two types as described")
    print("=" * 60)
    
    # Type 0: l0=1, l1=2 → 长度 1, 2, 完成于3
    # Type 1: l0=2, l1=5 → 长度 2, 3, 4, 5, 6, 完成于7
    request_types = [
        (1, 2),  # Type 0
        (2, 5),  # Type 1
    ]
    
    # 初始状态
    # 长度范围是1到6
    X0 = {
        1: [3.0, 0.0],   # 3个type 0在长度1
        2: [2.0, 4.0],   # 2个type 0在长度2, 4个type 1在长度2
        3: [0.0, 3.0],   # 3个type 1在长度3
        4: [0.0, 2.0],   # 2个type 1在长度4
        5: [0.0, 1.0],   # 1个type 1在长度5
        6: [0.0, 0.5],   # 0.5个type 1在长度6
    }
    
    sim = MultiTypeLLMScheduler(
        request_type_list=request_types,
        B=50,
        X0=X0,
        arrival_rates=[8.0, 4.0],
        b0=0.1,
        b1=0.01
    )
    
    sim.run(50)
    
    # print("\n\nExample 2: Three types with various configurations")
    # print("=" * 60)
    
    # # Type 0: l0=1, l1=3 → 长度 1, 2, 3, 完成于4
    # # Type 1: l0=2, l1=2 → 长度 2, 3, 完成于4  
    # # Type 2: l0=3, l1=4 → 长度 3, 4, 5, 6, 完成于7
    # request_types = [
    #     (1, 3),
    #     (2, 2),
    #     (3, 4),
    # ]
    
    # X0 = {}  # 从空状态开始
    
    # sim = MultiTypeLLMScheduler(
    #     request_type_list=request_types,
    #     B=80,
    #     X0=X0,
    #     arrival_rates=[6.0, 4.0, 2.0],
    #     b0=0.1,
    #     b1=0.01
    # )
    
    # sim.run(5)