"""
多类型LLM调度器模拟器 - 核心版本
专注于模拟逻辑和状态记录，用于后续可视化分析

使用decode次数作为优先级：decode次数 = 当前长度 - l0
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class MultiTypeLLMSimulator:
    """
    多类型LLM请求调度模拟器

    核心功能：
    1. 模拟overloaded系统的调度逻辑（admission/eviction by decode priority）
    2. 记录每个批次的状态快照（X_prime: after admission/eviction状态）
    3. 提供历史数据供后续分析和可视化
    """

    def __init__(self,
                 request_type_list: List[Tuple[int, int]],
                 B: int,
                 X0: Dict[int, List[float]],
                 arrival_rates: List[float],
                 b0: float,
                 b1: float,
                 verbose: bool = False,
                 precision: int = 10):
        """
        初始化多类型模拟器

        参数:
            request_type_list: 请求类型列表，每个元素是(l0, l1)元组
            B: GPU容量限制
            X0: 初始状态，X0[length][type_idx]表示长度为length的类型type_idx的请求数
            arrival_rates: 各类型的到达率列表
            b0, b1: 批处理时间参数 s(n) = b0 + b1*Z(n)
            verbose: 是否打印详细日志
            precision: 数值精度（小数位数，默认10）
        """
        self.request_types = request_type_list
        self.num_types = len(request_type_list)
        self.B = B
        self.arrival_rates = arrival_rates
        self.b0 = b0
        self.b1 = b1
        self.verbose = verbose
        self.precision = precision

        # 时间和批次计数
        self.T = 0.0
        self.n = 0

        # 计算长度范围
        self.min_length = min(l0 for l0, l1 in request_type_list)
        self.max_length = max(l0 + l1 - 1 for l0, l1 in request_type_list)

        # 计算最大decode次数
        self.max_decode = max(l1 for l0, l1 in request_type_list) - 1

        # 为每种类型构建有效长度集合和decode次数映射
        self.type_valid_lengths = {}
        self.type_completion_length = {}
        self.type_decode_mapping = {}
        self.decode_to_requests = {}

        for type_idx, (l0, l1) in enumerate(request_type_list):
            self.type_valid_lengths[type_idx] = set()
            self.type_decode_mapping[type_idx] = {}

            for length in range(l0, l0 + l1):
                self.type_valid_lengths[type_idx].add(length)
                decode_num = length - l0
                self.type_decode_mapping[type_idx][length] = decode_num

                if decode_num not in self.decode_to_requests:
                    self.decode_to_requests[decode_num] = []
                self.decode_to_requests[decode_num].append((type_idx, length))

            self.type_completion_length[type_idx] = l0 + l1

        # 初始化状态矩阵 X[length][type_idx]
        self.length_range = range(self.min_length, self.max_length + 1)
        self.X = {}
        for length in self.length_range:
            self.X[length] = [0.0] * self.num_types

        if X0:
            for length in X0:
                if length in self.X:
                    self.X[length] = X0[length].copy()

        self.completions = [0.0] * self.num_types
        self.total_completions = 0.0

        self.history = {
            'X_prime': [],
            'admissions': [],
            'evictions': [],
            'completions': [],
            'batch_info': [],
        }

        if self.verbose:
            self._print_init()

    def _print_init(self):
        """打印初始化信息"""
        print("=" * 80)
        print("Multi-Type LLM Simulator (Decode Priority Version)")
        print("=" * 80)
        print("Configuration:")
        print(f"  Request types: {self.request_types}")
        print(f"  Arrival rates λ: {self.arrival_rates}")
        print(f"  GPU capacity B: {self.B}")
        print(f"  Service time: s(n) = {self.b0} + {self.b1} * Z(n)")

        print("\nType decode mapping:")
        for type_idx in range(self.num_types):
            l0, l1 = self.request_types[type_idx]
            print(f"  Type {type_idx} (l0={l0}, l1={l1}):")
            decode_map = []
            for length in sorted(self.type_valid_lengths[type_idx]):
                decode_num = self.type_decode_mapping[type_idx][length]
                decode_map.append(f"L{length}->D{decode_num}")
            print(f"    {', '.join(decode_map)}, completes at L{self.type_completion_length[type_idx]}")

        print("\nInitial state X[length][type]:")
        self._print_state(self.X)
        print("=" * 80)

    def _print_state(self, state):
        """打印状态矩阵"""
        print("  Length", end="")
        for type_idx in range(self.num_types):
            print(f"    Type{type_idx}", end="")
        print()

        has_state = False
        for length in self.length_range:
            if any(state[length][t] > 0 for t in range(self.num_types)):
                has_state = True
                print(f"    {length:2d}  ", end="")
                for type_idx in range(self.num_types):
                    if length in self.type_valid_lengths[type_idx]:
                        print(f"   {state[length][type_idx]:6.2f}", end="")
                    else:
                        print("        -", end="")
                print()

        if not has_state:
            print("  (Empty state)")

    def compute_batch_size(self, X_state: Optional[Dict] = None) -> float:
        """计算当前批次大小Z(n)"""
        if X_state is None:
            X_state = self.X

        Z = 0.0
        for length in self.length_range:
            for type_idx in range(self.num_types):
                if X_state[length][type_idx] > 0:
                    Z += (length + 1) * X_state[length][type_idx]
        return Z

    def compute_service_time(self, Z: float) -> float:
        """计算服务时间s(n)"""
        return self.b0 + self.b1 * Z

    def update(self):
        """执行一次更新（处理一个批次）"""
        if self.verbose:
            print("\n" + "=" * 80)
            print(f"Batch {self.n}, T={self.T:.3f}")
            print("=" * 80)

        batch_admissions = {type_idx: 0.0 for type_idx in range(self.num_types)}
        batch_evictions = {type_idx: [] for type_idx in range(self.num_types)}

        # ===== 1. Admission/Eviction Phase =====
        X_prime = {}
        for length in self.length_range:
            X_prime[length] = [0.0] * self.num_types

        token_used = 0.0

        for decode_num in sorted(self.decode_to_requests.keys(), reverse=True):
            requests_at_decode = self.decode_to_requests[decode_num]
            types_at_decode = list(set(type_idx for type_idx, _ in requests_at_decode))

            if len(types_at_decode) == 1:
                for type_idx, length in requests_at_decode:
                    current_requests = self.X[length][type_idx]
                    if current_requests <= 0:
                        continue

                    needed_tokens = (length + 1) * current_requests
                    if token_used + needed_tokens <= self.B:
                        X_prime[length][type_idx] = current_requests
                        token_used += needed_tokens
                    else:
                        available_tokens = self.B - token_used
                        if available_tokens > 0:
                            can_admit = available_tokens / (length + 1)
                            X_prime[length][type_idx] = can_admit
                            evicted = current_requests - can_admit
                            batch_evictions[type_idx].append((length, evicted))
                            token_used = self.B
                        else:
                            batch_evictions[type_idx].append((length, current_requests))
                        break
            else:
                total_needed = 0.0
                request_counts = {}

                for type_idx, length in requests_at_decode:
                    current_requests = self.X[length][type_idx]
                    if current_requests <= 0:
                        continue

                    request_counts[(type_idx, length)] = current_requests
                    total_needed += (length + 1) * current_requests

                if total_needed > 0:
                    available_tokens = self.B - token_used

                    if available_tokens >= total_needed:
                        for (type_idx, length), current_requests in request_counts.items():
                            X_prime[length][type_idx] = current_requests
                            token_used += (length + 1) * current_requests
                    else:
                        if available_tokens > 0:
                            eviction_ratio = 1 - (available_tokens / total_needed)
                            for (type_idx, length), current_requests in request_counts.items():
                                retained_requests = current_requests * (1 - eviction_ratio)
                                X_prime[length][type_idx] = retained_requests

                                evicted = current_requests - retained_requests
                                if evicted > 0.001:
                                    batch_evictions[type_idx].append((length, evicted))

                                token_used += retained_requests * (length + 1)
                        else:
                            for (type_idx, length), current_requests in request_counts.items():
                                batch_evictions[type_idx].append((length, current_requests))

                        break

            if token_used >= self.B - 0.0001:
                break

        # ===== 2. 剩余容量准入新请求 =====
        if token_used < self.B:
            available_tokens = self.B - token_used

            denominator = 0.0
            for type_idx in range(self.num_types):
                l0, _ = self.request_types[type_idx]
                denominator += self.arrival_rates[type_idx] * (l0 + 1)

            if denominator > 0:
                for type_idx in range(self.num_types):
                    l0, _ = self.request_types[type_idx]
                    admission = (available_tokens / denominator) * self.arrival_rates[type_idx]
                    X_prime[l0][type_idx] += admission
                    batch_admissions[type_idx] = admission
                    token_used += admission * (l0 + 1)

        # ===== 3. 计算服务时间 =====
        Z = self.compute_batch_size(X_prime)
        s = self.compute_service_time(Z)

        epsilon = 1e-6
        assert abs(Z - self.B) < epsilon, f"Overloaded system must fill GPU: Z={Z}, B={self.B}"
        assert abs(token_used - self.B) < epsilon, "Token accounting must match B"

        if self.verbose:
            print(f"\nAfter Admission/Eviction (Z={Z:.2f}, s={s:.3f}):")
            self._print_state(X_prime)

        # ===== 4. 更新时间 =====
        self.T += s

        # ===== 5. 处理完成和推进阶段 =====
        X_new = {}
        for length in self.length_range:
            X_new[length] = [0.0] * self.num_types

        completions_this_batch = [0.0] * self.num_types

        for length in self.length_range:
            for type_idx in range(self.num_types):
                if X_prime[length][type_idx] <= 0:
                    continue

                next_length = length + 1

                if next_length == self.type_completion_length[type_idx]:
                    completions_this_batch[type_idx] += X_prime[length][type_idx]
                    self.completions[type_idx] += X_prime[length][type_idx]
                    self.total_completions += X_prime[length][type_idx]
                elif next_length in self.type_valid_lengths[type_idx]:
                    X_new[next_length][type_idx] = X_prime[length][type_idx]

        # ===== 6. 记录历史数据 =====
        self.history['X_prime'].append({
            'batch': self.n,
            'time': self.T,
            'state': {length: X_prime[length].copy() for length in self.length_range},
        })
        self.history['admissions'].append({
            'batch': self.n,
            'time': self.T,
            'admissions': batch_admissions.copy(),
        })
        self.history['evictions'].append({
            'batch': self.n,
            'time': self.T,
            'evictions': {k: v.copy() for k, v in batch_evictions.items()},
        })
        self.history['completions'].append({
            'batch': self.n,
            'time': self.T,
            'completions': completions_this_batch.copy(),
        })
        self.history['batch_info'].append({
            'batch': self.n,
            'time': self.T,
            'service_time': s,
            'batch_size': Z,
        })

        # ===== 7. 更新状态 =====
        self.X = X_new
        self.n += 1

        if self.verbose:
            print("\nAfter Execution (length++):")
            self._print_state(self.X)
            if any(c > 0 for c in completions_this_batch):
                print("\nCompletions:", end="")
                for type_idx in range(self.num_types):
                    if completions_this_batch[type_idx] > 0:
                        print(f" T{type_idx}={completions_this_batch[type_idx]:.2f}", end="")
                print()

    def run(self, steps: int):
        """运行模拟指定步数"""
        for _ in range(steps):
            self.update()

        if self.verbose:
            self._print_summary()

    def _print_summary(self):
        """打印模拟总结"""
        print("\n" + "=" * 80)
        print("SIMULATION SUMMARY")
        print("=" * 80)
        print(f"Total time: {self.T:.3f}")
        print(f"Total batches: {self.n}")
        print(f"Total completions: {self.total_completions:.2f}")
        print(f"Overall throughput: {self.total_completions / self.T:.3f}")

        print("\nCompletions by type:")
        for type_idx in range(self.num_types):
            l0, l1 = self.request_types[type_idx]
            throughput = self.completions[type_idx] / self.T
            print(
                f"  Type {type_idx} (l0={l0}, l1={l1}): "
                f"{self.completions[type_idx]:.2f} completions, throughput={throughput:.3f}"
            )

        print("\nFairness analysis:")
        print(f"  Arrival rates: λ = {self.arrival_rates}, sum={sum(self.arrival_rates):.3f}")
        if self.total_completions > 0:
            print("  Throughput ratios:")
            for type_idx in range(self.num_types):
                actual_ratio = self.completions[type_idx] / self.total_completions
                expected_ratio = self.arrival_rates[type_idx] / sum(self.arrival_rates)
                print(
                    f"    Type {type_idx}: actual={actual_ratio:.3f}, "
                    f"expected={expected_ratio:.3f}, "
                    f"deviation={(actual_ratio - expected_ratio) * 100:.1f}%"
                )

    def get_history(self):
        """获取历史记录数据"""
        return self.history

    def get_state_at_batch(self, batch_num: int):
        """获取指定批次的X_prime状态"""
        if batch_num < len(self.history['X_prime']):
            return self.history['X_prime'][batch_num]['state']
        return None


if __name__ == "__main__":
    print("Example 1: Two types with decode priority")
    print("=" * 80)

    request_types = [
        (2, 5),
        (5, 2),
    ]

    X0 = {
        2: [5.0, 0.0],
        3: [2.0, 0.0],
        4: [1.0, 0.0],
        5: [0.5, 2.0],
        6: [0.5, 1.0],
    }

    sim = MultiTypeLLMSimulator(
        request_type_list=request_types,
        B=50,
        X0=X0,
        arrival_rates=[8.0, 4.0],
        b0=0.1,
        b1=0.01,
        verbose=True,
    )

    sim.run(5)
    history = sim.get_history()
    print(f"\nRecorded {len(history['X_prime'])} batches of history data")

    print("\n" + "=" * 80)
    print("Example 2: Three types from empty state (non-verbose)")
    print("=" * 80)

    request_types = [
        (1, 3),
        (2, 2),
        (3, 4),
    ]

    sim2 = MultiTypeLLMSimulator(
        request_type_list=request_types,
        B=80,
        X0={},
        arrival_rates=[6.0, 4.0, 2.0],
        b0=0.1,
        b1=0.01,
        verbose=False,
    )

    sim2.run(100)
    sim2._print_summary()

    print(f"\nRecorded {len(sim2.get_history()['X_prime'])} batches of history data")
