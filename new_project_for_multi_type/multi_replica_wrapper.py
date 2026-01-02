#!/usr/bin/env python3
"""
Multi-Replica LLM Simulator Wrapper with Routing Strategies

Documentation:
    Usage: experiments/README.md
    Theoretical Basis: notes.tex - heterogeneous requests improve stability
    Related Theorem: GCD Stability Condition (multiple_discrete.tex)

Key Features:
    - Wraps multiple MultiTypeLLMSimulator instances (one per replica)
    - Supports routing strategies: 'segregated', 'mixed'
    - Tracks per-replica and aggregate metrics
    - Does not modify existing simulator code

Mathematical Correspondence:
    - Segregated: Type k → Replica k (homogeneous within replica)
    - Mixed: All types distributed evenly (heterogeneous within replica)
    - Total throughput = sum of replica throughputs
    - gcd(l_A, l_B) affects stability

Dependencies:
    - multi_type_simulator: Core single-replica simulator (reused, not modified)

Example:
    >>> wrapper = MultiReplicaSimulator(num_replicas=2, l0=5, B=500)
    >>> results = wrapper.run_scenario('segregated', l1_values=[129, 256],
    ...                                 arrival_rates=[2.0, 2.0], steps=1000)
    >>> print(results['total_throughput'])
"""

from multi_type_simulator import MultiTypeLLMSimulator
from typing import Dict, List, Literal, Tuple


class MultiReplicaSimulator:
    """Multi-replica wrapper supporting different routing strategies"""

    def __init__(self, num_replicas: int, B: int, b0: float = 0.1, b1: float = 0.01):
        """
        初始化Multi-replica模拟器

        参数:
            num_replicas: Replica数量
            B: 每个replica的GPU容量
            b0, b1: 服务时间参数
        """
        self.num_replicas = num_replicas
        self.B = B
        self.b0 = b0
        self.b1 = b1
        self.replicas = []

    def run_scenario(self, scenario: Literal['segregated', 'mixed'],
                    request_types: List[Tuple[int, int]], arrival_rates: List[float],
                    steps: int) -> Dict:
        """
        运行指定routing scenario

        参数:
            scenario: 'segregated' 或 'mixed'
            request_types: [(l0_A, l1_A), (l0_B, l1_B)] - 完整的请求类型列表
            arrival_rates: [lambda_A, lambda_B]
            steps: 模拟步数

        返回:
            结果字典（包含per-replica和aggregate指标）
        """
        if scenario == 'segregated':
            return self._run_segregated(request_types, arrival_rates, steps)
        elif scenario == 'mixed':
            return self._run_mixed(request_types, arrival_rates, steps)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

    def _run_segregated(self, request_types, arrival_rates, steps):
        """Type segregation: Type k → Replica k"""
        self.replicas = []

        for replica_id in range(self.num_replicas):
            # 每个replica只接收一种complete request type
            replica = MultiTypeLLMSimulator(
                request_type_list=[request_types[replica_id]],
                B=self.B,
                X0={},
                arrival_rates=[arrival_rates[replica_id]],
                b0=self.b0,
                b1=self.b1,
                verbose=False
            )
            replica.run(steps)
            self.replicas.append(replica)

        return self._aggregate_results(scenario='segregated')

    def _run_mixed(self, request_types, arrival_rates, steps):
        """Mixed routing: All types evenly distributed"""
        self.replicas = []

        # 每个replica接收所有types的均等份额
        mixed_rates = [rate / self.num_replicas for rate in arrival_rates]

        for replica_id in range(self.num_replicas):
            replica = MultiTypeLLMSimulator(
                request_type_list=request_types,
                B=self.B,
                X0={},
                arrival_rates=mixed_rates,
                b0=self.b0,
                b1=self.b1,
                verbose=False
            )
            replica.run(steps)
            self.replicas.append(replica)

        return self._aggregate_results(scenario='mixed')

    def _aggregate_results(self, scenario: str) -> Dict:
        """聚合所有replica的结果"""
        from replica_aggregator import aggregate_replica_results
        return aggregate_replica_results(self.replicas, scenario)

    def get_replicas(self) -> List:
        """获取所有replica实例（用于可视化）"""
        return self.replicas
