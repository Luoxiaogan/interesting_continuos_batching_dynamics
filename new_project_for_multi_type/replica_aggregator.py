#!/usr/bin/env python3
"""
Multi-Replica Result Aggregation Module

Documentation:
    Interface: experiments/README.md

Key Features:
    - Aggregates completions, throughput, latency from all replicas
    - Detects convergence (stability)
    - Computes load balance metrics

Dependencies:
    - pandas: Data processing

Example:
    >>> results = aggregate_replica_results(replicas, scenario='mixed')
    >>> print(results['total_throughput'])
"""

import pandas as pd
from typing import List, Dict


def aggregate_replica_results(replicas: List, scenario: str) -> Dict:
    """
    聚合所有replica的结果

    参数:
        replicas: MultiTypeLLMSimulator实例列表
        scenario: 'segregated' 或 'mixed'

    返回:
        聚合后的结果字典
    """
    num_replicas = len(replicas)
    per_replica_results = []

    # 收集每个replica的结果
    for idx, replica in enumerate(replicas):
        hist = replica.get_history()

        # 总完成数
        completions = sum([sum(c['completions']) for c in hist['completions']])

        # 总时间
        total_time = hist['batch_info'][-1]['time'] if hist['batch_info'] else 0

        # 收敛性检查
        converged = check_convergence(hist)

        per_replica_results.append({
            'replica_id': idx,
            'completions': completions,
            'total_time': total_time,
            'converged': converged
        })

    # 聚合指标
    total_completions = sum(r['completions'] for r in per_replica_results)
    max_time = max(r['total_time'] for r in per_replica_results)
    all_converged = all(r['converged'] for r in per_replica_results)

    # Throughput: requests per time unit
    total_throughput = total_completions / max_time if max_time > 0 else 0.0

    # 平均延迟（近似）
    avg_latency = max_time / total_completions if total_completions > 0 else float('inf')

    # 负载均衡
    completion_counts = [r['completions'] for r in per_replica_results]
    load_balance_std = pd.Series(completion_counts).std()

    return {
        'scenario': scenario,
        'total_throughput': float(total_throughput),
        'avg_latency': float(avg_latency),
        'total_time': float(max_time),
        'per_replica': [{
            'replica_id': int(r['replica_id']),
            'completions': float(r['completions']),
            'total_time': float(r['total_time']),
            'converged': bool(r['converged'])
        } for r in per_replica_results],
        'all_converged': bool(all_converged),
        'load_balance_std': float(load_balance_std) if pd.notna(load_balance_std) else 0.0
    }


def check_convergence(history: Dict, window_size: int = 50, threshold: float = 0.01) -> bool:
    """
    检查是否收敛（简化版本）

    参数:
        history: 模拟历史数据
        window_size: 最后多少个batch用于检查
        threshold: 方差阈值

    返回:
        True if converged, False otherwise
    """
    states = history['X_prime']
    if len(states) < window_size:
        return False

    # 简化：检查最后window_size个batch的状态方差
    final_states = states[-window_size:]

    # 计算所有(length, type)组合的count方差
    all_counts = []
    for state_record in final_states:
        state_dict = state_record['state']
        for length, type_counts in state_dict.items():
            all_counts.extend(type_counts)

    if not all_counts:
        return False

    variance = pd.Series(all_counts).var()
    return variance < threshold
