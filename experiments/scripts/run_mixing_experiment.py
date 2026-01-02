#!/usr/bin/env python3
"""
Multi-Replica Request Mixing Experiment

Documentation:
    Interface: experiments/README.md
    Config: experiments/exp_multi_replica_mixing_config.json

Purpose:
    验证heterogeneous request mixing对multi-replica LLM serving系统的影响。
    对比segregated routing (按type分组到不同replicas) vs mixed routing
    (所有types均匀分配到所有replicas) 的throughput和latency。

Experimental Design:
    - 4种request types分为2组，每组具有non-coprime的GCD特性
    - Group 1: (l0=4,l1=8), (l0=4,l1=16), gcd(8,16)=8
    - Group 2: (l0=3,l1=5), (l0=3,l1=15), gcd(5,15)=5
    - Segregated: Group1→Replica0, Group2→Replica1
    - Mixed: 全部4种types均分到2个replicas
    - GPU容量: B=500 tokens, 模拟步数: 1000 batches

Key Metrics:
    - Total throughput (requests/time)
    - Average latency (time/request)
    - Load balance (std of per-replica completions)
    - Convergence detection

Output:
    - experiments/multi_replica_mixing_results.json
    - experiments/mixing_results/performance_comparison.png
    - experiments/mixing_results/{segregated,mixed}/replica_*_gpu_state.png

Dependencies:
    - multi_type_simulator: Core LLM scheduling simulator
    - multi_replica_wrapper: Multi-replica orchestration
    - replica_aggregator: Result aggregation and metrics
    - visualization: GPU state and performance plots
"""
import sys, json
from math import gcd
sys.path.insert(0, 'new_project_for_multi_type')
from multi_type_simulator import MultiTypeLLMSimulator
from multi_replica_wrapper import MultiReplicaSimulator
from replica_aggregator import aggregate_replica_results
from visualization import plot_multi_replica_gpu_states, plot_performance_comparison, plot_batch_composition_comparison, plot_stage_distribution_comparison, plot_stage_distribution_stability

# 实验配置 - 4种types分2组 (更小的requests以适应B=500)
experiment = {'name': '4 Types: Group1(4,8),(4,16) vs Group2(3,5),(3,15)',
             'group1': [(4, 8), (4, 16)], 'group2': [(3, 5), (3, 15)],
             'rates': [1.0, 1.0, 1.0, 1.0], 'gcd_g1': gcd(8, 16), 'gcd_g2': gcd(5, 15)}
B, steps = 500, 1000
all_types = experiment['group1'] + experiment['group2']
print(f"\n{'='*60}\n{experiment['name']}\nGroup1: {experiment['group1']} gcd={experiment['gcd_g1']}")
print(f"Group2: {experiment['group2']} gcd={experiment['gcd_g2']}\n{'='*60}")

# Segregated: Group1→Replica0, Group2→Replica1
print("\n运行 Segregated...")
replicas_seg = []
for i, types in enumerate([experiment['group1'], experiment['group2']]):
    r = MultiTypeLLMSimulator(types, B, {}, experiment['rates'][i*2:(i+1)*2], 0.1, 0.01, False)
    r.run(steps)
    replicas_seg.append(r)
results_seg = aggregate_replica_results(replicas_seg, 'segregated')
simulator_seg = MultiReplicaSimulator(2, B)
simulator_seg.replicas = replicas_seg
plot_multi_replica_gpu_states(simulator_seg.get_replicas(), 'segregated', 'experiments/mixing_results/segregated', 0)

# Mixed: 所有4种types均匀分配
print("运行 Mixed...")
simulator_mix = MultiReplicaSimulator(2, B)
results_mix = simulator_mix.run_scenario('mixed', all_types, experiment['rates'], steps)
plot_multi_replica_gpu_states(simulator_mix.get_replicas(), 'mixed', 'experiments/mixing_results/mixed', 0)

# 对比
improvement = (results_mix['total_throughput'] - results_seg['total_throughput']) / results_seg['total_throughput'] * 100
print(f"\n对比:\n  Segregated: Throughput={results_seg['total_throughput']:.2f}, Latency={results_seg['avg_latency']:.4f}")
print(f"  Mixed:      Throughput={results_mix['total_throughput']:.2f}, Latency={results_mix['avg_latency']:.4f}")
print(f"  Improvement: {improvement:+.2f}%")

# 绘制性能对比图
plot_performance_comparison({'segregated': results_seg, 'mixed': results_mix},
                          'experiments/mixing_results/performance_comparison.png')

# 绘制batch组成对比图
plot_batch_composition_comparison(replicas_seg, simulator_mix.get_replicas(),
                                 'experiments/mixing_results/batch_composition_comparison.png')

# 绘制stage分布对比图
plot_stage_distribution_comparison(replicas_seg, simulator_mix.get_replicas(),
                                  'experiments/mixing_results/stage_distribution_comparison.png')

# 绘制stage分布稳定性分析图
plot_stage_distribution_stability(replicas_seg, simulator_mix.get_replicas(),
                                  'experiments/mixing_results/stage_stability_over_time.png')

# 保存
result = {'experiment': experiment['name'], 'group1': experiment['group1'], 'group2': experiment['group2'],
         'gcd_g1': experiment['gcd_g1'], 'gcd_g2': experiment['gcd_g2'], 'all_types': all_types,
         'segregated': results_seg, 'mixed': results_mix, 'improvement_pct': improvement}
with open('experiments/multi_replica_mixing_results.json', 'w') as f:
    json.dump(result, f, indent=2)
print("\n✅ 实验完成!")
