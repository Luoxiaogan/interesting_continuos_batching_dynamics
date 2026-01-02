#!/usr/bin/env python3
"""Multi-Replica Mixing结果可视化脚本 (简单脚本,仅调用库函数)"""
import json, matplotlib.pyplot as plt

# 加载结果
with open('experiments/multi_replica_mixing_results.json') as f:
    results = json.load(f)

# 提取数据
exp_names, throughput_seg, throughput_mix, latency_seg, latency_mix = [], [], [], [], []
for result in results:
    exp_names.append(f"{result['experiment']}\n(gcd={result['gcd']})")
    throughput_seg.append(result['segregated']['total_throughput'])
    throughput_mix.append(result['mixed']['total_throughput'])
    latency_seg.append(result['segregated']['avg_latency'])
    latency_mix.append(result['mixed']['avg_latency'])

# 绘制对比图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x, width = range(len(exp_names)), 0.35

# Throughput
axes[0].bar([i - width/2 for i in x], throughput_seg, width, label='Segregated', alpha=0.8, color='salmon')
axes[0].bar([i + width/2 for i in x], throughput_mix, width, label='Mixed', alpha=0.8, color='skyblue')
axes[0].set_xlabel('Experiment')
axes[0].set_ylabel('Total Throughput')
axes[0].set_title('Throughput Comparison')
axes[0].set_xticks(x)
axes[0].set_xticklabels(exp_names)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Latency
axes[1].bar([i - width/2 for i in x], latency_seg, width, label='Segregated', alpha=0.8, color='salmon')
axes[1].bar([i + width/2 for i in x], latency_mix, width, label='Mixed', alpha=0.8, color='skyblue')
axes[1].set_xlabel('Experiment')
axes[1].set_ylabel('Average Latency (lower is better)')
axes[1].set_title('Latency Comparison')
axes[1].set_xticks(x)
axes[1].set_xticklabels(exp_names)
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('experiments/mixing_comparison.png', dpi=300)
print("✅ 图表已保存到: experiments/mixing_comparison.png")
