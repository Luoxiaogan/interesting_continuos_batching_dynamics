"""
模拟运行脚本
负责运行模拟、保存数据到CSV文件
"""

import os
import sys
import json
import argparse
from datetime import datetime
import pandas as pd
from multi_type_simulator import MultiTypeLLMSimulator


def create_output_dir(base_dir="output"):
    """
    根据时间戳创建输出目录

    返回:
        str: 创建的目录路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"sim_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"创建输出目录: {output_dir}")
    return output_dir


def save_config(output_dir, config):
    """
    保存模拟配置到JSON文件

    参数:
        output_dir: 输出目录
        config: 配置字典
    """
    config_file = os.path.join(output_dir, "config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"配置已保存: {config_file}")


def save_history_to_csv(output_dir, history, num_types, length_range):
    """
    将模拟历史保存为CSV文件

    参数:
        output_dir: 输出目录
        history: 模拟器的历史数据
        num_types: 类型数量
        length_range: 长度范围
    """

    # 1. 保存X_prime状态（核心数据）
    print("\n保存X_prime状态数据...")
    x_prime_records = []
    for record in history['X_prime']:
        batch = record['batch']
        time = record['time']
        state = record['state']

        # 为每个(length, type)组合创建一行
        for length in length_range:
            for type_idx in range(num_types):
                x_prime_records.append({
                    'batch': batch,
                    'time': time,
                    'length': length,
                    'type': type_idx,
                    'count': state[length][type_idx]
                })

    df_x_prime = pd.DataFrame(x_prime_records)
    x_prime_file = os.path.join(output_dir, "x_prime_states.csv")
    df_x_prime.to_csv(x_prime_file, index=False)
    print(f"  ✓ X_prime状态已保存: {x_prime_file} ({len(x_prime_records)} 条记录)")

    # 2. 保存admissions数据
    print("保存admissions数据...")
    admission_records = []
    for record in history['admissions']:
        batch = record['batch']
        time = record['time']
        admissions = record['admissions']

        for type_idx, count in admissions.items():
            admission_records.append({
                'batch': batch,
                'time': time,
                'type': type_idx,
                'admitted_count': count
            })

    if admission_records:
        df_admissions = pd.DataFrame(admission_records)
        admissions_file = os.path.join(output_dir, "admissions.csv")
        df_admissions.to_csv(admissions_file, index=False)
        print(f"  ✓ Admissions已保存: {admissions_file} ({len(admission_records)} 条记录)")

    # 3. 保存evictions数据
    print("保存evictions数据...")
    eviction_records = []
    for record in history['evictions']:
        batch = record['batch']
        time = record['time']
        evictions = record['evictions']

        for type_idx, evict_list in evictions.items():
            for length, amount in evict_list:
                eviction_records.append({
                    'batch': batch,
                    'time': time,
                    'type': type_idx,
                    'length': length,
                    'evicted_count': amount
                })

    if eviction_records:
        df_evictions = pd.DataFrame(eviction_records)
        evictions_file = os.path.join(output_dir, "evictions.csv")
        df_evictions.to_csv(evictions_file, index=False)
        print(f"  ✓ Evictions已保存: {evictions_file} ({len(eviction_records)} 条记录)")

    # 4. 保存completions数据
    print("保存completions数据...")
    completion_records = []
    for record in history['completions']:
        batch = record['batch']
        time = record['time']
        completions = record['completions']

        for type_idx, count in enumerate(completions):
            if count > 0:
                completion_records.append({
                    'batch': batch,
                    'time': time,
                    'type': type_idx,
                    'completed_count': count
                })

    if completion_records:
        df_completions = pd.DataFrame(completion_records)
        completions_file = os.path.join(output_dir, "completions.csv")
        df_completions.to_csv(completions_file, index=False)
        print(f"  ✓ Completions已保存: {completions_file} ({len(completion_records)} 条记录)")

    # 5. 保存batch_info数据
    print("保存batch_info数据...")
    batch_info_records = []
    for record in history['batch_info']:
        batch_info_records.append({
            'batch': record['batch'],
            'time': record['time'],
            'service_time': record['service_time'],
            'batch_size': record['batch_size']
        })

    df_batch_info = pd.DataFrame(batch_info_records)
    batch_info_file = os.path.join(output_dir, "batch_info.csv")
    df_batch_info.to_csv(batch_info_file, index=False)
    print(f"  ✓ Batch info已保存: {batch_info_file} ({len(batch_info_records)} 条记录)")


def save_summary(output_dir, sim):
    """
    保存模拟总结到文本文件

    参数:
        output_dir: 输出目录
        sim: 模拟器对象
    """
    summary_file = os.path.join(output_dir, "summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("模拟总结 / SIMULATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"总时间: {sim.T:.3f}\n")
        f.write(f"总批次数: {sim.n}\n")
        f.write(f"总完成数: {sim.total_completions:.2f}\n")
        f.write(f"总体吞吐量: {sim.total_completions/sim.T:.3f}\n\n")

        f.write("各类型完成情况:\n")
        for type_idx in range(sim.num_types):
            l0, l1 = sim.request_types[type_idx]
            throughput = sim.completions[type_idx] / sim.T
            f.write(f"  Type {type_idx} (l0={l0}, l1={l1}): "
                   f"{sim.completions[type_idx]:.2f} completions, "
                   f"throughput={throughput:.3f}\n")

        f.write(f"\n公平性分析:\n")
        f.write(f"  到达率: λ = {sim.arrival_rates}, sum={sum(sim.arrival_rates):.3f}\n")

        if sim.total_completions > 0:
            f.write(f"  吞吐量比例:\n")
            for type_idx in range(sim.num_types):
                actual_ratio = sim.completions[type_idx] / sim.total_completions
                expected_ratio = sim.arrival_rates[type_idx] / sum(sim.arrival_rates)
                deviation = (actual_ratio - expected_ratio) * 100
                f.write(f"    Type {type_idx}: "
                       f"actual={actual_ratio:.3f}, "
                       f"expected={expected_ratio:.3f}, "
                       f"deviation={deviation:+.1f}%\n")

    print(f"\n总结已保存: {summary_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行多类型LLM调度模拟')

    # 请求类型参数
    parser.add_argument('--l0_0', type=int, required=True, help='Type 0的初始长度')
    parser.add_argument('--l1_0', type=int, required=True, help='Type 0的生成长度')
    parser.add_argument('--l0_1', type=int, required=True, help='Type 1的初始长度')
    parser.add_argument('--l1_1', type=int, required=True, help='Type 1的生成长度')

    # 到达率参数
    parser.add_argument('--lambda_0', type=float, required=True, help='Type 0的到达率')
    parser.add_argument('--lambda_1', type=float, required=True, help='Type 1的到达率')

    # 系统参数
    parser.add_argument('--B', type=int, required=True, help='GPU容量限制')
    parser.add_argument('--b0', type=float, default=0.1, help='服务时间基础参数')
    parser.add_argument('--b1', type=float, default=0.01, help='服务时间系数参数')

    # 模拟参数
    parser.add_argument('--steps', type=int, required=True, help='模拟步数')
    parser.add_argument('--x0', type=str, default='{}', help='初始状态JSON字符串')

    # 输出参数
    parser.add_argument('--output_base', type=str, default='output', help='输出基础目录')
    parser.add_argument('--verbose', action='store_true', help='是否打印详细日志')

    # 可视化参数
    parser.add_argument('--start_index', type=int, default=0, help='可视化开始的批次索引')
    parser.add_argument('--jump', type=int, default=1, help='差异计算的步长')
    parser.add_argument('--no_plot', action='store_true', help='禁用自动生成图表')

    args = parser.parse_args()

    # 解析初始状态
    X0 = json.loads(args.x0)
    # 将字符串键转换为整数键，值转换为列表
    X0 = {int(k): v for k, v in X0.items()}

    # 配置参数
    request_types = [
        (args.l0_0, args.l1_0),
        (args.l0_1, args.l1_1)
    ]
    arrival_rates = [args.lambda_0, args.lambda_1]

    config = {
        'request_types': request_types,
        'arrival_rates': arrival_rates,
        'B': args.B,
        'b0': args.b0,
        'b1': args.b1,
        'steps': args.steps,
        'X0': X0
    }

    print("=" * 80)
    print("多类型LLM调度模拟器")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  Request types: {request_types}")
    print(f"  Arrival rates: {arrival_rates}")
    print(f"  GPU capacity B: {args.B}")
    print(f"  Service time: s(n) = {args.b0} + {args.b1} * Z(n)")
    print(f"  Simulation steps: {args.steps}")
    print(f"  Initial state: {X0 if X0 else 'Empty'}")
    print("=" * 80)

    # 创建输出目录
    output_dir = create_output_dir(args.output_base)

    # 保存配置
    save_config(output_dir, config)

    # 创建模拟器
    print("\n初始化模拟器...")
    sim = MultiTypeLLMSimulator(
        request_type_list=request_types,
        B=args.B,
        X0=X0,
        arrival_rates=arrival_rates,
        b0=args.b0,
        b1=args.b1,
        verbose=args.verbose
    )

    # 运行模拟
    print(f"\n开始运行模拟 ({args.steps} 步)...")
    sim.run(args.steps)

    # 保存数据
    print("\n" + "=" * 80)
    print("保存数据到CSV文件")
    print("=" * 80)

    history = sim.get_history()
    save_history_to_csv(output_dir, history, sim.num_types, sim.length_range)

    # 保存总结
    save_summary(output_dir, sim)

    print("\n" + "=" * 80)
    print("模拟完成！")
    print("=" * 80)
    print(f"所有数据已保存到: {output_dir}")
    print(f"  - config.json: 模拟配置")
    print(f"  - x_prime_states.csv: 每批次的状态（核心数据）")
    print(f"  - admissions.csv: 准入记录")
    print(f"  - evictions.csv: 驱逐记录")
    print(f"  - completions.csv: 完成记录")
    print(f"  - batch_info.csv: 批次信息")
    print(f"  - summary.txt: 模拟总结")
    print("=" * 80)

    # 生成可视化图表（如果未禁用）
    if not args.no_plot:
        try:
            from visualization import generate_all_plots
            generate_all_plots(output_dir, start_index=args.start_index, jump=args.jump)
        except ImportError:
            print("\n警告: 无法导入visualization模块，跳过图表生成")
        except Exception as e:
            print(f"\n警告: 生成图表时出错: {e}")

    return output_dir


if __name__ == "__main__":
    output_dir = main()
