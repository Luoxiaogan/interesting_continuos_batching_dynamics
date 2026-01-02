"""
实验运行器
遍历 admission_threshold，收集 throughput 和 latency 数据
支持自适应步长
"""

import os
import json
import csv
from datetime import datetime
from multiprocessing import Pool, cpu_count
import numpy as np
from llm_scheduler_simulator_real import LLMSchedulerSimulator
from stability_detector import StabilityDetector


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_equilibrium_params(l0, l1, B, b0, b1, target_ratio=0.95):
    """
    根据稳态公式计算 lambda_rate 和 X0

    模拟器统一使用 active_memory = Σ(l0+i+1) × X[i]
    - eviction 判断：active_memory <= B
    - 服务时间：s = b0 + b1 × active_memory

    稳态条件：
    - X* = λ × s*
    - s* = b0 + b1 × γ × X*
    - M* = γ × X* = target_ratio × B

    其中 γ = Σ(l0+i+1) for i in [0, l1-1]

    解得：
    - X* = target_ratio × B / γ
    - λ = target_ratio × B / (γ × (b0 + b1 × target_ratio × B))
    - s* = b0 + b1 × target_ratio × B
    """
    # γ = Σ(l0 + i + 1) for i in [0, l1-1]
    gamma = l1 * (l0 + 1) + l1 * (l1 - 1) / 2

    # 目标 M* = target_ratio × B
    M_star = target_ratio * B

    # X* = M* / γ
    x_star = M_star / gamma

    # s* = b0 + b1 × M*
    s_star = b0 + b1 * M_star

    # λ = M* / (γ × s*)
    lambda_rate = M_star / (gamma * s_star)

    X0 = [x_star] * l1

    return {
        'lambda_rate': lambda_rate,
        'X0': X0,
        'gamma': gamma,
        'M_star': M_star,
        's_star': s_star,
        'actual_ratio': M_star / B
    }


def auto_configure(config, target_ratio=0.95):
    """
    自动配置 config，设置 lambda_rate 和 X0 使得稳态 memory 为 B 的 target_ratio

    参数:
        config: 配置字典
        target_ratio: M*/B 的目标比例 (默认 0.95)

    返回:
        config: 修改后的配置字典（原地修改）
    """
    sim_params = config['simulation_params']

    # 计算稳态参数
    eq_params = compute_equilibrium_params(
        l0=sim_params['l0'],
        l1=sim_params['l1'],
        B=sim_params['B'],
        b0=sim_params['b0'],
        b1=sim_params['b1'],
        target_ratio=target_ratio
    )

    # 更新配置
    sim_params['lambda_rate'] = eq_params['lambda_rate']
    config['initial_state']['X0'] = eq_params['X0']
    config['initial_state']['Qe0'] = 0  # 稳态时外部队列为 0

    # 打印信息
    print("=" * 60)
    print("Auto-configured equilibrium parameters:")
    print("=" * 60)
    print(f"  Target M*/B ratio: {target_ratio:.2%}")
    print(f"  γ (memory coef): {eq_params['gamma']:.4f}")
    print(f"  λ (lambda_rate): {eq_params['lambda_rate']:.6f}")
    print(f"  s* (service time): {eq_params['s_star']:.6f}")
    print(f"  X* (per stage): {eq_params['X0'][0]:.6f}")
    print(f"  M* (memory): {eq_params['M_star']:.4f}")
    print(f"  B (capacity): {sim_params['B']}")
    print("=" * 60)

    return config


def create_output_dir(base_dir="output"):
    """创建输出目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"experiment_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def run_single_experiment(sim_params, init_state, stability_params, admission_threshold, max_steps=10000):
    """
    运行单次实验（固定 admission_threshold）

    返回:
        (throughput, latency, steps, admission_history, batch_history)
    """
    # 创建模拟器
    sim = LLMSchedulerSimulator(
        l0=sim_params['l0'],
        l1=sim_params['l1'],
        B=sim_params['B'],
        X0=init_state['X0'].copy(),
        Qe0=init_state['Qe0'],
        lambda_rate=sim_params['lambda_rate'],
        b0=sim_params['b0'],
        b1=sim_params['b1'],
        if_float=sim_params['if_float'],
        admission_threshold=admission_threshold,
        admission_upper_bound=sim_params.get('admission_upper_bound', None),  # 大S
        verbose=False  # 批量实验时关闭打印
    )

    # 创建稳定性检测器
    detector = StabilityDetector(
        window_size=stability_params['window_size'],
        check_interval=stability_params['check_interval'],
        epsilon=stability_params['epsilon'],
        min_steps=stability_params.get('min_steps', 100)
    )

    # 运行模拟
    min_steps = stability_params.get('min_steps', 100)
    for step in range(max_steps):
        try:
            sim.update()
        except Exception as e:
            # 队列耗尽或其他错误
            print(f"    Simulation stopped at step {step}: {e}")
            break

        throughput = sim.get_current_throughput()
        latency = sim.get_current_avg_latency()
        detector.update(throughput, latency)

        # 必须至少运行 min_steps 步
        if step + 1 >= min_steps and detector.should_check() and detector.is_stable():
            throughput, latency = detector.get_stable_values()
            return throughput, latency, step + 1, sim.admission_history, sim.batch_history

    # 达到 max_steps，返回最后的值
    return sim.get_current_throughput(), sim.get_current_avg_latency(), max_steps, sim.admission_history, sim.batch_history


def _run_single_experiment_wrapper(args):
    """multiprocessing 的包装函数"""
    s, sim_params, init_state, stability_params, max_steps = args
    throughput, latency, steps, admission_history, batch_history = run_single_experiment(
        sim_params, init_state, stability_params, s, max_steps
    )
    avg_admission = np.mean(admission_history) if admission_history else 0
    std_admission = np.std(admission_history) if admission_history else 0
    return {
        'admission_threshold': s,
        'throughput': throughput,
        'latency': latency,
        'steps_to_converge': steps,
        'avg_admission': avg_admission,
        'std_admission': std_admission,
        'batch_history': batch_history
    }


def parallel_sweep(sim_params, init_state, stability_params, exp_params, n_workers=None):
    """
    并行扫描 admission_threshold

    参数:
        n_workers: 并行进程数，默认为 CPU 核心数
    """
    s_min = exp_params['s_min']
    s_max = exp_params['s_max']
    initial_step = exp_params.get('initial_step', 5)
    max_steps = exp_params.get('max_steps', 5000)

    if n_workers is None:
        n_workers = cpu_count()

    # 生成所有 s 值
    s_values = []
    s = s_min
    while s <= s_max:
        s_values.append(s)
        s += initial_step
    if s_max not in s_values:
        s_values.append(s_max)

    print(f"\n并行扫描 {len(s_values)} 个 s 值，使用 {n_workers} 个进程...")

    # 构建参数列表
    args_list = [(s, sim_params, init_state, stability_params, max_steps) for s in s_values]

    # 并行执行
    with Pool(n_workers) as pool:
        results_list = pool.map(_run_single_experiment_wrapper, args_list)

    # 按 s 排序
    results_list.sort(key=lambda x: x['admission_threshold'])

    # 打印结果
    for r in results_list:
        print(f"  s={r['admission_threshold']:.4f}: throughput={r['throughput']:.4f}, "
              f"latency={r['latency']:.4f}, avg_adm={r['avg_admission']:.4f}, steps={r['steps_to_converge']}")

    return results_list


def adaptive_sweep(sim_params, init_state, stability_params, exp_params):
    """
    自适应步长扫描 admission_threshold

    算法:
    1. 先用 initial_step 粗扫描整个范围
    2. 计算相邻点的 throughput 变化率
    3. 在变化率 > threshold 的区域，用更小步长重新扫描
    4. 合并结果并排序

    返回:
        results: [{admission_threshold, throughput, latency, steps_to_converge}, ...]
    """
    s_min = exp_params['s_min']
    s_max = exp_params['s_max']
    initial_step = exp_params.get('initial_step', 5)
    refine_step = exp_params.get('refine_step', 1)
    refine_threshold = exp_params.get('refine_threshold', 0.1)
    max_steps = exp_params.get('max_steps', 5000)

    results = {}  # {s: {throughput, latency, steps}}
    baseline_throughput = None

    # 第一阶段：粗扫描
    print("\n" + "=" * 60)
    print("Phase 1: Coarse scan")
    print("=" * 60)

    # 支持 float 类型的步长
    coarse_points = []
    s = s_min
    while s <= s_max:
        coarse_points.append(s)
        s += initial_step
    if s_max not in coarse_points:
        coarse_points.append(s_max)

    for s in coarse_points:
        print(f"  Running s = {s}...", end=" ")
        throughput, latency, steps, admission_history, batch_history = run_single_experiment(
            sim_params, init_state, stability_params, s, max_steps
        )
        avg_admission = np.mean(admission_history) if admission_history else 0
        std_admission = np.std(admission_history) if admission_history else 0
        results[s] = {
            'admission_threshold': s,
            'throughput': throughput,
            'latency': latency,
            'steps_to_converge': steps,
            'avg_admission': avg_admission,
            'std_admission': std_admission,
            'batch_history': batch_history
        }
        print(f"throughput={throughput:.4f}, latency={latency:.4f}, avg_adm={avg_admission:.4f}, steps={steps}")

        if s == s_min:
            baseline_throughput = throughput

        # 早停检测
        if baseline_throughput and throughput < baseline_throughput * 0.01:
            print(f"  Throughput dropped below 1% of baseline. Stopping coarse scan.")
            break

    # 第二阶段：识别需要加密的区域
    print("\n" + "=" * 60)
    print("Phase 2: Identify regions for refinement")
    print("=" * 60)

    sorted_s = sorted(results.keys())
    regions_to_refine = []

    for i in range(len(sorted_s) - 1):
        s1, s2 = sorted_s[i], sorted_s[i + 1]
        t1, t2 = results[s1]['throughput'], results[s2]['throughput']

        if t1 > 0:
            change_rate = abs(t2 - t1) / t1
        else:
            change_rate = float('inf') if t2 != t1 else 0

        if change_rate > refine_threshold:
            regions_to_refine.append((s1, s2))
            print(f"  Region [{s1}, {s2}]: change_rate = {change_rate:.4f} > {refine_threshold}")

    # 第三阶段：加密扫描
    if regions_to_refine:
        print("\n" + "=" * 60)
        print("Phase 3: Refined scan")
        print("=" * 60)

        for s_start, s_end in regions_to_refine:
            refine_points = []
            s = s_start + refine_step
            while s < s_end:
                # 使用近似比较（float 精度问题）
                if not any(abs(s - existing) < 1e-9 for existing in results.keys()):
                    refine_points.append(s)
                s += refine_step

            for s in refine_points:
                print(f"  Running s = {s}...", end=" ")
                throughput, latency, steps, admission_history, batch_history = run_single_experiment(
                    sim_params, init_state, stability_params, s, max_steps
                )
                avg_admission = np.mean(admission_history) if admission_history else 0
                std_admission = np.std(admission_history) if admission_history else 0
                results[s] = {
                    'admission_threshold': s,
                    'throughput': throughput,
                    'latency': latency,
                    'steps_to_converge': steps,
                    'avg_admission': avg_admission,
                    'std_admission': std_admission,
                    'batch_history': batch_history
                }
                print(f"throughput={throughput:.4f}, latency={latency:.4f}, avg_adm={avg_admission:.4f}, steps={steps}")

    # 返回排序后的结果
    return [results[s] for s in sorted(results.keys())]


def run_experiment_sweep(config, use_adaptive=True, parallel=False, n_workers=None):
    """
    扫描 admission_threshold 范围

    参数:
        config: 配置字典
        use_adaptive: 是否使用自适应步长
        parallel: 是否并行执行
        n_workers: 并行进程数

    返回:
        results: [{admission_threshold, throughput, latency, steps_to_converge}, ...]
    """
    sim_params = config['simulation_params']
    init_state = config['initial_state']
    exp_params = config['experiment_params']
    stability_params = config['stability_params']

    if parallel:
        return parallel_sweep(sim_params, init_state, stability_params, exp_params, n_workers)
    elif use_adaptive:
        return adaptive_sweep(sim_params, init_state, stability_params, exp_params)
    else:
        # 简单的均匀步长扫描
        s_min = exp_params['s_min']
        s_max = exp_params['s_max']
        s_step = exp_params.get('initial_step', 1)
        max_steps = exp_params.get('max_steps', 5000)

        results = []
        baseline_throughput = None

        for s in range(s_min, s_max + 1, s_step):
            print(f"Running s = {s}...", end=" ")
            throughput, latency, steps = run_single_experiment(
                sim_params, init_state, stability_params, s, max_steps
            )
            results.append({
                'admission_threshold': s,
                'throughput': throughput,
                'latency': latency,
                'steps_to_converge': steps
            })
            print(f"throughput={throughput:.4f}, latency={latency:.4f}, steps={steps}")

            if s == s_min:
                baseline_throughput = throughput

            if baseline_throughput and throughput < baseline_throughput * 0.01:
                print(f"Throughput dropped below 1% of baseline. Stopping.")
                break

        return results


def run_3d_experiment(config):
    """
    变化 arrival rate，生成 3D 数据

    返回:
        results_3d: {lambda: [{s, throughput, latency}, ...], ...}
    """
    vis_params = config.get('visualization_params', {})
    arrival_rates = vis_params.get('arrival_rates_for_3d', [])

    if not arrival_rates:
        return {}

    results_3d = {}

    for lambda_rate in arrival_rates:
        print("\n" + "#" * 60)
        print(f"Running 3D experiment with lambda_rate = {lambda_rate}")
        print("#" * 60)

        # 修改 config 中的 arrival rate
        config_copy = json.loads(json.dumps(config))
        config_copy['simulation_params']['lambda_rate'] = lambda_rate

        results = run_experiment_sweep(config_copy, use_adaptive=True)
        results_3d[lambda_rate] = results

    return results_3d


def save_results_to_csv(output_dir, results, filename="results.csv"):
    """保存 2D 实验结果到 CSV（高精度）"""
    csv_path = os.path.join(output_dir, filename)

    fieldnames = ['admission_threshold', 'throughput', 'latency', 'steps_to_converge',
                  'avg_admission', 'std_admission']

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for r in results:
            writer.writerow([
                f"{r['admission_threshold']:.10g}",
                f"{r['throughput']:.15g}",
                f"{r['latency']:.15g}",
                r['steps_to_converge'],
                f"{r['avg_admission']:.15g}",
                f"{r['std_admission']:.15g}"
            ])

    print(f"Results saved to: {csv_path}")
    return csv_path


def save_batch_history_csv(output_dir, results):
    """为每个 s 保存 batch_history 到单独的 CSV"""
    batch_dir = os.path.join(output_dir, "batch_history")
    os.makedirs(batch_dir, exist_ok=True)

    for r in results:
        s = r['admission_threshold']
        batch_history = r.get('batch_history', [])
        if not batch_history:
            continue

        csv_path = os.path.join(batch_dir, f"s_{s:.4g}.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['batch_idx', 'T', 'throughput', 'latency', 'cumulative_eviction',
                           'admission', 'queue_length', 'tokens_before_admission', 'tokens_after_admission'])
            for b in batch_history:
                writer.writerow([
                    b['batch_idx'],
                    f"{b['T']:.15g}",
                    f"{b['throughput']:.15g}",
                    f"{b['latency']:.15g}",
                    f"{b['cumulative_eviction']:.15g}",
                    f"{b['admission']:.15g}",
                    f"{b['queue_length']:.15g}",
                    f"{b['tokens_before_admission']:.15g}",
                    f"{b['tokens_after_admission']:.15g}"
                ])

    print(f"Batch history saved to: {batch_dir}")
    return batch_dir


def save_3d_results_to_csv(output_dir, results_3d, filename="results_3d.csv"):
    """保存 3D 实验结果到 CSV"""
    csv_path = os.path.join(output_dir, filename)

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'lambda_rate', 'admission_threshold', 'throughput', 'latency', 'steps_to_converge'
        ])
        writer.writeheader()

        for lambda_rate, results in results_3d.items():
            for r in results:
                row = {'lambda_rate': lambda_rate}
                row.update(r)
                writer.writerow(row)

    print(f"3D results saved to: {csv_path}")
    return csv_path


def main(config_path="config.json", run_3d=False, parallel=False, n_workers=None,
         auto_config=False, target_ratio=0.95):
    """主函数"""
    # 加载配置
    config = load_config(config_path)

    # 自动配置稳态参数
    if auto_config:
        config = auto_configure(config, target_ratio=target_ratio)

    # 创建输出目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = create_output_dir(os.path.join(script_dir, "output"))

    # 保存配置副本
    config_copy_path = os.path.join(output_dir, "config.json")
    with open(config_copy_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 运行 2D 实验
    print("\n" + "=" * 60)
    print("Starting 2D Experiment (s vs throughput/latency)")
    print("=" * 60)
    results_2d = run_experiment_sweep(config, use_adaptive=not parallel, parallel=parallel, n_workers=n_workers)
    save_results_to_csv(output_dir, results_2d)
    save_batch_history_csv(output_dir, results_2d)

    # 运行 3D 实验（如果配置了且用户请求）
    if run_3d and config.get('visualization_params', {}).get('arrival_rates_for_3d'):
        print("\n" + "=" * 60)
        print("Starting 3D Experiment (varying lambda)")
        print("=" * 60)
        results_3d = run_3d_experiment(config)
        save_3d_results_to_csv(output_dir, results_3d)

    # 生成可视化
    try:
        from visualization import generate_2d_plot, generate_3d_plot, generate_timeseries_plot
        generate_2d_plot(output_dir)
        # 传入 admission_upper_bound (大S) 用于标注
        admission_upper_bound = config['simulation_params'].get('admission_upper_bound')
        generate_timeseries_plot(output_dir, results_2d, admission_upper_bound=admission_upper_bound)
        if run_3d and config.get('visualization_params', {}).get('arrival_rates_for_3d'):
            generate_3d_plot(output_dir)
    except ImportError:
        print("Warning: visualization module not found. Skipping plot generation.")

    print(f"\nExperiment complete. All outputs saved to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run admission threshold experiments')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--run-3d', action='store_true', help='Run 3D experiment with varying arrival rates')
    parser.add_argument('--parallel', action='store_true', help='Run experiments in parallel')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--auto-config', action='store_true',
                        help='Auto-configure lambda_rate and X0 for equilibrium')
    parser.add_argument('--target-ratio', type=float, default=0.95,
                        help='Target M*/B ratio for auto-config (default: 0.95)')
    args = parser.parse_args()

    # 切换到脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    main(args.config, run_3d=args.run_3d, parallel=args.parallel, n_workers=args.workers,
         auto_config=args.auto_config, target_ratio=args.target_ratio)
