"""
更新 config.json 的稳态参数
根据目标 M*/B 比例自动计算 lambda_rate 和 X0
python3 update_config.py --target-ratio 0.95
"""

import json
import argparse
import os


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
    # = l1 × (l0 + 1) + l1 × (l1 - 1) / 2
    gamma = l1 * (l0 + 1) + l1 * (l1 - 1) / 2

    # 目标 M* = target_ratio × B
    M_star = target_ratio * B

    # X* = M* / γ
    x_star = M_star / gamma

    # s* = b0 + b1 × M*
    s_star = b0 + b1 * M_star

    # λ = M* / (γ × s*) = M* / (γ × (b0 + b1 × M*))
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


def update_config(config_path="config.json", target_ratio=0.95):
    """
    更新 config.json 中的 lambda_rate 和 X0

    参数:
        config_path: 配置文件路径
        target_ratio: 目标 M*/B 比例 (默认 0.95)
    """
    # 读取配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

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

    # 打印信息
    print("=" * 60)
    print("Equilibrium parameters:")
    print("=" * 60)
    print(f"  Target M*/B ratio: {target_ratio:.2%}")
    print(f"  γ (memory coef): {eq_params['gamma']:.4f}")
    print(f"  λ (lambda_rate): {eq_params['lambda_rate']:.6f}")
    print(f"  s* (service time): {eq_params['s_star']:.6f}")
    print(f"  X* (per stage): {eq_params['X0'][0]:.6f}")
    print(f"  M* (memory): {eq_params['M_star']:.4f}")
    print(f"  B (capacity): {sim_params['B']}")
    print("=" * 60)

    # 更新配置
    old_lambda = sim_params.get('lambda_rate', 'N/A')
    old_X0 = config['initial_state'].get('X0', 'N/A')
    old_s_max = config['experiment_params'].get('s_max', 'N/A')
    old_S = sim_params.get('admission_upper_bound', 'N/A')

    sim_params['lambda_rate'] = eq_params['lambda_rate']
    config['initial_state']['X0'] = eq_params['X0']
    config['initial_state']['Qe0'] = 0

    # 大S (admission_upper_bound) = B / γ
    # 这是当 Memory = B（满载）时的稳态 n*
    # 注意：X* = target_ratio × B / γ 是 target_ratio 负载下的稳态
    #       S = B / γ 是 100% 负载下的稳态，作为 admission 上界
    gamma = eq_params['gamma']
    B = sim_params['B']
    admission_upper_bound_S = B / gamma
    sim_params['admission_upper_bound'] = admission_upper_bound_S

    # s_max 应小于满载稳态 S = B/γ（否则 Qe + Qr 永远不会达到阈值）
    new_s_max = int(admission_upper_bound_S)
    config['experiment_params']['s_max'] = new_s_max

    print(f"\nUpdating {config_path}:")
    print(f"  lambda_rate: {old_lambda} -> {eq_params['lambda_rate']:.6f}")
    print(f"  X0: {old_X0} -> {[round(x, 6) for x in eq_params['X0']]}")
    print(f"  Qe0: -> 0")
    print(f"  s_max: {old_s_max} -> {new_s_max} (must be < S = {admission_upper_bound_S:.4f})")
    print(f"  S (admission_upper_bound): {old_S} -> {admission_upper_bound_S:.6f} (= B/γ, full capacity)")

    # 写回配置
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\nConfig updated successfully!")
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Update config.json with equilibrium parameters')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--target-ratio', type=float, default=0.95,
                        help='Target M*/B ratio (default: 0.95)')
    args = parser.parse_args()

    # 切换到脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    update_config(args.config, args.target_ratio)
