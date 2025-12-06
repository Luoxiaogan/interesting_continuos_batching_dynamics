"""
测试LLM调度模拟器 - 验证PDF中的均衡状态
"""

from llm_scheduler_simulator import LLMSchedulerSimulator
import matplotlib.pyplot as plt

def test_cyclic_equilibrium():
    """
    测试循环均衡：x(0) = (B/2, 0) 和 x(1) = (0, B/3)
    对于l0=1, l1=2的情况
    """
    print("="*50)
    print("测试循环均衡")
    print("="*50)
    
    l0 = 1
    l1 = 2 
    B = 12  # 使用12便于整除
    X0 = [B//2, 0]  # 初始状态 (B/2, 0)
    Qe0 = 0
    
    # 根据PDF，如果λ > B/(6(b0+b1B))，系统不稳定
    b0 = 0.1
    b1 = 0.01
    critical_lambda = B / (6 * (b0 + b1 * B))
    
    # 使用略低于临界值的λ
    lambda_rate = critical_lambda * 0.8
    
    print(f"参数设置:")
    print(f"  l0={l0}, l1={l1}, B={B}")
    print(f"  初始状态 X(0)={X0}")
    print(f"  临界到达率={critical_lambda:.4f}")
    print(f"  实际到达率={lambda_rate:.4f}\n")
    
    sim = LLMSchedulerSimulator(l0, l1, B, X0, Qe0, lambda_rate, b0, b1)
    sim.run(20)
    
    # 显示状态演变
    print("状态演变:")
    for i in range(min(10, len(sim.history))):
        record = sim.history[i]
        print(f"  n={record['n']}: X={record['X']}, Z={record['Z']:.1f}")
    
    return sim

def test_stable_equilibrium():
    """
    测试稳定均衡：x* = (B/5, B/5)
    对于l0=1, l1=2的情况
    """
    print("\n" + "="*50)
    print("测试稳定均衡")
    print("="*50)
    
    l0 = 1
    l1 = 2
    B = 10
    X0 = [B//5, B//5]  # 初始状态接近均衡点
    Qe0 = 0
    
    # 根据PDF，如果λ > B/(5(b0+b1B))，系统不稳定
    b0 = 0.1
    b1 = 0.01
    critical_lambda = B / (5 * (b0 + b1 * B))
    
    # 使用略低于临界值的λ
    lambda_rate = critical_lambda * 0.8
    
    print(f"参数设置:")
    print(f"  l0={l0}, l1={l1}, B={B}")
    print(f"  初始状态 X(0)={X0}")
    print(f"  临界到达率={critical_lambda:.4f}")
    print(f"  实际到达率={lambda_rate:.4f}\n")
    
    sim = LLMSchedulerSimulator(l0, l1, B, X0, Qe0, lambda_rate, b0, b1)
    sim.run(20)
    
    # 显示状态演变
    print("状态演变:")
    for i in range(min(10, len(sim.history))):
        record = sim.history[i]
        print(f"  n={record['n']}: X={record['X']}, Z={record['Z']:.1f}")
    
    return sim

def test_three_stages():
    """
    测试三阶段系统：l0=1, l1=3
    """
    print("\n" + "="*50)
    print("测试三阶段系统")
    print("="*50)
    
    l0 = 1
    l1 = 3
    B = 12
    X0 = [B//3, 0, 0]  # 初始状态
    Qe0 = 0
    
    # 根据PDF，临界到达率
    b0 = 0.1
    b1 = 0.01
    critical_lambda = B / (12 * (b0 + b1 * B))
    
    lambda_rate = critical_lambda * 0.8
    
    print(f"参数设置:")
    print(f"  l0={l0}, l1={l1}, B={B}")
    print(f"  初始状态 X(0)={X0}")
    print(f"  临界到达率={critical_lambda:.4f}")
    print(f"  实际到达率={lambda_rate:.4f}\n")
    
    sim = LLMSchedulerSimulator(l0, l1, B, X0, Qe0, lambda_rate, b0, b1)
    sim.run(30)
    
    # 显示状态演变
    print("状态演变:")
    for i in range(min(10, len(sim.history))):
        record = sim.history[i]
        print(f"  n={record['n']}: X={record['X']}, Z={record['Z']:.1f}")
    
    return sim

def plot_simulation(sim):
    """绘制模拟结果"""
    history = sim.get_history()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 批大小Z随时间变化
    Z_values = [record['Z'] for record in history]
    axes[0, 0].plot(Z_values, marker='o')
    axes[0, 0].set_title('批大小 Z(n)')
    axes[0, 0].set_xlabel('批次 n')
    axes[0, 0].set_ylabel('Z')
    axes[0, 0].grid(True)
    
    # 队列长度随时间变化
    Qe_values = [record['Qe'] for record in history]
    Qr_values = [record['Qr'] for record in history]
    axes[0, 1].plot(Qe_values, label='外部队列 Qe', marker='s')
    axes[0, 1].plot(Qr_values, label='重排队列 Qr', marker='^')
    axes[0, 1].set_title('队列长度')
    axes[0, 1].set_xlabel('批次 n')
    axes[0, 1].set_ylabel('队列长度')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 各阶段作业数
    l1 = sim.l1
    for i in range(l1):
        Xi_values = [record['X'][i] for record in history]
        axes[1, 0].plot(Xi_values, label=f'X{i}', marker='o')
    axes[1, 0].set_title('各阶段作业数')
    axes[1, 0].set_xlabel('批次 n')
    axes[1, 0].set_ylabel('作业数')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 服务时间
    s_values = [record['s'] for record in history]
    axes[1, 1].plot(s_values, marker='d')
    axes[1, 1].set_title('服务时间 s(n)')
    axes[1, 1].set_xlabel('批次 n')
    axes[1, 1].set_ylabel('s')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # 测试不同的均衡状态
    sim1 = test_cyclic_equilibrium()
    sim2 = test_stable_equilibrium()
    sim3 = test_three_stages()
    
    # 绘制第一个模拟的结果
    try:
        fig = plot_simulation(sim1)
        plt.savefig('/home/claude/simulation_results.png')
        print("\n结果已保存到 simulation_results.png")
        plt.show()
    except:
        print("\n(绘图功能需要matplotlib)")
