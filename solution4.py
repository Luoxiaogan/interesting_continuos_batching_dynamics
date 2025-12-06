import numpy as np
from scipy.special import comb
import numpy.linalg as la
from decimal import Decimal, getcontext

# 设置高精度
getcontext().prec = 50  # 50位精度

def create_matrices_float(l0, l1):
    """创建矩阵 B1 和 B2（使用高精度浮点数）"""
    m = l1
    
    # 计算参数
    alpha = -1.0 / (l0 + 1)
    beta = (l0 + l1) / (l0 + 1)
    a = 1.0 / (l0 + 2)
    
    # 创建 B1 矩阵
    B1 = np.zeros((m, m), dtype=np.float64)
    # 第一行
    for j in range(m-1):
        B1[0, j] = alpha
    B1[0, m-1] = beta
    # 子对角线
    for i in range(1, m):
        B1[i, i-1] = 1.0
    
    # 创建 B2 矩阵
    B2 = np.zeros((m, m), dtype=np.float64)
    # 第一行全为0
    # 第二行
    if m > 1:
        B2[1, 0] = 1.0 - a
        for j in range(1, m-1):
            B2[1, j] = -a
        B2[1, m-1] = 0.0
    # 子对角线
    for i in range(2, m):
        B2[i, i-1] = 1.0
    
    return B1, B2, alpha, beta, a

def create_test_vector(m):
    """创建测试向量 v = [0, 2, 1, 1, ..., 1]"""
    v = np.zeros(m)
    v[0] = 0.0
    v[1] = 2.0
    for i in range(2, m):
        v[i] = 1.0
    return v.reshape(-1, 1)

def verify_eigenvalue_1(P, v, tolerance=1e-10):
    """验证v是否是P的特征值1对应的特征向量"""
    # 计算 P*v
    Pv = P @ v
    
    # 计算 P*v - v
    error = Pv - v
    
    # 计算误差范数
    error_norm = np.linalg.norm(error)
    
    # 检查是否接近零向量
    is_eigenvector = error_norm < tolerance
    
    return Pv, error, is_eigenvector, error_norm

def batch_test(max_l1=25):
    """批量测试不同的参数组合"""
    print("="*80)
    print("批量测试：验证向量 (0, 2, 1, 1, ..., 1) 是否为特征值1的特征向量")
    print("="*80)
    
    results = []
    
    # 生成所有测试案例
    for l1 in range(3, max_l1 + 1):
        for l0 in range(1, l1):
            m = l1
            
            # 创建矩阵
            B1, B2, alpha, beta, a = create_matrices_float(l0, l1)
            
            # 计算 P = B2 * B1^(m-1)
            B1_power = np.linalg.matrix_power(B1, m-1)
            P = B2 @ B1_power
            
            # 创建测试向量
            v = create_test_vector(m)
            
            # 验证是否为特征向量
            Pv, error, is_eigenvector, error_norm = verify_eigenvalue_1(P, v)
            
            # 计算特征值
            eigenvalues = la.eigvals(P)
            eigenvalues_real = np.real(eigenvalues)
            has_eigenvalue_1 = any(np.abs(ev - 1.0) < 1e-10 for ev in eigenvalues)
            
            # 找最接近1的特征值
            distances_to_1 = np.abs(eigenvalues_real - 1.0)
            closest_to_1 = eigenvalues_real[np.argmin(distances_to_1)]
            
            results.append({
                'l0': l0,
                'l1': l1,
                'm': m,
                'is_eigenvector': is_eigenvector,
                'error_norm': error_norm,
                'has_eigenvalue_1': has_eigenvalue_1,
                'closest_eigenvalue': closest_to_1,
                'all_eigenvalues': eigenvalues_real
            })
    
    return results

def analyze_results(results):
    """分析测试结果"""
    print("\n" + "="*80)
    print("结果分析")
    print("="*80)
    
    # 统计
    total = len(results)
    success = sum(1 for r in results if r['is_eigenvector'])
    has_ev1 = sum(1 for r in results if r['has_eigenvalue_1'])
    
    print(f"\n总测试案例数: {total}")
    print(f"向量是特征向量的案例数: {success} ({100*success/total:.1f}%)")
    print(f"矩阵有特征值1的案例数: {has_ev1} ({100*has_ev1/total:.1f}%)")
    
    # 显示成功的案例
    success_cases = [r for r in results if r['is_eigenvector']]
    if success_cases:
        print("\n成功案例 (向量是特征向量):")
        print("  l0   l1    误差范数")
        print("  ---  ---   ----------")
        for r in success_cases[:20]:  # 最多显示前20个
            print(f"  {r['l0']:3d}  {r['l1']:3d}   {r['error_norm']:.2e}")
        if len(success_cases) > 20:
            print(f"  ... 还有 {len(success_cases)-20} 个成功案例")
    
    # 显示失败的案例
    failed_cases = [r for r in results if not r['is_eigenvector']]
    if failed_cases:
        print("\n失败案例 (向量不是特征向量):")
        print("  l0   l1    误差范数      最接近1的特征值")
        print("  ---  ---   ----------    ---------------")
        for r in failed_cases[:20]:  # 最多显示前20个
            print(f"  {r['l0']:3d}  {r['l1']:3d}   {r['error_norm']:.2e}    {r['closest_eigenvalue']:.6f}")
        if len(failed_cases) > 20:
            print(f"  ... 还有 {len(failed_cases)-20} 个失败案例")
    
    # 寻找模式
    print("\n模式分析:")
    
    # 检查是否与 l0 + l1 的关系
    if success_cases:
        print("\n成功案例的 l0 + l1 值:")
        sums = [(r['l0'] + r['l1'], r['l0'], r['l1']) for r in success_cases]
        sums_unique = sorted(set([s[0] for s in sums]))
        for s in sums_unique[:10]:
            cases = [(l0, l1) for total, l0, l1 in sums if total == s]
            print(f"  l0 + l1 = {s}: {cases[:5]}{'...' if len(cases) > 5 else ''}")
    
    # 检查 l0 = 1 的特殊情况
    l0_1_cases = [r for r in results if r['l0'] == 1]
    l0_1_success = sum(1 for r in l0_1_cases if r['is_eigenvector'])
    print(f"\nl0 = 1 的案例: {l0_1_success}/{len(l0_1_cases)} 成功")
    
    # 检查 l0 = l1 - 1 的特殊情况
    l0_l1_minus_1 = [r for r in results if r['l0'] == r['l1'] - 1]
    l0_l1_minus_1_success = sum(1 for r in l0_l1_minus_1 if r['is_eigenvector'])
    print(f"l0 = l1 - 1 的案例: {l0_l1_minus_1_success}/{len(l0_l1_minus_1)} 成功")
    
    return success_cases, failed_cases

def detailed_test(l0, l1):
    """对特定参数进行详细测试"""
    print(f"\n详细测试: l0 = {l0}, l1 = {l1}")
    print("-" * 50)
    
    m = l1
    
    # 创建矩阵
    B1, B2, alpha, beta, a = create_matrices_float(l0, l1)
    
    print(f"参数:")
    print(f"  m = {m}")
    print(f"  alpha = -1/{l0+1} = {alpha:.6f}")
    print(f"  beta = {l0+l1}/{l0+1} = {beta:.6f}")
    print(f"  a = 1/{l0+2} = {a:.6f}")
    
    # 计算 P = B2 * B1^(m-1)
    B1_power = np.linalg.matrix_power(B1, m-1)
    P = B2 @ B1_power
    
    # 创建测试向量
    v = create_test_vector(m)
    print(f"\n测试向量 v = {v.flatten()}")
    
    # 验证
    Pv, error, is_eigenvector, error_norm = verify_eigenvalue_1(P, v)
    print(f"\nPv = {Pv.flatten()[:10]}{'...' if m > 10 else ''}")
    print(f"误差范数: {error_norm:.2e}")
    print(f"是特征向量: {'✓' if is_eigenvector else '✗'}")
    
    # 计算所有特征值
    eigenvalues = la.eigvals(P)
    eigenvalues_real = np.sort(np.real(eigenvalues))[::-1]
    print(f"\n前5个最大特征值: {eigenvalues_real[:5]}")

def main():
    print("扩展参数范围测试程序")
    print("="*80)
    print("测试范围：l1 ∈ [3, 25], l0 ∈ [1, 40]")
    print("注意：l0 可以大于 l1")
    print("="*80)
    
    results = []
    total_tests = 0
    
    # 测试所有组合
    for l1 in range(3, 51):  # l1 从 3 到 25
        print(f"\n处理 l1 = {l1}...")
        l1_results = []
        
        for l0 in range(1, 101):  # l0 从 1 到 40
            m = l1
            total_tests += 1
            
            try:
                # 创建矩阵
                B1, B2, alpha, beta, a = create_matrices_float(l0, l1)
                
                # 计算 P = B2 * B1^(m-1)
                B1_power = np.linalg.matrix_power(B1, m-1)
                P = B2 @ B1_power
                
                # 创建测试向量
                v = create_test_vector(m)
                
                # 验证是否为特征向量
                Pv, error, is_eigenvector, error_norm = verify_eigenvalue_1(P, v, tolerance=1e-10)
                
                # 计算特征值
                eigenvalues = la.eigvals(P)
                eigenvalues_real = np.real(eigenvalues)
                has_eigenvalue_1 = any(np.abs(ev - 1.0) < 1e-10 for ev in eigenvalues)
                
                # 找最接近1的特征值
                distances_to_1 = np.abs(eigenvalues_real - 1.0)
                closest_to_1 = eigenvalues_real[np.argmin(distances_to_1)]
                
                result = {
                    'l0': l0,
                    'l1': l1,
                    'm': m,
                    'is_eigenvector': is_eigenvector,
                    'error_norm': error_norm,
                    'has_eigenvalue_1': has_eigenvalue_1,
                    'closest_eigenvalue': closest_to_1,
                    'l0_over_l1': l0 / l1,
                    'l0_plus_l1': l0 + l1
                }
                
                results.append(result)
                l1_results.append(result)
                
            except Exception as e:
                print(f"  错误: l0={l0}, l1={l1}: {str(e)}")
                continue
        
        # 每个 l1 的统计
        success_count = sum(1 for r in l1_results if r['is_eigenvector'])
        print(f"  l1={l1}: {success_count}/{len(l1_results)} 成功 ({100*success_count/len(l1_results):.1f}%)")
    
    print(f"\n总共测试了 {total_tests} 个参数组合")
    
    # 详细分析
    print("\n" + "="*80)
    print("结果分析")
    print("="*80)
    
    # 基本统计
    total = len(results)
    success = sum(1 for r in results if r['is_eigenvector'])
    has_ev1 = sum(1 for r in results if r['has_eigenvalue_1'])
    
    print(f"\n有效测试案例数: {total}")
    print(f"向量是特征向量的案例数: {success} ({100*success/total:.1f}%)")
    print(f"矩阵有特征值1的案例数: {has_ev1} ({100*has_ev1/total:.1f}%)")
    
    # 按 l0 < l1, l0 = l1, l0 > l1 分类
    results_l0_less = [r for r in results if r['l0'] < r['l1']]
    results_l0_equal = [r for r in results if r['l0'] == r['l1']]
    results_l0_greater = [r for r in results if r['l0'] > r['l1']]
    
    print("\n按 l0 与 l1 的关系分类:")
    
    if results_l0_less:
        success_less = sum(1 for r in results_l0_less if r['is_eigenvector'])
        print(f"  l0 < l1: {success_less}/{len(results_l0_less)} 成功 ({100*success_less/len(results_l0_less):.1f}%)")
    
    if results_l0_equal:
        success_equal = sum(1 for r in results_l0_equal if r['is_eigenvector'])
        print(f"  l0 = l1: {success_equal}/{len(results_l0_equal)} 成功 ({100*success_equal/len(results_l0_equal):.1f}%)")
    
    if results_l0_greater:
        success_greater = sum(1 for r in results_l0_greater if r['is_eigenvector'])
        print(f"  l0 > l1: {success_greater}/{len(results_l0_greater)} 成功 ({100*success_greater/len(results_l0_greater):.1f}%)")
    
    # 显示所有成功案例
    success_cases = [r for r in results if r['is_eigenvector']]
    if success_cases:
        print(f"\n找到 {len(success_cases)} 个成功案例:")
        print("  l0   l1   l0/l1    l0+l1    误差范数")
        print("  ---  ---  ------   -----    ----------")
        
        # 按 l1 然后 l0 排序
        success_cases_sorted = sorted(success_cases, key=lambda x: (x['l1'], x['l0']))
        
        for r in success_cases_sorted[:30]:  # 显示前30个
            print(f"  {r['l0']:3d}  {r['l1']:3d}  {r['l0_over_l1']:6.3f}   {r['l0_plus_l1']:4d}     {r['error_norm']:.2e}")
        
        if len(success_cases) > 30:
            print(f"  ... 还有 {len(success_cases)-30} 个成功案例")
    
    # 寻找模式
    print("\n" + "="*80)
    print("模式分析")
    print("="*80)
    
    # 检查特定的 l0 值
    special_l0_values = [1, 2, 3, 4, 5, 10, 20, 30, 40]
    print("\n特定 l0 值的成功率:")
    for l0_val in special_l0_values:
        l0_cases = [r for r in results if r['l0'] == l0_val]
        if l0_cases:
            l0_success = sum(1 for r in l0_cases if r['is_eigenvector'])
            print(f"  l0 = {l0_val:2d}: {l0_success:2d}/{len(l0_cases):2d} 成功 ({100*l0_success/len(l0_cases):5.1f}%)")
    
    # 检查 l0 + l1 的值
    print("\n按 l0 + l1 的值分组（仅显示有成功案例的）:")
    if success_cases:
        sum_groups = {}
        for r in success_cases:
            s = r['l0_plus_l1']
            if s not in sum_groups:
                sum_groups[s] = []
            sum_groups[s].append((r['l0'], r['l1']))
        
        for s in sorted(sum_groups.keys())[:20]:
            pairs = sum_groups[s]
            print(f"  l0 + l1 = {s:3d}: {pairs[:5]}{'...' if len(pairs) > 5 else ''}")
    
    # 创建热图数据
    print("\n" + "="*80)
    print("成功案例热图（部分）")
    print("="*80)
    
    # 选择显示范围
    max_display_l0 = 20
    max_display_l1 = 15
    min_display_l1 = 3
    
    print(f"\n显示范围: l1 ∈ [{min_display_l1}, {max_display_l1}], l0 ∈ [1, {max_display_l0}]")
    print("符号: ✓=成功, ✗=失败, .=未显示")
    
    print("\n    l0→", end="")
    for l0 in range(1, max_display_l0 + 1):
        print(f" {l0:2d}", end="")
    print("\n  l1↓")
    
    for l1 in range(min_display_l1, max_display_l1 + 1):
        print(f"  {l1:2d}  ", end="")
        for l0 in range(1, max_display_l0 + 1):
            result = next((r for r in results if r['l0'] == l0 and r['l1'] == l1), None)
            if result:
                print("  ✓" if result['is_eigenvector'] else "  ✗", end="")
            else:
                print("  .", end="")
        print()
    
    # 导出结果到文件（可选）
    print("\n" + "="*80)
    print("结果汇总")
    print("="*80)
    
    # 保存成功案例到列表
    if success_cases:
        print("\n成功案例的 (l0, l1) 对:")
        success_pairs = [(r['l0'], r['l1']) for r in success_cases]
        success_pairs_sorted = sorted(success_pairs)
        
        # 按行输出，每行10个
        for i in range(0, len(success_pairs_sorted), 10):
            batch = success_pairs_sorted[i:i+10]
            print("  " + ", ".join([f"({l0},{l1})" for l0, l1 in batch]))
    
    # 尝试找出规律
    print("\n" + "="*80)
    print("规律探索")
    print("="*80)
    
    if success_cases:
        # 检查是否所有成功案例都满足某个条件
        print("\n检查可能的规律:")
        
        # 规律1: l0 = k * l1 - c 的形式？
        print("\n1. 检查 l0 与 l1 的线性关系:")
        for k in [0, 1, 2]:
            for c in range(-5, 6):
                matching = [r for r in success_cases if r['l0'] == k * r['l1'] + c]
                if len(matching) >= 3:  # 至少3个匹配
                    print(f"   l0 = {k}*l1 + ({c}): {len(matching)} 个案例")
        
        # 规律2: l0 + l1 = 常数？
        print("\n2. l0 + l1 为特定值时的成功率:")
        sum_stats = {}
        for r in results:
            s = r['l0_plus_l1']
            if s not in sum_stats:
                sum_stats[s] = {'total': 0, 'success': 0}
            sum_stats[s]['total'] += 1
            if r['is_eigenvector']:
                sum_stats[s]['success'] += 1
        
        for s in sorted(sum_stats.keys())[:15]:
            if sum_stats[s]['total'] > 0:
                rate = 100 * sum_stats[s]['success'] / sum_stats[s]['total']
                print(f"   l0 + l1 = {s:3d}: {sum_stats[s]['success']:2d}/{sum_stats[s]['total']:2d} ({rate:5.1f}%)")
    
    print("\n测试完成！")
    
    # 返回结果供进一步分析
    return results, success_cases

if __name__ == "__main__":
    results, success_cases = main()