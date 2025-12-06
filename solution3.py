import numpy as np
from fractions import Fraction
from scipy.special import comb
import numpy.linalg as la

def create_matrices(l0, l1):
    """创建矩阵 B1 和 B2"""
    m = l1
    
    # 计算参数
    alpha = Fraction(-1, l0 + 1)
    beta = Fraction(l0 + l1, l0 + 1)
    a = Fraction(1, l0 + 2)
    
    # 创建 B1 矩阵
    B1 = np.zeros((m, m), dtype=object)
    # 第一行
    for j in range(m-1):
        B1[0, j] = alpha
    B1[0, m-1] = beta
    # 子对角线
    for i in range(1, m):
        B1[i, i-1] = Fraction(1)
    
    # 创建 B2 矩阵
    B2 = np.zeros((m, m), dtype=object)
    # 第一行全为0
    # 第二行
    if m > 1:
        B2[1, 0] = Fraction(1) - a
        for j in range(1, m-1):
            B2[1, j] = -a
        B2[1, m-1] = Fraction(0)
    # 子对角线
    for i in range(2, m):
        B2[i, i-1] = Fraction(1)
    
    return B1, B2, alpha, beta, a

def matrix_multiply(A, B):
    """矩阵乘法（用于Fraction类型）"""
    m, n = A.shape
    n2, p = B.shape
    assert n == n2, "矩阵维度不匹配"
    
    C = np.zeros((m, p), dtype=object)
    for i in range(m):
        for j in range(p):
            C[i, j] = Fraction(0)
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

def matrix_power(A, n):
    """计算矩阵的n次幂"""
    m = A.shape[0]
    if n == 0:
        # 返回单位矩阵
        I = np.zeros((m, m), dtype=object)
        for i in range(m):
            I[i, i] = Fraction(1)
        return I
    
    result = A.copy()
    for _ in range(n-1):
        result = matrix_multiply(result, A)
    return result

def create_test_vector(m):
    """创建测试向量 v = [0, 2, 1, 1, ..., 1]"""
    v = np.zeros(m, dtype=object)
    v[0] = Fraction(0)
    v[1] = Fraction(2)
    for i in range(2, m):
        v[i] = Fraction(1)
    return v.reshape(-1, 1)

def print_matrix(M, name):
    """打印矩阵"""
    print(f"\n{name}:")
    m, n = M.shape
    for i in range(m):
        row = []
        for j in range(n):
            if isinstance(M[i, j], Fraction):
                if M[i, j].denominator == 1:
                    row.append(str(M[i, j].numerator))
                else:
                    row.append(f"{M[i, j].numerator}/{M[i, j].denominator}")
            else:
                row.append(str(M[i, j]))
        print("[" + ", ".join(f"{x:>8}" for x in row) + "]")

def print_vector(v, name):
    """打印向量"""
    print(f"\n{name}:")
    vector_str = "["
    for i in range(len(v)):
        if isinstance(v[i, 0], Fraction):
            if v[i, 0].denominator == 1:
                vector_str += f"{v[i, 0].numerator}"
            else:
                vector_str += f"{v[i, 0].numerator}/{v[i, 0].denominator}"
        else:
            vector_str += str(v[i, 0])
        if i < len(v) - 1:
            vector_str += ", "
    vector_str += "]"
    print(f"  {vector_str}")

def verify_eigenvalue_1(P, v):
    """验证v是否是P的特征值1对应的特征向量"""
    m = P.shape[0]
    
    # 计算 P*v
    Pv = matrix_multiply(P, v)
    
    # 计算 P*v - v
    error = np.zeros((m, 1), dtype=object)
    for i in range(m):
        error[i, 0] = Pv[i, 0] - v[i, 0]
    
    # 检查是否为零向量
    is_eigenvector = all(error[i, 0] == Fraction(0) for i in range(m))
    
    return Pv, error, is_eigenvector

def main():
    print("="*70)
    print("验证向量 (0, 2, 1, 1, ..., 1) 是否为特征值1的特征向量")
    print("="*70)
    
    # 测试不同的参数组合
    test_cases = [
        (1, 3),   # m = 3
        (2, 3),   # m = 3
        (1, 4),   # m = 4
        (2, 4),   # m = 4
        (3, 4),   # m = 4
        (1, 5),   # m = 5
        (2, 5),   # m = 5
        (3, 5),   # m = 5
        (4, 5),   # m = 5
        (1, 6),   # m = 6
        (2, 6),   # m = 6
        (3, 6),   # m = 6
        (3, 20),   # m = 6
        (3, 33),   # m = 6
    ]
    
    results_summary = []
    
    for l0, l1 in test_cases:
        if l1 < 3:
            continue  # 跳过 l1 < 3 的情况
            
        print("\n" + "="*70)
        print(f"测试案例: l0 = {l0}, l1 = {l1}")
        print("="*70)
        
        m = l1
        
        # 创建矩阵
        B1, B2, alpha, beta, a = create_matrices(l0, l1)
        
        print(f"\n参数:")
        print(f"  m = {m}")
        print(f"  alpha = -1/{l0+1} = {alpha}")
        print(f"  beta = {l0+l1}/{l0+1} = {beta}")
        print(f"  a = 1/{l0+2} = {a}")
        
        # 计算 P = B2 * B1^(m-1)
        B1_power = matrix_power(B1, m-1)
        P = matrix_multiply(B2, B1_power)
        
        print_matrix(P, f"P = B2 * B1^{m-1}")
        
        # 创建测试向量
        v = create_test_vector(m)
        print_vector(v, "测试向量 v = (0, 2, 1, ..., 1)")
        
        # 验证是否为特征向量
        Pv, error, is_eigenvector = verify_eigenvalue_1(P, v)
        
        print_vector(Pv, "P * v")
        print_vector(error, "P*v - v (误差)")
        
        # 输出验证结果
        if is_eigenvector:
            print(f"\n✓ 验证成功: v 是特征值1对应的特征向量")
            results_summary.append((l0, l1, "✓"))
        else:
            print(f"\n✗ 验证失败: v 不是特征值1对应的特征向量")
            results_summary.append((l0, l1, "✗"))
            
        # 计算并显示实际特征值
        P_float = np.array([[float(P[i, j]) for j in range(m)] for i in range(m)])
        eigenvalues = la.eigvals(P_float)
        eigenvalues_real = np.real(eigenvalues)
        eigenvalues_sorted = np.sort(eigenvalues_real)[::-1]
        
        print("\n特征值（降序）:")
        has_eigenvalue_1 = False
        for i, ev in enumerate(eigenvalues_sorted):
            print(f"  λ{i+1} = {ev:.6f}")
            if np.abs(ev - 1.0) < 1e-10:
                has_eigenvalue_1 = True
        
        if not has_eigenvalue_1:
            print("  注意：此矩阵没有特征值1")
    
    # 打印总结
    print("\n" + "="*70)
    print("总结：向量 (0, 2, 1, ..., 1) 的验证结果")
    print("="*70)
    print("\n  l0   l1   结果")
    print("  ---  ---  ----")
    for l0, l1, result in results_summary:
        print(f"  {l0:2d}   {l1:2d}    {result}")
    
    # 分析模式
    print("\n分析:")
    success_count = sum(1 for _, _, r in results_summary if r == "✓")
    total_count = len(results_summary)
    
    if success_count == 0:
        print(f"  向量 (0, 2, 1, ..., 1) 在所有 {total_count} 个测试案例中都不是特征向量")
        print("  这表明此向量可能不是一般情况下的特征向量")
    elif success_count == total_count:
        print(f"  向量 (0, 2, 1, ..., 1) 在所有 {total_count} 个测试案例中都是特征向量")
        print("  这是一个有趣的发现！")
    else:
        print(f"  向量 (0, 2, 1, ..., 1) 在 {success_count}/{total_count} 个案例中是特征向量")
        print("  需要进一步分析成功和失败案例的模式")
        
        # 分析成功案例的模式
        success_cases = [(l0, l1) for l0, l1, r in results_summary if r == "✓"]
        if success_cases:
            print("\n  成功的案例 (l0, l1):")
            for l0, l1 in success_cases:
                print(f"    ({l0}, {l1})")
                
    # 额外分析：检查是否与 l0 + l1 = 常数有关
    print("\n检查 l0 + l1 的值:")
    for l0, l1, result in results_summary:
        print(f"  l0={l0}, l1={l1}: l0+l1={l0+l1}, 结果={result}")

if __name__ == "__main__":
    main()