import numpy as np
from fractions import Fraction
from scipy.special import comb
import numpy.linalg as la
from math import gcd
from functools import reduce

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

def create_eigenvector(m):
    """创建特征向量 w = [0, C(m-1,1), C(m-1,2), ..., C(m-1,m-1)]"""
    w = np.zeros(m, dtype=object)
    w[0] = Fraction(0)
    for i in range(1, m):
        w[i] = Fraction(int(comb(m-1, i)))
    return w.reshape(-1, 1)

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
    for i in range(len(v)):
        if isinstance(v[i, 0], Fraction):
            if v[i, 0].denominator == 1:
                print(f"  {v[i, 0].numerator}")
            else:
                print(f"  {v[i, 0].numerator}/{v[i, 0].denominator}")
        else:
            print(f"  {v[i, 0]}")

def lcm(a, b):
    """计算最小公倍数"""
    return abs(a * b) // gcd(a, b)

def find_eigenvector_for_eigenvalue_1(P):
    """找到特征值1对应的特征向量（使用精确分数运算）"""
    m = P.shape[0]
    
    # 构造 (P - I)
    P_minus_I = np.zeros((m, m), dtype=object)
    for i in range(m):
        for j in range(m):
            P_minus_I[i, j] = P[i, j] - (Fraction(1) if i == j else Fraction(0))
    
    # 使用高斯消元法求解 (P - I)v = 0
    # 这里我们寻找核空间的一个非零向量
    
    # 增广矩阵（不需要右边的0向量部分）
    A = P_minus_I.copy()
    
    # 高斯消元
    for k in range(min(m-1, m)):
        # 找主元
        pivot_row = k
        for i in range(k+1, m):
            if A[i, k] != 0:
                pivot_row = i
                break
        
        if A[pivot_row, k] == 0:
            continue
            
        # 交换行
        if pivot_row != k:
            A[[k, pivot_row]] = A[[pivot_row, k]]
        
        # 消元
        for i in range(k+1, m):
            if A[i, k] != 0:
                factor = A[i, k] / A[k, k]
                for j in range(k, m):
                    A[i, j] -= factor * A[k, j]
    
    # 回代求解
    # 假设最后一个变量为自由变量，设为1
    v = np.zeros(m, dtype=object)
    v[-1] = Fraction(1)
    
    # 从倒数第二行开始回代
    for i in range(m-2, -1, -1):
        if all(A[i, j] == 0 for j in range(m)):
            # 自由变量
            v[i] = Fraction(1)
        else:
            # 找到第一个非零元素
            pivot_col = -1
            for j in range(i, m):
                if A[i, j] != 0:
                    pivot_col = j
                    break
            
            if pivot_col >= 0:
                sum_val = Fraction(0)
                for j in range(pivot_col+1, m):
                    sum_val += A[i, j] * v[j]
                if A[i, pivot_col] != 0:
                    v[pivot_col] = -sum_val / A[i, pivot_col]
    
    return v

def simplify_eigenvector(v):
    """将特征向量转换为最简整数形式"""
    # 找到所有分母的最小公倍数
    denominators = []
    for val in v:
        if isinstance(val, Fraction) and val.denominator != 1:
            denominators.append(val.denominator)
    
    if denominators:
        lcm_denom = reduce(lcm, denominators)
    else:
        lcm_denom = 1
    
    # 乘以最小公倍数得到整数
    v_int = []
    for val in v:
        if isinstance(val, Fraction):
            v_int.append(int(val * lcm_denom))
        else:
            v_int.append(int(val * lcm_denom))
    
    # 找到所有整数的最大公约数
    gcd_all = reduce(gcd, [abs(x) for x in v_int if x != 0])
    
    # 除以最大公约数得到最简形式
    v_simplified = [x // gcd_all for x in v_int]
    
    return v_simplified, lcm_denom, gcd_all

def main():
    # 硬编码 l0 和 l1
    test_cases = [
        (2050, 25),  # 案例3
    ]
    
    for l0, l1 in test_cases:
        print("="*60)
        print(f"测试案例: l0 = {l0}, l1 = {l1}")
        print("="*60)
        
        m = l1
        
        # 创建矩阵
        B1, B2, alpha, beta, a = create_matrices(l0, l1)
        
        print(f"\n参数:")
        print(f"  m = {m}")
        print(f"  alpha = {alpha}")
        print(f"  beta = {beta}")
        print(f"  a = {a}")
        
        # 打印矩阵
        print_matrix(B1, "B1")
        print_matrix(B2, "B2")
        
        # 计算 B1^(m-1)
        B1_power = matrix_power(B1, m-1)
        print_matrix(B1_power, f"B1^{m-1}")
        
        # 计算 P = B2 * B1^(m-1)
        P = matrix_multiply(B2, B1_power)
        print_matrix(P, f"P = B2 * B1^{m-1}")
        
        # 创建理论特征向量 w
        w = create_eigenvector(m)
        print_vector(w, "理论特征向量 w (二项式系数)")
        
        # 计算 P*w
        Pw = matrix_multiply(P, w)
        print_vector(Pw, "P * w")
        
        # 计算误差 P*w - w
        error = np.zeros((m, 1), dtype=object)
        for i in range(m):
            error[i, 0] = Pw[i, 0] - w[i, 0]
        print_vector(error, "P*w - w (应该全为0)")
        
        # 验证是否为特征向量
        is_eigenvector = all(error[i, 0] == Fraction(0) for i in range(m))
        print(f"\n验证结果: {'✓ w 是特征值1对应的特征向量' if is_eigenvector else '✗ w 不是特征向量'}")
        
        # 计算特征值（使用浮点数近似）
        print("\n特征值（浮点数近似）:")
        P_float = np.array([[float(P[i, j]) for j in range(m)] for i in range(m)])
        eigenvalues = la.eigvals(P_float)
        eigenvalues_real = np.real(eigenvalues)
        eigenvalues_sorted = np.sort(eigenvalues_real)[::-1]
        
        has_eigenvalue_1 = False
        for i, ev in enumerate(eigenvalues):
            if np.abs(ev - 1.0) < 1e-10:
                has_eigenvalue_1 = True
            if np.isreal(ev):
                print(f"  λ{i+1} = {ev.real:.6f}")
            else:
                print(f"  λ{i+1} = {ev:.6f}")
        
        # 如果存在特征值1，计算对应的特征向量
        if has_eigenvalue_1:
            print("\n计算特征值1对应的特征向量（数值方法）:")
            
            # 使用numpy计算特征向量
            eigenvalues_np, eigenvectors_np = la.eig(P_float)
            
            # 找到特征值1对应的索引
            idx_1 = np.argmin(np.abs(eigenvalues_np - 1.0))
            
            if np.abs(eigenvalues_np[idx_1] - 1.0) < 1e-10:
                v_numerical = eigenvectors_np[:, idx_1]
                
                # 归一化使第一个非零元素为正
                first_nonzero = np.argmax(np.abs(v_numerical) > 1e-10)
                if v_numerical[first_nonzero] < 0:
                    v_numerical = -v_numerical
                
                print("  数值解（浮点数）:")
                for i, val in enumerate(v_numerical):
                    print(f"    v[{i}] = {val:.6f}")
                
                # 尝试将数值解转换为分数形式
                print("\n  尝试转换为分数形式:")
                v_fractions = []
                for val in v_numerical:
                    # 使用limit_denominator限制分母大小
                    frac = Fraction(val).limit_denominator(1000)
                    v_fractions.append(frac)
                
                for i, frac in enumerate(v_fractions):
                    if frac.denominator == 1:
                        print(f"    v[{i}] = {frac.numerator}")
                    else:
                        print(f"    v[{i}] = {frac.numerator}/{frac.denominator}")
                
                # 转换为最简整数形式
                v_simplified, lcm_used, gcd_used = simplify_eigenvector(v_fractions)
                
                print(f"\n  最简整数形式（乘以{lcm_used}，除以{gcd_used}）:")
                print(f"  v = {v_simplified}")
                
                # 验证理论向量
                w_flat = [int(w[i, 0]) for i in range(m)]
                print(f"\n  理论特征向量（整数形式）: {w_flat}")
                
                # 检查两个向量是否成比例
                ratio_check = True
                ratio = None
                for i in range(m):
                    if v_simplified[i] != 0 and w_flat[i] != 0:
                        current_ratio = Fraction(v_simplified[i], w_flat[i])
                        if ratio is None:
                            ratio = current_ratio
                        elif ratio != current_ratio:
                            ratio_check = False
                            break
                
                if ratio_check and ratio is not None:
                    print(f"  ✓ 数值解与理论解成比例，比例系数: {ratio}")
                else:
                    print(f"  ✗ 数值解与理论解不成比例")

if __name__ == "__main__":
    main()