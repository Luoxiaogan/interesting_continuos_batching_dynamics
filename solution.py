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

def main():
    # 硬编码 l0 和 l1
    test_cases = [
        (1, 3),  # 案例1
        (2, 3),  # 案例2
        (2, 4),  # 案例3
        (3, 5),  # 额外测试
        (3, 6),  # 额外测试
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
        
        # 创建特征向量 w
        w = create_eigenvector(m)
        print_vector(w, "特征向量 w")
        
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
        
        # 计算特征值（可选：使用浮点数近似）
        print("\n特征值（浮点数近似）:")
        P_float = np.array([[float(P[i, j]) for j in range(m)] for i in range(m)])
        eigenvalues = la.eigvals(P_float)
        eigenvalues = np.sort(eigenvalues)[::-1]
        for i, ev in enumerate(eigenvalues):
            if np.isreal(ev):
                print(f"  λ{i+1} = {ev.real:.6f}")
            else:
                print(f"  λ{i+1} = {ev:.6f}")

if __name__ == "__main__":
    main()