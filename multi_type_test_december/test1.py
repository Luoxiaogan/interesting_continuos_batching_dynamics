import numpy as np

def get_B1(l0, lA, lB):
    """
    构造矩阵 B1
    
    参数:
        l0: 整数参数
        lA: 整数参数
        lB: 整数参数，矩阵的维度 (lB x lB)
    
    返回:
        B1: lB x lB 的 numpy 矩阵
    """
    # 计算 alpha 和 beta
    alpha = -1 / (l0 + 1)
    beta = (l0 + lB) / (l0 + 1)
    
    # 初始化 lB x lB 的零矩阵
    B1 = np.zeros((lB, lB))
    
    # 填充第一行：前 lB-1 列都是 alpha，最后一列是 beta
    B1[0, :-1] = alpha
    B1[0, -1] = beta
    
    # 填充下三角部分：从第二行开始，每行在对角线下方位置填1
    for i in range(1, lB):
        B1[i, i-1] = 1
    
    return B1


def get_B1_prime(l0, lA, lB, p):
    """
    构造矩阵 B1'
    
    参数:
        l0: 整数参数
        lA: 整数参数，指定非零元素的列位置
        lB: 整数参数，矩阵的维度 (lB x lB)
        p: 概率参数
    
    返回:
        B1_prime: lB x lB 的 numpy 矩阵
    """
    # 先获取 B1 矩阵
    B1 = get_B1(l0, lA, lB)
    
    # 创建修正矩阵（全零矩阵）
    correction = np.zeros((lB, lB))
    
    # 计算要添加的值
    value_1_lA = ((l0 + lA + 1) / (l0 + 1)) * p  # 位置 [1, l_A] 的值（注意：索引从0开始，所以是 [0, lA-1]）
    value_lA1_lA = 1 - p  # 位置 [l_A+1, l_A] 的值（注意：索引从0开始，所以是 [lA, lA-1]）
    
    # 在修正矩阵中填入非零项
    # 位置 [1, l_A] -> 索引 [0, lA-1]
    correction[0, lA - 1] = value_1_lA
    
    # 位置 [l_A+1, l_A] -> 索引 [lA, lA-1]
    correction[lA, lA - 1] = value_lA1_lA
    
    # B1' = B1 + 修正矩阵
    B1_prime = B1 + correction
    
    return B1_prime


def get_B1_power_k(l0, lA, lB, k):
    """
    计算矩阵 B1 的 k 次幂
    
    参数:
        l0: 整数参数
        lA: 整数参数
        lB: 整数参数
        k: 幂次
    
    返回:
        B1^k: B1 的 k 次幂矩阵
    """
    B1 = get_B1(l0, lA, lB)
    
    # 使用 numpy 的矩阵幂运算
    B1_k = np.linalg.matrix_power(B1, k)
    
    return B1_k

def get_B1_prime_power_k(l0, lA, lB, p, k):
    """
    计算矩阵 B1' 的 k 次幂
    
    参数:
        l0: 整数参数
        lA: 整数参数
        lB: 整数参数
        p: 概率参数
        k: 幂次
    
    返回:
        B1'^k: B1' 的 k 次幂矩阵
    """
    B1 = get_B1_prime(l0, lA, lB, p)
    
    # 使用 numpy 的矩阵幂运算
    B1_k = np.linalg.matrix_power(B1, k)
    
    return B1_k

def analyze_eigenvalues(matrix, name):
    """
    分析矩阵的特征值和特征向量
    
    参数:
        matrix: 要分析的矩阵
        name: 矩阵的名称（用于显示）
    """
    print(f"\n{'='*60}")
    print(f"{name} 的特征值分析")
    print(f"{'='*60}")
    
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    
    # 按特征值的模（绝对值）降序排序
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print(f"\n特征值 (按模排序):")
    for i, eigenvalue in enumerate(eigenvalues):
        if np.isreal(eigenvalue):
            print(f"  λ_{i+1} = {eigenvalue.real:.6f}, 模 = {np.abs(eigenvalue):.6f}")
        else:
            print(f"  λ_{i+1} = {eigenvalue.real:.6f} + {eigenvalue.imag:.6f}i, 模 = {np.abs(eigenvalue):.6f}")
    
    print(f"\n对应的特征向量:")
    for i in range(len(eigenvalues)):
        print(f"\n特征值 λ_{i+1} 对应的特征向量:")
        eigenvector = eigenvectors[:, i]
        if np.all(np.isreal(eigenvector)):
            eigenvector = eigenvector.real
            print(f"  {eigenvector}")
        else:
            print(f"  实部: {eigenvector.real}")
            print(f"  虚部: {eigenvector.imag}")
    
    # 找出主特征值（模最大的特征值）
    dominant_eigenvalue = eigenvalues[0]
    print(f"\n主特征值（模最大）: λ_1 = {dominant_eigenvalue}")
    print(f"主特征值的模: |λ_1| = {np.abs(dominant_eigenvalue):.6f}")

# ============= 主程序：可以在这里硬编码参数 =============
if __name__ == "__main__":
    # 硬编码参数（可以根据需要修改）
    l0 = 5
    lA = 3
    lB = 7
    p = 0.6
    k = 1000

    assert (lB>lA)
    
    # 获取 B1 矩阵
    B1 = get_B1(l0, lA, lB)
    print("矩阵 B1:")
    print(B1)
    print()
    
    # 获取 B1' 矩阵
    B1_prime = get_B1_prime(l0, lA, lB, p)
    print("矩阵 B1':")
    print(B1_prime)
    print()
    
    # 获取 B1^k 矩阵
    B1_k = get_B1_power_k(l0, lA, lB, k)
    print(f"矩阵 B1^{k}:")
    print(B1_k)

    # 获取 B1'^k 矩阵
    B1_prime_k = get_B1_prime_power_k(l0, lA, lB, p, k)
    print(f"\n矩阵 B1'^{k}:")
    print(B1_prime_k)
    
    # 分析 B1 的特征值和特征向量
    analyze_eigenvalues(B1, "B1")
    
    # 分析 B1' 的特征值和特征向量
    analyze_eigenvalues(B1_prime, "B1'")