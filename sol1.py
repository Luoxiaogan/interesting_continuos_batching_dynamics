"""
计算矩阵 P = B2^{(i)} * B1^{j} 的特征值和特征向量
简化版本 - 只显示核心结果
"""

import numpy as np
from scipy import linalg

def generate_B1(l0, l1, j=1):
    """生成矩阵 B1^j"""
    m = l1
    alpha = -1 / (l0 + 1)
    beta = (l0 + l1) / (l0 + 1)
    
    B1 = np.zeros((m, m))
    B1[0, :-1] = alpha
    B1[0, -1] = beta
    
    for i in range(1, m):
        B1[i, i-1] = 1
    
    if j == 1:
        return B1
    else:
        return np.linalg.matrix_power(B1, j)

def generate_B2(l0, l1, i):
    """生成矩阵 B2^{(i)}"""
    m = l1
    B2 = np.zeros((m, m))
    
    I_sub = np.eye(m-1)
    e_i = np.zeros(m-1)
    e_i[i-1] = 1
    ones = np.ones(m-1)
    
    correction = (1 / (l0 + i + 1)) * np.outer(e_i, ones)
    sub_matrix = I_sub - correction
    
    B2[1:, :-1] = sub_matrix
    
    return B2

def main():
    # ========== 参数设置 ==========
    l0 = 1
    l1 = 4
    i = 1  # 驱逐层级
    j = l1 - 1  # B1的幂次
    B = 300
    w = [l0+i+1 for i in range(0,l1)]
    z = [0,2] + [1]*(l1-2)
    
    print("=" * 60)
    print(f"参数: l0={l0}, l1={l1}, i={i}, j={j}")
    # print(f"计算: P = B2^{{({i})}} * B1^{{{j}}}")
    print("=" * 60)
    
    # ========== 计算 P ==========
    P =  generate_B2(l0, l1, 1) @ generate_B1(l0, l1, 3)
    # P = generate_B1(l0, l1, 4)
    print(P,'\n')
    print(f"w={w}\nz={z}\n")
    x_star = B/sum([ i*j for i, j in zip(w,z) ])
    print(f"sum={sum([ i*j for i, j in zip(w,z) ])}, x^*={x_star}")
    X = [i*x_star for i in z]
    print(f"X^*={X}, w^TX^*={sum([ i*j for i, j in zip(w,X) ])}")
    X0 = X.copy()
    a = 1+0.001
    if a<1:
        X0[l1-1]=X0[l1-1]*a
        X0[1] += (X0[l1-1]/a*(1-a)*(l0+l1))/(l0+2) 
    if a>1:
        X0[l1-1]=X0[l1-1]*a
        X0[1] -= (X0[l1-1]/a*(a-1)*(l0+l1))/(l0+2) 
    print(f"X0={X0}, w^TX0={sum([ i*j for i, j in zip(w,X0) ])}")
    one = [1]*l1

    
    # P = P.T
    # ========== 特征值和特征向量 ==========
    eigenvalues, eigenvectors = linalg.eig(P)
    
    # 按特征值从小到大排序
    idx = np.argsort(eigenvalues.real)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print("\n特征值与特征向量（从小到大）:")
    print("=" * 60)
    
    for k in range(len(eigenvalues)):
        ev = eigenvalues[k]
        vec = eigenvectors[:, k]
        
        # 特征值
        if np.abs(ev.imag) < 1e-10:
            print(f"\nλ_{k+1} = {ev.real:.10f}")
        else:
            print(f"\nλ_{k+1} = {ev.real:.10f} + {ev.imag:.10f}i")
        
        # 特征向量
        print(f"v_{k+1} = [", end="")
        for idx, component in enumerate(vec):
            if np.abs(component.imag) < 1e-10:
                print(f"{component.real:9.6f}", end="")
            else:
                print(f"({component.real:9.6f}+{component.imag:9.6f}i)", end="")
            if idx < len(vec) - 1:
                print(", ", end="")
        print("]")
    print("=" * 60)

    T = 10
    for i in range(T):
        print("\n")
        wTX0 = sum([ i*j for i, j in zip(w,X0) ])
        oneTX0 = sum([ i*j for i, j in zip(one,X0) ])
        mean = oneTX0/l1
        print(f"P^{i}, B1^{0},   w^TX={wTX0:.2f}, 1^TX={oneTX0:.5f}, X={X0}, mean = {mean}")
        for j in range(1,l1):
            X0 = generate_B1(l0, l1, 1) @ X0
            wTX0 = sum([ i*j for i, j in zip(w,X0) ])
            oneTX0 = sum([ i*j for i, j in zip(one,X0) ])
            print(f"P^{i}, B1^{j},   w^TX={wTX0:.2f}, 1^TX={oneTX0:.5f}, X={X0}")
        X0 = generate_B2(l0, l1, 1)@ X0
        wTX0 = sum([ i*j for i, j in zip(w,X0) ])
        oneTX0 = sum([ i*j for i, j in zip(one,X0) ])
        print(f"P^{i}, B2B1^{l1-1}, w^TX={wTX0:.2f}, 1^TX={oneTX0:.5f}, X={X0}")



    

if __name__ == "__main__":
    main()