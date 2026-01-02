import numpy as np
import matplotlib.pyplot as plt

def compute_F_coefficients(l_A, l_B, l0_A, l0_B, p_A):
    p_B = 1 - p_A
    max_l = max(l_A, l_B)
    coeffs = np.zeros(max_l)
    for m in range(0, l_A):
        power = max_l - 1 - m
        if power >= 0:
            coeffs[power] += p_A * (l0_A + m + 1)
    for m in range(0, l_B):
        power = max_l - 1 - m
        if power >= 0:
            coeffs[power] += p_B * (l0_B + m + 1)
    return coeffs

def compute_A_coefficients(l_A, l_B, p_A):
    p_B = 1 - p_A
    max_l = max(l_A, l_B)
    coeffs = np.zeros(max_l)
    coeffs[max_l - 1] = 1.0
    for m in range(1, l_A):
        power = max_l - 1 - m
        coeffs[power] += p_A
    for m in range(1, l_B):
        power = max_l - 1 - m
        coeffs[power] += p_B
    return coeffs

def find_roots(coeffs):
    return np.roots(coeffs[::-1])

def key_ratio(l0_A, l0_B, p_A):
    p_B = 1 - p_A
    return np.abs(l0_A - l0_B) / ((l0_A + 1) * p_A + (l0_B + 1) * p_B)

# Parameters
l_A, l_B = 2, 5
p_A = 0.3
p_B = 1 - p_A

# A roots
A_coeffs = compute_A_coefficients(l_A, l_B, p_A)
A_roots = find_roots(A_coeffs)
max_alpha = np.max(np.abs(A_roots))

# Unity roots
l_A_unity = [np.exp(2j * np.pi * k / l_A) for k in range(1, l_A)]  # -1 for l_A=2
l_B_unity = [np.exp(2j * np.pi * k / l_B) for k in range(1, l_B)]

print("=" * 80)
print("系统分析：Key Ratio 的三种行为")
print("=" * 80)

print(r"""
Key Ratio 定义：
                |l₀ᴬ - l₀ᴮ|              |Δ|
    R := ─────────────────────────── = ─────
         (l₀ᴬ+1)pᴬ + (l₀ᴮ+1)pᴮ         l̄₀

扰动公式中的关键项：
    λⱼ = αⱼ[1 + ε + ε·μⱼ] + O(ε²)
    
    其中 ε·μⱼ ∝ R（key ratio）
""")

print("=" * 80)
print("情况 1：R → 0（扰动成功）")
print("=" * 80)
print(r"""
【公式形式】
    l₀ᴬ = l₀ᴮ + Δ，其中 Δ 是固定常数
    
    当 l₀ᴮ → ∞ 时：
              |Δ|           |Δ|
    R = ─────────────── ≈ ────── → 0
        l₀ᴮ(pᴬ+pᴮ) + ...    l₀ᴮ

【扰动公式】
    ε·μⱼ → 0，所以：
    λⱼ ≈ αⱼ(1 + ε) → αⱼ
    
【结果】
    F(λ) 的根趋近于 A(λ) 的根 αⱼ
    max|λ| → max|αⱼ| < 1（对于互素情况）
""")

print("\n数值验证：l₀ᴬ = l₀ᴮ + 10")
print("-" * 70)
print(f"{'l₀ᴮ':>10} | {'l₀ᴬ':>10} | {'R':>12} | {'max|λ|':>12} | {'max|αⱼ|':>12}")
print("-" * 70)
for l0_B in [50, 200, 1000, 5000, 20000]:
    l0_A = l0_B + 10
    R = key_ratio(l0_A, l0_B, p_A)
    F_coeffs = compute_F_coefficients(l_A, l_B, l0_A, l0_B, p_A)
    F_roots = find_roots(F_coeffs)
    max_mod = np.max(np.abs(F_roots))
    print(f"{l0_B:>10} | {l0_A:>10} | {R:>12.6f} | {max_mod:>12.6f} | {max_alpha:>12.6f}")
print(f"{'∞':>10} | {'∞':>10} | {'0':>12} | {max_alpha:>12.6f} | {max_alpha:>12.6f}")

print("\n" + "=" * 80)
print("情况 2：R → 常数 ∈ (0, 1)（扰动失败，但根仍在单位圆内）")
print("=" * 80)
print(r"""
【公式形式】
    l₀ᴬ = k · l₀ᴮ，其中 k > 0 是固定常数（k ≠ 1）
    
    当 l₀ᴮ → ∞ 时：
           |k-1| · l₀ᴮ           |k-1|
    R = ─────────────────── = ──────────────── → 常数
        (k·pᴬ + pᴮ) · l₀ᴮ      k·pᴬ + pᴮ

【极限值】
           |k-1|
    R∞ = ────────────
         1 + (k-1)pᴬ

    k=2:   R∞ = 1/(1+pᴬ) = 1/1.3 ≈ 0.769
    k=0.5: R∞ = 0.5/(1-0.5pᴬ) ≈ 0.588
    k=3:   R∞ = 2/(1+2pᴬ) ≈ 1.25

【扰动公式】
    ε·μⱼ → 常数 ≠ 0
    扰动公式 λⱼ ≈ αⱼ(1 + ε + ε·μⱼ) 不再有效
    
【结果】
    根不趋近于 αⱼ，而是趋近于某个中间位置
    但只要 k 不太极端，max|λ| 仍然 < 1
""")

print("\n数值验证：l₀ᴬ = 2 · l₀ᴮ (k=2)")
print("-" * 70)
R_limit = (2-1) / (1 + (2-1)*p_A)
print(f"理论极限 R∞ = {R_limit:.6f}")
print("-" * 70)
print(f"{'l₀ᴮ':>10} | {'l₀ᴬ':>10} | {'R':>12} | {'max|λ|':>12}")
print("-" * 70)
for l0_B in [50, 200, 1000, 5000, 20000]:
    l0_A = 2 * l0_B
    R = key_ratio(l0_A, l0_B, p_A)
    F_coeffs = compute_F_coefficients(l_A, l_B, l0_A, l0_B, p_A)
    F_roots = find_roots(F_coeffs)
    max_mod = np.max(np.abs(F_roots))
    print(f"{l0_B:>10} | {l0_A:>10} | {R:>12.6f} | {max_mod:>12.6f}")

# 找到 k=2 时的极限根
print("\n当 k=2, l₀ᴮ → ∞ 时，根趋近于：")
l0_B_large = 100000
l0_A_large = 2 * l0_B_large
F_coeffs = compute_F_coefficients(l_A, l_B, l0_A_large, l0_B_large, p_A)
F_roots = find_roots(F_coeffs)
for i, r in enumerate(sorted(F_roots, key=lambda x: -np.abs(x))):
    print(f"  λ_{i} = {r:.6f}, |λ_{i}| = {np.abs(r):.6f}")

print("\n" + "=" * 80)
print("情况 3：R → +∞（根趋近单位根，边际稳定）")
print("=" * 80)
print(r"""
【公式形式】
    l₀ᴬ = l₀ᴮ^α，其中 α > 1（超线性增长）
    
    例如 l₀ᴬ = l₀ᴮ²：
           l₀ᴮ² - l₀ᴮ         l₀ᴮ
    R ≈ ───────────── ≈ ────── → +∞
           pᴬ · l₀ᴮ²           pᴬ

【扰动公式】
    ε·μⱼ → +∞
    扰动完全失效
    
【结果】
    系统退化为单一类型主导
    - 若 l₀ᴬ >> l₀ᴮ：A 类型主导，根 → l_A 次单位根
    - 若 l₀ᴮ >> l₀ᴬ：B 类型主导，根 → l_B 次单位根
    max|λ| → 1（边际稳定）
""")

print("\n数值验证：l₀ᴬ = l₀ᴮ² (超线性增长)")
print("-" * 70)
print(f"{'l₀ᴮ':>10} | {'l₀ᴬ':>12} | {'R':>12} | {'max|λ|':>12} | 最大根")
print("-" * 70)
for l0_B in [10, 20, 50, 100, 200]:
    l0_A = l0_B ** 2
    R = key_ratio(l0_A, l0_B, p_A)
    F_coeffs = compute_F_coefficients(l_A, l_B, l0_A, l0_B, p_A)
    F_roots = find_roots(F_coeffs)
    max_idx = np.argmax(np.abs(F_roots))
    max_root = F_roots[max_idx]
    max_mod = np.abs(max_root)
    print(f"{l0_B:>10} | {l0_A:>12} | {R:>12.4f} | {max_mod:>12.6f} | {max_root:.4f}")

print(f"\nl_A = {l_A} 次单位根（除1外）: {l_A_unity[0]:.4f}")
print("当 R → +∞ 且 l₀ᴬ >> l₀ᴮ 时，最大根 → -1")

print("\n" + "=" * 80)
print("反方向：l₀ᴮ = l₀ᴬ² (B 类型主导)")
print("-" * 70)
print(f"{'l₀ᴬ':>10} | {'l₀ᴮ':>12} | {'R':>12} | {'max|λ|':>12}")
print("-" * 70)
for l0_A in [10, 20, 50, 100, 200]:
    l0_B = l0_A ** 2
    R = key_ratio(l0_A, l0_B, p_A)
    F_coeffs = compute_F_coefficients(l_A, l_B, l0_A, l0_B, p_A)
    F_roots = find_roots(F_coeffs)
    max_mod = np.max(np.abs(F_roots))
    print(f"{l0_A:>10} | {l0_B:>12} | {R:>12.4f} | {max_mod:>12.6f}")

print(f"\nl_B = {l_B} 次单位根（除1外）:")
for i, r in enumerate(l_B_unity):
    print(f"  ω_{i+1} = {r:.4f}, |ω| = {np.abs(r):.4f}")
print("当 R → +∞ 且 l₀ᴮ >> l₀ᴬ 时，根 → l_B 次单位根")

print("\n" + "=" * 80)
print("总结")
print("=" * 80)
print(r"""
┌────────────────────┬─────────────────────┬──────────────────┬─────────────────┐
│     公式形式       │    Key Ratio R      │    根的极限      │   max|λ|        │
├────────────────────┼─────────────────────┼──────────────────┼─────────────────┤
│ l₀ᴬ = l₀ᴮ + Δ     │    R → 0            │    → αⱼ          │  → max|αⱼ| < 1  │
│ (固定差)           │                     │   (A的根)         │   ✓ 稳定        │
├────────────────────┼─────────────────────┼──────────────────┼─────────────────┤
│ l₀ᴬ = k·l₀ᴮ       │    R → |k-1|        │    → 中间位置     │  < 1            │
│ (固定比例)         │        ────────     │   (不在αⱼ附近)    │   ✓ 仍稳定      │
│                    │        1+(k-1)pᴬ    │                   │                 │
├────────────────────┼─────────────────────┼──────────────────┼─────────────────┤
│ l₀ᴬ = l₀ᴮ^α       │    R → +∞           │ → l_A次单位根     │  → 1            │
│ (超线性, α>1)      │                     │   (若A主导)       │   ⚠ 边际稳定    │
├────────────────────┼─────────────────────┼──────────────────┼─────────────────┤
│ l₀ᴮ = l₀ᴬ^α       │    R → +∞           │ → l_B次单位根     │  → 1            │
│ (超线性, α>1)      │                     │   (若B主导)       │   ⚠ 边际稳定    │
└────────────────────┴─────────────────────┴──────────────────┴─────────────────┘

核心洞察：
• R → 0:     扰动成功，根在 A(λ) 根附近
• R → 常数:  扰动失败，但根仍在单位圆内（互素情况）
• R → +∞:    系统退化为单类型，根趋向单位根
""")

#=============================================================================
# 可视化
#=============================================================================
fig = plt.figure(figsize=(16, 12))

# Unit circle
theta = np.linspace(0, 2*np.pi, 100)
unit_circle_x = np.cos(theta)
unit_circle_y = np.sin(theta)

# Row 1: Key ratio behavior
ax1 = fig.add_subplot(2, 3, 1)
l0_B_range = np.logspace(1, 4, 100)

# Case 1: Fixed Delta
R_case1 = [10 / ((l0_B + 10 + 1) * p_A + (l0_B + 1) * p_B) for l0_B in l0_B_range]
ax1.loglog(l0_B_range, R_case1, 'b-', linewidth=2, label=r'$l_0^A = l_0^B + 10$ (Fixed $\Delta$)')

# Case 2: Fixed k
R_case2 = [key_ratio(2*l0_B, l0_B, p_A) for l0_B in l0_B_range]
ax1.loglog(l0_B_range, R_case2, 'g-', linewidth=2, label=r'$l_0^A = 2 l_0^B$ (Fixed $k$)')

# Case 3: Superlinear
R_case3 = [key_ratio(l0_B**1.5, l0_B, p_A) for l0_B in l0_B_range]
ax1.loglog(l0_B_range, R_case3, 'r-', linewidth=2, label=r'$l_0^A = (l_0^B)^{1.5}$ (Superlinear)')

ax1.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
ax1.set_xlabel(r'$l_0^B$', fontsize=11)
ax1.set_ylabel('Key Ratio R', fontsize=11)
ax1.set_title('Key Ratio vs $l_0^B$', fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Row 1: max|lambda| behavior
ax2 = fig.add_subplot(2, 3, 2)
l0_B_range_dense = np.logspace(1.3, 3.5, 50)

max_mods_case1 = []
max_mods_case2 = []
max_mods_case3 = []

for l0_B in l0_B_range_dense:
    # Case 1
    F_coeffs = compute_F_coefficients(l_A, l_B, l0_B + 10, l0_B, p_A)
    max_mods_case1.append(np.max(np.abs(find_roots(F_coeffs))))
    # Case 2
    F_coeffs = compute_F_coefficients(l_A, l_B, 2*l0_B, l0_B, p_A)
    max_mods_case2.append(np.max(np.abs(find_roots(F_coeffs))))
    # Case 3
    F_coeffs = compute_F_coefficients(l_A, l_B, l0_B**1.5, l0_B, p_A)
    max_mods_case3.append(np.max(np.abs(find_roots(F_coeffs))))

ax2.semilogx(l0_B_range_dense, max_mods_case1, 'b-', linewidth=2, label='Case 1: R→0')
ax2.semilogx(l0_B_range_dense, max_mods_case2, 'g-', linewidth=2, label='Case 2: R→const')
ax2.semilogx(l0_B_range_dense, max_mods_case3, 'r-', linewidth=2, label='Case 3: R→∞')
ax2.axhline(y=max_alpha, color='blue', linestyle='--', alpha=0.7, label=f'max|α_j|={max_alpha:.3f}')
ax2.axhline(y=1, color='black', linestyle=':', alpha=0.5, label='|λ|=1')
ax2.set_xlabel(r'$l_0^B$', fontsize=11)
ax2.set_ylabel(r'max|$\lambda$|', fontsize=11)
ax2.set_title(r'max|$\lambda$| vs $l_0^B$', fontsize=12)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.85, 1.02)

# Row 1: Parametric plot
ax3 = fig.add_subplot(2, 3, 3)
R_vals_case1 = [key_ratio(l0_B + 10, l0_B, p_A) for l0_B in l0_B_range_dense]
R_vals_case2 = [key_ratio(2*l0_B, l0_B, p_A) for l0_B in l0_B_range_dense]
R_vals_case3 = [key_ratio(l0_B**1.5, l0_B, p_A) for l0_B in l0_B_range_dense]

ax3.plot(R_vals_case1, max_mods_case1, 'b-', linewidth=2, label='Case 1')
ax3.plot(R_vals_case2, max_mods_case2, 'g-', linewidth=2, label='Case 2')
ax3.plot(R_vals_case3, max_mods_case3, 'r-', linewidth=2, label='Case 3')
ax3.axhline(y=1, color='black', linestyle=':', alpha=0.5)
ax3.set_xlabel('Key Ratio R', fontsize=11)
ax3.set_ylabel(r'max|$\lambda$|', fontsize=11)
ax3.set_title('Parametric: R vs max|λ|', fontsize=12)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 5)

# Row 2: Root trajectories
cases = [
    ("Case 1: R → 0\nRoots → α_j", lambda l0_B: (l0_B + 10, l0_B), 'blue', [30, 100, 300, 1000]),
    ("Case 2: R → const\nRoots → intermediate", lambda l0_B: (2 * l0_B, l0_B), 'green', [30, 100, 300, 1000]),
    ("Case 3: R → ∞\nRoots → unity roots", lambda l0_B: (l0_B**1.5, l0_B), 'red', [20, 50, 100, 200]),
]

for idx, (title, get_l0s, color, l0_B_vals) in enumerate(cases):
    ax = fig.add_subplot(2, 3, idx + 4)
    ax.plot(unit_circle_x, unit_circle_y, 'k--', alpha=0.3, linewidth=1)
    
    # A roots
    for r in A_roots:
        ax.plot(r.real, r.imag, 'ko', markersize=10, zorder=100)
    
    # Unity roots for Case 3
    if idx == 2:
        for r in l_A_unity:
            ax.plot(r.real, r.imag, 'r*', markersize=15, zorder=100)
    
    # Root trajectories
    cmap = plt.cm.viridis
    for i, l0_B in enumerate(l0_B_vals):
        l0_A, l0_B_actual = get_l0s(l0_B)
        R = key_ratio(l0_A, l0_B_actual, p_A)
        F_coeffs = compute_F_coefficients(l_A, l_B, l0_A, l0_B_actual, p_A)
        F_roots = find_roots(F_coeffs)
        c = cmap(i / (len(l0_B_vals) - 1))
        size = 5 + 5 * (i / (len(l0_B_vals) - 1))
        ax.plot(F_roots.real, F_roots.imag, 'o', color=c, markersize=size, 
                alpha=0.8, label=f'$l_0^B$={l0_B}, R={R:.2f}')
    
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11, color=color)
    ax.set_xlabel('Re(λ)')
    ax.set_ylabel('Im(λ)')
    ax.legend(loc='lower left', fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('three_cases_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nSaved three_cases_analysis.png")