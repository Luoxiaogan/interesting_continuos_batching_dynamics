import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm

# Parameters
l_A, l_B = 2, 6
p = 0.5
q = 1 - p
g = np.gcd(l_A, l_B)

def A(lam):
    """A(λ) = λ^{l_B-1} + Σ_{m=1}^{l_A-1} λ^{l_B-1-m} + q·Σ_{m=l_A}^{l_B-1} λ^{l_B-1-m}"""
    result = lam**(l_B-1)
    for m in range(1, l_A):
        result += lam**(l_B-1-m)
    for m in range(l_A, l_B):
        result += q * lam**(l_B-1-m)
    return result

def A_prime(lam):
    """A'(λ)"""
    result = (l_B-1) * lam**(l_B-2)
    for m in range(1, l_A):
        result += (l_B-1-m) * lam**(l_B-2-m)
    for m in range(l_A, l_B):
        result += q * (l_B-1-m) * lam**(l_B-2-m)
    return result

def g_func(lam, l_0):
    """g(λ) = A(λ) / λ^{l_0+l_B}"""
    with np.errstate(divide='ignore', invalid='ignore'):
        return A(lam) / (lam**(l_0 + l_B))

def get_A_roots():
    """Get roots of A(λ)"""
    coeffs = [0] * l_B
    coeffs[l_B - 1] = 1
    for m in range(1, l_A):
        coeffs[l_B - 1 - m] += 1
    for m in range(l_A, l_B):
        coeffs[l_B - 1 - m] += q
    return np.roots(coeffs[::-1])

def get_F_roots(l_0):
    """Get roots of F(λ) = critical points of g(λ)"""
    coeffs = [0] * l_B
    coeffs[l_B - 1] = l_0 + 1
    for m in range(1, l_A):
        coeffs[l_B - 1 - m] += (l_0 + m + 1)
    for m in range(l_A, l_B):
        coeffs[l_B - 1 - m] += q * (l_0 + m + 1)
    return np.roots(coeffs[::-1])

# Create mesh grid for complex plane
x = np.linspace(-1.5, 1.5, 300)
y = np.linspace(-1.5, 1.5, 300)
X, Y = np.meshgrid(x, y)
Z_complex = X + 1j * Y

# Get roots
A_roots = get_A_roots()

# Different l_0 values to show evolution
l_0_values = [2, 5, 10, 50]

# ============== Figure 1: 3D surface plots ==============
fig1 = plt.figure(figsize=(16, 12))

for idx, l_0 in enumerate(l_0_values):
    ax = fig1.add_subplot(2, 2, idx + 1, projection='3d')
    
    # Compute |g(λ)|
    G = np.abs(g_func(Z_complex, l_0))
    G = np.clip(G, 1e-3, 10)  # Clip for visualization
    G = np.log10(G)  # Log scale
    
    # Plot surface
    surf = ax.plot_surface(X, Y, G, cmap=cm.viridis, alpha=0.8,
                           linewidth=0, antialiased=True)
    
    # Get F roots (critical points)
    F_roots = get_F_roots(l_0)
    
    # Mark A(λ) zeros (zeros of g)
    for root in A_roots:
        ax.scatter(root.real, root.imag, -3, c='red', s=100, marker='*', 
                   label='$A(\\lambda)=0$' if root == A_roots[0] else '')
    
    # Mark F(λ) zeros (critical points of g)
    for root in F_roots:
        g_val = np.log10(np.clip(np.abs(g_func(root, l_0)), 1e-3, 10))
        ax.scatter(root.real, root.imag, g_val, c='orange', s=100, marker='D',
                   label='$F(\\lambda)=0$' if root == F_roots[0] else '')
    
    # Draw unit circle on z=-3 plane
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), -3*np.ones_like(theta), 
            'k--', alpha=0.5, linewidth=2)
    
    ax.set_xlabel(r'Re$(\lambda)$')
    ax.set_ylabel(r'Im$(\lambda)$')
    ax.set_zlabel(r'$\log_{10}|g(\lambda)|$')
    ax.set_title(f'$l_0 = {l_0}$', fontsize=14)
    ax.set_zlim(-3, 2)
    
fig1.suptitle(f'3D Surface of $|g(\\lambda)| = |A(\\lambda)/\\lambda^{{l_0+l_B}}|$\n'
              f'$l_A={l_A}$, $l_B={l_B}$, $p={p}$, $\\gcd={g}$', fontsize=14)
plt.tight_layout()
plt.savefig('g_function_3d.png', dpi=150, bbox_inches='tight')

# ============== Figure 2: Contour plots with zeros and critical points ==============
fig2, axes = plt.subplots(2, 2, figsize=(14, 12))

for idx, l_0 in enumerate(l_0_values):
    ax = axes[idx // 2, idx % 2]
    
    # Compute |g(λ)|
    G = np.abs(g_func(Z_complex, l_0))
    G = np.clip(G, 1e-3, 100)
    
    # Contour plot (log scale)
    levels = np.logspace(-2, 2, 20)
    contour = ax.contourf(X, Y, G, levels=levels, norm=LogNorm(), cmap='viridis', alpha=0.8)
    ax.contour(X, Y, G, levels=levels, colors='white', linewidths=0.5, alpha=0.3)
    
    # Unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'w--', linewidth=2, label='Unit circle')
    
    # A(λ) zeros
    ax.scatter(A_roots.real, A_roots.imag, s=150, c='red', marker='*', 
               edgecolors='darkred', linewidths=1.5, zorder=10, label=r'$A(\lambda)=0$ (zeros of $g$)')
    
    # F(λ) zeros = critical points of g
    F_roots = get_F_roots(l_0)
    ax.scatter(F_roots.real, F_roots.imag, s=120, c='orange', marker='D',
               edgecolors='darkorange', linewidths=1.5, zorder=10, label=r'$F(\lambda)=0$ (critical pts)')
    
    # g-th roots of unity
    if g > 1:
        for k in range(1, g):
            omega = np.exp(2j * np.pi * k / g)
            ax.scatter(omega.real, omega.imag, s=200, c='cyan', marker='s',
                       edgecolors='blue', linewidths=2, zorder=5)
        ax.scatter([], [], s=200, c='cyan', marker='s', edgecolors='blue', 
                   label=f'$g$-th roots of unity')
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_xlabel(r'Re$(\lambda)$', fontsize=11)
    ax.set_ylabel(r'Im$(\lambda)$', fontsize=11)
    ax.set_title(f'$l_0 = {l_0}$', fontsize=14)
    ax.legend(loc='upper left', fontsize=8)
    
    # Colorbar
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
    cbar.set_label(r'$|g(\lambda)|$', fontsize=10)

fig2.suptitle(f'Contour of $|g(\\lambda)|$ with Zeros and Critical Points\n'
              f'$l_A={l_A}$, $l_B={l_B}$, $p={p}$, $\\gcd={g}$', fontsize=14)
plt.tight_layout()
plt.savefig('g_function_contour.png', dpi=150, bbox_inches='tight')

# ============== Figure 3: Root trajectory + phase portrait ==============
fig3, axes = plt.subplots(1, 3, figsize=(18, 6))

# Left: Root trajectory
ax1 = axes[0]
theta = np.linspace(0, 2*np.pi, 100)
ax1.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, linewidth=2, label='Unit circle')

l_0_range = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100, 200]
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(l_0_range)))

for i, l_0 in enumerate(l_0_range):
    F_roots = get_F_roots(l_0)
    ax1.scatter(F_roots.real, F_roots.imag, s=40, c=[colors[i]], alpha=0.8)

# Connect trajectories
l_0_fine = np.arange(1, 201)
all_roots = np.array([get_F_roots(l_0) for l_0 in l_0_fine])
for j in range(l_B - 1):
    ax1.plot(all_roots[:, j].real, all_roots[:, j].imag, 'gray', alpha=0.3, linewidth=1)

ax1.scatter(A_roots.real, A_roots.imag, s=200, c='red', marker='*', 
            edgecolors='darkred', zorder=10, label=r'$A(\lambda)$ roots (limit)')

if g > 1:
    for k in range(1, g):
        omega = np.exp(2j * np.pi * k / g)
        ax1.scatter(omega.real, omega.imag, s=200, c='cyan', marker='s',
                    edgecolors='blue', linewidths=2, zorder=5)

ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)
ax1.set_aspect('equal')
ax1.set_xlabel(r'Re$(\lambda)$', fontsize=12)
ax1.set_ylabel(r'Im$(\lambda)$', fontsize=12)
ax1.set_title('Root Trajectories as $l_0$ increases', fontsize=14)
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)

# Middle: |λ| vs l_0
ax2 = axes[1]
l_0_range_full = np.arange(1, 201)
all_moduli = np.array([np.abs(get_F_roots(l_0)) for l_0 in l_0_range_full])

for j in range(l_B - 1):
    ax2.plot(l_0_range_full, all_moduli[:, j], linewidth=2, alpha=0.8)

ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Stability boundary')
ax2.axhline(y=np.max(np.abs(A_roots)), color='green', linestyle=':', linewidth=2, 
            label=f'$\\max|A$ roots$|={np.max(np.abs(A_roots)):.3f}$')
ax2.set_xlabel(r'$l_0$', fontsize=12)
ax2.set_ylabel(r'$|\lambda|$', fontsize=12)
ax2.set_title(r'Root Moduli vs $l_0$', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1, 200)

# Right: Phase of g(λ) for fixed l_0
ax3 = axes[2]
l_0_show = 10
G_phase = np.angle(g_func(Z_complex, l_0_show))

phase_plot = ax3.contourf(X, Y, G_phase, levels=50, cmap='twilight', alpha=0.8)
ax3.contour(X, Y, np.abs(g_func(Z_complex, l_0_show)), levels=[0.1, 0.5, 1, 2, 5], 
            colors='white', linewidths=1, alpha=0.5)
ax3.plot(np.cos(theta), np.sin(theta), 'w--', linewidth=2)

F_roots_show = get_F_roots(l_0_show)
ax3.scatter(A_roots.real, A_roots.imag, s=150, c='red', marker='*', edgecolors='white', linewidths=1)
ax3.scatter(F_roots_show.real, F_roots_show.imag, s=120, c='orange', marker='D', edgecolors='white', linewidths=1)

ax3.set_xlim(-1.5, 1.5)
ax3.set_ylim(-1.5, 1.5)
ax3.set_aspect('equal')
ax3.set_xlabel(r'Re$(\lambda)$', fontsize=12)
ax3.set_ylabel(r'Im$(\lambda)$', fontsize=12)
ax3.set_title(f'Phase of $g(\\lambda)$, $l_0={l_0_show}$', fontsize=14)
cbar = plt.colorbar(phase_plot, ax=ax3, shrink=0.8)
cbar.set_label(r'$\arg(g(\lambda))$', fontsize=10)

fig3.suptitle(f'Complete Analysis: $l_A={l_A}$, $l_B={l_B}$, $p={p}$, $\\gcd={g}$ (Non-coprime → Unstable)', 
              fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('complete_analysis.png', dpi=150, bbox_inches='tight')

# ============== Figure 4: Compare coprime vs non-coprime ==============
fig4, axes = plt.subplots(2, 2, figsize=(14, 12))

cases = [
    (2, 6, 'Non-coprime: $\\gcd(2,6)=2$'),  # gcd = 2
    (2, 5, 'Coprime: $\\gcd(2,5)=1$'),       # gcd = 1
]

for case_idx, (lA, lB, title) in enumerate(cases):
    # Redefine A and F for this case
    def A_case(lam, lA=lA, lB=lB):
        result = lam**(lB-1)
        for m in range(1, lA):
            result += lam**(lB-1-m)
        for m in range(lA, lB):
            result += q * lam**(lB-1-m)
        return result
    
    def get_A_roots_case(lA=lA, lB=lB):
        coeffs = [0] * lB
        coeffs[lB - 1] = 1
        for m in range(1, lA):
            coeffs[lB - 1 - m] += 1
        for m in range(lA, lB):
            coeffs[lB - 1 - m] += q
        return np.roots(coeffs[::-1])
    
    def get_F_roots_case(l_0, lA=lA, lB=lB):
        coeffs = [0] * lB
        coeffs[lB - 1] = l_0 + 1
        for m in range(1, lA):
            coeffs[lB - 1 - m] += (l_0 + m + 1)
        for m in range(lA, lB):
            coeffs[lB - 1 - m] += q * (l_0 + m + 1)
        return np.roots(coeffs[::-1])
    
    A_roots_case = get_A_roots_case()
    gcd_case = np.gcd(lA, lB)
    
    # Left column: Root trajectory
    ax1 = axes[case_idx, 0]
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, linewidth=2)
    
    l_0_fine = np.arange(1, 201)
    all_roots_case = np.array([get_F_roots_case(l_0) for l_0 in l_0_fine])
    for j in range(lB - 1):
        ax1.plot(all_roots_case[:, j].real, all_roots_case[:, j].imag, 'gray', alpha=0.5, linewidth=1)
    
    ax1.scatter(A_roots_case.real, A_roots_case.imag, s=200, c='red', marker='*', 
                edgecolors='darkred', zorder=10, label=r'$A(\lambda)$ roots')
    
    if gcd_case > 1:
        for k in range(1, gcd_case):
            omega = np.exp(2j * np.pi * k / gcd_case)
            ax1.scatter(omega.real, omega.imag, s=200, c='cyan', marker='s',
                        edgecolors='blue', linewidths=2, zorder=5)
    
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.set_xlabel(r'Re$(\lambda)$', fontsize=11)
    ax1.set_ylabel(r'Im$(\lambda)$', fontsize=11)
    ax1.set_title(f'{title}\nRoot Trajectories', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Right column: |λ| vs l_0
    ax2 = axes[case_idx, 1]
    all_moduli_case = np.array([np.abs(get_F_roots_case(l_0)) for l_0 in l_0_fine])
    
    for j in range(lB - 1):
        ax2.plot(l_0_fine, all_moduli_case[:, j], linewidth=2, alpha=0.8)
    
    ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, label='$|\\lambda|=1$')
    ax2.fill_between(l_0_fine, 1, 1.5, alpha=0.2, color='red', label='Unstable region')
    ax2.set_xlabel(r'$l_0$', fontsize=11)
    ax2.set_ylabel(r'$|\lambda|$', fontsize=11)
    ax2.set_title(f'{title}\nRoot Moduli', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, 200)
    ax2.set_ylim(0.4, 1.3)

fig4.suptitle(f'Comparison: Coprime vs Non-coprime ($p={p}$)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('coprime_comparison.png', dpi=150, bbox_inches='tight')

plt.show()

print("Saved figures:")
print("  - g_function_3d.png")
print("  - g_function_contour.png") 
print("  - complete_analysis.png")
print("  - coprime_comparison.png")