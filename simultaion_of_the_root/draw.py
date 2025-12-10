import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial as P

# Parameters
l_A, l_B = 2, 6
p = 0.5
q = 1 - p
g = np.gcd(l_A, l_B)  # g = 2

def get_A_coeffs():
    """
    A(λ) = λ^{l_B-1} + Σ_{m=1}^{l_A-1} λ^{l_B-1-m} + q·Σ_{m=l_A}^{l_B-1} λ^{l_B-1-m}
    """
    coeffs = [0] * l_B  # coefficients for λ^0 to λ^{l_B-1}
    coeffs[l_B - 1] = 1  # λ^{l_B-1}
    for m in range(1, l_A):
        coeffs[l_B - 1 - m] += 1
    for m in range(l_A, l_B):
        coeffs[l_B - 1 - m] += q
    return coeffs

def get_F_coeffs(l_0):
    """
    F(λ) = (l_0+1)λ^{l_B-1} + Σ_{m=1}^{l_A-1}(l_0+m+1)λ^{l_B-1-m} 
           + q·Σ_{m=l_A}^{l_B-1}(l_0+m+1)λ^{l_B-1-m}
    """
    coeffs = [0] * l_B
    coeffs[l_B - 1] = l_0 + 1
    for m in range(1, l_A):
        coeffs[l_B - 1 - m] += (l_0 + m + 1)
    for m in range(l_A, l_B):
        coeffs[l_B - 1 - m] += q * (l_0 + m + 1)
    return coeffs

# Get roots of A(λ)
A_coeffs = get_A_coeffs()
A_roots = np.roots(A_coeffs[::-1])

# Get roots of F(λ) for different l_0 values
l_0_values = [1, 2, 5, 10, 20, 50, 100, 500]
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(l_0_values)))

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ============== Left plot: Root trajectories ==============
ax1 = axes[0]

# Draw unit circle
theta = np.linspace(0, 2*np.pi, 100)
ax1.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, label='Unit circle')

# Draw A(λ) roots
ax1.scatter(A_roots.real, A_roots.imag, s=150, c='red', marker='*', 
            zorder=10, label=r'$A(\lambda)$ roots (limit)', edgecolors='darkred')

# Draw F(λ) roots for each l_0
for i, l_0 in enumerate(l_0_values):
    F_coeffs = get_F_coeffs(l_0)
    F_roots = np.roots(F_coeffs[::-1])
    ax1.scatter(F_roots.real, F_roots.imag, s=50, c=[colors[i]], 
                marker='o', alpha=0.8, label=f'$l_0={l_0}$')

# Draw g-th roots of unity (the critical ones on unit circle)
if g > 1:
    for k in range(1, g):
        omega = np.exp(2j * np.pi * k / g)
        ax1.scatter(omega.real, omega.imag, s=200, c='orange', marker='D',
                    zorder=5, edgecolors='darkorange', linewidths=2)
    ax1.scatter([], [], s=200, c='orange', marker='D', 
                label=f'$g$-th roots of unity ($g={g}$)', edgecolors='darkorange')

ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)
ax1.set_aspect('equal')
ax1.axhline(y=0, color='gray', linewidth=0.5)
ax1.axvline(x=0, color='gray', linewidth=0.5)
ax1.set_xlabel(r'Re$(\lambda)$', fontsize=12)
ax1.set_ylabel(r'Im$(\lambda)$', fontsize=12)
ax1.set_title(r'Roots of $F(\lambda)$ as $l_0$ increases', fontsize=14)
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)

# ============== Right plot: |λ| vs l_0 ==============
ax2 = axes[1]

l_0_range = np.arange(1, 201)
max_modulus = []
all_moduli = []

for l_0 in l_0_range:
    F_coeffs = get_F_coeffs(l_0)
    F_roots = np.roots(F_coeffs[::-1])
    moduli = np.abs(F_roots)
    max_modulus.append(np.max(moduli))
    all_moduli.append(moduli)

all_moduli = np.array(all_moduli)

# Plot each root's modulus trajectory
for j in range(l_B - 1):
    ax2.plot(l_0_range, all_moduli[:, j], alpha=0.6, linewidth=1.5)

ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, label='$|\\lambda|=1$ (stability boundary)')
ax2.set_xlabel(r'$l_0$', fontsize=12)
ax2.set_ylabel(r'$|\lambda|$', fontsize=12)
ax2.set_title(r'Root moduli vs $l_0$', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1, 200)

# Add caption
fig.suptitle(f'Root Analysis: $l_A={l_A}$, $l_B={l_B}$, $p={p}$, $\\gcd(l_A,l_B)={g}$ (Non-coprime: Unstable)', 
             fontsize=14, y=1.02)

plt.tight_layout()
plt.savefig('root_trajectory.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Parameters: l_A={l_A}, l_B={l_B}, p={p}, gcd={g}")
print(f"\nA(λ) roots: {A_roots}")
print(f"|A roots|: {np.abs(A_roots)}")
print(f"\nFor l_0=100, F(λ) roots: {np.roots(get_F_coeffs(100)[::-1])}")
print(f"Max |root|: {max_modulus[-1]:.6f}")