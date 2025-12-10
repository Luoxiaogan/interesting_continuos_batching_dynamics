import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# Parameters
l_A, l_B = 2, 4
p, q = 0.5, 0.5

def get_F_roots(l_0):
    """Get roots of F(λ) for given l_0"""
    coeffs = [l_0+1, l_0+2, q*(l_0+3), q*(l_0+4)]
    return np.roots(coeffs)

def get_stability_normal(l_0):
    """
    Get normal vector to stable manifold in D-space.
    Stability condition: λ_2λ_3·D_0 - (λ_2+λ_3)·D_1 + D_2 = 0
    Normal vector: n = (λ_2λ_3, -(λ_2+λ_3), 1)
    """
    roots = get_F_roots(l_0)
    idx = np.argmax(np.abs(roots))
    lam1 = roots[idx]  # unstable
    stable_roots = np.delete(roots, idx)
    lam2, lam3 = stable_roots[0], stable_roots[1]
    
    s1 = lam2 + lam3      # sum of stable roots
    s2 = lam2 * lam3      # product of stable roots
    
    # Normal vector (real part, since complex conjugate pairs give real coefficients)
    n = np.array([s2.real, -s1.real, 1.0])
    return n / np.linalg.norm(n), lam1, lam2, lam3

def get_stable_manifold_basis(normal):
    """Get two basis vectors spanning the stable manifold (plane orthogonal to normal)"""
    n = normal
    # Find two vectors orthogonal to n
    if np.abs(n[0]) < 0.9:
        v1 = np.cross(n, [1, 0, 0])
    else:
        v1 = np.cross(n, [0, 1, 0])
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(n, v1)
    v2 = v2 / np.linalg.norm(v2)
    return v1, v2

# ============================================================
# Figure 1: Stable manifold evolution with l_0
# ============================================================
fig = plt.figure(figsize=(16, 12))

l_0_values = [2, 5, 10, 50]

for idx, l_0 in enumerate(l_0_values):
    ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
    
    # Get stability condition
    normal, lam1, lam2, lam3 = get_stability_normal(l_0)
    v1, v2 = get_stable_manifold_basis(normal)
    
    # Create mesh for the stable plane
    s = np.linspace(-2, 2, 20)
    t = np.linspace(-2, 2, 20)
    S, T = np.meshgrid(s, t)
    
    # Points on the plane: P = s*v1 + t*v2
    X_plane = S * v1[0] + T * v2[0]
    Y_plane = S * v1[1] + T * v2[1]
    Z_plane = S * v1[2] + T * v2[2]
    
    # Plot stable manifold (plane)
    ax.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.3, color='blue', 
                    label='Stable manifold')
    
    # Plot normal vector (unstable direction)
    origin = np.array([0, 0, 0])
    ax.quiver(*origin, *normal, color='red', arrow_length_ratio=0.1, 
              linewidth=2, label='Unstable direction')
    
    # Plot coordinate axes
    ax.quiver(0, 0, 0, 2, 0, 0, color='gray', alpha=0.5, arrow_length_ratio=0.05)
    ax.quiver(0, 0, 0, 0, 2, 0, color='gray', alpha=0.5, arrow_length_ratio=0.05)
    ax.quiver(0, 0, 0, 0, 0, 2, color='gray', alpha=0.5, arrow_length_ratio=0.05)
    
    # Mark origin (zero solution)
    ax.scatter([0], [0], [0], color='green', s=100, marker='o', label='Zero solution')
    
    # Sample stable initial conditions
    np.random.seed(42)
    n_samples = 20
    for _ in range(n_samples):
        s_rand, t_rand = np.random.uniform(-1.5, 1.5, 2)
        point = s_rand * v1 + t_rand * v2
        ax.scatter(*point, color='blue', s=20, alpha=0.6)
    
    # Sample unstable initial conditions
    for _ in range(10):
        point = np.random.uniform(-1.5, 1.5, 3)
        # Project out the stable component to get a point off the plane
        point_off = point - np.dot(point, v1)*v1 - np.dot(point, v2)*v2
        if np.linalg.norm(point_off) > 0.3:  # ensure it's actually off the plane
            ax.scatter(*point, color='red', s=20, alpha=0.6, marker='x')
    
    ax.set_xlabel(r'$D_0$', fontsize=11)
    ax.set_ylabel(r'$D_1$', fontsize=11)
    ax.set_zlabel(r'$D_2$', fontsize=11)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    
    # Title with eigenvalue info
    ax.set_title(f'$l_0 = {l_0}$\n'
                 f'$\\lambda_1 = {lam1.real:.3f}$ ($|\\lambda_1| = {np.abs(lam1):.3f}$)\n'
                 f'$|\\lambda_{{2,3}}| = {np.abs(lam2):.3f}$', fontsize=11)

fig.suptitle(f'Stable Manifold in $D$-space: $l_A={l_A}, l_B={l_B}, p={p}$\n'
             f'Blue plane: Stable manifold ($c_1=0$) | Red arrow: Unstable direction | '
             f'Green dot: Zero solution', fontsize=12)
plt.tight_layout()
plt.savefig('stable_manifold_D_space.png', dpi=150, bbox_inches='tight')

# ============================================================
# Figure 2: Normal vector evolution
# ============================================================
fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

l_0_range = np.arange(2, 101)
normals = []
unstable_eigenvalues = []

for l_0 in l_0_range:
    n, lam1, _, _ = get_stability_normal(l_0)
    normals.append(n)
    unstable_eigenvalues.append(lam1)

normals = np.array(normals)
unstable_eigenvalues = np.array(unstable_eigenvalues)

# Plot normal vector components
ax1 = axes[0]
ax1.plot(l_0_range, normals[:, 0], 'b-', linewidth=2, label=r'$n_0$ (coeff of $D_0$)')
ax1.plot(l_0_range, normals[:, 1], 'r-', linewidth=2, label=r'$n_1$ (coeff of $D_1$)')
ax1.plot(l_0_range, normals[:, 2], 'g-', linewidth=2, label=r'$n_2$ (coeff of $D_2$)')
ax1.set_xlabel(r'$l_0$', fontsize=12)
ax1.set_ylabel('Normal vector components', fontsize=12)
ax1.set_title('Evolution of Stable Manifold Normal Vector', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot unstable eigenvalue
ax2 = axes[1]
ax2.plot(l_0_range, np.abs(unstable_eigenvalues), 'b-', linewidth=2)
ax2.axhline(y=1, color='red', linestyle='--', linewidth=1.5, label='Stability boundary')
ax2.fill_between(l_0_range, 1, np.abs(unstable_eigenvalues), alpha=0.3, color='red')
ax2.set_xlabel(r'$l_0$', fontsize=12)
ax2.set_ylabel(r'$|\lambda_1|$', fontsize=12)
ax2.set_title('Unstable Eigenvalue Modulus vs $l_0$', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

fig2.suptitle(f'Stable Manifold Characterization: $l_A={l_A}, l_B={l_B}, p={p}$', fontsize=13)
plt.tight_layout()
plt.savefig('stable_manifold_evolution.png', dpi=150, bbox_inches='tight')

# ============================================================
# Figure 3: 3D trajectory visualization
# ============================================================
fig3 = plt.figure(figsize=(16, 6))

def simulate_D(D_init, l_0, n_steps=50):
    """Simulate the D recurrence"""
    D = list(D_init)
    for t in range(3, n_steps):
        D_new = -((l_0+2)*D[t-1] + q*(l_0+3)*D[t-2] + q*(l_0+4)*D[t-3]) / (l_0+1)
        D.append(D_new)
    return np.array(D)

l_0 = 10
normal, lam1, lam2, lam3 = get_stability_normal(l_0)
v1, v2 = get_stable_manifold_basis(normal)

# Left: Stable trajectory
ax1 = fig3.add_subplot(1, 2, 1, projection='3d')

# Plot stable manifold
s = np.linspace(-2, 2, 20)
t = np.linspace(-2, 2, 20)
S, T = np.meshgrid(s, t)
X_plane = S * v1[0] + T * v2[0]
Y_plane = S * v1[1] + T * v2[1]
Z_plane = S * v1[2] + T * v2[2]
ax1.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.2, color='blue')

# Stable initial condition
D0, D1 = 0.5, -0.3
D2 = -(normal[0]*D0 + normal[1]*D1) / normal[2] * np.linalg.norm(normal)
# Recompute using actual stability condition
s2 = (lam2 * lam3).real
s1 = (lam2 + lam3).real
D2 = (s2*D0 - s1*D1)  # From: s2*D0 - s1*D1 + D2 = 0 => D2 = -s2*D0 + s1*D1

D_stable = [D0, D1, D2]
D_traj = simulate_D(D_stable, l_0, 30)

# Plot trajectory
ax1.plot(D_traj[:-2], D_traj[1:-1], D_traj[2:], 'g-', linewidth=2, label='Trajectory')
ax1.scatter(*D_stable, color='green', s=100, marker='o', label='Initial condition')
ax1.scatter([0], [0], [0], color='black', s=100, marker='*', label='Origin (attractor)')

ax1.set_xlabel(r'$D_t$')
ax1.set_ylabel(r'$D_{t+1}$')
ax1.set_zlabel(r'$D_{t+2}$')
ax1.set_title(f'Stable Initial Condition ($l_0={l_0}$)\nTrajectory converges to origin', fontsize=11)
ax1.legend(loc='upper left', fontsize=9)

# Right: Unstable trajectory
ax2 = fig3.add_subplot(1, 2, 2, projection='3d')

# Plot stable manifold
ax2.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.2, color='blue')

# Unstable initial condition (off the plane)
D_unstable = [0.5, -0.3, 0.2]  # Not satisfying stability condition
D_traj_unstable = simulate_D(D_unstable, l_0, 20)

# Clip for visualization
clip_idx = np.where(np.abs(D_traj_unstable) > 5)[0]
if len(clip_idx) > 0:
    end_idx = clip_idx[0]
else:
    end_idx = len(D_traj_unstable)

ax2.plot(D_traj_unstable[:end_idx-2], D_traj_unstable[1:end_idx-1], 
         D_traj_unstable[2:end_idx], 'r-', linewidth=2, label='Trajectory')
ax2.scatter(*D_unstable, color='red', s=100, marker='o', label='Initial condition')
ax2.scatter([0], [0], [0], color='black', s=100, marker='*')

ax2.set_xlabel(r'$D_t$')
ax2.set_ylabel(r'$D_{t+1}$')
ax2.set_zlabel(r'$D_{t+2}$')
ax2.set_xlim(-3, 3)
ax2.set_ylim(-3, 3)
ax2.set_zlim(-3, 3)
ax2.set_title(f'Unstable Initial Condition ($l_0={l_0}$)\nTrajectory diverges', fontsize=11)
ax2.legend(loc='upper left', fontsize=9)

fig3.suptitle(f'Phase Space Trajectories: $l_A={l_A}, l_B={l_B}, p={p}$\n'
              f'Blue plane: Stable manifold | Stability condition: '
              f'${(lam2*lam3).real:.3f} D_0 {(-(lam2+lam3)).real:+.3f} D_1 + D_2 = 0$', 
              fontsize=12)
plt.tight_layout()
plt.savefig('phase_space_trajectories.png', dpi=150, bbox_inches='tight')

# ============================================================
# Figure 4: Angle between stable manifold and coordinate planes
# ============================================================
fig4 = plt.figure(figsize=(12, 5))

ax1 = fig4.add_subplot(1, 2, 1)

l_0_range = np.arange(2, 201)
angles_xy = []  # angle with D_0-D_1 plane (normal = [0,0,1])
angles_xz = []  # angle with D_0-D_2 plane (normal = [0,1,0])
angles_yz = []  # angle with D_1-D_2 plane (normal = [1,0,0])

for l_0 in l_0_range:
    n, _, _, _ = get_stability_normal(l_0)
    # Angle between planes = angle between normals
    angles_xy.append(np.arccos(np.abs(n[2])) * 180 / np.pi)
    angles_xz.append(np.arccos(np.abs(n[1])) * 180 / np.pi)
    angles_yz.append(np.arccos(np.abs(n[0])) * 180 / np.pi)

ax1.plot(l_0_range, angles_xy, 'b-', linewidth=2, label=r'Angle with $D_0$-$D_1$ plane')
ax1.plot(l_0_range, angles_xz, 'r-', linewidth=2, label=r'Angle with $D_0$-$D_2$ plane')
ax1.plot(l_0_range, angles_yz, 'g-', linewidth=2, label=r'Angle with $D_1$-$D_2$ plane')
ax1.set_xlabel(r'$l_0$', fontsize=12)
ax1.set_ylabel('Angle (degrees)', fontsize=12)
ax1.set_title('Stable Manifold Orientation vs $l_0$', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Right: coefficients of stability condition
ax2 = fig4.add_subplot(1, 2, 2)

coeff_D0 = []
coeff_D1 = []

for l_0 in l_0_range:
    roots = get_F_roots(l_0)
    idx = np.argmax(np.abs(roots))
    stable_roots = np.delete(roots, idx)
    lam2, lam3 = stable_roots[0], stable_roots[1]
    s2 = (lam2 * lam3).real
    s1 = (lam2 + lam3).real
    coeff_D0.append(s2)
    coeff_D1.append(-s1)

ax2.plot(l_0_range, coeff_D0, 'b-', linewidth=2, label=r'$\lambda_2\lambda_3$ (coeff of $D_0$)')
ax2.plot(l_0_range, coeff_D1, 'r-', linewidth=2, label=r'$-(\lambda_2+\lambda_3)$ (coeff of $D_1$)')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel(r'$l_0$', fontsize=12)
ax2.set_ylabel('Coefficient value', fontsize=12)
ax2.set_title('Stability Condition Coefficients vs $l_0$', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

fig4.suptitle(f'Stable Manifold Geometry: $l_A={l_A}, l_B={l_B}, p={p}$\n'
              f'Stability condition: $\\lambda_2\\lambda_3 \\cdot D_0 - (\\lambda_2+\\lambda_3) \\cdot D_1 + D_2 = 0$',
              fontsize=12)
plt.tight_layout()
plt.savefig('stable_manifold_geometry.png', dpi=150, bbox_inches='tight')

plt.show()

print("Saved figures:")
print("  - stable_manifold_D_space.png: 3D view of stable manifold for different l_0")
print("  - stable_manifold_evolution.png: Normal vector and eigenvalue evolution")
print("  - phase_space_trajectories.png: Stable vs unstable trajectories")
print("  - stable_manifold_geometry.png: Manifold orientation analysis")