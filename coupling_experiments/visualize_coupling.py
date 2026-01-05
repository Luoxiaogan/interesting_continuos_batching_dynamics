"""
Visualization functions for coupling experiments.

Generates comparison plots showing Theory vs Simulation trajectories.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path


def plot_comparison(trajectory: List[Dict],
                   output_path: Optional[Path] = None,
                   title: str = "Theory vs Simulation Comparison"):
    """
    Plot admission/eviction comparison between Theory and Simulation.

    Args:
        trajectory: List of dicts with batch data
        output_path: Path to save figure (optional)
        title: Plot title
    """
    batches = [row['batch'] for row in trajectory]
    theory_adm = [row['theory_admission'] for row in trajectory]
    sim_adm = [row['sim_admission'] for row in trajectory]
    sim_evic = [row['sim_eviction_total'] for row in trajectory]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Theory admission line (can be negative)
    ax.plot(batches, theory_adm, 'b-', linewidth=2, label='Theory Admission', marker='o', markersize=4)

    # Simulation admission line (>= 0)
    ax.plot(batches, sim_adm, 'g-', linewidth=2, label='Simulation Admission', marker='s', markersize=4)

    # Simulation eviction as negative bars
    evic_negative = [-e for e in sim_evic]
    ax.bar(batches, evic_negative, color='red', alpha=0.6, label='Simulation Eviction (negative)')

    # Zero line
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    ax.set_xlabel('Batch Index', fontsize=12)
    ax.set_ylabel('Number of Requests', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Highlight divergence region
    # Find batches where theory < 0 or eviction > 0
    diverge_batches = [b for i, b in enumerate(batches)
                       if theory_adm[i] < 0 or sim_evic[i] > 0.01]
    if diverge_batches:
        ax.axvspan(min(diverge_batches) - 0.5, max(diverge_batches) + 0.5,
                  alpha=0.1, color='yellow', label='Divergence Region')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {output_path}")

    plt.close()


def plot_eviction_detail(trajectory: List[Dict],
                        max_stage: int,
                        output_path: Optional[Path] = None,
                        title: str = "Eviction by Stage"):
    """
    Plot eviction distribution by stage as stacked bar chart.

    Args:
        trajectory: List of dicts with batch data
        max_stage: Maximum stage number
        output_path: Path to save figure (optional)
        title: Plot title
    """
    batches = [row['batch'] for row in trajectory]

    # Extract eviction by stage
    eviction_by_stage = {}
    for stage in range(max_stage + 1):
        key = f'sim_eviction_stage{stage}'
        eviction_by_stage[stage] = [row.get(key, 0.0) for row in trajectory]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Stacked bar chart
    bottom = np.zeros(len(batches))
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, max_stage + 1))

    for stage in range(max_stage + 1):
        values = eviction_by_stage[stage]
        ax.bar(batches, values, bottom=bottom, color=colors[stage],
               label=f'Stage {stage}', alpha=0.8)
        bottom = bottom + np.array(values)

    ax.set_xlabel('Batch Index', fontsize=12)
    ax.set_ylabel('Evicted Requests', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', title='Stage')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved eviction detail plot to {output_path}")

    plt.close()


def plot_G_comparison(trajectory: List[Dict],
                      output_path: Optional[Path] = None,
                      title: str = "G (max-min) Comparison"):
    """
    Plot G metric comparison between Theory and Simulation.

    G = max(state values) - min(state values)
    This measures the "imbalance" in the state vector.

    Args:
        trajectory: List of dicts with batch data (must include 'theory_G' and 'sim_G')
        output_path: Path to save figure (optional)
        title: Plot title
    """
    batches = [row['batch'] for row in trajectory]
    theory_G = [row.get('theory_G', 0.0) for row in trajectory]
    sim_G = [row.get('sim_G', 0.0) for row in trajectory]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Theory G line
    ax.plot(batches, theory_G, 'b-', linewidth=2, label='Theory G', marker='o', markersize=4)

    # Simulation G line
    ax.plot(batches, sim_G, 'g-', linewidth=2, label='Simulation G', marker='s', markersize=4)

    ax.set_xlabel('Batch Index', fontsize=12)
    ax.set_ylabel('G = max(state) - min(state)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Add equilibrium reference line if G stabilizes
    if len(theory_G) > 10:
        eq_G = np.mean(theory_G[-10:])
        ax.axhline(y=eq_G, color='blue', linestyle=':', linewidth=1, alpha=0.5,
                   label=f'Theory Equilibrium ≈ {eq_G:.2f}')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved G comparison plot to {output_path}")

    plt.close()


def plot_G_decomposed(trajectory: List[Dict],
                      output_path: Optional[Path] = None,
                      title: str = "G Decomposition"):
    """
    Plot G decomposition: 2x4 grid of subplots.

    Top row: G_A, G_B, G_merge (compensated), G_merged_raw (no compensation)
             Each shows Theory (blue) vs Simulation (green)
    Bottom row: Difference (Theory - Simulation) for each metric

    Args:
        trajectory: List of dicts with batch data
                    Must include: theory_G_A, theory_G_B, theory_G_merge, theory_G_merged_raw,
                                  sim_G_A, sim_G_B, sim_G_merge, sim_G_merged_raw
        output_path: Path to save figure (optional)
        title: Plot title
    """
    batches = [row['batch'] for row in trajectory]

    # Extract data
    theory_G_A = [row.get('theory_G_A', 0.0) for row in trajectory]
    theory_G_B = [row.get('theory_G_B', 0.0) for row in trajectory]
    theory_G_merge = [row.get('theory_G_merge', 0.0) for row in trajectory]
    theory_G_merged_raw = [row.get('theory_G_merged_raw', 0.0) for row in trajectory]

    sim_G_A = [row.get('sim_G_A', 0.0) for row in trajectory]
    sim_G_B = [row.get('sim_G_B', 0.0) for row in trajectory]
    sim_G_merge = [row.get('sim_G_merge', 0.0) for row in trajectory]
    sim_G_merged_raw = [row.get('sim_G_merged_raw', 0.0) for row in trajectory]

    # Compute differences (Theory - Simulation)
    diff_G_A = [t - s for t, s in zip(theory_G_A, sim_G_A)]
    diff_G_B = [t - s for t, s in zip(theory_G_B, sim_G_B)]
    diff_G_merge = [t - s for t, s in zip(theory_G_merge, sim_G_merge)]
    diff_G_merged_raw = [t - s for t, s in zip(theory_G_merged_raw, sim_G_merged_raw)]

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))

    # ===== Top row: Theory vs Simulation =====

    # Subplot (0,0): G_A
    ax = axes[0, 0]
    ax.plot(batches, theory_G_A, 'b-', linewidth=2, label='Theory', marker='o', markersize=3)
    ax.plot(batches, sim_G_A, 'g-', linewidth=2, label='Simulation', marker='s', markersize=3)
    ax.set_xlabel('Batch Index', fontsize=11)
    ax.set_ylabel('G_A = max - min', fontsize=11)
    ax.set_title('G_A (Type A only)', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Subplot (0,1): G_B
    ax = axes[0, 1]
    ax.plot(batches, theory_G_B, 'b-', linewidth=2, label='Theory', marker='o', markersize=3)
    ax.plot(batches, sim_G_B, 'g-', linewidth=2, label='Simulation', marker='s', markersize=3)
    ax.set_xlabel('Batch Index', fontsize=11)
    ax.set_ylabel('G_B = max - min', fontsize=11)
    ax.set_title('G_B (Type B only)', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Subplot (0,2): G_merge (compensated)
    ax = axes[0, 2]
    ax.plot(batches, theory_G_merge, 'b-', linewidth=2, label='Theory', marker='o', markersize=3)
    ax.plot(batches, sim_G_merge, 'g-', linewidth=2, label='Simulation', marker='s', markersize=3)
    ax.set_xlabel('Batch Index', fontsize=11)
    ax.set_ylabel('G_merge = max - min', fontsize=11)
    ax.set_title('G_merge (compensated)', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Subplot (0,3): G_merged_raw (no compensation)
    ax = axes[0, 3]
    ax.plot(batches, theory_G_merged_raw, 'b-', linewidth=2, label='Theory', marker='o', markersize=3)
    ax.plot(batches, sim_G_merged_raw, 'g-', linewidth=2, label='Simulation', marker='s', markersize=3)
    ax.set_xlabel('Batch Index', fontsize=11)
    ax.set_ylabel('G_merged_raw = max - min', fontsize=11)
    ax.set_title('G_merged_raw (no compensation)', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # ===== Bottom row: Difference (Theory - Simulation) =====
    # Negative points are marked in red

    def plot_diff_with_negative_highlight(ax, batches, diff_values, title):
        """Plot difference with negative points highlighted in red."""
        # Split into positive/zero and negative
        pos_mask = [d >= 0 for d in diff_values]
        neg_mask = [d < 0 for d in diff_values]

        # Plot line
        ax.plot(batches, diff_values, 'purple', linewidth=2, alpha=0.7)

        # Plot positive/zero points in purple
        pos_batches = [b for b, m in zip(batches, pos_mask) if m]
        pos_values = [v for v, m in zip(diff_values, pos_mask) if m]
        ax.scatter(pos_batches, pos_values, c='purple', s=20, zorder=5)

        # Plot negative points in red
        neg_batches = [b for b, m in zip(batches, neg_mask) if m]
        neg_values = [v for v, m in zip(diff_values, neg_mask) if m]
        ax.scatter(neg_batches, neg_values, c='red', s=30, zorder=6, label='< 0')

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel('Batch Index', fontsize=11)
        ax.set_ylabel('Theory - Simulation', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        if neg_batches:
            ax.legend(loc='best')

    # Subplot (1,0): Diff G_A
    plot_diff_with_negative_highlight(axes[1, 0], batches, diff_G_A, 'Δ G_A')

    # Subplot (1,1): Diff G_B
    plot_diff_with_negative_highlight(axes[1, 1], batches, diff_G_B, 'Δ G_B')

    # Subplot (1,2): Diff G_merge
    plot_diff_with_negative_highlight(axes[1, 2], batches, diff_G_merge, 'Δ G_merge')

    # Subplot (1,3): Diff G_merged_raw
    plot_diff_with_negative_highlight(axes[1, 3], batches, diff_G_merged_raw, 'Δ G_merged_raw')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved G decomposed plot to {output_path}")

    plt.close()


def plot_state_comparison(theory_states: List[Dict],
                         sim_states: List[Dict],
                         l0: int,
                         output_path: Optional[Path] = None,
                         title: str = "State Comparison"):
    """
    Plot state evolution comparison (optional, for debugging).

    Args:
        theory_states: Theory simulator state history
        sim_states: Simulation state history (needs conversion)
        l0: Initial length
        output_path: Path to save figure
        title: Plot title
    """
    # This is a more detailed plot for debugging
    # Can be expanded as needed
    pass


def plot_roots_analysis(l0: int, l_A: int, l_B: int,
                        lambda_A: float, lambda_B: float,
                        output_path: Optional[Path] = None,
                        title: str = "Roots Analysis"):
    """
    Plot characteristic equation and limit equation roots on complex plane.

    Characteristic equation F(λ) = 0:
        - |λ| > 1: Red × (unstable)
        - |λ| ≤ 1: Orange ×

    Limit equation (1-λ)A(λ) = 0:
        - |λ| = 1: Blue ○
        - |λ| < 1: Green ○

    Args:
        l0: Initial/prefill length
        l_A: Decode length for Type A
        l_B: Decode length for Type B
        lambda_A: Arrival rate for Type A
        lambda_B: Arrival rate for Type B
        output_path: Path to save figure (optional)
        title: Plot title
    """
    from metrics import compute_characteristic_roots, compute_limit_roots
    import math

    char_roots = compute_characteristic_roots(l0, l_A, l_B, lambda_A, lambda_B)
    limit_roots = compute_limit_roots(l_A, l_B, lambda_A, lambda_B)
    gcd = math.gcd(l_A, l_B)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw unit circle
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'gray', linestyle='--',
            linewidth=1.5, label='Unit Circle')

    # Track legend entries to avoid duplicates
    limit_on_circle_plotted = False
    limit_inside_plotted = False
    char_unstable_plotted = False
    char_stable_plotted = False

    # Plot limit equation roots
    for root in limit_roots:
        norm = abs(root)
        if abs(norm - 1.0) < 1e-6:
            label = 'Limit: |λ|=1' if not limit_on_circle_plotted else None
            ax.scatter(root.real, root.imag, c='blue', s=100, marker='o',
                      zorder=5, label=label)
            limit_on_circle_plotted = True
        else:
            label = 'Limit: |λ|<1' if not limit_inside_plotted else None
            ax.scatter(root.real, root.imag, c='green', s=100, marker='o',
                      zorder=5, label=label)
            limit_inside_plotted = True

    # Plot characteristic equation roots
    for root in char_roots:
        norm = abs(root)
        if norm > 1.0 + 1e-6:
            label = 'Char: |λ|>1 (unstable)' if not char_unstable_plotted else None
            ax.scatter(root.real, root.imag, c='red', s=120, marker='x',
                      linewidths=2, zorder=6, label=label)
            char_unstable_plotted = True
        else:
            label = 'Char: |λ|≤1' if not char_stable_plotted else None
            ax.scatter(root.real, root.imag, c='orange', s=120, marker='x',
                      linewidths=2, zorder=6, label=label)
            char_stable_plotted = True

    # Count roots
    n_limit = len(limit_roots)
    n_char = len(char_roots)
    n_char_unstable = sum(1 for r in char_roots if abs(r) > 1.0 + 1e-6)

    # Set title with parameters
    stability = "Stable (coprime)" if gcd == 1 else f"Unstable (gcd={gcd})"
    ax.set_title(f'{title}\n'
                 f'l₀={l0}, l_A={l_A}, l_B={l_B}, gcd={gcd}\n'
                 f'{stability} | Limit roots: {n_limit}, Char roots: {n_char} '
                 f'(unstable: {n_char_unstable})',
                 fontsize=12)

    ax.set_xlabel('Real', fontsize=11)
    ax.set_ylabel('Imaginary', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)

    # Set axis limits with some padding
    all_roots = np.concatenate([char_roots, limit_roots])
    max_extent = max(1.2, np.max(np.abs(all_roots)) * 1.1)
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved roots analysis plot to {output_path}")

    plt.close()


# ============================================================
# NEW VISUALIZATION FUNCTIONS FOR DELTA ANALYSIS (4.1-4.4)
# ============================================================

def plot_delta_extrema(delta_data: List[Dict],
                       output_path: Optional[Path] = None,
                       title: str = "Delta at Extrema Tracking"):
    """
    Plot Delta values at Simulation's extrema positions.

    Top subplot: Δ_{m^S}(t) and Δ_{n^S}(t) values over time
    Bottom subplot: m^S(t) and n^S(t) positions over time

    This is the KEY plot for verifying Δ_{m^S} ≥ 0 conjecture.

    Args:
        delta_data: List of dicts with delta analysis data
        output_path: Path to save figure (optional)
        title: Plot title
    """
    if not delta_data:
        print("Warning: Empty delta_data, skipping plot_delta_extrema")
        return

    batches = [row['batch'] for row in delta_data]
    delta_at_m_S = [row['delta_at_m_S'] for row in delta_data]
    delta_at_n_S = [row['delta_at_n_S'] for row in delta_data]
    m_S = [row['m_S'] for row in delta_data]
    n_S = [row['n_S'] for row in delta_data]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # ===== Top subplot: Delta values at extrema =====
    ax = axes[0]

    # Plot Δ_{m^S} (should be >= 0)
    ax.plot(batches, delta_at_m_S, 'b-', linewidth=2, label='Δ_{m^S} (at Sim argmax)',
            marker='o', markersize=3)

    # Plot Δ_{n^S} (should be <= 0)
    ax.plot(batches, delta_at_n_S, 'r-', linewidth=2, label='Δ_{n^S} (at Sim argmin)',
            marker='s', markersize=3)

    # Critical reference line: y = 0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, label='y = 0')

    # Highlight violations (Δ_{m^S} < 0)
    violations = [i for i, d in enumerate(delta_at_m_S) if d < 0]
    if violations:
        for v in violations:
            ax.axvspan(batches[v] - 0.5, batches[v] + 0.5, alpha=0.3, color='red')
        ax.scatter([batches[v] for v in violations],
                   [delta_at_m_S[v] for v in violations],
                   c='red', s=100, marker='x', zorder=10, label=f'Violations ({len(violations)})')

    ax.set_ylabel('Delta Value', fontsize=12)
    ax.set_title('Δ at Simulation Extrema (Key: Δ_{m^S} should be ≥ 0)', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Add statistics text
    delta_m_geq_0_rate = sum(1 for d in delta_at_m_S if d >= 0) / len(delta_at_m_S) * 100
    ax.text(0.02, 0.98, f'Δ_{{m^S}} ≥ 0: {delta_m_geq_0_rate:.1f}%',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ===== Bottom subplot: Extrema positions =====
    ax = axes[1]

    # Plot m^S positions (step plot for discrete values)
    ax.step(batches, m_S, 'b-', linewidth=2, label='m^S (Sim argmax)', where='mid')
    ax.scatter(batches, m_S, c='blue', s=15, zorder=5)

    # Plot n^S positions
    ax.step(batches, n_S, 'r-', linewidth=2, label='n^S (Sim argmin)', where='mid')
    ax.scatter(batches, n_S, c='red', s=15, zorder=5)

    ax.set_xlabel('Batch Index', fontsize=12)
    ax.set_ylabel('Stage Position', fontsize=12)
    ax.set_title('Simulation Extrema Positions (m^S = argmax, n^S = argmin)', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Set integer ticks for stage positions
    max_stage = max(max(m_S), max(n_S))
    ax.set_yticks(range(max_stage + 1))

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved delta extrema plot to {output_path}")

    plt.close()


def plot_delta_heatmap(delta_data: List[Dict],
                       output_path: Optional[Path] = None,
                       title: str = "Delta Vector Heatmap"):
    """
    Plot Delta vector as a heatmap (stages × time).

    Colors: Red = positive (Theory > Sim), Blue = negative (Theory < Sim), White = 0

    Args:
        delta_data: List of dicts with delta analysis data (must include 'delta_vec')
        output_path: Path to save figure (optional)
        title: Plot title
    """
    if not delta_data:
        print("Warning: Empty delta_data, skipping plot_delta_heatmap")
        return

    batches = [row['batch'] for row in delta_data]

    # Parse delta_vec strings to numpy array
    def parse_delta_vec(s):
        """Parse '[0.1, -0.2, ...]' string to list of floats."""
        s = s.strip('[]')
        return [float(x) for x in s.split(',')]

    delta_matrix = []
    for row in delta_data:
        delta_vec = row['delta_vec']
        if isinstance(delta_vec, str):
            delta_matrix.append(parse_delta_vec(delta_vec))
        else:
            delta_matrix.append(delta_vec)

    delta_matrix = np.array(delta_matrix).T  # Shape: (stages, time_steps)
    n_stages, n_steps = delta_matrix.shape

    fig, ax = plt.subplots(figsize=(14, 6))

    # Compute symmetric color limits
    max_abs = np.max(np.abs(delta_matrix))
    vmin, vmax = -max_abs, max_abs

    # Plot heatmap
    im = ax.imshow(delta_matrix, aspect='auto', cmap='RdBu_r',
                   vmin=vmin, vmax=vmax,
                   extent=[batches[0] - 0.5, batches[-1] + 0.5, n_stages - 0.5, -0.5])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Δ = Theory - Simulation')

    # Mark extrema positions
    m_S = [row['m_S'] for row in delta_data]
    n_S = [row['n_S'] for row in delta_data]
    ax.scatter(batches, m_S, c='lime', s=10, marker='^', label='m^S (argmax)', alpha=0.7)
    ax.scatter(batches, n_S, c='cyan', s=10, marker='v', label='n^S (argmin)', alpha=0.7)

    ax.set_xlabel('Batch Index', fontsize=12)
    ax.set_ylabel('Stage', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_yticks(range(n_stages))
    ax.legend(loc='upper right')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved delta heatmap to {output_path}")

    plt.close()


def plot_lyapunov_energy(lyapunov_data: List[Dict],
                         output_path: Optional[Path] = None,
                         title: str = "Lyapunov Energy Analysis"):
    """
    Plot Lyapunov energy comparison between Theory and Simulation.

    Top subplot: V^T(t) and V^S(t) on log scale
    Bottom subplot: Energy ratio V^T(t) / V^S(t)

    Args:
        lyapunov_data: List of dicts with 'V_theory' and 'V_sim'
        output_path: Path to save figure (optional)
        title: Plot title
    """
    if not lyapunov_data:
        print("Warning: Empty lyapunov_data, skipping plot_lyapunov_energy")
        return

    batches = [row['batch'] for row in lyapunov_data]
    V_theory = [row.get('V_theory', 0.0) for row in lyapunov_data]
    V_sim = [row.get('V_sim', 0.0) for row in lyapunov_data]

    # Filter out zero/negative values for log scale
    valid_indices = [i for i in range(len(V_theory)) if V_theory[i] > 0 and V_sim[i] > 0]
    if not valid_indices:
        print("Warning: No valid energy data, skipping plot_lyapunov_energy")
        return

    batches_valid = [batches[i] for i in valid_indices]
    V_T_valid = [V_theory[i] for i in valid_indices]
    V_S_valid = [V_sim[i] for i in valid_indices]
    ratio = [V_theory[i] / V_sim[i] for i in valid_indices]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # ===== Top subplot: Energy evolution (log scale) =====
    ax = axes[0]
    ax.semilogy(batches_valid, V_T_valid, 'b-', linewidth=2, label='V^T (Theory)',
                marker='o', markersize=3)
    ax.semilogy(batches_valid, V_S_valid, 'g-', linewidth=2, label='V^S (Simulation)',
                marker='s', markersize=3)

    ax.set_ylabel('Energy V (log scale)', fontsize=12)
    ax.set_title('Lyapunov Energy Evolution', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Add dominance statistic
    V_dominance = sum(1 for i in valid_indices if V_theory[i] >= V_sim[i]) / len(valid_indices) * 100
    ax.text(0.02, 0.02, f'V^T ≥ V^S: {V_dominance:.1f}%',
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ===== Bottom subplot: Energy ratio =====
    ax = axes[1]
    ax.plot(batches_valid, ratio, 'purple', linewidth=2, marker='o', markersize=3)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, label='Ratio = 1')

    # Highlight where ratio < 1 (violations)
    violations = [i for i, r in enumerate(ratio) if r < 1]
    if violations:
        ax.scatter([batches_valid[i] for i in violations],
                   [ratio[i] for i in violations],
                   c='red', s=50, marker='x', zorder=10, label=f'< 1 ({len(violations)})')

    ax.set_xlabel('Batch Index', fontsize=12)
    ax.set_ylabel('V^T / V^S', fontsize=12)
    ax.set_title('Energy Ratio (should be ≥ 1 if Theory dominates)', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Show stable ratio
    if len(ratio) > 10:
        stable_ratio = np.mean(ratio[-10:])
        ax.text(0.98, 0.98, f'Final ratio ≈ {stable_ratio:.2f}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved Lyapunov energy plot to {output_path}")

    plt.close()


def plot_case_distribution(delta_data: List[Dict],
                           output_path: Optional[Path] = None,
                           title: str = "Case Distribution Analysis"):
    """
    Plot Case A/B/C/D distribution.

    Left subplot: Pie chart of overall distribution
    Right subplot: Cumulative case counts over time

    Case definitions:
        A: Δ_{m^S} ≥ 0 AND Δ_{n^S} ≤ 0 (ideal)
        B: Δ_{m^S} ≥ 0 AND Δ_{n^S} > 0
        C: Δ_{m^S} < 0 AND Δ_{n^S} ≤ 0
        D: Δ_{m^S} < 0 AND Δ_{n^S} > 0 (worst)

    Args:
        delta_data: List of dicts with 'case' field
        output_path: Path to save figure (optional)
        title: Plot title
    """
    if not delta_data:
        print("Warning: Empty delta_data, skipping plot_case_distribution")
        return

    batches = [row['batch'] for row in delta_data]
    cases = [row['case'] for row in delta_data]

    # Count cases
    case_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    for c in cases:
        case_counts[c] = case_counts.get(c, 0) + 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ===== Left subplot: Pie chart =====
    ax = axes[0]
    colors = {'A': '#2ecc71', 'B': '#f1c40f', 'C': '#e67e22', 'D': '#e74c3c'}
    labels = []
    sizes = []
    pie_colors = []
    for case in ['A', 'B', 'C', 'D']:
        if case_counts[case] > 0:
            labels.append(f'Case {case}\n({case_counts[case]})')
            sizes.append(case_counts[case])
            pie_colors.append(colors[case])

    if sizes:
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=pie_colors,
                                           autopct='%1.1f%%', startangle=90,
                                           textprops={'fontsize': 10})
        ax.set_title('Case Distribution', fontsize=12)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')

    # Add legend explaining cases
    legend_text = ('Case A: Δ_{m^S}≥0, Δ_{n^S}≤0 (ideal)\n'
                   'Case B: Δ_{m^S}≥0, Δ_{n^S}>0\n'
                   'Case C: Δ_{m^S}<0, Δ_{n^S}≤0\n'
                   'Case D: Δ_{m^S}<0, Δ_{n^S}>0 (worst)')
    ax.text(0.5, -0.15, legend_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    # ===== Right subplot: Cumulative case counts =====
    ax = axes[1]

    # Compute cumulative counts for each case
    cumul = {'A': [], 'B': [], 'C': [], 'D': []}
    running = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    for c in cases:
        running[c] += 1
        for case in ['A', 'B', 'C', 'D']:
            cumul[case].append(running[case])

    # Stacked area plot
    ax.stackplot(batches,
                 cumul['A'], cumul['B'], cumul['C'], cumul['D'],
                 labels=['Case A', 'Case B', 'Case C', 'Case D'],
                 colors=[colors['A'], colors['B'], colors['C'], colors['D']],
                 alpha=0.8)

    ax.set_xlabel('Batch Index', fontsize=12)
    ax.set_ylabel('Cumulative Count', fontsize=12)
    ax.set_title('Cumulative Case Distribution Over Time', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved case distribution plot to {output_path}")

    plt.close()


def plot_G_vs_V_relation(lyapunov_data: List[Dict],
                         delta_data: List[Dict],
                         output_path: Optional[Path] = None,
                         title: str = "G² vs V Relationship"):
    """
    Plot G² vs V scatter plot to explore norm equivalence.

    Args:
        lyapunov_data: List of dicts with 'V_theory', 'V_sim'
        delta_data: List of dicts with 'theory_G_merge', 'sim_G_merge'
        output_path: Path to save figure (optional)
        title: Plot title
    """
    if not lyapunov_data or not delta_data:
        print("Warning: Empty data, skipping plot_G_vs_V_relation")
        return

    # Extract data (aligned by batch)
    V_theory = [row.get('V_theory', 0.0) for row in lyapunov_data]
    V_sim = [row.get('V_sim', 0.0) for row in lyapunov_data]
    G_theory = [row.get('theory_G_merge', 0.0) for row in delta_data]
    G_sim = [row.get('sim_G_merge', 0.0) for row in delta_data]

    # Align lengths
    n = min(len(V_theory), len(G_theory))
    V_theory = V_theory[:n]
    V_sim = V_sim[:n]
    G_theory = G_theory[:n]
    G_sim = G_sim[:n]

    # Filter positive values
    valid_T = [(V_theory[i], G_theory[i]**2) for i in range(n) if V_theory[i] > 0]
    valid_S = [(V_sim[i], G_sim[i]**2) for i in range(n) if V_sim[i] > 0]

    if not valid_T or not valid_S:
        print("Warning: No valid data points, skipping plot_G_vs_V_relation")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot Theory points
    V_T, G2_T = zip(*valid_T)
    ax.scatter(V_T, G2_T, c='blue', s=30, alpha=0.6, label='Theory', marker='o')

    # Plot Simulation points
    V_S, G2_S = zip(*valid_S)
    ax.scatter(V_S, G2_S, c='green', s=30, alpha=0.6, label='Simulation', marker='s')

    ax.set_xlabel('V (Lyapunov Energy)', fontsize=12)
    ax.set_ylabel('G² (Spread squared)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Use log scale if range is large
    if max(V_T + V_S) / min(V_T + V_S) > 100:
        ax.set_xscale('log')
        ax.set_yscale('log')

    # Compute and show G²/V ratio statistics
    ratio_T = [g2 / v for v, g2 in valid_T]
    ratio_S = [g2 / v for v, g2 in valid_S]
    ax.text(0.02, 0.98,
            f'G²/V ratio:\n  Theory: {np.mean(ratio_T):.4f} ± {np.std(ratio_T):.4f}\n'
            f'  Sim: {np.mean(ratio_S):.4f} ± {np.std(ratio_S):.4f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved G vs V relation plot to {output_path}")

    plt.close()


def plot_eviction_delta_relation(trajectory: List[Dict],
                                  delta_data: List[Dict],
                                  output_path: Optional[Path] = None,
                                  title: str = "Eviction vs Delta Relation"):
    """
    Plot relationship between eviction events and Delta values.

    Top subplot: Δ_{m^S} curve with eviction events marked
    Bottom subplot: Eviction stage distribution vs m^S position

    Args:
        trajectory: List of dicts with eviction info
        delta_data: List of dicts with delta analysis
        output_path: Path to save figure (optional)
        title: Plot title
    """
    if not trajectory or not delta_data:
        print("Warning: Empty data, skipping plot_eviction_delta_relation")
        return

    batches = [row['batch'] for row in delta_data]
    delta_at_m_S = [row['delta_at_m_S'] for row in delta_data]
    m_S = [row['m_S'] for row in delta_data]

    # Find eviction events
    eviction_batches = []
    eviction_stages = []
    for row in trajectory:
        total_evic = row.get('sim_eviction_total', 0.0)
        if total_evic > 0.01:
            eviction_batches.append(row['batch'])
            # Find which stage had eviction
            for stage in range(10):  # Check up to 10 stages
                key = f'sim_eviction_stage{stage}'
                if row.get(key, 0.0) > 0.01:
                    eviction_stages.append(stage)
                    break
            else:
                eviction_stages.append(-1)  # Unknown

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # ===== Top subplot: Δ_{m^S} with eviction markers =====
    ax = axes[0]
    ax.plot(batches, delta_at_m_S, 'b-', linewidth=2, label='Δ_{m^S}',
            marker='o', markersize=3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5)

    # Mark eviction events
    if eviction_batches:
        for b in eviction_batches:
            ax.axvline(x=b, color='red', linestyle=':', alpha=0.5)

        # Get Δ_{m^S} at eviction batches
        evic_idx = [batches.index(b) if b in batches else -1 for b in eviction_batches]
        evic_delta = [delta_at_m_S[i] for i in evic_idx if i >= 0]
        evic_b = [eviction_batches[j] for j, i in enumerate(evic_idx) if i >= 0]
        if evic_delta:
            ax.scatter(evic_b, evic_delta, c='red', s=100, marker='v',
                       zorder=10, label=f'Eviction events ({len(eviction_batches)})')

    ax.set_ylabel('Δ_{m^S}', fontsize=12)
    ax.set_title('Δ_{m^S} with Eviction Events Marked', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # ===== Bottom subplot: Eviction stage vs m^S =====
    ax = axes[1]

    # Plot m^S as background
    ax.step(batches, m_S, 'b-', linewidth=1, alpha=0.5, label='m^S (Sim argmax)', where='mid')

    # Plot eviction stages
    if eviction_batches and eviction_stages:
        valid_evic = [(b, s) for b, s in zip(eviction_batches, eviction_stages) if s >= 0]
        if valid_evic:
            evic_b, evic_s = zip(*valid_evic)
            ax.scatter(evic_b, evic_s, c='red', s=80, marker='x',
                       linewidths=2, zorder=10, label='Eviction stage')

            # Check if eviction happened at m^S
            at_m_S = sum(1 for b, s in valid_evic if b in batches and m_S[batches.index(b)] == s)
            ax.text(0.02, 0.98, f'Eviction at m^S: {at_m_S}/{len(valid_evic)}',
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Batch Index', fontsize=12)
    ax.set_ylabel('Stage', fontsize=12)
    ax.set_title('Eviction Stage vs Simulation Argmax Position', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved eviction-delta relation plot to {output_path}")

    plt.close()


def plot_analysis_dashboard(trajectory: List[Dict],
                            lyapunov_data: List[Dict],
                            delta_data: List[Dict],
                            output_path: Optional[Path] = None,
                            title: str = "Analysis Dashboard"):
    """
    Create a comprehensive 2x3 dashboard of key metrics.

    Layout:
        (0,0) G^T vs G^S
        (0,1) V^T vs V^S (if available)
        (0,2) Δ_{m^S} time series
        (1,0) Case pie chart
        (1,1) Energy ratio V^T/V^S (if available)
        (1,2) G dominance summary

    Args:
        trajectory: List of dicts with trajectory data
        lyapunov_data: List of dicts with energy data (can be empty)
        delta_data: List of dicts with delta analysis
        output_path: Path to save figure (optional)
        title: Plot title
    """
    if not delta_data:
        print("Warning: Empty delta_data, skipping plot_analysis_dashboard")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    batches = [row['batch'] for row in delta_data]

    # ===== (0,0) G^T vs G^S =====
    ax = axes[0, 0]
    G_theory = [row.get('theory_G_merge', 0.0) for row in delta_data]
    G_sim = [row.get('sim_G_merge', 0.0) for row in delta_data]
    ax.plot(batches, G_theory, 'b-', linewidth=2, label='G^T', marker='o', markersize=2)
    ax.plot(batches, G_sim, 'g-', linewidth=2, label='G^S', marker='s', markersize=2)
    ax.set_xlabel('Batch')
    ax.set_ylabel('G (Spread)')
    ax.set_title('G Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ===== (0,1) V^T vs V^S =====
    ax = axes[0, 1]
    if lyapunov_data:
        V_theory = [row.get('V_theory', 0.0) for row in lyapunov_data]
        V_sim = [row.get('V_sim', 0.0) for row in lyapunov_data]
        ly_batches = [row['batch'] for row in lyapunov_data]
        valid = [(ly_batches[i], V_theory[i], V_sim[i])
                 for i in range(len(V_theory)) if V_theory[i] > 0 and V_sim[i] > 0]
        if valid:
            b, vt, vs = zip(*valid)
            ax.semilogy(b, vt, 'b-', linewidth=2, label='V^T', marker='o', markersize=2)
            ax.semilogy(b, vs, 'g-', linewidth=2, label='V^S', marker='s', markersize=2)
            ax.set_title('Energy V (log scale)')
        else:
            ax.text(0.5, 0.5, 'No energy data', ha='center', va='center')
            ax.set_title('Energy V')
    else:
        ax.text(0.5, 0.5, 'No energy data', ha='center', va='center')
        ax.set_title('Energy V')
    ax.set_xlabel('Batch')
    ax.set_ylabel('V')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ===== (0,2) Δ_{m^S} time series =====
    ax = axes[0, 2]
    delta_at_m_S = [row['delta_at_m_S'] for row in delta_data]
    ax.plot(batches, delta_at_m_S, 'b-', linewidth=2, marker='o', markersize=2)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Batch')
    ax.set_ylabel('Δ_{m^S}')
    ax.set_title('Δ at Sim argmax (should be ≥ 0)')
    ax.grid(True, alpha=0.3)
    # Mark violations
    violations = [i for i, d in enumerate(delta_at_m_S) if d < 0]
    if violations:
        ax.scatter([batches[i] for i in violations],
                   [delta_at_m_S[i] for i in violations],
                   c='red', s=50, marker='x', zorder=10)

    # ===== (1,0) Case pie chart =====
    ax = axes[1, 0]
    cases = [row['case'] for row in delta_data]
    case_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    for c in cases:
        case_counts[c] = case_counts.get(c, 0) + 1
    colors = {'A': '#2ecc71', 'B': '#f1c40f', 'C': '#e67e22', 'D': '#e74c3c'}
    labels = [f'{k}: {v}' for k, v in case_counts.items() if v > 0]
    sizes = [v for v in case_counts.values() if v > 0]
    pie_colors = [colors[k] for k, v in case_counts.items() if v > 0]
    if sizes:
        ax.pie(sizes, labels=labels, colors=pie_colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Case Distribution')

    # ===== (1,1) Energy ratio =====
    ax = axes[1, 1]
    if lyapunov_data:
        V_theory = [row.get('V_theory', 0.0) for row in lyapunov_data]
        V_sim = [row.get('V_sim', 0.0) for row in lyapunov_data]
        ly_batches = [row['batch'] for row in lyapunov_data]
        ratio = [(ly_batches[i], V_theory[i] / V_sim[i])
                 for i in range(len(V_theory)) if V_sim[i] > 0]
        if ratio:
            b, r = zip(*ratio)
            ax.plot(b, r, 'purple', linewidth=2, marker='o', markersize=2)
            ax.axhline(y=1.0, color='gray', linestyle='--')
            ax.set_title('Energy Ratio V^T/V^S')
        else:
            ax.text(0.5, 0.5, 'No ratio data', ha='center', va='center')
            ax.set_title('Energy Ratio')
    else:
        ax.text(0.5, 0.5, 'No energy data', ha='center', va='center')
        ax.set_title('Energy Ratio')
    ax.set_xlabel('Batch')
    ax.set_ylabel('V^T / V^S')
    ax.grid(True, alpha=0.3)

    # ===== (1,2) G dominance summary =====
    ax = axes[1, 2]
    G_dominance = [row.get('G_theory_geq_G_sim', True) for row in delta_data]
    dom_rate = sum(G_dominance) / len(G_dominance) * 100
    delta_m_geq_0 = sum(1 for row in delta_data if row['delta_at_m_S'] >= 0) / len(delta_data) * 100

    summary_text = (
        f"G Dominance Analysis\n"
        f"{'='*30}\n\n"
        f"G^T ≥ G^S: {sum(G_dominance)}/{len(G_dominance)} ({dom_rate:.1f}%)\n\n"
        f"Δ_{{m^S}} ≥ 0: {sum(1 for row in delta_data if row['delta_at_m_S'] >= 0)}/{len(delta_data)} ({delta_m_geq_0:.1f}%)\n\n"
        f"Case A rate: {case_counts['A']}/{len(cases)} ({case_counts['A']/len(cases)*100:.1f}%)"
    )
    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
            fontsize=14, verticalalignment='center', horizontalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    ax.axis('off')
    ax.set_title('Summary Statistics')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved analysis dashboard to {output_path}")

    plt.close()


# Quick test
if __name__ == "__main__":
    # Test with dummy data
    trajectory = [
        {'batch': 0, 'theory_admission': 0.0, 'sim_admission': 0.0,
         'sim_eviction_total': 0.0, 'sim_eviction_stage0': 0.0,
         'sim_eviction_stage1': 0.0, 'sim_eviction_stage2': 0.0},
        {'batch': 1, 'theory_admission': -7.5, 'sim_admission': 0.0,
         'sim_eviction_total': 7.5, 'sim_eviction_stage0': 0.0,
         'sim_eviction_stage1': 7.5, 'sim_eviction_stage2': 0.0},
        {'batch': 2, 'theory_admission': 5.0, 'sim_admission': 5.0,
         'sim_eviction_total': 0.0, 'sim_eviction_stage0': 0.0,
         'sim_eviction_stage1': 0.0, 'sim_eviction_stage2': 0.0},
        {'batch': 3, 'theory_admission': 3.0, 'sim_admission': 3.0,
         'sim_eviction_total': 0.0, 'sim_eviction_stage0': 0.0,
         'sim_eviction_stage1': 0.0, 'sim_eviction_stage2': 0.0},
    ]

    print("Testing visualization functions...")
    plot_comparison(trajectory, title="Test Comparison")
    plot_eviction_detail(trajectory, max_stage=2, title="Test Eviction Detail")
    print("Test complete (plots not saved, just validated).")
