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
