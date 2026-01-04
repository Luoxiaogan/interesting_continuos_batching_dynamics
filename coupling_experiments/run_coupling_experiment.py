#!/usr/bin/env python3
"""
Coupling Experiment: Theory vs Simulation Trajectory Comparison

This script runs both Theory (unconstrained) and Simulation (with eviction)
from the same initial condition and compares their trajectories.

Expected behavior: "diverge then converge" - the two trajectories diverge
when Theory has negative admission while Simulation has eviction, but both
eventually converge to the same fluid equilibrium (for coprime l_A, l_B).
"""

import sys
import os
import json
import csv
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'new_project_for_multi_type'))

from theory_simulator import TheorySimulator
from multi_type_simulator import MultiTypeLLMSimulator
from visualize_coupling import plot_comparison, plot_eviction_detail, plot_G_comparison, plot_G_decomposed, plot_roots_analysis
from metrics import compute_G, compute_G_A, compute_G_B, compute_G_merge, compute_G_merged_raw


def get_git_commit():
    """Get current git commit hash."""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, cwd=os.path.dirname(__file__)
        )
        return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def run_experiment(config: dict) -> dict:
    """
    Run coupling experiment comparing Theory vs Simulation.

    Args:
        config: Experiment configuration dictionary

    Returns:
        results: Dictionary containing trajectory data
    """
    # Extract parameters
    l0 = config['l0']
    l_A = config['l_A']
    l_B = config['l_B']
    B = config['B']
    lambda_A = config['lambda_A']
    lambda_B = config['lambda_B']
    steps = config['steps']

    # Compute derived values
    p = lambda_A / (lambda_A + lambda_B)
    N = B / (l0 + 1)

    print(f"Running coupling experiment:")
    print(f"  l0={l0}, l_A={l_A}, l_B={l_B}, B={B}")
    print(f"  lambda_A={lambda_A}, lambda_B={lambda_B}, p={p:.4f}")
    print(f"  N={N:.2f}, steps={steps}")
    print()

    # ===== Initialize Theory Simulator =====
    theory = TheorySimulator(l0, l_A, l_B, B, lambda_A, lambda_B)

    # Set initial condition: only stage 0, proportional
    theory_init = {0: [p * N, (1 - p) * N]}
    theory.set_initial_state(theory_init)

    # ===== Initialize Simulation =====
    # request_type_list format: [(l0, l1), ...] where l1 is decode length
    request_types = [(l0, l_A), (l0, l_B)]

    # Initial state for Simulation: X0[length][type_idx]
    # stage 0 corresponds to length = l0
    sim_init = {
        l0: [p * N, (1 - p) * N]
    }

    simulation = MultiTypeLLMSimulator(
        request_type_list=request_types,
        B=B,
        X0=sim_init,
        arrival_rates=[lambda_A, lambda_B],
        b0=0.1,  # Not important for comparison
        b1=0.01,
        verbose=False
    )

    # ===== Run both simulators =====
    print("Running simulations...")
    theory.run(steps)
    simulation.run(steps)

    # ===== Extract and align data =====
    results = {
        'config': config,
        'trajectory': []
    }

    # Theory data
    theory_admissions = theory.get_admission_history()
    theory_states = theory.get_state_history()

    # Simulation data
    sim_history = simulation.get_history()
    sim_admissions_list = sim_history['admissions']
    sim_evictions_list = sim_history['evictions']
    sim_states_list = sim_history['X_prime']  # State after admission/eviction

    max_stage = max(l_A, l_B) - 1

    for batch in range(steps):
        # Theory admission (can be negative)
        theory_adm = theory_admissions[batch] if batch < len(theory_admissions) else 0.0

        # Theory state and G metrics
        if batch < len(theory_states):
            theory_state = theory_states[batch]['state']
            theory_G = compute_G(theory_state, l0, l_A, l_B, state_format='stage')
            theory_G_A = compute_G_A(theory_state, l0, l_A, l_B, state_format='stage')
            theory_G_B = compute_G_B(theory_state, l0, l_A, l_B, state_format='stage')
            theory_G_merge = compute_G_merge(theory_state, l0, l_A, l_B, lambda_A, lambda_B, state_format='stage')
            theory_G_merged_raw = compute_G_merged_raw(theory_state, l0, l_A, l_B, state_format='stage')
        else:
            theory_G = 0.0
            theory_G_A = 0.0
            theory_G_B = 0.0
            theory_G_merge = 0.0
            theory_G_merged_raw = 0.0

        # Simulation admission (>= 0)
        sim_adm_data = sim_admissions_list[batch] if batch < len(sim_admissions_list) else {}
        sim_adm = sum(sim_adm_data.get('admissions', {}).values())

        # Simulation state and G metrics
        if batch < len(sim_states_list):
            sim_state = sim_states_list[batch]['state']
            sim_G = compute_G(sim_state, l0, l_A, l_B, state_format='length')
            sim_G_A = compute_G_A(sim_state, l0, l_A, l_B, state_format='length')
            sim_G_B = compute_G_B(sim_state, l0, l_A, l_B, state_format='length')
            sim_G_merge = compute_G_merge(sim_state, l0, l_A, l_B, lambda_A, lambda_B, state_format='length')
            sim_G_merged_raw = compute_G_merged_raw(sim_state, l0, l_A, l_B, state_format='length')
        else:
            sim_G = 0.0
            sim_G_A = 0.0
            sim_G_B = 0.0
            sim_G_merge = 0.0
            sim_G_merged_raw = 0.0

        # Simulation eviction by stage
        sim_evic_data = sim_evictions_list[batch] if batch < len(sim_evictions_list) else {}
        evictions_raw = sim_evic_data.get('evictions', {})

        # Convert eviction from (type_idx -> [(length, amount), ...]) to stage-based
        eviction_by_stage = {stage: 0.0 for stage in range(max_stage + 1)}
        eviction_total = 0.0

        for type_idx, evic_list in evictions_raw.items():
            for length, amount in evic_list:
                stage = length - l0
                if 0 <= stage <= max_stage:
                    eviction_by_stage[stage] += amount
                    eviction_total += amount

        # Build row
        row = {
            'batch': batch,
            'theory_admission': theory_adm,
            'sim_admission': sim_adm,
            'sim_eviction_total': eviction_total,
            'theory_G': theory_G,
            'sim_G': sim_G,
            'theory_G_A': theory_G_A,
            'theory_G_B': theory_G_B,
            'theory_G_merge': theory_G_merge,
            'theory_G_merged_raw': theory_G_merged_raw,
            'sim_G_A': sim_G_A,
            'sim_G_B': sim_G_B,
            'sim_G_merge': sim_G_merge,
            'sim_G_merged_raw': sim_G_merged_raw,
        }
        for stage in range(max_stage + 1):
            row[f'sim_eviction_stage{stage}'] = eviction_by_stage[stage]

        results['trajectory'].append(row)

    print(f"Simulation complete. {len(results['trajectory'])} batches recorded.")
    return results


def save_results(results: dict, output_dir: Path):
    """Save results to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    config = results['config']

    # Save config.json
    config_full = config.copy()
    config_full['p'] = config['lambda_A'] / (config['lambda_A'] + config['lambda_B'])
    config_full['N'] = config['B'] / (config['l0'] + 1)
    config_full['timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_full['git_commit'] = get_git_commit()

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config_full, f, indent=2)

    # Save trajectory.csv
    trajectory = results['trajectory']
    if trajectory:
        fieldnames = list(trajectory[0].keys())
        with open(output_dir / 'trajectory.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trajectory)

    print(f"Results saved to {output_dir}")


def main():
    """Main entry point."""
    # Load configuration from config.json
    config_path = Path(__file__).parent / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / 'outputs' / timestamp

    # Run experiment
    results = run_experiment(config)

    # Save results
    save_results(results, output_dir)

    # Generate visualizations
    print("\nGenerating visualizations...")
    trajectory = results['trajectory']

    plot_comparison(
        trajectory,
        output_path=output_dir / 'comparison.png',
        title=f"Theory vs Simulation: l0={config['l0']}, l_A={config['l_A']}, l_B={config['l_B']}"
    )

    plot_eviction_detail(
        trajectory,
        max_stage=max(config['l_A'], config['l_B']) - 1,
        output_path=output_dir / 'eviction_detail.png',
        title=f"Eviction by Stage: l0={config['l0']}, l_A={config['l_A']}, l_B={config['l_B']}"
    )

    plot_G_comparison(
        trajectory,
        output_path=output_dir / 'G_comparison.png',
        title=f"G (max-min) Comparison: l0={config['l0']}, l_A={config['l_A']}, l_B={config['l_B']}"
    )

    plot_G_decomposed(
        trajectory,
        output_path=output_dir / 'G_decomposed.png',
        title=f"G Decomposition: l0={config['l0']}, l_A={config['l_A']}, l_B={config['l_B']}"
    )

    plot_roots_analysis(
        l0=config['l0'], l_A=config['l_A'], l_B=config['l_B'],
        lambda_A=config['lambda_A'], lambda_B=config['lambda_B'],
        output_path=output_dir / 'roots_analysis.png',
        title="Characteristic & Limit Equation Roots"
    )

    print(f"\nExperiment complete!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
