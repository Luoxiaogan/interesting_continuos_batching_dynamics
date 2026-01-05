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

import numpy as np

from theory_simulator import TheorySimulator
from multi_type_simulator import MultiTypeLLMSimulator
from visualize_coupling import (
    plot_comparison, plot_eviction_detail, plot_G_comparison,
    plot_G_decomposed, plot_roots_analysis,
    # New visualization functions for 4.1-4.4 analysis
    plot_delta_extrema, plot_delta_heatmap, plot_lyapunov_energy,
    plot_case_distribution, plot_G_vs_V_relation,
    plot_eviction_delta_relation, plot_analysis_dashboard
)
from metrics import (compute_G, compute_G_A, compute_G_B, compute_G_merge,
                     compute_G_merged_raw, compute_merged_compensated_vector,
                     # 4.1 Lyapunov energy functions
                     compute_transition_matrix, compute_weight_vector,
                     compute_equilibrium_state, compute_lyapunov_matrix,
                     compute_lyapunov_energy,
                     # 4.2 Delta vector functions
                     compute_delta_vector, compute_delta_stats,
                     # 4.3 Argmax/Argmin tracking
                     compute_argmax_argmin, check_delta_at_extrema,
                     # 4.4 Norm relation functions
                     compute_G_squared_over_V, compute_G_based_energy)


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

    # ===== Initialize Lyapunov analysis (4.1) =====
    print("Initializing Lyapunov analysis...")

    # Use characteristic equation roots to check stability (more reliable)
    from metrics import compute_characteristic_roots
    char_roots = compute_characteristic_roots(l0, l_A, l_B, lambda_A, lambda_B)
    char_max_abs = np.max(np.abs(char_roots))
    is_stable = char_max_abs < 1.0
    print(f"  Characteristic equation: max |root| = {char_max_abs:.6f}")
    print(f"  System is {'stable (coprime)' if is_stable else 'UNSTABLE (non-coprime)'}")

    # Compute weight vector and equilibrium state
    W = compute_weight_vector(l0, l_A, l_B, lambda_A, lambda_B)
    M_star = compute_equilibrium_state(l0, l_A, l_B, B, lambda_A, lambda_B)
    print(f"  Equilibrium N = {M_star[0]:.4f}")

    # Try to compute transition matrix and Lyapunov matrix
    A = None
    P = None
    try:
        A = compute_transition_matrix(l0, l_A, l_B, lambda_A, lambda_B)
        A_eigenvalues = np.linalg.eigvals(A)
        A_max_abs = np.max(np.abs(A_eigenvalues))
        print(f"  Transition matrix A: max |eigenvalue| = {A_max_abs:.6f}")

        # Only compute Lyapunov matrix if A is stable
        if A_max_abs < 1.0:
            P = compute_lyapunov_matrix(A)
            P_eigenvalues = np.linalg.eigvals(P)
            print(f"  Lyapunov matrix P computed (eigenvalues: {np.sort(np.real(P_eigenvalues))[:3]}...)")
        else:
            print("  Note: A matrix unstable, using simple identity for energy (||M - M*||^2)")
    except Exception as e:
        print(f"  Warning: Could not compute transition matrix: {e}")

    # ===== Extract and align data =====
    results = {
        'config': config,
        'trajectory': [],
        'state_vectors': [],  # For state_vectors.csv and aligned txt
        'lyapunov_analysis': [],  # For 4.1 energy analysis
        'delta_analysis': [],  # For 4.2, 4.3, 4.4 analysis
        'lyapunov_info': {
            'A': A.tolist() if A is not None else None,
            'W': W.tolist(),
            'M_star': M_star.tolist(),
            'P': P.tolist() if P is not None else None,
            'max_characteristic_root': float(char_max_abs),
            'is_stable': is_stable,
        }
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
            theory_vec = compute_merged_compensated_vector(theory_state, l0, l_A, l_B, lambda_A, lambda_B, state_format='stage')
        else:
            theory_G = 0.0
            theory_G_A = 0.0
            theory_G_B = 0.0
            theory_G_merge = 0.0
            theory_G_merged_raw = 0.0
            theory_vec = [0.0] * max(l_A, l_B)

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
            sim_vec = compute_merged_compensated_vector(sim_state, l0, l_A, l_B, lambda_A, lambda_B, state_format='length')
        else:
            sim_G = 0.0
            sim_G_A = 0.0
            sim_G_B = 0.0
            sim_G_merge = 0.0
            sim_G_merged_raw = 0.0
            sim_vec = [0.0] * max(l_A, l_B)

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

        # Store state vectors for separate output
        # Compute max and min with indices
        theory_max_val = max(theory_vec) if theory_vec else 0.0
        theory_min_val = min(theory_vec) if theory_vec else 0.0
        theory_max_idx = theory_vec.index(theory_max_val) if theory_vec else 0
        theory_min_idx = theory_vec.index(theory_min_val) if theory_vec else 0

        sim_max_val = max(sim_vec) if sim_vec else 0.0
        sim_min_val = min(sim_vec) if sim_vec else 0.0
        sim_max_idx = sim_vec.index(sim_max_val) if sim_vec else 0
        sim_min_idx = sim_vec.index(sim_min_val) if sim_vec else 0

        results['state_vectors'].append({
            'batch': batch,
            'theory_vec': theory_vec,
            'sim_vec': sim_vec,
            'theory_max_min': [(theory_max_val, theory_max_idx), (theory_min_val, theory_min_idx)],
            'sim_max_min': [(sim_max_val, sim_max_idx), (sim_min_val, sim_min_idx)],
            'theory_G': theory_G_merge,
            'sim_G': sim_G_merge,
        })

        # ===== 4.1 Lyapunov Energy Analysis =====
        # Always compute energy (use P if available, else use identity = ||M - M*||^2)
        theory_vec_arr = np.array(theory_vec)
        sim_vec_arr = np.array(sim_vec)

        if P is not None:
            V_theory = compute_lyapunov_energy(theory_vec_arr, P, M_star)
            V_sim = compute_lyapunov_energy(sim_vec_arr, P, M_star)
        else:
            # Fallback: simple L2 energy ||M - M*||^2
            V_theory = float(np.sum((theory_vec_arr - M_star) ** 2))
            V_sim = float(np.sum((sim_vec_arr - M_star) ** 2))

        # G-based energy (alternative)
        U_theory = compute_G_based_energy(theory_vec, M_star)
        U_sim = compute_G_based_energy(sim_vec, M_star)

        # G^2/V ratio (4.4)
        G2_over_V_theory = compute_G_squared_over_V(theory_G_merge, V_theory)
        G2_over_V_sim = compute_G_squared_over_V(sim_G_merge, V_sim)

        results['lyapunov_analysis'].append({
            'batch': batch,
            'V_theory': V_theory,
            'V_sim': V_sim,
            'V_theory_geq_V_sim': V_theory >= V_sim - 1e-9,
            'U_theory': U_theory,
            'U_sim': U_sim,
            'G2_over_V_theory': G2_over_V_theory,
            'G2_over_V_sim': G2_over_V_sim,
        })

        # ===== 4.2, 4.3 Delta Vector Analysis =====
        delta_vec = compute_delta_vector(theory_vec, sim_vec)
        delta_stats = compute_delta_stats(delta_vec, W)
        extrema_check = check_delta_at_extrema(theory_vec, sim_vec)

        # G dominance check
        G_theory_geq_G_sim = theory_G_merge >= sim_G_merge - 1e-9

        results['delta_analysis'].append({
            'batch': batch,
            'delta_vec': delta_vec,
            # Delta stats
            'weighted_sum': delta_stats['weighted_sum'],
            'max_delta': delta_stats['max_delta'],
            'min_delta': delta_stats['min_delta'],
            'argmax_delta': delta_stats['argmax_delta'],
            'argmin_delta': delta_stats['argmin_delta'],
            'num_positive': delta_stats['num_positive'],
            'num_negative': delta_stats['num_negative'],
            # Extrema check
            'm_S': extrema_check['m_S'],
            'n_S': extrema_check['n_S'],
            'delta_at_m_S': extrema_check['delta_at_m_S'],
            'delta_at_n_S': extrema_check['delta_at_n_S'],
            'delta_m_geq_0': extrema_check['delta_m_geq_0'],
            'delta_m_geq_delta_n': extrema_check['delta_m_geq_delta_n'],
            'case': extrema_check['case'],
            # G dominance
            'theory_G_merge': theory_G_merge,
            'sim_G_merge': sim_G_merge,
            'G_theory_geq_G_sim': G_theory_geq_G_sim,
        })

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

    # Save state_vectors.csv and state_vectors_aligned.txt
    save_state_vectors(results['state_vectors'], output_dir)

    # Save lyapunov_analysis.csv (4.1)
    save_lyapunov_analysis(results.get('lyapunov_analysis', []), output_dir)

    # Save delta_analysis.csv (4.2, 4.3, 4.4)
    save_delta_analysis(results.get('delta_analysis', []), output_dir)

    # Save analysis_summary.json
    save_analysis_summary(results, output_dir)

    print(f"Results saved to {output_dir}")


def save_state_vectors(state_vectors: list, output_dir: Path):
    """Save state vectors to CSV and aligned TXT files."""
    if not state_vectors:
        return

    # Helper function to format vector as string
    def vec_to_str(vec):
        return '[' + ','.join(f'{v:.5f}' for v in vec) + ']'

    # Helper function to format max_min as string
    def max_min_to_str(mm):
        return f'[({mm[0][0]:.5f},{mm[0][1]}),({mm[1][0]:.5f},{mm[1][1]})]'

    # ===== Save state_vectors.csv =====
    with open(output_dir / 'state_vectors.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['batch', 'theory_state_vector_merged_compensated',
                        'simulation_state_vector_merged_compensated',
                        'theory_max_min', 'simulation_max_min',
                        'theory_G', 'simulation_G'])
        for row in state_vectors:
            writer.writerow([
                row['batch'],
                vec_to_str(row['theory_vec']),
                vec_to_str(row['sim_vec']),
                max_min_to_str(row['theory_max_min']),
                max_min_to_str(row['sim_max_min']),
                f"{row['theory_G']:.5f}",
                f"{row['sim_G']:.5f}"
            ])

    # ===== Save state_vectors_aligned.txt =====
    # Calculate max width needed for each position in the vector
    vec_len = len(state_vectors[0]['theory_vec']) if state_vectors else 0

    # Find the maximum width needed for any number (5 decimals + sign + integer part)
    max_width = 12  # Default width for numbers like -12345.12345

    with open(output_dir / 'state_vectors_aligned.txt', 'w') as f:
        for row in state_vectors:
            batch = row['batch']
            theory_vec = row['theory_vec']
            sim_vec = row['sim_vec']

            # Format each number with fixed width
            theory_nums = [f'{v:>{max_width}.5f}' for v in theory_vec]
            sim_nums = [f'{v:>{max_width}.5f}' for v in sim_vec]

            theory_str = '[' + ', '.join(theory_nums) + ']'
            sim_str = '[' + ', '.join(sim_nums) + ']'

            f.write(f"Step = {batch}\n")
            f.write(f"Theory:     {theory_str}\n")
            f.write(f"Simulation: {sim_str}\n")
            f.write("\n")

    print(f"Saved state_vectors.csv and state_vectors_aligned.txt")


def save_lyapunov_analysis(lyapunov_data: list, output_dir: Path):
    """Save Lyapunov energy analysis to CSV."""
    if not lyapunov_data:
        return

    fieldnames = ['batch', 'V_theory', 'V_sim', 'V_theory_geq_V_sim',
                  'U_theory', 'U_sim', 'G2_over_V_theory', 'G2_over_V_sim']

    with open(output_dir / 'lyapunov_analysis.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in lyapunov_data:
            writer.writerow({k: row[k] for k in fieldnames})

    print(f"Saved lyapunov_analysis.csv")


def save_delta_analysis(delta_data: list, output_dir: Path):
    """Save Delta vector analysis to CSV."""
    if not delta_data:
        return

    # Helper to format delta_vec as string
    def vec_to_str(vec):
        return '[' + ','.join(f'{v:.5f}' for v in vec) + ']'

    fieldnames = ['batch', 'delta_vec', 'weighted_sum',
                  'max_delta', 'min_delta', 'argmax_delta', 'argmin_delta',
                  'num_positive', 'num_negative',
                  'm_S', 'n_S', 'delta_at_m_S', 'delta_at_n_S',
                  'delta_m_geq_0', 'delta_m_geq_delta_n', 'case',
                  'theory_G_merge', 'sim_G_merge', 'G_theory_geq_G_sim']

    with open(output_dir / 'delta_analysis.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for row in delta_data:
            writer.writerow([
                row['batch'],
                vec_to_str(row['delta_vec']),
                f"{row['weighted_sum']:.6f}",
                f"{row['max_delta']:.5f}",
                f"{row['min_delta']:.5f}",
                row['argmax_delta'],
                row['argmin_delta'],
                row['num_positive'],
                row['num_negative'],
                row['m_S'],
                row['n_S'],
                f"{row['delta_at_m_S']:.5f}",
                f"{row['delta_at_n_S']:.5f}",
                row['delta_m_geq_0'],
                row['delta_m_geq_delta_n'],
                row['case'],
                f"{row['theory_G_merge']:.5f}",
                f"{row['sim_G_merge']:.5f}",
                row['G_theory_geq_G_sim'],
            ])

    print(f"Saved delta_analysis.csv")


def save_analysis_summary(results: dict, output_dir: Path):
    """Save summary statistics of the analysis."""
    summary = {
        'config': results['config'],
        'lyapunov_info': results.get('lyapunov_info', {}),
    }

    # Lyapunov energy statistics
    lyapunov_data = results.get('lyapunov_analysis', [])
    if lyapunov_data:
        V_theory_list = [r['V_theory'] for r in lyapunov_data]
        V_sim_list = [r['V_sim'] for r in lyapunov_data]
        V_geq_count = sum(1 for r in lyapunov_data if r['V_theory_geq_V_sim'])

        summary['lyapunov_summary'] = {
            'V_theory_initial': V_theory_list[0],
            'V_theory_final': V_theory_list[-1],
            'V_sim_initial': V_sim_list[0],
            'V_sim_final': V_sim_list[-1],
            'V_theory_geq_V_sim_count': V_geq_count,
            'V_theory_geq_V_sim_rate': V_geq_count / len(lyapunov_data),
            'V_theory_over_V_sim_mean': sum(v_t / v_s if v_s > 0 else 0
                                            for v_t, v_s in zip(V_theory_list, V_sim_list)) / len(V_theory_list),
        }

    # Delta analysis statistics
    delta_data = results.get('delta_analysis', [])
    if delta_data:
        # Case distribution
        case_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        for r in delta_data:
            case_counts[r['case']] += 1

        # Key checks
        delta_m_geq_0_count = sum(1 for r in delta_data if r['delta_m_geq_0'])
        delta_m_geq_n_count = sum(1 for r in delta_data if r['delta_m_geq_delta_n'])
        G_dominance_count = sum(1 for r in delta_data if r['G_theory_geq_G_sim'])

        # Eviction correlation with m_S
        eviction_at_m_S = 0
        for i, r in enumerate(delta_data):
            if i < len(results['trajectory']):
                traj = results['trajectory'][i]
                m_S = r['m_S']
                if traj.get(f'sim_eviction_stage{m_S}', 0) > 1e-9:
                    eviction_at_m_S += 1

        summary['delta_summary'] = {
            'case_distribution': case_counts,
            'delta_m_geq_0_count': delta_m_geq_0_count,
            'delta_m_geq_0_rate': delta_m_geq_0_count / len(delta_data),
            'delta_m_geq_delta_n_count': delta_m_geq_n_count,
            'delta_m_geq_delta_n_rate': delta_m_geq_n_count / len(delta_data),
            'G_dominance_count': G_dominance_count,
            'G_dominance_rate': G_dominance_count / len(delta_data),
            'eviction_at_m_S_count': eviction_at_m_S,
        }

    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    with open(output_dir / 'analysis_summary.json', 'w') as f:
        json.dump(convert_to_native(summary), f, indent=2)

    # Also print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    if 'lyapunov_summary' in summary:
        ls = summary['lyapunov_summary']
        print(f"\nLyapunov Energy:")
        print(f"  V^T >= V^S: {ls['V_theory_geq_V_sim_count']}/{len(lyapunov_data)} "
              f"({ls['V_theory_geq_V_sim_rate']*100:.1f}%)")
        print(f"  V^T / V^S mean: {ls['V_theory_over_V_sim_mean']:.3f}")

    if 'delta_summary' in summary:
        ds = summary['delta_summary']
        print(f"\nDelta Analysis:")
        print(f"  Case distribution: {ds['case_distribution']}")
        print(f"  Δ_{{m^S}} >= 0: {ds['delta_m_geq_0_count']}/{len(delta_data)} "
              f"({ds['delta_m_geq_0_rate']*100:.1f}%)")
        print(f"  G^T >= G^S (dominance): {ds['G_dominance_count']}/{len(delta_data)} "
              f"({ds['G_dominance_rate']*100:.1f}%)")
        print(f"  Eviction at m^S: {ds['eviction_at_m_S_count']}")

    print("=" * 60)
    print(f"Saved analysis_summary.json")


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

    # ===== NEW VISUALIZATIONS FOR 4.1-4.4 ANALYSIS =====
    delta_data = results.get('delta_analysis', [])
    lyapunov_data = results.get('lyapunov_analysis', [])

    # 1. Delta extrema tracking (KEY: verifies Δ_{m^S} >= 0)
    if delta_data:
        plot_delta_extrema(
            delta_data,
            output_path=output_dir / 'delta_extrema.png',
            title=f"Delta at Extrema: l0={config['l0']}, l_A={config['l_A']}, l_B={config['l_B']}"
        )

    # 2. Delta heatmap
    if delta_data:
        plot_delta_heatmap(
            delta_data,
            output_path=output_dir / 'delta_heatmap.png',
            title=f"Delta Vector Heatmap: l0={config['l0']}, l_A={config['l_A']}, l_B={config['l_B']}"
        )

    # 3. Lyapunov energy analysis
    if lyapunov_data:
        plot_lyapunov_energy(
            lyapunov_data,
            output_path=output_dir / 'lyapunov_energy.png',
            title=f"Lyapunov Energy: l0={config['l0']}, l_A={config['l_A']}, l_B={config['l_B']}"
        )

    # 4. Case distribution
    if delta_data:
        plot_case_distribution(
            delta_data,
            output_path=output_dir / 'case_distribution.png',
            title=f"Case Distribution: l0={config['l0']}, l_A={config['l_A']}, l_B={config['l_B']}"
        )

    # 5. G vs V relation
    if lyapunov_data and delta_data:
        plot_G_vs_V_relation(
            lyapunov_data,
            delta_data,
            output_path=output_dir / 'G_vs_V_relation.png',
            title=f"G² vs V: l0={config['l0']}, l_A={config['l_A']}, l_B={config['l_B']}"
        )

    # 6. Eviction-Delta relation
    if delta_data:
        plot_eviction_delta_relation(
            trajectory,
            delta_data,
            output_path=output_dir / 'eviction_delta_relation.png',
            title=f"Eviction-Delta: l0={config['l0']}, l_A={config['l_A']}, l_B={config['l_B']}"
        )

    # 7. Analysis dashboard (comprehensive summary)
    if delta_data:
        plot_analysis_dashboard(
            trajectory,
            lyapunov_data,
            delta_data,
            output_path=output_dir / 'analysis_dashboard.png',
            title=f"Analysis Dashboard: l0={config['l0']}, l_A={config['l_A']}, l_B={config['l_B']}"
        )

    print(f"\nExperiment complete!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
