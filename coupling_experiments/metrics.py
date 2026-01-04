"""
Metrics for coupling experiments.

This module contains functions to compute various metrics for comparing
Theory and Simulation trajectories.
"""

from typing import Dict, List, Tuple
import numpy as np


def compute_G(state: Dict, l0: int, l_A: int, l_B: int,
              state_format: str = 'stage') -> float:
    """
    Compute G = max(state values) - min(state values).

    This is a measure of "imbalance" in the system state.
    When the system reaches equilibrium, G should stabilize (or decrease for stable systems).

    Args:
        state: State dictionary
               - If state_format='stage': state[stage][type_idx], stage in 0..max_stage
               - If state_format='length': state[length][type_idx], length in l0..l0+max_stage
        l0: Initial/prefill length
        l_A: Decode length for Type A
        l_B: Decode length for Type B
        state_format: 'stage' or 'length'

    Returns:
        G = max - min of all valid state values
    """
    values = []

    if state_format == 'stage':
        # Theory format: state[stage][type]
        max_stage = max(l_A, l_B) - 1
        for stage in range(max_stage + 1):
            if stage not in state:
                continue
            # Type A valid for stage 0..l_A-1
            if stage < l_A:
                values.append(state[stage][0])
            # Type B valid for stage 0..l_B-1
            if stage < l_B:
                values.append(state[stage][1])

    elif state_format == 'length':
        # Simulation format: state[length][type]
        for length in state:
            # Type A: length in [l0, l0+l_A-1]
            if l0 <= length < l0 + l_A:
                values.append(state[length][0])
            # Type B: length in [l0, l0+l_B-1]
            if l0 <= length < l0 + l_B:
                values.append(state[length][1])

    if not values:
        return 0.0

    return max(values) - min(values)


def compute_G_weighted(state: Dict, l0: int, l_A: int, l_B: int,
                       state_format: str = 'stage') -> float:
    """
    Compute G_weighted = max(x/w) - min(x/w), where w = l0 + stage + 1.

    This is the metric from single_discrete.tex, measuring "flow imbalance".

    Args:
        state: State dictionary (same format as compute_G)
        l0: Initial/prefill length
        l_A: Decode length for Type A
        l_B: Decode length for Type B
        state_format: 'stage' or 'length'

    Returns:
        G_weighted = max - min of all valid (state value / weight)
    """
    weighted_values = []

    if state_format == 'stage':
        max_stage = max(l_A, l_B) - 1
        for stage in range(max_stage + 1):
            if stage not in state:
                continue
            weight = l0 + stage + 1
            if stage < l_A:
                weighted_values.append(state[stage][0] / weight)
            if stage < l_B:
                weighted_values.append(state[stage][1] / weight)

    elif state_format == 'length':
        for length in state:
            weight = length + 1  # = l0 + stage + 1
            if l0 <= length < l0 + l_A:
                weighted_values.append(state[length][0] / weight)
            if l0 <= length < l0 + l_B:
                weighted_values.append(state[length][1] / weight)

    if not weighted_values:
        return 0.0

    return max(weighted_values) - min(weighted_values)


def compute_G_A(state: Dict, l0: int, l_A: int, l_B: int,
                state_format: str = 'stage') -> float:
    """
    Compute G_A = max - min for Type A only.

    Args:
        state: State dictionary
        l0: Initial/prefill length
        l_A: Decode length for Type A
        l_B: Decode length for Type B (not used, for API consistency)
        state_format: 'stage' or 'length'

    Returns:
        G_A = max - min of Type A values across valid stages
    """
    values = []

    if state_format == 'stage':
        for stage in range(l_A):
            if stage in state:
                values.append(state[stage][0])

    elif state_format == 'length':
        for length in range(l0, l0 + l_A):
            if length in state:
                values.append(state[length][0])

    if not values:
        return 0.0

    return max(values) - min(values)


def compute_G_B(state: Dict, l0: int, l_A: int, l_B: int,
                state_format: str = 'stage') -> float:
    """
    Compute G_B = max - min for Type B only.

    Args:
        state: State dictionary
        l0: Initial/prefill length
        l_A: Decode length for Type A (not used, for API consistency)
        l_B: Decode length for Type B
        state_format: 'stage' or 'length'

    Returns:
        G_B = max - min of Type B values across valid stages
    """
    values = []

    if state_format == 'stage':
        for stage in range(l_B):
            if stage in state:
                values.append(state[stage][1])

    elif state_format == 'length':
        for length in range(l0, l0 + l_B):
            if length in state:
                values.append(state[length][1])

    if not values:
        return 0.0

    return max(values) - min(values)


def compute_G_merge(state: Dict, l0: int, l_A: int, l_B: int,
                    lambda_A: float, lambda_B: float,
                    state_format: str = 'stage') -> float:
    """
    Compute G_merge = max - min of merged state with compensation.

    For stages where only one type exists (after the shorter type completes),
    we compensate by scaling up to account for the missing type's flow.

    Example (l_A=2, l_B=3, λ_A=λ_B=1):
        Stage 0: merged = A + B
        Stage 1: merged = A + B
        Stage 2: merged = B * (λ_A + λ_B) / λ_B  (compensate for missing A)

    Args:
        state: State dictionary
        l0: Initial/prefill length
        l_A: Decode length for Type A
        l_B: Decode length for Type B
        lambda_A: Arrival rate for Type A
        lambda_B: Arrival rate for Type B
        state_format: 'stage' or 'length'

    Returns:
        G_merge = max - min of compensated merged values
    """
    merged_values = []
    max_stage = max(l_A, l_B) - 1
    min_l = min(l_A, l_B)
    total_lambda = lambda_A + lambda_B

    if state_format == 'stage':
        for stage in range(max_stage + 1):
            if stage not in state:
                continue

            if stage < min_l:
                # Both types present: simple sum
                merged = state[stage][0] + state[stage][1]
            elif l_A < l_B:
                # Only Type B present (A has completed)
                # Compensate: B * (λ_A + λ_B) / λ_B
                merged = state[stage][1] * total_lambda / lambda_B
            else:
                # Only Type A present (B has completed)
                # Compensate: A * (λ_A + λ_B) / λ_A
                merged = state[stage][0] * total_lambda / lambda_A

            merged_values.append(merged)

    elif state_format == 'length':
        for stage in range(max_stage + 1):
            length = l0 + stage
            if length not in state:
                continue

            if stage < min_l:
                # Both types present
                merged = state[length][0] + state[length][1]
            elif l_A < l_B:
                # Only Type B present
                merged = state[length][1] * total_lambda / lambda_B
            else:
                # Only Type A present
                merged = state[length][0] * total_lambda / lambda_A

            merged_values.append(merged)

    if not merged_values:
        return 0.0

    return max(merged_values) - min(merged_values)


def compute_G_merged_raw(state: Dict, l0: int, l_A: int, l_B: int,
                         state_format: str = 'stage') -> float:
    """
    Compute G_merged_raw = max - min of merged state WITHOUT compensation.

    Simply sums A + B at each stage. For stages where only one type exists,
    the other type contributes 0 (no compensation).

    Args:
        state: State dictionary
        l0: Initial/prefill length
        l_A: Decode length for Type A
        l_B: Decode length for Type B
        state_format: 'stage' or 'length'

    Returns:
        G_merged_raw = max - min of raw merged values (no compensation)
    """
    merged_values = []
    max_stage = max(l_A, l_B) - 1

    if state_format == 'stage':
        for stage in range(max_stage + 1):
            if stage not in state:
                continue
            # Sum both types (0 if not valid)
            val_A = state[stage][0] if stage < l_A else 0.0
            val_B = state[stage][1] if stage < l_B else 0.0
            merged_values.append(val_A + val_B)

    elif state_format == 'length':
        for stage in range(max_stage + 1):
            length = l0 + stage
            if length not in state:
                continue
            val_A = state[length][0] if stage < l_A else 0.0
            val_B = state[length][1] if stage < l_B else 0.0
            merged_values.append(val_A + val_B)

    if not merged_values:
        return 0.0

    return max(merged_values) - min(merged_values)


def compute_characteristic_roots(l0: int, l_A: int, l_B: int,
                                  lambda_A: float, lambda_B: float) -> np.ndarray:
    """
    Compute roots of the characteristic equation F(λ) = 0.

    F(λ) = (l_0+1)λ^{l_B-1} + Σ_{m=1}^{l_A-1}(l_0+m+1)λ^{l_B-1-m}
           + (1-p)Σ_{m=l_A}^{l_B-1}(l_0+m+1)λ^{l_B-1-m}

    Args:
        l0: Initial/prefill length
        l_A: Decode length for Type A
        l_B: Decode length for Type B (l_A < l_B)
        lambda_A: Arrival rate for Type A
        lambda_B: Arrival rate for Type B

    Returns:
        Array of complex roots
    """
    p = lambda_A / (lambda_A + lambda_B)

    # Polynomial degree is l_B - 1
    # Coefficients from highest degree to lowest (for np.roots)
    coeffs = [0.0] * l_B

    # λ^{l_B-1} coefficient (m=0)
    coeffs[0] = l0 + 1

    # λ^{l_B-1-m} for m = 1 to l_A-1
    for m in range(1, l_A):
        coeffs[m] = l0 + m + 1

    # λ^{l_B-1-m} for m = l_A to l_B-1
    for m in range(l_A, l_B):
        coeffs[m] = (1 - p) * (l0 + m + 1)

    return np.roots(coeffs)


def compute_limit_roots(l_A: int, l_B: int,
                        lambda_A: float, lambda_B: float) -> np.ndarray:
    """
    Compute roots of the limit equation (1-λ)A(λ) = 0.

    (1-λ)A(λ) = -λ^{l_B} + p·λ^{l_B-l_A} + q = 0

    where p = λ_A/(λ_A+λ_B), q = 1 - p

    Args:
        l_A: Decode length for Type A
        l_B: Decode length for Type B (l_A < l_B)
        lambda_A: Arrival rate for Type A
        lambda_B: Arrival rate for Type B

    Returns:
        Array of complex roots
    """
    p = lambda_A / (lambda_A + lambda_B)
    q = 1 - p

    # Polynomial: -λ^{l_B} + p·λ^{l_B-l_A} + q = 0
    # Degree is l_B
    # Coefficients from highest degree to lowest
    coeffs = [0.0] * (l_B + 1)
    coeffs[0] = -1.0            # λ^{l_B}
    coeffs[l_A] = p             # λ^{l_B-l_A}
    coeffs[l_B] = q             # λ^0 (constant term)

    return np.roots(coeffs)


# Test
if __name__ == "__main__":
    print("Testing metrics...")

    # Test with stage format (Theory)
    l0, l_A, l_B = 3, 2, 3
    state_stage = {
        0: [7.5, 7.5],  # Both types at stage 0
        1: [0.0, 0.0],  # Both types at stage 1
        2: [0.0, 0.0],  # Only type B at stage 2
    }

    G = compute_G(state_stage, l0, l_A, l_B, state_format='stage')
    print(f"G (stage format): {G}")

    G_w = compute_G_weighted(state_stage, l0, l_A, l_B, state_format='stage')
    print(f"G_weighted (stage format): {G_w}")

    # Test with length format (Simulation)
    state_length = {
        3: [7.5, 7.5],  # l0 = 3, stage 0
        4: [0.0, 0.0],  # stage 1
        5: [0.0, 0.0],  # stage 2 (only B valid)
    }

    G = compute_G(state_length, l0, l_A, l_B, state_format='length')
    print(f"G (length format): {G}")

    G_w = compute_G_weighted(state_length, l0, l_A, l_B, state_format='length')
    print(f"G_weighted (length format): {G_w}")

    print("Test complete.")
