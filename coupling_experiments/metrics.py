"""
Metrics for coupling experiments.

This module contains functions to compute various metrics for comparing
Theory and Simulation trajectories.
"""

from typing import Dict, List, Tuple


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
