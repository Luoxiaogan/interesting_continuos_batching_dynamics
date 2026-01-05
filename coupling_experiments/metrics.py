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


def compute_merged_compensated_vector(state: Dict, l0: int, l_A: int, l_B: int,
                                       lambda_A: float, lambda_B: float,
                                       state_format: str = 'stage') -> List[float]:
    """
    Compute merged state vector with compensation.

    Returns a vector of length max(l_A, l_B) where each element is the
    merged (A+B) value at that stage, with compensation for stages where
    only one type exists.

    Args:
        state: State dictionary
        l0: Initial/prefill length
        l_A: Decode length for Type A
        l_B: Decode length for Type B
        lambda_A: Arrival rate for Type A
        lambda_B: Arrival rate for Type B
        state_format: 'stage' or 'length'

    Returns:
        List of merged values with compensation, length = max(l_A, l_B)
    """
    max_stage = max(l_A, l_B)
    min_l = min(l_A, l_B)
    total_lambda = lambda_A + lambda_B
    result = []

    if state_format == 'stage':
        for stage in range(max_stage):
            if stage not in state:
                result.append(0.0)
                continue

            if stage < min_l:
                # Both types present: simple sum
                merged = state[stage][0] + state[stage][1]
            elif l_A < l_B:
                # Only Type B present (A has completed)
                merged = state[stage][1] * total_lambda / lambda_B
            else:
                # Only Type A present (B has completed)
                merged = state[stage][0] * total_lambda / lambda_A
            result.append(merged)

    elif state_format == 'length':
        for stage in range(max_stage):
            length = l0 + stage
            if length not in state:
                result.append(0.0)
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
            result.append(merged)

    return result


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


# ============================================================================
# 4.1 Lyapunov Energy Functions
# ============================================================================

def compute_transition_matrix(l0: int, l_A: int, l_B: int,
                               lambda_A: float, lambda_B: float) -> np.ndarray:
    """
    Compute the state transition matrix A for the Theory system.

    The merged state vector M evolves as: M(t+1) = A * M(t) + b
    where b depends on the equilibrium admission.

    For the deviation from equilibrium: δ(t+1) = A * δ(t)

    Args:
        l0: Initial/prefill length
        l_A: Decode length for Type A
        l_B: Decode length for Type B (l_A < l_B)
        lambda_A: Arrival rate for Type A
        lambda_B: Arrival rate for Type B

    Returns:
        Transition matrix A of shape (l_B, l_B)
    """
    n = l_B  # Dimension of merged state vector
    p = lambda_A / (lambda_A + lambda_B)
    q = 1 - p

    A = np.zeros((n, n))

    # Row 0: admission equation
    # a(t) = (completion_tokens - increment_tokens) / W_0
    # M_0(t+1) = a(t) depends on M(t)
    W_0 = l0 + 1

    for s in range(n):
        if s < l_A:
            # Both types at stage s, weight = l0 + s + 1
            W_s = l0 + s + 1
        else:
            # Only Type B at stage s, effective weight for merged = (l0+s+1)*q
            W_s = (l0 + s + 1) * q

        if s == n - 1:
            # Completion stage: releases (l0 + l_B) tokens per request
            # For merged state: completion releases W_{l_B-1} * (l0+l_B) / W_{l_B-1} = (l0+l_B)
            # But we need to account for the shift...
            # Actually for completion: tokens released = (l0 + l_B) per request
            completion_coeff = (l0 + l_B) * q if l_A < l_B else (l0 + l_B)
            A[0, s] = completion_coeff / W_0
        else:
            # Non-completion stage: each request gains 1 token
            # Contribution to admission = -1 / W_0 per request
            A[0, s] = -W_s / W_0  # Normalized by effective weight

    # Rows 1 to n-1: shift (advance) operation
    # M_s(t+1) = M_{s-1}(t) for s = 1, ..., n-1
    for s in range(1, n):
        A[s, s-1] = 1.0

    return A


def compute_weight_vector(l0: int, l_A: int, l_B: int,
                          lambda_A: float, lambda_B: float) -> np.ndarray:
    """
    Compute the weight vector W for token balance.

    Token Balance: sum_s W_s * M_s = B

    Args:
        l0, l_A, l_B, lambda_A, lambda_B: System parameters

    Returns:
        Weight vector W of length l_B
    """
    n = l_B
    q = lambda_B / (lambda_A + lambda_B)

    W = np.zeros(n)
    for s in range(n):
        if s < l_A:
            W[s] = l0 + s + 1
        else:
            W[s] = (l0 + s + 1) * q

    return W


def compute_equilibrium_state(l0: int, l_A: int, l_B: int, B: float,
                               lambda_A: float, lambda_B: float) -> np.ndarray:
    """
    Compute the equilibrium merged state vector M*.

    At equilibrium, all stages have the same merged value N:
    M* = [N, N, ..., N]
    where N = B / sum(W_s)

    Args:
        l0, l_A, l_B, B, lambda_A, lambda_B: System parameters

    Returns:
        Equilibrium state vector M* of length l_B
    """
    W = compute_weight_vector(l0, l_A, l_B, lambda_A, lambda_B)
    N = B / np.sum(W)
    return np.full(l_B, N)


def compute_lyapunov_matrix(A: np.ndarray, Q: np.ndarray = None) -> np.ndarray:
    """
    Solve the discrete Lyapunov equation: A^T P A - P = -Q

    For stable A (all eigenvalues |λ| < 1), there exists unique P > 0.

    Args:
        A: Transition matrix (must be stable)
        Q: Positive definite matrix (default: identity)

    Returns:
        P: Solution to the Lyapunov equation
    """
    from scipy import linalg

    n = A.shape[0]
    if Q is None:
        Q = np.eye(n)

    # scipy.linalg.solve_discrete_lyapunov solves: A X A^T - X + Q = 0
    # We need: A^T P A - P = -Q  =>  A^T P A - P + Q = 0
    # Let X = P, then we need: A^T X A - X + Q = 0
    # scipy solves: A X A^T - X + Q = 0
    # So we pass A^T to get: (A^T) X (A^T)^T - X + Q = 0 => A^T X A - X + Q = 0

    P = linalg.solve_discrete_lyapunov(A.T, Q)

    return P


def compute_lyapunov_energy(M: np.ndarray, P: np.ndarray, M_star: np.ndarray) -> float:
    """
    Compute Lyapunov energy V(M) = (M - M*)^T P (M - M*)

    Args:
        M: Current merged state vector
        P: Lyapunov matrix
        M_star: Equilibrium state vector

    Returns:
        V: Lyapunov energy (scalar)
    """
    delta = M - M_star
    return float(delta @ P @ delta)


def compute_lyapunov_energy_from_state(state: Dict, l0: int, l_A: int, l_B: int,
                                        lambda_A: float, lambda_B: float, B: float,
                                        P: np.ndarray, M_star: np.ndarray,
                                        state_format: str = 'stage') -> float:
    """
    Compute Lyapunov energy from a state dictionary.

    Convenience function that converts state dict to merged vector,
    then computes energy.

    Args:
        state: State dictionary
        l0, l_A, l_B, lambda_A, lambda_B, B: System parameters
        P: Lyapunov matrix
        M_star: Equilibrium state vector
        state_format: 'stage' or 'length'

    Returns:
        V: Lyapunov energy
    """
    M = np.array(compute_merged_compensated_vector(
        state, l0, l_A, l_B, lambda_A, lambda_B, state_format
    ))
    return compute_lyapunov_energy(M, P, M_star)


# ============================================================================
# 4.2 Delta Vector Functions
# ============================================================================

def compute_delta_vector(M_theory: List[float], M_sim: List[float]) -> List[float]:
    """
    Compute Δ = M^T - M^S (Theory minus Simulation).

    Args:
        M_theory: Theory merged state vector
        M_sim: Simulation merged state vector

    Returns:
        Delta vector Δ
    """
    return [t - s for t, s in zip(M_theory, M_sim)]


def compute_delta_stats(delta: List[float], W: np.ndarray) -> Dict:
    """
    Compute statistics about the Delta vector.

    Args:
        delta: Delta vector Δ = M^T - M^S
        W: Weight vector

    Returns:
        Dictionary with:
        - weighted_sum: sum(W_s * Δ_s) (should be ~0 by Token Balance)
        - max_delta: max(Δ)
        - min_delta: min(Δ)
        - argmax_delta: position of max(Δ)
        - argmin_delta: position of min(Δ)
        - num_positive: count of Δ_s > 0
        - num_negative: count of Δ_s < 0
    """
    delta_arr = np.array(delta)

    return {
        'weighted_sum': float(np.sum(W * delta_arr)),
        'max_delta': float(np.max(delta_arr)),
        'min_delta': float(np.min(delta_arr)),
        'argmax_delta': int(np.argmax(delta_arr)),
        'argmin_delta': int(np.argmin(delta_arr)),
        'num_positive': int(np.sum(delta_arr > 1e-9)),
        'num_negative': int(np.sum(delta_arr < -1e-9)),
    }


# ============================================================================
# 4.3 Argmax/Argmin Tracking
# ============================================================================

def compute_argmax_argmin(M: List[float]) -> Tuple[int, int, float, float]:
    """
    Compute argmax and argmin positions and values.

    Args:
        M: Merged state vector

    Returns:
        (argmax, argmin, max_value, min_value)
    """
    M_arr = np.array(M)
    argmax = int(np.argmax(M_arr))
    argmin = int(np.argmin(M_arr))
    return argmax, argmin, float(M_arr[argmax]), float(M_arr[argmin])


def check_delta_at_extrema(M_theory: List[float], M_sim: List[float]) -> Dict:
    """
    Check Delta values at Simulation's argmax and argmin positions.

    This is the key observation: Δ_{m^S} >= 0 (numerically verified).

    Args:
        M_theory: Theory merged state vector
        M_sim: Simulation merged state vector

    Returns:
        Dictionary with:
        - m_S: argmax position of Simulation
        - n_S: argmin position of Simulation
        - delta_at_m_S: Δ_{m^S} = M^T_{m^S} - M^S_{m^S}
        - delta_at_n_S: Δ_{n^S} = M^T_{n^S} - M^S_{n^S}
        - delta_m_geq_0: whether Δ_{m^S} >= 0
        - delta_m_geq_delta_n: whether Δ_{m^S} >= Δ_{n^S}
        - case: A/B/C/D classification
    """
    m_S, n_S, max_sim, min_sim = compute_argmax_argmin(M_sim)

    delta_at_m_S = M_theory[m_S] - M_sim[m_S]
    delta_at_n_S = M_theory[n_S] - M_sim[n_S]

    # Case classification from theory.md
    if delta_at_m_S >= 0 and delta_at_n_S <= 0:
        case = 'A'  # Obvious case
    elif delta_at_m_S >= 0 and delta_at_n_S > 0:
        case = 'B'  # Need Δ_{m^S} >= Δ_{n^S}
    elif delta_at_m_S < 0 and delta_at_n_S <= 0:
        case = 'C'  # Need |Δ_{m^S}| <= |Δ_{n^S}|
    else:  # delta_at_m_S < 0 and delta_at_n_S > 0
        case = 'D'  # Should not happen

    return {
        'm_S': m_S,
        'n_S': n_S,
        'max_sim': max_sim,
        'min_sim': min_sim,
        'max_theory': M_theory[m_S],
        'min_theory_at_n_S': M_theory[n_S],
        'delta_at_m_S': delta_at_m_S,
        'delta_at_n_S': delta_at_n_S,
        'delta_m_geq_0': delta_at_m_S >= -1e-9,
        'delta_m_geq_delta_n': delta_at_m_S >= delta_at_n_S - 1e-9,
        'case': case,
    }


# ============================================================================
# 4.4 Norm Relation Functions
# ============================================================================

def compute_G_squared_over_V(G: float, V: float) -> float:
    """
    Compute the ratio G^2 / V.

    This measures how G (L_inf spread) relates to V (L_2 energy).

    Args:
        G: Spread (max - min)
        V: Lyapunov energy

    Returns:
        G^2 / V ratio (or 0 if V is too small)
    """
    if V < 1e-12:
        return 0.0
    return G * G / V


def compute_G_based_energy(M: List[float], M_star: np.ndarray) -> float:
    """
    Compute G-based energy: U(M) = G(M - M*)^2 = (max(M-M*) - min(M-M*))^2

    This is an alternative Lyapunov-like function based on spread.

    Args:
        M: Current merged state vector
        M_star: Equilibrium state vector

    Returns:
        U: G-based energy
    """
    delta = np.array(M) - M_star
    G = float(np.max(delta) - np.min(delta))
    return G * G


# ============================================================================
# 4.5 P-Norm and Inner Product Functions (Experiments 1 & 2)
# ============================================================================

def compute_P_norm_squared(x: np.ndarray, P: np.ndarray) -> float:
    """
    Compute ||x||_P^2 = x^T P x.

    This is the squared P-norm (Lyapunov energy without centering).

    Args:
        x: Vector
        P: Positive definite matrix

    Returns:
        ||x||_P^2
    """
    return float(x @ P @ x)


def compute_P_inner_product(x: np.ndarray, y: np.ndarray, P: np.ndarray) -> float:
    """
    Compute <x, y>_P = x^T P y.

    This is the P-weighted inner product.

    Args:
        x: First vector
        y: Second vector
        P: Positive definite matrix

    Returns:
        <x, y>_P
    """
    return float(x @ P @ y)


def compute_eviction_P_norm_analysis(M_sim_after: np.ndarray,
                                      eviction_by_stage: Dict[int, float],
                                      P: np.ndarray,
                                      M_star: np.ndarray,
                                      l0: int, l_A: int, l_B: int,
                                      lambda_A: float, lambda_B: float) -> Dict:
    """
    Experiment 1: Analyze the effect of eviction on P-norm.

    Computes the P-norm distance to M* before and after eviction.
    "Before eviction" state is reconstructed by adding back evicted requests.

    Args:
        M_sim_after: Simulation state vector AFTER eviction (merged compensated)
        eviction_by_stage: Dict mapping stage -> eviction amount
        P: Lyapunov matrix
        M_star: Equilibrium state
        l0, l_A, l_B, lambda_A, lambda_B: System parameters

    Returns:
        Dictionary with:
        - has_eviction: whether eviction occurred
        - dist_before: ||M_before - M*||_P^2
        - dist_after: ||M_after - M*||_P^2
        - dist_decreased: whether distance decreased (dist_after <= dist_before)
        - dist_change: dist_after - dist_before (negative means decrease)
    """
    # Check if any eviction occurred
    total_eviction = sum(eviction_by_stage.values())
    has_eviction = total_eviction > 1e-9

    if not has_eviction:
        # No eviction, before = after
        dist_after = compute_P_norm_squared(M_sim_after - M_star, P)
        return {
            'has_eviction': False,
            'dist_before': dist_after,
            'dist_after': dist_after,
            'dist_decreased': True,  # Trivially true (no change)
            'dist_change': 0.0,
        }

    # Reconstruct "before eviction" state by adding back evicted requests
    # Eviction is in terms of raw requests, need to convert to merged compensated
    q = lambda_B / (lambda_A + lambda_B)
    total_lambda = lambda_A + lambda_B

    M_before = M_sim_after.copy()
    for stage, amount in eviction_by_stage.items():
        if amount > 1e-9 and 0 <= stage < len(M_before):
            # For stage < l_A: both types, eviction adds directly
            # For stage >= l_A: only Type B, need to compensate
            if stage < l_A:
                # Both types present, eviction is sum of both types
                M_before[stage] += amount
            else:
                # Only Type B, compensated value = B * (total_lambda / lambda_B)
                M_before[stage] += amount * total_lambda / lambda_B

    # Compute P-norm distances
    dist_before = compute_P_norm_squared(M_before - M_star, P)
    dist_after = compute_P_norm_squared(M_sim_after - M_star, P)

    return {
        'has_eviction': True,
        'dist_before': dist_before,
        'dist_after': dist_after,
        'dist_decreased': dist_after <= dist_before + 1e-9,
        'dist_change': dist_after - dist_before,
        'M_before': M_before.tolist(),
        'M_after': M_sim_after.tolist(),
    }


def compute_inner_product_analysis(M_theory: np.ndarray,
                                    M_sim: np.ndarray,
                                    P: np.ndarray,
                                    M_star: np.ndarray) -> Dict:
    """
    Experiment 2: Analyze the sign of <x^S, Δ>_P.

    For V^T - V^S = 2<x^S, Δ>_P + ||Δ||_P^2, we need <x^S, Δ>_P >= 0
    to guarantee V^T >= V^S (since ||Δ||_P^2 >= 0 always).

    Args:
        M_theory: Theory state vector (merged compensated)
        M_sim: Simulation state vector (merged compensated)
        P: Lyapunov matrix
        M_star: Equilibrium state

    Returns:
        Dictionary with:
        - x_S: M^S - M*
        - Delta: M^T - M^S
        - inner_prod_P: <x^S, Δ>_P
        - Delta_P_norm_sq: ||Δ||_P^2
        - inner_prod_geq_0: whether <x^S, Δ>_P >= 0
        - V_diff_decomposition: V^T - V^S = 2<x^S, Δ>_P + ||Δ||_P^2
    """
    x_S = M_sim - M_star
    Delta = M_theory - M_sim

    inner_prod_P = compute_P_inner_product(x_S, Delta, P)
    Delta_P_norm_sq = compute_P_norm_squared(Delta, P)

    # V^T - V^S decomposition
    V_diff = 2 * inner_prod_P + Delta_P_norm_sq

    # Also compute V^T and V^S directly for verification
    V_theory = compute_P_norm_squared(M_theory - M_star, P)
    V_sim = compute_P_norm_squared(M_sim - M_star, P)

    return {
        'x_S_norm': float(np.linalg.norm(x_S)),
        'Delta_norm': float(np.linalg.norm(Delta)),
        'inner_prod_P': inner_prod_P,
        'Delta_P_norm_sq': Delta_P_norm_sq,
        'inner_prod_geq_0': inner_prod_P >= -1e-9,
        'V_diff_decomposition': V_diff,
        'V_diff_actual': V_theory - V_sim,
        'decomposition_error': abs(V_diff - (V_theory - V_sim)),
    }


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
