"""
Theory Simulator: Unconstrained Version for Coupling Experiments

This simulator implements the theoretical model from multiple_discrete.tex:
- No eviction (allows negative admission)
- No non-negativity constraint (state can be negative)
- Token balance: completion_tokens - increment_tokens = admission_tokens

Used to compare with the actual Simulation which has eviction and non-negativity.
"""

from typing import Dict, List, Tuple


class TheorySimulator:
    """
    Unconstrained simulator matching the theoretical analysis.

    Key differences from MultiTypeLLMSimulator:
    - No eviction: when available_tokens < 0, admission becomes negative
    - State can be negative: X[stage][type] can be < 0
    - Still maintains token balance equation
    """

    def __init__(self, l0: int, l_A: int, l_B: int, B: int,
                 lambda_A: float, lambda_B: float):
        """
        Initialize Theory Simulator.

        Args:
            l0: Initial/prefill length (same for both types)
            l_A: Decode length for Type A (number of decode steps)
            l_B: Decode length for Type B (number of decode steps), l_A < l_B
            B: GPU capacity (total tokens)
            lambda_A: Arrival rate for Type A
            lambda_B: Arrival rate for Type B
        """
        self.l0 = l0
        self.l_A = l_A
        self.l_B = l_B
        self.B = B
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B

        # Proportion p = lambda_A / (lambda_A + lambda_B)
        self.p = lambda_A / (lambda_A + lambda_B)
        self.q = 1 - self.p  # = lambda_B / (lambda_A + lambda_B)

        # Max stage (0-indexed): Type A goes 0..l_A-1, Type B goes 0..l_B-1
        self.max_stage = max(l_A, l_B) - 1

        # State: X[stage][type], type 0 = A, type 1 = B
        # Initialized to zeros
        self.X: Dict[int, List[float]] = {}
        for stage in range(self.max_stage + 1):
            self.X[stage] = [0.0, 0.0]  # [Type A, Type B]

        # Batch counter
        self.n = 0

        # History
        self.history = {
            'admissions': [],  # Total admission each batch (can be negative)
            'states': [],      # State snapshots
        }

    def set_initial_state(self, X_init: Dict[int, List[float]]):
        """Set initial state from external specification."""
        for stage in X_init:
            if stage in self.X:
                self.X[stage] = X_init[stage].copy()

    def _is_valid_stage(self, stage: int, type_idx: int) -> bool:
        """Check if a stage is valid for a given type."""
        if type_idx == 0:  # Type A
            return 0 <= stage < self.l_A
        else:  # Type B
            return 0 <= stage < self.l_B

    def _compute_length(self, stage: int) -> int:
        """Compute length from stage: length = l0 + stage."""
        return self.l0 + stage

    def _compute_tokens(self, stage: int, count: float) -> float:
        """Compute tokens for requests at a stage."""
        # At processing time, request has (length+1) tokens in batch
        length = self._compute_length(stage)
        return (length + 1) * count

    def update(self):
        """
        Execute one batch update following the theoretical model.

        Token balance:
        - completion_tokens: tokens released by completing requests
        - increment_tokens: tokens increased by stage advancement
        - admission_tokens: new tokens admitted (can be negative)

        completion_tokens - increment_tokens = admission_tokens
        """
        # ===== 1. Calculate completion tokens =====
        # Type A completes at stage l_A - 1
        # Type B completes at stage l_B - 1
        completion_A = self.X[self.l_A - 1][0] if self.l_A - 1 in self.X else 0.0
        completion_B = self.X[self.l_B - 1][1] if self.l_B - 1 in self.X else 0.0

        # Tokens released by completion
        # When completing, request had (l0 + l_A) tokens for A, (l0 + l_B) for B
        completion_tokens = (self.l0 + self.l_A) * completion_A + \
                           (self.l0 + self.l_B) * completion_B

        # ===== 2. Calculate increment tokens =====
        # All non-completing requests advance one stage, each gains 1 token
        increment_count = 0.0
        for stage in range(self.max_stage + 1):
            for type_idx in range(2):
                if self._is_valid_stage(stage, type_idx):
                    # Exclude completing requests
                    if type_idx == 0 and stage == self.l_A - 1:
                        continue  # Type A completing
                    if type_idx == 1 and stage == self.l_B - 1:
                        continue  # Type B completing
                    increment_count += self.X[stage][type_idx]

        increment_tokens = increment_count  # Each request gains 1 token

        # ===== 3. Calculate admission (can be negative) =====
        available_tokens = completion_tokens - increment_tokens

        # Admission distributed by arrival rate ratio
        # Total new requests take (l0 + 1) tokens each
        # admission_A * (l0+1) + admission_B * (l0+1) = available_tokens
        # admission_A / admission_B = lambda_A / lambda_B

        denominator = (self.l0 + 1)
        total_admission = available_tokens / denominator

        admission_A = total_admission * self.p
        admission_B = total_admission * self.q

        # ===== 4. State advancement (before adding new admissions) =====
        X_new: Dict[int, List[float]] = {}
        for stage in range(self.max_stage + 1):
            X_new[stage] = [0.0, 0.0]

        # Advance: stage -> stage + 1 (except completing requests)
        for stage in range(self.max_stage + 1):
            for type_idx in range(2):
                if self._is_valid_stage(stage, type_idx):
                    next_stage = stage + 1
                    # Check if next stage is valid (not completing)
                    if self._is_valid_stage(next_stage, type_idx):
                        X_new[next_stage][type_idx] = self.X[stage][type_idx]
                    # If next stage is completion stage, requests complete (removed)

        # ===== 5. Add new admissions to stage 0 =====
        X_new[0][0] += admission_A
        X_new[0][1] += admission_B

        # ===== 6. Record history =====
        self.history['admissions'].append({
            'batch': self.n,
            'admission_A': admission_A,
            'admission_B': admission_B,
            'admission_total': admission_A + admission_B,
            'completion_tokens': completion_tokens,
            'increment_tokens': increment_tokens,
            'available_tokens': available_tokens,
        })

        self.history['states'].append({
            'batch': self.n,
            'state': {stage: self.X[stage].copy() for stage in self.X}
        })

        # ===== 7. Update state =====
        self.X = X_new
        self.n += 1

    def run(self, steps: int):
        """Run simulation for specified number of steps."""
        for _ in range(steps):
            self.update()

    def get_admission_history(self) -> List[float]:
        """Get list of total admissions per batch."""
        return [h['admission_total'] for h in self.history['admissions']]

    def get_state_history(self) -> List[Dict]:
        """Get list of state snapshots."""
        return self.history['states']


# Test
if __name__ == "__main__":
    print("Theory Simulator Test")
    print("=" * 60)

    # Parameters from design
    l0, l_A, l_B = 3, 2, 3
    B = 60
    lambda_A, lambda_B = 1.0, 1.0

    sim = TheorySimulator(l0, l_A, l_B, B, lambda_A, lambda_B)

    # Initial condition: only stage 0, proportional
    p = lambda_A / (lambda_A + lambda_B)
    N = B / (l0 + 1)

    X_init = {
        0: [p * N, (1 - p) * N],  # stage 0
    }
    sim.set_initial_state(X_init)

    print(f"Parameters: l0={l0}, l_A={l_A}, l_B={l_B}, B={B}")
    print(f"p = {p}, N = {N}")
    print(f"Initial: X[0] = {sim.X[0]}")
    print()

    # Run a few steps
    for step in range(10):
        sim.update()
        adm = sim.history['admissions'][-1]
        print(f"Batch {step}: admission_total = {adm['admission_total']:.2f}, "
              f"completion = {adm['completion_tokens']:.2f}, "
              f"increment = {adm['increment_tokens']:.2f}")

    print()
    print("Final state:")
    for stage in range(sim.max_stage + 1):
        if any(abs(x) > 0.01 for x in sim.X[stage]):
            print(f"  Stage {stage}: {sim.X[stage]}")
