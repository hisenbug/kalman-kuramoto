"""Cost accounting, with interaction and erasure channels kept separate.

Two channels, never summed with a magic constant:

  interaction_cost[t]  proportional to bandwidth consumed this step.
      Baseline (all-to-all):   N**2
      Predictive (epsilon):    N**2 * epsilon

  erasure_cost[t]  Landauer-style cost to reduce posterior entropy given
      the observation, in nats * N. The agent pays
          0.5 * log(1 + Var[innovation] / R)
      per update (information gain in nats), times N agents.

The poster plots them stacked so the viewer sees the composition, not a
weighted scalar.
"""
from __future__ import annotations

import numpy as np


def interaction_cost_per_step(N: int, epsilon: float = 1.0) -> float:
    """Per-step interaction cost. epsilon=1.0 for baseline, else sparse."""
    return float(N * N) * float(epsilon)


def erasure_cost_per_step(innovation_var: float, obs_variance_R: float, N: int) -> float:
    """Landauer erasure cost for one Kalman update, in nats * agents.

    Information gain per agent = 0.5 * log(1 + Var[y-y_hat] / R). Multiplied
    by N to get the total thermodynamic floor for one timestep of the
    population's belief update.
    """
    innovation_var = max(innovation_var, 1e-12)
    obs_variance_R = max(obs_variance_R, 1e-12)
    return 0.5 * np.log(1.0 + innovation_var / obs_variance_R) * N
