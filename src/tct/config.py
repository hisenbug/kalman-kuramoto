from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class PhysicsParams:
    """Shared physics constants. Values match Jan 30 research note."""
    N: int = 2000
    dt: float = 0.05
    T: int = 4000
    coupling_J: float = 1.5
    noise_sigma: float = 3.0
    omega_std: float = 0.5


@dataclass(frozen=True)
class PredictiveParams:
    """Parameters specific to the Kuramoto-Kalman predictive agent.

    K is 'model precision' — the inverse-variance of the prior on the internal
    estimate of the global order-parameter phase psi. High K => heavy mass
    (slow belief, low Kalman gain). This is the 'inertial' interpretation
    from the singular-perturbation framing in the EPL letter.

    epsilon is the fraction of the N agents each agent samples per timestep.
    """
    epsilon: float = 0.005
    K: float = 50.0


def to_dict(obj: Any) -> dict:
    return asdict(obj)
