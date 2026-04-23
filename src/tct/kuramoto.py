"""All-to-all Kuramoto baseline. No internal model, no sparse observations.

This is the reference point for the 'predictive uses ~5% bandwidth' claim.
Each agent couples to the full population every step:

    dtheta_i/dt = omega_i + J * r * sin(psi - theta_i)

where (r, psi) are the global order parameter, computed directly (i.e. every
agent 'observes' every other agent). Bandwidth = N^2 per step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import PhysicsParams
from .cost import interaction_cost_per_step
from .device import sync


@dataclass
class BaselineResult:
    r_t: np.ndarray                 # shape (T,)
    interaction_cost_total: float   # cumulative over T steps
    erasure_cost_total: float       # 0 for baseline (no internal model to erase)
    r_final: float
    seed: int
    wall_time_s: float


def run_baseline(
    phys: PhysicsParams,
    seed: int,
    device: torch.device,
) -> BaselineResult:
    """Classic Kuramoto with global mean-field coupling. No Kalman filter."""
    import time

    t_start = time.time()
    g = torch.Generator(device=device).manual_seed(seed)

    omega = torch.randn(phys.N, generator=g, device=device) * phys.omega_std
    theta = torch.rand(phys.N, generator=g, device=device) * (2.0 * np.pi)

    r_t_gpu = torch.zeros(phys.T, device=device)

    for t in range(phys.T):
        z = torch.mean(torch.exp(1j * theta))
        r = torch.abs(z)
        psi = torch.angle(z)
        r_t_gpu[t] = r

        dtheta = omega + phys.coupling_J * r * torch.sin(psi - theta)
        theta = theta + dtheta * phys.dt

    sync(device)
    r_t = r_t_gpu.detach().cpu().numpy()

    interaction = interaction_cost_per_step(phys.N, epsilon=1.0) * phys.T

    return BaselineResult(
        r_t=r_t,
        interaction_cost_total=float(interaction),
        erasure_cost_total=0.0,
        r_final=float(r_t[-1]),
        seed=seed,
        wall_time_s=time.time() - t_start,
    )
