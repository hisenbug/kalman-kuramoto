"""Predictive Kuramoto-Kalman agent.

Two timescales, kept structurally visible (matched asymptotics):

  FAST  — phase correction via the Kalman gain K_gain = P / (P + R).
          P is the prior variance on the internal phase estimate (= 1/K,
          where K is the model precision / inertial mass). R is the
          observation variance from sparse sampling, sigma^2 / (epsilon * N).

  SLOW  — frequency integration. We compute the instantaneous frequency
          innovation omega_err = error / dt (rad/s), then integrate with
          forward-Euler:

              internal_freq += alpha_freq * omega_err * dt

          This is algebraically the same as `alpha_freq * error`, but the
          explicit rate form makes the fast/slow structure visible: error
          is the fast-timescale inner solution, the integration over dt is
          the slow outer integration.

Coupling uses the agent's INTERNAL estimate of the order parameter (honest
coupling): if r_est is low, the agent doesn't over-commit force.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import PhysicsParams, PredictiveParams
from .cost import interaction_cost_per_step, erasure_cost_per_step
from .device import sync


@dataclass
class PredictiveResult:
    r_t: np.ndarray
    interaction_cost_total: float
    erasure_cost_total: float
    r_final: float
    seed: int
    wall_time_s: float
    # Explicit matched-condition quantities, saved so the manifest shows them:
    P_prior: float
    R_obs: float
    K_gain: float
    alpha_freq: float


def run_predictive(
    phys: PhysicsParams,
    pred: PredictiveParams,
    seed: int,
    device: torch.device,
) -> PredictiveResult:
    import time

    t_start = time.time()
    g = torch.Generator(device=device).manual_seed(seed)

    eps = max(pred.epsilon, 1e-6)
    K = max(pred.K, 1e-3)

    # Matched-condition Kalman gain -- kept in explicit P, R form.
    P_prior = 1.0 / K                                   # prior variance
    R_obs = (phys.noise_sigma ** 2) / (eps * phys.N)    # observation variance
    K_gain = P_prior / (P_prior + R_obs)                # <<< P / (P + R)

    # Slow-dynamics learning rate for frequency integration.
    alpha_freq = 0.5 / (K + 1.0)

    # State.
    omega = torch.randn(phys.N, generator=g, device=device) * phys.omega_std
    theta = torch.rand(phys.N, generator=g, device=device) * (2.0 * np.pi)
    internal_psi = torch.rand(phys.N, generator=g, device=device) * (2.0 * np.pi)
    internal_freq = omega.clone()

    r_t_gpu = torch.zeros(phys.T, device=device)
    erasure_t = np.zeros(phys.T, dtype=np.float64)

    samples = max(int(phys.N * eps), 1)
    obs_noise_std = phys.noise_sigma / np.sqrt(samples)

    for t in range(phys.T):
        # --- Environment ---
        z_true = torch.mean(torch.exp(1j * theta))
        psi_true = torch.angle(z_true)
        r_t_gpu[t] = torch.abs(z_true)

        # --- Prediction (internal model) ---
        predicted_psi = internal_psi + internal_freq * phys.dt

        # --- Observation (sparse, noisy, scalar per agent) ---
        obs_noise = torch.randn(phys.N, generator=g, device=device) * obs_noise_std
        observed_psi = psi_true + obs_noise

        error = observed_psi - predicted_psi
        error = (error + np.pi) % (2.0 * np.pi) - np.pi  # wrap to [-pi, pi]

        # --- Thermodynamic cost of the update (Landauer) ---
        innov_var = float(torch.var(error).item())
        erasure_t[t] = erasure_cost_per_step(innov_var, R_obs, phys.N)

        # --- Fast update: phase correction with Kalman gain ---
        internal_psi = predicted_psi + K_gain * error

        # --- Slow update: frequency integration via explicit innovation rate ---
        freq_innovation = error / phys.dt        # [rad/s]
        internal_freq = internal_freq + alpha_freq * freq_innovation * phys.dt

        # --- Action: honest mean-field coupling via internal estimate ---
        z_est = torch.mean(torch.exp(1j * internal_psi))
        r_est = torch.abs(z_est)
        psi_est = torch.angle(z_est)
        dtheta = omega + phys.coupling_J * r_est * torch.sin(psi_est - theta)
        theta = theta + dtheta * phys.dt

    sync(device)
    r_t = r_t_gpu.detach().cpu().numpy()

    interaction = interaction_cost_per_step(phys.N, eps) * phys.T
    erasure = float(erasure_t.sum())

    return PredictiveResult(
        r_t=r_t,
        interaction_cost_total=float(interaction),
        erasure_cost_total=erasure,
        r_final=float(r_t[-1]),
        seed=seed,
        wall_time_s=time.time() - t_start,
        P_prior=float(P_prior),
        R_obs=float(R_obs),
        K_gain=float(K_gain),
        alpha_freq=float(alpha_freq),
    )
