from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PendulumParams:
    """Physical parameters for the driven spherical pendulum."""

    mass: float = 1.0
    length: float = 1.0
    gravity: float = 9.81
    pivot_radius: float = 1.0
    drive_omega: float = 2.0
    singularity_eps: float = 1.0e-6


def validate_params(params: PendulumParams, dt: float | None = None) -> None:
    if dt is not None and dt <= 0.0:
        raise ValueError("dt must be positive")
    if params.length <= 0.0:
        raise ValueError("pendulum length must be positive")
    if params.mass <= 0.0:
        raise ValueError("mass must be positive")
    if params.pivot_radius < 0.0:
        raise ValueError("pivot radius must be non-negative")


def rhs(t: float, state: np.ndarray, params: PendulumParams) -> np.ndarray:
    """Return time derivative of [theta, phi, theta_dot, phi_dot]."""

    theta, phi, theta_dot, phi_dot = state
    s = np.sin(theta)
    c = np.cos(theta)
    delta = phi - params.drive_omega * t
    forcing = params.pivot_radius * params.drive_omega**2 / params.length

    theta_ddot = (
        s * c * phi_dot**2
        + forcing * c * np.cos(delta)
        - (params.gravity / params.length) * s
    )
    s_safe = s
    if abs(s_safe) < params.singularity_eps:
        s_safe = params.singularity_eps if s >= 0.0 else -params.singularity_eps
    phi_ddot = -2.0 * (c / s_safe) * theta_dot * phi_dot
    phi_ddot -= forcing * np.sin(delta) / s_safe

    return np.array([theta_dot, phi_dot, theta_ddot, phi_ddot], dtype=float)


def rk4_step(t: float, state: np.ndarray, dt: float, params: PendulumParams) -> np.ndarray:
    """Advance one fixed-size RK4 step."""

    k1 = rhs(t, state, params)
    k2 = rhs(t + 0.5 * dt, state + 0.5 * dt * k1, params)
    k3 = rhs(t + 0.5 * dt, state + 0.5 * dt * k2, params)
    k4 = rhs(t + dt, state + dt * k3, params)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate(
    params: PendulumParams,
    initial_state: np.ndarray,
    t_final: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the angle equations and return time and state histories."""

    validate_params(params, dt)
    if t_final <= 0.0:
        raise ValueError("t_final must be positive")

    n_steps = int(np.ceil(t_final / dt))
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)
    states = np.zeros((n_steps + 1, 4), dtype=float)
    states[0] = initial_state

    for k in range(n_steps):
        states[k + 1] = rk4_step(t[k], states[k], dt, params)
    return t, states
