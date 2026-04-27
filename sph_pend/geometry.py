from __future__ import annotations

import numpy as np

from .dynamics import PendulumParams


def pivot_positions(t: np.ndarray, params: PendulumParams) -> np.ndarray:
    """Vectorized pivot position for all times."""

    alpha = params.drive_omega * t
    return np.column_stack(
        (
            params.pivot_radius * np.cos(alpha),
            params.pivot_radius * np.sin(alpha),
            np.zeros_like(t),
        )
    )


def trajectory_from_state(
    t: np.ndarray,
    states: np.ndarray,
    params: PendulumParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pivot and bob positions for a whole state history."""

    theta = states[:, 0]
    phi = states[:, 1]
    pivot = pivot_positions(t, params)
    relative = np.column_stack(
        (
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            -np.cos(theta),
        )
    )
    return pivot, pivot + params.length * relative


def position_from_state(
    t: float,
    state: np.ndarray,
    params: PendulumParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pivot and bob positions for one state."""

    theta, phi = state[0], state[1]
    alpha = params.drive_omega * t
    pivot = np.array(
        (
            params.pivot_radius * np.cos(alpha),
            params.pivot_radius * np.sin(alpha),
            0.0,
        ),
        dtype=float,
    )
    relative = np.array(
        (
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            -np.cos(theta),
        ),
        dtype=float,
    )
    return pivot, pivot + params.length * relative


def energy_history(
    t: np.ndarray,
    states: np.ndarray,
    params: PendulumParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return kinetic, potential, and total energy histories."""

    theta, phi = states[:, 0], states[:, 1]
    theta_dot, phi_dot = states[:, 2], states[:, 3]
    alpha = params.drive_omega * t
    pivot_velocity = np.column_stack(
        (
            -params.pivot_radius * params.drive_omega * np.sin(alpha),
            params.pivot_radius * params.drive_omega * np.cos(alpha),
            np.zeros_like(t),
        )
    )
    u_dot = np.column_stack(
        (
            theta_dot * np.cos(theta) * np.cos(phi)
            - phi_dot * np.sin(theta) * np.sin(phi),
            theta_dot * np.cos(theta) * np.sin(phi)
            + phi_dot * np.sin(theta) * np.cos(phi),
            theta_dot * np.sin(theta),
        )
    )
    velocity = pivot_velocity + params.length * u_dot
    kinetic = 0.5 * params.mass * np.sum(velocity * velocity, axis=1)
    _, bob = trajectory_from_state(t, states, params)
    potential = params.mass * params.gravity * bob[:, 2]
    return kinetic, potential, kinetic + potential
