from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .dynamics import PendulumParams
from .geometry import energy_history, trajectory_from_state

MPL_CONFIG_DIR = Path(__file__).resolve().parents[1] / ".mplconfig"
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))


def _set_axes_equal(ax: object, points: np.ndarray) -> None:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    centers = 0.5 * (mins + maxs)
    radius = max(0.5 * np.max(maxs - mins), 0.5) * 1.12
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def build_visualizer(
    t: np.ndarray,
    states: np.ndarray,
    params: PendulumParams,
    fps: int,
    playback_speed: float,
    trail_seconds: float,
    animate: bool,
) -> tuple[object, object | None]:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    pivot, bob = trajectory_from_state(t, states, params)
    kinetic, potential, total = energy_history(t, states, params)
    theta_deg = np.rad2deg(states[:, 0])
    phi_deg = np.rad2deg(np.unwrap(states[:, 1]))
    fig = plt.figure(figsize=(13.0, 7.2))
    grid = fig.add_gridspec(2, 2, width_ratios=(2.25, 1.0), wspace=0.28, hspace=0.34)
    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    ax_angles = fig.add_subplot(grid[0, 1])
    ax_energy = fig.add_subplot(grid[1, 1])

    circle = np.linspace(0.0, 2.0 * np.pi, 240)
    ax3d.plot(params.pivot_radius * np.cos(circle), params.pivot_radius * np.sin(circle), 0 * circle, "--", color="0.25")
    ax3d.plot(bob[:, 0], bob[:, 1], bob[:, 2], color="#2a9d8f", alpha=0.18)
    rod_line, = ax3d.plot([], [], [], color="#1f2933", linewidth=2.6)
    trail_line, = ax3d.plot([], [], [], color="#2a9d8f", linewidth=2.0)
    pivot_marker, = ax3d.plot([], [], [], "o", color="#e76f51", markersize=7)
    bob_marker, = ax3d.plot([], [], [], "o", color="#264653", markersize=11)
    time_label = ax3d.text2D(0.03, 0.94, "", transform=ax3d.transAxes)
    _set_axes_equal(ax3d, np.vstack((pivot, bob)))
    ax3d.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]", title="Driven spherical pendulum")

    ax_angles.plot(t, theta_deg, label="theta")
    ax_angles.plot(t, phi_deg, label="phi unwrapped")
    theta_point, = ax_angles.plot([], [], "o")
    phi_point, = ax_angles.plot([], [], "o")
    angle_cursor = ax_angles.axvline(t[0], color="0.15", alpha=0.45)
    ax_angles.set(xlabel="time [s]", ylabel="angle [deg]")
    ax_angles.grid(True, alpha=0.25)
    ax_angles.legend()

    ax_energy.plot(t, kinetic, label="T")
    ax_energy.plot(t, potential, label="V")
    ax_energy.plot(t, total, label="T + V")
    energy_point, = ax_energy.plot([], [], "o")
    energy_cursor = ax_energy.axvline(t[0], color="0.15", alpha=0.45)
    ax_energy.set(xlabel="time [s]", ylabel="energy [J]")
    ax_energy.grid(True, alpha=0.25)
    ax_energy.legend()

    dt = t[1] - t[0] if len(t) > 1 else 1.0
    stride = max(1, int(round(playback_speed / max(fps * dt, 1.0e-12))))
    frames = np.arange(0, len(t), stride, dtype=int)
    frames = np.append(frames, len(t) - 1) if frames[-1] != len(t) - 1 else frames
    trail_count = max(2, int(round(trail_seconds / dt)))

    def update(index: int):
        start = max(0, index - trail_count)
        rod_line.set_data_3d([pivot[index, 0], bob[index, 0]], [pivot[index, 1], bob[index, 1]], [0, bob[index, 2]])
        trail_line.set_data_3d(bob[start : index + 1, 0], bob[start : index + 1, 1], bob[start : index + 1, 2])
        pivot_marker.set_data_3d([pivot[index, 0]], [pivot[index, 1]], [pivot[index, 2]])
        bob_marker.set_data_3d([bob[index, 0]], [bob[index, 1]], [bob[index, 2]])
        theta_point.set_data([t[index]], [theta_deg[index]])
        phi_point.set_data([t[index]], [phi_deg[index]])
        angle_cursor.set_xdata([t[index], t[index]])
        energy_point.set_data([t[index]], [total[index]])
        energy_cursor.set_xdata([t[index], t[index]])
        time_label.set_text(f"t = {t[index]:.2f} s\ntheta = {theta_deg[index]:.1f} deg\nphi = {phi_deg[index]:.1f} deg")

    animation = FuncAnimation(fig, update, frames=frames, interval=1000.0 / fps) if animate else None
    update(frames[0] if animate else len(t) - 1)
    return fig, animation
