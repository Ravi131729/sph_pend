from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LOCAL_MPL_CONFIG_DIR = Path(__file__).with_name(".mplconfig")
LOCAL_MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_MPL_CONFIG_DIR))


@dataclass(frozen=True)
class PendulumParams:
    """Physical parameters for the driven spherical pendulum."""

    mass: float = 1.0
    length: float = 1.0
    gravity: float = 9.81
    pivot_radius: float = 1
    drive_omega: float = 2.0
    singularity_eps: float = 1.0e-6


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

    if abs(s) < params.singularity_eps:
        s_safe = params.singularity_eps if s >= 0.0 else -params.singularity_eps
    else:
        s_safe = s

    phi_ddot = (
        -2.0 * (c / s_safe) * theta_dot * phi_dot
        - forcing * np.sin(delta) / s_safe
    )

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

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if t_final <= 0.0:
        raise ValueError("t_final must be positive")
    if params.length <= 0.0:
        raise ValueError("pendulum length must be positive")
    if params.mass <= 0.0:
        raise ValueError("mass must be positive")
    if params.pivot_radius < 0.0:
        raise ValueError("pivot radius must be non-negative")

    n_steps = int(np.ceil(t_final / dt))
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)
    states = np.zeros((n_steps + 1, 4), dtype=float)
    states[0] = initial_state

    for k in range(n_steps):
        states[k + 1] = rk4_step(t[k], states[k], dt, params)

    return t, states


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
    bob = pivot + params.length * relative
    return pivot, bob


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
    bob = pivot + params.length * relative
    return pivot, bob


def energy_history(
    t: np.ndarray,
    states: np.ndarray,
    params: PendulumParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return kinetic, potential, and total energy histories."""

    theta = states[:, 0]
    phi = states[:, 1]
    theta_dot = states[:, 2]
    phi_dot = states[:, 3]
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


def set_axes_equal(ax: object, points: np.ndarray, pad_fraction: float = 0.12) -> None:
    """Set equal-looking 3-D limits around a point cloud."""

    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    radius = max(radius, 0.5)
    radius *= 1.0 + pad_fraction

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
    """Create either an animated or final-frame 3-D visualization."""

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    pivot, bob = trajectory_from_state(t, states, params)
    kinetic, potential, total = energy_history(t, states, params)
    theta_deg = np.rad2deg(states[:, 0])
    phi_deg = np.rad2deg(np.unwrap(states[:, 1]))

    fig = plt.figure(figsize=(13.0, 7.2))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(2.25, 1.0),
        height_ratios=(1.0, 1.0),
        wspace=0.28,
        hspace=0.34,
    )
    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    ax_angles = fig.add_subplot(grid[0, 1])
    ax_energy = fig.add_subplot(grid[1, 1])

    circle_angle = np.linspace(0.0, 2.0 * np.pi, 240)
    ax3d.plot(
        params.pivot_radius * np.cos(circle_angle),
        params.pivot_radius * np.sin(circle_angle),
        np.zeros_like(circle_angle),
        color="0.25",
        linestyle="--",
        linewidth=1.0,
        alpha=0.55,
        label="pivot path",
    )
    ax3d.plot(
        bob[:, 0],
        bob[:, 1],
        bob[:, 2],
        color="#2a9d8f",
        linewidth=1.0,
        alpha=0.18,
    )

    rod_line, = ax3d.plot([], [], [], color="#1f2933", linewidth=2.6)
    trail_line, = ax3d.plot([], [], [], color="#2a9d8f", linewidth=2.0, alpha=0.85)
    pivot_marker, = ax3d.plot([], [], [], "o", color="#e76f51", markersize=7)
    bob_marker, = ax3d.plot([], [], [], "o", color="#264653", markersize=11)
    time_label = ax3d.text2D(0.03, 0.94, "", transform=ax3d.transAxes)

    all_points = np.vstack((pivot, bob))
    set_axes_equal(ax3d, all_points)
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.set_title("Driven spherical pendulum")
    ax3d.view_init(elev=22.0, azim=-45.0)

    ax_angles.plot(t, theta_deg, color="#0077b6", linewidth=1.7, label="theta")
    ax_angles.plot(t, phi_deg, color="#d62828", linewidth=1.3, label="phi unwrapped")
    theta_point, = ax_angles.plot([], [], "o", color="#0077b6", markersize=5)
    phi_point, = ax_angles.plot([], [], "o", color="#d62828", markersize=5)
    angle_cursor = ax_angles.axvline(t[0], color="0.15", linewidth=1.0, alpha=0.45)
    ax_angles.set_xlabel("time [s]")
    ax_angles.set_ylabel("angle [deg]")
    ax_angles.grid(True, alpha=0.25)
    ax_angles.legend(loc="best")

    ax_energy.plot(t, kinetic, color="#6a4c93", linewidth=1.2, label="T")
    ax_energy.plot(t, potential, color="#f77f00", linewidth=1.2, label="V")
    ax_energy.plot(t, total, color="#111827", linewidth=1.6, label="T + V")
    energy_point, = ax_energy.plot([], [], "o", color="#111827", markersize=5)
    energy_cursor = ax_energy.axvline(t[0], color="0.15", linewidth=1.0, alpha=0.45)
    ax_energy.set_xlabel("time [s]")
    ax_energy.set_ylabel("energy [J]")
    ax_energy.grid(True, alpha=0.25)
    ax_energy.legend(loc="best")

    dt = t[1] - t[0] if len(t) > 1 else 1.0
    stride = max(1, int(round(playback_speed / max(fps * dt, 1.0e-12))))
    frame_indices = np.arange(0, len(t), stride, dtype=int)
    if frame_indices[-1] != len(t) - 1:
        frame_indices = np.append(frame_indices, len(t) - 1)

    trail_count = max(2, int(round(trail_seconds / dt)))

    def update(index: int):
        start = max(0, index - trail_count)
        rod_line.set_data_3d(
            [pivot[index, 0], bob[index, 0]],
            [pivot[index, 1], bob[index, 1]],
            [pivot[index, 2], bob[index, 2]],
        )
        trail_line.set_data_3d(
            bob[start : index + 1, 0],
            bob[start : index + 1, 1],
            bob[start : index + 1, 2],
        )
        pivot_marker.set_data_3d(
            [pivot[index, 0]],
            [pivot[index, 1]],
            [pivot[index, 2]],
        )
        bob_marker.set_data_3d(
            [bob[index, 0]],
            [bob[index, 1]],
            [bob[index, 2]],
        )

        theta_point.set_data([t[index]], [theta_deg[index]])
        phi_point.set_data([t[index]], [phi_deg[index]])
        angle_cursor.set_xdata([t[index], t[index]])
        energy_point.set_data([t[index]], [total[index]])
        energy_cursor.set_xdata([t[index], t[index]])

        time_label.set_text(
            f"t = {t[index]:.2f} s\n"
            f"theta = {theta_deg[index]:.1f} deg\n"
            f"phi = {phi_deg[index]:.1f} deg"
        )
        return (
            rod_line,
            trail_line,
            pivot_marker,
            bob_marker,
            theta_point,
            phi_point,
            angle_cursor,
            energy_point,
            energy_cursor,
            time_label,
        )

    if animate:
        animation = FuncAnimation(
            fig,
            update,
            frames=frame_indices,
            interval=1000.0 / fps,
            blit=False,
            repeat=True,
        )
    else:
        update(len(t) - 1)
        animation = None

    return fig, animation


def make_polyline(points: np.ndarray):
    """Create a PyVista polyline from an ordered point array."""

    import pyvista as pv

    points = np.asarray(points, dtype=float)
    polyline = pv.PolyData(points)
    if len(points) > 1:
        polyline.lines = np.hstack(
            ([len(points)], np.arange(len(points), dtype=np.int64))
        )
    return polyline


def show_pyvista_visualizer(
    initial_state: np.ndarray,
    params: PendulumParams,
    dt: float,
    fps: int,
    playback_speed: float,
    trail_seconds: float,
    start_paused: bool,
    save: Path | None,
    no_show: bool,
) -> None:
    """Open a live PyVista 3-D visualizer that integrates until closed."""

    try:
        import pyvista as pv
    except ImportError as exc:
        raise SystemExit(
            "PyVista is not installed. Install it with:\n\n"
            "    python3 -m pip install pyvista\n\n"
            "Then run this script again with '--visualizer pyvista'."
        ) from exc

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if params.length <= 0.0:
        raise ValueError("pendulum length must be positive")
    if params.mass <= 0.0:
        raise ValueError("mass must be positive")
    if params.pivot_radius < 0.0:
        raise ValueError("pivot radius must be non-negative")

    state = np.asarray(initial_state, dtype=float).copy()
    reset_state = state.copy()
    sim_time = 0.0
    steps_per_frame = max(1, int(round(playback_speed / max(fps * dt, 1.0e-12))))
    rendered_trail_count = max(2, int(round(trail_seconds * fps / playback_speed)))
    pivot, bob = position_from_state(sim_time, state, params)

    plotter = pv.Plotter(
        window_size=(1200, 820),
        title="Driven spherical pendulum",
        off_screen=no_show,
    )
    plotter.set_background("#f8fafc")
    plotter.add_axes(line_width=2, labels_off=False)

    circle_angle = np.linspace(0.0, 2.0 * np.pi, 240)
    pivot_circle = np.column_stack(
        (
            params.pivot_radius * np.cos(circle_angle),
            params.pivot_radius * np.sin(circle_angle),
            np.zeros_like(circle_angle),
        )
    )
    plotter.add_mesh(
        make_polyline(pivot_circle),
        color="#6b7280",
        line_width=2,
        label="pivot path",
    )

    rod_mesh = make_polyline(np.vstack((pivot, bob)))
    trail_points = np.repeat(bob[None, :], rendered_trail_count, axis=0)
    trail_mesh = make_polyline(trail_points)

    pivot_marker_radius = max(0.025 * params.length, 0.012)
    bob_radius = max(0.055 * params.length, 0.025)
    pivot_mesh = pv.Sphere(
        radius=pivot_marker_radius,
        theta_resolution=32,
        phi_resolution=16,
    )
    bob_mesh = pv.Sphere(
        radius=bob_radius,
        theta_resolution=40,
        phi_resolution=20,
    )
    pivot_base_points = pivot_mesh.points.copy()
    bob_base_points = bob_mesh.points.copy()

    plotter.add_mesh(rod_mesh, color="#111827", line_width=7)
    plotter.add_mesh(trail_mesh, color="#0f766e", line_width=4)
    plotter.add_mesh(pivot_mesh, color="#e76f51", smooth_shading=True)
    plotter.add_mesh(bob_mesh, color="#264653", smooth_shading=True)

    center = np.zeros(3, dtype=float)
    radius = max(params.pivot_radius + params.length, params.length, 0.5)
    plotter.show_bounds(
        bounds=(
            center[0] - radius,
            center[0] + radius,
            center[1] - radius,
            center[1] + radius,
            center[2] - radius,
            center[2] + radius,
        ),
        grid="front",
        location="outer",
        all_edges=True,
        color="#475569",
    )
    plotter.camera_position = [
        (center[0] + 2.3 * radius, center[1] - 2.6 * radius, center[2] + 1.45 * radius),
        tuple(center),
        (0.0, 0.0, 1.0),
    ]

    playback = {"playing": not start_paused}

    def update_geometry(clear_trail: bool = False) -> None:
        nonlocal trail_points

        pivot_now, bob_now = position_from_state(sim_time, state, params)
        if clear_trail:
            trail_points = np.repeat(bob_now[None, :], rendered_trail_count, axis=0)
        else:
            trail_points = np.roll(trail_points, -1, axis=0)
            trail_points[-1] = bob_now

        rod_mesh.points = np.vstack((pivot_now, bob_now))
        trail_mesh.points = trail_points
        pivot_mesh.points = pivot_base_points + pivot_now
        bob_mesh.points = bob_base_points + bob_now

        theta_deg = np.rad2deg(state[0])
        phi_deg = np.rad2deg(state[1])

        controls = "space: play/pause   n: step   c: clear trail   r: reset"
        if save is not None:
            controls += "   s: screenshot"

        plotter.add_text(
            (
                f"t = {sim_time:.2f} s\n"
                f"pivot radius = {params.pivot_radius:.3g} m\n"
                f"theta = {theta_deg:.1f} deg\n"
                f"phi = {phi_deg:.1f} deg\n"
                f"{controls}"
            ),
            name="state_text",
            position="upper_left",
            font_size=11,
            color="#111827",
        )
        plotter.render()

    def toggle_playback() -> None:
        playback["playing"] = not playback["playing"]

    def reset_playback() -> None:
        nonlocal sim_time, state

        playback["playing"] = False
        sim_time = 0.0
        state = reset_state.copy()
        update_geometry(clear_trail=True)

    def step_forward() -> None:
        nonlocal sim_time, state

        playback["playing"] = False
        for _ in range(steps_per_frame):
            state = rk4_step(sim_time, state, dt, params)
            sim_time += dt
        update_geometry()

    def timer_callback(_step: int | None = None) -> None:
        nonlocal sim_time, state

        if not playback["playing"]:
            return
        for _ in range(steps_per_frame):
            state = rk4_step(sim_time, state, dt, params)
            sim_time += dt
        update_geometry()

    def clear_trail() -> None:
        update_geometry(clear_trail=True)

    def save_screenshot() -> None:
        if save is not None:
            plotter.screenshot(str(save))
            print(f"Saved PyVista screenshot to {save}")

    update_geometry(clear_trail=True)
    plotter.add_key_event("space", toggle_playback)
    plotter.add_key_event("r", reset_playback)
    plotter.add_key_event("n", step_forward)
    plotter.add_key_event("c", clear_trail)
    if save is not None:
        plotter.add_key_event("s", save_screenshot)
    plotter.add_timer_event(
        max_steps=2_147_483_647,
        duration=int(round(1000.0 / fps)),
        callback=timer_callback,
    )

    if save is not None and save.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise SystemExit(
            "The PyVista backend currently saves screenshots only. "
            "Use .png/.jpg, or use '--visualizer matplotlib' for GIF export."
        )

    if save is not None and no_show:
        plotter.show(screenshot=str(save), auto_close=no_show)
        print(f"Saved PyVista screenshot to {save}")
    elif no_show:
        plotter.close()
    else:
        plotter.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate a spherical pendulum whose pivot moves in a circle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mass",
        type=float,
        default=PendulumParams.mass,
        help="bob mass m [kg]",
    )
    parser.add_argument(
        "--length",
        type=float,
        default=PendulumParams.length,
        help="pendulum length l [m]",
    )
    parser.add_argument(
        "--gravity",
        type=float,
        default=PendulumParams.gravity,
        help="gravity g [m/s^2]",
    )
    parser.add_argument(
        "--pivot-radius",
        type=float,
        default=PendulumParams.pivot_radius,
        help="radius a of the pivot circle [m]",
    )
    parser.add_argument(
        "--drive-omega",
        type=float,
        default=PendulumParams.drive_omega,
        help="pivot angular velocity Omega [rad/s]",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=14.0,
        help="Matplotlib simulation duration [s]; PyVista runs live until closed",
    )
    parser.add_argument("--dt", type=float, default=0.002, help="integration step [s]")
    parser.add_argument("--theta0", type=float, default=35.0, help="initial theta [deg]")
    parser.add_argument("--phi0", type=float, default=20.0, help="initial phi [deg]")
    parser.add_argument(
        "--theta-dot0",
        type=float,
        default=0.0,
        help="initial theta rate [deg/s]",
    )
    parser.add_argument(
        "--phi-dot0",
        type=float,
        default=65.0,
        help="initial phi rate [deg/s]",
    )
    parser.add_argument("--fps", type=int, default=30, help="animation frames per second")
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="simulated seconds per real second in the animation",
    )
    parser.add_argument(
        "--trail-seconds",
        type=float,
        default=2.5,
        help="length of the bright bob trail [s]",
    )
    parser.add_argument(
        "--visualizer",
        choices=("matplotlib", "pyvista"),
        default="matplotlib",
        help="3-D visualization backend",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Matplotlib: show final frame; PyVista: start paused",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="optional output path; PyVista saves screenshots with the s key",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="do not open a plotting window after running",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if args.playback_speed <= 0.0:
        raise ValueError("playback-speed must be positive")

    params = PendulumParams(
        mass=args.mass,
        length=args.length,
        gravity=args.gravity,
        pivot_radius=args.pivot_radius,
        drive_omega=args.drive_omega,
    )
    initial_state = np.deg2rad(
        np.array([args.theta0, args.phi0, args.theta_dot0, args.phi_dot0], dtype=float)
    )

    if args.visualizer == "pyvista":
        show_pyvista_visualizer(
            initial_state=initial_state,
            params=params,
            dt=args.dt,
            fps=args.fps,
            playback_speed=args.playback_speed,
            trail_seconds=args.trail_seconds,
            start_paused=args.static,
            save=args.save,
            no_show=args.no_show,
        )
    else:
        t, states = simulate(params, initial_state, args.time, args.dt)
        fig, animation = build_visualizer(
            t=t,
            states=states,
            params=params,
            fps=args.fps,
            playback_speed=args.playback_speed,
            trail_seconds=args.trail_seconds,
            animate=not args.static,
        )

        if args.save is not None:
            suffix = args.save.suffix.lower()
            if animation is None:
                fig.savefig(args.save, dpi=180)
            elif suffix == ".gif":
                animation.save(args.save, writer="pillow", fps=args.fps)
            else:
                animation.save(args.save, fps=args.fps)
            print(f"Saved visualization to {args.save}")

        if args.no_show:
            import matplotlib.pyplot as plt

            plt.close(fig)
        else:
            import matplotlib.pyplot as plt

            plt.show()


if __name__ == "__main__":
    main()
