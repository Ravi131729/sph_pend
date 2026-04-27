from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .dynamics import PendulumParams, simulate
from .mpl_viewer import build_visualizer
from .pyvista_viewer import show_pyvista_visualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate a spherical pendulum whose pivot moves in a circle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mass", type=float, default=PendulumParams.mass, help="bob mass m [kg]")
    parser.add_argument("--length", type=float, default=PendulumParams.length, help="pendulum length l [m]")
    parser.add_argument("--gravity", type=float, default=PendulumParams.gravity, help="gravity g [m/s^2]")
    parser.add_argument("--pivot-radius", type=float, default=PendulumParams.pivot_radius, help="pivot circle radius a [m]")
    parser.add_argument("--drive-omega", type=float, default=PendulumParams.drive_omega, help="pivot angular velocity Omega [rad/s]")
    parser.add_argument("--time", type=float, default=14.0, help="Matplotlib duration [s]; PyVista runs until closed")
    parser.add_argument("--dt", type=float, default=0.002, help="integration step [s]")
    parser.add_argument("--theta0", type=float, default=35.0, help="initial theta [deg]")
    parser.add_argument("--phi0", type=float, default=20.0, help="initial phi [deg]")
    parser.add_argument("--theta-dot0", type=float, default=0.0, help="initial theta rate [deg/s]")
    parser.add_argument("--phi-dot0", type=float, default=65.0, help="initial phi rate [deg/s]")
    parser.add_argument("--fps", type=int, default=30, help="animation frames per second")
    parser.add_argument("--playback-speed", type=float, default=1.0, help="simulated seconds per real second")
    parser.add_argument("--trail-seconds", type=float, default=2.5, help="bright trail length [s]")
    parser.add_argument("--visualizer", choices=("matplotlib", "pyvista"), default="matplotlib", help="3-D backend")
    parser.add_argument("--static", action="store_true", help="Matplotlib final frame; PyVista start paused")
    parser.add_argument("--save", type=Path, default=None, help="optional output path")
    parser.add_argument("--no-show", action="store_true", help="do not open a plotting window")
    return parser.parse_args()


def params_from_args(args: argparse.Namespace) -> PendulumParams:
    return PendulumParams(
        mass=args.mass,
        length=args.length,
        gravity=args.gravity,
        pivot_radius=args.pivot_radius,
        drive_omega=args.drive_omega,
    )


def initial_state_from_args(args: argparse.Namespace) -> np.ndarray:
    return np.deg2rad(
        np.array([args.theta0, args.phi0, args.theta_dot0, args.phi_dot0], dtype=float)
    )


def run_matplotlib(args: argparse.Namespace, params: PendulumParams, state: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    t, states = simulate(params, state, args.time, args.dt)
    fig, animation = build_visualizer(
        t, states, params, args.fps, args.playback_speed, args.trail_seconds, not args.static
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
    plt.close(fig) if args.no_show else plt.show()


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if args.playback_speed <= 0.0:
        raise ValueError("playback-speed must be positive")
    params = params_from_args(args)
    state = initial_state_from_args(args)
    if args.visualizer == "pyvista":
        show_pyvista_visualizer(
            state, params, args.dt, args.fps, args.playback_speed,
            args.trail_seconds, args.static, args.save, args.no_show
        )
    else:
        run_matplotlib(args, params, state)
