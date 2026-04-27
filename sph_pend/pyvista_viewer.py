from __future__ import annotations
from pathlib import Path
import numpy as np
from .dynamics import PendulumParams, rk4_step, validate_params
from .geometry import position_from_state

def make_polyline(points: np.ndarray):
    import pyvista as pv
    points = np.asarray(points, dtype=float)
    polyline = pv.PolyData(points)
    if len(points) > 1:
        polyline.lines = np.hstack(([len(points)], np.arange(len(points), dtype=np.int64)))
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
    try:
        import pyvista as pv
    except ImportError as exc:
        raise SystemExit("Install PyVista with: python3 -m pip install pyvista") from exc
    validate_params(params, dt)
    if save is not None and save.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise SystemExit("PyVista saves screenshots only. Use .png/.jpg.")
    state = np.asarray(initial_state, dtype=float).copy()
    reset_state = state.copy()
    sim_time = 0.0
    steps = max(1, int(round(playback_speed / max(fps * dt, 1.0e-12))))
    trail_count = max(2, int(round(trail_seconds * fps / playback_speed)))
    pivot, bob = position_from_state(sim_time, state, params)
    trail_points = np.repeat(bob[None, :], trail_count, axis=0)
    plotter = pv.Plotter(window_size=(1200, 820), title="Driven spherical pendulum", off_screen=no_show)
    plotter.set_background("#f8fafc")
    plotter.add_axes(line_width=2, labels_off=False)
    circle = np.linspace(0.0, 2.0 * np.pi, 240)
    pivot_circle = np.column_stack((params.pivot_radius * np.cos(circle), params.pivot_radius * np.sin(circle), 0 * circle))
    plotter.add_mesh(make_polyline(pivot_circle), color="#6b7280", line_width=2)
    rod_mesh = make_polyline(np.vstack((pivot, bob)))
    trail_mesh = make_polyline(trail_points)
    pivot_mesh = pv.Sphere(radius=max(0.025 * params.length, 0.012))
    bob_mesh = pv.Sphere(radius=max(0.055 * params.length, 0.025))
    pivot_base, bob_base = pivot_mesh.points.copy(), bob_mesh.points.copy()
    plotter.add_mesh(rod_mesh, color="#111827", line_width=7)
    plotter.add_mesh(trail_mesh, color="#0f766e", line_width=4)
    plotter.add_mesh(pivot_mesh, color="#e76f51", smooth_shading=True)
    plotter.add_mesh(bob_mesh, color="#264653", smooth_shading=True)
    radius = max(params.pivot_radius + params.length, params.length, 0.5)
    plotter.show_bounds(bounds=(-radius, radius, -radius, radius, -radius, radius), grid="front", location="outer")
    plotter.camera_position = [(2.3 * radius, -2.6 * radius, 1.45 * radius), (0, 0, 0), (0, 0, 1)]
    playback = {"playing": not start_paused}
    def update_geometry(clear_trail: bool = False) -> None:
        nonlocal trail_points
        pivot_now, bob_now = position_from_state(sim_time, state, params)
        trail_points = np.repeat(bob_now[None, :], trail_count, axis=0) if clear_trail else np.roll(trail_points, -1, axis=0)
        trail_points[-1] = bob_now
        rod_mesh.points = np.vstack((pivot_now, bob_now))
        trail_mesh.points = trail_points
        pivot_mesh.points = pivot_base + pivot_now
        bob_mesh.points = bob_base + bob_now
        controls = "space: play/pause  n: step  c: clear trail  r: reset"
        controls += "  s: screenshot" if save is not None else ""
        plotter.add_text(f"t = {sim_time:.2f} s\npivot radius = {params.pivot_radius:.3g} m\n{controls}", name="state_text", position="upper_left", font_size=11)
        plotter.render()
    def advance() -> None:
        nonlocal sim_time, state
        for _ in range(steps):
            state = rk4_step(sim_time, state, dt, params)
            sim_time += dt
        update_geometry()
    def reset() -> None:
        nonlocal sim_time, state
        playback["playing"] = False
        sim_time, state = 0.0, reset_state.copy()
        update_geometry(clear_trail=True)
    def timer(_step=None):
        if playback["playing"]:
            advance()
    def screenshot():
        if save is not None:
            plotter.screenshot(str(save))
    update_geometry(clear_trail=True)
    plotter.add_key_event("space", lambda: playback.update(playing=not playback["playing"]))
    plotter.add_key_event("n", lambda: (playback.update(playing=False), advance()))
    plotter.add_key_event("c", lambda: update_geometry(clear_trail=True))
    plotter.add_key_event("r", reset)
    if save is not None:
        plotter.add_key_event("s", screenshot)
    plotter.add_timer_event(max_steps=2_147_483_647, duration=int(round(1000.0 / fps)), callback=timer)
    plotter.close() if no_show else plotter.show()
