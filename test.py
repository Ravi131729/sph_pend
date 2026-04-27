import numpy as np
import matplotlib.pyplot as plt

E1 = np.array([1.0, 0.0, 0.0])
E2 = np.array([0.0, 1.0, 0.0])
E3 = np.array([0.0, 0.0, 1.0])


def hat(v):
    v = np.asarray(v).reshape(3)
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=float)


def project_to_so3(r):
    u, _, vt = np.linalg.svd(r)
    rn = u @ vt
    if np.linalg.det(rn) < 0:
        u[:, -1] *= -1
        rn = u @ vt
    return rn


def solve_accels(r, omega, m, big_m, inertia, pc, g):
    hpc = hat(pc)
    hom = hat(omega)
    a = np.zeros((5, 5), dtype=float)
    b = np.zeros(5, dtype=float)
    a[0:3, 0:3] = inertia
    a[0:3, 3] = m * (hpc @ (r.T @ E1))
    a[0:3, 4] = m * (hpc @ (r.T @ E2))
    b[0:3] = m * g * (hpc @ (r.T @ E3)) - hom @ inertia @ omega
    a[3, 0:3] = -m * (E1 @ r @ hpc)
    a[3, 3] = big_m + m
    b[3] = -m * (E1 @ r @ hom @ hom @ pc)
    a[4, 0:3] = -m * (E2 @ r @ hpc)
    a[4, 4] = big_m + m
    b[4] = -m * (E2 @ r @ hom @ hom @ pc)
    solution = np.linalg.solve(a, b)
    return solution[0:3], solution[3], solution[4]


def derivative(state, m, big_m, inertia, pc, g):
    r = state[4:13].reshape(3, 3)
    omega = state[13:16]
    omega_dot, x_ddot, y_ddot = solve_accels(r, omega, m, big_m, inertia, pc, g)
    out = np.zeros_like(state)
    out[0:4] = [state[2], state[3], x_ddot, y_ddot]
    out[4:13] = (r @ hat(omega)).reshape(-1)
    out[13:16] = omega_dot
    return out


def rk4_step(state, dt, m, big_m, inertia, pc, g):
    k1 = derivative(state, m, big_m, inertia, pc, g)
    k2 = derivative(state + 0.5 * dt * k1, m, big_m, inertia, pc, g)
    k3 = derivative(state + 0.5 * dt * k2, m, big_m, inertia, pc, g)
    k4 = derivative(state + dt * k3, m, big_m, inertia, pc, g)
    new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    new_state[4:13] = project_to_so3(new_state[4:13].reshape(3, 3)).reshape(-1)
    return new_state


def run():
    m, big_m, g = 0.5, 0.0, 9.81
    inertia = np.diag([0.02, 0.03, 0.04])
    pc = np.array([0.1, 0.0, 0.05])
    state = np.zeros(16)
    state[4:13] = np.eye(3).reshape(-1)
    state[13:16] = np.array([0.0, 0.2, 0.0])
    dt, final_time = 0.001, 5.0
    steps = int(final_time / dt)
    history = np.zeros((steps + 1, 6))
    history[0, 2:5] = state[13:16]

    for k in range(steps):
        state = rk4_step(state, dt, m, big_m, inertia, pc, g)
        history[k + 1, 0] = (k + 1) * dt
        history[k + 1, 1] = state[0]
        history[k + 1, 2:5] = state[13:16]
        history[k + 1, 5] = state[1]
    return history


if __name__ == "__main__":
    hist = run()
    plt.figure()
    plt.plot(hist[:, 0], hist[:, 1], label="x")
    plt.plot(hist[:, 0], hist[:, 5], label="y")
    plt.legend()
    plt.grid(True)
    plt.figure()
    for i, name in enumerate((r"$\Omega_x$", r"$\Omega_y$", r"$\Omega_z$"), start=2):
        plt.plot(hist[:, 0], hist[:, i], label=name)
    plt.legend()
    plt.grid(True)
    plt.show()
