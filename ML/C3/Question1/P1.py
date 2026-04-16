from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parent
_CACHE_ROOT = _SCRIPT_ROOT / ".cache"
_MPLCONFIGDIR = _CACHE_ROOT / "mplconfig"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MU_VALUES = np.arange(-3.0, 4.0, 1.0)
DEFAULT_INITIAL_STATE = np.array([0.5, 0.5], dtype=float)
PART_C_INITIAL_STATES = [
    np.array([0.5, 0.5], dtype=float),
    np.array([1.0, 2.5], dtype=float),
    np.array([4.0, 4.0], dtype=float),
]
DEFAULT_DT = 0.01
DEFAULT_PART_B_TIME = 30.0
DEFAULT_PART_C_TIME = 35.0


def code_root() -> Path:
    return _SCRIPT_ROOT


def ensure_output_dirs() -> tuple[Path, Path]:
    images_dir = code_root() / "images"
    results_dir = code_root() / "results"
    images_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, results_dir


def van_der_pol_rhs(state: np.ndarray, mu: float) -> np.ndarray:
    x, y = state
    return np.array([y, mu * (1.0 - x**2) * y - x], dtype=float)


def rk4_step(state: np.ndarray, mu: float, dt: float) -> np.ndarray:
    k1 = van_der_pol_rhs(state, mu)
    k2 = van_der_pol_rhs(state + 0.5 * dt * k1, mu)
    k3 = van_der_pol_rhs(state + 0.5 * dt * k2, mu)
    k4 = van_der_pol_rhs(state + dt * k3, mu)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_van_der_pol(
    mu: float,
    initial_state: np.ndarray,
    total_time: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    steps = int(round(total_time / dt))
    times = np.linspace(0.0, steps * dt, steps + 1)
    states = np.empty((steps + 1, 2), dtype=float)
    states[0] = initial_state
    for idx in range(steps):
        states[idx + 1] = rk4_step(states[idx], mu, dt)
    return times, states


def linearized_jacobian(mu: float) -> np.ndarray:
    return np.array([[0.0, 1.0], [-1.0, mu]], dtype=float)


def linearized_eigenvalues(mu: float) -> np.ndarray:
    return np.linalg.eigvals(linearized_jacobian(mu))


def classify_local_regime(mu: float) -> str:
    if np.isclose(mu, -2.0):
        return "Repeated stable node"
    if np.isclose(mu, 0.0):
        return "Marginal center"
    if np.isclose(mu, 2.0):
        return "Repeated unstable node"
    if mu < -2.0:
        return "Stable node"
    if -2.0 < mu < 0.0:
        return "Stable focus"
    if 0.0 < mu < 2.0:
        return "Unstable focus"
    return "Unstable node"


def qualitative_prediction(mu: float) -> str:
    if mu < 0.0:
        return "Trajectories contract toward the origin."
    if np.isclose(mu, 0.0):
        return "Linearized motion is a center with bounded oscillations."
    return "The origin repels and the nonlinear dynamics approach a limit cycle."


def tail_statistics(states: np.ndarray) -> dict[str, float]:
    tail = states[int(0.8 * len(states)) :]
    radius = np.linalg.norm(tail, axis=1)
    return {
        "tail_radius_mean": float(radius.mean()),
        "tail_radius_std": float(radius.std()),
        "tail_x_peak_to_peak": float(np.ptp(tail[:, 0])),
        "tail_y_peak_to_peak": float(np.ptp(tail[:, 1])),
    }


def run_mu_sweep(total_time: float, dt: float) -> dict[float, dict[str, object]]:
    sweep: dict[float, dict[str, object]] = {}
    for mu in MU_VALUES:
        times, states = integrate_van_der_pol(mu, DEFAULT_INITIAL_STATE, total_time, dt)
        eigvals = linearized_eigenvalues(mu)
        summary = {
            "mu": float(mu),
            "local_regime": classify_local_regime(mu),
            "qualitative_prediction": qualitative_prediction(mu),
            "linearized_eigenvalues": [
                {"real": float(value.real), "imag": float(value.imag)} for value in eigvals
            ],
            "final_state": {
                "x": float(states[-1, 0]),
                "y": float(states[-1, 1]),
            },
        }
        summary.update(tail_statistics(states))
        sweep[float(mu)] = {
            "times": times,
            "states": states,
            "summary": summary,
        }
    return sweep


def plot_mu_sweep_timeseries(sweep: dict[float, dict[str, object]], out_path: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(14, 15), sharex=False)
    axes = axes.ravel()
    for idx, mu in enumerate(MU_VALUES):
        payload = sweep[float(mu)]
        ax = axes[idx]
        times = payload["times"]
        states = payload["states"]
        summary = payload["summary"]
        ax.plot(times, states[:, 0], color="#1f77b4", lw=1.4, label="x(t)")
        ax.plot(times, states[:, 1], color="#d62728", lw=1.2, label="y(t)")
        ax.set_title(
            rf"$\mu={mu:g}$ | {summary['local_regime']}",
            fontsize=10,
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("State value")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=9, loc="upper right")
    axes[-1].axis("off")
    fig.suptitle(
        "Question 1(b): Van der Pol oscillator time traces from RK4\n"
        r"Initial condition $(x_0, y_0) = (0.5, 0.5)$",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_mu_sweep_phase(sweep: dict[float, dict[str, object]], out_path: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(14, 15), sharex=False, sharey=False)
    axes = axes.ravel()
    for idx, mu in enumerate(MU_VALUES):
        payload = sweep[float(mu)]
        ax = axes[idx]
        states = payload["states"]
        ax.plot(states[:, 0], states[:, 1], color="#264653", lw=1.4)
        ax.scatter(states[0, 0], states[0, 1], color="#2a9d8f", s=35, label="start")
        ax.scatter(states[-1, 0], states[-1, 1], color="#e76f51", s=35, label="end")
        ax.set_title(rf"$\mu={mu:g}$", fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=9, loc="best")
    axes[-1].axis("off")
    fig.suptitle(
        "Question 1(b): Phase-space trajectories from RK4",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_part_c(total_time: float, dt: float) -> list[dict[str, object]]:
    trajectories: list[dict[str, object]] = []
    for initial_state in PART_C_INITIAL_STATES:
        times, states = integrate_van_der_pol(1.0, initial_state, total_time, dt)
        payload = {
            "initial_state": initial_state.copy(),
            "times": times,
            "states": states,
        }
        trajectories.append(payload)
    return trajectories


def plot_part_c_grid(trajectories: list[dict[str, object]], out_path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 13), sharex=False, sharey=False)
    for row, payload in enumerate(trajectories):
        times = payload["times"]
        states = payload["states"]
        initial_state = payload["initial_state"]

        ax_time = axes[row, 0]
        ax_phase = axes[row, 1]

        ax_time.plot(times, states[:, 0], color="#1f77b4", lw=1.4, label="x(t)")
        ax_time.plot(times, states[:, 1], color="#d62728", lw=1.2, label="y(t)")
        ax_time.set_title(
            rf"Time traces for $(x_0, y_0)=({initial_state[0]:g}, {initial_state[1]:g})$",
            fontsize=10,
        )
        ax_time.set_xlabel("Time")
        ax_time.set_ylabel("State value")
        ax_time.grid(True, alpha=0.3)
        if row == 0:
            ax_time.legend(fontsize=9, loc="upper right")

        ax_phase.plot(states[:, 0], states[:, 1], color="#264653", lw=1.4)
        ax_phase.scatter(states[0, 0], states[0, 1], color="#2a9d8f", s=35, label="start")
        ax_phase.scatter(states[-1, 0], states[-1, 1], color="#e76f51", s=35, label="end")
        ax_phase.set_title(
            rf"Phase portrait for $(x_0, y_0)=({initial_state[0]:g}, {initial_state[1]:g})$",
            fontsize=10,
        )
        ax_phase.set_xlabel("x")
        ax_phase.set_ylabel("y")
        ax_phase.grid(True, alpha=0.3)
        if row == 0:
            ax_phase.legend(fontsize=9, loc="best")

    fig.suptitle(
        r"Question 1(c): $\mu=1$ trajectories from three initial conditions",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_part_c_overlay(trajectories: list[dict[str, object]], out_path: Path) -> None:
    colors = ["#1f77b4", "#d62728", "#2a9d8f"]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for color, payload in zip(colors, trajectories):
        states = payload["states"]
        initial_state = payload["initial_state"]
        label = rf"$({initial_state[0]:g}, {initial_state[1]:g})$"
        ax.plot(states[:, 0], states[:, 1], lw=1.4, color=color, label=label)
        ax.scatter(states[0, 0], states[0, 1], color=color, s=28)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"Question 1(c): Convergence toward the $\mu=1$ limit cycle")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Initial state", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_question1_summary(
    sweep: dict[float, dict[str, object]],
    trajectories: list[dict[str, object]],
    results_dir: Path,
) -> None:
    sweep_summary = [sweep[float(mu)]["summary"] for mu in MU_VALUES]
    part_c_summary = []
    for payload in trajectories:
        states = payload["states"]
        initial_state = payload["initial_state"]
        part_c_summary.append(
            {
                "initial_state": [float(initial_state[0]), float(initial_state[1])],
                "final_state": [float(states[-1, 0]), float(states[-1, 1])],
                "tail_x_peak_to_peak": float(np.ptp(states[int(0.8 * len(states)) :, 0])),
                "tail_y_peak_to_peak": float(np.ptp(states[int(0.8 * len(states)) :, 1])),
            }
        )

    payload = {
        "linearization": {
            "jacobian_at_origin": [[0.0, 1.0], [-1.0, "mu"]],
            "eigenvalues_formula": "(mu +/- sqrt(mu^2 - 4)) / 2",
            "stability_statement": "The origin is asymptotically stable for mu < 0, marginal for mu = 0, and unstable for mu > 0.",
        },
        "part_b_mu_sweep": sweep_summary,
        "part_c_mu_equals_1": part_c_summary,
        "discussion": {
            "match_statement": [
                "For mu < 0, the RK4 trajectories decay toward the origin, matching the linearized stability result.",
                "For mu = 0, the motion remains bounded and oscillatory, consistent with a center in the linearized system.",
                "For mu > 0, the origin is unstable as predicted, but the full nonlinear system saturates onto a limit cycle instead of diverging without bound.",
            ],
            "cause_of_difference": "The nonlinear damping term -mu*x^2*y, neglected by the linearization, limits the amplitude and creates a stable limit cycle for positive mu.",
        },
    }
    (results_dir / "question1_summary.json").write_text(json.dumps(payload, indent=2))

    lines = [
        "Question 1 summary",
        "",
        "Linearized system at the origin:",
        "  J(0,0) = [[0, 1], [-1, mu]]",
        "  eigenvalues = (mu +/- sqrt(mu^2 - 4)) / 2",
        "  stable for mu < 0, marginal at mu = 0, unstable for mu > 0",
        "",
        "Part (b) comparison with part (a):",
        "  For mu < 0 the RK4 trajectories collapse to the origin as expected.",
        "  For mu = 0 the trajectories remain bounded and oscillatory.",
        "  For mu > 0 the origin repels trajectories, but the full nonlinear model settles onto a finite-amplitude limit cycle.",
        "  This difference appears because the linearization ignores the nonlinear term -mu*x^2*y.",
        "",
        "Part (c): for mu = 1, all tested initial conditions approach the same attracting limit cycle.",
    ]
    (results_dir / "question1_summary.txt").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve and plot the Van der Pol oscillator tasks from Assignment C3 Question 1."
    )
    parser.add_argument("--dt", type=float, default=DEFAULT_DT, help="Time step for RK4 integration.")
    parser.add_argument(
        "--part-b-time",
        type=float,
        default=DEFAULT_PART_B_TIME,
        help="Simulation horizon for Question 1(b).",
    )
    parser.add_argument(
        "--part-c-time",
        type=float,
        default=DEFAULT_PART_C_TIME,
        help="Simulation horizon for Question 1(c).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_dir, results_dir = ensure_output_dirs()

    print("Running Question 1(b) RK4 sweep for mu = -3, ..., 3")
    sweep = run_mu_sweep(total_time=args.part_b_time, dt=args.dt)
    plot_mu_sweep_timeseries(sweep, images_dir / "Q1_part_b_timeseries.png")
    plot_mu_sweep_phase(sweep, images_dir / "Q1_part_b_phase.png")

    print("Running Question 1(c) RK4 trajectories for mu = 1")
    trajectories = run_part_c(total_time=args.part_c_time, dt=args.dt)
    plot_part_c_grid(trajectories, images_dir / "Q1_part_c_grid.png")
    plot_part_c_overlay(trajectories, images_dir / "Q1_part_c_overlay.png")

    write_question1_summary(sweep, trajectories, results_dir)

    print(f"Saved figures to {images_dir}")
    print(f"Saved summaries to {results_dir}")


if __name__ == "__main__":
    main()
