from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent

GAMMA = 1.76e11
MU0 = 4.0 * np.pi * 1e-7
HBAR = 1.054571817e-34
E_CHARGE = 1.602176634e-19

MS = 1313e3
OE_TO_A_PER_M = 1e3 / (4.0 * np.pi)
HK = 17.9e3 * OE_TO_A_PER_M
HAPP = 1.0e3 * OE_TO_A_PER_M
HK_EFF = HK - MS

ALPHA = 0.005
ETA = 0.33
LAMBDA = 0.38
BETA_FIELDLIKE = 0.0
THICKNESS = 2.0e-9
POLARIZER = np.array([1.0, 0.0, 0.0])

RADIUS_X = 50e-9
RADIUS_Y = 50e-9
CROSS_SECTION = np.pi * RADIUS_X * RADIUS_Y

DT = 5.0e-13
LOW_CURRENT_SETTLE_TIME = 6.0e-8
HIGH_CURRENT_SETTLE_TIME = 2.0e-8
MEASURE_TIME = 1.2e-8
VALIDATION_SETTLE_TIME = 1.0e-8
VALIDATION_TIME = 8.0e-9
SCAN_SAMPLE_STRIDE = 80
VALIDATION_SAMPLE_STRIDE = 80

CURRENTS_MA = np.linspace(1.2, 2.0, 9)
VALIDATION_MA = np.array([1.6, 1.8, 2.0])


def normalize(vec):
    return vec / np.linalg.norm(vec)


def current_density_from_ma(current_ma):
    return current_ma * 1e-3 / CROSS_SECTION


def spherical_to_cartesian(state):
    theta, phi = state
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])


def cartesian_to_spherical(vec):
    m = normalize(vec)
    theta = np.arccos(np.clip(m[2], -1.0, 1.0))
    phi = np.arctan2(m[1], m[0])
    return np.array([theta, phi])


def spin_torque_strength(current_density, theta, phi):
    mx = np.sin(theta) * np.cos(phi)
    return HBAR * ETA * current_density / (
        2.0 * E_CHARGE * MS * THICKNESS * (1.0 + LAMBDA * mx)
    )


def spherical_rhs(state, current_density, beta_fieldlike=BETA_FIELDLIKE):
    theta, phi = state
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_theta_safe = np.sign(sin_theta) * max(abs(sin_theta), 1e-8)

    hz = HAPP + HK_EFF * cos_theta
    hs = spin_torque_strength(current_density, theta, phi)

    term_dl_theta = cos_theta * np.cos(phi) + ALPHA * np.sin(phi)
    term_dl_phi = np.sin(phi) - ALPHA * cos_theta * np.cos(phi)

    dtheta_dt = (
        -ALPHA * GAMMA * MU0 * hz * sin_theta
        - GAMMA * hs * (term_dl_theta + beta_fieldlike * term_dl_phi)
    ) / (1.0 + ALPHA**2)

    dphi_dt = (
        GAMMA * MU0 * hz * sin_theta
        + GAMMA * hs * (term_dl_phi - beta_fieldlike * term_dl_theta)
    ) / ((1.0 + ALPHA**2) * sin_theta_safe)

    return np.array([dtheta_dt, dphi_dt])


def vector_rhs(m, current_density, beta_fieldlike=BETA_FIELDLIKE):
    hz = HAPP + HK_EFF * m[2]
    h_vec = np.array([0.0, 0.0, hz])
    h_s = HBAR * ETA * current_density / (
        2.0 * E_CHARGE * MS * THICKNESS * (1.0 + LAMBDA * m[0])
    )

    torque_field = -GAMMA * MU0 * np.cross(m, h_vec)
    torque_dl = -GAMMA * h_s * np.cross(m, np.cross(POLARIZER, m))
    torque_fl = -GAMMA * beta_fieldlike * h_s * np.cross(m, POLARIZER)
    torque_total = torque_field + torque_dl + torque_fl

    return (torque_total + ALPHA * np.cross(m, torque_total)) / (1.0 + ALPHA**2)


def rk4_step_state(state, current_density):
    k1 = spherical_rhs(state, current_density)
    k2 = spherical_rhs(state + 0.5 * DT * k1, current_density)
    k3 = spherical_rhs(state + 0.5 * DT * k2, current_density)
    k4 = spherical_rhs(state + DT * k3, current_density)

    updated = state + DT * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    updated[0] = np.clip(updated[0], 1e-5, np.pi - 1e-5)
    return updated


def rk4_step_vector(m, current_density):
    k1 = vector_rhs(m, current_density)
    k2 = vector_rhs(normalize(m + 0.5 * DT * k1), current_density)
    k3 = vector_rhs(normalize(m + 0.5 * DT * k2), current_density)
    k4 = vector_rhs(normalize(m + DT * k3), current_density)
    return normalize(m + DT * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0)


def simulate_spherical(current_density, total_time, initial_state, sample_stride):
    n_steps = int(total_time / DT)
    state = initial_state.copy()
    sampled = []

    for idx in range(n_steps):
        if idx % sample_stride == 0:
            sampled.append(state.copy())
        state = rk4_step_state(state, current_density)

    return np.asarray(sampled), state


def simulate_vector(current_density, total_time, initial_m, sample_stride):
    n_steps = int(total_time / DT)
    m = normalize(initial_m.copy())
    sampled = []

    for idx in range(n_steps):
        if idx % sample_stride == 0:
            sampled.append(m.copy())
        m = rk4_step_vector(m, current_density)

    return np.asarray(sampled), m


def advance_spherical(current_density, total_time, initial_state):
    n_steps = int(total_time / DT)
    state = initial_state.copy()

    for _ in range(n_steps):
        state = rk4_step_state(state, current_density)

    return state


def advance_vector(current_density, total_time, initial_m):
    n_steps = int(total_time / DT)
    m = normalize(initial_m.copy())

    for _ in range(n_steps):
        m = rk4_step_vector(m, current_density)

    return m


def extract_metrics_from_spherical(sampled_states, sample_stride):
    theta = sampled_states[:, 0]
    phi = np.unwrap(sampled_states[:, 1])
    half = len(theta) // 2

    theta_mean_deg = np.degrees(np.mean(theta[half:]))
    theta_std_deg = np.degrees(np.std(theta[half:]))
    inst_phi_rate_hz = np.diff(phi[half:]) / (sample_stride * DT * 2.0 * np.pi)
    phi_frequency_hz = np.mean(inst_phi_rate_hz)
    phi_jitter_ghz = np.std(inst_phi_rate_hz) / 1e9
    return phi_frequency_hz, theta_mean_deg, theta_std_deg, phi_jitter_ghz


def extract_metrics_from_vector(sampled_m, sample_stride):
    theta = np.arccos(np.clip(sampled_m[:, 2], -1.0, 1.0))
    phi = np.unwrap(np.arctan2(sampled_m[:, 1], sampled_m[:, 0]))
    half = len(theta) // 2

    theta_mean_deg = np.degrees(np.mean(theta[half:]))
    theta_std_deg = np.degrees(np.std(theta[half:]))
    inst_phi_rate_hz = np.diff(phi[half:]) / (sample_stride * DT * 2.0 * np.pi)
    phi_frequency_hz = np.mean(inst_phi_rate_hz)
    phi_jitter_ghz = np.std(inst_phi_rate_hz) / 1e9
    return phi_frequency_hz, theta_mean_deg, theta_std_deg, phi_jitter_ghz


def initial_state():
    return cartesian_to_spherical(np.array([0.02, 0.0, 1.0]))


def stable_orbit_flag(theta_mean_deg, theta_std_deg, phi_jitter_ghz):
    return theta_mean_deg >= 8.0 and theta_std_deg <= 3.0 and phi_jitter_ghz <= 0.6


def run_frequency_scan():
    state = initial_state()
    rows = []
    steady_states = {}

    for current_ma in CURRENTS_MA:
        current_density = current_density_from_ma(current_ma)
        settle_time = LOW_CURRENT_SETTLE_TIME if current_ma < 1.5 else HIGH_CURRENT_SETTLE_TIME
        state = advance_spherical(current_density, settle_time, state)
        sampled, state = simulate_spherical(
            current_density,
            MEASURE_TIME,
            state,
            SCAN_SAMPLE_STRIDE,
        )
        phi_frequency_hz, theta_mean_deg, theta_std_deg, phi_jitter_ghz = extract_metrics_from_spherical(
            sampled,
            SCAN_SAMPLE_STRIDE,
        )
        stable = stable_orbit_flag(theta_mean_deg, theta_std_deg, phi_jitter_ghz)

        rows.append({
            "current_ma": current_ma,
            "current_density": current_density,
            "phi_frequency_hz": phi_frequency_hz,
            "theta_mean_deg": theta_mean_deg,
            "theta_std_deg": theta_std_deg,
            "phi_jitter_ghz": phi_jitter_ghz,
            "stable": stable,
        })
        steady_states[round(current_ma, 3)] = state.copy()

    return rows, steady_states


def save_scan_csv(rows):
    csv_path = SCRIPT_DIR / "part_e_scan.csv"
    header = (
        "current_ma,current_density_A_per_m2,current_density_1e11_A_per_m2,"
        "phi_frequency_GHz,theta_mean_deg,theta_std_deg,phi_jitter_GHz,stable_orbit\n"
    )

    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write(header)
        for row in rows:
            handle.write(
                f"{row['current_ma']:.3f},"
                f"{row['current_density']:.6e},"
                f"{row['current_density'] / 1e11:.6f},"
                f"{row['phi_frequency_hz'] / 1e9:.6f},"
                f"{row['theta_mean_deg']:.6f},"
                f"{row['theta_std_deg']:.6f},"
                f"{row['phi_jitter_ghz']:.6f},"
                f"{int(row['stable'])}\n"
            )


def run_validation(steady_states):
    validation_rows = []

    for current_ma in VALIDATION_MA:
        key = round(float(current_ma), 3)
        current_density = current_density_from_ma(current_ma)
        state0 = steady_states[key]
        m0 = spherical_to_cartesian(state0)

        state0 = advance_spherical(current_density, VALIDATION_SETTLE_TIME, state0)
        m0 = advance_vector(current_density, VALIDATION_SETTLE_TIME, m0)

        spherical_samples, _ = simulate_spherical(
            current_density,
            VALIDATION_TIME,
            state0,
            VALIDATION_SAMPLE_STRIDE,
        )
        vector_samples, _ = simulate_vector(
            current_density,
            VALIDATION_TIME,
            m0,
            VALIDATION_SAMPLE_STRIDE,
        )

        sph_phi_frequency, sph_theta_mean, _, _ = extract_metrics_from_spherical(
            spherical_samples,
            VALIDATION_SAMPLE_STRIDE,
        )
        vec_phi_frequency, vec_theta_mean, _, _ = extract_metrics_from_vector(
            vector_samples,
            VALIDATION_SAMPLE_STRIDE,
        )

        validation_rows.append({
            "current_ma": current_ma,
            "sph_phi_frequency_ghz": sph_phi_frequency / 1e9,
            "vec_phi_frequency_ghz": vec_phi_frequency / 1e9,
            "phi_frequency_diff_mhz": abs(sph_phi_frequency - vec_phi_frequency) / 1e6,
            "sph_theta_deg": sph_theta_mean,
            "vec_theta_deg": vec_theta_mean,
            "theta_diff_deg": abs(sph_theta_mean - vec_theta_mean),
        })

    csv_path = SCRIPT_DIR / "part_e_validation.csv"
    header = (
        "current_ma,spherical_phi_frequency_GHz,vector_phi_frequency_GHz,phi_frequency_difference_MHz,"
        "spherical_theta_deg,vector_theta_deg,theta_difference_deg\n"
    )
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write(header)
        for row in validation_rows:
            handle.write(
                f"{row['current_ma']:.3f},"
                f"{row['sph_phi_frequency_ghz']:.6f},"
                f"{row['vec_phi_frequency_ghz']:.6f},"
                f"{row['phi_frequency_diff_mhz']:.6f},"
                f"{row['sph_theta_deg']:.6f},"
                f"{row['vec_theta_deg']:.6f},"
                f"{row['theta_diff_deg']:.6f}\n"
            )

    return validation_rows


def plot_phi_frequency_and_theta(rows):
    current_density = np.array([row["current_density"] for row in rows]) / 1e11
    phi_frequency = np.array([row["phi_frequency_hz"] for row in rows]) / 1e9
    theta_mean = np.array([row["theta_mean_deg"] for row in rows])
    stable = np.array([row["stable"] for row in rows], dtype=bool)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(current_density, phi_frequency, "o", color="#94a3b8", ms=6, label="all scan points")
    if np.any(~stable):
        ax1.plot(current_density[~stable], phi_frequency[~stable], "--", color="#94a3b8", lw=1.1)
        ax1.scatter(
            current_density[~stable],
            phi_frequency[~stable],
            facecolors="white",
            edgecolors="#64748b",
            s=58,
            linewidths=1.3,
            label="threshold-sensitive points",
            zorder=3,
        )
    if np.any(stable):
        ax1.plot(current_density[stable], phi_frequency[stable], "o-", color="#264653", lw=1.8,
                 label="stable auto-oscillation")
    ax1.set_xlabel(r"Current density [$10^{11}$ A/m$^2$]", fontsize=12)
    ax1.set_ylabel(r"$\dot{\phi}/2\pi$ [GHz]", fontsize=12)
    ax1.set_title("Oscillation frequency", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc="upper right")

    ax2.plot(current_density, theta_mean, "o", color="#fdc5b6", ms=6, label="all scan points")
    if np.any(~stable):
        ax2.plot(current_density[~stable], theta_mean[~stable], "--", color="#f4a261", lw=1.1)
        ax2.scatter(
            current_density[~stable],
            theta_mean[~stable],
            facecolors="white",
            edgecolors="#e76f51",
            s=58,
            linewidths=1.3,
            label="threshold-sensitive points",
            zorder=3,
        )
    if np.any(stable):
        ax2.plot(current_density[stable], theta_mean[stable], "o-", color="#e76f51", lw=1.8,
                 label="stable auto-oscillation")
    ax2.set_xlabel(r"Current density [$10^{11}$ A/m$^2$]", fontsize=12)
    ax2.set_ylabel(r"Precession angle $\theta$ [deg]", fontsize=12)
    ax2.set_title("Average precession angle", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc="upper left")

    fig.suptitle(
        "Part e - PMA SHNO dynamics in spherical coordinates\n"
        r"$\mu_0 H_{\rm app}=0.10$ T, $\mu_0 H_{k,{\rm eff}}\approx 0.14$ T, "
        rf"$\alpha={ALPHA}$, $\eta={ETA}$, $\lambda={LAMBDA}$",
        fontsize=12, y=1.03,
    )
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / "part_e_phi_theta_scan.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    rows, steady_states = run_frequency_scan()
    save_scan_csv(rows)
    validation_rows = run_validation(steady_states)
    plot_phi_frequency_and_theta(rows)

    max_phi_frequency_diff = max(row["phi_frequency_diff_mhz"] for row in validation_rows)
    max_theta_diff = max(row["theta_diff_deg"] for row in validation_rows)

    print("=" * 64)
    print("Part e - SHNO spherical LLGS simulation")
    print("=" * 64)
    print("Saved figure:")
    print("  - part_e_phi_theta_scan.png")
    print("Saved tables:")
    print("  - part_e_scan.csv")
    print("  - part_e_validation.csv")
    print(
        f"Validation summary: max |Δf_phi| = {max_phi_frequency_diff:.3f} MHz, "
        f"max |Δtheta| = {max_theta_diff:.4f} deg"
    )


if __name__ == "__main__":
    main()
