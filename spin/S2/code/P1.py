import numpy as np
import matplotlib.pyplot as plt

GAMMA = 1.76e11
MU0 = 4.0 * np.pi * 1e-7

MS = 1.0e6
KU = 5.0e5
DEFAULT_ALPHA = 0.1

H_K = 2.0 * KU / (MU0 * MS)
DT = 2.5e-13
SCAN_TIME = 3.0e-9
TRAJ_TIME = 4.0e-9

CRITICAL_FIELD_CACHE = {}


def normalize(vec):
    return vec / np.linalg.norm(vec)


def llg_rhs(m, h_eff, alpha):
    pref = 1.0 / (1.0 + alpha**2)
    mxh = np.cross(m, h_eff)
    return -GAMMA * MU0 * pref * (mxh + alpha * np.cross(m, mxh))


def rk4_step(m, t, dt, alpha, h_func):
    def rhs(m_val, t_val):
        return llg_rhs(m_val, h_func(t_val, m_val), alpha)

    k1 = rhs(m, t)

    m2 = normalize(m + 0.5 * dt * k1)
    k2 = rhs(m2, t + 0.5 * dt)

    m3 = normalize(m + 0.5 * dt * k2)
    k3 = rhs(m3, t + 0.5 * dt)

    m4 = normalize(m + dt * k3)
    k4 = rhs(m4, t + dt)

    return normalize(m + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))


def simulate_reverse_field(h_rev, alpha, total_time=SCAN_TIME, dt=DT, store=False):
    m = normalize(np.array([0.02, 0.0, 1.0]))
    n_steps = int(total_time / dt) + 1

    if store:
        times_ns = np.arange(n_steps) * dt * 1e9
        traj = np.empty((n_steps, 3))

    def h_func(_t, m_val):
        return np.array([0.0, 0.0, H_K * m_val[2] - h_rev])

    min_mz = m[2]
    switch_time_ns = np.nan

    for idx in range(n_steps):
        if store:
            traj[idx] = m

        min_mz = min(min_mz, m[2])
        if np.isnan(switch_time_ns) and m[2] <= 0.0:
            switch_time_ns = idx * dt * 1e9

        if idx < n_steps - 1:
            m = rk4_step(m, idx * dt, dt, alpha, h_func)

    if store:
        return times_ns, traj, min_mz, switch_time_ns
    return m, min_mz, switch_time_ns


def switches_under_field(h_rev, alpha):
    final_m, min_mz, _ = simulate_reverse_field(h_rev, alpha)
    return min_mz < -0.2 and final_m[2] < -0.8


def find_critical_field(alpha, high_guess=1.15 * H_K, n_iter=8):
    key = round(alpha, 8)
    if key in CRITICAL_FIELD_CACHE:
        return CRITICAL_FIELD_CACHE[key]

    low = 0.0
    high = high_guess

    while not switches_under_field(high, alpha):
        high *= 1.12
        if high > 2.0 * H_K:
            return np.nan

    for _ in range(n_iter):
        mid = 0.5 * (low + high)
        if switches_under_field(mid, alpha):
            high = mid
        else:
            low = mid

    CRITICAL_FIELD_CACHE[key] = high
    return high


def plot_field_scan():
    h_sw = find_critical_field(DEFAULT_ALPHA)
    fields = np.linspace(0.75 * H_K, 1.35 * H_K, 18)

    final_mz = np.empty_like(fields)
    switch_times = np.full_like(fields, np.nan)

    for idx, h_rev in enumerate(fields):
        final_m, _, t_sw = simulate_reverse_field(h_rev, DEFAULT_ALPHA)
        final_mz[idx] = final_m[2]
        switch_times[idx] = t_sw

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(MU0 * fields * 1e3, final_mz, "o-", lw=1.4, color="#264653")
    ax1.axvline(MU0 * h_sw * 1e3, color="#e63946", ls="--", lw=1.2,
                label=rf"$\mu_0 H_{{sw}} \approx {MU0 * h_sw:.3f}$ T")
    ax1.set_xlabel(r"Reverse field $\mu_0 H_{\rm rev}$ [mT]", fontsize=12)
    ax1.set_ylabel(r"Final $m_z$", fontsize=12)
    ax1.set_title("Static outcome after applying a reverse field", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1.1, 1.1)
    ax1.legend(fontsize=9)

    mask = np.isfinite(switch_times)
    ax2.plot(MU0 * fields[mask] * 1e3, switch_times[mask], "o-", lw=1.4,
             color="#457b9d")
    ax2.set_xlabel(r"Reverse field $\mu_0 H_{\rm rev}$ [mT]", fontsize=12)
    ax2.set_ylabel("Switching time [ns]", fontsize=12)
    ax2.set_title("Faster reversal once the threshold is exceeded", fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "Part 1 - Field-Driven PMA Switching\n"
        rf"$K_u = {KU/1e5:.1f}\times10^5$ J/m$^3$, $M_s = {MS/1e6:.1f}\times10^6$ A/m, "
        rf"$\alpha = {DEFAULT_ALPHA}$",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig("P1_field_scan.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_trajectory():
    h_sw = find_critical_field(DEFAULT_ALPHA)
    h_demo = 1.05 * h_sw

    times_ns, traj, _, switch_time_ns = simulate_reverse_field(
        h_demo, DEFAULT_ALPHA, total_time=TRAJ_TIME, store=True
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(times_ns, traj[:, 0], color="#e63946", lw=1.1, label=r"$m_x$")
    ax1.plot(times_ns, traj[:, 1], color="#2a9d8f", lw=1.1, label=r"$m_y$")
    ax1.plot(times_ns, traj[:, 2], color="#1d3557", lw=1.4, label=r"$m_z$")
    ax1.axhline(0.0, color="grey", lw=0.6, ls="--")
    if np.isfinite(switch_time_ns):
        ax1.axvline(switch_time_ns, color="#e63946", lw=1.0, ls="--")
    ax1.set_xlabel("Time [ns]", fontsize=12)
    ax1.set_ylabel("Magnetization component", fontsize=12)
    ax1.set_title(
        rf"Component dynamics at $\mu_0 H_{{\rm rev}} = {MU0 * h_demo:.3f}$ T",
        fontsize=11,
    )
    ax1.set_ylim(-1.1, 1.1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    ax2.plot(traj[:, 0], traj[:, 2], color="#264653", lw=1.4)
    ax2.scatter(traj[0, 0], traj[0, 2], color="#2a9d8f", s=40, label="start")
    ax2.scatter(traj[-1, 0], traj[-1, 2], color="#e63946", s=40, label="end")
    ax2.set_xlabel(r"$m_x$", fontsize=12)
    ax2.set_ylabel(r"$m_z$", fontsize=12)
    ax2.set_title("Phase-plane view of the reversal path", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-1.05, 1.05)
    ax2.set_ylim(-1.05, 1.05)
    ax2.legend(fontsize=9)

    fig.suptitle(
        "Part 1 - Field-Driven Switching Trajectory",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig("P1_trajectory.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_damping_dependence():
    alphas = np.array([0.03, 0.07, 0.10, 0.15, 0.22])
    critical_fields = np.array([find_critical_field(alpha) for alpha in alphas])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(alphas, MU0 * critical_fields * 1e3, "o-", lw=1.5, color="#e76f51")
    ax.set_xlabel(r"Damping parameter $\alpha$", fontsize=12)
    ax.set_ylabel(r"Critical reverse field $\mu_0 H_{\rm sw}$ [mT]", fontsize=12)
    ax.set_title("Influence of damping on field-driven switching", fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("P1_damping_dependence.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    print("=" * 54)
    print("  Part 1 - Magnetic-Field-Driven PMA Switching")
    print("=" * 54)

    print("\n[1/3] Reverse-field scan ...")
    plot_field_scan()

    print("[2/3] Switching trajectory ...")
    plot_trajectory()

    print("[3/3] Damping study ...")
    plot_damping_dependence()

    print("\nDone - figures saved.")
