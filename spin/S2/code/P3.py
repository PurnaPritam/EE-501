import numpy as np
import matplotlib.pyplot as plt

GAMMA = 1.76e11
MU0 = 4.0 * np.pi * 1e-7
HBAR = 1.054571817e-34
E_CHARGE = 1.602176634e-19

MS = 1.0e6
KU = 5.0e5
DEFAULT_ALPHA = 0.1

ETA_STT = 0.6
T_FREE_STT = 1.5e-9

ETA_SOT = 1.0
T_FREE_SOT = 0.8e-9
SIGMA_SOT = np.array([1.0, 0.0, 0.0])
ASSIST_FIELD_Y = 0.20

H_K = 2.0 * KU / (MU0 * MS)
DT = 2.5e-13

NO_FIELD_CURRENT = 2.0e12
NO_FIELD_PULSE = 1.0e-9
NO_FIELD_TOTAL = 2.5e-9
NO_FIELD_RELAX_BIAS_Z = 1.0e-4

ASSISTED_PULSE = 2.0e-9
ASSISTED_TOTAL = 2.0e-9

FIELD_THRESHOLD_CACHE = {}
STT_THRESHOLD_CACHE = {}
SOT_THRESHOLD_CACHE = {}


def normalize(vec):
    return vec / np.linalg.norm(vec)


def torque_prefactor(current_density, eta, thickness):
    return GAMMA * HBAR * eta * current_density / (2.0 * E_CHARGE * MS * thickness)


def llg_rhs(m, h_eff, alpha, torques=()):
    pref = 1.0 / (1.0 + alpha**2)
    mxh = np.cross(m, h_eff)
    rhs = -GAMMA * MU0 * pref * (mxh + alpha * np.cross(m, mxh))

    for coeff, polarizer in torques:
        mxp = np.cross(m, polarizer)
        rhs += -coeff * pref * (np.cross(m, mxp) - alpha * mxp)

    return rhs


def rk4_step(m, t, dt, alpha, h_func, torque_func):
    def rhs(m_val, t_val):
        return llg_rhs(
            m_val,
            h_func(t_val, m_val),
            alpha,
            torque_func(t_val, m_val),
        )

    k1 = rhs(m, t)

    m2 = normalize(m + 0.5 * dt * k1)
    k2 = rhs(m2, t + 0.5 * dt)

    m3 = normalize(m + 0.5 * dt * k2)
    k3 = rhs(m3, t + 0.5 * dt)

    m4 = normalize(m + dt * k3)
    k4 = rhs(m4, t + dt)

    return normalize(m + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))


def simulate_protocol(m0, alpha, total_time, h_func, torque_func, dt=DT,
                      switch_threshold=0.0, store=False):
    m = normalize(m0.astype(float))
    n_steps = int(total_time / dt) + 1

    if store:
        times_ns = np.arange(n_steps) * dt * 1e9
        traj = np.empty((n_steps, 3))

    min_mz = m[2]
    switch_time_ns = np.nan

    for idx in range(n_steps):
        if store:
            traj[idx] = m

        min_mz = min(min_mz, m[2])
        if np.isnan(switch_time_ns) and m[2] <= switch_threshold:
            switch_time_ns = idx * dt * 1e9

        if idx < n_steps - 1:
            m = rk4_step(m, idx * dt, dt, alpha, h_func, torque_func)

    if store:
        return times_ns, traj, min_mz, switch_time_ns
    return m, min_mz, switch_time_ns


def simulate_sot_pulse(current_density, assist_field_frac, alpha=DEFAULT_ALPHA,
                       pulse_time=ASSISTED_PULSE, total_time=ASSISTED_TOTAL,
                       post_pulse_bias_z_frac=0.0, switch_threshold=0.0,
                       store=False):
    m0 = np.array([1e-3, 0.0, 1.0])
    assist_field_y = assist_field_frac * H_K
    coeff = torque_prefactor(current_density, ETA_SOT, T_FREE_SOT)

    def h_func(t_val, m_val):
        hz_bias = post_pulse_bias_z_frac * H_K if t_val > pulse_time else 0.0
        return np.array([0.0, assist_field_y, H_K * m_val[2] + hz_bias])

    def torque_func(t_val, _m_val):
        if t_val <= pulse_time:
            return ((coeff, SIGMA_SOT),)
        return ()

    return simulate_protocol(
        m0,
        alpha,
        total_time,
        h_func,
        torque_func,
        switch_threshold=switch_threshold,
        store=store,
    )


def simulate_field_switch(h_rev, alpha, total_time=4.0e-9):
    m0 = np.array([0.02, 0.0, 1.0])

    def h_func(_t, m_val):
        return np.array([0.0, 0.0, H_K * m_val[2] - h_rev])

    def torque_func(_t, _m_val):
        return ()

    return simulate_protocol(m0, alpha, total_time, h_func, torque_func)


def simulate_stt_switch(current_density, alpha, total_time=6.0e-9):
    m0 = np.array([0.02, 0.0, 1.0])
    coeff = torque_prefactor(current_density, ETA_STT, T_FREE_STT)
    pinned_layer = np.array([0.0, 0.0, -1.0])

    def h_func(_t, m_val):
        return np.array([0.0, 0.0, H_K * m_val[2]])

    def torque_func(_t, _m_val):
        return ((coeff, pinned_layer),)

    return simulate_protocol(m0, alpha, total_time, h_func, torque_func)


def switches_under_field(h_rev, alpha):
    final_m, min_mz, _ = simulate_field_switch(h_rev, alpha)
    return min_mz < -0.2 and final_m[2] < -0.8


def switches_under_stt(current_density, alpha):
    final_m, min_mz, _ = simulate_stt_switch(current_density, alpha)
    return min_mz < -0.2 and final_m[2] < -0.8


def switches_under_sot(current_density, alpha, assist_field_frac=ASSIST_FIELD_Y):
    final_m, min_mz, _ = simulate_sot_pulse(
        current_density,
        assist_field_frac,
        alpha=alpha,
    )
    return min_mz < -0.2 and final_m[2] < -0.8


def find_critical_field(alpha, high_guess=1.15 * H_K, n_iter=8):
    key = round(alpha, 8)
    if key in FIELD_THRESHOLD_CACHE:
        return FIELD_THRESHOLD_CACHE[key]

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

    FIELD_THRESHOLD_CACHE[key] = high
    return high


def find_critical_stt_current(alpha, high_guess=1.2e12, n_iter=8):
    key = round(alpha, 8)
    if key in STT_THRESHOLD_CACHE:
        return STT_THRESHOLD_CACHE[key]

    low = 0.0
    high = high_guess

    while not switches_under_stt(high, alpha):
        high *= 1.18
        if high > 5.5e12:
            return np.nan

    for _ in range(n_iter):
        mid = 0.5 * (low + high)
        if switches_under_stt(mid, alpha):
            high = mid
        else:
            low = mid

    STT_THRESHOLD_CACHE[key] = high
    return high


def find_critical_sot_current(alpha, assist_field_frac=ASSIST_FIELD_Y, high_guess=1.0e12,
                              n_iter=8):
    key = (round(alpha, 8), round(assist_field_frac, 8))
    if key in SOT_THRESHOLD_CACHE:
        return SOT_THRESHOLD_CACHE[key]

    low = 0.0
    high = high_guess

    while not switches_under_sot(high, alpha, assist_field_frac):
        high *= 1.18
        if high > 4.5e12:
            return np.nan

    for _ in range(n_iter):
        mid = 0.5 * (low + high)
        if switches_under_sot(mid, alpha, assist_field_frac):
            high = mid
        else:
            low = mid

    SOT_THRESHOLD_CACHE[key] = high
    return high


def plot_unassisted_sot():
    times_ns, traj, _, _ = simulate_sot_pulse(
        NO_FIELD_CURRENT,
        assist_field_frac=0.0,
        alpha=DEFAULT_ALPHA,
        pulse_time=NO_FIELD_PULSE,
        total_time=NO_FIELD_TOTAL,
        post_pulse_bias_z_frac=NO_FIELD_RELAX_BIAS_Z,
        store=True,
    )

    pulse_idx = np.searchsorted(times_ns, NO_FIELD_PULSE * 1e9)
    m_inplane = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.axvspan(0.0, NO_FIELD_PULSE * 1e9, color="#f4a261", alpha=0.18,
                label="SOT on")
    ax1.axvline(NO_FIELD_PULSE * 1e9, color="#f4a261", lw=1.0, ls="--")
    ax1.plot(times_ns, m_inplane, color="#2a9d8f", lw=1.4,
             label=r"$m_{\parallel}=\sqrt{m_x^2+m_y^2}$")
    ax1.plot(times_ns, traj[:, 2], color="#1d3557", lw=1.6, label=r"$m_z$")
    ax1.text(
        0.05,
        0.09,
        "SOT keeps the magnet near-plane",
        transform=ax1.transAxes,
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
    )
    ax1.text(
        0.63,
        0.87,
        "After removal,\nPMA restores +z",
        transform=ax1.transAxes,
        fontsize=9,
        ha="left",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
    )
    ax1.set_xlabel("Time [ns]", fontsize=12)
    ax1.set_ylabel("Magnetization component", fontsize=12)
    ax1.set_title("Part 3a - While SOT is on, the magnet stays near-plane", fontsize=11)
    ax1.set_ylim(-1.1, 1.1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    ax2.plot(m_inplane, traj[:, 2], color="#264653", lw=1.4)
    ax2.scatter(m_inplane[0], traj[0, 2], color="#2a9d8f", s=40, label="start")
    ax2.scatter(m_inplane[pulse_idx], traj[pulse_idx, 2], color="#f4a261", s=40,
                label="SOT removed")
    ax2.scatter(m_inplane[-1], traj[-1, 2], color="#e63946", s=40, label="end")
    ax2.annotate(
        "",
        xy=(m_inplane[min(pulse_idx + 200, len(m_inplane) - 1)],
            traj[min(pulse_idx + 200, len(traj) - 1), 2]),
        xytext=(m_inplane[min(pulse_idx + 40, len(m_inplane) - 1)],
                traj[min(pulse_idx + 40, len(traj) - 1), 2]),
        arrowprops=dict(arrowstyle="->", color="#264653", lw=1.0),
    )
    ax2.set_xlabel(r"$m_{\parallel}$", fontsize=12)
    ax2.set_ylabel(r"$m_z$", fontsize=12)
    ax2.set_title("After SOT is removed, the magnet relaxes out of plane", fontsize=11)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-1.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    fig.suptitle(
        "Part 3a - Unassisted Spin-Orbit Torque",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig("P3a_unassisted_sot.png", dpi=180, bbox_inches="tight")
    plt.show()


def plot_polarity_study():
    combos = [
        (+1.0e12, +ASSIST_FIELD_Y, "#e63946", "-", r"$+J,\ +H_y$"),
        (+1.0e12, -ASSIST_FIELD_Y, "#457b9d", "-", r"$+J,\ -H_y$"),
        (-1.0e12, +ASSIST_FIELD_Y, "#2a9d8f", "--", r"$-J,\ +H_y$"),
        (-1.0e12, -ASSIST_FIELD_Y, "#e9c46a", "--", r"$-J,\ -H_y$"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    final_mz = []
    labels = []

    for current_density, assist_field_frac, color, line_style, label in combos:
        times_ns, traj, _, _ = simulate_sot_pulse(
            current_density,
            assist_field_frac,
            alpha=DEFAULT_ALPHA,
            pulse_time=ASSISTED_PULSE,
            total_time=ASSISTED_TOTAL,
            store=True,
        )
        ax1.plot(times_ns, traj[:, 2], lw=1.8, ls=line_style, color=color, label=label)
        final_mz.append(traj[-1, 2])
        labels.append(label)

    ax1.axhline(1.0, color="grey", lw=0.6, ls=":")
    ax1.axhline(-1.0, color="grey", lw=0.6, ls=":")
    ax1.set_xlabel("Time [ns]", fontsize=12)
    ax1.set_ylabel(r"$m_z$", fontsize=12)
    ax1.set_title("Part 3b - $m_z(t)$ for the four current/field combinations", fontsize=11)
    ax1.set_ylim(-1.1, 1.1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc="lower left")

    state_map = np.array([
        [final_mz[0], final_mz[1]],
        [final_mz[2], final_mz[3]],
    ])
    im = ax2.imshow(state_map, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax2.set_xticks([0, 1], labels=[r"$+H_y$", r"$-H_y$"])
    ax2.set_yticks([0, 1], labels=[r"$+J$", r"$-J$"])
    ax2.set_title("Final state selected by the signs of $J$ and $H_y$", fontsize=11)
    for row in range(2):
        for col in range(2):
            value = state_map[row, col]
            state_name = "+z" if value > 0 else "-z"
            ax2.text(
                col,
                row,
                f"{state_name}\n({value:+.2f})",
                ha="center",
                va="center",
                fontsize=10,
                color="black",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
            )
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label(r"Final $m_z$", fontsize=10)

    fig.tight_layout()
    fig.savefig("P3b_polarity_study.png", dpi=180, bbox_inches="tight")
    plt.show()


def plot_assisted_trajectory():
    j_crit = find_critical_sot_current(DEFAULT_ALPHA)
    j_demo = 1.10 * j_crit

    times_ns, traj, _, switch_time_ns = simulate_sot_pulse(
        j_demo,
        assist_field_frac=ASSIST_FIELD_Y,
        alpha=DEFAULT_ALPHA,
        pulse_time=ASSISTED_PULSE,
        total_time=ASSISTED_TOTAL,
        switch_threshold=-0.9,
        store=True,
    )
    switch_idx = None
    if np.isfinite(switch_time_ns):
        switch_idx = min(np.searchsorted(times_ns, switch_time_ns), len(times_ns) - 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(times_ns, traj[:, 0], color="#e63946", lw=1.1, label=r"$m_x$")
    ax1.plot(times_ns, traj[:, 1], color="#2a9d8f", lw=1.1, label=r"$m_y$")
    ax1.plot(times_ns, traj[:, 2], color="#1d3557", lw=1.4, label=r"$m_z$")
    ax1.axhline(-0.9, color="#6c757d", lw=0.9, ls=":", label=r"target $m_z=-0.9$")
    if np.isfinite(switch_time_ns):
        ax1.axvline(switch_time_ns, color="#e63946", lw=1.0, ls="--",
                    label=rf"$t_{{\rm sw}}={switch_time_ns:.2f}$ ns")
    ax1.text(
        0.03,
        0.08,
        rf"$H_y={ASSIST_FIELD_Y:.2f}H_k$, current kept on",
        transform=ax1.transAxes,
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
    )
    ax1.set_xlabel("Time [ns]", fontsize=12)
    ax1.set_ylabel("Magnetization component", fontsize=12)
    ax1.set_title(
        rf"Part 3c - Continuous assisted SOT at $J = {j_demo/1e11:.2f}\times10^{{11}}$ A/m$^2$",
        fontsize=11,
    )
    ax1.set_ylim(-1.1, 1.1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    ax2.plot(traj[:, 0], traj[:, 2], color="#264653", lw=1.4)
    ax2.scatter(traj[0, 0], traj[0, 2], color="#2a9d8f", s=40, label="start")
    if switch_idx is not None:
        ax2.scatter(traj[switch_idx, 0], traj[switch_idx, 2], color="#f4a261", s=42,
                    label=r"$m_z=-0.9$")
    ax2.scatter(traj[-1, 0], traj[-1, 2], color="#e63946", s=40, label="end")
    ax2.set_xlabel(r"$m_x$", fontsize=12)
    ax2.set_ylabel(r"$m_z$", fontsize=12)
    ax2.set_title("Phase path of the continuous SOT-assisted reversal", fontsize=11)
    ax2.set_xlim(-1.05, 1.05)
    ax2.set_ylim(-1.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig("P3c_assisted_trajectory.png", dpi=180, bbox_inches="tight")
    plt.show()


def plot_damping_comparison():
    alphas = np.array([0.05, 0.10, 0.20])
    field_thresholds = np.array([find_critical_field(alpha) for alpha in alphas])
    stt_thresholds = np.array([find_critical_stt_current(alpha) for alpha in alphas])
    sot_thresholds = np.array([find_critical_sot_current(alpha) for alpha in alphas])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(alphas, MU0 * field_thresholds * 1e3, "o-", lw=1.5, color="#e76f51")
    axes[0].set_title("Field-driven", fontsize=11)
    axes[0].set_ylabel(r"Critical threshold", fontsize=12)
    axes[0].set_xlabel(r"$\alpha$", fontsize=12)
    axes[0].set_ylabel(r"$\mu_0 H_{\rm sw}$ [mT]", fontsize=12)
    axes[0].set_xticks(alphas)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(alphas, stt_thresholds / 1e11, "o-", lw=1.5, color="#457b9d")
    axes[1].set_title("Spin-transfer torque", fontsize=11)
    axes[1].set_xlabel(r"$\alpha$", fontsize=12)
    axes[1].set_ylabel(r"$J_c$ [$10^{11}$ A/m$^2$]", fontsize=12)
    axes[1].set_xticks(alphas)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(alphas, sot_thresholds / 1e11, "o-", lw=1.5, color="#2a9d8f")
    axes[2].set_title("Assisted spin-orbit torque", fontsize=11)
    axes[2].set_xlabel(r"$\alpha$", fontsize=12)
    axes[2].set_ylabel(r"$J_c$ [$10^{11}$ A/m$^2$]", fontsize=12)
    axes[2].set_xticks(alphas)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        "Part 3c - Damping Dependence Compared with Parts 1 and 2",
        fontsize=12, y=1.03,
    )
    fig.tight_layout()
    fig.savefig("P3c_damping_comparison.png", dpi=180, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    print("=" * 54)
    print("  Part 3 - Spin-Orbit-Torque Switching")
    print("=" * 54)

    print("\n[1/4] Unassisted SOT pulse (part 3a) ...")
    plot_unassisted_sot()

    print("[2/4] Polarity study with an assist field (part 3b) ...")
    plot_polarity_study()

    print("[3/4] Successful assisted-switching trajectory (part 3c) ...")
    plot_assisted_trajectory()

    print("[4/4] Damping comparison with parts 1 and 2 ...")
    plot_damping_comparison()

    print("\nDone - figures saved.")
