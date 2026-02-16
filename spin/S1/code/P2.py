
import numpy as np
import matplotlib.pyplot as plt

GAMMA = 1.76e11
MU0   = 4.0 * np.pi * 1e-7

MS    = 1.0e6
K0    = 5.0e5
ALPHA = 0.1


def dmdt(m, H_eff, alpha):
    gp  = GAMMA * MU0 / (1.0 + alpha**2)
    mxH = np.cross(m, H_eff)
    return -gp * (mxH + alpha * np.cross(m, mxH))


def rk4_step(m, H_eff_func, alpha, dt):
    """H_eff_func(m) returns H_eff for a given m."""
    k1 = dmdt(m,              H_eff_func(m),              alpha)
    k2 = dmdt(m + 0.5*dt*k1, H_eff_func(m + 0.5*dt*k1), alpha)
    k3 = dmdt(m + 0.5*dt*k2, H_eff_func(m + 0.5*dt*k2), alpha)
    k4 = dmdt(m +     dt*k3, H_eff_func(m +     dt*k3), alpha)
    m_new = m + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    nrm = np.linalg.norm(m_new)
    if nrm == 0.0:
        return m / np.linalg.norm(m)
    return m_new / nrm


def settle_pma_inplane(m, Hx, Hk, alpha, dt, n_steps):
    def H_eff(m_):
        return np.array([Hx, 0.0, Hk * m_[2]])
    for _ in range(n_steps):
        m = rk4_step(m, H_eff, alpha, dt)
    return m


def hard_axis_loop(K, Ms, alpha, dt, n_settle, n_field=80):
    Hk    = 2 * K / (MU0 * Ms)
    H_max = 1.5 * Hk

    H_pos = np.linspace(0, H_max, n_field)
    m = np.array([1e-3, 0.0, 1.0]);  m /= np.linalg.norm(m)
    mx_pos = np.empty(n_field)
    for i, Hx in enumerate(H_pos):
        m = settle_pma_inplane(m, Hx, Hk, alpha, dt, n_settle)
        mx_pos[i] = m[0]

    H_neg = np.linspace(0, -H_max, n_field)
    m = np.array([-1e-3, 0.0, 1.0]);  m /= np.linalg.norm(m)
    mx_neg = np.empty(n_field)
    for i, Hx in enumerate(H_neg):
        m = settle_pma_inplane(m, Hx, Hk, alpha, dt, n_settle)
        mx_neg[i] = m[0]

    H_full  = np.concatenate([H_neg[::-1], H_pos[1:]])
    mx_full = np.concatenate([mx_neg[::-1], mx_pos[1:]])

    return H_full, mx_full, Hk


def plot_hard_axis():
    K, alpha = K0, ALPHA
    Hk = 2 * K / (MU0 * MS)
    dt, n_s = 1e-12, 2000

    H, mx, _ = hard_axis_loop(K, MS, alpha, dt, n_s)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(H / Hk,        mx, lw=1.5, color='#457b9d')
    ax2.plot(H * MU0 * 1e3, mx, lw=1.5, color='#457b9d')

    for ax in (ax1, ax2):
        ax.set_ylabel(r"$m_x$  (along applied field)", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.15, 1.15)
        ax.axhline(0, c='grey', lw=0.5, ls='--')
        ax.axvline(0, c='grey', lw=0.5, ls='--')

    ax1.set_xlabel(r"$H_x / H_k$", fontsize=13)
    ax1.set_title("Normalised field", fontsize=12)
    ax1.set_xlim(-1.6, 1.6)

    ax2.set_xlabel(r"$\mu_0 H_x$ [mT]", fontsize=13)
    ax2.set_title("Absolute field", fontsize=12)

    fig.suptitle(
        "Part 2 — PMA Hard-Axis Curve  (field applied in-plane)\n"
        rf"$K = {K/1e5:.0f}\times10^5$ J/m³,  $\alpha = {alpha}$",
        fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig("P2_hard_axis_pma.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_comparison():
    K, alpha = K0, ALPHA
    Hk = 2 * K / (MU0 * MS)
    dt, n_s = 1e-12, 5000

    H_max = 1.5 * Hk
    H_up   = np.linspace(-H_max,  H_max, 100)
    H_down = np.linspace( H_max, -H_max, 100)
    H_sweep = np.concatenate([H_up, H_down])

    m = np.array([0.0, 0.0, -1.0])
    mz_easy = np.empty(len(H_sweep))
    for i, Hz in enumerate(H_sweep):
        m += np.array([0.02, 0.02, 0.0])
        m /= np.linalg.norm(m)
        for _ in range(n_s):
            H_eff = np.array([0.0, 0.0, Hz + Hk * m[2]])
            k1 = dmdt(m, H_eff, alpha)
            k2 = dmdt(m + 0.5*dt*k1, H_eff, alpha)
            k3 = dmdt(m + 0.5*dt*k2, H_eff, alpha)
            k4 = dmdt(m + dt*k3,     H_eff, alpha)
            m = m + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            m /= np.linalg.norm(m)
        mz_easy[i] = m[2]

    H_h, mx_hard, _ = hard_axis_loop(K, MS, alpha, dt, 2000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(H_sweep / Hk, mz_easy, lw=1.5, color='#e63946')
    ax1.set_xlabel(r"$H_z / H_k$", fontsize=13)
    ax1.set_ylabel(r"$m_z$", fontsize=13)
    ax1.set_title("Easy-axis loop (field ∥ z)", fontsize=12)

    ax2.plot(H_h / Hk, mx_hard, lw=1.5, color='#457b9d')
    ax2.set_xlabel(r"$H_x / H_k$", fontsize=13)
    ax2.set_ylabel(r"$m_x$", fontsize=13)
    ax2.set_title("Hard-axis curve (field ∥ x)", fontsize=12)

    for ax in (ax1, ax2):
        ax.set_xlim(-1.6, 1.6);  ax.set_ylim(-1.15, 1.15)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, c='grey', lw=0.5, ls='--')
        ax.axvline(0, c='grey', lw=0.5, ls='--')

    fig.suptitle(
        "PMA — Easy-axis vs Hard-axis Response\n"
        rf"$K = {K/1e5:.0f}\times10^5$ J/m³,  $\alpha = {alpha}$",
        fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig("P2_easy_vs_hard.png", dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("=" * 50)
    print("  Part 2 – PMA with In-Plane Field")
    print("=" * 50)

    print("\n[1/2] Hard-axis curve ...")
    plot_hard_axis()

    print("[2/2] Easy-axis vs Hard-axis comparison ...")
    plot_comparison()

    print("\nDone – figures saved.")
