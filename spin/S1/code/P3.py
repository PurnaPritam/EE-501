
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
    k1 = dmdt(m,              H_eff_func(m),              alpha)
    k2 = dmdt(m + 0.5*dt*k1, H_eff_func(m + 0.5*dt*k1), alpha)
    k3 = dmdt(m + 0.5*dt*k2, H_eff_func(m + 0.5*dt*k2), alpha)
    k4 = dmdt(m +     dt*k3, H_eff_func(m +     dt*k3), alpha)
    m_new = m + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    nrm = np.linalg.norm(m_new)
    if nrm == 0.0:
        return m / np.linalg.norm(m)
    return m_new / nrm


def settle_easyplane(m, H_ext, Hk, alpha, dt, n_steps):
    def H_eff(m_):
        return H_ext + np.array([0.0, 0.0, -Hk * m_[2]])
    for _ in range(n_steps):
        m = rk4_step(m, H_eff, alpha, dt)
    return m


def inplane_hysteresis(K, Ms, alpha, dt, n_settle, phi_deg=0, n_field=120):
    Hk    = 2 * K / (MU0 * Ms)
    H_max = 1.5 * Hk
    phi   = np.radians(phi_deg)
    e_hat = np.array([np.cos(phi), np.sin(phi), 0.0])
    e_perp = np.array([-np.sin(phi), np.cos(phi), 0.0])

    H_up    = np.linspace(-H_max,  H_max, n_field)
    H_down  = np.linspace( H_max, -H_max, n_field)
    H_sweep = np.concatenate([H_up, H_down])

    m = -e_hat.copy() + 0.01 * e_perp
    m /= np.linalg.norm(m)

    m_par = np.empty(len(H_sweep))
    for i, H_mag in enumerate(H_sweep):
        m += 0.01 * e_perp
        m /= np.linalg.norm(m)
        H_ext = H_mag * e_hat
        m = settle_easyplane(m, H_ext, Hk, alpha, dt, n_settle)
        m_par[i] = np.dot(m, e_hat)

    return H_sweep, m_par, Hk


def plot_inplane_hysteresis():
    K, alpha = K0, ALPHA
    Hk = 2 * K / (MU0 * MS)
    dt, n_s = 1e-12, 5000

    angles  = [0, 30, 60, 90]
    colors  = ['#264653', '#2a9d8f', '#e9c46a', '#e76f51']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for phi, clr in zip(angles, colors):
        H, m_par, _ = inplane_hysteresis(K, MS, alpha, dt, n_s, phi_deg=phi)
        lbl = rf"$\varphi = {phi}°$"
        axes[0].plot(H / Hk,        m_par, lw=1.4, color=clr, label=lbl)
        axes[1].plot(H * MU0 * 1e3, m_par, lw=1.4, color=clr, label=lbl)

    for ax in axes:
        ax.set_ylabel(r"$m_\parallel$ (along field)", fontsize=13)
        ax.grid(True, alpha=0.3);  ax.set_ylim(-1.15, 1.15)
        ax.legend(fontsize=10)
        ax.axhline(0, c='grey', lw=0.5, ls='--')
        ax.axvline(0, c='grey', lw=0.5, ls='--')

    axes[0].set_xlabel(r"$H / H_k$", fontsize=13)
    axes[0].set_title("Normalised", fontsize=12)
    axes[0].set_xlim(-1.6, 1.6)

    axes[1].set_xlabel(r"$\mu_0 H$ [mT]", fontsize=13)
    axes[1].set_title("Absolute field", fontsize=12)

    fig.suptitle(
        "Part 3 – Easy-Plane Anisotropy: In-Plane Hysteresis\n"
        rf"$K = {K/1e5:.0f}\times10^5$ J/m³,  $\alpha = {alpha}$  "
        r"(all in-plane directions are equivalent)",
        fontsize=12, y=1.04)
    fig.tight_layout()
    fig.savefig("P3_inplane_anisotropy.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_pma_vs_easyplane():
    K, alpha = K0, ALPHA
    Hk = 2 * K / (MU0 * MS)
    dt, n_s = 1e-12, 5000

    H_max = 1.5 * Hk
    H_up   = np.linspace(-H_max,  H_max, 100)
    H_down = np.linspace( H_max, -H_max, 100)
    Hz_sweep = np.concatenate([H_up, H_down])

    m = np.array([0.0, 0.0, -1.0])
    mz_pma = np.empty(len(Hz_sweep))
    for i, Hz in enumerate(Hz_sweep):
        m += np.array([0.02, 0.02, 0.0])
        m /= np.linalg.norm(m)
        for _ in range(n_s):
            H_eff = np.array([0.0, 0.0, Hz + Hk * m[2]])
            k1 = dmdt(m, H_eff, alpha);  k2 = dmdt(m+0.5*dt*k1, H_eff, alpha)
            k3 = dmdt(m+0.5*dt*k2, H_eff, alpha);  k4 = dmdt(m+dt*k3, H_eff, alpha)
            m = m + (dt/6)*(k1+2*k2+2*k3+k4);  m /= np.linalg.norm(m)
        mz_pma[i] = m[2]

    Hx, mx_ep, _ = inplane_hysteresis(K, MS, alpha, dt, n_s, phi_deg=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(Hz_sweep / Hk, mz_pma, lw=1.5, color='#e63946')
    ax1.set_title("PMA easy-axis (field ∥ z)\n→ large switching field", fontsize=11)
    ax1.set_xlabel(r"$H_z / H_k$", fontsize=13)
    ax1.set_ylabel(r"$m_z$", fontsize=13)

    ax2.plot(Hx / Hk, mx_ep, lw=1.5, color='#457b9d')
    ax2.set_title("Easy-plane, in-plane field (field ∥ x)\n→ NO switching field",
                   fontsize=11)
    ax2.set_xlabel(r"$H_x / H_k$", fontsize=13)
    ax2.set_ylabel(r"$m_x$", fontsize=13)

    for ax in (ax1, ax2):
        ax.set_xlim(-1.6, 1.6);  ax.set_ylim(-1.15, 1.15)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, c='grey', lw=0.5, ls='--')
        ax.axvline(0, c='grey', lw=0.5, ls='--')

    fig.suptitle(
        "Part 3 – PMA vs Easy-Plane Anisotropy Comparison\n"
        rf"$K = {K/1e5:.0f}\times10^5$ J/m³,  $\alpha = {alpha}$",
        fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig("P3_pma_vs_easyplane.png", dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("=" * 50)
    print("  Part 3 – Easy-Plane Anisotropy")
    print("=" * 50)

    print("\n[1/2] In-plane hysteresis at various φ ...")
    plot_inplane_hysteresis()

    print("[2/2] PMA vs easy-plane comparison ...")
    plot_pma_vs_easyplane()

    print("\nDone – figures saved.")
