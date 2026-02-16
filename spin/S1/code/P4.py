
import numpy as np
import matplotlib.pyplot as plt

GAMMA = 1.76e11
MU0   = 4.0 * np.pi * 1e-7

MS    = 1.0e6
KU    = 5.0e5
KS    = 1.0e5
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


def settle(m, H_ext, Hk_plane, Hk_shape, alpha, dt, n_steps):
    def H_eff(m_):
        return H_ext + np.array([Hk_shape * m_[0],
                                 0.0,
                                 -Hk_plane * m_[2]])
    for _ in range(n_steps):
        m = rk4_step(m, H_eff, alpha, dt)
    return m


def hysteresis_phi(Ku, Ks, Ms, alpha, dt, n_settle, phi_deg, n_field=120):
    Hk_plane = 2 * Ku / (MU0 * Ms)
    Hk_shape = 2 * Ks / (MU0 * Ms)
    H_max = 3.0 * Hk_shape
    phi   = np.radians(phi_deg)
    e_hat = np.array([np.cos(phi), np.sin(phi), 0.0])
    e_perp = np.array([-np.sin(phi), np.cos(phi), 0.0])

    H_up   = np.linspace(-H_max,  H_max, n_field)
    H_down = np.linspace( H_max, -H_max, n_field)
    H_sweep = np.concatenate([H_up, H_down])

    m = -e_hat.copy() + 0.01 * e_perp
    m /= np.linalg.norm(m)

    m_par = np.empty(len(H_sweep))
    for i, H_mag in enumerate(H_sweep):
        m += 0.01 * e_perp
        m /= np.linalg.norm(m)
        H_ext = H_mag * e_hat
        m = settle(m, H_ext, Hk_plane, Hk_shape, alpha, dt, n_settle)
        m_par[i] = np.dot(m, e_hat)

    return H_sweep, m_par, Hk_shape


def plot_angle_comparison():
    dt, n_s = 1e-12, 2000
    Hk_shape = 2 * KS / (MU0 * MS)

    angles  = [0, 30, 45, 60, 90]
    colors  = ['#264653', '#2a9d8f', '#e9c46a', '#e76f51', '#e63946']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for phi, clr in zip(angles, colors):
        H, m_par, _ = hysteresis_phi(KU, KS, MS, ALPHA, dt, n_s, phi)
        lbl = rf"$\varphi = {phi}°$"
        ax1.plot(H / Hk_shape,   m_par, lw=1.4, color=clr, label=lbl)
        ax2.plot(H * MU0 * 1e3,  m_par, lw=1.4, color=clr, label=lbl)

    for ax in (ax1, ax2):
        ax.set_ylabel(r"$m_\parallel$ (along field)", fontsize=13)
        ax.grid(True, alpha=0.3);  ax.set_ylim(-1.15, 1.15)
        ax.legend(fontsize=10)
        ax.axhline(0, c='grey', lw=0.5, ls='--')
        ax.axvline(0, c='grey', lw=0.5, ls='--')

    ax1.set_xlabel(r"$H / H_{k,\rm shape}$", fontsize=13)
    ax1.set_title("Normalised by shape anisotropy field", fontsize=12)
    ax1.set_xlim(-3.5, 3.5)

    ax2.set_xlabel(r"$\mu_0 H$ [mT]", fontsize=13)
    ax2.set_title("Absolute field", fontsize=12)

    fig.suptitle(
        "Part 4 – Easy-Plane + Shape Anisotropy: Direction-Dependent Hysteresis\n"
        rf"$K_u = {KU/1e5:.0f}\times10^5$ J/m³,  "
        rf"$K_s = {KS/1e5:.0f}\times10^5$ J/m³,  "
        rf"$\alpha = {ALPHA}$",
        fontsize=12, y=1.04)
    fig.tight_layout()
    fig.savefig("P4_angle_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_grid():
    dt, n_s = 1e-12, 2000
    Hk_shape = 2 * KS / (MU0 * MS)

    key_angles = [0, 45, 90]
    labels = [r"$\varphi=0°$ (shape easy axis, x)",
              r"$\varphi=45°$",
              r"$\varphi=90°$ (shape hard axis, y)"]
    colors = ['#264653', '#e9c46a', '#e63946']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, phi, lbl, clr in zip(axes, key_angles, labels, colors):
        H, m_par, _ = hysteresis_phi(KU, KS, MS, ALPHA, dt, n_s, phi)
        ax.plot(H / Hk_shape, m_par, lw=1.5, color=clr)
        ax.set_title(lbl, fontsize=11)
        ax.set_xlabel(r"$H / H_{k,\rm shape}$", fontsize=12)
        ax.set_xlim(-3.5, 3.5);  ax.set_ylim(-1.15, 1.15)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, c='grey', lw=0.5, ls='--')
        ax.axvline(0, c='grey', lw=0.5, ls='--')

        if phi == 0:
            half = len(H) // 2
            idx  = np.argmin(np.abs(m_par[:half]))
            ax.annotate(rf"$H_{{sw}}/H_{{k,s}}\approx{H[idx]/Hk_shape:+.2f}$",
                        xy=(0.05, 0.88), xycoords='axes fraction',
                        fontsize=9, color='#e63946',
                        bbox=dict(boxstyle='round,pad=0.2', fc='w', alpha=0.8))

    axes[0].set_ylabel(r"$m_\parallel$", fontsize=13)
    fig.suptitle(
        "Part 4 – Individual Loops Along Key Directions\n"
        rf"$K_u = {KU/1e5:.0f}\times10^5$,  $K_s = {KS/1e5:.0f}\times10^5$ J/m³,"
        rf"  $\alpha = {ALPHA}$",
        fontsize=12, y=1.04)
    fig.tight_layout()
    fig.savefig("P4_key_directions.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_shape_strength():
    dt, n_s = 1e-12, 2000

    Ks_vals = [0.5e5, 1.0e5, 2.0e5]
    colors  = ['#f4a261', '#e76f51', '#264653']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for Ks, clr in zip(Ks_vals, colors):
        Hk_s = 2 * Ks / (MU0 * MS)
        H0, m0, _ = hysteresis_phi(KU, Ks, MS, ALPHA, dt, n_s, phi_deg=0)
        lbl = rf"$K_s = {Ks/1e5:.1f}\times10^5$"
        ax1.plot(H0 * MU0 * 1e3, m0, lw=1.4, color=clr, label=lbl)
        H90, m90, _ = hysteresis_phi(KU, Ks, MS, ALPHA, dt, n_s, phi_deg=90)
        ax2.plot(H90 * MU0 * 1e3, m90, lw=1.4, color=clr, label=lbl)

    ax1.set_title(r"Field along x ($\varphi=0°$, shape easy)", fontsize=11)
    ax2.set_title(r"Field along y ($\varphi=90°$, shape hard)", fontsize=11)
    for ax in (ax1, ax2):
        ax.set_xlabel(r"$\mu_0 H$ [mT]", fontsize=13)
        ax.set_ylabel(r"$m_\parallel$", fontsize=13)
        ax.grid(True, alpha=0.3);  ax.set_ylim(-1.15, 1.15)
        ax.legend(fontsize=10)
        ax.axhline(0, c='grey', lw=0.5, ls='--')
        ax.axvline(0, c='grey', lw=0.5, ls='--')

    fig.suptitle(
        "Part 4 – Effect of Shape Anisotropy Strength\n"
        rf"$K_u = {KU/1e5:.0f}\times10^5$ J/m³,  $\alpha = {ALPHA}$",
        fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig("P4_shape_strength.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_coercivity_polar():
    dt, n_s = 1e-12, 2000
    Hk_shape = 2 * KS / (MU0 * MS)

    phis = np.arange(0, 360, 10)
    Hc   = np.empty(len(phis))

    for idx, phi in enumerate(phis):
        H, m_par, _ = hysteresis_phi(KU, KS, MS, ALPHA, dt, n_s, phi, n_field=60)
        half = len(H) // 2
        zero_crossings = np.where(np.diff(np.sign(m_par[:half])))[0]
        if len(zero_crossings) > 0:
            i0 = zero_crossings[0]
            Hc[idx] = abs(H[i0] - m_par[i0]*(H[i0+1]-H[i0])/(m_par[i0+1]-m_par[i0]))
        else:
            Hc[idx] = 0.0

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
    ax.plot(np.radians(phis), Hc / Hk_shape, 'o-', color='#264653', lw=1.2, ms=3)
    ax.set_title(
        "Angular Dependence of Coercive Field\n"
        rf"$K_u = {KU/1e5:.0f}\times10^5$,  $K_s = {KS/1e5:.0f}\times10^5$ J/m³",
        fontsize=11, pad=20)
    ax.set_rlabel_position(135)
    ax.set_ylabel(r"$H_c / H_{k,\rm shape}$", fontsize=11, labelpad=30)
    fig.tight_layout()
    fig.savefig("P4_coercivity_polar.png", dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("=" * 50)
    print("  Part 4 – Easy-Plane + Shape Anisotropy")
    print("=" * 50)

    print("\n[1/4] Angle-dependent hysteresis overlay ...")
    plot_angle_comparison()

    print("[2/4] Key directions (0°, 45°, 90°) ...")
    plot_grid()

    print("[3/4] Shape-strength study ...")
    plot_shape_strength()

    print("[4/4] Coercivity polar plot ...")
    plot_coercivity_polar()

    print("\nDone – figures saved.")
