import numpy as np
import matplotlib.pyplot as plt

GAMMA = 1.76e11
MU0   = 4.0 * np.pi * 1e-7

MS    = 1.0e6
K0    = 5.0e5
ALPHA = 0.1


def dmdt(m, H_eff, alpha):
    gp = GAMMA * MU0 / (1.0 + alpha**2)
    mxH = np.cross(m, H_eff)
    return -gp * (mxH + alpha * np.cross(m, mxH))


def rk4_step(m, H_eff, alpha, dt):
    k1 = dmdt(m,              H_eff, alpha)
    k2 = dmdt(m + 0.5*dt*k1, H_eff, alpha)
    k3 = dmdt(m + 0.5*dt*k2, H_eff, alpha)
    k4 = dmdt(m +     dt*k3, H_eff, alpha)
    m_new = m + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return m_new / np.linalg.norm(m_new)


def settle(m, H_ext_z, Hk, alpha, dt, n_steps):
    for _ in range(n_steps):
        H_eff = np.array([0.0, 0.0, H_ext_z + Hk * m[2]])
        m = rk4_step(m, H_eff, alpha, dt)
    return m


def hysteresis(K, Ms, alpha, dt, n_settle, n_field=80):
    Hk    = 2*K / (MU0*Ms)
    H_max = 1.5 * Hk

    H_up   = np.linspace(-H_max,  H_max, n_field)
    H_down = np.linspace( H_max, -H_max, n_field)
    H_sweep = np.concatenate([H_up, H_down])

    m = np.array([0.0, 0.0, -1.0])

    mz = np.empty(len(H_sweep))
    for i, Hz in enumerate(H_sweep):
        m += np.array([0.02, 0.02, 0.0])
        m /= np.linalg.norm(m)
        m = settle(m, Hz, Hk, alpha, dt, n_settle)
        mz[i] = m[2]

    return H_sweep, mz, Hk


def plot_ramp_rates():
    K, alpha = K0, ALPHA
    Hk = 2*K / (MU0*MS)
    dt = 1e-12

    configs = [
        (50,   "Fast  (50 steps)",    '#e63946'),
        (500,  "Medium (500 steps)",  '#457b9d'),
        (5000, "Slow  (5000 steps)",  '#2a9d8f'),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    for n_s, lbl, clr in configs:
        H, mz, _ = hysteresis(K, MS, alpha, dt, n_s)
        ax.plot(H/Hk, mz, lw=1.5, color=clr, label=lbl)

    ax.set_xlabel(r"$H_{\rm ext} / H_k$", fontsize=13)
    ax.set_ylabel(r"$m_z$", fontsize=13)
    ax.set_title("Effect of Ramp Rate on Hysteresis\n"
                  rf"($K={K/1e5:.0f}\times10^5$ J/m³, $\alpha={alpha}$)",
                  fontsize=12)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.15, 1.15)
    ax.axhline(0, c='grey', lw=0.5, ls='--'); ax.axvline(0, c='grey', lw=0.5, ls='--')
    fig.tight_layout(); fig.savefig("hysteresis_ramp_rate.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_anisotropy():
    alpha = ALPHA
    dt, n_s = 1e-12, 5000

    K_vals = [2e5, 5e5, 1e6]
    colors = ['#f4a261', '#e76f51', '#264653']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for K, clr in zip(K_vals, colors):
        H, mz, Hk = hysteresis(K, MS, alpha, dt, n_s)
        lbl = rf"$K={K/1e5:.0f}\times10^5$ J/m³"
        ax1.plot(H * MU0 * 1e3, mz, lw=1.5, color=clr, label=lbl)
        ax2.plot(H / Hk,        mz, lw=1.5, color=clr, label=lbl)

    for ax in (ax1, ax2):
        ax.set_ylabel(r"$m_z$", fontsize=13); ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.15, 1.15); ax.legend(fontsize=10)
        ax.axhline(0, c='grey', lw=0.5, ls='--'); ax.axvline(0, c='grey', lw=0.5, ls='--')
    ax1.set_xlabel(r"$\mu_0 H_{\rm ext}$ [mT]", fontsize=13)
    ax1.set_title("Absolute Field", fontsize=12)
    ax2.set_xlabel(r"$H_{\rm ext}/H_k$", fontsize=13)
    ax2.set_title("Normalised (loops collapse)", fontsize=12)
    ax2.set_xlim(-1.6, 1.6)
    fig.suptitle(f"Effect of Anisotropy Constant  "
                 rf"($\alpha={alpha}$)", fontsize=13, y=1.02)
    fig.tight_layout(); fig.savefig("hysteresis_anisotropy.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_combined():
    alpha = ALPHA;  dt = 1e-12
    K_vals     = [2e5, 5e5, 1e6]
    n_settles  = [50, 500, 5000]

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey=True)
    for i, K in enumerate(K_vals):
        Hk = 2*K/(MU0*MS)
        for j, n_s in enumerate(n_settles):
            H, mz, _ = hysteresis(K, MS, alpha, dt, n_s, n_field=60)
            ax = axes[i, j]
            ax.plot(H/Hk, mz, lw=1.3, color='#264653')
            ax.axhline(0, c='grey', lw=0.4, ls='--')
            ax.axvline(0, c='grey', lw=0.4, ls='--')
            ax.grid(True, alpha=0.25)
            half = len(H)//2
            idx = np.argmin(np.abs(mz[:half]))
            ax.annotate(rf"$H_{{sw}}/H_k\approx{H[idx]/Hk:+.2f}$",
                        xy=(0.05,0.88), xycoords='axes fraction',
                        fontsize=8, color='#e63946',
                        bbox=dict(boxstyle='round,pad=0.2', fc='w', alpha=0.8))
            if i == 0: ax.set_title(f"settle = {n_s} steps", fontsize=11)
            if j == 0: ax.set_ylabel(rf"$m_z$"+"\n"+rf"$K={K/1e5:.0f}\times10^5$", fontsize=10)
            if i == 2: ax.set_xlabel(r"$H_{\rm ext}/H_k$", fontsize=11)
            ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.15, 1.15)

    fig.suptitle("LLG Hysteresis – Comparative Study\n"
                 rf"(Rows: $K$,  Columns: ramp rate,  $\alpha={alpha}$)",
                 fontsize=13, y=1.01)
    fig.tight_layout(); fig.savefig("hysteresis_combined.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_dynamics():
    B_app = 0.1

    m0 = np.array([1.0, 0.0, 0.0])

    t_end = 2.0e-9
    dt    = 1e-13
    N     = int(t_end / dt)
    t_ns  = np.arange(N) * dt * 1e9

    B_eff = np.array([0.0, 0.0, B_app])

    alpha_vals = [0.05, 0.01]
    trajs = []

    for alpha in alpha_vals:
        m = m0.copy()
        traj = np.empty((N, 3))
        gp = GAMMA / (1.0 + alpha**2)
        for i in range(N):
            traj[i] = m
            mxB = np.cross(m, B_eff)
            k1 = -gp * (mxB + alpha * np.cross(m, mxB))

            m1 = m + 0.5*dt*k1
            mxB = np.cross(m1, B_eff)
            k2 = -gp * (mxB + alpha * np.cross(m1, mxB))

            m2 = m + 0.5*dt*k2
            mxB = np.cross(m2, B_eff)
            k3 = -gp * (mxB + alpha * np.cross(m2, mxB))

            m3 = m + dt*k3
            mxB = np.cross(m3, B_eff)
            k4 = -gp * (mxB + alpha * np.cross(m3, mxB))

            m = m + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            m /= np.linalg.norm(m)
        trajs.append(traj)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, traj, alpha in zip([ax1, ax2], trajs, alpha_vals):
        ax.plot(t_ns, traj[:,0], 'r', lw=1.0, label=r'along x ($M_x$)')
        ax.plot(t_ns, traj[:,1], 'g', lw=1.0, label=r'along y ($M_y$)')
        ax.plot(t_ns, traj[:,2], 'b', lw=1.4, label=r'along z ($M_z$)')
        ax.set_xlabel(r"Time (in s)          $\times 10^{-9}$", fontsize=12)
        ax.set_ylabel("Magnetization components", fontsize=12)
        ax.set_title(f"Applied field = 0.1 T, damping ratio = {alpha}",
                     fontsize=11)
        ax.legend(fontsize=9, loc='right')
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlim(0, 2.0)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Numerically solving LLG (precession)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig("reversal_dynamics.png", dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("=" * 50)
    print("  LLG – PMA Hysteresis Simulation")
    print("=" * 50)

    print("\n[1/4] Reversal dynamics ...")
    plot_dynamics()

    print("[2/4] Ramp-rate study ...")
    plot_ramp_rates()

    print("[3/4] Anisotropy study ...")
    plot_anisotropy()

    print("[4/4] Combined grid ...")
    plot_combined()
