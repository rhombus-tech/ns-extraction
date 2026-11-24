"""Real 3D unsteady data: decaying Taylor-Green (sim_ns3d.py, 64^3, nu = 0.01, unforced).

Enstrophy budget on the native grid; weighted-identity budgets on a 128^3 spectrally upsampled grid with du/dt recomputed
WITHOUT dealiasing truncation (the solver's own du/dt drops nonlinear modes above 2/3 of the band; the enstrophy budget
cannot see them, but a weighted identity multiplies by f^alpha and can).
alpha = 1 has smooth integrands (closure -> round-off); alpha = 0.5 has |w| kinks at zeros of w, so its quadrature
error converges only as O(h^2) -- reported so the limit is visible.
"""
import glob
import sys
import numpy as np
import fd3d
from sim_ns3d import NS3D
from spectral import upsample

fd3d.PERIODIC_METHOD = "spectral"
M = 128


def exact_dudt(u_up, nu):
    sim = NS3D(u_up.shape[-1], nu)
    sim.dealias = np.ones_like(sim.dealias)
    return np.fft.ifftn(sim.rhs(np.fft.fftn(u_up, axes=(1, 2, 3))), axes=(1, 2, 3)).real


def main():
    files = sorted(glob.glob("data/tg_*.npz"))
    if not files:
        print("no data/tg_*.npz; run: python3 sim_ns3d.py tg 64 0.01 8.0")
        return 1
    print(" t    enstrophy: prod / unsteady / viscous  closure(64^3) | weighted a=1.0: lhs  unsteady  closure(128^3) | a=0.5 closure | R<1  EulerDev")
    worst_smooth = 0.0
    for f in files[::2]:
        d = np.load(f)
        u, dudt, t, nu, h = d["u"].astype(float), d["dudt"].astype(float), float(d["t"]), float(d["nu"]), float(d["h"])
        eb = fd3d.enstrophy_budget(u, (h,) * 3, nu, periodic=True, dudt=dudt)
        uu = upsample(u, M)
        hh = (2 * np.pi / M,) * 3
        dd = exact_dudt(uu, nu)
        w1 = fd3d.weighted_budget(uu, hh, nu, 1.0, periodic=True, dudt=dd)
        w5 = fd3d.weighted_budget(uu, hh, nu, 0.5, periodic=True, dudt=dd)
        rs = fd3d.regularity_summary(u, (h,) * 3, nu, periodic=True)
        worst_smooth = max(worst_smooth, abs(eb["closure_error"]), abs(w1["closure_error"]))
        print(f"{t:4.1f}   {eb['production']:+8.3f} {eb['unsteady']:+8.3f} {eb['viscous']:8.3f}   {eb['closure_error']:+.1e}      |"
              f"        {w1['lhs']:+9.3f} {w1['unsteady']:+9.3f}   {w1['closure_error']:+.1e}       | {w5['closure_error']:+.1e} |"
              f" {rs['R_below_1_fraction']:.2f}  {rs['euler_deviation_score']:.3f}", flush=True)
    print(f"worst closure error, smooth budgets (enstrophy, alpha=1): {worst_smooth:.1e}")
    return 0 if worst_smooth < 1e-6 else 1


if __name__ == "__main__":
    sys.exit(main())
