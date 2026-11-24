"""The 5D (rho^3-weighted, p = 3) operator: validation of the generalised solver, then the core mode on a Gaussian vortex for p = 1 and p = 3.

EXPLORATORY. Convention fixed before running: for each p, kappa = 3e-4 on two grids (h = 0.006, 0.0045); counted only if they agree within 10%.
The companion's operator is Delta_5 (p = 3); the closed form orbits.core_mode is p-independent at leading order (the drift p psi_rho / rho vanishes at the core).
"""
import sys
import numpy as np
from scipy.sparse.linalg import eigs
import true_operator as to
import orbits as ob
from run_compact_vortex_exponent import build


def pure_diffusion(p):
    n = 100
    h = 1.0 / n
    rho = 1.0 + (np.arange(n) + 0.5) * h
    L, _ = to.full_window_operator(np.zeros((n, n)), rho, h, h, 1.0, p=p)  # pads internally; absorbing ghost cells = Dirichlet walls
    lam = -eigs(L, k=1, sigma=0.0, which="LM", return_eigenvectors=False)[0].real
    # separable: z part pi^2; rho part: lowest eigenvalue of -(f'' + p f'/rho), f(1)=f(2)=0
    m = n - 1
    r = 1.0 + h * np.arange(1, n)
    D = np.zeros((m, m))
    for k in range(m):
        D[k, k] = 2 / h**2
        if k > 0:
            D[k, k - 1] = -1 / h**2 + p / (2 * h * r[k])
        if k < m - 1:
            D[k, k + 1] = -1 / h**2 - p / (2 * h * r[k])
    return lam, np.min(np.linalg.eigvals(D).real) + np.pi**2


def rate(h, kappa, p):
    psi, rho, R, dr, dz, region = build(h)
    _, om = ob.orbit_frequency(psi, R, dr, dz, region, nlevels=40)
    L, _ = to.full_window_operator(psi, rho, dr, dz, kappa, p=p)
    r, ev, frac = to.atmosphere_modes(L, region, (om.min(), om.max()), im_min=0.3 * om.min(), n_shifts=5, real_guess=0.05)
    return r, ob.core_mode(psi, R, dr, dz, region, kappa)["rate"]


def main():
    ok = True
    for p in (1, 3):
        lam, exact = pure_diffusion(p)
        good = abs(lam - exact) / exact < 0.01
        ok &= good
        print(f"pure-diffusion Dirichlet block, p={p}: solver {lam:.4f} vs separable {exact:.4f} {'PASS' if good else 'FAIL'}", flush=True)
    print("\nGaussian vortex, core at rho = 1, kappa = 3e-4", flush=True)
    res = {}
    for p in (1, 3):
        rs = []
        for hh in (0.006, 0.0045):
            r, closed = rate(hh, 3e-4, p)
            rs.append(r)
        conv = abs(rs[0] - rs[1]) / rs[1] <= 0.10
        res[p] = (rs[1], closed, conv)
        print(f"  p={p}: 2D operator {rs[0]:.4f} / {rs[1]:.4f} ({'converged' if conv else 'UNCONVERGED'}); closed form {closed:.4f}; closed/true {closed/rs[1]:.3f}", flush=True)
    if res[1][2] and res[3][2]:
        print(f"p=3 / p=1 rate ratio: {res[3][0]/res[1][0]:.3f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
