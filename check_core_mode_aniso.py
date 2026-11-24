"""Two verification checks, run after the audit.

(1) Observed grid-convergence ORDER of the 2D operator's core-mode rate (Gaussian vortex, 3 grids h, 0.75h, 0.5625h), to test the
    p = 2 assumption used when Richardson-extrapolating.
(2) The anisotropic extension of orbits.core_mode, tested on elliptical Gaussian vortices of ellipticity e = 1, 1.5, 2, 3 in the PLANAR
    limit (axisymmetric operator with the core at rho0 = 50, psi scaled by rho0, so the geometry is planar to O(1/rho0)).
EXPLORATORY. Convention fixed before running: ellipticity cases use kappa = 3e-4 on two grids (h, 0.75 h) and count only if they agree within 10%.
"""
import sys
import numpy as np
import orbits as ob
import true_operator as to

PSI0, A = 0.18, 0.3
RHO0 = 50.0


def vortex(h, ecc, kappa, planar=True, half=0.9):
    nx = int(round(2 * half / h))
    nz = int(round(2 * half * max(1.0, ecc) / h))
    dr = 2 * half / nx
    dz = 2 * half * max(1.0, ecc) / nz
    x = -half + (np.arange(nx) + 0.5) * dr
    z = -half * max(1.0, ecc) + (np.arange(nz) + 0.5) * dz
    X, Z = np.meshgrid(x, z, indexing="ij")
    psi = -PSI0 * np.exp(-(X**2 / (2 * A**2) + Z**2 / (2 * (A * ecc) ** 2)))
    rho = RHO0 + x
    psi_eff = RHO0 * psi
    region = psi < 0.15 * psi.min()
    Rr, _ = np.meshgrid(rho, z, indexing="ij")
    return psi_eff, rho, Rr, dr, dz, region


def rate(h, ecc, kappa):
    psi, rho, R, dr, dz, region = vortex(h, ecc, kappa)
    _, om = ob.orbit_frequency(psi, R, dr, dz, region, nlevels=40)
    L, _ = to.full_window_operator(psi, rho, dr, dz, kappa)
    r, ev, frac = to.atmosphere_modes(L, region, (om.min(), om.max()), im_min=0.3 * om.min(), n_shifts=5, real_guess=0.05)
    return r, ev, frac, ob.core_mode(psi, R, dr, dz, region, kappa)


def main():
    print("(1) observed order of grid convergence, Gaussian vortex (planar limit, e=1)", flush=True)
    for kappa, h in ((1e-3, 0.008), (1e-4, 0.0045)):
        rs = [rate(hh, 1.0, kappa)[0] for hh in (h, 0.75 * h, 0.5625 * h)]
        d1, d2 = rs[0] - rs[1], rs[1] - rs[2]
        p = np.log(abs(d1 / d2)) / np.log(1 / 0.75) if d2 != 0 else float("nan")
        print(f"  kappa={kappa:.0e}: rates {rs[0]:.5f}, {rs[1]:.5f}, {rs[2]:.5f}; observed order p = {p:.2f}; "
              f"Richardson(p_obs) limit {rs[2] - d2 / ((1 / 0.75) ** p - 1):.5f}", flush=True)
    print("\n(2) anisotropy: core_mode vs 2D operator, kappa = 3e-4", flush=True)
    worst = 0.0
    for ecc in (1.0, 1.5, 2.0, 3.0):
        rs, cm = [], None
        for hh in (0.0065, 0.0049):
            r, ev, frac, cm = rate(hh, ecc, 3e-4)
            rs.append((r, frac))
        ok = np.isfinite(rs[0][0]) and np.isfinite(rs[1][0]) and abs(rs[0][0] - rs[1][0]) / rs[1][0] <= 0.10
        t = rs[1][0]
        ratio = cm["rate"] / t if ok else float("nan")
        if ok:
            worst = max(worst, abs(ratio - 1))
        print(f"  e={ecc}: Hessian ({cm['hessian_eigenvalues'][0]:.2f}, {cm['hessian_eigenvalues'][1]:.2f}), core_mode {cm['rate']:.4f}, "
              f"2D operator {rs[0][0]:.4f} / {rs[1][0]:.4f} (loc {rs[1][1]:.2f}), ratio {ratio:.3f}{'' if ok else '  [UNCONVERGED]'}", flush=True)
    print(f"worst deviation over converged ellipticities: {worst:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
