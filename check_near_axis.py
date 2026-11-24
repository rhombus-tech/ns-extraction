"""Core NEAR THE AXIS: does the closed-form core mode (orbits.core_mode) still match the 2D operator for p = 1 and p = 3 when the core is not far from rho = 0?

EXPLORATORY, not preregistered. Convention fixed BEFORE running:
  - flow: psi = -0.18 rho^2/rho0^2 exp(-((rho-rho0)^2 + z^2)/(2*0.3^2)) (vanishes like rho^2 at the axis, as a physical stream function must), rho0 in (1.0, 0.6, 0.4);
    window rho in [0, 1.8], z in [-0.8, 0.8] (touches the axis: uses the reflecting-axis fix of Amendment 2), absorbing outer edges;
  - closed region: psi < 0.15 psi_min; mode = slowest oscillatory eigenmode with >= 90% of |v|^2 in the region and |Im| >= 0.3 min(omega); kappa = 3e-4;
  - each case on two grids (h = 0.006, 0.0045): counted only if they agree within 10%; no pass/fail threshold is set: the closed/true ratio is reported as is.
  - rho0 = 1.0 is a control (the window now touches the axis, unlike run_compact_vortex_exponent).
"""
import sys
import time
import numpy as np
import orbits as ob
import true_operator as to

A, PSI0, KAPPA = 0.3, 0.18, 3e-4
RHO_HI, Z_HALF = 1.8, 0.8


def build(h, rho0):
    nr = int(round(RHO_HI / h))
    nz = int(round(2 * Z_HALF / h))
    dr, dz = RHO_HI / nr, 2 * Z_HALF / nz
    rho = (np.arange(nr) + 0.5) * dr
    z = -Z_HALF + (np.arange(nz) + 0.5) * dz
    R, Z = np.meshgrid(rho, z, indexing="ij")
    psi = -PSI0 * (R / rho0) ** 2 * np.exp(-(((R - rho0) ** 2) + Z**2) / (2 * A**2))
    region = psi < 0.15 * psi.min()
    return psi, rho, R, dr, dz, region


def rate(h, rho0, p):
    psi, rho, R, dr, dz, region = build(h, rho0)
    lev, om = ob.orbit_frequency(psi, R, dr, dz, region, nlevels=40)
    L, _ = to.full_window_operator(psi, rho, dr, dz, KAPPA, p=p)
    r, ev, frac = to.atmosphere_modes(L, region, (om.min(), om.max()), im_min=0.3 * om.min(), n_shifts=5, real_guess=0.05)
    cm = ob.core_mode(psi, R, dr, dz, region, KAPPA)
    return r, (cm["rate"] if cm else None), (cm["ellipticity"] if cm else None), om.min(), om.max()


def main():
    print(f"kappa = {KAPPA}; core radius {A}; closed/true = closed-form core rate / 2D-operator rate", flush=True)
    # default (1.0, 0.6, 0.4); pass centres on the command line, e.g. `check_near_axis.py 0.3 0.2 0.15` (core radius 0.3 exceeds rho0 for rho0 < 0.3; same convention)
    for rho0 in ([float(a) for a in sys.argv[1:]] or (1.0, 0.6, 0.4)):
        for p in (1, 3):
            t0 = time.time()
            rs = []
            for h in (0.006, 0.0045):
                r, closed, ell, wlo, whi = rate(h, rho0, p)
                rs.append(r)
            conv = np.isfinite(rs[0]) and np.isfinite(rs[1]) and abs(rs[0] - rs[1]) / rs[1] <= 0.10
            print(f"rho0={rho0:.1f} p={p}: 2D operator {rs[0]:.4f} / {rs[1]:.4f} ({'converged' if conv else 'UNCONVERGED'}); closed form {closed:.4f}; "
                  f"closed/true {closed / rs[1]:.3f}; ellipticity {ell:.2f}; omega range {wlo:.2f}-{whi:.2f}  [{time.time() - t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
