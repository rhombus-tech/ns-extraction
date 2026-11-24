"""Controlled-flow exponent run: does the true decay-rate exponent drift from ~0.47 toward 1/3 as kappa decreases?

EXPLORATORY (not preregistered). Convention fixed BEFORE running:
  - flow: compact axisymmetric Gaussian vortex, psi = -0.18 exp(-((rho-1)^2 + z^2)/(2*0.3^2)); orbit frequency ~2 at the centre, monotone
    (checked below), so non-degenerate shear; window rho in [0.2,1.8], z in [-0.8,0.8], absorbing edges (same operator as true_rates);
  - closed region: psi < 0.15 * psi_min; mode = slowest oscillatory eigenmode with >= 90% of |v|^2 in the region and |Im| >= 0.3*min(omega);
  - each kappa is solved on two grids (spacings h, 0.75 h); it is CONVERGED iff the two rates agree within 10%; only converged kappa enter the
    exponent fit; unconverged ones are reported and excluded.
"""
import sys
import time
import numpy as np
import orbits as ob
import true_operator as to

PSI0, A = 0.18, 0.3
KAPPAS = ((1e-3, 0.008), (3e-4, 0.006), (1e-4, 0.0045), (5e-5, 0.0035))
RHO_LO, RHO_HI, Z_HALF = 0.2, 1.8, 0.8


def build(h):
    nr = int(round((RHO_HI - RHO_LO) / h))
    nz = int(round(2 * Z_HALF / h))
    dr = (RHO_HI - RHO_LO) / nr
    dz = 2 * Z_HALF / nz
    rho = RHO_LO + (np.arange(nr) + 0.5) * dr
    z = -Z_HALF + (np.arange(nz) + 0.5) * dz
    R, Z = np.meshgrid(rho, z, indexing="ij")
    psi = -PSI0 * np.exp(-(((R - 1.0) ** 2) + Z**2) / (2 * A**2))
    region = psi < 0.15 * psi.min()
    return psi, rho, R, dr, dz, region


def mode_rate(kappa, h):
    psi, rho, R, dr, dz, region = build(h)
    lev, om = ob.orbit_frequency(psi, R, dr, dz, region, nlevels=40)
    L, _ = to.full_window_operator(psi, rho, dr, dz, kappa)
    t0 = time.time()
    rate, ev, frac = to.atmosphere_modes(L, region, (om.min(), om.max()), im_min=0.3 * om.min(), n_shifts=5, real_guess=0.05)
    return rate, ev, frac, psi.size, time.time() - t0


def main():
    psi, rho, R, dr, dz, region = build(0.006)
    lev, om = ob.orbit_frequency(psi, R, dr, dz, region, nlevels=60)
    u, ws, sh, noise = ob.smooth_shear(lev, om)
    print(f"compact vortex: omega from {om.max():.2f} (core) to {om.min():.2f} (edge of region); shear range [{sh.min():.2f},{sh.max():.2f}], "
          f"zeros {ob.shear_zero_order(u, sh)}, SNR {np.abs(sh).max()/noise:.0f}", flush=True)
    good = []
    for kappa, h in KAPPAS:
        rates = []
        for hh in (h, 0.75 * h):
            r, ev, frac, n, dt = mode_rate(kappa, hh)
            rates.append(r)
            txt = "none" if ev is None else f"{r:.4f} (loc {frac:.3f}, Im {ev.imag:+.3f})"
            print(f"kappa={kappa:.1e} h={hh:.4f} ({n} cells, {dt:.0f}s): {txt}", flush=True)
        ok = all(np.isfinite(x) for x in rates) and abs(rates[0] - rates[1]) / rates[1] <= 0.10
        print(f"   -> {'CONVERGED' if ok else 'UNCONVERGED'}", flush=True)
        if ok:
            good.append((kappa, rates[1]))
    print("\nconverged (kappa, rate):", [(f"{k:.1e}", round(r, 4)) for k, r in good])
    for (k1, r1), (k2, r2) in zip(good[:-1], good[1:]):
        print(f"  local exponent {k1:.1e} -> {k2:.1e}: {np.log(r1 / r2) / np.log(k1 / k2):.3f}")
    if len(good) >= 3:
        print(f"overall exponent over {len(good)} converged kappa: {np.polyfit(np.log([k for k, _ in good]), np.log([r for _, r in good]), 1)[0]:.3f} "
              f"(large-Pe, non-degenerate: 0.333)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
