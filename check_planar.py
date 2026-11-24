"""Planar (2D) geometry: rho -> 1, Omega is the out-of-plane vorticity (ocean-eddy style flows)."""
import sys
import numpy as np
import diagnose as dg


def grid(n=500):
    dr = 3.0 / n
    dz = 3.0 / n
    x = (np.arange(n) + 0.5) * dr
    z = -1.5 + (np.arange(n) + 0.5) * dz
    X, Z = np.meshgrid(x, z, indexing="ij")
    return X, Z, dr, dz


def main():
    X, Z, dr, dz = grid()
    ok = True
    psi_rad = -np.exp(-(((X - 1.5) ** 2) + Z**2) / 0.5)
    d = dg.diagnose_axisymmetric(psi_rad, X, dr, dz, geometry="planar", level=-0.05)
    good = d["rel_residual"] < 5e-3
    ok &= good
    print(f"radial Gaussian vortex (planar): ||dOmega||/||Omega|| = {d['rel_residual']:.1e} (expect ~0) -> {d['verdict'][:70]} "
          f"{'PASS' if good else 'FAIL'}")

    psi_ell = -np.exp(-((((X - 1.5) ** 2) / 0.5) + (Z**2) / 0.25))
    d2 = dg.diagnose_axisymmetric(psi_ell, X, dr, dz, geometry="planar", level=-0.05)
    good = d2["rel_residual"] > 0.02
    ok &= good
    print(f"elliptical vortex (planar): ||dOmega||/||Omega|| = {d2['rel_residual']:.3f} (expect clearly nonzero) "
          f"{'PASS' if good else 'FAIL'}")

    d3 = dg.diagnose_axisymmetric(psi_ell, X, dr, dz, geometry="planar", level=-0.05, kappa=1e-4,
                                  experimental_rate=True)
    est = d3.get("rate_estimate")
    good = isinstance(est, dict) and est["range"][0] > 0
    ok &= good
    print(f"planar rate estimate at kappa=1e-4: {est['range'] if isinstance(est, dict) else est} {'PASS' if good else 'FAIL'}")
    print("ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
