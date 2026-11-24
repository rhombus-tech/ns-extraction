"""The closed-form core-mode rate (orbits.core_mode) against the 2D-operator rates on the controlled Gaussian vortex.

True rates come from run_compact_vortex_exponent.py (converged on two grids to <= 0.7%): hard-coded here from compact_vortex.out.
"""
import ast
import re
import sys
import numpy as np
import orbits as ob
from run_compact_vortex_exponent import build


def main():
    true = {}
    for line in open("compact_vortex.out"):
        m = re.match(r"converged \(kappa, rate\): (.*)", line)
        if m:
            true = dict((float(k), float(v)) for k, v in ast.literal_eval(m.group(1)))
    if not true:
        print("compact_vortex.out not found or incomplete")
        return 1
    psi, rho, R, dr, dz, region = build(0.006)
    ok = True
    worst = 0.0
    for kappa, t in true.items():
        r = ob.core_mode(psi, R, dr, dz, region, kappa)
        ratio = r["rate"] / t
        worst = max(worst, abs(ratio - 1))
        print(f"kappa={kappa:.1e}: core_mode {r['rate']:.4f} vs 2D operator {t:.4f}, ratio {ratio:.3f}")
    good = worst < 0.08
    print(f"worst deviation {worst:.3f}; Hessian {r['hessian_eigenvalues']}, core shear {r['core_shear']:.2f} {'PASS' if good else 'FAIL'}")
    # a non-elliptic 'core' must return None: a saddle-like field
    X, Z = np.meshgrid(np.linspace(-1, 1, 200), np.linspace(-1, 1, 200), indexing="ij")
    sad = X**2 - Z**2
    reg = np.ones_like(sad, dtype=bool)
    reg[0] = reg[-1] = False
    reg[:, 0] = reg[:, -1] = False
    none = ob.core_mode(sad, np.ones_like(sad), 0.01, 0.01, reg, 1e-3)
    ok = good and (none is None or none["hessian_eigenvalues"][0] > 0)
    print("ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
