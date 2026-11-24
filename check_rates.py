"""Calibrate the orbit-reduced rate estimate (rates.py) against the direct 2D axisymmetric solves.

Reads the measured late-time rates from axisym_direct.out (written by check_axisym_direct.py) and compares.
Flows: psi = h(s^2) circular orbits around rho=1, annulus 0.1<s<0.7, p=3 (vorticity Omega), Dirichlet walls.
"""
import re
import sys
import numpy as np
import rates

S_A, S_B, Q0 = 0.1, 0.7, 0.16


def hfun(q, m):
    return q + q * q / 2 if m == 1 else q + (q - Q0) ** 3 + Q0**3  # integral of h' = 1+q  /  1+3(q-q0)^2


def build(m, n=800):
    dr = 2.0 / n
    dz = 2.0 / n
    rho = (np.arange(n) + 0.5) * dr
    z = -1 + (np.arange(n) + 0.5) * dz
    R, Z = np.meshgrid(rho, z, indexing="ij")
    s2 = (R - 1) ** 2 + Z**2
    psi = hfun(s2, m)
    region = (s2 > S_A**2) & (s2 < S_B**2)
    return psi, R, dr, dz, region


def measured():
    out = {}
    for line in open("axisym_direct.out"):
        mm = re.match(r"\s+m=(\d) kappa=([0-9.e+-]+): late-time rate = ([0-9.e+-]+)", line)
        if mm:
            out.setdefault(int(mm.group(1)), []).append((float(mm.group(2)), float(mm.group(3))))
    return out


def main():
    meas = measured()
    if not meas:
        print("no measured rates found in axisym_direct.out; run check_axisym_direct.py first")
        return 1
    ratios = []
    for m in (1, 2):
        psi, R, dr, dz, region = build(m)
        for kappa, lam_direct in meas[m]:
            r = rates.reduced_rate(psi, R, dr, dz, region, kappa, p=3, k_modes=(1, 2), m_grid=4000)
            ratio = r["rate"] / lam_direct
            ratios.append(ratio)
            print(f"m={m} kappa={kappa:.0e}: reduced {r['rate']:.4e} (k={r['mode_k']}, at psi={r['localised_at_psi']:.3f})  "
                  f"direct {lam_direct:.4e}  ratio {ratio:.3f}", flush=True)
    ratios = np.array(ratios)
    lo, hi = ratios.min(), ratios.max()
    good = 0.7 < lo and hi < 1.4
    print(f"ratio range reduced/direct: [{lo:.3f}, {hi:.3f}]  (accuracy claim: within a factor of {max(hi, 1/lo):.2f}) "
          f"{'PASS' if good else 'FAIL'}")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
