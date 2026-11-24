"""Verification of the orbit-averaged-diffusion core formula  Re lambda = sqrt(2 kappa |omega'| (h1 + h2)).

PART A is RETROSPECTIVE (the formula was found by explaining the planar elliptical data, so that agreement is not independent evidence).
PART B is OUT OF SAMPLE and its criterion is fixed here before running:
  B1  planar elliptical Gaussian vortices, ellipticity e in {1.25, 2.5, 3}, kappa = 1e-4 (not used in the derivation), each solved on two grids
      (h, 0.75 h); counts only if the two agree within 10%.  PASS iff |formula / 2D operator - 1| <= 0.06 for every counted case.
  B2  the real ring (nu = 0.01, frozen t = 2.5) at kappa = 6e-4, lower than any kappa used before, on refinements 2.5 and 3.0; counts only if the two
      agree within 10%.  PASS iff |formula / 2D operator - 1| <= 0.06.
"""
import sys
import numpy as np
import io_fields as io
import orbits as ob
import run_ring_case as rc
import scalar_experiment as se
import true_rates as tr
import check_core_mode_aniso as ca
from run_compact_vortex_exponent import build


def part_a():
    print("PART A (retrospective): new formula / 2D operator on all earlier data", flush=True)
    psi, rho, R, dr, dz, region = build(0.006)
    for k, t in ((1e-3, 0.2821), (3e-4, 0.1575), (1e-4, 0.0915), (5e-5, 0.0648)):
        cm = ob.core_mode(psi, R, dr, dz, region, k)
        print(f"  Gaussian vortex kappa={k:.0e}: new {cm['rate']:.4f} / true {t:.4f} = {cm['rate']/t:.3f} (old {cm['rate_cartesian_oscillator']/t:.3f})")
    ref = np.load("stage1_frozen.npz")
    u5, _, _, _, h = rc.load(5)
    a5, _ = se.extraction(u5, h, float(ref["zc"]))
    p1 = io.stream_function(a5["u_rho"], a5["u_z"], a5["rho"], a5["dr"], a5["dz"])[0]
    pc1 = io.to_comoving(p1, a5["rho"], float(ref["speed"]))
    core = np.unravel_index(np.argmax(np.abs(a5["omega_phi"])), a5["omega_phi"].shape)
    lab = (-1.0 if pc1[core] > 0 else 1.0) * pc1
    R1, _ = np.meshgrid(a5["rho"], a5["z"], indexing="ij")
    reg1 = ob.closed_region(lab, np.ones_like(lab, dtype=bool), 0.02 * lab.min())
    for k, t in ((1.25e-3, 0.2418), (1.7e-3, 0.2800), (2.5e-3, 0.3356), (3.3e-3, 0.3809), (5e-3, 0.4582)):
        cm = ob.core_mode(lab, R1, a5["dr"], a5["dz"], reg1, k)
        print(f"  ring 1 kappa={k:.2e}: new {cm['rate']:.4f} / true {t:.4f} = {cm['rate']/t:.3f} (old {cm['rate_cartesian_oscillator']/t:.3f})")
    return pc1, a5


def part_b1():
    print("\nPART B1 (out of sample): planar elliptical vortices, kappa = 1e-4", flush=True)
    ok_all = True
    for ecc in (1.25, 2.5, 3.0):
        rs, cm = [], None
        for hh in (0.0065, 0.0049):
            r, ev, frac, cm = ca.rate(hh, ecc, 1e-4)
            rs.append(r)
        conv = np.isfinite(rs[0]) and np.isfinite(rs[1]) and abs(rs[0] - rs[1]) / rs[1] <= 0.10
        ratio = cm["rate"] / rs[1] if conv else float("nan")
        good = conv and abs(ratio - 1) <= 0.06
        ok_all &= (good or not conv)
        print(f"  e={ecc}: new {cm['rate']:.4f}, old {cm['rate_cartesian_oscillator']:.4f}; 2D operator {rs[0]:.4f} / {rs[1]:.4f}; "
              f"new/true {ratio:.3f} (old/true {cm['rate_cartesian_oscillator']/rs[1]:.3f}) -> {'PASS' if good else ('FAIL' if conv else 'UNCONVERGED, not counted')}", flush=True)
    return ok_all


def part_b2(pc1, a5):
    print("\nPART B2 (out of sample): real ring 1, kappa = 6e-4", flush=True)
    kap = 6e-4
    core = np.unravel_index(np.argmax(np.abs(a5["omega_phi"])), a5["omega_phi"].shape)
    lab = (-1.0 if pc1[core] > 0 else 1.0) * pc1
    R1, _ = np.meshgrid(a5["rho"], a5["z"], indexing="ij")
    reg1 = ob.closed_region(lab, np.ones_like(lab, dtype=bool), 0.02 * lab.min())
    cm = ob.core_mode(lab, R1, a5["dr"], a5["dz"], reg1, kap)
    rs = []
    for k in (2.5, 3.0):
        r = tr.slowest_mode(pc1, a5["omega_phi"], a5["rho"], a5["z"], a5["dr"], a5["dz"], kap, refine=k)
        rs.append(r["rate"])
        print(f"  refine {k}: {r['cells']} cells, rate {r['rate']}, localisation {r.get('localisation')}", flush=True)
    conv = all(x is not None for x in rs) and abs(rs[0] - rs[1]) / rs[1] <= 0.10
    ratio = cm["rate"] / rs[1] if conv else float("nan")
    good = conv and abs(ratio - 1) <= 0.06
    print(f"  new {cm['rate']:.4f} (old {cm['rate_cartesian_oscillator']:.4f}); 2D operator {rs}; new/true {ratio:.3f} -> "
          f"{'PASS' if good else ('FAIL' if conv else 'UNCONVERGED, not counted')}")
    return good or not conv


def main():
    pc1, a5 = part_a()
    b1 = part_b1()
    b2 = part_b2(pc1, a5)
    print(f"\nB1 {'PASS' if b1 else 'FAIL'}; B2 {'PASS' if b2 else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
