"""Checks for the orbit-decomposition pipeline: Hill's vortex, GS depletion, perturbed ring, shear-zero order."""
import sys
import numpy as np
import orbits as ob


def grid(rho_max, z_max, nr, nz):
    dr = rho_max / nr
    dz = 2 * z_max / nz
    rho = (np.arange(nr) + 0.5) * dr
    z = -z_max + (np.arange(nz) + 0.5) * dz
    R, Z = np.meshgrid(rho, z, indexing="ij")
    return R, Z, dr, dz


def hill_psi(R, Z):
    r2 = R**2 + Z**2
    return -0.75 * R**2 * (1 - r2), r2 < 1.0


def test_hill():
    R, Z, dr, dz = grid(1.2, 1.2, 300, 600)
    psi, inside = hill_psi(R, Z)
    ok = True
    for eps in (0.0, 0.01, 0.02, 0.04):
        phi = R**2 * (1 - (R**2 + Z**2)) ** 2 * Z
        p = psi + eps * phi
        region = ob.closed_region(p, inside, -0.01)
        u_r, u_z = ob.poloidal_velocity(p, R, dr, dz)
        om = ob.azimuthal_vorticity(u_r, u_z, dr, dz) / R
        # drop the outermost region cells (finite-difference stencil touches the sphere boundary)
        region = region & (R**2 + Z**2 < 0.97**2)
        centers, g, d_om = ob.decompose(p, R, dr, dz, om, region)
        rel = ob.weighted_l2(d_om, R, dr, dz, region) / ob.weighted_l2(om, R, dr, dz, region)
        if eps == 0.0:
            spread = np.std(om[region]) / abs(np.mean(om[region]))
            print(f"Hill: Omega over interior = {np.mean(om[region]):+.4f} (|Omega| should be 15/2 = 7.5), "
                  f"relative spread {spread:.2e}")
            sh = ob.shear_profile(centers, g)
            print(f"      max |g'(psi)| = {np.max(np.abs(sh)):.2e} -> shear zero of infinite order, beta -> 2 (no enhanced mixing)")
            good = abs(abs(np.mean(om[region])) - 7.5) < 0.02 and rel < 5e-3
            ok &= good
        print(f"  eps={eps:5.3f}: ||dOmega||/||Omega|| = {rel:.3e}")
        if eps == 0.01:
            r1 = rel
        if eps == 0.02:
            r2 = rel
        if eps == 0.04:
            r4 = rel
    lin1, lin2 = r2 / r1, r4 / r2
    good = abs(lin1 - 2) < 0.15 and abs(lin2 - 2) < 0.15
    ok &= good
    print(f"  linear scaling in eps: ratios {lin1:.3f}, {lin2:.3f} (expect 2) {'PASS' if good else 'FAIL'}")
    return ok


def test_gs_depletion():
    R, Z, dr, dz = grid(1.2, 1.2, 300, 600)
    psi, inside = hill_psi(R, Z)
    region = ob.closed_region(psi, inside, -0.01)
    u_r, _ = ob.poloidal_velocity(psi, R, dr, dz)
    ok = True
    for name, g in (("g=1", lambda s: np.ones_like(s)),
                    ("g=sin(20 psi)", lambda s: np.sin(20 * s)),
                    ("g=30 psi^2 + psi", lambda s: 30 * s**2 + s)):
        val = np.sum(np.where(region, R**2 * g(psi) ** 2 * u_r, 0.0)) * dr * dz
        scale = np.sum(np.where(region, R**2 * g(psi) ** 2 * np.abs(u_r), 0.0)) * dr * dz
        good = abs(val) / scale < 1e-3
        ok &= good
        print(f"GS depletion {name:18s}: integral/scale = {abs(val)/scale:.2e} {'PASS' if good else 'FAIL'}")
    return ok


def test_shear_order():
    R, Z, dr, dz = grid(2.0, 1.0, 400, 400)
    psi = -np.exp(-(((R - 1.0) ** 2) + Z**2) / 0.08)
    region = ob.closed_region(psi, np.ones_like(psi, dtype=bool), -0.02)
    ok = True
    for k in (1, 2, 3):
        g_true = lambda s: (s + 0.4) ** k
        om = g_true(psi) + 0.05 * Z * np.exp(-(((R - 1.0) ** 2) + Z**2) / 0.08)
        centers, g, d_om = ob.decompose(psi, R, dr, dz, om, region, nbins=120)
        err = np.max(np.abs(g - g_true(centers)))
        sh = ob.shear_profile(centers, g)
        zeros = ob.shear_zero_order(centers, sh)
        if k == 1:
            good = len(zeros) == 0 and err < 5e-3
            print(f"shear order k=1 (non-degenerate): zeros found = {len(zeros)}, g recovery err {err:.1e} "
                  f"{'PASS' if good else 'FAIL'}")
        else:
            good = len(zeros) == 1 and abs(zeros[0][0] + 0.4) < 0.01 and abs(zeros[0][1] - k) < 0.35 and err < 5e-3
            z0 = zeros[0] if zeros else (float('nan'), float('nan'))
            print(f"shear order k={k}: psi0 = {z0[0]:+.3f} (true -0.400), m = {z0[1]:.2f} (true {k}), "
                  f"beta = {ob.beta_from_order(z0[1]):.3f} (true {ob.beta_from_order(k):.3f}), "
                  f"g recovery err {err:.1e} {'PASS' if good else 'FAIL'}")
        ok &= good
    return ok


def test_diagnose_orbit_frequency():
    """diagnose must take its shear from the ORBIT FREQUENCY omega(psi)=2pi/T, not from the vorticity profile g(psi)."""
    import diagnose as dg
    R, Z, dr, dz = grid(2.0, 1.0, 500, 500)
    Z = Z - 0.0
    s2 = (R - 1) ** 2 + Z**2
    mask = (s2 > 0.1**2) & (s2 < 0.7**2)
    q0 = 0.16
    bump = 0.05 * Z * np.exp(-s2 / 0.1)
    ok = True
    psi1 = s2 + s2**2 / 2
    d1 = dg.diagnose_axisymmetric(psi1, R, dr, dz, omega_phi=R * (1.0 + bump), mask=mask, level=1e9)
    good = not d1["zeros"] and d1.get("order_m") == 1.0
    ok &= good
    print(f"diagnose, omega=2(1+q) (non-degenerate): zeros {d1['zeros']}, m={d1.get('order_m')} {'PASS' if good else 'FAIL'}")
    psi2 = s2 + (s2 - q0) ** 3 + q0**3
    d2 = dg.diagnose_axisymmetric(psi2, R, dr, dz, omega_phi=R * (1.0 + bump), mask=mask, level=1e9)
    good = len(d2["zeros"]) == 1 and abs(d2["zeros"][0][1] - 2) < 0.2 and abs(d2["zeros"][0][0] - 0.1641) < 0.005
    ok &= good
    print(f"diagnose, omega=2(1+3(q-q0)^2) (simple shear zero): zeros {[(round(a, 4), round(b, 2)) for a, b in d2['zeros']]} {'PASS' if good else 'FAIL'}")
    # same Omega profile g(psi) must NOT create a zero when the flow is non-degenerate (the old bug keyed on g, not omega)
    om_g = (psi1 + 0.2) ** 2 + bump
    d3 = dg.diagnose_axisymmetric(psi1, R, dr, dz, omega_phi=R * om_g, mask=mask, level=1e9)
    good = not d3["zeros"]
    ok &= good
    print(f"diagnose ignores zeros of the vorticity profile g(psi): zeros {d3['zeros']} {'PASS' if good else 'FAIL'}")
    return ok


def main():
    results = [test_hill(), test_gs_depletion(), test_shear_order(), test_diagnose_orbit_frequency()]
    print("ALL PASS" if all(results) else "FAILURES")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
