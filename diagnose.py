"""Practitioner entry points: run the extracted diagnostics on a velocity field.

    python3 diagnose.py demo        # runs both diagnostics on built-in test flows

diagnose_axisymmetric(psi, rho, dr, dz, ...)  -> orbit decomposition, shear zeros, rate exponent
diagnose_periodic_3d(u, nu)                   -> Euler deviation score, regularity-indicator map

What these do NOT give you: a mixing-rate number. The exponent is known; the prefactor is not.
"""
import sys
import numpy as np
import orbits as ob
import rates
from spectral import Grid

AXI_WARNINGS = [
    "Axisymmetric, no swirl, closed streamlines only; work in the frame where streamlines close (e.g. ring co-moving frame).",
    "delta-Omega is the vorticity-derived quantity Omega = omega_phi/rho. For a passive scalar the same exponent is expected but not proved.",
    "The reported exponent is the scaling of the decay rate with diffusivity (kappa^p); the prefactor is NOT determined.",
    "Treats the velocity as frozen (linearisation): valid for small delta-Omega and mixing faster than mean-flow evolution.",
]
PERIODIC_WARNINGS = [
    "Identities proved for steady flow on R^3; on a periodic box there are no boundary terms but forcing and unsteadiness add terms.",
    "In a forced flow the zero-sum identity is violated by -int f^a w.curl(F): a nonzero zero-sum residual is not by itself a defect.",
    "int w.S.w = 0 needs steady Euler (inviscid, unforced). Deviation measures departure from that ideal, nothing more.",
]


def diagnose_axisymmetric(psi, rho, dr, dz, omega_phi=None, level=None, mask=None, nbins=ob.DEFAULT_BINS,
                          geometry="axisym", kappa=None, quantity="omega", experimental_rate=False, accurate_rate=False,
                          applicability=None, override=False):
    """geometry 'axisym' (rho,z) or 'planar' (rho -> 1, e.g. 2D ocean eddies); omega_phi is the (out-of-plane) vorticity.

    WITHDRAWN: the rate estimate failed its preregistered test on a simulated vortex ring (PREREGISTERED_STAGE1_E1.md:
    measured/predicted 2.42 at kappa = 3.3e-3). It is computed only if experimental_rate=True, for research use.
    kappa: diffusivity; if given (and experimental_rate=True), an orbit-reduced decay-RATE estimate is added, bracketed by
    Dirichlet / Neumann conditions at the edge of the closed-streamline region. quantity 'omega' (p=3) or 'scalar' (p=1);
    planar uses p=0.
    applicability: the (ok, reasons) tuple returned by io_fields.applicability(extraction, psi_residual). If given and not ok, NO numbers are
    computed (the flow is not axisymmetric / is swirling / has an inconsistent stream function) unless override=True.
    """
    if applicability is not None and not applicability[0] and not override:
        return {"applicable": False, "reasons": list(applicability[1]),
                "verdict": "NOT APPLICABLE: the axisymmetric pipeline does not describe this flow, so no shear, core or rate numbers are given."}
    if geometry == "planar":
        rho = np.ones_like(psi)
    if omega_phi is None:
        u_r, u_z = ob.poloidal_velocity(psi, rho, dr, dz)
        omega_phi = ob.azimuthal_vorticity(u_r, u_z, dr, dz)
    om = omega_phi / rho
    mask = np.ones_like(psi, dtype=bool) if mask is None else mask
    level = 0.5 * (psi[mask].min() + psi[mask].max()) if level is None else level
    region = ob.closed_region(psi, mask, level)
    centers, g, d_om = ob.decompose(psi, rho, dr, dz, om, region, nbins)
    rel = ob.weighted_l2(d_om, rho, dr, dz, region) / max(ob.weighted_l2(om, rho, dr, dz, region), 1e-300)
    # shear of the ORBIT FREQUENCY omega(psi) = 2 pi / T(psi) governs mixing across streamlines (not g'(psi))
    lev, omega_orb = ob.orbit_frequency(psi, rho, dr, dz, region)
    if len(lev) < 8:
        return {"closed_cells": int(region.sum()), "rel_residual": rel, "applicable": False,
                "verdict": (f"fewer than 8 closed streamlines resolved ({len(lev)}): no shear or rate numbers. diagnose expects the stream function to have its "
                            "MINIMUM at the vortex core (pass -psi if the core is a maximum), a `level` that encloses the core, and a core several cells across.")}
    psi_u, om_s, shear, shear_noise = ob.smooth_shear(lev, omega_orb)
    zeros = ob.shear_zero_order(psi_u, shear)
    max_shear = float(np.max(np.abs(shear)))
    flat = rel < 1e-3  # delta-Omega negligible: nothing for shear-enhanced mixing to act on
    snr = max_shear / max(shear_noise, 1e-300)
    resolved = snr > 3 and not flat  # 3 < snr <= 10 is reported but flagged marginal
    out = {"closed_cells": int(region.sum()), "rel_residual": rel, "max_shear": max_shear,
           "shear_noise_floor": float(shear_noise), "shear_snr": float(snr),
           "shear_confidence": "no delta-Omega" if flat else ("high" if snr > 10 else ("marginal" if snr > 3 else "unresolved")),
           "zeros": zeros, "warnings": list(AXI_WARNINGS)}
    if flat:
        out["verdict"] = ("Omega is (numerically) constant on streamlines (delta-Omega ~ 0): nothing for shear-enhanced mixing "
                          "of Omega to act on. A passive scalar would still mix, governed by the orbit-frequency shear.")
    elif not resolved:
        out["verdict"] = ("shear is not resolved above numerical noise: no rate exponent reported. "
                          "Refine the grid or check the closed-streamline region before drawing conclusions.")
    elif not zeros:
        out["order_m"] = 1.0
        out["beta"] = ob.beta_from_order(1.0)
        out["rate_exponent"] = 1 / 3
        out["verdict"] = ("orbit-frequency shear resolved, no zero detected: non-degenerate (m = 1) at interior orbits (large-Pe "
                          "scaling kappa^(1/3) there). The slowest oscillatory mode of a smooth vortex CORE is different: kappa^(1/2), "
                          "see core_mode")
    else:
        m = min(z[1] for z in zeros)
        out["order_m"] = m
        out["beta"] = ob.beta_from_order(m)
        out["rate_exponent"] = m / (m + 2)
        out["verdict"] = (f"worst shear zero order m = {m:.2f}: decay rate ~ kappa^{m / (m + 2):.3f} "
                          f"(beta = {ob.beta_from_order(m):.3f}); prefactor undetermined")
    if resolved and not flat and out["shear_confidence"] == "marginal":
        out["verdict"] += f" [MARGINAL: shear signal-to-noise {snr:.1f}; indicative only]"
    if kappa is not None:
        cm = ob.core_mode(psi, rho, dr, dz, region, kappa)
        out["core_mode"] = cm if cm is not None else "no elliptic core in the closed region"
        if cm is not None:
            out["verdict"] += (f"; vortex-core mode decays at ~{cm['rate']:.3g} (1/time) at kappa = {kappa:g} "
                               f"(closed form; ellipticity {cm['ellipticity']:.2f}: {cm['accuracy_note']})")
    if kappa is not None and accurate_rate and geometry == "axisym":
        import true_rates
        nz = psi.shape[1]
        out["core_mode_2d_operator"] = true_rates.slowest_mode(psi, omega_phi, rho[:, 0], (np.arange(nz) + 0.5) * dz, dr, dz, kappa)
        out["core_mode_2d_operator"]["note"] = ("solves the full 2D operator on the whole window (minutes of compute): validated to <1% against an "
                                                "independent 3D evolution; preferred over the closed form when better than ~10% is needed")
    if kappa is not None and not experimental_rate:
        out["rate_estimate"] = ("withdrawn: the orbit-reduced estimate failed a preregistered test and under-predicts by 1.4-2.9x on a "
                                "simulated ring. Use true_rates.slowest_mode / decay_curve (solves the 2D operator directly; "
                                "validated on one flow). experimental_rate=True computes the withdrawn estimate for research only")
    elif kappa is not None and (flat or not resolved):
        out["rate_estimate"] = "not computed: shear is absent or unresolved"
    elif kappa is not None:
        p = 0 if geometry == "planar" else (3 if quantity == "omega" else 1)
        est = {bc: rates.reduced_rate(psi, rho, dr, dz, region, kappa, p, k_modes=(1,), bc=bc)["rate"]
               for bc in ("dirichlet", "neumann")}
        lo, hi = min(est.values()), max(est.values())
        out["rate_estimate"] = {"range": (lo, hi), "by_edge_condition": est,
                                "note": "orbit-reduced model estimate for angular mode k=1 (see check_rates.py for measured accuracy)"}
        out["verdict"] += f"; estimated decay rate {lo:.3g}-{hi:.3g} (1/time) at kappa = {kappa:g}"
    return out


def diagnose_periodic_3d(u, nu):
    n = u.shape[-1]
    g = Grid(n)
    w = g.curl(u)
    S = g.strain_from_vorticity(w)
    ws = np.einsum("i...,ij...,j...->...", w, S, w)
    w2 = g.integrate(np.sum(w * w, axis=0))
    s_inf = float(np.max(np.sqrt(np.einsum("ij...,ij...->...", S, S))))
    score = abs(g.integrate(ws)) / (w2 * s_inf)
    G = g.grad_tensor(w)
    mag = np.maximum(np.sqrt(np.sum(w * w, axis=0)), 1e-14)
    dmag = np.einsum("i...,ij...->j...", w, G) / mag
    dwhat = G / mag - np.einsum("i...,j...->ij...", w, dmag) / mag**2
    rot = np.sum(dwhat * dwhat, axis=(0, 1))
    what = w / mag
    sw = np.einsum("i...,ij...,j...->...", what, S, what)
    m = sw > 1e-8 * np.max(np.abs(sw))
    R = nu * rot[m] / sw[m]
    return {
        "euler_deviation_score": float(score),
        "stretching_region_fraction": float(m.mean()),
        "R_below_1_fraction": float(np.mean(R < 1)),
        "R_median": float(np.median(R)),
        "warnings": list(PERIODIC_WARNINGS),
    }


def _print(title, d):
    print(f"== {title}")
    for k, v in d.items():
        if k == "warnings":
            for w in v:
                print(f"   ! {w}")
        else:
            print(f"   {k}: {v}")


def demo():
    nr, nz = 300, 600
    dr, dz = 1.2 / nr, 2.4 / nz
    rho = (np.arange(nr) + 0.5) * dr
    z = -1.2 + (np.arange(nz) + 0.5) * dz
    R, Z = np.meshgrid(rho, z, indexing="ij")
    r2 = R**2 + Z**2
    psi_hill = -0.75 * R**2 * (1 - r2)
    inside = r2 < 0.97**2
    _print("Hill's vortex (expect no enhanced mixing)", diagnose_axisymmetric(psi_hill, R, dr, dz, mask=inside, level=-0.01))

    eps = 0.04
    psi_pert = psi_hill + eps * R**2 * (1 - r2) ** 2 * Z
    _print("perturbed ring, eps=0.04 (delta-Omega nonzero)", diagnose_axisymmetric(psi_pert, R, dr, dz, mask=inside, level=-0.01))

    g = Grid(32)
    x, y, zz = np.meshgrid(g.x, g.x, g.x, indexing="ij")
    A, B, C = 1.0, 0.7, 0.43
    u = np.stack([A * np.sin(zz) + C * np.cos(y), B * np.sin(x) + A * np.cos(zz), C * np.sin(y) + B * np.cos(x)])
    _print("ABC flow, nu=0.1 (steady Euler, forced)", diagnose_periodic_3d(u, 0.1))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        print(__doc__)
