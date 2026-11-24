"""Orbit-reduced decay-rate estimate: a NUMBER for the enhanced-dissipation rate, not just an exponent.

For  d_t f + u_p.grad f = kappa * rho^(-p) div(rho^p grad f)  on closed streamlines (p = 3: vorticity-derived Omega,
p = 1: passive scalar in axisymmetric flow, p = 0: planar flow with rho = 1), orbit averaging gives, for angular mode k,

    H_k f = -kappa (A(psi) f'' + B(psi) f') + i k w(psi) f ,
    A = <|grad psi|^2>,   B = < Lap psi + p psi_rho / rho >,   w = 2 pi / T(psi),   T = d(Phi)/d(psi),  Phi = int rho d rho dz,
averages taken with the invariant (time) measure rho d rho dz per psi-shell.
The estimate is  lambda = min_k  min Re spec(H_k).  It is a MODEL estimate: it neglects the theta-dependence of the
diffusion coefficients within an orbit. check_rates.py measures its accuracy against direct 2D solves.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs


def orbit_profiles(psi, rho, dr, dz, region, p, nbins=400):
    w = np.where(region, rho * dr * dz, 0.0)
    psi_r = np.gradient(psi, dr, axis=0)
    psi_z = np.gradient(psi, dz, axis=1)
    grad2 = psi_r**2 + psi_z**2
    lap = np.gradient(psi_r, dr, axis=0) + np.gradient(psi_z, dz, axis=1)
    lp = lap + p * psi_r / rho
    vals = psi[region]
    edges = np.linspace(vals.min(), vals.max(), nbins + 1)
    idx = np.clip(np.digitize(psi, edges) - 1, 0, nbins - 1)
    r = region
    den = np.bincount(idx[r], weights=w[r], minlength=nbins)
    good = den > 0
    cen = np.bincount(idx[r], weights=(w * psi)[r], minlength=nbins)[good] / den[good]
    A = np.bincount(idx[r], weights=(w * grad2)[r], minlength=nbins)[good] / den[good]
    B = np.bincount(idx[r], weights=(w * lp)[r], minlength=nbins)[good] / den[good]
    dpsi = np.diff(edges)[good]
    T = den[good] / dpsi
    return cen, A, B, T, (edges[0], edges[-1])


def _operator(kappa, k, psi, A, B, omega, bc):
    m = len(psi)
    h = psi[1] - psi[0]
    sgn = 1.0
    main = 2 * kappa * A / h**2 + 1j * k * omega
    low = -kappa * A / h**2 + kappa * B / (2 * h) * sgn
    up = -kappa * A / h**2 - kappa * B / (2 * h) * sgn
    H = sp.diags([low[1:], main, up[:-1]], [-1, 0, 1], format="lil", dtype=complex)
    if bc == "neumann":
        H[0, 1] = -2 * kappa * A[0] / h**2
        H[m - 1, m - 2] = -2 * kappa * A[-1] / h**2
    elif bc == "mixed":  # regular (Neumann) vortex centre, absorbing (Dirichlet) edge: scalar leaks into the exterior flow
        H[0, 1] = -2 * kappa * A[0] / h**2
    return H.tocsc()


def spectral_abscissa(kappa, k, psi, A, B, omega, bc="dirichlet"):
    H = _operator(kappa, k, psi, A, B, omega, bc)
    m = len(psi)
    h = psi[1] - psi[0]
    dw = np.gradient(omega, h)
    ddw = np.gradient(dw, h)
    cands = {0, m - 1}
    a = np.abs(dw)
    for i in range(2, m - 2):
        if a[i] <= a[i - 1] and a[i] <= a[i + 1] and a[i] < 0.5 * a.max():
            cands.add(i)
    best = (np.inf, None, None)
    for c in sorted(cands):
        s = abs(dw[c]) + 1e-300
        g_nd = (kappa * A[c] * (k * s) ** 2) ** (1 / 3)
        g_dg = np.sqrt(kappa * A[c] * k * abs(ddw[c]) / 2 + 1e-300)
        for scale in (g_nd, g_dg):
            for ph in (np.pi / 3, np.pi / 4):
                sigma = 1j * k * omega[c] + scale * np.exp(1j * ph)
                try:
                    ev = eigs(H, k=6, sigma=sigma, which="LM", return_eigenvectors=True)
                except Exception:
                    continue
                for lam, vec in zip(*ev):
                    if lam.real < best[0] and lam.real > 0:
                        loc = int(np.argmax(np.abs(vec)))
                        best = (float(lam.real), complex(lam), float(psi[loc]))
    return best


def reduced_rate(psi, rho, dr, dz, region, kappa, p, k_modes=(1, 2, 3), nbins=400, m_grid=4000, bc="dirichlet"):
    cen, A, B, T, (lo, hi) = orbit_profiles(psi, rho, dr, dz, region, p, nbins)
    grid = np.linspace(cen[0], cen[-1], m_grid)
    Ag = np.interp(grid, cen, A)
    Bg = np.interp(grid, cen, B)
    Tg = np.interp(grid, cen, T)
    omega = 2 * np.pi / Tg
    inner = slice(1, -1)  # unknowns exclude the Dirichlet end points
    best = (np.inf, None, None, None)
    for k in k_modes:
        lam, ev, loc = spectral_abscissa(kappa, k, grid[inner], Ag[inner], Bg[inner], omega[inner], bc)
        if lam < best[0]:
            best = (lam, ev, loc, k)
    return {"rate": best[0], "eigenvalue": best[1], "localised_at_psi": best[2], "mode_k": best[3],
            "psi_range": (float(cen[0]), float(cen[-1])), "model": "orbit-reduced (estimate)"}
