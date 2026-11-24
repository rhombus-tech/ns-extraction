"""Experimental rate tool for FROZEN closed-streamline flows: solve the 2D axisymmetric advection-diffusion operator directly.

Validated (PREREGISTERED_TRUE_OPERATOR.md, one flow): reproduces an independent 3D pseudo-spectral scalar evolution to < 1%
(orbit-varying norm vs time) and its slowest atmosphere eigenvalue is converged to 0.3% between two grids.

    decay_curve(...)   predicted ||delta_c(t)||/||delta_c(0)|| for an initial scalar field you supply
    slowest_mode(...)  slowest atmosphere eigenmode (rate, Im, localisation) or None if no localised mode exists

Inputs: the stream function in the frame where the structure is steady, on the WHOLE (rho,z) window (exterior flow included),
the azimuthal vorticity omega_phi (only to locate the core), and the diffusivity kappa. The window edges are absorbing.
A 'rate' is meaningful only when a localised mode exists; at larger kappa the region leaks and there may be none.
"""
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.sparse.linalg import expm_multiply
import orbits as ob
import rates
import true_operator as to


def _setup(psi_co, omega_phi, rho, z, dr, dz, refine):
    core = np.unravel_index(np.argmax(np.abs(omega_phi)), omega_phi.shape)
    sign = -1.0 if psi_co[core] > 0 else 1.0
    spl = RectBivariateSpline(rho, z, psi_co, kx=3, ky=3)
    rf = (np.arange(int(round(len(rho) * refine))) + 0.5) * dr / refine
    zf = z[0] - dz / 2 + (np.arange(int(round(len(z) * refine))) + 0.5) * dz / refine
    pf = spl(rf, zf)
    lab = sign * pf
    R, Z = np.meshgrid(rf, zf, indexing="ij")
    region = ob.closed_region(lab, np.ones_like(lab, dtype=bool), 0.02 * lab.min())
    return pf, lab, rf, zf, R, Z, dr / refine, dz / refine, region


def slowest_mode(psi_co, omega_phi, rho, z, dr, dz, kappa, refine=1.5, n_shifts=6, min_localisation=0.9):
    pf, lab, rf, zf, R, Z, drf, dzf, region = _setup(psi_co, omega_phi, rho, z, dr, dz, refine)
    _, _, _, T, _ = rates.orbit_profiles(lab, R, drf, dzf, region, 1)
    w = 2 * np.pi / T
    L, _ = to.full_window_operator(pf, rf, drf, dzf, kappa)
    rate, ev, frac = to.atmosphere_modes(L, region, (w.min(), w.max()), im_min=0.3 * w.min(), n_shifts=n_shifts,
                                         min_weight=min_localisation)
    if ev is None:
        return {"rate": None, "message": "no single rate: no eigenmode is localised in the closed region at this kappa "
                                         "(the region leaks into the exterior flow)", "cells": pf.size}
    return {"rate": rate, "eigenvalue": ev, "localisation": frac, "cells": pf.size,
            "message": "slowest eigenmode localised in the closed-streamline region (frozen flow, absorbing window edges)"}


def decay_curve(psi_co, omega_phi, rho, z, dr, dz, kappa, c0, t_end, n_out=25, refine=2):
    """c0: function (R, Z) -> initial scalar on the refined grid. Returns times, r(t) = ||dc(t)||/||dc(0)|| (orbit-varying, closed region)."""
    pf, lab, rf, zf, R, Z, drf, dzf, region = _setup(psi_co, omega_phi, rho, z, dr, dz, refine)
    L, _ = to.full_window_operator(pf, rf, drf, dzf, kappa)
    frames = expm_multiply(L, c0(R, Z).ravel(), start=0.0, stop=t_end, num=n_out, endpoint=True)
    norms = []
    for c in frames:
        _, _, dc = ob.decompose(lab, R, drf, dzf, c.reshape(lab.shape), region, 160)
        norms.append(ob.weighted_l2(dc, R, drf, dzf, region))
    norms = np.array(norms)
    return np.linspace(0.0, t_end, n_out), norms / norms[0]
