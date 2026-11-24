"""Ground truth for the decay rate: slowest ORBIT-VARYING eigenvalue of the full 2D axisymmetric advection-diffusion operator
on a real (frozen) closed-streamline region. No time series, no fitting window.

  rho d_t c = -div(rho u c) + kappa div(rho grad c)      (passive scalar, p = 1), area element d rho dz
Cell-centred finite volume on a masked (rho,z) grid, rho = 1D array of cell-centre radii.
Advective face fluxes come from corner values of the stream function (rho u_rho = -psi_z, rho u_z = psi_rho):
  outward flux through E = -(pc[i,j]-pc[i,j-1]) c_f,  W = +(pc[i-1,j]-pc[i-1,j-1]) c_f,
                    N = +(pc[i,j]-pc[i-1,j]) c_f,     S = -(pc[i,j-1]-pc[i-1,j-1]) c_f,
with pc[i,j] the stream function at corner (i+1/2, j+1/2); the discrete advection is then exactly divergence-free.
Faces on the edge of the closed region carry no advective flux (the edge is a streamline); diffusion there is no-flux ('neumann')
or absorbing with c = 0 outside ('dirichlet').
Orbit-varying modes = eigenvalues with |Im| above a threshold (orbit-constant modes are real).
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs


def build_operator(psi, rho, dr, dz, region, kappa, bc, p=1):
    nr, nz = psi.shape
    assert not (region[0].any() or region[-1].any() or region[:, 0].any() or region[:, -1].any()), "region touches the grid edge"
    idx = -np.ones(psi.shape, dtype=np.int64)
    idx[region] = np.arange(int(region.sum()))
    n = int(region.sum())
    pc = 0.25 * (psi[:-1, :-1] + psi[1:, :-1] + psi[:-1, 1:] + psi[1:, 1:])
    rows, cols, vals = [], [], []
    ii, jj = np.nonzero(region)
    for i, j in zip(ii, jj):
        r0 = idx[i, j]
        rho_i = rho[i]
        V = rho_i * dr * dz
        # c_t + u.grad c = kappa rho^-p div(rho^p grad c); times rho, with div(rho u) = 0:  rho c_t + div(rho u c) = kappa rho^(1-p) div(rho^p grad c)
        # so a diffusive face conductance is rho_f^p rho_i^(1-p) (len/dist); for p = 1 this is the original rho_f (len/dist)
        wf = lambda rf: rf**p * rho_i ** (1 - p)
        faces = (
            (i + 1, j, -(pc[i, j] - pc[i, j - 1]), wf(rho_i + 0.5 * dr) * dz / dr),
            (i - 1, j, +(pc[i - 1, j] - pc[i - 1, j - 1]), wf(rho_i - 0.5 * dr) * dz / dr),
            (i, j + 1, +(pc[i, j] - pc[i - 1, j]), wf(rho_i) * dr / dz),
            (i, j - 1, -(pc[i, j - 1] - pc[i - 1, j - 1]), wf(rho_i) * dr / dz),
        )
        for ni, nj, A, g in faces:
            if region[ni, nj]:
                r1 = idx[ni, nj]
                rows += [r0, r0]
                cols += [r0, r1]
                vals += [(-kappa * g - 0.5 * A) / V, (kappa * g - 0.5 * A) / V]
            elif bc == "dirichlet":  # absorbing exterior (c = 0): upwind advection, ghost diffusion at half-cell distance
                rows.append(r0)
                cols.append(r0)
                vals.append((-2.0 * kappa * g - max(A, 0.0)) / V)
            else:  # 'neumann': mirrored ghost value keeps constants in the null space
                rows.append(r0)
                cols.append(r0)
                vals.append(-A / V)
    return sp.csc_matrix((vals, (rows, cols)), shape=(n, n))


def full_window_operator(psi, rho, dr, dz, kappa, p=1):
    """Operator on the WHOLE (rho,z) window (real exterior flow, absorbing edges, reflecting axis).

    psi must be the stream function in the frame where the structure is steady. It is padded by one cell with linear
    extrapolation so that the edge faces carry the correct advective flux.
    """
    nr, nz = psi.shape
    pp = np.empty((nr + 2, nz + 2))
    pp[1:-1, 1:-1] = psi
    pp[0, 1:-1] = 2 * psi[0] - psi[1]
    if abs(rho[0] - 0.5 * dr) < 1e-9 * dr:
        # window touches the axis (first face at rho = 0): psi = 0 there (stream function integrated from the axis), so the ghost row is -psi[0]
        # making the face value exactly 0 and the axis a true no-flux wall (linear extrapolation leaked O(dr^2) flux, see PREREGISTERED_TRUE_OPERATOR_AMENDMENT.md)
        pp[0, 1:-1] = -psi[0]
    pp[-1, 1:-1] = 2 * psi[-1] - psi[-2]
    pp[:, 0] = 2 * pp[:, 1] - pp[:, 2]
    pp[:, -1] = 2 * pp[:, -2] - pp[:, -3]
    rho_p = np.concatenate([[rho[0] - dr], rho, [rho[-1] + dr]])
    reg = np.zeros((nr + 2, nz + 2), dtype=bool)
    reg[1:-1, 1:-1] = True
    return build_operator(pp, rho_p, dr, dz, reg, kappa, "dirichlet", p=p), reg[1:-1, 1:-1]


def atmosphere_modes(L, inside, omega_range, im_min, n_shifts=6, real_guess=0.08, k_eigs=8, min_weight=0.9):
    """Smallest decay rate among eigenmodes localised (>= min_weight of |v|^2) inside the mask `inside` with |Im| >= im_min."""
    best = (np.inf, None, None)
    lo, hi = omega_range
    inside_flat = inside.ravel()
    for s in np.linspace(lo, hi, n_shifts):
        sigma = complex(-real_guess, s)
        try:
            ev, vec = eigs(L, k=k_eigs, sigma=sigma, which="LM")
        except Exception as e:
            print("   eigs failed:", type(e).__name__, flush=True)
            continue
        for lam, v in zip(ev, vec.T):
            w = np.abs(v) ** 2
            frac = float(w[inside_flat].sum() / w.sum())
            if abs(lam.imag) >= im_min and frac >= min_weight and -lam.real < best[0]:
                best = (float(-lam.real), complex(lam), frac)
    return best


def orbit_varying_rate(L, omega_range, im_min, n_shifts=12, real_guess=0.08, k_eigs=6):
    """Smallest decay rate -Re(lambda) among eigenvalues with |Im lambda| >= im_min, via shift-invert near -g + i*s."""
    best = (np.inf, None)
    lo, hi = omega_range
    for s in np.linspace(lo, hi, n_shifts):
        for sign in (+1,):  # L is real: eigenvalues come in conjugate pairs
            sigma = complex(-real_guess, sign * s)
            try:
                ev = eigs(L, k=k_eigs, sigma=sigma, which="LM", return_eigenvectors=False)
            except Exception as e:
                print("   eigs failed:", type(e).__name__, flush=True)
                continue
            for lam in ev:
                if abs(lam.imag) >= im_min and -lam.real < best[0]:
                    best = (float(-lam.real), complex(lam))
    return best
