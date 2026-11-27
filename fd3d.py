"""Finite-difference 3D diagnostics for non-periodic / wall-bounded / unsteady data.

Fields live on a regular grid, u has shape (3, nx, ny, nz), spacing h = (hx, hy, hz).
Every budget returns each term separately so that boundary flux, unsteadiness and forcing are visible
instead of silently violating the idealised (steady, unforced, decaying) identities of the paper.

Enstrophy budget on a region V (f = |w|^2):
    int w.S.w = (1/2) d/dt int f  +  (1/2) oint f u.n  -  (1/2) nu oint d_n f  +  nu int |grad w|^2  -  int w.curl F
Weighted identity (alpha > -1/2):
    int f^(a+1) (w^.S.w^ - nu|grad w^|^2) = (1+2a) nu int f^a |grad|w||^2  +  U  +  B  -  Fterm
  with  U = int f^a w.w_t,   B = oint [ f^(a+1) u.n /(2(a+1)) - (nu/2) f^a d_n f ],   Fterm = int f^a w.curl F.
"""
import numpy as np

EDGE = 3  # cells excluded next to non-periodic boundaries (finite-difference stencils)


PERIODIC_METHOD = "fd"  # set to "spectral" for periodic data: exponentially accurate on smooth fields


def diff(a, axis, h, periodic):
    if periodic and PERIODIC_METHOD == "spectral":
        n = a.shape[axis]
        k = 2 * np.pi * np.fft.fftfreq(n, d=h)
        if n % 2 == 0:
            k[n // 2] = 0.0
        shape = [1] * a.ndim
        shape[axis] = n
        return np.fft.ifft(1j * k.reshape(shape) * np.fft.fft(a, axis=axis), axis=axis).real
    if periodic:
        return (np.roll(a, -1, axis) - np.roll(a, 1, axis)) / (2 * h)
    return np.gradient(a, h, axis=axis, edge_order=2)


def grad_tensor(u, h, periodic):
    """G[i,j] = d_j u_i."""
    g = np.empty((3, 3) + u.shape[1:])
    for i in range(3):
        for j in range(3):
            g[i, j] = diff(u[i], j, h[j], periodic)
    return g


def curl_from_grad(g):
    return np.stack([g[2, 1] - g[1, 2], g[0, 2] - g[2, 0], g[1, 0] - g[0, 1]])


def curl(u, h, periodic):
    return curl_from_grad(grad_tensor(u, h, periodic))


class Quantities:
    def __init__(self, u, h, periodic):
        self.h = h
        self.periodic = periodic
        gu = grad_tensor(u, h, periodic)
        self.S = 0.5 * (gu + np.swapaxes(gu, 0, 1))
        self.w = curl_from_grad(gu)
        self.G = grad_tensor(self.w, h, periodic)  # d_j w_i
        self.f = np.sum(self.w * self.w, axis=0)
        self.mag = np.maximum(np.sqrt(self.f), 1e-14)
        self.dmag = np.einsum("i...,ij...->j...", self.w, self.G) / self.mag
        what = self.w / self.mag
        self.dwhat = self.G / self.mag - np.einsum("i...,j...->ij...", self.w, self.dmag) / self.mag**2
        self.wSw = np.einsum("i...,ij...,j...->...", self.w, self.S, self.w)
        self.whSwh = np.einsum("i...,ij...,j...->...", what, self.S, what)
        self.rot = np.sum(self.dwhat * self.dwhat, axis=(0, 1))
        self.dfdx = np.stack([diff(self.f, j, h[j], periodic) for j in range(3)])


def region(shape, periodic, margin=EDGE):
    if periodic:
        return tuple(slice(0, n) for n in shape)
    return tuple(slice(margin, n - margin) for n in shape)


def integral(a, h, box):
    return float(np.sum(a[box]) * h[0] * h[1] * h[2])


def face_integral(fn, h, box, axis, side):
    """Integral over one face of the box of fn (3D array), averaged to the true cell face."""
    sl = box[axis]
    # inner cell index: last cell inside the box on this side
    inner = sl.start if side < 0 else sl.stop - 1
    # outer cell index: first cell outside the box on this side
    outer = inner + side
    if not 0 <= outer < fn.shape[axis]:
        outer = inner  # box touches the array edge: no outer cell, fall back to the inner value (no wrap-around)
    idx_in = [slice(None)] * 3
    idx_out = [slice(None)] * 3
    idx_in[axis] = inner
    idx_out[axis] = outer
    for a in range(3):
        if a != axis:
            idx_in[a] = box[a]
            idx_out[a] = box[a]
    other = [h[a] for a in range(3) if a != axis]
    face_val = 0.5 * (fn[tuple(idx_in)] + fn[tuple(idx_out)])
    return float(np.sum(face_val) * other[0] * other[1])


def boundary_flux(u, q, h, box, nu, alpha):
    """B = oint [ f^(a+1) u.n/(2(a+1)) - (nu/2) f^a d_n f ] over the 6 faces of the box."""
    total = 0.0
    adv = q.f ** (alpha + 1) / (2 * (alpha + 1))
    for axis in range(3):
        for side in (-1, +1):
            fn = side * (adv * u[axis] - 0.5 * nu * q.f**alpha * q.dfdx[axis])
            total += face_integral(fn, h, box, axis, side)
    return total


def enstrophy_flux_terms(u, q, h, box, nu):
    adv = diff_free = 0.0
    for axis in range(3):
        for side in (-1, +1):
            adv += face_integral(side * 0.5 * q.f * u[axis], h, box, axis, side)
            diff_free += face_integral(side * 0.5 * nu * q.dfdx[axis], h, box, axis, side)
    return adv, diff_free


def enstrophy_budget(u, h, nu, periodic=False, force=None, dudt=None, box=None):
    """Closure check of the enstrophy balance; returns every term."""
    q = Quantities(u, h, periodic)
    box = box or region(u.shape[1:], periodic)
    prod = integral(q.wSw, h, box)
    visc = nu * integral(np.sum(q.G * q.G, axis=(0, 1)), h, box)
    out = {"production": prod, "viscous": visc}
    adv, dif = (0.0, 0.0) if periodic else enstrophy_flux_terms(u, q, h, box, nu)
    out["flux_advective"], out["flux_diffusive"] = adv, -dif
    dt_term = 0.0
    if dudt is not None:
        wt = curl(dudt, h, periodic)
        dt_term = integral(np.sum(q.w * wt, axis=0), h, box)
    out["unsteady"] = dt_term
    fterm = 0.0
    if force is not None:
        cf = curl(force, h, periodic)
        fterm = integral(np.sum(q.w * cf, axis=0), h, box)
    out["forcing"] = -fterm
    rhs = dt_term + adv - dif + visc - fterm
    scale = max(abs(prod), abs(rhs), abs(visc), abs(dt_term), abs(adv), abs(dif), abs(fterm), 1e-30)
    out["closure_error"] = (prod - rhs) / scale
    out["steady_euler_deviation"] = abs(prod) / max(integral(np.abs(q.wSw), h, box), 1e-30)
    return out


def weighted_budget(u, h, nu, alpha, periodic=False, force=None, dudt=None, box=None):
    q = Quantities(u, h, periodic)
    box = box or region(u.shape[1:], periodic)
    # Compute using well-defined products that avoid 0/0 at vorticity zeros:
    #   f^(alpha+1) * hat{w}.S.hat{w} = f^alpha * w.S.w  (always exact)
    #   f^alpha * |grad|w||^2 = f^(alpha-1) * sum_j (w.G_j)^2
    # For alpha >= 1, f^(alpha-1) is safe.  For alpha = 0, use wG2/max(f, tiny)
    # where wG2 <= f * |G|^2 (Cauchy-Schwarz), so the ratio is bounded.
    fa_wSw = integral(q.f ** alpha * q.wSw, h, box)
    G2 = np.sum(q.G * q.G, axis=(0, 1))
    wG = np.einsum("i...,ij...->j...", q.w, q.G)
    wG2 = np.sum(wG * wG, axis=0)
    fa_G2 = integral(q.f ** alpha * G2, h, box)
    if alpha >= 1:
        fa_dmag2 = integral(q.f ** (alpha - 1) * wG2, h, box)
    else:
        fa_dmag2 = integral(q.f ** alpha * wG2 / np.maximum(q.f, 1e-30), h, box)
    stretch = fa_wSw - nu * fa_G2 + nu * fa_dmag2
    rhs_visc = (1 + 2 * alpha) * nu * fa_dmag2
    U = integral(q.f**alpha * np.sum(q.w * curl(dudt, h, periodic), axis=0), h, box) if dudt is not None else 0.0
    Fterm = integral(q.f**alpha * np.sum(q.w * curl(force, h, periodic), axis=0), h, box) if force is not None else 0.0
    B = 0.0 if periodic else boundary_flux(u, q, h, box, nu, alpha)
    resid = stretch - rhs_visc
    scale = max(abs(stretch), abs(rhs_visc), abs(U), abs(B), abs(Fterm), 1e-30)
    return {"lhs": stretch, "rhs_viscous": rhs_visc, "residual": resid, "unsteady": U, "boundary": B,
            "forcing": -Fterm, "closure_error": (resid - (U + B - Fterm)) / scale}


def regularity_summary(u, h, nu, periodic=False, box=None):
    q = Quantities(u, h, periodic)
    box = box or region(u.shape[1:], periodic)
    sw = q.whSwh[box]
    m = sw > 1e-8 * max(np.max(np.abs(sw)), 1e-30)
    R = nu * q.rot[box][m] / sw[m]
    w2 = integral(q.f, h, box)
    s_inf = float(np.max(np.sqrt(np.einsum("ij...,ij...->...", q.S[(slice(None), slice(None)) + box], q.S[(slice(None), slice(None)) + box]))))
    return {
        "stretching_fraction": float(m.mean()),
        "R_below_1_fraction": float(np.mean(R < 1)) if m.any() else float("nan"),
        "R_median": float(np.median(R)) if m.any() else float("nan"),
        "euler_deviation_score": abs(integral(q.wSw, h, box)) / (w2 * s_inf) if w2 > 0 else float("nan"),
    }
