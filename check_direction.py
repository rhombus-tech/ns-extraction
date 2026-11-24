"""Checks for the vorticity-direction tools on a steady forced Beltrami (ABC) flow on the periodic box.

ABC flow: curl u = u, steady NS with forcing F = nu*u (since (u.grad)u = grad|u|^2/2 and Lap u = -u).
Derived family (forcing included):
  int f^(a+1) (w^ S w^ - nu |grad w^|^2) = (1+2a) nu int f^a |grad|w||^2 - int f^a w.curl F,   f=|w|^2.
With F = 0 this is the paper's Theorem; here curl F = nu*w so the forcing term is nu*int f^(a+1).
"""
import sys
import numpy as np
from spectral import Grid, random_vorticity

N = 48
NU = 0.1


def abc_velocity(g, A=1.0, B=0.7, C=0.43):
    x, y, z = np.meshgrid(g.x, g.x, g.x, indexing="ij")
    return np.stack([A * np.sin(z) + C * np.cos(y),
                     B * np.sin(x) + A * np.cos(z),
                     C * np.sin(y) + B * np.cos(x)])


def fields(g, w):
    G = g.grad_tensor(w)  # G[i,j] = d_j w_i
    mag = np.maximum(np.sqrt(np.sum(w * w, axis=0)), 1e-14)
    dmag = np.einsum("i...,ij...->j...", w, G) / mag
    what = w / mag
    dwhat = G / mag - np.einsum("i...,j...->ij...", w, dmag) / mag**2
    S = g.strain_from_vorticity(w)
    return mag, dmag, what, dwhat, G, S


def main():
    g = Grid(N)
    ok = True
    u = abc_velocity(g)
    w = g.curl(u)
    print(f"ABC: max |curl u - u| = {np.max(np.abs(w - u)):.2e}, max |div u| = {np.max(np.abs(g.divergence(u))):.2e}")
    mag, dmag, what, dwhat, G, S = fields(g, w)

    lhs = np.sum(G * G, axis=(0, 1))
    rhs = np.sum(dmag * dmag, axis=0) + mag**2 * np.sum(dwhat * dwhat, axis=(0, 1))
    err = np.max(np.abs(lhs - rhs) / np.maximum(lhs, 1e-12))
    good = err < 1e-9
    ok &= good
    print(f"algebraic decomposition |grad w|^2 = |grad|w||^2 + |w|^2|grad w^|^2: max rel err {err:.2e} {'PASS' if good else 'FAIL'}")

    ws = np.einsum("i...,ij...,j...->...", w, S, w)
    prod = g.integrate(ws)
    scale = g.integrate(np.abs(ws))
    good = abs(prod) / scale < 1e-9
    ok &= good
    print(f"Euler production int w.S.w for steady Euler (ABC, periodic): {prod:.2e} (scale {scale:.2e}) {'PASS' if good else 'FAIL'}")

    rng = np.random.default_rng(3)
    wr = random_vorticity(g, rng, kmax=3.0)
    Sr = g.strain_from_vorticity(wr)
    pr = g.integrate(np.einsum("i...,ij...,j...->...", wr, Sr, wr))
    sr = g.integrate(np.abs(np.einsum("i...,ij...,j...->...", wr, Sr, wr)))
    print(f"control, random field (not steady Euler): int w.S.w / scale = {abs(pr)/sr:.2e} (nonzero expected)")

    sw = np.einsum("i...,ij...,j...->...", what, S, what)
    rot = np.sum(dwhat * dwhat, axis=(0, 1))
    f = mag**2
    for a in (-0.5, 0.0, 0.5, 1.0):
        lhs_a = g.integrate(f ** (a + 1) * (sw - NU * rot))
        rhs_unforced = (1 + 2 * a) * NU * g.integrate(f**a * np.sum(dmag * dmag, axis=0))
        forcing = NU * g.integrate(f ** (a + 1))
        resid_unforced = lhs_a - rhs_unforced
        resid_forced = lhs_a - (rhs_unforced - forcing)
        norm = max(abs(lhs_a), abs(rhs_unforced), abs(forcing), 1e-12)
        good = abs(resid_forced) / norm < 2e-2
        ok &= good
        print(f"weighted identity alpha={a:+.1f}: LHS={lhs_a:+.4f}  residual unforced form = {resid_unforced:+.4f}, "
              f"with forcing term = {resid_forced:+.2e} (rel {abs(resid_forced)/norm:.1e}) {'PASS' if good else 'FAIL'}")

    mask = sw > 1e-8
    for nu in (0.01, 0.1, 0.5, 2.0):
        R = nu * rot[mask] / sw[mask]
        frac = np.mean(R < 1)
        print(f"regularity indicator nu={nu:5.2f}: volume fraction (of stretching region) with R<1 = {frac:.2f}; median R = {np.median(R):.3g}")
    print("ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
