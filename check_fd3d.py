"""Validate the finite-difference diagnostics (fd3d) against spectral results and exact solutions.

Exact solutions used: ABC (Beltrami) flow.
  steady forced : u, F = nu*u
  decaying      : u(x,t) = exp(-nu t) u_ABC(x), unforced, exact unsteady NS solution (du/dt = -nu u)
"""
import sys
import numpy as np
import fd3d
from spectral import Grid

NU = 0.1


def abc(n, scale=1.0):
    g = Grid(n)
    x, y, z = np.meshgrid(g.x, g.x, g.x, indexing="ij")
    A, B, C = 1.0, 0.7, 0.43
    u = np.stack([A * np.sin(z) + C * np.cos(y), B * np.sin(x) + A * np.cos(z), C * np.sin(y) + B * np.cos(x)])
    return g, scale * u


def main():
    ok = True
    n = 64
    g, u = abc(n)
    h = (2 * np.pi / n,) * 3

    w_sp = g.curl(u)
    w_fd = fd3d.curl(u, h, True)
    err = np.max(np.abs(w_sp - w_fd)) / np.max(np.abs(w_sp))
    good = err < 2e-2
    ok &= good
    print(f"FD vs spectral vorticity (N={n}): max rel err {err:.2e} {'PASS' if good else 'FAIL'}")

    out = fd3d.enstrophy_budget(u, h, NU, periodic=True, force=NU * u)
    good = abs(out["closure_error"]) < 5e-3
    ok &= good
    print(f"periodic steady forced ABC, enstrophy budget: closure error {out['closure_error']:.2e}, "
          f"production {out['production']:.2e}, viscous {out['viscous']:.3f}, forcing {out['forcing']:.3f} {'PASS' if good else 'FAIL'}")

    for a in (0.0, 0.5, 1.0):
        wb = fd3d.weighted_budget(u, h, NU, a, periodic=True, force=NU * u)
        wb0 = fd3d.weighted_budget(u, h, NU, a, periodic=True)
        good = abs(wb["closure_error"]) < 5e-3
        ok &= good
        print(f"periodic weighted budget alpha={a}: closure error with forcing {wb['closure_error']:.1e}; "
              f"residual if forcing ignored = {wb0['residual']:+.2f} (lhs {wb['lhs']:+.2f}) {'PASS' if good else 'FAIL'}")

    t = 0.7
    u_t = np.exp(-NU * t) * u
    dudt = -NU * u_t
    out = fd3d.enstrophy_budget(u_t, h, NU, periodic=True, dudt=dudt)
    good = abs(out["closure_error"]) < 5e-3
    ok &= good
    print(f"decaying exact solution, unsteady enstrophy budget: closure {out['closure_error']:.1e}, "
          f"unsteady {out['unsteady']:+.3f}, viscous {out['viscous']:+.3f} {'PASS' if good else 'FAIL'}")
    wb = fd3d.weighted_budget(u_t, h, NU, 0.5, periodic=True, dudt=dudt)
    good = abs(wb["closure_error"]) < 5e-3
    ok &= good
    print(f"decaying weighted budget alpha=0.5: closure {wb['closure_error']:.1e}, unsteady term {wb['unsteady']:+.3f} "
          f"vs lhs {wb['lhs']:+.3f} {'PASS' if good else 'FAIL'}")

    box = (slice(10, 44), slice(8, 50), slice(12, 40))
    out = fd3d.enstrophy_budget(u, h, NU, periodic=False, force=NU * u, box=box)
    good = abs(out["closure_error"]) < 1e-2
    ok &= good
    print(f"bounded sub-box, steady forced ABC, enstrophy budget: closure {out['closure_error']:.1e}; "
          f"production {out['production']:+.3f}, adv.flux {out['flux_advective']:+.3f}, diff.flux {out['flux_diffusive']:+.3f} "
          f"{'PASS' if good else 'FAIL'}")
    for a in (0.5, 1.0):
        wb = fd3d.weighted_budget(u, h, NU, a, periodic=False, force=NU * u, box=box)
        good = abs(wb["closure_error"]) < 1e-2
        ok &= good
        print(f"bounded sub-box weighted alpha={a}: closure {wb['closure_error']:.1e}; boundary term {wb['boundary']:+.3f} "
              f"vs lhs {wb['lhs']:+.3f} (ignoring it would be wrong by {abs(wb['boundary'])/max(abs(wb['lhs']),1e-30):.0%}) "
              f"{'PASS' if good else 'FAIL'}")
    print("ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
