"""End-to-end run on REAL simulation data: a viscous vortex ring from sim_ns3d.py (periodic 96^3, nu = 0.01).

Pipeline per snapshot: load -> cylindrical extraction -> stream function -> co-moving frame -> orbit decomposition
-> reduced rate estimate;  plus the 3D enstrophy / weighted budgets with the solver's exact du/dt.
Compares the ESTIMATED decay rate of the orbit-varying vorticity with what the simulation actually does.
"""
import glob
import sys
import numpy as np
import io_fields as io
import orbits as ob
import diagnose as dg
import fd3d

CENTER = (np.pi, np.pi)
fd3d.PERIODIC_METHOD = "spectral"  # periodic solver data: finite differences close the budgets only to ~3-5% at 5 cells/core


def load(i, case="ring"):
    d = np.load(f"data/{case}_{i:03d}.npz")
    return d["u"].astype(float), d["dudt"].astype(float), float(d["t"]), float(d["nu"]), float(d["h"])


def axisym(u, h, z_center, rho_max=2.4, z_half=2.0):
    return io.sample_axisymmetric(u, h, CENTER, rho_max=rho_max, z_half=z_half, nrho=120, nz=240, nphi=24, z_center=z_center)


def main():
    n_snap = len(glob.glob("data/ring_*.npz"))
    u0, _, t0, nu, h = load(0)
    a0 = axisym(u0, h, np.pi)
    u1, _, t1, _, _ = load(1)
    a1 = axisym(u1, h, np.pi + 0.5)
    speed0 = io.ring_speed(a0, t0, a1, t1)
    print(f"viscosity nu = {nu}; initial ring speed from centroid shift = {speed0:.3f}")

    zc, rows = np.pi, []
    prev = None
    for i in range(n_snap):
        u, dudt, t, nu, h = load(i)
        zc = np.pi + speed0 * t
        a = axisym(u, h, zc)
        w_prof = np.abs(a["omega_phi"]) ** 2
        zc2 = float(np.sum(w_prof * a["z"][None, :]) / np.sum(w_prof))
        a = axisym(u, h, zc2)
        if prev is not None:
            speed = io.ring_speed(prev[0], prev[1], a, t)
        else:
            speed = speed0
        prev = (a, t)
        psi, res = io.stream_function(a["u_rho"], a["u_z"], a["rho"], a["dr"], a["dz"])
        psi_co = io.to_comoving(psi, a["rho"], speed)
        core = np.unravel_index(np.argmax(np.abs(a["omega_phi"])), a["omega_phi"].shape)
        sign = -1.0 if psi_co[core] > 0 else 1.0
        lab = sign * psi_co
        R, _ = np.meshgrid(a["rho"], a["z"], indexing="ij")
        level = 0.02 * lab.min()
        d = dg.diagnose_axisymmetric(lab, R, a["dr"], a["dz"], omega_phi=a["omega_phi"], level=level, kappa=nu, quantity="omega",
                                      experimental_rate=True)
        est = d.get("rate_estimate")
        rate_txt = f"{est['range'][0]:.3g}-{est['range'][1]:.3g}" if isinstance(est, dict) else str(est)
        rows.append((t, d["rel_residual"], rate_txt, est, d.get("order_m"), a["nonaxisym"], a["swirl"], res, speed))
        print(f"t={t:4.1f} speed={speed:.3f} nonaxisym={a['nonaxisym']:.1e} swirl={a['swirl']:.1e} psi-resid={res:.1e} "
              f"| ||dOmega||/||Omega|| = {d['rel_residual']:.3f} | shear zero order {d.get('order_m')} | est. rate {rate_txt}", flush=True)

    ts = np.array([r[0] for r in rows])
    rel = np.array([r[1] for r in rows])
    sel = ts >= 1.5
    if sel.sum() >= 3 and np.all(rel[sel] > 0):
        slope = np.polyfit(ts[sel], np.log(rel[sel]), 1)[0]
        print(f"observed trend of ||dOmega||/||Omega|| for t >= 1.5: d ln/dt = {slope:+.3f}")
    last = [r for r in rows if isinstance(r[3], dict)]
    if last:
        lo, hi = last[-1][3]["range"]
        print(f"estimated decay rate of orbit-varying vorticity at kappa=nu (last snapshot): {lo:.3g}-{hi:.3g}")

    print("\n3D budgets (periodic, spectral-solver data, exact du/dt):")
    for i in (3, 7, n_snap - 1):
        u, dudt, t, nu, h = load(i)
        hh = (h,) * 3
        eb = fd3d.enstrophy_budget(u, hh, nu, periodic=True, dudt=dudt)
        wb = fd3d.weighted_budget(u, hh, nu, 0.5, periodic=True, dudt=dudt)
        print(f" t={t:.1f}: enstrophy closure {eb['closure_error']:+.1e}  (production {eb['production']:.3f}, "
              f"unsteady {eb['unsteady']:+.3f}, viscous {eb['viscous']:.3f});  weighted(a=0.5) closure {wb['closure_error']:+.1e}, "
              f"unsteady term {wb['unsteady']:+.3f} vs lhs {wb['lhs']:+.3f}")
    u, dudt, t, nu, h = load(5)
    hh = (h,) * 3
    print(f"\nbounded sub-boxes around the ring (t={t:.1f}): boundary-flux share of the weighted identity (alpha=0.5)")
    c = u.shape[1] // 2
    for half in (12, 20, 30, 40):
        box = tuple(slice(c - half, c + half) for _ in range(3))
        wb = fd3d.weighted_budget(u, hh, nu, 0.5, periodic=False, dudt=dudt, box=box)
        print(f"  box half-width {half * h:.2f}: lhs {wb['lhs']:+.3f}, unsteady {wb['unsteady']:+.3f}, boundary {wb['boundary']:+.3f}, "
              f"closure {wb['closure_error']:+.1e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
