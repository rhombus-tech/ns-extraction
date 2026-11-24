"""2D advection-diffusion of a dye blob in a Lamb-Oseen (or point) vortex, polar coordinates.

Advection by the azimuthal flow is exact (each angular Fourier mode m is rotated by exp(-i m Omega dt));
diffusion is a Crank-Nicolson radial solve per mode (finite volume, no-flux ends); Strang splitting.
Parameters are the PUBLISHED values of Meunier & Villermaux, J. Fluid Mech. 476 (2003): Gamma=14.2 cm^2/s,
a0=0.3 cm, D=5e-6 cm^2/s, s0=0.22 cm (blob width), nu=0.01 cm^2/s for a^2=a0^2+4 nu t (their eq. 2.2).

  python3 mv_sim.py verify      solver check against the paper's closed form (steady point vortex); NO measured data used
  python3 mv_sim.py run B       full run, evolving Lamb-Oseen (primary);  'run A' = steady point vortex (secondary)
"""
import sys, json, time
import numpy as np
from math import erf, sqrt, pi

G, D, S0, A0, NU = 14.2, 5e-6, 0.22, 0.3, 0.01


def theory_cM(r, t, Gam=G, D=D, s0=S0):
    tau = D * t / s0**2 + D * Gam**2 * t**3 / (3 * pi**2 * r**4 * s0**2)
    return erf(1 / (4 * sqrt(tau)))


def solve(point=False, Nr=2400, M=2048, dt=0.02, R=3.2, rin=0.6, rout=2.4, tmax=20.0,
          rec_r=(1.32,), snap=(5.0, 10.0, 20.0), Gam=G, Dif=D, s0=S0, a0=A0, nu=NU, verbose=True):
    dr = R / Nr
    r = (np.arange(Nr) + 0.5) * dr
    rf = np.arange(Nr + 1) * dr
    Nth = 2 * M
    th = np.arange(Nth) * (2 * pi / Nth)
    dth = np.minimum(th, 2 * pi - th)
    c = np.zeros((Nr, Nth))
    for j in range(Nr):
        if rin <= r[j] <= rout:
            c[j, dth < s0 / (2 * r[j])] = 1.0
    ch = np.fft.rfft(c, axis=1)
    m = np.arange(M + 1, dtype=float)
    al = Dif * dt / 2
    lo = -al * rf[:-1] / (r * dr**2)
    up = -al * rf[1:] / (r * dr**2)
    diag0 = 1 + al * (rf[1:] + rf[:-1]) / (r * dr**2)
    b = diag0[:, None] + al * (m[None, :] ** 2) / (r[:, None] ** 2)
    bp = np.empty_like(b)
    w = np.zeros_like(b)
    bp[0] = b[0]
    for j in range(1, Nr):
        w[j] = lo[j] / bp[j - 1]
        bp[j] = b[j] - w[j] * up[j - 1]
    inv_bp = 1.0 / bp
    ex_diag = 1 - al * (rf[1:] + rf[:-1]) / (r * dr**2)
    ex_lo = al * rf[:-1] / (r * dr**2)
    ex_up = al * rf[1:] / (r * dr**2)
    mm = al * (m[None, :] ** 2) / (r[:, None] ** 2)
    rec_idx = [int(np.argmin(np.abs(r - x))) for x in rec_r]
    nst = int(round(tmax / dt))
    times = []
    series = {x: [] for x in rec_r}
    snaps = {}
    snap_steps = {int(round(s / dt)): s for s in snap}
    t = 0.0
    t0 = time.time()
    for n in range(nst):
        if point:
            Om = Gam / (2 * pi * r**2)
        else:
            a2 = a0**2 + 4 * nu * (t + dt / 2)
            Om = Gam / (2 * pi * r**2) * (-np.expm1(-r**2 / a2))
        wph = np.exp(-1j * Om * dt / 2)
        E = np.cumprod(np.concatenate([np.ones((Nr, 1), complex), np.repeat(wph[:, None], M, axis=1)], axis=1), axis=1)
        ch *= E
        rhs = ch * (ex_diag[:, None] - mm)
        rhs[1:] += ex_lo[1:, None] * ch[:-1]
        rhs[:-1] += ex_up[:-1, None] * ch[1:]
        for j in range(1, Nr):
            rhs[j] -= w[j] * rhs[j - 1]
        x = np.empty_like(rhs)
        x[-1] = rhs[-1] * inv_bp[-1]
        for j in range(Nr - 2, -1, -1):
            x[j] = (rhs[j] - up[j] * x[j + 1]) * inv_bp[j]
        ch = x * E
        t += dt
        times.append(t)
        for x_, i in zip(rec_r, rec_idx):
            series[x_].append(float(np.fft.irfft(ch[i], n=Nth).max()))
        if (n + 1) in snap_steps:
            f = np.fft.irfft(ch, n=Nth, axis=1)
            snaps[snap_steps[n + 1]] = f.max(axis=1).tolist()
        if verbose and (n + 1) % 100 == 0:
            print(f"  step {n+1}/{nst}  t={t:.2f}  {time.time()-t0:.0f}s", flush=True)
    return dict(times=times, series={str(k): v for k, v in series.items()}, snaps=snaps, r=r.tolist(),
                params=dict(point=point, Nr=Nr, M=M, dt=dt, R=R, rin=rin, rout=rout, tmax=tmax))


def verify():
    """Steady point vortex: the solver must reproduce the paper's closed form (their eq. 3.11) where that form applies."""
    res = solve(point=True, Nr=1600, M=1024, dt=0.04, tmax=20.0, rec_r=(1.32, 1.5, 1.8), snap=(5.0, 10.0, 20.0), verbose=False)
    t = np.array(res["times"])
    worst = 0
    for rr in (1.32, 1.5, 1.8):
        num = np.array(res["series"][str(rr)])
        th = np.array([theory_cM(rr, tt) for tt in t])
        mask = t >= 3.0
        dev = np.max(np.abs(num[mask] - th[mask]))
        worst = max(worst, dev)
        print(f"point vortex r={rr}: max |numerical - eq.3.11| for t in [3,20] = {dev:.4f}")
    r = np.array(res["r"])
    for s in (5.0, 10.0, 20.0):
        f = np.array(res["snaps"][s])
        sel = [(1.0, 1.0), (1.3, 1.3), (1.6, 1.6), (1.9, 1.9)]
        out = []
        for rr, _ in sel:
            j = int(np.argmin(np.abs(r - rr)))
            out.append((rr, round(f[j], 3), round(theory_cM(rr, s), 3)))
        print(f"  snapshot t={s}: (r, numerical, eq3.11) {out}")
    print("SOLVER VERIFY:", "OK" if worst < 0.05 else "NOT OK", f"(worst {worst:.4f}, limit 0.05)")


if __name__ == "__main__":
    if sys.argv[1] == "verify":
        verify()
    elif sys.argv[1] == "run":
        variant = sys.argv[2]
        tag = sys.argv[3] if len(sys.argv) > 3 else "fine"
        cfg = dict(fine=dict(Nr=2400, M=2048, dt=0.02), coarse=dict(Nr=1200, M=1024, dt=0.04))[tag]
        rec = tuple(np.round(np.arange(0.6, 2.01, 0.1), 2).tolist()) + (1.32,)
        res = solve(point=(variant == "A"), rec_r=rec, **cfg)
        json.dump(res, open(f"result_{variant}_{tag}.json", "w"))
        print("saved", f"result_{variant}_{tag}.json")
