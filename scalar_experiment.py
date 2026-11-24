"""Stage 1 experiment (PREREGISTERED_STAGE1.md): passive scalar in the vortex-ring flow.

    python3 scalar_experiment.py frozen   # F: velocity frozen at the t=2.5 snapshot, in the ring frame
    python3 scalar_experiment.py full     # S: scalar carried by the evolving flow, t = 10
Writes stage1_frozen.npz / stage1_full.npz (rho,z slices + resolution diagnostics); analysis is analyze_stage1.py.
"""
import sys
import time
import numpy as np
import io_fields as io
import run_ring_case as rc
from sim_ns3d import NS3D, ring_vorticity

N = 96
NU = 0.01
KAPPAS = (3e-2, 1e-2, 3.3e-3, 1.1e-3)
DT = 0.02
OUT_EVERY = 25
CENTER = (np.pi, np.pi)
RHO_MAX, Z_HALF = 2.4, 2.0
SL = dict(nrho=120, nz=240, nphi=24)


class Scalar:
    def __init__(self, n):
        self.n = n
        k1 = np.fft.fftfreq(n, d=1.0 / n)
        kz = np.fft.rfftfreq(n, d=1.0 / n)
        self.k = np.stack(np.meshgrid(k1, k1, kz, indexing="ij"))
        self.k2 = np.sum(self.k**2, axis=0)
        lim = n / 3.0
        self.dealias = (np.abs(self.k[0]) < lim) & (np.abs(self.k[1]) < lim) & (np.abs(self.k[2]) < lim)

    def rhs(self, u, ch, kappa):
        adv = 0.0
        for i in range(3):
            adv = adv + u[i] * np.fft.irfftn(1j * self.k[i] * ch, s=(self.n,) * 3)
        return -np.fft.rfftn(adv) * self.dealias - kappa * self.k2 * ch

    def tail_fraction(self, c):
        ch = np.fft.fftn(c)
        n = self.n
        k1 = np.fft.fftfreq(n, d=1.0 / n)
        kk = np.sqrt(sum(a**2 for a in np.meshgrid(k1, k1, k1, indexing="ij")))
        p = np.abs(ch) ** 2
        p[0, 0, 0] = 0.0
        tot = p.sum()
        return float(p[kk > 0.75 * (n / 3.0)].sum() / tot) if tot > 0 else 0.0


def blob(x, zc):
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    rho = np.sqrt((X - np.pi) ** 2 + (Y - np.pi) ** 2)
    return np.exp(-(((rho - 1.0) ** 2 + (Z - zc - 0.25) ** 2) / (2 * 0.2**2)))


def extraction(u, h, z_guess):
    a = rc.axisym(u, h, z_guess)
    w = np.abs(a["omega_phi"]) ** 2
    zc = float(np.sum(w * a["z"][None, :]) / np.sum(w))
    return rc.axisym(u, h, zc), zc


def run_frozen(t_end=24.0):
    u4, _, t4, nu, h = rc.load(4)
    u5, _, t5, _, _ = rc.load(5)
    u6, _, t6, _, _ = rc.load(6)
    a4, _ = extraction(u4, h, np.pi + 0.5 * t4)
    a5, zc5 = extraction(u5, h, np.pi + 0.5 * t5)
    a6, _ = extraction(u6, h, np.pi + 0.5 * t6)
    speed = io.ring_speed(a4, t4, a6, t6)
    uf = u5.copy()
    uf[2] -= speed
    print(f"frozen flow: snapshot t={t5}, ring speed U={speed:.4f}, ring centroid z={zc5:.4f}", flush=True)

    psi, res = io.stream_function(a5["u_rho"], a5["u_z"], a5["rho"], a5["dr"], a5["dz"])
    psi_co = io.to_comoving(psi, a5["rho"], speed)
    core = np.unravel_index(np.argmax(np.abs(a5["omega_phi"])), a5["omega_phi"].shape)
    lab = (-1.0 if psi_co[core] > 0 else 1.0) * psi_co

    x = np.arange(N) * (2 * np.pi / N)
    sc = Scalar(N)
    chs = [np.fft.rfftn(blob(x, zc5)) * sc.dealias for _ in KAPPAS]
    steps = int(round(t_end / DT))
    times, slices, tails = [], [], []
    t0 = time.time()
    for s in range(steps + 1):
        if s % OUT_EVERY == 0:
            row, trow = [], []
            for ch in chs:
                c = np.fft.irfftn(ch, s=(N,) * 3)
                row.append(io.sample_axisym_scalar(c, h, CENTER, RHO_MAX, Z_HALF, z_center=zc5, **SL))
                trow.append(sc.tail_fraction(c))
            times.append(s * DT)
            slices.append(row)
            tails.append(trow)
            if (s // OUT_EVERY) % 6 == 0:
                print(f"  t={s*DT:5.1f} wall {time.time()-t0:5.0f}s tails {['%.1e' % v for v in trow]}", flush=True)
        if s == steps:
            break
        new = []
        for ch, kap in zip(chs, KAPPAS):
            k1 = sc.rhs(uf, ch, kap)
            k2 = sc.rhs(uf, ch + 0.5 * DT * k1, kap)
            k3 = sc.rhs(uf, ch + 0.5 * DT * k2, kap)
            k4 = sc.rhs(uf, ch + DT * k3, kap)
            new.append(ch + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
        chs = new
    np.savez_compressed("stage1_frozen.npz", times=np.array(times), kappas=np.array(KAPPAS), slices=np.array(slices),
                        tails=np.array(tails), psi=lab, rho=a5["rho"], z=a5["z"], dr=a5["dr"], dz=a5["dz"],
                        speed=speed, zc=zc5, psi_resid=res)
    print("saved stage1_frozen.npz", flush=True)


def run_full(t_end=10.0):
    sim = NS3D(N, NU)
    sc = Scalar(N)
    h = 2 * np.pi / N
    uh = sim.velocity_from_vorticity_field(ring_vorticity(sim))
    x = sim.x
    chs = [np.fft.rfftn(blob(x, np.pi)) * sc.dealias for _ in KAPPAS]
    steps = int(round(t_end / DT))
    times, slices, tails, urs, uzs, wps, zcs = [], [], [], [], [], [], []
    last_zc, last_speed = np.pi, 0.66
    t0 = time.time()
    for s in range(steps + 1):
        if s % OUT_EVERY == 0:
            u = np.fft.ifftn(uh, axes=(1, 2, 3)).real
            a, zc = extraction(u, h, last_zc + last_speed * (OUT_EVERY * DT if s else 0.0))
            if s:
                last_speed = (zc - last_zc) / (OUT_EVERY * DT)
            last_zc = zc
            row, trow = [], []
            for ch in chs:
                c = np.fft.irfftn(ch, s=(N,) * 3)
                row.append(io.sample_axisym_scalar(c, h, CENTER, RHO_MAX, Z_HALF, z_center=zc, **SL))
                trow.append(sc.tail_fraction(c))
            times.append(s * DT)
            slices.append(row)
            tails.append(trow)
            urs.append(a["u_rho"])
            uzs.append(a["u_z"])
            wps.append(a["omega_phi"])
            zcs.append(zc)
            if (s // OUT_EVERY) % 4 == 0:
                print(f"  t={s*DT:5.1f} wall {time.time()-t0:5.0f}s zc={zc:.3f} tails {['%.1e' % v for v in trow]}", flush=True)
        if s == steps:
            break
        k1 = sim.rhs(uh)
        k2 = sim.rhs(uh + 0.5 * DT * k1)
        k3 = sim.rhs(uh + 0.5 * DT * k2)
        k4 = sim.rhs(uh + DT * k3)
        u_s = [np.fft.ifftn(uh, axes=(1, 2, 3)).real,
               np.fft.ifftn(uh + 0.5 * DT * k1, axes=(1, 2, 3)).real,
               np.fft.ifftn(uh + 0.5 * DT * k2, axes=(1, 2, 3)).real,
               np.fft.ifftn(uh + DT * k3, axes=(1, 2, 3)).real]
        new = []
        for ch, kap in zip(chs, KAPPAS):
            c1 = sc.rhs(u_s[0], ch, kap)
            c2 = sc.rhs(u_s[1], ch + 0.5 * DT * c1, kap)
            c3 = sc.rhs(u_s[2], ch + 0.5 * DT * c2, kap)
            c4 = sc.rhs(u_s[3], ch + DT * c3, kap)
            new.append(ch + DT / 6 * (c1 + 2 * c2 + 2 * c3 + c4))
        chs = new
        uh = uh + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    a0 = rc.axisym(np.fft.ifftn(uh, axes=(1, 2, 3)).real, h, last_zc)
    np.savez_compressed("stage1_full.npz", times=np.array(times), kappas=np.array(KAPPAS), slices=np.array(slices),
                        tails=np.array(tails), u_rho=np.array(urs), u_z=np.array(uzs), omega_phi=np.array(wps),
                        zc=np.array(zcs), rho=a0["rho"], z0=a0["z"], dr=a0["dr"], dz=a0["dz"])
    print("saved stage1_full.npz", flush=True)


if __name__ == "__main__":
    {"frozen": run_frozen, "full": run_full}[sys.argv[1]]()
