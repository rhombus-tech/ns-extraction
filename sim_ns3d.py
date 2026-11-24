"""Pseudo-spectral 3D incompressible Navier-Stokes on [0,2pi)^3 (rotational form, 2/3 dealiasing, RK4).

Generates real (nonlinear, viscous, unsteady) simulation data to exercise the adapters and diagnostics.

    python3 sim_ns3d.py ring  N nu T          # axisymmetric viscous vortex ring, translating along +z
    python3 sim_ns3d.py tg    N nu T          # decaying Taylor-Green
Snapshots (u, dudt exact from the RHS) are written to data/<case>_NNN.npz
"""
import os
import sys
import time
import numpy as np


class NS3D:
    def __init__(self, n, nu):
        self.n, self.nu = n, nu
        self.x = np.arange(n) * (2 * np.pi / n)
        k1 = np.fft.fftfreq(n, d=1.0 / n)
        self.k = np.stack(np.meshgrid(k1, k1, k1, indexing="ij"))
        self.k2 = np.sum(self.k**2, axis=0)
        self.k2s = np.where(self.k2 == 0, 1.0, self.k2)
        kmax = n / 3.0
        self.dealias = (np.abs(self.k[0]) < kmax) & (np.abs(self.k[1]) < kmax) & (np.abs(self.k[2]) < kmax)

    def project(self, vh):
        kv = np.sum(self.k * vh, axis=0)
        return vh - self.k * kv / self.k2s

    def curl_h(self, vh):
        k = self.k
        return 1j * np.stack([k[1] * vh[2] - k[2] * vh[1], k[2] * vh[0] - k[0] * vh[2], k[0] * vh[1] - k[1] * vh[0]])

    def rhs(self, uh):
        u = np.fft.ifftn(uh, axes=(1, 2, 3)).real
        w = np.fft.ifftn(self.curl_h(uh), axes=(1, 2, 3)).real
        cross = np.stack([u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]])
        nh = np.fft.fftn(cross, axes=(1, 2, 3)) * self.dealias
        return self.project(nh) - self.nu * self.k2 * uh

    def step(self, uh, dt):
        k1 = self.rhs(uh)
        k2 = self.rhs(uh + 0.5 * dt * k1)
        k3 = self.rhs(uh + 0.5 * dt * k2)
        k4 = self.rhs(uh + dt * k3)
        return uh + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def velocity_from_vorticity_field(self, w):
        wh = self.project(np.fft.fftn(w, axes=(1, 2, 3)))
        uh = 1j * np.stack([
            self.k[1] * wh[2] - self.k[2] * wh[1],
            self.k[2] * wh[0] - self.k[0] * wh[2],
            self.k[0] * wh[1] - self.k[1] * wh[0]]) / self.k2s
        uh[:, self.k2 == 0] = 0
        return uh * self.dealias


def ring_vorticity(sim, gamma=3.0, r0=1.0, a=0.35):
    c = np.pi
    X, Y, Z = np.meshgrid(sim.x, sim.x, sim.x, indexing="ij")
    xs, ys, zs = X - c, Y - c, Z - c
    rho = np.sqrt(xs**2 + ys**2)
    wphi = gamma / (np.pi * a * a) * np.exp(-((rho - r0) ** 2 + zs**2) / (a * a))
    rs = np.maximum(rho, 1e-12)
    return np.stack([-wphi * ys / rs, wphi * xs / rs, np.zeros_like(wphi)])


def taylor_green(sim):
    X, Y, Z = np.meshgrid(sim.x, sim.x, sim.x, indexing="ij")
    u = np.stack([np.sin(X) * np.cos(Y) * np.cos(Z), -np.cos(X) * np.sin(Y) * np.cos(Z), np.zeros_like(X)])
    return np.fft.fftn(u, axes=(1, 2, 3)) * sim.dealias


def run(case, n, nu, t_end, dt=0.02, save_dt=0.5, outdir="data"):
    os.makedirs(outdir, exist_ok=True)
    sim = NS3D(n, nu)
    uh = sim.velocity_from_vorticity_field(ring_vorticity(sim)) if case == "ring" else taylor_green(sim)
    t, nxt, snap = 0.0, 0.0, 0
    t0 = time.time()
    while t <= t_end + 1e-9:
        if t >= nxt - 1e-9:
            u = np.fft.ifftn(uh, axes=(1, 2, 3)).real
            dudt = np.fft.ifftn(sim.rhs(uh), axes=(1, 2, 3)).real
            np.savez_compressed(os.path.join(outdir, f"{case}_{snap:03d}.npz"), u=u.astype(np.float32),
                                dudt=dudt.astype(np.float32), t=t, nu=nu, h=2 * np.pi / n)
            print(f"snapshot {snap} t={t:.2f} max|u|={np.max(np.abs(u)):.3f} wall {time.time()-t0:.0f}s", flush=True)
            snap += 1
            nxt += save_dt
        uh = sim.step(uh, dt)
        t += dt
    return snap


if __name__ == "__main__":
    case, n, nu, t_end = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    run(case, n, nu, t_end)
