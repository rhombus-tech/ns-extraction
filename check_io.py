"""Tests for the input adapters (io_fields)."""
import os
import sys
import tempfile
import numpy as np
import io_fields as io


def abc(n):
    x = np.arange(n) * (2 * np.pi / n)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    A, B, C = 1.0, 0.7, 0.43
    return np.stack([A * np.sin(Z) + C * np.cos(Y), B * np.sin(X) + A * np.cos(Z), C * np.sin(Y) + B * np.cos(X)]), x


def analytic_axisym(n, swirl=0.0, m2=0.0, shift=0.0):
    """psi = rho^2 exp(-(rho^2+(z-zc)^2)/0.5); u_rho = -psi_z/rho, u_z = psi_rho/rho (+ optional swirl, m=2 term)."""
    h = 2 * np.pi / n
    x = np.arange(n) * h
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    cx = cy = np.pi
    xs, ys = X - cx, Y - cy
    rho = np.maximum(np.sqrt(xs**2 + ys**2), 1e-12)
    zs = Z - np.pi - shift
    E = np.exp(-(rho**2 + zs**2) / 0.5)
    psi = rho**2 * E
    u_rho = -(rho**2 * E * (-2 * zs / 0.5)) / rho
    u_z = (2 * rho * E + rho**2 * E * (-2 * rho / 0.5)) / rho
    phi = np.arctan2(ys, xs)
    u_rho = u_rho * (1 + m2 * np.cos(2 * phi))
    u_phi = swirl * rho * E
    c, s = xs / rho, ys / rho
    u = np.stack([u_rho * c - u_phi * s, u_rho * s + u_phi * c, u_z])
    return u, h, psi


def main():
    ok = True
    with tempfile.TemporaryDirectory() as d:
        u, _ = abc(16)
        h = 2 * np.pi / 16
        for binary in (True, False):
            p = os.path.join(d, f"t_{binary}.vtk")
            io.write_vtk_structured_points(p, u, (h,) * 3, binary=binary)
            r = io.read_vtk(p)
            back = r["point_data"]["velocity"]
            err = np.max(np.abs(back - u))
            good = back.shape == u.shape and err < 1e-6 and r["kind"] == "STRUCTURED_POINTS" and abs(r["spacing"][0] - h) < 1e-12
            ok &= good
            print(f"VTK structured points round trip ({'binary' if binary else 'ascii'}): max err {err:.1e} {'PASS' if good else 'FAIL'}")

        p = os.path.join(d, "t.npz")
        np.savez(p, u=u, h=h)
        good = np.array_equal(io.load_npz(p)["u"], u)
        ok &= good
        print(f"npz round trip {'PASS' if good else 'FAIL'}")
        try:
            io.load_npz(p, keys=("u", "nope"))
            good = False
        except KeyError:
            good = True
        ok &= good
        print(f"npz missing-key error reported {'PASS' if good else 'FAIL'}")

        import h5py
        p = os.path.join(d, "t.h5")
        with h5py.File(p, "w") as f:
            f["u"] = u
            for i, c in enumerate(("ux", "uy", "uz")):
                f[c] = u[i]
        good = np.array_equal(io.load_hdf5(p), u) and np.array_equal(io.load_hdf5(p, components=("ux", "uy", "uz")), u)
        ok &= good
        print(f"HDF5 round trip (single dataset and three components) {'PASS' if good else 'FAIL'}")

        rng = np.random.default_rng(0)
        pts = rng.uniform(0, 2 * np.pi, (30000, 3))
        vel = np.stack([np.sin(pts[:, 2]) + 0.43 * np.cos(pts[:, 1]), 0.7 * np.sin(pts[:, 0]) + np.cos(pts[:, 2]),
                        0.43 * np.sin(pts[:, 1]) + 0.7 * np.cos(pts[:, 0])], axis=1)
        import pandas as pd
        p = os.path.join(d, "t.csv")
        pd.DataFrame(np.hstack([pts, vel]), columns=list("xyzuvw")).to_csv(p, index=False)
        pts2, vel2 = io.load_csv_points(p)
        good = np.allclose(pts, pts2) and np.allclose(vel, vel2)
        ok &= good
        print(f"CSV point cloud round trip {'PASS' if good else 'FAIL'}")

        ax = np.linspace(0.4, 2 * np.pi - 0.4, 24)
        g, mask = io.points_to_grid(pts2, vel2, (ax, ax, ax))
        Xg, Yg, Zg = np.meshgrid(ax, ax, ax, indexing="ij")
        exact = np.stack([np.sin(Zg) + 0.43 * np.cos(Yg), 0.7 * np.sin(Xg) + np.cos(Zg), 0.43 * np.sin(Yg) + 0.7 * np.cos(Xg)])
        err = np.max(np.abs(g - exact)[:, mask])
        good = mask.mean() > 0.99 and err < 0.1
        ok &= good
        print(f"scattered -> grid: coverage {mask.mean():.3f}, max err {err:.3f} (30k random points) {'PASS' if good else 'FAIL'}")
        _, mask2 = io.points_to_grid(pts2[pts2[:, 0] < 3.0], vel2[pts2[:, 0] < 3.0], (ax, ax, ax))
        good = 0.2 < mask2.mean() < 0.7
        ok &= good
        print(f"coverage mask flags the unsampled region (fraction covered {mask2.mean():.2f}) {'PASS' if good else 'FAIL'}")

    n = 64
    u, h, psi_exact = analytic_axisym(n)
    a = io.sample_axisymmetric(u, h, (np.pi, np.pi), rho_max=2.0, z_half=2.0, nrho=100, nz=200, nphi=16)
    rho, z = a["rho"], a["z"]
    R, Z = np.meshgrid(rho, z, indexing="ij")
    zs = Z - np.pi
    E = np.exp(-(R**2 + zs**2) / 0.5)
    ur_ex = -(R**2 * E * (-2 * zs / 0.5)) / R
    uz_ex = (2 * R * E + R**2 * E * (-2 * R / 0.5)) / R
    e1 = max(np.max(np.abs(a["u_rho"] - ur_ex)), np.max(np.abs(a["u_z"] - uz_ex)))
    good = e1 < 2e-3 and a["nonaxisym"] < 1e-3 and a["swirl"] < 1e-6
    ok &= good
    print(f"cylindrical extraction (analytic axisymmetric flow): max err {e1:.1e}, nonaxisym {a['nonaxisym']:.1e}, swirl {a['swirl']:.1e} "
          f"{'PASS' if good else 'FAIL'}")
    psi, res = io.stream_function(a["u_rho"], a["u_z"], rho, a["dr"], a["dz"])
    psi_ex = R**2 * E
    e2 = np.max(np.abs(psi - psi_ex))
    good = e2 < 5e-3 and res < 5e-2
    ok &= good
    print(f"stream function reconstruction: max err {e2:.1e}, consistency residual {res:.1e} {'PASS' if good else 'FAIL'}")

    us, _, _ = analytic_axisym(n, swirl=0.5)
    a2 = io.sample_axisymmetric(us, h, (np.pi, np.pi), 2.0, 2.0, 100, 200, 16)
    um, _, _ = analytic_axisym(n, m2=0.5)
    a3 = io.sample_axisymmetric(um, h, (np.pi, np.pi), 2.0, 2.0, 100, 200, 16)
    good = a2["swirl"] > 0.1 and a3["nonaxisym"] > 0.1 and a3["nonaxisym"] > 100 * a["nonaxisym"]
    ok &= good
    print(f"honesty metrics: swirl flagged ({a2['swirl']:.2f}), non-axisymmetry flagged ({a3['nonaxisym']:.2f}) {'PASS' if good else 'FAIL'}")

    ua, _, _ = analytic_axisym(n, shift=0.0)
    ub, _, _ = analytic_axisym(n, shift=0.3)
    A = io.sample_axisymmetric(ua, h, (np.pi, np.pi), 2.0, 2.0, 100, 200, 16)
    B = io.sample_axisymmetric(ub, h, (np.pi, np.pi), 2.0, 2.0, 100, 200, 16)
    sp = io.ring_speed(A, 0.0, B, 0.5)
    good = abs(sp - 0.6) < 0.02
    ok &= good
    print(f"ring speed from centroid shift: {sp:.3f} (true 0.600) {'PASS' if good else 'FAIL'}")

    pc = io.to_comoving(psi, rho, 0.6)
    uz_new = np.gradient(pc, a["dr"], axis=0) / rho[:, None]
    uz_old = np.gradient(psi, a["dr"], axis=0) / rho[:, None]
    e3 = np.max(np.abs((uz_new - uz_old) + 0.6)[2:-2])
    good = e3 < 1e-6
    ok &= good
    print(f"co-moving shift: u_z changes by -U to {e3:.1e} {'PASS' if good else 'FAIL'}")
    print("ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
