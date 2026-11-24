"""Input adapters: real CFD output -> the regular-grid fields the diagnostics expect.

    load_npz / load_hdf5 / load_csv_points / read_vtk / write_vtk_structured_points
    points_to_grid                 scattered (unstructured) velocity samples -> regular grid (+ coverage mask)
    sample_axisymmetric            3D Cartesian field -> (rho,z) axisymmetric slice, phi-averaged, with honesty metrics
    ring_speed, to_comoving        co-moving frame for a translating structure
    stream_function                Stokes stream function psi from (u_rho, u_z) with a consistency residual
All functions return plain NumPy arrays / dicts. Axis conventions: array axes are (x, y, z); u has shape (3,nx,ny,nz).
"""
import numpy as np
from scipy import ndimage, interpolate


# ---------------------------------------------------------------- generic loaders
def load_npz(path, keys=("u",)):
    d = np.load(path)
    out = {k: d[k] for k in d.files}
    missing = [k for k in keys if k not in out]
    if missing:
        raise KeyError(f"{path}: missing {missing}; available {list(out)}")
    return out


def load_hdf5(path, u="u", components=None):
    """Read a velocity field from HDF5: either one dataset of shape (3,nx,ny,nz) or three scalar datasets."""
    import h5py
    with h5py.File(path, "r") as f:
        if components is None:
            arr = np.asarray(f[u])
        else:
            arr = np.stack([np.asarray(f[c]) for c in components])
    if arr.shape[0] != 3:
        raise ValueError(f"expected leading axis of length 3, got {arr.shape}")
    return arr


def load_csv_points(path, cols=("x", "y", "z", "u", "v", "w")):
    import pandas as pd
    df = pd.read_csv(path)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{path}: missing columns {missing}; has {list(df.columns)}")
    pts = df[list(cols[:3])].to_numpy(float)
    vel = df[list(cols[3:])].to_numpy(float)
    return pts, vel


def points_to_grid(points, values, axes, fill=np.nan):
    """Scattered samples -> regular grid by Delaunay linear interpolation.

    axes = (xs, ys, zs) 1D coordinate arrays. values: (npts,) or (npts,3).
    Returns (grid array, coverage mask); cells outside the convex hull of the samples are masked False.
    """
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    q = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    values = np.asarray(values, float)
    vals = values[:, None] if values.ndim == 1 else values
    interp = interpolate.LinearNDInterpolator(points, vals, fill_value=fill)
    g = interp(q).reshape(X.shape + (vals.shape[1],))
    mask = np.all(np.isfinite(g), axis=-1)
    g = np.moveaxis(g, -1, 0)
    return (g[0] if vals.shape[1] == 1 else g), mask


# ---------------------------------------------------------------- legacy VTK
class _Reader:
    def __init__(self, data):
        self.d, self.p = data, 0

    def line(self):
        e = self.d.index(b"\n", self.p)
        s = self.d[self.p:e].decode("ascii", "replace").strip()
        self.p = e + 1
        return s

    def token(self):
        d = self.d
        n = len(d)
        while self.p < n and d[self.p:self.p + 1].isspace():
            self.p += 1
        s = self.p
        while self.p < n and not d[self.p:self.p + 1].isspace():
            self.p += 1
        return d[s:self.p].decode("ascii", "replace")

    def array(self, count, dtype, binary):
        if binary:
            dt = np.dtype(dtype).newbyteorder(">")
            while self.d[self.p:self.p + 1] in (b"\n", b"\r", b" "):
                self.p += 1
            nb = count * dt.itemsize
            a = np.frombuffer(self.d, dtype=dt, count=count, offset=self.p).astype(dtype)
            self.p += nb
            return a
        vals = [float(self.token()) for _ in range(count)]
        return np.array(vals, dtype)


def read_vtk(path):
    """Legacy VTK reader: STRUCTURED_POINTS, RECTILINEAR_GRID, STRUCTURED_GRID, UNSTRUCTURED_GRID, POLYDATA.

    Returns dict(kind, dims, origin, spacing, axes, points, point_data{name: array}).
    Structured data are returned with array axes (x, y, z).
    """
    with open(path, "rb") as f:
        r = _Reader(f.read())
    r.line()
    r.line()
    binary = r.line().upper() == "BINARY"
    kind = r.line().split()[1].upper()
    out = {"kind": kind, "point_data": {}, "points": None, "dims": None, "origin": None, "spacing": None, "axes": None}
    npts = None
    dtmap = {"float": np.float32, "double": np.float64, "int": np.int32}
    while True:
        tok = r.token()
        if tok == "":
            break
        tk = tok.upper()
        if tk == "DIMENSIONS":
            out["dims"] = tuple(int(r.token()) for _ in range(3))
            npts = int(np.prod(out["dims"]))
        elif tk == "ORIGIN":
            out["origin"] = tuple(float(r.token()) for _ in range(3))
        elif tk in ("SPACING", "ASPECT_RATIO"):
            out["spacing"] = tuple(float(r.token()) for _ in range(3))
        elif tk.endswith("_COORDINATES") and tk[0] in "XYZ":
            n = int(r.token())
            dt = dtmap.get(r.token().lower(), np.float32)
            ax = r.array(n, dt, binary)
            out["axes"] = (out["axes"] or {})
            out["axes"]["XYZ".index(tk[0])] = ax
        elif tk == "POINTS":
            n = int(r.token())
            dt = dtmap.get(r.token().lower(), np.float32)
            out["points"] = r.array(3 * n, dt, binary).reshape(n, 3)
            npts = n
        elif tk == "POINT_DATA":
            npts = int(r.token())
        elif tk in ("VECTORS", "SCALARS"):
            name = r.token()
            dt = dtmap.get(r.token().lower(), np.float32)
            nc = 3
            if tk == "SCALARS":
                nc = 1
                nxt = r.token()
                if nxt.upper() == "LOOKUP_TABLE":
                    r.token()
                else:
                    nc = int(nxt)
                    if r.token().upper() == "LOOKUP_TABLE":
                        r.token()
            else:
                pass
            if tk == "SCALARS" and nc == 1:
                pass
            a = r.array(nc * npts, dt, binary)
            out["point_data"][name] = a.reshape(npts, nc) if nc > 1 else a
    if out["axes"] is not None:
        out["axes"] = tuple(out["axes"][i] for i in range(3))
    if out["dims"] is not None:
        nx, ny, nz = out["dims"]
        for k, a in list(out["point_data"].items()):
            shp = (nz, ny, nx) + ((a.shape[1],) if a.ndim == 2 else ())
            b = a.reshape(shp)
            b = np.transpose(b, (2, 1, 0) + ((3,) if a.ndim == 2 else ()))
            out["point_data"][k] = np.moveaxis(b, -1, 0) if a.ndim == 2 else b
    return out


def write_vtk_structured_points(path, u, h, origin=(0.0, 0.0, 0.0), name="velocity", binary=True):
    nx, ny, nz = u.shape[1:]
    arr = np.transpose(u, (3, 2, 1, 0)).reshape(-1, 3)  # x fastest
    with open(path, "wb") as f:
        f.write(b"# vtk DataFile Version 3.0\nvelocity\n")
        f.write(b"BINARY\n" if binary else b"ASCII\n")
        f.write(b"DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\nORIGIN {origin[0]} {origin[1]} {origin[2]}\n".encode())
        f.write(f"SPACING {h[0]} {h[1]} {h[2]}\nPOINT_DATA {nx * ny * nz}\n".encode())
        f.write(f"VECTORS {name} float\n".encode())
        if binary:
            f.write(arr.astype(">f4").tobytes() + b"\n")
        else:
            f.write(("\n".join(" ".join(f"{v:.9g}" for v in row) for row in arr) + "\n").encode())


# ---------------------------------------------------------------- axisymmetric extraction
def sample_axisymmetric(u, h, center, rho_max, z_half, nrho=160, nz=320, nphi=24, w=None, z_center=None, origin=0.0, mode="grid-wrap"):
    """Sample a periodic 3D Cartesian field on a (rho,z) half-plane about the axis through `center`=(cx,cy).

    u: (3,nx,ny,nz) with axis z the symmetry axis. Returns dict with phi-averaged u_rho, u_z, u_phi, omega_phi
    (omega from `w` if given, else curl of u by periodic central differences) and honesty metrics:
      nonaxisym  rms phi-variation of (u_rho,u_z) relative to their rms
      swirl      rms(u_phi)/rms(u_p)
    """
    import fd3d
    hh = (h,) * 3 if np.isscalar(h) else tuple(h)
    if w is None:
        w = fd3d.curl(u, hh, True)
    n = u.shape[1]
    dr = rho_max / nrho
    dz = 2 * z_half / nz
    rho = (np.arange(nrho) + 0.5) * dr
    zc = (hh[2] * n / 2 if z_center is None else z_center)
    zz = zc - z_half + (np.arange(nz) + 0.5) * dz
    phis = 2 * np.pi * np.arange(nphi) / nphi
    R, Z, P = np.meshgrid(rho, zz, phis, indexing="ij")
    ix = (center[0] + R * np.cos(P) - origin) / hh[0]
    iy = (center[1] + R * np.sin(P) - origin) / hh[1]
    iz = (Z - origin) / hh[2]
    coords = np.stack([ix, iy, iz])

    def samp(a):
        return ndimage.map_coordinates(a, coords, order=3, mode=mode)  # 'grid-wrap' for periodic boxes, 'nearest' for measured, bounded data

    c, s = np.cos(P), np.sin(P)
    ux, uy, uz = samp(u[0]), samp(u[1]), samp(u[2])
    wx, wy = samp(w[0]), samp(w[1])
    u_rho = ux * c + uy * s
    u_phi = -ux * s + uy * c
    w_phi = -wx * s + wy * c
    mean = lambda a: a.mean(axis=2)
    ur, uzm, uph, wph = mean(u_rho), mean(uz), mean(u_phi), mean(w_phi)
    var = np.sqrt(np.mean((u_rho - ur[..., None]) ** 2 + (uz - uzm[..., None]) ** 2))
    rms_p = np.sqrt(np.mean(ur**2 + uzm**2))
    swirl = np.sqrt(np.mean(uph**2)) / max(rms_p, 1e-300)
    return {"rho": rho, "z": zz, "dr": dr, "dz": dz, "u_rho": ur, "u_z": uzm, "u_phi": uph, "omega_phi": wph,
            "nonaxisym": float(var / max(rms_p, 1e-300)), "swirl": float(swirl)}


def sample_axisym_scalar(c, h, center, rho_max, z_half, nrho=120, nz=240, nphi=24, z_center=None, origin=0.0):
    """phi-average of a periodic 3D scalar on the same (rho,z) grid as sample_axisymmetric."""
    hh = (h,) * 3 if np.isscalar(h) else tuple(h)
    n = c.shape[0]
    dr = rho_max / nrho
    dz = 2 * z_half / nz
    rho = (np.arange(nrho) + 0.5) * dr
    zc = (hh[2] * n / 2 if z_center is None else z_center)
    zz = zc - z_half + (np.arange(nz) + 0.5) * dz
    phis = 2 * np.pi * np.arange(nphi) / nphi
    R, Z, P = np.meshgrid(rho, zz, phis, indexing="ij")
    coords = np.stack([(center[0] + R * np.cos(P) - origin) / hh[0], (center[1] + R * np.sin(P) - origin) / hh[1],
                       (Z - origin) / hh[2]])
    return ndimage.map_coordinates(c, coords, order=3, mode="grid-wrap").mean(axis=2)


def applicability(a, psi_residual=None, max_nonaxisym=0.05, max_swirl=0.10, max_psi_residual=0.05):
    """Gate for the axisymmetric pipeline. Returns (ok, reasons).

    Thresholds are conventions set from my own cases: simulated rings gave nonaxisym ~1e-3 and psi residual ~2e-3, analytic flows 1e-5;
    a measured experimental ring gave 0.33 and 45. The pipeline's numbers (core Hessian, shear, rates) must not be used when this returns False.
    """
    reasons = []
    if a["nonaxisym"] > max_nonaxisym:
        reasons.append(f"not axisymmetric: phi-variation is {a['nonaxisym']:.2f} of the poloidal rms (limit {max_nonaxisym})")
    if a["swirl"] > max_swirl:
        reasons.append(f"significant swirl: {a['swirl']:.2f} of the poloidal rms (limit {max_swirl})")
    if psi_residual is not None and psi_residual > max_psi_residual:
        reasons.append(f"stream function inconsistent with u_rho: residual {psi_residual:.2f} (limit {max_psi_residual})")
    return (not reasons), reasons


def stream_function(u_rho, u_z, rho, dr, dz):
    """psi(rho,z) = int_0^rho rho' u_z d rho' (u_z = psi_rho/rho, u_rho = -psi_z/rho); returns (psi, residual)."""
    r = rho[:, None]
    integrand = r * u_z
    psi = np.cumsum(integrand, axis=0) * dr - 0.5 * integrand * dr
    resid = np.sqrt(np.mean((u_rho + np.gradient(psi, dz, axis=1) / r) ** 2)) / max(np.sqrt(np.mean(u_rho**2)), 1e-300)
    return psi, float(resid)


def ring_speed(ax_a, t_a, ax_b, t_b):
    """Translation speed of a vortex ring along z from the shift of the |omega_phi| centroid in z."""
    def zc(ax):
        wgt = np.abs(ax["omega_phi"]) ** 2
        return float(np.sum(wgt * ax["z"][None, :]) / np.sum(wgt))
    return (zc(ax_b) - zc(ax_a)) / (t_b - t_a)


def to_comoving(psi, rho, speed):
    """Frame moving with +speed along z: u_z -> u_z - speed  <=>  psi -> psi - speed*rho^2/2."""
    return psi - 0.5 * speed * rho[:, None] ** 2
