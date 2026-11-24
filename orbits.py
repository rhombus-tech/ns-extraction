"""Orbit decomposition of an axisymmetric (rho,z) field: Omega = g(psi) + dOmega.

Orbit average uses the advective time measure: in a thin psi-shell the weight is rho*drho*dz
(dt = rho dl / |grad psi| on a closed streamline of the poloidal flow).
"""
import numpy as np
from scipy import ndimage
from scipy.signal import savgol_filter

DEFAULT_BINS = 160


def poloidal_velocity(psi, rho, dr, dz):
    """u_z = psi_rho/rho, u_rho = -psi_z/rho (Stokes stream function, rho > 0)."""
    psi_r = np.gradient(psi, dr, axis=0)
    psi_z = np.gradient(psi, dz, axis=1)
    return -psi_z / rho, psi_r / rho


def azimuthal_vorticity(u_rho, u_z, dr, dz):
    return np.gradient(u_rho, dz, axis=1) - np.gradient(u_z, dr, axis=0)


def closed_region(psi, mask, level):
    """Connected component of {psi < level} & mask containing the global minimum of psi (closed streamlines)."""
    cand = mask & (psi < level)
    lab, _ = ndimage.label(cand)
    i = np.unravel_index(np.argmin(np.where(cand, psi, np.inf)), psi.shape)
    return lab == lab[i]


def decompose(psi, rho, dr, dz, omega_over_rho, region, nbins=DEFAULT_BINS):
    w = np.where(region, rho * dr * dz, 0.0)
    vals = psi[region]
    edges = np.linspace(vals.min(), vals.max(), nbins + 1)
    idx = np.clip(np.digitize(psi, edges) - 1, 0, nbins - 1)
    num = np.bincount(idx[region], weights=(w * omega_over_rho)[region], minlength=nbins)
    den = np.bincount(idx[region], weights=w[region], minlength=nbins)
    good = den > 0
    centers = np.bincount(idx[region], weights=(w * psi)[region], minlength=nbins)[good] / den[good]
    g_bins = num[good] / den[good]
    g_cell = np.interp(psi, centers, g_bins)
    d_omega = np.where(region, omega_over_rho - g_cell, 0.0)
    return centers, g_bins, d_omega


def weighted_l2(f, rho, dr, dz, region):
    w = np.where(region, rho * dr * dz, 0.0)
    return float(np.sqrt(np.sum(w * f * f)))


def shear_profile(centers, g_bins):
    return np.gradient(g_bins, centers)


def smooth_shear(centers, g_bins, frac=0.12):
    """Savitzky-Golay smoothed g and shear on a uniform psi grid, plus a noise-floor estimate for the shear.

    Returns (psi_u, g_s, shear_s, shear_noise). The shear is 'resolved' only if max|shear_s| >> shear_noise.
    """
    n = len(centers)
    u = np.linspace(centers[0], centers[-1], n)
    gu = np.interp(u, centers, g_bins)
    w = max(7, (int(frac * n) // 2) * 2 + 1)
    du = u[1] - u[0]
    g_s = savgol_filter(gu, w, 3)
    sh = savgol_filter(gu, w, 3, deriv=1, delta=du)
    noise = float(np.std(gu - g_s))
    shear_noise = 2.0 * noise * np.sqrt(12.0) / (du * w**1.5)
    return u, g_s, sh, shear_noise


def shear_zero_order(centers, shear, lo=0.03, hi=0.3, min_r2=0.9):
    """Locate sign changes of the shear and estimate the zero order m from |shear| ~ |psi-psi0|^(m-1)."""
    out = []
    span = centers[-1] - centers[0]
    s = np.sign(shear)
    cands = []
    for i in np.where(s[:-1] * s[1:] < 0)[0]:
        cands.append(centers[i] - shear[i] * (centers[i + 1] - centers[i]) / (shear[i + 1] - shear[i]))
    # even-order shear zeros (m odd) touch zero without a sign change: look for deep local minima of |shear|
    a = np.abs(shear)
    scale = a.max()
    for i in range(2, len(a) - 2):
        if a[i] < 0.02 * scale and a[i] <= a[i - 1] and a[i] <= a[i + 1] and a[i] < a[i - 2] and a[i] < a[i + 2]:
            if all(abs(centers[i] - c) > 0.05 * span for c in cands):
                cands.append(centers[i])
    for psi0 in cands:
        d = np.abs(centers - psi0)
        sel = (d > lo * span) & (d < hi * span) & (np.abs(shear) > 0)
        if sel.sum() < 6:
            continue
        x, y = np.log(d[sel]), np.log(np.abs(shear[sel]))
        slope, icpt = np.polyfit(x, y, 1)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - np.sum((y - (slope * x + icpt)) ** 2) / ss_tot if ss_tot > 0 else 0.0
        m = 1 + slope
        if r2 >= min_r2 and 0.8 <= m <= 10:
            out.append((float(psi0), float(m)))
    return out


def orbit_frequency(psi, rho, dr, dz, region, nlevels=60):
    """Orbit angular frequency omega(psi) = 2 pi / T(psi), the quantity whose shear controls mixing across streamlines.

    T = closed contour integral of rho dl / |grad psi| along each streamline (the time to go round), from extracted contours.
    This needs one derivative less than differencing a cumulative area and is accurate to ~1e-3, which the SHEAR of omega requires.
    Returns (levels, omega). NOT the orbit-averaged vorticity g(psi): that is a property of the advected field, not of the flow.
    """
    import contourpy
    nr, nz = psi.shape
    rho1 = (np.arange(nr) + 0.5) * dr  # coordinate along axis 0 (rho, or x for planar flow)
    weight1 = rho[:, 0]  # rho for axisymmetric flow, 1 for planar flow
    zc = (np.arange(nz) + 0.5) * dz
    ps_all = psi[region]
    core = np.unravel_index(np.argmin(np.where(region, psi, np.inf)), psi.shape)
    core_pt = np.array([zc[core[1]], rho1[core[0]]])
    gr = np.gradient(psi, dr, axis=0)
    gz = np.gradient(psi, dz, axis=1)
    gmag = np.sqrt(gr**2 + gz**2)
    from scipy.interpolate import RegularGridInterpolator
    gi = RegularGridInterpolator((rho1, zc), gmag, bounds_error=False, fill_value=np.nan)
    gen = contourpy.contour_generator(zc, rho1, psi)
    levels = np.linspace(ps_all.min(), ps_all.max(), nlevels + 2)[1:-1]
    out_l, out_T = [], []
    for lv in levels:
        best = None
        for line in gen.lines(lv):
            if len(line) < 8 or np.hypot(*(line[0] - line[-1])) > 1e-9:
                continue
            x, y = line[:, 0], line[:, 1]
            # winding test: does this closed loop encircle the core?
            ang = np.unwrap(np.arctan2(y - core_pt[1], x - core_pt[0]))
            if abs(ang[-1] - ang[0]) > np.pi:
                best = line
                break
        if best is None:
            continue
        mid = 0.5 * (best[1:] + best[:-1])
        seg = np.hypot(*(best[1:] - best[:-1]).T)
        g = gi(np.stack([mid[:, 1], mid[:, 0]], axis=1))
        if np.any(~np.isfinite(g)) or np.any(g <= 0):
            continue
        out_l.append(lv)
        out_T.append(float(np.sum(np.interp(mid[:, 1], rho1, weight1) * seg / g)))
    return np.array(out_l), 2 * np.pi / np.array(out_T)


def core_mode(psi, rho, dr, dz, region, kappa, fit_fraction=0.12, n_levels=8):
    """Closed-form decay rate of the vortex-core mode (the slowest oscillatory mode of a smooth closed-streamline core).

    Near an elliptic centre psi ~ psi_c + (h1 x^2 + h2 y^2)/2 and the orbit frequency is omega(psi) ~ omega_c + omega'(psi - psi_c). In
    coordinates where the core flow is a pure rotation the potential is isotropic and the diffusion anisotropic; the fast rotation averages it
    (orbit-averaged diffusion A(psi) = <|grad psi|^2> = psi (h1 + h2)), so the isotropic complex-harmonic-oscillator result applies with the
    averaged diffusivity, giving the basis-independent
        Re lambda = sqrt( 2 kappa |omega'| (h1 + h2) )         (angular mode 1; h1 + h2 = Hessian trace, the core vorticity in planar flow).
    It scales as kappa^(1/2) even though the shear in psi is nonzero, because the cross-streamline diffusion vanishes at the centre.
    VALIDATION (formula / 2D operator): Gaussian vortex 1.02-1.05 (kappa 5e-5..1e-3); planar elliptical vortices, ellipticity 1..3: 1.03-1.04
    (kappa 3e-4) and, OUT OF SAMPLE, 1.02-1.04 at ellipticity 1.25, 2.5, 3 (kappa 1e-4); real rings: ring 1 1.05-1.12 and ring 2 1.09-1.12 for
    kappa 6e-4..5e-3, the excess shrinking roughly like sqrt(kappa) (finite-size correction); and the exact radial Lamb-Oseen passive-scalar
    operator (ratio -> 1 as alpha -> infinity, deficit ~ 5/sqrt(alpha)). It over-predicts at moderate kappa by up to ~12%.
    The earlier Cartesian-oscillator form (rate_cartesian_oscillator) is exact only at isotropy: it under-predicts by 9/18/30% at ellipticity
    1.5/2/3, and looked better on the rings only through cancellation with the finite-kappa excess.
    Returns None if the core is not elliptic.
    """
    nr, nz = psi.shape
    x1 = (np.arange(nr) + 0.5) * dr
    z1 = (np.arange(nz) + 0.5) * dz
    X, Z = np.meshgrid(x1, z1, indexing="ij")
    core = np.unravel_index(np.argmin(np.where(region, psi, np.inf)), psi.shape)
    xc, zc = x1[core[0]], z1[core[1]]
    radius = fit_fraction * np.sqrt(region.sum() * dr * dz / np.pi)
    m = (X - xc) ** 2 + (Z - zc) ** 2 < radius**2
    x, y, v = (X - xc)[m], (Z - zc)[m], psi[m]
    M = np.stack([np.ones_like(x), x, y, 0.5 * x * x, x * y, 0.5 * y * y], axis=1)
    c = np.linalg.lstsq(M, v, rcond=None)[0]
    h1, h2 = np.linalg.eigvalsh(np.array([[c[3], c[4]], [c[4], c[5]]]))
    if h1 <= 0:
        return None
    lev, om = orbit_frequency(psi, rho, dr, dz, region, nlevels=60)
    slope = abs(np.polyfit(lev[:n_levels], om[:n_levels], 1)[0])
    # Orbit-averaged diffusion: the fast core rotation averages the anisotropic diffusion, A(psi) = <|grad psi|^2> = psi (h1 + h2), which is the
    # isotropic problem with D -> kappa (h1 + h2) / (2 sqrt(h1 h2)); the result is basis independent and depends on the Hessian trace only:
    rate = np.sqrt(2 * kappa * slope * (h1 + h2))
    # the earlier Cartesian-oscillator form (min over the two excited states) is exact only at isotropy; kept for comparison
    rate_cartesian = np.cos(np.pi / 4) * np.sqrt(kappa * slope / 2) * min(3 * np.sqrt(h1) + np.sqrt(h2), np.sqrt(h1) + 3 * np.sqrt(h2))
    ecc = float(np.sqrt(h2 / h1))
    note = (f"ellipticity {ecc:.2f}; derived closed form (no fitted constant), tested to ellipticity 3; it over-predicts the 2D operator by 2-4% at small "
            "kappa and up to ~12% at moderate kappa on real rings (finite-size excess shrinking ~sqrt(kappa))")
    return {"rate": float(rate), "rate_cartesian_oscillator": float(rate_cartesian), "exponent": 0.5,
            "hessian_eigenvalues": (float(h1), float(h2)), "core_shear": float(slope),
            "core_frequency": float(om[0]), "core_position": (float(xc), float(zc)), "ellipticity": ecc, "accuracy_note": note}


def beta_from_order(m):
    return 2 * (m + 1) / (m + 2)
