"""Out-of-sample test of 'a frozen-flow decay rate overstates an evolving flow's decay' on a SECOND ring (nu = 0.005). See PREREGISTERED_UNSTEADY_RING2.md.

    python3 unsteady_ring2.py frozen    # velocity frozen at snapshot 5 (t = 2.5) of data_nu005, in the ring frame
    python3 unsteady_ring2.py full      # scalar carried by the evolving flow, t = 10
    python3 unsteady_ring2.py analyze   # mechanical application of the preregistered criteria
Same procedure as scalar_experiment.py (Stage 1) except: nu = 0.005, kappa in {3e-2, 1e-2}, outputs unrelated to the ring-1 files.
"""
import json
import sys
import time
import numpy as np
import io_fields as io
import analyze_stage1 as a
import scalar_experiment as se
from sim_ns3d import NS3D, ring_vorticity

N = 96
NU = 0.005
KAPPAS = (3e-2, 1e-2)
DATA = "data_nu005"
FLOW_TAIL_LIMIT = 1e-6


def flow_tail(u):
    """Fraction of the kinetic energy at |k| > 0.75 n/3 (resolution of the velocity field)."""
    n = u.shape[-1]
    uh = np.fft.fftn(u, axes=(1, 2, 3))
    k1 = np.fft.fftfreq(n, d=1.0 / n)
    kk = np.sqrt(sum(g**2 for g in np.meshgrid(k1, k1, k1, indexing="ij")))
    p = np.sum(np.abs(uh) ** 2, axis=0)
    return float(p[kk > 0.75 * (n / 3.0)].sum() / p.sum())


def run_frozen(t_end=24.0):
    snaps = {i: np.load(f"{DATA}/ring_{i:03d}.npz") for i in (4, 5, 6)}
    h = float(snaps[5]["h"])
    ex = {i: se.extraction(snaps[i]["u"].astype(float), h, np.pi + 0.5 * float(snaps[i]["t"])) for i in snaps}
    tt = {i: float(snaps[i]["t"]) for i in snaps}
    speed = io.ring_speed(ex[4][0], tt[4], ex[6][0], tt[6])
    a5, zc5 = ex[5]
    u5 = snaps[5]["u"].astype(float)
    uf = u5.copy()
    uf[2] -= speed
    tail_u = flow_tail(u5)
    print(f"frozen flow: snapshot t={tt[5]}, ring speed U={speed:.4f}, ring centroid z={zc5:.4f}, flow tail {tail_u:.1e}", flush=True)
    psi, res = io.stream_function(a5["u_rho"], a5["u_z"], a5["rho"], a5["dr"], a5["dz"])
    psi_co = io.to_comoving(psi, a5["rho"], speed)
    core = np.unravel_index(np.argmax(np.abs(a5["omega_phi"])), a5["omega_phi"].shape)
    lab = (-1.0 if psi_co[core] > 0 else 1.0) * psi_co
    x = np.arange(N) * (2 * np.pi / N)
    sc = se.Scalar(N)
    chs = [np.fft.rfftn(se.blob(x, zc5)) * sc.dealias for _ in KAPPAS]
    steps = int(round(t_end / se.DT))
    times, slices, tails = [], [], []
    t0 = time.time()
    for s in range(steps + 1):
        if s % se.OUT_EVERY == 0:
            row, trow = [], []
            for ch in chs:
                c = np.fft.irfftn(ch, s=(N,) * 3)
                row.append(io.sample_axisym_scalar(c, h, se.CENTER, se.RHO_MAX, se.Z_HALF, z_center=zc5, **se.SL))
                trow.append(sc.tail_fraction(c))
            times.append(s * se.DT)
            slices.append(row)
            tails.append(trow)
            if (s // se.OUT_EVERY) % 6 == 0:
                print(f"  t={s * se.DT:5.1f} wall {time.time() - t0:5.0f}s tails {['%.1e' % v for v in trow]}", flush=True)
        if s == steps:
            break
        new = []
        for ch, kap in zip(chs, KAPPAS):
            k1 = sc.rhs(uf, ch, kap)
            k2 = sc.rhs(uf, ch + 0.5 * se.DT * k1, kap)
            k3 = sc.rhs(uf, ch + 0.5 * se.DT * k2, kap)
            k4 = sc.rhs(uf, ch + se.DT * k3, kap)
            new.append(ch + se.DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
        chs = new
    np.savez_compressed("unsteady2_frozen.npz", times=np.array(times), kappas=np.array(KAPPAS), slices=np.array(slices), tails=np.array(tails),
                        psi=lab, rho=a5["rho"], z=a5["z"], dr=a5["dr"], dz=a5["dz"], speed=speed, zc=zc5, psi_resid=res, flow_tail=tail_u)
    print("saved unsteady2_frozen.npz", flush=True)


def run_full(t_end=10.0):
    sim = NS3D(N, NU)
    sc = se.Scalar(N)
    h = 2 * np.pi / N
    uh = sim.velocity_from_vorticity_field(ring_vorticity(sim))
    chs = [np.fft.rfftn(se.blob(sim.x, np.pi)) * sc.dealias for _ in KAPPAS]
    steps = int(round(t_end / se.DT))
    times, slices, tails, urs, uzs, wps, zcs, ftails = [], [], [], [], [], [], [], []
    last_zc, last_speed = np.pi, 0.66
    t0 = time.time()
    for s in range(steps + 1):
        if s % se.OUT_EVERY == 0:
            u = np.fft.ifftn(uh, axes=(1, 2, 3)).real
            ar, zc = se.extraction(u, h, last_zc + last_speed * (se.OUT_EVERY * se.DT if s else 0.0))
            if s:
                last_speed = (zc - last_zc) / (se.OUT_EVERY * se.DT)
            last_zc = zc
            row, trow = [], []
            for ch in chs:
                c = np.fft.irfftn(ch, s=(N,) * 3)
                row.append(io.sample_axisym_scalar(c, h, se.CENTER, se.RHO_MAX, se.Z_HALF, z_center=zc, **se.SL))
                trow.append(sc.tail_fraction(c))
            times.append(s * se.DT)
            slices.append(row)
            tails.append(trow)
            urs.append(ar["u_rho"])
            uzs.append(ar["u_z"])
            wps.append(ar["omega_phi"])
            zcs.append(zc)
            ftails.append(flow_tail(u))
            if (s // se.OUT_EVERY) % 4 == 0:
                print(f"  t={s * se.DT:5.1f} wall {time.time() - t0:5.0f}s zc={zc:.3f} tails {['%.1e' % v for v in trow]} flow {ftails[-1]:.1e}", flush=True)
        if s == steps:
            break
        k1 = sim.rhs(uh)
        k2 = sim.rhs(uh + 0.5 * se.DT * k1)
        k3 = sim.rhs(uh + 0.5 * se.DT * k2)
        k4 = sim.rhs(uh + se.DT * k3)
        u_s = [np.fft.ifftn(uh, axes=(1, 2, 3)).real,
               np.fft.ifftn(uh + 0.5 * se.DT * k1, axes=(1, 2, 3)).real,
               np.fft.ifftn(uh + 0.5 * se.DT * k2, axes=(1, 2, 3)).real,
               np.fft.ifftn(uh + se.DT * k3, axes=(1, 2, 3)).real]
        new = []
        for ch, kap in zip(chs, KAPPAS):
            c1 = sc.rhs(u_s[0], ch, kap)
            c2 = sc.rhs(u_s[1], ch + 0.5 * se.DT * c1, kap)
            c3 = sc.rhs(u_s[2], ch + 0.5 * se.DT * c2, kap)
            c4 = sc.rhs(u_s[3], ch + se.DT * c3, kap)
            new.append(ch + se.DT / 6 * (c1 + 2 * c2 + 2 * c3 + c4))
        chs = new
        uh = uh + se.DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    ar0, _ = se.extraction(np.fft.ifftn(uh, axes=(1, 2, 3)).real, h, last_zc)
    np.savez_compressed("unsteady2_full.npz", times=np.array(times), kappas=np.array(KAPPAS), slices=np.array(slices), tails=np.array(tails),
                        u_rho=np.array(urs), u_z=np.array(uzs), omega_phi=np.array(wps), zc=np.array(zcs), flow_tails=np.array(ftails),
                        rho=ar0["rho"], z0=ar0["z"], dr=ar0["dr"], dz=ar0["dz"])
    print("saved unsteady2_full.npz", flush=True)


def rates_of(d, get_labels):
    out = []
    for j, kap in enumerate(d["kappas"]):
        norms = []
        for i in range(len(d["times"])):
            lab, R, dr, dz = get_labels(i)
            norms.append(a.dc_norm(d["slices"][i, j], lab, R, dr, dz, a.region_of(lab)))
        norms = np.array(norms)
        rate, win = a.measured_rate(d["times"], norms)
        tail = float(d["tails"][-1, j])
        out.append({"kappa": float(kap), "rate": rate, "window": win, "tail_last": tail, "valid": tail < a.TAIL_LIMIT and rate is not None,
                    "final_ratio": float(norms[-1] / norms[0])})
    return out


def analyze():
    df = np.load("unsteady2_frozen.npz")
    rows_f = rates_of(df, lambda i: a.frozen_labels(df))
    dfu = np.load("unsteady2_full.npz")
    n = len(dfu["times"])
    dz = float(dfu["dz"])
    axes = [{"omega_phi": dfu["omega_phi"][i], "z": dfu["zc"][i] - a.Z_HALF + (np.arange(dfu["u_z"].shape[2]) + 0.5) * dz} for i in range(n)]
    speeds = [io.ring_speed(axes[i - 1], dfu["times"][i - 1], axes[i], dfu["times"][i]) for i in range(1, n)]
    speeds = [speeds[0]] + speeds
    rows_s = rates_of(dfu, lambda i: a.full_labels(dfu, i, speeds))
    flow_ok = bool(float(df["flow_tail"]) < FLOW_TAIL_LIMIT and float(np.max(dfu["flow_tails"])) < FLOW_TAIL_LIMIT)
    print(f"flow resolution: frozen tail {float(df['flow_tail']):.1e}, evolving max tail {float(np.max(dfu['flow_tails'])):.1e} -> {'OK' if flow_ok else 'NOT RESOLVED'}")
    qs = []
    for rf, rs in zip(rows_f, rows_s):
        ok = rf["valid"] and rs["valid"]
        q = rs["rate"] / rf["rate"] if ok else None
        qs.append(q)
        print(f"kappa={rf['kappa']:.1e}: frozen rate {rf['rate']} window {rf['window']} valid {rf['valid']}; evolving rate {rs['rate']} window {rs['window']} valid {rs['valid']}; "
              f"q = evolving/frozen = {q if q is None else round(q, 3)}")
    valid = [q for q in qs if q is not None]
    if not flow_ok or len(valid) < 2:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "PASS" if all(0.5 <= q <= 0.9 for q in valid) else "FAIL"
    print(f"\nU1: {verdict}  (valid q: {[round(q, 3) for q in valid]}; PASS iff >=2 valid kappa, flow resolved, and every q in [0.5, 0.9])")
    if verdict == "PASS":
        rule = "1: the frozen-flow rate overstates the evolving decay by 10-100% in a second ring; still two rings, one geometry"
    elif verdict == "FAIL":
        rule = "2: the ring-1 ratio does not generalise; the guide says it was specific to that ring and gives both"
    else:
        rule = "3: inconclusive; the guide keeps the single-ring statement and says it is untested elsewhere"
    print(f"Pre-declared interpretation {rule}")
    json.dump({"frozen": rows_f, "evolving": rows_s, "q": qs, "U1": verdict, "rule": rule, "flow_ok": flow_ok}, open("unsteady2_results.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    {"frozen": run_frozen, "full": run_full, "analyze": analyze}[sys.argv[1]]()
