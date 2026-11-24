"""Mechanical application of PREREGISTERED_STAGE1.md to stage1_frozen.npz and stage1_full.npz. No discretion."""
import json
import sys
import numpy as np
import io_fields as io
import orbits as ob
import rates

Z_HALF = 2.0
TAIL_LIMIT = 1e-6
WINDOW = (0.03, 0.3)
MIN_POINTS = 4


def region_of(lab):
    return ob.closed_region(lab, np.ones_like(lab, dtype=bool), 0.02 * lab.min())


def dc_norm(c, lab, R, dr, dz, region):
    _, _, dc = ob.decompose(lab, R, dr, dz, c, region, 160)
    return ob.weighted_l2(dc, R, dr, dz, region)


def measured_rate(times, norms):
    r = norms / norms[0]
    sel = (r <= WINDOW[1]) & (r >= WINDOW[0])
    if sel.sum() < MIN_POINTS:
        return None, None
    return float(-np.polyfit(times[sel], np.log(norms[sel]), 1)[0]), (float(times[sel][0]), float(times[sel][-1]))


def prediction(lab, R, dr, dz, kappa):
    region = region_of(lab)
    est = {bc: rates.reduced_rate(lab, R, dr, dz, region, kappa, 1, k_modes=(1,), bc=bc)["rate"]
           for bc in ("dirichlet", "neumann")}
    return est, float(np.sqrt(est["dirichlet"] * est["neumann"]))


def frozen_labels(d):
    R, _ = np.meshgrid(d["rho"], d["z"], indexing="ij")
    return d["psi"], R, float(d["dr"]), float(d["dz"])


def full_labels(d, i, speeds):
    rho = d["rho"]
    dr, dz = float(d["dr"]), float(d["dz"])
    nz = d["u_z"].shape[2]
    z = d["zc"][i] - Z_HALF + (np.arange(nz) + 0.5) * dz
    psi, _ = io.stream_function(d["u_rho"][i], d["u_z"][i], rho, dr, dz)
    psi_co = io.to_comoving(psi, rho, speeds[i])
    core = np.unravel_index(np.argmax(np.abs(d["omega_phi"][i])), d["omega_phi"][i].shape)
    lab = (-1.0 if psi_co[core] > 0 else 1.0) * psi_co
    R, _ = np.meshgrid(rho, z, indexing="ij")
    return lab, R, dr, dz


def verdict(rows, label):
    valid = [r for r in rows if r["valid"] and r["rate"] is not None]
    if len(valid) < 2:
        return "INCONCLUSIVE", "fewer than 2 valid kappa"
    ok = all(0.5 <= r["ratio"] <= 2.0 for r in valid)
    return ("PASS" if ok else "FAIL"), f"{len(valid)} valid kappa, ratios " + ", ".join(f"{r['ratio']:.2f}" for r in valid)


def scaling(rows):
    valid = [r for r in rows if r["valid"] and r["rate"] is not None]
    if len(valid) < 3:
        return "INCONCLUSIVE", "fewer than 3 valid kappa"
    lk = np.log([r["kappa"] for r in valid])
    sm = np.polyfit(lk, np.log([r["rate"] for r in valid]), 1)[0]
    sp = np.polyfit(lk, np.log([r["pred"] for r in valid]), 1)[0]
    return ("PASS" if abs(sm - sp) <= 0.15 else "FAIL"), f"slope measured {sm:.3f}, predicted {sp:.3f}"


def run_flow(name, d, get_labels, midpoint_index):
    times = d["times"]
    rows = []
    for j, kap in enumerate(d["kappas"]):
        norms = []
        for i in range(len(times)):
            lab, R, dr, dz = get_labels(i)
            norms.append(dc_norm(d["slices"][i, j], lab, R, dr, dz, region_of(lab)))
        norms = np.array(norms)
        rate, win = measured_rate(times, norms)
        tail = float(d["tails"][-1, j])
        row = {"kappa": float(kap), "tail_last": tail, "valid": tail < TAIL_LIMIT, "rate": rate, "window": win,
               "final_ratio": float(norms[-1] / norms[0]), "pred": None, "pred_bounds": None, "ratio": None}
        if rate is not None and row["valid"]:
            mid = midpoint_index(win)
            lab, R, dr, dz = get_labels(mid)
            est, geo = prediction(lab, R, dr, dz, float(kap))
            row.update(pred=geo, pred_bounds=est, ratio=rate / geo)
        rows.append(row)
        print(f"[{name}] kappa={kap:.1e}: tail(last)={tail:.1e} valid={row['valid']} final r={row['final_ratio']:.3f} "
              f"rate={rate if rate is None else f'{rate:.4f}'} window={win} "
              + (f"pred={row['pred']:.4f} {row['pred_bounds']} ratio={row['ratio']:.2f}" if row["pred"] else "(no prediction)"), flush=True)
    return rows


def main():
    out = {}
    df = np.load("stage1_frozen.npz")
    rows_f = run_flow("F", df, lambda i: frozen_labels(df), lambda win: 0)
    out["F"] = rows_f

    dfu = np.load("stage1_full.npz")
    n = len(dfu["times"])
    axes = []
    dz = float(dfu["dz"])
    for i in range(n):
        z = dfu["zc"][i] - Z_HALF + (np.arange(dfu["u_z"].shape[2]) + 0.5) * dz
        axes.append({"omega_phi": dfu["omega_phi"][i], "z": z})
    speeds = [io.ring_speed(axes[i - 1], dfu["times"][i - 1], axes[i], dfu["times"][i]) for i in range(1, n)]
    speeds = [speeds[0]] + speeds
    rows_s = run_flow("S", dfu, lambda i: full_labels(dfu, i, speeds),
                      lambda win: int(np.argmin(np.abs(dfu["times"] - 0.5 * (win[0] + win[1])))))
    out["S"] = rows_s

    f1, f1s = verdict(rows_f, "F1")
    f2, f2s = scaling(rows_f)
    s1, s1s = verdict(rows_s, "S1")
    print("\n=== PREREGISTERED CRITERIA ===")
    print(f"F1 (frozen rate):    {f1}  [{f1s}]")
    print(f"F2 (frozen scaling): {f2}  [{f2s}]")
    print(f"S1 (full flow rate): {s1}  [{s1s}]")
    if f1 == "FAIL":
        rule = "1: rate estimate falsified on the real ring flow -> remove from guide, change paper text"
    elif f1 == "PASS" and s1 == "FAIL":
        rule = "2: valid for frozen / quasi-steady flow only; discrepancy attributed to flow evolution (not tested further)"
    elif f1 == "PASS" and s1 == "PASS":
        rule = "3: supported on this flow family only (one Reynolds number, one geometry)"
    else:
        rule = "4: at least one criterion INCONCLUSIVE; reported as inconclusive, not as a pass"
    print(f"Pre-declared interpretation {rule}")
    out["verdict"] = {"F1": f1, "F2": f2, "S1": s1, "rule": rule}
    json.dump(out, open("stage1_results.json", "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
