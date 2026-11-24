"""U2: rerun of the EVOLVING ring-2 scalar at 128^3 (dt = 0.01), removing the 96^3 resolution failure. See PREREGISTERED_UNSTEADY_RING2_U2.md.

    python3 unsteady_ring2_128.py run       # writes unsteady2_full_128.npz (never touches the 96^3 files)
    python3 unsteady_ring2_128.py analyze   # mechanical application of the U2 criteria
Reuses unsteady_ring2.py (hashed, unchanged) with N = 128, DT = 0.01, output every 50 steps (0.5 time units).
"""
import json
import os
import sys
import numpy as np
import analyze_stage1 as a
import io_fields as io
import scalar_experiment as se
import unsteady_ring2 as u

OUT = "unsteady2_full_128.npz"


def run():
    if os.path.exists(OUT):
        sys.exit(f"{OUT} exists; refusing to overwrite")
    u.N = 128
    se.DT = 0.01
    se.OUT_EVERY = 50
    u.run_full()
    os.replace("unsteady2_full.npz", OUT)  # run_full writes its fixed name; the 96^3 copy is kept as unsteady2_full_96.npz
    print(f"moved to {OUT}", flush=True)


def analyze():
    df = np.load("unsteady2_frozen.npz")
    rows_f = u.rates_of(df, lambda i: a.frozen_labels(df))
    d9 = np.load("unsteady2_full_96.npz")
    d12 = np.load(OUT)

    def evolving(d):
        n = len(d["times"])
        dz = float(d["dz"])
        axes = [{"omega_phi": d["omega_phi"][i], "z": d["zc"][i] - a.Z_HALF + (np.arange(d["u_z"].shape[2]) + 0.5) * dz} for i in range(n)]
        sp = [io.ring_speed(axes[i - 1], d["times"][i - 1], axes[i], d["times"][i]) for i in range(1, n)]
        sp = [sp[0]] + sp
        return u.rates_of(d, lambda i: a.full_labels(d, i, sp))

    rows_96, rows_128 = evolving(d9), evolving(d12)
    flow_ok = bool(float(df["flow_tail"]) < u.FLOW_TAIL_LIMIT and float(np.max(d12["flow_tails"])) < u.FLOW_TAIL_LIMIT)
    print(f"flow resolution: frozen (96^3) tail {float(df['flow_tail']):.1e}; evolving 128^3 max tail {float(np.max(d12['flow_tails'])):.1e} "
          f"(96^3 was {float(np.max(d9['flow_tails'])):.1e}) -> {'OK' if flow_ok else 'NOT RESOLVED'}")
    qs = []
    for rf, r9, r12 in zip(rows_f, rows_96, rows_128):
        ok = rf["valid"] and r12["valid"]
        q = r12["rate"] / rf["rate"] if ok else None
        qs.append(q)
        sens = abs(r12["rate"] - r9["rate"]) / r12["rate"] if (r12["rate"] and r9["rate"]) else None
        print(f"kappa={rf['kappa']:.1e}: frozen {rf['rate']:.4f} | evolving 128^3 {r12['rate']:.4f} (valid {r12['valid']}, window {r12['window']}) "
              f"| evolving 96^3 {r9['rate']:.4f} | resolution sensitivity |128-96|/128 = {sens if sens is None else round(sens, 3)} | q = {q if q is None else round(q, 3)}")
    valid = [q for q in qs if q is not None]
    if not flow_ok or len(valid) < 2:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "PASS" if all(q < 0.95 for q in valid) else "FAIL"
    print(f"\nU2a: {verdict}  (valid q: {[round(q, 3) for q in valid]}; PASS iff flow resolved, >= 2 valid kappa, every q < 0.95)")
    print(f"Reported only (NOT a criterion): q in the old band [0.5, 0.9]? {[0.5 <= q <= 0.9 for q in valid]}")
    json.dump({"frozen": rows_f, "evolving_96": rows_96, "evolving_128": rows_128, "q": qs, "U2a": verdict, "flow_ok": flow_ok},
              open("unsteady2_results_128.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    {"run": run, "analyze": analyze}[sys.argv[1]]()
