"""Mechanical analysis for PREREGISTERED_MV2003.md (criteria fixed there). Usage: python3 analyze_mv2003.py [B|A] [tag_fine tag_coarse]"""
import sys, json
import numpy as np
from mv_sim import theory_cM, A0

TS = 6.460981094815063
R44 = 4.4 * A0
fa = json.load(open("fig4a_digitised.json"))
fb = json.load(open("fig4b_digitised.json"))
variant = sys.argv[1] if len(sys.argv) > 1 else "B"
tags = sys.argv[2:4] if len(sys.argv) > 3 else ["fine", "coarse"]


def load(tag):
    return json.load(open(f"result_{variant}_{tag}.json"))


def c_at(res, t):
    tt = np.array(res["times"]); cc = np.array(res["series"]["1.32"])
    return float(np.interp(t, tt, cc))


def snap_at(res, s, rr):
    r = np.array(res["r"]); f = np.array(res["snaps"][str(float(s))] if str(float(s)) in res["snaps"] else res["snaps"][str(s)])
    return float(np.interp(rr, r, f))


def dev_b(res):
    return np.array([c_at(res, x * TS) - c for x, c in fb["pts"]])


def dev_a(res, s):
    pts = [(x, c) for x, c in fa[f"t{s}"] if x >= 3.0]
    return np.array([snap_at(res, s, x * A0) - c for x, c in pts])


def slope_model(res):
    ts_ = np.linspace(2 * TS, 3 * TS, 40)
    cs = np.array([c_at(res, t) for t in ts_])
    return float(np.polyfit(np.log(ts_), np.log(cs), 1)[0])


def slope_data():
    p = [(x, c) for x, c in fb["pts"] if 2.0 <= x <= 3.0]
    return float(np.polyfit(np.log([x for x, _ in p]), np.log([c for _, c in p]), 1)[0])


fine, coarse = load(tags[0]), load(tags[1])
rep = {}
for name, res in (("fine", fine), ("coarse", coarse)):
    d = dev_b(res)
    rep[name] = dict(C1_rms=float(np.sqrt(np.mean(d**2))), C1_max=float(np.max(np.abs(d))),
                     C2={s: dict(rms=float(np.sqrt(np.mean(dev_a(res, s)**2))), mean=float(np.mean(dev_a(res, s)))) for s in (5, 10, 20)},
                     slope=slope_model(res))
base_b = np.array([theory_cM(R44, x * TS) - c for x, c in fb["pts"]])
base_a = {s: np.array([theory_cM(x * A0, s) - c for x, c in fa[f"t{s}"] if x >= 3.0]) for s in (5, 10, 20)}
sd = slope_data()
# C4
c4 = 0.0
for x, _ in fb["pts"]:
    c4 = max(c4, abs(c_at(fine, x * TS) - c_at(coarse, x * TS)))
for s in (5, 10, 20):
    for x, _ in fa[f"t{s}"]:
        c4 = max(c4, abs(snap_at(fine, s, x * A0) - snap_at(coarse, s, x * A0)))
f = rep["fine"]
C1 = f["C1_rms"] <= 0.05 and f["C1_max"] <= 0.10
C2 = all(f["C2"][s]["rms"] <= 0.12 and abs(f["C2"][s]["mean"]) <= 0.08 for s in (5, 10, 20))
C3 = abs(f["slope"] + 1.5) <= 0.2 and abs(f["slope"] - sd) <= 0.2
C4 = c4 <= 0.02
print(f"variant {variant}")
print(f"C1 Fig4b: model RMS {f['C1_rms']:.4f}, max {f['C1_max']:.4f}   (paper's closed form RMS {np.sqrt(np.mean(base_b**2)):.4f}, max {np.max(np.abs(base_b)):.4f})  -> {'PASS' if C1 else 'FAIL'}")
for s in (5, 10, 20):
    print(f"C2 Fig4a t={s:2d}s: model RMS {f['C2'][s]['rms']:.4f} mean {f['C2'][s]['mean']:+.4f}   (paper's closed form RMS {np.sqrt(np.mean(base_a[s]**2)):.4f} mean {np.mean(base_a[s]):+.4f})")
print(f"   -> C2 {'PASS' if C2 else 'FAIL'}")
print(f"C3 slope t/ts in [2,3]: model {f['slope']:+.3f}, digitised data {sd:+.3f}, theory -1.5  -> {'PASS' if C3 else 'FAIL'}")
print(f"C4 fine-vs-coarse max difference {c4:.4f} (limit 0.02) -> {'OK' if C4 else 'NOT OK'}")
verdict = "INCONCLUSIVE" if not C4 else ("PASS" if (C1 and C2 and C3) else "FAIL")
print("VERDICT (this variant):", verdict, "" if variant == "B" else "(secondary variant: does not decide the preregistered verdict)")
