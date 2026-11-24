"""Mechanical application of PREREGISTERED_STAGE1_E1.md to stage1_e1_frozen.npz (functions reused from analyze_stage1)."""
import json
import sys
import numpy as np
import analyze_stage1 as a


def main():
    d = np.load("stage1_e1_frozen.npz")
    rows = a.run_flow("E1", d, lambda i: a.frozen_labels(d), lambda win: 0)
    f1, f1s = a.verdict(rows, "F1'")
    f2, f2s = a.scaling(rows)
    print("\n=== PREREGISTERED CRITERIA (E1, frozen flow, 128^3) ===")
    print(f"F1' (rate):    {f1}  [{f1s}]")
    print(f"F2' (scaling): {f2}  [{f2s}]")
    try:
        old = json.load(open("stage1_results.json"))["F"]
        print("\nresolution-sensitivity check (96^3 vs 128^3 measured rates):")
        for r96 in old:
            r128 = [r for r in rows if abs(r["kappa"] - r96["kappa"]) < 1e-12][0]
            if r96["rate"] and r128["rate"]:
                print(f"  kappa={r96['kappa']:.1e}: 96^3 rate {r96['rate']:.4f} (valid {r96['valid']}), "
                      f"128^3 rate {r128['rate']:.4f} (valid {r128['valid']})")
    except FileNotFoundError:
        pass
    if f1 == "FAIL":
        rule = "2: rate estimate falsified on the frozen real flow -> remove from guide"
    elif f2 == "FAIL":
        rule = "1: kappa-dependence not supported -> guide says order-of-magnitude only at Pe ~ 100-300, scaling not reproduced"
    elif f1 == "PASS" and f2 == "PASS":
        rule = "3: scaling supported on the frozen ring flow (one flow, one Reynolds number)"
    else:
        rule = "4: INCONCLUSIVE criterion reported as inconclusive"
    print(f"Pre-declared interpretation {rule}")
    json.dump({"rows": rows, "F1": f1, "F2": f2, "rule": rule}, open("stage1_e1_results.json", "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
