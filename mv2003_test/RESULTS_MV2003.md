# Result of the preregistered published-data test (Meunier & Villermaux 2003) - run 2026-11-23

Registered criteria: PREREGISTERED_MV2003.md (hashes taken before the run; they match the files as run).

## Verdict for the registered model B (evolving Lamb-Oseen vortex, measured Gamma, a0, nu; D = 5e-6 cm^2/s): FAIL (by one criterion, by a small margin)

| Criterion | Model B | Limit | Outcome |
|---|---|---|---|
| C1 Fig. 4b, 36 points: RMS / max deviation in c/c0 | 0.033 / 0.075 | 0.05 / 0.10 | pass |
| C2 Fig. 4a, t = 5 / 10 / 20 s: RMS | 0.030 / 0.091 / 0.077 | 0.12 | pass |
| C2 mean deviation | -0.011 / -0.048 / -0.046 | 0.08 | pass |
| C3 late-time slope d ln c / d ln t, t/t_s in [2,3] | -1.289 (vs -1.5: 0.211 away; vs data -1.453: 0.164 away) | within 0.2 of -1.5 AND of data | **FAIL on the -1.5 part (0.211 > 0.2)** |
| C4 fine vs coarse numerics | 0.009 | 0.02 | OK |

By the rule fixed in advance, the verdict is FAIL. The criterion was not changed after seeing the result.

## Secondary and exploratory results (do not alter the verdict)
- Model A (steady point vortex, the flow the paper's own theory assumes): passes all criteria (C1 RMS 0.026, slope -1.431) and reproduces the paper's closed form (eq. 3.11) to within 0.001 in RMS against the data; the numerical solver therefore agrees with their near-exact strip theory.
- Against Fig. 4a the evolving-core model B is slightly closer to the measured points than the point-vortex theory at t = 10 and 20 s (RMS 0.091 vs 0.099, 0.077 vs 0.092) and the sign of the offset shrinks (mean -0.048 vs -0.060, -0.046 vs -0.066), but it flattens the late decay: slope -1.29 against -1.43 (A) and -1.45 (digitised data in the same window).
- The digitised slope itself moves with the window (-1.41 for t/t_s in [1.5,2.5], -1.45 for [2,3], -1.78 for [2.5,3] with 7 points), so the data cannot decide between A and B on slope alone: the A-B slope difference (0.14) is of the order of the data's own window-to-window variation.
- The result is independent of the assumed radial extent of the blob (three choices give identical RMS and slope).
- At r/a0 = 4.4 and t = 13-19 s the Lamb-Oseen core has grown to a = 0.78-0.93 cm, so the swirl at the measurement radius is 6-14% below the point-vortex value; that is why B decays more slowly than A in this window.

## What this does and does not show
Shows: with no fitted parameter, the full 2D advection-diffusion operator driven by the measured vortex reproduces the measured maximum dye concentration over a range from c/c0 = 1 down to about 0.2 to an RMS of about 0.03 (at one radius over time) and 0.03-0.09 (across radii), the same accuracy as the paper's own closed-form theory.
Does not show: the registered evolving-vortex model matching the late-time slope; anything about the core mode or the sqrt(2 kappa |omega'|(h1+h2)) closed form (the dye was far from the core); the kappa-exponent (one Peclet number); anything beyond one laboratory vortex. The measured data are read from a published figure, not raw images.

Files: result_B_fine.json, result_B_coarse.json, result_A_fine.json, result_A_coarse.json (hashes in RESULTS_MV2003.sha256).
