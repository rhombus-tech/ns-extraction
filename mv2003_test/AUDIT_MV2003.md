# Audit of the MV2003 write-up (2026-11-23, after RESULTS_MV2003.md; the hashed result files are unchanged)

Findings, all exploratory (they do not change the registered verdict, FAIL on C3 for model B):

1. The registered verdict is FRAGILE with respect to an assumed input. The paper does not give the viscosity; nu = 0.01 cm^2/s (water, ~20 C) was assumed. Slope over t/t_s in [2,3] (coarse runs): nu = 0.008 -> -1.343 (would PASS C3), 0.010 -> -1.294 (registered, FAIL by 0.011 margin), 0.012 -> -1.245 (FAIL). 20% in nu is ~8-10 C in water.
2. The published D and s0 are not consistent with the paper's own Fig. 4a curves. Eq. 3.11 depends only on eps = D/s0^2. The plotted Fig. 4a curves correspond to eps = 7.67e-5 (D/s0^2, units 1/s), i.e. 0.74 of the published 1.033e-4; a least-squares fit of eq. 3.11 to the MEASURED Fig. 4a points gives 7.56e-5 (0.73). With the published values, the model (and eq. 3.11 itself) sits ~0.05 below the data at t = 10, 20 s.
3. The earlier statement "same accuracy as the authors' own closed form" compared with eq. 3.11 evaluated at the published parameters (RMS 0.026; 0.035/0.099/0.092), NOT with the curve they plot (RMS 0.030/0.076/0.076).
4. Fig. 4b time axis: the 36 points fit equally well (RMS 0.0258 / 0.0257) with (eps_published, t_s = 6.46 s) or (eps_fitted, t_s = 7.15 s) because the plotted curve is universal in t/t_s; which t_s the authors used is not stated. Model B with eps fitted to the Fig. 4a points and t_s = 7.15 s: 4b RMS 0.040, slope -1.269 (still fails C3), 4a RMS 0.034/0.078/0.065 with means +0.016/+0.023/-0.002.
5. Calibration claim: "plotted theory curve reproduced to 0.0012" is true for Fig. 4b only; for Fig. 4a the plotted curves match eq. 3.11 only after item 2.
6. A numerical slip in my own correction ("5 C") was fixed to 8-10 C.

Table (coarse numerics, tmax 22): model | Fig4b RMS | slope[2,3] | Fig4a (RMS, mean) at 5/10/20 s
- B, published, nu .01, t_s 6.461: 0.036 | -1.294 | (0.03,-0.01) (0.09,-0.044) (0.076,-0.044)
- B, eps_fit, t_s 7.15: 0.040 | -1.269 | (0.034,+0.016) (0.078,+0.023) (0.065,-0.002)
- B, nu .008: 0.032 | -1.343 | (0.031,-0.012) (0.092,-0.048) (0.081,-0.051)
- B, nu .012: 0.041 | -1.245 | (0.029,-0.009) (0.087,-0.039) (0.071,-0.035)
- A, published, t_s 6.461: 0.028 | -1.437 | (0.034,-0.014) (0.096,-0.055) (0.091,-0.063)
- A, eps_fit, t_s 7.15: 0.029 | -1.438 | (0.031,+0.013) (0.077,+0.012) (0.076,-0.024)

Conclusion: the operator reproduces the measured dye decay to about the accuracy that the published parameters allow (a ~26% uncertainty in D/s0^2 and an unreported viscosity). The late-time slope separates the evolving-core model (-1.25 to -1.34) from the steady point vortex (-1.44) more than it separates either from the data (-1.45, window-dependent), so the experiment does not settle whether the core spreading matters at r/a0 = 4.4.
