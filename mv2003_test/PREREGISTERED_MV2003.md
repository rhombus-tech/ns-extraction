# Preregistered published-data test: 2D advection-diffusion vs measured dye decay in a vortex (Meunier & Villermaux 2003)

Written 2026-11-23 BEFORE any model-vs-data comparison was computed. Hashes in PREREGISTERED_MV2003.sha256.

## What is tested, and what is not
TESTED: the premise behind the guide's `true_rates` tool: the full 2D advection-diffusion operator, driven by the MEASURED vortex (circulation, core size, viscous growth) and the dye's molecular diffusivity, predicts the MEASURED decay of the maximum dye concentration, with no fitted parameter.
NOT TESTED: the core-mode closed form sqrt(2 kappa |omega'|(h1+h2)) (the blob was deliberately placed far from the core, r/a0 >= 2.5), and anything about real flows other than one laboratory vortex.
The regime is the interior-orbit (differential-rotation) regime, where the classical Rhines-Young / strip theory already applies; a pass therefore validates the operator-level premise on real data, not a new prediction. The paper's own closed form (their eq. 3.11) is reported as a baseline.

## Measured inputs (all from the published paper, hal-00014833)
Gamma = 14.2 cm^2/s; a0 = 0.3 cm; a^2 = a0^2 + 4 nu t with nu = 0.01 cm^2/s (water; their eq. 2.2, Fig. 2c); dye diffusivity D = 5e-6 cm^2/s; blob width s0 = 0.22 cm (their section 3).
Measured dye data: the paper's Fig. 4, digitised from the vector graphics of the PDF: `fig4a_digitised.json` (max concentration c/c0 against r/a0 at t = 5, 10, 20 s) and `fig4b_digitised.json` (c/c0 against t/t_s at r/a0 = 4.4, t_s = 6.461 s from their eq. 3.12).
Digitisation accuracy: axes calibrated from the plotted tick marks; the paper's own plotted theory line in Fig. 4b is reproduced to max 0.0012 in c/c0 (so calibration error is far below the criteria below).

## Models (`mv_sim.py`; solver verified BEFORE this file against the paper's closed form on a steady point vortex: worst deviation 0.0062 in c/c0)
B (PRIMARY): evolving Lamb-Oseen vortex, Omega(r,t) = Gamma/(2 pi r^2) (1 - exp(-r^2/a(t)^2)), a^2 = a0^2 + 4 nu t.
A (secondary, reported, does not affect the verdict): steady point vortex, the flow the paper's own theory assumes.
Initial dye: uniform c0, tangential width s0, radial extent r in [0.6, 2.4] cm (the paper does not give the radial extent for the Fig. 4 runs; this choice is fixed here; a sensitivity run with [0.4, 3.0] is reported but is not a criterion).
Numerics: polar grid, exact azimuthal advection, Crank-Nicolson radial diffusion, Strang splitting; fine run (2400 radial cells, 2048 modes, dt 0.02) and coarse run (1200, 1024, 0.04).

## Criteria (fixed now; thresholds were chosen knowing that the paper's own closed form has RMS 0.026 against Fig. 4b and RMS 0.037 / 0.094 / 0.091 against Fig. 4a at t = 5 / 10 / 20 s)
C1 (Fig. 4b, 36 points): model c_max(t) at r = 1.32 cm, t = (t/t_s) * 6.461 s: RMS deviation <= 0.05 AND max |deviation| <= 0.10.
C2 (Fig. 4a, each of t = 5, 10, 20 s, points with r/a0 >= 3.0): RMS deviation <= 0.12 AND |mean deviation| <= 0.08.
C3 (decay law after the mixing time): slope d ln c / d ln t of the model over t/t_s in [2, 3] within 0.2 of -1.5, AND within 0.2 of the slope of the digitised points in the same interval.
C4 (numerics): fine and coarse runs agree to <= 0.02 in c/c0 at every compared point; otherwise the verdict is INCONCLUSIVE.
VERDICT (model B): PASS if C1, C2, C3 all hold and C4 holds; FAIL if C4 holds and any of C1-C3 fails; INCONCLUSIVE if C4 fails.

## Caveats stated in advance
The dye data are read from a figure (not raw images); the measured vortex parameters are the paper's fits (a few percent); the dye diffusivity has a stated value without uncertainty; the blob's radial extent is assumed; the experiment is one vortex at one Peclet number (Gamma/D = 2.8e6), so it cannot test the kappa-exponent.
