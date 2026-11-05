# Water–nitrogen: inverse/sonic ablation

The earlier failures involved very large inverse actions of the physical-flux Jacobian K in admissible interior states, Newton directions with an extremely narrow descent region, and eventual CFL collapse. The native face flux and nonconservative correction both use that inverse. A separate hard sonic-activation switch produced a finite flux jump. This four-arm study tests replacing those inverse-based operations with primitive-state-jump viscosity and direct nonconservative path integration, smoothing the sonic policy alone, or combining both packages. Physical inputs, boundary equations and the implicit solver remain fixed.

The native baseline converged in 15/24 configurations. Keeping the old sonic law, the inverse-free package converged in 24/24; sonic-only converged in 18/24; the combined method converged in 24/24.

Among 9 baseline attempts without convergence within the declared limits and with all four arms available, inverse-free/old-sonic recovered 9, sonic-only recovered 3, and both recovered 9. In 0 of those configurations only the combined arm recovered. Regressions from a converged baseline were 0, 0 and 0, respectively. The inverse-free package recovered every unconverged baseline configuration in this tested set while retaining the old sonic law. Of these baseline failures, 1 occurred in the subsonic case; inverse-free/old-sonic recovered 1. The difficulty is therefore not confined to transonic expansion. These are conditional implementation-package effects, not proof of universal convergence or of the unique cause of every rejected step.

Published-profile screening passed for Native baseline: 13/15 converged roots; Inverse-free / old sonic: 14/24 converged roots; New sonic only: 14/18 converged roots; Inverse-free + new sonic: 14/24 converged roots. A recovered root can still be too diffusive or displaced on a coarse mesh. The field comparisons and unchanged profile limits test that separately. For the combined method, 3/12 constant configurations pass and 11/12 linear configurations pass. Even at 400 cells, its constant shock gas-temperature error is 1.08% against the 1% limit.

For the 400-cell linear shock case, all four variants converged. Their single warm observations were Native baseline: 744.7 s (535 accepted steps), Inverse-free / old sonic: 23.3 s (50 accepted steps), New sonic only: 14.3 s (45 accepted steps), Inverse-free + new sonic: 25.21 s (53 accepted steps). The full matrix below also includes the cases with similar runtime and every failed attempt.

Root convergence, physical branch, profile accuracy and runtime are separate. The reported warm times are one compilation-free observation per configuration. Failed runs show time until termination, not time to solution.

Callback-checked wall budgets are 1800 s for a warm solve and 1800 s for an excluded compile/load solve. Budget stops are reported separately from stagnation and iteration limits.

| Case | Reconstruction | Cells | Variant | Root / branch | Profile | Max RMS | Global | Accepted / rejected | Warm [s] | Timing meaning |
|---|---|---|---|---|---|---|---|---|---|---|
| subsonic | Constant | 50 | Native baseline | Pass | disagrees | 2.975e-10 | 5.01e-11 | 35 / 0 | 1.741 | time to solution |
| subsonic | Constant | 50 | Inverse-free / old sonic | Pass | disagrees | 2.841e-10 | 4.76e-11 | 35 / 0 | 1.269 | time to solution |
| subsonic | Constant | 50 | New sonic only | Pass | disagrees | 2.975e-10 | 5.01e-11 | 35 / 0 | 1.613 | time to solution |
| subsonic | Constant | 50 | Inverse-free + new sonic | Pass | disagrees | 2.841e-10 | 4.76e-11 | 35 / 0 | 1.326 | time to solution |
| shock | Constant | 50 | Native baseline | stagnation | not_converged | 1.784e+04 | 33 | 142 / 88 | 18.87 | time to termination |
| shock | Constant | 50 | Inverse-free / old sonic | Pass | disagrees | 2.501e-10 | 2.181e-11 | 35 / 0 | 1.255 | time to solution |
| shock | Constant | 50 | New sonic only | Pass | disagrees | 5.232e-10 | 7.243e-11 | 39 / 1 | 1.857 | time to solution |
| shock | Constant | 50 | Inverse-free + new sonic | Pass | disagrees | 2.486e-10 | 2.182e-11 | 35 / 0 | 1.207 | time to solution |
| adapted | Constant | 50 | Native baseline | max_steps | not_converged | 0.004094 | 0.0002694 | 2000 / 549 | 257.7 | time to termination |
| adapted | Constant | 50 | Inverse-free / old sonic | Pass | disagrees | 3.174e-10 | 5.33e-11 | 35 / 0 | 1.319 | time to solution |
| adapted | Constant | 50 | New sonic only | Pass | disagrees | 3.074e-10 | 3.452e-11 | 36 / 0 | 1.858 | time to solution |
| adapted | Constant | 50 | Inverse-free + new sonic | Pass | disagrees | 3.171e-10 | 5.301e-11 | 35 / 0 | 1.444 | time to solution |
| subsonic | Linear (Van Albada) | 50 | Native baseline | Pass | matches | 6.789e-10 | 7.713e-11 | 34 / 0 | 1.484 | time to solution |
| subsonic | Linear (Van Albada) | 50 | Inverse-free / old sonic | Pass | matches | 2.97e-10 | 3.149e-11 | 34 / 0 | 1.703 | time to solution |
| subsonic | Linear (Van Albada) | 50 | New sonic only | Pass | matches | 6.789e-10 | 7.713e-11 | 34 / 0 | 1.423 | time to solution |
| subsonic | Linear (Van Albada) | 50 | Inverse-free + new sonic | Pass | matches | 2.97e-10 | 3.149e-11 | 34 / 0 | 1.446 | time to solution |
| shock | Linear (Van Albada) | 50 | Native baseline | stagnation | not_converged | 17.72 | 0.4007 | 131 / 74 | 16.84 | time to termination |
| shock | Linear (Van Albada) | 50 | Inverse-free / old sonic | Pass | disagrees | 8.497e-10 | 6.888e-11 | 34 / 0 | 1.524 | time to solution |
| shock | Linear (Van Albada) | 50 | New sonic only | stagnation | not_converged | 20.99 | 0.4746 | 140 / 77 | 15.4 | time to termination |
| shock | Linear (Van Albada) | 50 | Inverse-free + new sonic | Pass | disagrees | 8.497e-10 | 6.888e-11 | 34 / 0 | 1.487 | time to solution |
| adapted | Linear (Van Albada) | 50 | Native baseline | Pass | matches | 2.846e-10 | 5.034e-11 | 512 / 126 | 92.37 | time to solution |
| adapted | Linear (Van Albada) | 50 | Inverse-free / old sonic | Pass | matches | 4.316e-10 | 7.203e-11 | 34 / 0 | 1.373 | time to solution |
| adapted | Linear (Van Albada) | 50 | New sonic only | Pass | matches | 3.681e-10 | 6.485e-11 | 508 / 125 | 85.69 | time to solution |
| adapted | Linear (Van Albada) | 50 | Inverse-free + new sonic | Pass | matches | 4.316e-10 | 7.203e-11 | 34 / 0 | 1.32 | time to solution |
| subsonic | Constant | 100 | Native baseline | stagnation | not_converged | 1.813e+06 | 1.827 | 137 / 92 | 26.76 | time to termination |
| subsonic | Constant | 100 | Inverse-free / old sonic | Pass | disagrees | 7.238e-10 | 9.27e-11 | 38 / 0 | 2.223 | time to solution |
| subsonic | Constant | 100 | New sonic only | stagnation | not_converged | 1.749e+06 | 1.762 | 123 / 88 | 23.52 | time to termination |
| subsonic | Constant | 100 | Inverse-free + new sonic | Pass | disagrees | 7.238e-10 | 9.27e-11 | 38 / 0 | 2.024 | time to solution |
| shock | Constant | 100 | Native baseline | stagnation | not_converged | 6.765e+05 | 3.891 | 140 / 91 | 27.84 | time to termination |
| shock | Constant | 100 | Inverse-free / old sonic | Pass | disagrees | 2.59e-10 | 1.574e-11 | 39 / 0 | 2.323 | time to solution |
| shock | Constant | 100 | New sonic only | stagnation | not_converged | 6.473e+05 | 3.722 | 162 / 97 | 28.37 | time to termination |
| shock | Constant | 100 | Inverse-free + new sonic | Pass | disagrees | 2.587e-10 | 1.568e-11 | 39 / 0 | 2.148 | time to solution |
| adapted | Constant | 100 | Native baseline | stagnation | not_converged | 2.214e+06 | 2.322 | 133 / 91 | 27.34 | time to termination |
| adapted | Constant | 100 | Inverse-free / old sonic | Pass | disagrees | 5.978e-10 | 1.061e-10 | 38 / 0 | 2.082 | time to solution |
| adapted | Constant | 100 | New sonic only | stagnation | not_converged | 2.16e+06 | 2.265 | 141 / 93 | 26.5 | time to termination |
| adapted | Constant | 100 | Inverse-free + new sonic | Pass | disagrees | 5.965e-10 | 1.06e-10 | 38 / 0 | 1.876 | time to solution |
| subsonic | Linear (Van Albada) | 100 | Native baseline | Pass | matches | 2.209e-10 | 2.116e-11 | 38 / 0 | 3.046 | time to solution |
| subsonic | Linear (Van Albada) | 100 | Inverse-free / old sonic | Pass | matches | 6.061e-10 | 6.182e-11 | 37 / 0 | 2.799 | time to solution |
| subsonic | Linear (Van Albada) | 100 | New sonic only | Pass | matches | 2.209e-10 | 2.116e-11 | 38 / 0 | 2.493 | time to solution |
| subsonic | Linear (Van Albada) | 100 | Inverse-free + new sonic | Pass | matches | 6.061e-10 | 6.182e-11 | 37 / 0 | 2.66 | time to solution |
| shock | Linear (Van Albada) | 100 | Native baseline | Pass | matches | 9.696e-10 | 5.349e-11 | 38 / 0 | 3.457 | time to solution |
| shock | Linear (Van Albada) | 100 | Inverse-free / old sonic | Pass | matches | 6.518e-10 | 3.702e-11 | 38 / 0 | 3.512 | time to solution |
| shock | Linear (Van Albada) | 100 | New sonic only | Pass | matches | 9.695e-10 | 5.349e-11 | 38 / 0 | 3.442 | time to solution |
| shock | Linear (Van Albada) | 100 | Inverse-free + new sonic | Pass | matches | 6.52e-10 | 3.703e-11 | 38 / 0 | 3.411 | time to solution |
| adapted | Linear (Van Albada) | 100 | Native baseline | Pass | matches | 3.017e-10 | 5.157e-11 | 38 / 0 | 3.045 | time to solution |
| adapted | Linear (Van Albada) | 100 | Inverse-free / old sonic | Pass | matches | 7.49e-10 | 1.295e-10 | 37 / 0 | 3.193 | time to solution |
| adapted | Linear (Van Albada) | 100 | New sonic only | Pass | matches | 3.017e-10 | 5.157e-11 | 38 / 0 | 2.966 | time to solution |
| adapted | Linear (Van Albada) | 100 | Inverse-free + new sonic | Pass | matches | 7.49e-10 | 1.295e-10 | 37 / 0 | 3.347 | time to solution |
| subsonic | Constant | 200 | Native baseline | Pass | matches | 3.439e-10 | 4.344e-11 | 42 / 0 | 5.73 | time to solution |
| subsonic | Constant | 200 | Inverse-free / old sonic | Pass | disagrees | 3.424e-10 | 3.477e-11 | 42 / 0 | 4.586 | time to solution |
| subsonic | Constant | 200 | New sonic only | Pass | matches | 3.439e-10 | 4.344e-11 | 42 / 0 | 5.22 | time to solution |
| subsonic | Constant | 200 | Inverse-free + new sonic | Pass | disagrees | 3.424e-10 | 3.477e-11 | 42 / 0 | 4.482 | time to solution |
| shock | Constant | 200 | Native baseline | Pass | disagrees | 8.657e-10 | 5.796e-11 | 46 / 1 | 9.185 | time to solution |
| shock | Constant | 200 | Inverse-free / old sonic | Pass | disagrees | 9.221e-10 | 4.325e-11 | 42 / 0 | 5.522 | time to solution |
| shock | Constant | 200 | New sonic only | Pass | disagrees | 2.342e-10 | 1.625e-11 | 43 / 0 | 6.289 | time to solution |
| shock | Constant | 200 | Inverse-free + new sonic | Pass | disagrees | 9.23e-10 | 4.325e-11 | 42 / 0 | 5.105 | time to solution |
| adapted | Constant | 200 | Native baseline | Pass | matches | 3.482e-10 | 5.297e-11 | 46 / 1 | 9.176 | time to solution |
| adapted | Constant | 200 | Inverse-free / old sonic | Pass | matches | 2.919e-10 | 4.864e-11 | 42 / 0 | 5.686 | time to solution |
| adapted | Constant | 200 | New sonic only | Pass | matches | 5.387e-10 | 8.279e-11 | 42 / 0 | 5.647 | time to solution |
| adapted | Constant | 200 | Inverse-free + new sonic | Pass | matches | 2.921e-10 | 4.866e-11 | 42 / 0 | 3.957 | time to solution |
| subsonic | Linear (Van Albada) | 200 | Native baseline | Pass | matches | 3.051e-10 | 2.82e-11 | 41 / 0 | 5.894 | time to solution |
| subsonic | Linear (Van Albada) | 200 | Inverse-free / old sonic | Pass | matches | 8.811e-10 | 1.122e-10 | 40 / 0 | 6.385 | time to solution |
| subsonic | Linear (Van Albada) | 200 | New sonic only | Pass | matches | 3.051e-10 | 2.82e-11 | 41 / 0 | 5.978 | time to solution |
| subsonic | Linear (Van Albada) | 200 | Inverse-free + new sonic | Pass | matches | 8.811e-10 | 1.122e-10 | 40 / 0 | 6.092 | time to solution |
| shock | Linear (Van Albada) | 200 | Native baseline | stagnation | not_converged | 54.47 | 0.5072 | 134 / 75 | 50.85 | time to termination |
| shock | Linear (Van Albada) | 200 | Inverse-free / old sonic | Pass | matches | 4.51e-10 | 1.889e-11 | 42 / 0 | 7.083 | time to solution |
| shock | Linear (Van Albada) | 200 | New sonic only | stagnation | not_converged | 23.48 | 0.2186 | 130 / 73 | 50.89 | time to termination |
| shock | Linear (Van Albada) | 200 | Inverse-free + new sonic | Pass | matches | 4.509e-10 | 1.889e-11 | 42 / 0 | 6.902 | time to solution |
| adapted | Linear (Van Albada) | 200 | Native baseline | stagnation | not_converged | 53.04 | 0.4939 | 144 / 78 | 56.93 | time to termination |
| adapted | Linear (Van Albada) | 200 | Inverse-free / old sonic | Pass | matches | 3.769e-10 | 6.045e-11 | 41 / 0 | 5.844 | time to solution |
| adapted | Linear (Van Albada) | 200 | New sonic only | stagnation | not_converged | 27.99 | 0.2607 | 141 / 76 | 49.88 | time to termination |
| adapted | Linear (Van Albada) | 200 | Inverse-free + new sonic | Pass | matches | 3.769e-10 | 6.045e-11 | 41 / 0 | 6.005 | time to solution |
| subsonic | Constant | 400 | Native baseline | Pass | matches | 6.2e-10 | 6.668e-11 | 45 / 0 | 11.36 | time to solution |
| subsonic | Constant | 400 | Inverse-free / old sonic | Pass | matches | 7.12e-10 | 6.882e-11 | 45 / 0 | 9.383 | time to solution |
| subsonic | Constant | 400 | New sonic only | Pass | matches | 6.2e-10 | 6.668e-11 | 45 / 0 | 9.442 | time to solution |
| subsonic | Constant | 400 | Inverse-free + new sonic | Pass | matches | 7.12e-10 | 6.882e-11 | 45 / 0 | 8.203 | time to solution |
| shock | Constant | 400 | Native baseline | stagnation | not_converged | 3593 | 1.419 | 209 / 100 | 153.8 | time to termination |
| shock | Constant | 400 | Inverse-free / old sonic | Pass | disagrees | 6.324e-10 | 2.16e-11 | 46 / 0 | 11.76 | time to solution |
| shock | Constant | 400 | New sonic only | Pass | matches | 9.586e-10 | 8.104e-11 | 106 / 17 | 68.22 | time to solution |
| shock | Constant | 400 | Inverse-free + new sonic | Pass | disagrees | 6.322e-10 | 2.16e-11 | 46 / 0 | 10.46 | time to solution |
| adapted | Constant | 400 | Native baseline | Pass | matches | 8.396e-10 | 1.242e-10 | 45 / 0 | 10.73 | time to solution |
| adapted | Constant | 400 | Inverse-free / old sonic | Pass | matches | 5.663e-10 | 9.254e-11 | 45 / 0 | 8.268 | time to solution |
| adapted | Constant | 400 | New sonic only | Pass | matches | 8.364e-10 | 1.244e-10 | 45 / 0 | 11.08 | time to solution |
| adapted | Constant | 400 | Inverse-free + new sonic | Pass | matches | 5.662e-10 | 9.252e-11 | 45 / 0 | 7.822 | time to solution |
| subsonic | Linear (Van Albada) | 400 | Native baseline | Pass | matches | 5.418e-10 | 5.336e-11 | 44 / 0 | 12.16 | time to solution |
| subsonic | Linear (Van Albada) | 400 | Inverse-free / old sonic | Pass | matches | 6.896e-10 | 7.76e-11 | 44 / 0 | 12.99 | time to solution |
| subsonic | Linear (Van Albada) | 400 | New sonic only | Pass | matches | 5.418e-10 | 5.336e-11 | 44 / 0 | 11.67 | time to solution |
| subsonic | Linear (Van Albada) | 400 | Inverse-free + new sonic | Pass | matches | 6.896e-10 | 7.76e-11 | 44 / 0 | 12.27 | time to solution |
| shock | Linear (Van Albada) | 400 | Native baseline | Pass | matches | 5.07e-10 | 3.453e-11 | 535 / 130 | 744.7 | time to solution |
| shock | Linear (Van Albada) | 400 | Inverse-free / old sonic | Pass | matches | 1.92e-10 | 5.949e-12 | 50 / 1 | 23.3 | time to solution |
| shock | Linear (Van Albada) | 400 | New sonic only | Pass | matches | 8.803e-10 | 2.663e-11 | 45 / 0 | 14.3 | time to solution |
| shock | Linear (Van Albada) | 400 | Inverse-free + new sonic | Pass | matches | 5.694e-10 | 1.605e-11 | 53 / 2 | 25.21 | time to solution |
| adapted | Linear (Van Albada) | 400 | Native baseline | Pass | matches | 6.013e-10 | 9.268e-11 | 44 / 0 | 12.18 | time to solution |
| adapted | Linear (Van Albada) | 400 | Inverse-free / old sonic | Pass | matches | 6.877e-10 | 1.202e-10 | 44 / 0 | 12.02 | time to solution |
| adapted | Linear (Van Albada) | 400 | New sonic only | Pass | matches | 5.998e-10 | 9.23e-11 | 44 / 0 | 11.91 | time to solution |
| adapted | Linear (Van Albada) | 400 | Inverse-free + new sonic | Pass | matches | 6.877e-10 | 1.202e-10 | 44 / 0 | 11.27 | time to solution |

All runs use independent prepared published-profile seeds, implicit backward Euler, colored AD/banded Newton and unchanged $10^{-9}$ absolute targets. No sequencing or multigrid is used.

The sonic policy maps to signed weights in the native operator and to viscosity magnitudes in the inverse-free operator. Therefore these are conditional package effects; the finite-jump mapping, NC path and individual regularization changes are not separately identified by four arms.

The new NC path can change nonconservative shock selection. Equal-slip viscosity is approximate, and stationary void contacts are diffused. Cold-start and perturbed-seed robustness are outside this campaign.

![subsonic__constant__p](outputs/report/plots/subsonic__constant__p.png)

![subsonic__constant__T_g](outputs/report/plots/subsonic__constant__T_g.png)

![subsonic__constant__residual](outputs/report/plots/subsonic__constant__residual.png)

![subsonic__muscl_van_albada__p](outputs/report/plots/subsonic__muscl_van_albada__p.png)

![subsonic__muscl_van_albada__T_g](outputs/report/plots/subsonic__muscl_van_albada__T_g.png)

![subsonic__muscl_van_albada__residual](outputs/report/plots/subsonic__muscl_van_albada__residual.png)

![supersonic_shock__constant__p](outputs/report/plots/supersonic_shock__constant__p.png)

![supersonic_shock__constant__T_g](outputs/report/plots/supersonic_shock__constant__T_g.png)

![supersonic_shock__constant__residual](outputs/report/plots/supersonic_shock__constant__residual.png)

![supersonic_shock__muscl_van_albada__p](outputs/report/plots/supersonic_shock__muscl_van_albada__p.png)

![supersonic_shock__muscl_van_albada__T_g](outputs/report/plots/supersonic_shock__muscl_van_albada__T_g.png)

![supersonic_shock__muscl_van_albada__residual](outputs/report/plots/supersonic_shock__muscl_van_albada__residual.png)

![supersonic_adapted__constant__p](outputs/report/plots/supersonic_adapted__constant__p.png)

![supersonic_adapted__constant__T_g](outputs/report/plots/supersonic_adapted__constant__T_g.png)

![supersonic_adapted__constant__residual](outputs/report/plots/supersonic_adapted__constant__residual.png)

![supersonic_adapted__muscl_van_albada__p](outputs/report/plots/supersonic_adapted__muscl_van_albada__p.png)

![supersonic_adapted__muscl_van_albada__T_g](outputs/report/plots/supersonic_adapted__muscl_van_albada__T_g.png)

![supersonic_adapted__muscl_van_albada__residual](outputs/report/plots/supersonic_adapted__muscl_van_albada__residual.png)
