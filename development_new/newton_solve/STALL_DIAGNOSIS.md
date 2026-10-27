# Saved-state diagnostics of stalled direct solves

These diagnostics use the first saved repeat (index 0) in `results_comparison/summary.json`, the 201-node shared spatial problem, and the original initial state. No production code or solver settings were changed. `diagnose_stalls.py` reproduces the calculations; `results_investigation_diagnostics/diagnostics.json` records all values.

Run from the repository root:

```powershell
poetry run python -m development_new.newton_solve.diagnose_stalls
```

The script accepts `--source`, `--output`, and `--repeat-index`. Relative source/output paths are resolved beside the script, independent of the working directory. The defaults are `results_comparison/summary.json`, `results_investigation_diagnostics`, and repeat index 0. The source must have the `compare_solvers` summary format: case IDs `shock35`/`shock70`, `settings.points`, case definitions, and saved `x`/`root_success` values under `runs.newton`, `runs.slsqp`, and `runs.implicit`. The same zero-based repeat index is selected for all methods. This script performs diagnostics only; it does not rerun solver comparisons.

Reproduction uses the current shared problem code and settings to reconstruct the residual and initial field. The summary does not archive every physical/discretization setting or the operator implementation. To reproduce the values below, use the same code revision and baseline settings as the saved comparison; a summary from different equations or settings is insufficient by itself.

The results support a globalization/basin-of-convergence difficulty around choking. They do not provide evidence of an incorrect compressed Jacobian or inaccurate linear solve at these sampled states. They do not prove that every piecewise branch of the residual is correct.

The later [mathematical analysis](../../docs/MATHEMATICAL_STAGNATION_ANALYSIS.md)
goes beyond these local derivative checks. It isolates severe limiter curvature
in the saved `shock70` Newton direction and demonstrates independent flux
continuity/consistency defects. Matching AD derivatives does not establish
regularity across branch boundaries.

| Case and state | Residual RMS | Scaled Jacobian condition number | Minimum singular value | Maximum interior Mach | Full Newton direction positivity limit |
| --- | ---: | ---: | ---: | ---: | ---: |
| shock35, initial | 1.545e-3 | 6.815e3 | 8.197e-4 | 0.4860 | 2.948 |
| shock35, failed Newton | 6.021e-4 | 8.837e4 | 8.672e-5 | 1.0204 | 0.05411 |
| shock35, failed SLSQP | 4.994e-4 | 7.364e4 | 8.535e-5 | 1.0008 | 0.07428 |
| shock35, implicit root | 5.824e-9 | 4.361e4 | See JSON | 1.3952 | 2.404e5 |
| shock70, initial | 1.313e-3 | 4.429e3 | 1.104e-3 | 0.4860 | 1.632 |
| shock70, failed Newton | 8.906e-4 | 1.415e5 | 5.842e-5 | 1.0223 | 0.009116 |
| shock70, failed SLSQP | 8.671e-4 | 3.742e4 | 1.941e-4 | 1.0214 | 0.02537 |
| shock70, implicit root | 7.485e-9 | 2.787e4 | See JSON | 1.8401 | 5.009e4 |

The positivity limit is the largest fraction $\alpha$ for which every interior component of $z+\alpha d$ stays positive, where $Jd=-R$. Pressure near the throat limits all four failed-state Newton directions. The saved fields themselves are comfortably positive: minimum pressure is about 102–110 kPa, minimum velocity 110–119 m/s, and minimum enthalpy 250–251 kJ/kg. Thus the domain rejections are caused by the proposed directions, rather than an already invalid accepted field.

At all initial, failed, and implicit-converged states, the compressed scaled Jacobian matches dense AD of the complete steady residual to about $1.6\times10^{-16}$ relative Frobenius error. The relative residual of a dense solve of $Jd=-R$ is approximately $4\times10^{-14}$ to $2.3\times10^{-12}$ at the initial and failed states. Conditioning worsens near the failed states, but these linear systems are still solved accurately in double precision.

Central directional finite differences approach the AD derivative as the perturbation shrinks, before cancellation dominates. The best relative errors in the failed states are approximately $4.9\times10^{-6}$, $5.9\times10^{-6}$, $2.0\times10^{-4}$, and $5.8\times10^{-6}$, respectively, in the table order. The shock70 Newton state is especially strongly curved: shrinking the normalized-direction perturbation from $10^{-4}$ to $10^{-7}$ reduces the central error from roughly $7.9\times10^3$ to $2.0\times10^{-4}$.

For the shock70 failed Newton field, even the positive fraction $\alpha=10^{-3}$ increases the residual norm by a factor 14.26; $10^{-4}$ increases it by 1.91, and $10^{-6}$ still increases it slightly. A fraction of $10^{-7}$ finally gives a tiny reduction. The linear model predicts a reduction for every fraction, but describes the residual accurately only in a very narrow neighborhood. At the shock35 failed Newton field, $\alpha=10^{-2}$ increases the residual by 2.62, while $10^{-3}$ produces only a 0.05% decrease. This explains why repeated backtracking can consume the Newton trial budget with little useful progress.

All saved failed states have only a small supersonic patch near the throat, while the converged implicit solutions contain a developed supersonic region before the shock. The largest scaled residual component in every initial and failed state is the momentum equation at the first physical cell, $x=0.001$ m. A residual plateau should therefore not be interpreted as a localized shock residual alone.

The FVS implementation contains hard characteristic sign choices, a strict entropy-fix activation condition, a narrow global floor, reconstruction-limiter branches, and EOS-range fallback branches. Those features, combined with the large required field change from the fully subsonic initial state, are plausible contributors to the narrow useful Newton neighborhood. The saved-state tests establish the narrow neighborhood but do not isolate a single branch as its cause. Initial-fraction sweeps and implicit warm starts are needed to determine which approach reaches the desired root reliably.
