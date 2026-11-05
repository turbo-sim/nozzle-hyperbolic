# Two-fluid v2: single-grid inverse/sonic ablation

This study compares four numerical variants of the existing water–nitrogen
Elliot nozzle, retaining the physical storage, flux, EOS tables, interphase
closures, virtual mass, stabilization and physical boundary equations. The
baseline is the actual `TwoFluidModel` and implicit solver imported from `src/`.
No package source is changed. No grid sequencing, multigrid, RK methods,
continuation between configurations or imported numerical solutions are used.

There are 96 configurations: subsonic flow, internal shock and adapted
expansion; 50, 100, 200 and 400 physical cells; constant and linear MUSCL Van
Albada reconstruction; four variants. Every configuration starts independently
from the existing published primitive profile, with a common boundary-prepared
seed shared by the four methods. This is explicitly reference-seeded recovery,
not a uniform physical cold-start test.

## Four variants

| Variant | Flux/NC construction | Sonic policy |
|---|---|---|
| `baseline` | Unmodified package compatibility flux and $XK^{-1}\Delta F$ NC correction | Unmodified native law |
| `inverse_old_sonic` | Primitive-jump viscosity and direct $X\,dU$ path integration | Old signed weights converted to viscosity magnitudes $m_k=\lambda_k s_k$ |
| `sonic_only` | Native $K^{-1}$ face-flux construction and native NC formula | Continuous Harten magnitude in the signed-weight denominator |
| `both` | Same inverse-free construction as the second arm | Harten viscosity magnitudes, as in v1 with native boundary floors |

The inverse-free package changes both the face flux and NC discretization.
It does not separate those two changes, the primitive-jump choice, the analytic
storage bridge or the small-slip grouping from one another.

For full-slip states, write $P_k=r_k\ell_k/D_k$, with $z_k$ the acoustic
low-Mach multiplier (one on the other families). The native construction is

$$
F^* = \frac{F_L+F_R}{2}
-\frac12 K\left(\sum_k z_k s_kP_k\right)K^{-1}(F_R-F_L).
$$

The inverse-free construction is

$$
F^* = \frac{F_L+F_R}{2}
-\frac12 J_{\mathrm{policy}}\left(\sum_k z_km_kP_k\right)(U_R-U_L).
$$

The old scalar weight uses the native expansion predicate
$\lambda_{k,L}<0<\lambda_{k,R}$ and native background width
$\delta_0=10^{-6}a_{\mathrm{mix}}$. At a crossing its denominator is
$(\bar\lambda_k^2+\Delta\lambda_k^2)/(2\Delta\lambda_k)$; otherwise it is
$\max(|\bar\lambda_k|,\delta_0)$. The old signed weight is the mean speed divided
by this denominator. The native common material block retains its own
Harten-smoothed sign with the old background width.

The new policy uses $\delta=0.02a_{\mathrm{mix}}$ and

$$
H_\delta(\lambda)=
\begin{cases}
|\lambda|,&|\lambda|\ge\delta,\\
(\lambda^2+\delta^2)/(2\delta),&|\lambda|<\delta.
\end{cases}
$$

In `sonic_only`, $s_k=\lambda_k/H_\delta(\lambda_k)$. In `both`,
$m_k=H_\delta(\lambda_k)$; in `inverse_old_sonic`, $m_k=\lambda_ks_k^{\rm old}$.
This distinction is essential: the native operator consumes signed weights,
whereas the new state-jump operator consumes magnitudes. The experiment tests
implementable packages and their interaction, not one identical scalar
substitution in otherwise identical matrix contractions. In particular,
the old-law magnitude vanishes at zero speed, while $H_\delta(0)=\delta/2$.

All arms retain the case's low-Mach floors: one internally and zero at the
two boundary faces. The actual acoustic multiplier is
$\operatorname{clip}(|u_{\mathrm{mix}}|/a_{\mathrm{mix}},\text{floor},1)$;
a zero floor does not mean zero dissipation. Native physical boundary laws,
slip-blend thresholds 5/20 m/s and stabilization parameters are unchanged.

The inverse-free NC integral uses the straight path in
$U=(p,u_g,u_\ell,\alpha_g,h_g,h_\ell)$, with two-point Gauss quadrature. Both
half-interface contributions and the within-cell reconstructed path are
included. Constant reconstruction has an identically zero within-cell path.
The two mass NC rows vanish and the two momentum rows cancel by construction.

## Physical and numerical qualifications

The new NC treatment preserves the intended $F_x+XU_x$ balance. With independently
interpolated EOS tables, the literal native NC smooth limit is
$XK_{\mathrm{policy}}^{-1}K_{\mathrm{AD}}U_x$, where the tabulated flux derivative
need not equal the analytic policy matrix. This is not an algebraically
equivalent conditioning-only transformation. A primitive integration path also
specifies a nonconservative shock prescription; refinement and reference
profiles therefore remain separate gates from root convergence.

The small-slip inverse-free viscosity is a bounded grouping approximation. It
does not include every material Jordan-block term of an exact matrix absolute
value. The combined magnitude-based arm diffuses stationary void contacts;
the old-law magnitude $\lambda s$ instead vanishes at zero speed. V1's
manufactured smooth test measured approximately first order for constant and
1.6 order for linear reconstruction; full second order is not assumed here.

## Solver and timing protocol

All methods use coupled implicit backward Euler, colored AD, the banded Newton
solver, 20 Newton corrections, 20 backtracking halvings, CFL 2 growing by 1.2
to $10^6$, at most 2000 accepted steps and 64 unchanged-state stagnation steps.
The original fixed-scale volume-weighted transport RMS, boundary, EOS, global
and total-mass invariant targets are all $10^{-9}$.
The recorded campaign uses callback-checked wall budgets of 1800 s for both
excluded compile/load solves and warm solves. The warm cap was raised before
the first timed sample after an excluded native solve converged slowly.
A budget stop is a time-limited unresolved attempt,
reported separately from stagnation and the iteration limit.

One excluded complete solve per configuration compiles the required kernels.
These excluded jobs may run concurrently. Only after they finish, warm workers
run serially. Each warm worker loads the cached step/monitor kernels using a
discarded one-step preflight, resets the identical original seed and performs
one timed complete solve. Timing includes the native monitoring, accepted and
rejected work; setup, property preparation, cache loading, final independent
audits and output are excluded. JAX work is synchronized before stopping the
clock. Zero observed backend compile/load requests is required for the reported
warm timing. If a lazy request appears, that contaminated attempt is retained
and one cache-ready retry is allowed; it does not create a repeated sample set.

These are single observations, not statistical speed benchmarks. Failed solves
report time until failure, never time to solution. Different discretizations
need not have the same finite-grid root, so runtime ratios are not presented as
same-root speedups. Before/after Python CPU snapshots are supplementary evidence
of competing activity, not a guarantee of an idle machine.

## Reproduce

Run from the repository root using the existing environment and a fresh output
directory. `--jobs` controls excluded compilation only; warm timing is serial.

```powershell
.venv/Scripts/python.exe -B -m development.two_fluid_test_v2.scripts.audit kernels
.venv/Scripts/python.exe -B -u -m development.two_fluid_test_v2.scripts.run --output development/two_fluid_test_v2/outputs/new_campaign --jobs 3 --warm-budget 1800
.venv/Scripts/python.exe -B -m development.two_fluid_test_v2.scripts.audit saved --directory development/two_fluid_test_v2/outputs/new_campaign
.venv/Scripts/python.exe -B -m development.two_fluid_test_v2.scripts.report development/two_fluid_test_v2/outputs/new_campaign
```

`--phase compile` and `--phase warm` allow the two phases to be run separately;
use `--resume` only with the identical numerical source and campaign settings.
`--cache` selects an existing hash-addressed JAX cache without changing the
physical or numerical settings. Failures, complete states, histories, excluded
attempts and source archives remain under the campaign's `outputs/` directory.

The implementation is limited to `numerics.py`, `run.py` and small common
helpers. Independent checks, saved-data reporting and browser QA are separate.
The [HTML report](REPORT.html) is standalone and contains embedded data and
libraries; [RESULTS.md](RESULTS.md) and exported PNG/PDF figures are generated
from the same saved records.
