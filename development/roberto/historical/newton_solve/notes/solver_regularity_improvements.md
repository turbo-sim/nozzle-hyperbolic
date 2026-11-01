# LM/HYBR tests and Euler flux/MUSCL corrections

> Historical analysis of the pre-shared-solver equations and recorded experiments.
> Removed one-off drivers and their original commands are not current
> reproduction instructions. Read the [retained study](../README.md) and
> [offline report](../REPORT.html) for the preserved comparison. Current
> production flux variants are documented in the
> [flux studies](../../../../../development/flux_methods/README.md).

The flux and limiter defects are corrected, but the tested changes do **not**
make direct cold-start solves converge on the two shocked cases. Constant
reconstruction also fails with the original equations, so the remaining
failure is not isolated to MUSCL. Implicit marching continues to converge.

Work remains in `nozzle-hyperbolic` on `roberto-branch`; there is no nozzleX
integration. All comparisons use the original 201-point mesh, perfect-gas Air
backend, simple initial field, and common scaled residual RMS target $10^{-8}$.

## Tests performed before changing the numerical kernels

LM and HYBR were tested first on the unchanged Van Leer MUSCL equations, using
the exact JAX Jacobian and MINPACK initial trust-region factors 0.1, 1, and 10.
Neither method reached the root target on either shock case.

LM's best final RMS values among those settings were about $6.24\times10^{-5}$
and $2.90\times10^{-4}$. It frequently reported native success on relative
step or objective-progress criteria while the equations remained unsatisfied.
HYBR stagnated, or, with larger initial radii in the 35% case, requested an
invalid physical state. Invalid requests abort explicitly; no artificial
residual is supplied to MINPACK.

The original constant-reconstruction equations were then tested with LM,
HYBR, Newton, and SLSQP. All four missed the target on both shocks. Thus the
earlier MUSCL diagnosis identifies an obstruction at a sampled Newton state,
but is not a complete explanation of all global convergence failures.

`results_root_methods_before/` and `results_root_methods_constant_before/`
under `development/roberto/newton_solve/` preserve these measurements, native
messages, settings, states, histories, and source-file fingerprints.

## Corrected characteristic flux

The public `_jax_F_fvs_stadtke` now uses a regularized characteristic flux
with a conservative sonic-viscosity correction. Its original implementation,
including its function body, comments, and docstring, is retained under
`_jax_F_fvs_stadtke_legacy` for reproducibility. The implicit factory and driver
accept `flux_variant='regularized'` (new default) or `'legacy'`.

### Regular projectors at rest and sonic flow

With $C=\partial Q/\partial(p,u,h)$, conservative projectors are evaluated as

$$
\Pi_i=Cv_i\ell_iC^{-1}.
$$

The explicit thermodynamic expression for $C^{-1}$ avoids inverting
$K=\partial F/\partial(p,u,h)$, whose determinant vanishes at $u=0,\pm a$.
There is no division by $u$ or $u\pm a$ in the corrected projectors.

### Continuous sonic regularization with unchanged central transport

Let $\lambda_i$ be the averaged characteristic speeds. A strictly positive
width is formed without the historical strict sign-change activation:

$$
\delta_0=\texttt{delta\_factor}\,a_{\rm avg},\qquad
\delta_i=\delta_0+\tfrac12\operatorname{pos}_{\delta_0}
  (\lambda_i^R-\lambda_i^L),
$$

$$
\operatorname{pos}_{\delta_0}(d)
=\tfrac12\left(d+\sqrt{d^2+\delta_0^2}\right).
$$

Harten's regularized absolute value is

$$
\phi_i=
\begin{cases}
(\lambda_i^2+\delta_i^2)/(2\delta_i),&|\lambda_i|<\delta_i,\\
|\lambda_i|,&\text{otherwise}.
\end{cases}
\qquad s_i=\lambda_i/\phi_i.
$$

The numerical flux is

$$
\widehat F
=\tfrac12(F_L+F_R)
-\tfrac12\sum_i\Pi_i\left[
s_i(F_R-F_L)
+(\phi_i-s_i\lambda_i)(Q_R-Q_L)
\right].
$$

This preserves $\widehat F(U,U)=F(U)$, including inside the sonic band. The
regularization is $C^1$ where the EOS is smooth; Harten's band-edge formula is
not $C^2$. It has no finite jump at the historical entropy activation boundary.
For frozen linear equations the complete characteristic viscosity is exactly
$\phi_i$. Smoothing flux weights alone would instead reduce that viscosity to
$\lambda_i^2/\phi_i$.

Outside the regularization band, the additional conservative viscosity is
zero and the original intended characteristic upwinding is recovered. The
discretization deliberately changes inside the band. Arithmetic averaging
is not a general-EOS Roe linearization, and these properties do not establish
a full nonlinear entropy or update-positivity proof. The existing adaptive
physical-state guards remain necessary. Positive finite widths are required;
the regularized factory rejects zero/negative/nonfinite values. The legacy
variant still permits its historical zero-width configuration.

## MUSCL changes

The ordinary Van Leer limiter keeps its active formula. Its inactive
denominator is made safe **before division**, removing the reverse-AD NaNs
at flat or cancelling differences. This numerical safeguard does not remove
Van Leer's mathematical derivative kinks.

A separate opt-in `reconstruction='muscl_van_leer_smooth'` supplies a $C^2$
limited slope. For differences normalized by fixed pressure, velocity, and
enthalpy scales, let $t=\max(ab,0)$ and $s=a+b$:

$$
\sigma_\epsilon
=\frac{2t^3s}{(t^2+\epsilon^4)(s^2+\epsilon^2)},\qquad
\epsilon=\frac{\texttt{smooth\_limiter\_coefficient}}{(N-1)^2}.
$$

The comparison uses coefficient 1, hence $\epsilon=2.5\times10^{-5}$ at
201 points, with fixed reservoir scales $(p_0,a_0,h_0)$. The factory also
supports explicit `limiter_scales`. Fixed scales preserve the local stencil
and the colored Jacobian's validity.

The slope retains Van Leer's sign and componentwise reconstruction bounds,
is zero for opposite slopes, and preserves constant fields. It intentionally
damps nearly flat linear slopes. Global differentiability, exact TVD axis
zeros, and exact linear preservation at arbitrarily small slopes are
incompatible at the origin; smoothing cannot promise all three. Finite
$\epsilon$ therefore changes the discrete solution, which is audited against
its own implicit reference. This is limited piecewise-linear MUSCL, not
unlimited central reconstruction.

The limiter's $C^2$ property does not make every solver operation globally
smooth: boundary clamps and the existing EOS-range fallback remain. Other
historical limiter options have not been rewritten. Normal float64 widths
are used; arbitrarily subnormal smoothing parameters are not a supported
numerical regime.

## Results after the corrections

The following are final residual RMS values with initial trust factor 0.1.
LM/HYBR receive up to 1000 residual requests; Newton has 200 trial attempts
and SLSQP 200 major iterations. Those budgets count different work units.

| Reconstruction | Shock location | LM | HYBR | Newton | SLSQP |
|---|---|---:|---:|---:|---:|
| Constant | 35% | $5.28\times10^{-5}$ | $4.24\times10^{-4}$ | $4.92\times10^{-4}$ | $4.37\times10^{-4}$ |
| Constant | 70% | $2.73\times10^{-4}$ | $4.68\times10^{-4}$ | $8.25\times10^{-4}$ | $8.16\times10^{-4}$ |
| Van Leer MUSCL | 35% | $6.24\times10^{-5}$ | $2.98\times10^{-4}$ | $6.02\times10^{-4}$ | $4.99\times10^{-4}$ |
| Van Leer MUSCL | 70% | $2.89\times10^{-4}$ | $4.71\times10^{-4}$ | $8.91\times10^{-4}$ | $8.69\times10^{-4}$ |
| Smooth MUSCL | 35% | $6.24\times10^{-5}$ | $3.07\times10^{-4}$ | $5.89\times10^{-4}$ | $4.93\times10^{-4}$ |
| Smooth MUSCL | 70% | $2.95\times10^{-4}$ | $4.71\times10^{-4}$ | $8.87\times10^{-4}$ | $8.73\times10^{-4}$ |

All direct runs in this table miss $10^{-8}$. Every corresponding implicit
reference reaches it. These are failures, including native LM success above
tolerance. Budget/domain aborts return an explicitly tagged best evaluated
field, never a claimed converged final iterate.

Two additional smooth-MUSCL probes use $z=\exp(y)$ with the exact chain
Jacobian $J_y=J_z\operatorname{diag}(z)$. This preserves the equations and
positive variables while changing trust-region geometry. Neither cold log
coordinates nor an actual 20-update implicit prefix followed by LM/HYBR
reaches the target within the tested 300-request budget. After that prefix,
LM reaches $4.87\times10^{-5}$ and $1.87\times10^{-4}$; HYBR reaches
$1.32\times10^{-4}$ and $4.99\times10^{-4}$. Thus this short prefix is not
an established remedy either.

No direct-solve speedup is claimed. Broad-sweep archives use each wrapper's
local solve timer, whose construction scope differs between backends. The
subsequent log/prefix probes and current runner use enclosing timers and
aligned history origins. Final audits, plotting, and output are outside solve
times. All timed solve calls were serial with one-thread BLAS. Timing
differences between separate batches are not evidence of acceleration.

## What LM's terminal fields show

A separate diagnostic examines the corrected **constant** reconstruction,
isolating it from MUSCL. It reproduces the archived LM residuals exactly.

| Quantity | Shock 35% | Shock 70% |
|---|---:|---:|
| Jacobian condition number | $1.58\times10^5$ | $3.02\times10^4$ |
| Numerical rank | 597/597 | 597/597 |
| $\|J^TR\|/(\|J\|_2\|R\|)$ | $7.95\times10^{-6}$ | $3.96\times10^{-5}$ |
| Residual energy in weakest left singular mode | 99.99992% | 99.96733% |
| Maximum Mach | 1.0476 | 1.1193 |
| Throat Mach | 0.9725 | 0.9062 |

The residual lies almost entirely in a weak local mode. The current fields
have not developed the supersonic region and shock seen in the implicit
solutions. This supports weak least-squares geometry and progress stagnation.
It is **not** proof of an exact non-root stationary minimum or a continuum
fold: both sampled Jacobians have full numerical rank and nonzero gradients.
MINPACK stopped on relative objective/step progress, not its orthogonality
criterion. Scalar damping, trust-region changes, and regularity repairs have
not supplied a reliable global path from this initial field.

The appropriate remaining direction is pseudo-transient/parameter
continuation or a defensible improved initial field, with final convergence
audited against the intended spatial residual. The current results do not
justify replacing implicit marching with LM/HYBR for these cases.

## Verification and outputs

New tests cover flux consistency at rest/sonic/reverse states for ideal and
stiffened-gas EOS, threshold continuity, forward/reverse AD and finite
differences, supersonic upwinding, frozen-linear viscosity, limiter bounds and
derivatives, and independent dense/colored/banded operator checks for all three
reconstructions. They also check invalid regularization settings and current
four/five-return API parity.

The full suite reports **62 passed, 3 expected failures, and 1 unchanged
failure**. All 37 new tests pass. The remaining unit failure is
`tests/test_fvs.py::test_lowmach_flux_dissipation`: its old mock EOS omits
`T`, which the separate two-fluid `phase_props` implementation requires. That
fixture was left untouched. The log is
`../poetry_setup_audit/full_tests_after_flux_fix.log`.

The five-regime steady baseline converges in every regime after these changes.
Supersonic, both shock cases, and fully subsonic accuracy checks pass. The
pre-existing exactly-sonic/subsonic accuracy gap remains: normalized Mach L1
error about 0.03119 against a 0.03 threshold, with numerical throat Mach 0.8464
versus the reference 1.0. Consequently that baseline script still exits nonzero;
this is not presented as an all-green analytical validation.

Results live under `development/roberto/newton_solve/results_root_methods_*`:
`summary.json`, `timings.csv`, raw history CSVs, and PNG/PDF residual/profile
plots. LM/HYBR histories are **all residual evaluations**, including duplicate
and unsuccessful trial requests, because MINPACK has no accepted-iteration
callback. Newton/SLSQP/implicit histories are accepted updates. Plots label
this distinction, mark the authoritative returned point, and label aborted
best-evaluated fields explicitly. Positive-variable and warm-prefix runs keep
their actual phase costs and separate histories. Generated results remain
ignored by Git; the scripts and this report are preserved.

Independent post-processing checked all 66 recorded solves (52 direct
attempts and 14 implicit references): root/status logic, exact request counts,
monotone timestamps, enclosing costs where recorded, and independently
recomputed differences from each reconstruction's implicit root.

## Commands

Run from the repository root:

```powershell
# Historical residual: LM/HYBR first, ordinary MUSCL, three trust factors.
poetry run python -m development.roberto.newton_solve.compare_root_methods --stage before --no-show

# Historical residual: isolate constant reconstruction.
poetry run python -m development.roberto.newton_solve.compare_root_methods --stage constant_before --reconstructions constant --methods lm hybr newton slsqp --factors 0.1 --no-show

# Corrected flux: compare constant and both piecewise-linear reconstructions.
poetry run python -m development.roberto.newton_solve.compare_root_methods --stage after --reconstructions constant muscl_van_leer muscl_van_leer_smooth --methods lm hybr newton slsqp --factors 0.1 --no-show

# Positive coordinates, then the tested short actual warm start.
poetry run python -m development.roberto.newton_solve.compare_root_methods --stage after_log --reconstructions muscl_van_leer_smooth --coordinates log --factors 0.1 --max-evaluations 300 --no-show
poetry run python -m development.roberto.newton_solve.compare_root_methods --stage warm20_log --reconstructions muscl_van_leer_smooth --coordinates log --factors 0.1 --warm-steps 20 --max-evaluations 300 --no-show

# Redraw saved observations without rerunning solvers.
poetry run python -m development.roberto.newton_solve.compare_root_methods --stage after --plot-only --no-show
```

`--flux-variant` overrides the stage selection; `before`/`constant_before`
default to legacy, other stages to regularized. `--delta-factor` and
`--smooth-limiter-coefficient` expose the numerical regularization parameters.
These alter the discretization and are not claimed convergent tuning recipes.
MINPACK's factor is an initial trust-region radius scale, not a Newton step
fraction. Native options follow the primary
[SciPy LM](https://docs.scipy.org/doc/scipy/reference/optimize.root-lm.html)
and [HYBR](https://docs.scipy.org/doc/scipy/reference/optimize.root-hybr.html)
documentation. The limiter/convergence tradeoff is discussed in
[Venkatakrishnan's original study](https://doi.org/10.1006/jcph.1995.1084).
