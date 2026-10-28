# Euler numerical changes: what changed and why

These changes address singular flux algebra, discontinuous sonic regularization,
and unsafe differentiation of reconstruction. The first and third repairs
preserve their intended mathematical formulas where those formulas are defined.
The sonic regularization and optional smooth limiter change the discretization.

The [illustrated steady-nozzle report](results_steady/REPORT.md) contains the
numerical findings and figures. [RUNNING.md](RUNNING.md) explains reproduction.
This README describes the changes in `src` and their mathematical motivation.

| Source | Changes |
|---|---|
| [Euler flux](../../../src/nozzle_hyperbolic/models/euler_nozzle/flux_jax.py) | Regular characteristic projectors and continuous, consistent sonic regularization; preserved legacy flux |
| [Reconstruction](../../../src/nozzle_hyperbolic/numerics/reconstruction.py) | Safe ordinary Van Leer arithmetic and optional smooth limiter |
| [Euler FVM](../../../src/nozzle_hyperbolic/models/euler_nozzle/fvm.py) | Select flux and reconstruction consistently throughout residual/Jacobian evaluation; accept fixed smoothing scales |

## 1. Characteristic projectors: remove an unnecessary singular inverse

The solver uses primitive variables $W=(p,u,h)$ and conservative variables

$$
Q=(\rho,\rho u,\rho(h+u^2/2)-p).
$$

Let $C=\partial Q/\partial W$ be the storage Jacobian and
$K=\partial F/\partial W$ the physical-flux Jacobian. The old implementation
uses $K^{-1}$ when transforming characteristic projectors. But

$$
\det K=\frac{\rho^2}{a^2}u(u^2-a^2),
\qquad
\det C=\frac{\rho^2}{a^2}.
$$

Thus $K$ is singular at rest and sonic flow, even with a perfectly admissible
thermodynamic state. These are common flow states, not exceptional EOS errors.
Where $K$ is invertible, the projector expression can be rewritten exactly as

$$
K v_i\ell_iK^{-1}=C v_i\ell_iC^{-1}=\Pi_i.
$$

The implementation evaluates the right-hand expression using the EOS density,
sound speed, and Grüneisen derivatives. It avoids divisions by $u$ or
$u\pm a$. This representation is regular for positive density and sound speed.
It is an algebraic repair where the old expression is defined, with a regular
extension through rest and sonic states. It does not require guessing a shock.

In floating point, an exactly sonic state can sometimes return a finite old
flux because roundoff masks the singularity. A finite value is insufficient:
its derivatives can still be non-finite or extremely sensitive.

## 2. Sonic regularization: remove a discontinuous activation and retain viscosity

The old entropy-fix activation is the strict condition
$\lambda_L<0<\lambda_R$. Crossing its boundary switches between different
regularization widths. Although each selected formula can be differentiable
inside its branch, the combined interface flux has a finite jump at the switch.
AD differentiates the selected formula and cannot reveal that jump by itself.

The corrected formulation builds a positive width continuously from a small
floor and the left/right characteristic-speed spread. With $\phi_i$ the
positive Harten regularization of $|\lambda_i|$, it uses

$$
s_i=\lambda_i/\phi_i,\qquad
c_i=\phi_i-\lambda_i^2/\phi_i\geq0,
$$

$$
\widehat F=\tfrac12(F_L+F_R)
-\tfrac12\sum_i\Pi_i\left[s_i(F_R-F_L)+c_i(Q_R-Q_L)\right].
$$

Equal states give the physical flux exactly because both jumps vanish.
The conservative-state jump supplies the viscosity lost if one smooths the
flux weights alone. For a frozen linear system the resulting characteristic
viscosity is $\phi_i$, rather than $\lambda_i^2/\phi_i$.

**This part changes the numerical method in the regularization band.** It is
not merely a different way to evaluate the original formula. The width also
has a different definition from the legacy width; equal `delta_factor`
therefore does not imply identical smoothing bands. Outside the band the
additional conservative viscosity vanishes.

For smooth admissible EOS data and positive width, the corrected flux is
$C^1$; Harten's band edges are not $C^2$. This is not a proof of nonlinear
Euler entropy stability, positivity of a full timestep, or global Newton
convergence. Those stronger properties have not been established here.

## 3. Ordinary Van Leer: make inactive arithmetic safe for reverse AD

The mathematical limiter is

$$
\sigma(a,b)=
\begin{cases}2ab/(a+b),&ab>0,\\0,&ab\leq0.\end{cases}
$$

The original JAX expression forms the quotient inside `where`. At zero or
cancelling differences, the inactive quotient can overflow or have undefined
sensitivities. Reverse AD can then produce NaNs even though the selected
primal result is zero. Merely putting `where` around a dangerous quotient
does not ensure safe differentiation.

The repair replaces the denominator with a harmless value on the inactive
branch **before division**. The active Van Leer formula is unchanged. This
fixes the tested AD failure without making Van Leer mathematically smooth:
its derivative kinks remain, and the origin lacks a unique linear derivative.

## 4. Optional smooth MUSCL: deliberately change slopes near flat regions

For normalized differences, the optional limiter uses

$$
t=\max(ab,0),\quad s=a+b,\qquad
\sigma_\epsilon=\frac{2t^3s}{(t^2+\epsilon^4)(s^2+\epsilon^2)}.
$$

It is $C^2$, preserves constants, remains zero for opposite slopes, and
attenuates rather than enlarges the ordinary same-sign slope. At finite
$\epsilon$ it does not exactly preserve arbitrarily small linear slopes.
For example, the probe with $a=b=10^{-6}$ returns approximately
$1.63\times10^{-14}$ instead of $10^{-6}$ at $\epsilon=2.5\times10^{-5}$.

The smoothing width is controlled by a coefficient and mesh size,
$\epsilon=\text{coefficient}/(N-1)^2$, with fixed component scales such as
$(p_0,a_0,h_0)$. These scales do not depend on the current field and therefore
preserve the local stencil used by the colored Jacobian.
Smoothness of this limiter does not make all boundary/EOS/flux operations
globally smooth, and its component bounds do not prove full-system TVD.

## Defaults and legacy reproduction

`flux_variant="regularized"` is the corrected default. Set
`flux_variant="legacy"` to select the preserved original flux.
`reconstruction="muscl_van_leer"` remains ordinary Van Leer with safe inactive
arithmetic. The smooth limiter is opt-in through
`reconstruction="muscl_van_leer_smooth"`; it is not the default in the latest
PTC experiments.

The legacy selector restores the original **flux**, not the old unsafe shared
limiter expression. This distinction matters when reproducing AD failures.
The corrected flux resolves specific regularity defects; it does not guarantee
that a global nonlinear solver will reach a shocked steady solution.

These numerical changes are separate from the structural refactor;
see the [change inventory](../../../docs/NUMERICAL_CHANGES_REVIEW.md) and the earlier
[implementation report](../../../docs/SOLVER_REGULARITY_IMPROVEMENTS.md).

For the verification study, the FVM driver also supports optional
`record_history=True`. This records diagnostics after each accepted implicit
update; it does not alter the residual, timestep controller, or acceptance
rules. Recording is disabled by default. The plotted SSR is the native
state-normalized steady spatial residual, rather than the inner Newton residual.
