# Stadtke baseline and explicitly custom corrections

This note defines the production `flux="stadtke"` family tested here. The
baseline is the implemented generalized characteristic FVS. The two terms
named `custom_jump_correction` and `custom_endpoint_envelope` are project
modifications, **not terms attributed to the literature Stadtke method**.
Their switches are independent and default to `True`. Set both to `False`
when requesting the original baseline.

Numerical evidence is tied to the recorded source/case/client hashes. A later
external Roe-source change was preserved; these recorded checks are not a
claim that the current working checkout is byte-identical or newly validated.

## Original $F^+/F^-$ split

At one face, $W_L,W_R$ are the usual constant or MUSCL reconstructed states.
The model supplies conservative storage $U(W)$, physical flux $F(W)$ and
mean-state characteristic projectors/speeds $(\bar P_k,\bar\lambda_k)$ in a
common face frame. Write $\Delta U=U_R-U_L$, $\Delta F=F_R-F_L$ and
$\sum_k\bar P_k=I$. Define the Harten absolute speed by

$$
\phi_\delta(\lambda)=
\begin{cases}
(\lambda^2+\delta^2)/(2\delta),&\delta>0,\ |\lambda|<\delta,\\
|\lambda|,&\text{otherwise},
\end{cases}
\qquad \bar\phi_k=\phi_{\bar\delta_k}(\bar\lambda_k),\quad
\bar s_k=\bar\lambda_k/\bar\phi_k,\quad
\bar b_k=\bar\phi_k-\bar s_k\bar\lambda_k.
$$

Production Stadtke requires positive characteristic widths. At a zero wave
speed, $\bar\phi_k=\bar\delta_k/2>0$, so $\bar s_k=0$ without division by
zero. No zero-width operating limit is claimed for the unchanged baseline
kernel. With the shared mean data held fixed, the implemented split is

$$
F^\pm(W;\bar W)=\frac12\sum_k\bar P_k
\left[(1\pm\bar s_k)F(W)\pm\bar b_k U(W)\right],
\qquad \widehat F_0=F^+(W_L;\bar W)+F^-(W_R;\bar W).
$$

Equivalently,

$$
\boxed{\widehat F_0=\frac{F_L+F_R}{2}
-\frac12\sum_k\bar P_k
\left[\frac{\bar\lambda_k}{\bar\phi_k}\Delta F+
\left(\bar\phi_k-\frac{\bar\lambda_k^2}{\bar\phi_k}\right)\Delta U\right].}
$$

This specifies the actual baseline algebra, including its sonic viscosity.
The model owns averaging, wave families, frame changes and thermodynamics.
The shared operator owns reconstruction and applies the physical face area
once. Geometry/source and boundary equations are separate from this
homogeneous face flux.

## Exact custom terms and the four variants

The custom conservative-jump term is

$$
\boxed{C_{\rm jump}=\frac12\sum_k
\frac{\bar\lambda_k}{\bar\phi_k}\bar P_k
\left(\Delta F-\bar\lambda_k\Delta U\right).}
$$

It corrects the difference between a finite physical flux jump and the
existing mean-state characteristic linearization. It vanishes for a frozen
linear system or compatible finite-jump data. Arithmetic averaging does not
generally satisfy a Roe jump identity: this correction is **not a Roe solver**.

The separate endpoint envelope is

$$
\alpha_k=\max\left(\bar\phi_k,
|\lambda_k(W_L)|,|\lambda_k(W_R)|\right),\qquad
\boxed{C_{\rm end}=-\frac12\sum_k
(\alpha_k-\bar\phi_k)\bar P_k\Delta U.}
$$

Endpoint families must use the model's consistent family labels. The full
switchable flux is

$$
\boxed{\widehat F=\widehat F_0+
\chi_J C_{\rm jump}+\chi_E C_{\rm end},\qquad
\chi_J,\chi_E\in\{0,1\}.}
$$

| Saved variant | Jump flag | Endpoint flag | Exact viscosity |
|---|---|---|---|
| `stadtke_original` | `False` | `False` | Original mixed $\Delta F,\Delta U$ dissipation |
| `stadtke_jump` | `True` | `False` | $\sum_k\bar\phi_k\bar P_k\Delta U$ |
| `stadtke_endpoint` | `False` | `True` | Original dissipation plus the endpoint increment |
| `stadtke_both` | `True` | `True` | $\sum_k\alpha_k\bar P_k\Delta U$ |

In particular,

$$
\widehat F_{\rm jump}=\frac{F_L+F_R}{2}
-\frac12\sum_k\bar\phi_k\bar P_k\Delta U,\qquad
\widehat F_{\rm both}=\frac{F_L+F_R}{2}
-\frac12\sum_k\alpha_k\bar P_k\Delta U.
$$

**Endpoint-only is not the combined correction.** It retains the original
mixed flux-jump part. Its new independent experiment cannot inherit a pass
from jump-only or both-on results. Actual booleans are retained with every
saved row and checked against the variant label.

## Why this can change a shock root

When all mean waves point downstream and regularization is inactive, the
original mixed formula can reduce to the incoming physical flux despite a
large finite jump. Conservative-jump viscosity changes that finite-jump
treatment. The endpoint term additionally accounts for endpoint speeds that
exceed the regularized mean speed. These are algebraic, model-independent
constructions; no empirical pressure sensor or replacement-flux blend is used.

One shared flux per face preserves telescoping conservation. Coincident-state
consistency and linear-system equivalence do not imply a maximum principle.
Extra dissipation may alter shock width and thermodynamic internal-layer
errors. Removing a pressure spike does not by itself verify total enthalpy,
physical branch selection or analytical accuracy.

## Per-face cost and explicit timestep guard

The jump correction reuses the existing mean projectors/speeds and endpoint
storage/physical fluxes. Its additional work is contractions; evaluating the
equivalent matrix form can even avoid some original flux-jump contractions.
The endpoint envelope requests two coincident-state spectral callbacks.
Endpoint fused EOS auxiliary data are reused where the model supports that
interface. This is different from integrating many new path states. AD,
compiler fusion and property-backend cost prevent a universal timing factor;
fewer outer iterations also need not mean a cheaper complete solve.

For explicit Euler/SSPRK3, either enabled correction additionally guards the
step using selected regularized interface speeds: the mean $\bar\phi_k$, or
the endpoint envelope $\alpha_k$ when enabled, augmented by physical endpoint
bounds where available. With the resulting face bound $s_f$,

$$
\Delta t\le\mathrm{CFL}\min_i
\frac{\Delta x_i}{\max(s_{i-1/2},s_{i+1/2})}.
$$

Existing cell/model/source bounds can further reduce the step. Both-off
Stadtke preserves its historical timestep arithmetic. The interface guard is
not a universal nonlinear stability or positivity theorem. This study uses
backward Euler: its implicit CFL controller is unchanged.

## A representation limit on a universal primitive no-peak claim

Even exact conservative cell averages need not preserve extrema of a nonlinear
primitive variable: $p(\overline U)\ne\overline p$. As a mathematical example,
consider the thermodynamically consistent EOS

$$
p=R\rho T-a\rho^2,\qquad e=c_vT-a\rho,\qquad
\gamma=1+R/c_v.
$$

A stationary contact has $u_L=u_R=0$ and $p_L=p_R=p_0$. An exact cell average
$\overline U=\theta U_L+(1-\theta)U_R$ recovers

$$
\boxed{p(\overline U)=p_0+(2-\gamma)a\theta(1-\theta)
(\rho_L-\rho_R)^2.}
$$

Take $R=1$, $c_v=2.5$, $a=1$, $\rho_L=1$, $\rho_R=2$, $p_0=10$.
Then $\gamma=1.4$, $T_L=11$, $T_R=7$, and $\theta=1/2$ gives
$p(\overline U)=10.15$: a legitimate 1.5% primitive pressure excess. Both endpoint states
have positive heat capacity, positive isothermal pressure derivative (9 and
3), and positive sound-speed squared (13.4 and 5.8).

This is a representation counterexample, **not a new result of this campaign
and not an explanation of the abnormal ideal-gas FVS shock root**. It shows why
generic conservative algebraic corrections cannot imply a universal
primitive-variable no-peak theorem. Analytical conservative projections and
primitive point branches must be assessed separately. The preserved algebraic
record is `outputs/general_algebra/exact_projection_pressure_counterexample.json`;
its authoritative retrieval path is the
[legacy inventory and verified archive locator](../LEGACY_SPIKE_INVENTORY.md#verified-archive-recorded-on-2026-11-01).

## Verification and limits

This independent study requests four nozzle/annular shock cases, four grids
(50/100/200/400 cells), constant and Van Leer MUSCL reconstruction and all four
flag combinations: 128 cold solves. Each grid is initialized independently;
no analytical seed, grid continuation or hidden warm start is used.

The public production driver retains its boundary equations, backward Euler,
Newton tolerances, residual stopping and adaptive-CFL rules. A changed flux
changes the spatial residual and its AD Jacobian; it does not constitute a new
Newton globalization algorithm. Accepted steps, rejected attempts and all
Newton corrections are different work measures.

Saved original-equation/BC/global checks, admissibility, analytical branch
checks and finite-volume projection errors are reported separately. Failed,
missing and budget-limited attempts remain visible. New results are assessed
from their own source and case identities, not inherited from another study.

No unconditional positivity, nonlinear entropy-stability or componentwise
monotonicity theorem is claimed for these custom terms or for the complete
source/boundary/implicit update. General model callbacks do not prove that a
future equation of state, nonconvex regime or two-fluid system is covered.

The separate [Osher study](../osher/THEORY.md) integrates viscosity along a
model-declared path. It is a comparison family, not required for either custom
Stadtke switch. The [Roe theory](../roe/roe_theory.md) instead specifies a
model-owned finite-jump matrix and should not be confused with mean-state
projectors used here.
