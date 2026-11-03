# Numerical fluxes and entropy fixes

At a face, all quantities below use the model's common **unweighted conservative
face frame**. Define $\Delta Q=Q_R-Q_L$, $\Delta F=F_R-F_L$ and
$F_c=(F_L+F_R)/2$. Geometry conversion and face area are applied afterward.

| `flux` | Data supplied by the physical model | Availability |
|---|---|---|
| `rusanov` | $Q,F$, signal-speed bounds | Euler and capable custom models |
| `hll` | $Q,F$, lower/upper wave bounds | Euler and capable custom models |
| `stadtke` | Conservative projectors, eigenvalues, widths | Euler and capable custom models; default |
| `roe` | Finite-jump Roe projectors/speeds/widths and validity | Nozzle/annular Euler and capable custom models |
| `osher` | Admissible hyperbolic state path, storage derivative and eigenspaces | Models supporting the required path differentiation |
| `model_compatibility` | Explicit model flux and nonconservative action | Required by the active two-fluid model |

## Rusanov and HLL

Rusanov uses the largest absolute endpoint signal bound $s$:

$$
\widehat F=F_c-\tfrac12s\Delta Q.
$$

HLL retains left and right signal bounds $S_L,S_R$, formed by taking the
minimum and maximum of the model's endpoint bounds:

$$
\widehat F=\begin{cases}
F_L,&S_L\geq0,\\
\dfrac{S_RF_L-S_LF_R+S_LS_R\Delta Q}{S_R-S_L},&S_L<0<S_R,\\
F_R,&S_R\leq0.
\end{cases}
$$

These methods do not need eigenvectors. HLL represents a two-wave fan, so it
does not resolve every contact/shear field separately. Their robustness still
depends on valid signal bounds, physical reconstruction and appropriate stepping.

## Harten regularized absolute value

The characteristic methods regularize acoustic viscosity near sonic speeds
with the piecewise quadratic function

$$
\phi(\lambda,\delta)=\begin{cases}
(\lambda^2+\delta^2)/(2\delta),&|\lambda|<\delta,\\
|\lambda|,&|\lambda|\geq\delta.
\end{cases}
$$

For positive $\delta$, it is $C^1$, matches $|\lambda|$ and its slope at the
joins, and supplies positive viscosity at $\lambda=0$. It is not $C^2$.
This is the implemented Harten entropy regularization. A sonic expansion can
need this dissipation even when an unregularized approximate wave speed is zero.
The fix alone does not prove an entropy inequality for the full method.

Euler Stadtke widths use the mean sound speed and endpoint family spread.
For $d_k=\lambda_{R,k}-\lambda_{L,k}$ and
$\delta_0=\texttt{delta_factor}\,\bar a$,

$$
\delta_k=\delta_0+\tfrac14\left(d_k+\sqrt{d_k^2+\delta_0^2}\right).
$$

Other models own their width policy. Roe permits zero widths: those groups use
the exact absolute value with its kink at zero, with safe inactive arithmetic.

## Stadtke and the two custom corrections

Let $P_k$ be the model's conservative projectors, $\lambda_k$ its mean speeds,
$\phi_k=\phi(\lambda_k,\delta_k)$ and $s_k=\lambda_k/\phi_k$.
With both project correction flags disabled, the retained FVS contraction is

$$
\widehat F=F_c-\tfrac12\sum_k P_k
\left[s_k\Delta F+(\phi_k-s_k\lambda_k)\Delta Q\right].
$$

The second term supplies sonic viscosity omitted by merely smoothing the flux
weights. For a frozen linear system the whole expression has viscosity
$\phi_k$. At a finite nonlinear jump, $\Delta F$ need not equal
$\sum_k\lambda_kP_k\Delta Q$.

`custom_jump_correction=True` replaces that finite-flux-jump dissipation by
mean-characteristic conservative-jump viscosity:

$$
\widehat F=F_c-\tfrac12\sum_k\phi_kP_k\Delta Q.
$$

`custom_endpoint_envelope=True` independently adds

$$
-\tfrac12\sum_k(\mu_k-\phi_k)P_k\Delta Q,\qquad
\mu_k=\max(\phi_k,|\lambda_{L,k}|,|\lambda_{R,k}|).
$$

Both flags default to `True`. Disabling both recovers the original implemented
contraction; the names denote project additions and are not attributed to the
original method. With both on, the dissipation is simply
$\sum_k\mu_kP_k\Delta Q$. A mean-characteristic jump correction is not a
finite-jump Roe construction and does not establish general monotonicity.

## Roe finite-jump linearization

Roe's flux is

$$
\widehat F=F_c-\tfrac12\sum_k
\phi(\widetilde\lambda_k,\delta_k)\widetilde P_k\Delta Q.
$$

The physical model supplies a finite-jump matrix satisfying consistency and
the secant relation $\widetilde A\Delta Q=\Delta F$ in its documented domain.
The generic numerical kernel only contracts the projectors with the jump.
An arbitrary Jacobian at an averaged state does not meet that requirement.

The shared Euler PH construction uses square-root-density velocity/total-enthalpy
averages and a normalized pressure-secant projection for general-EOS data.
Annular Euler uses a common face radius and the corresponding transverse
velocity. Its repeated contact/shear speed uses a rank-two projector.
Acoustic groups receive Harten regularization; the convective group uses zero
width, preserving the exact absolute value.

Near dimensionless thermodynamic jumps below $\epsilon_{\rm mach}^{1/3}$,
symmetric provisional slopes avoid amplifying property subtraction noise. A
$C^2$ blend reaches the full secant projection by $2\epsilon_{\rm mach}^{1/3}$.
The metadata documents the reference scales and the mixed pressure-secant
audit tolerance $4096\epsilon_{\rm mach}$. This is a face-data validity policy,
separate from nonlinear convergence. Invalid endpoints, nonpositive Roe sound
speed squared, nonfinite data or excessive secant defect reject the candidate;
there is no replacement numerical flux.

## Osher path viscosity

With path $W(\theta)$, $0\leq\theta\leq1$, Osher evaluates

$$
\widehat F=F_c-\tfrac12\int_0^1
\sum_k\phi(\lambda_k(\theta),\delta_k(\theta))P_k(\theta)
\frac{dQ(W(\theta))}{d\theta}\,d\theta.
$$

The default path is affine in the model's reconstruction variables. A model
can supply `face_state_path` instead. JAX directional differentiation supplies
the conservative storage tangent. A property backend must support that
derivative, beyond merely returning traceable property values.

`quadrature_points=3`, `5` (default), or `9` selects Gauss nodes **per panel**.
`integration="gauss"` is the default unsplit panel.
`integration="sonic_partitioned"` splits at captured sonic/regularization
crossings and applies the chosen rule in each panel. Nine nodes on one panel
are not equivalent to sonic partitioning. Finite sampling does not certify
unseen crossings or admissibility of the entire continuous path. Invalid
sampled states reject the candidate. The model owns the path and hyperbolicity.

## Two-fluid compatibility flux

The active two-fluid model accepts only `flux="model_compatibility"`. Its
frozen-reference FVS has its own low-Mach acoustic factor, entropy width,
zero/full-slip branches and an $XK^{-1}\Delta\widehat F$ correction. These are
controlled by `CompatibilityConfig`; they are independent of the two Euler
Stadtke correction flags. See the [two-fluid derivation](../theory/two_fluid.md).

## Optional fourth-difference damping

Positive `filter_strength` adds an internal-face flux proportional to a third
difference of transported conserved densities:

$$
\widehat F_{j}^{\rm filt}=\widehat F_j+
\epsilon_f s_j(Q_{j+2}-3Q_{j+1}+3Q_j-Q_{j-1}),
$$

where the four point values straddle the internal face. Its divergence acts
as fourth-difference damping. Endpoint additions vanish and the correction
remains conservative. The default strength is zero; model filter callbacks
are required. The wider stencil changes Jacobian bandwidth and explicit
stability limits. Two-fluid-specific filters have separate physical-policy
configuration and should not be confused with this generic filter.

Implementations: {mod}`nozzle_hyperbolic.numerics.fluxes`,
{mod}`nozzle_hyperbolic.numerics.characteristic`,
{mod}`nozzle_hyperbolic.numerics.roe`, {mod}`nozzle_hyperbolic.numerics.osher`
and {mod}`nozzle_hyperbolic.numerics.filters`.
