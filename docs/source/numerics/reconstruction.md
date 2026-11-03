# Reconstruction and limiters

Reconstruction supplies a pair of states $(W_L,W_R)$ at every physical face.
It is independent of flux selection and time integration. The default
`Scheme(reconstruction="constant")` uses the adjacent point values unchanged.

## Coordinate-aware MUSCL

Let $Z_i$ be the reconstruction variables and $x_i$ their actual point
coordinates, including endpoint traces. The one-sided gradients are

$$
a_i=\frac{Z_i-Z_{i-1}}{x_i-x_{i-1}},\qquad
b_i=\frac{Z_{i+1}-Z_i}{x_{i+1}-x_i}.
$$

A componentwise limiter gives $\sigma_i=\mathcal L(a_i,b_i)$. Extrapolate to
the physical face coordinate $x_f$:

$$
Z_{L,f}=Z_i+\sigma_i(x_f-x_i),\qquad
Z_{R,f}=Z_{i+1}+\sigma_{i+1}(x_f-x_{i+1}).
$$

Endpoint traces have zero slope. Interior face states are thus coordinate
aware on nonuniform grids; a fixed half-cell jump formula is not used.
Piecewise linear reconstruction gives second-order spatial behavior in smooth
regions under the usual consistency assumptions. At extrema or discontinuities
the limiter reduces the slope and can reduce local order.

By default $Z=W$. Two-fluid flow maps its eight-state iteration variables to
six enthalpy primitives, reconstructs those, then maps face states back through
the EOS. This does not reduce the number of cell unknowns or remove the EOS
constraints. Model-provided maps must be local and have fixed scales.

## Implemented slope limiters

For $ab\leq0$ the following gated limiters return zero. For $ab>0$, let
$s=\operatorname{sign}(a)$:

| `reconstruction` | $\mathcal L(a,b)$ for like signs | Behavior |
|---|---|---|
| `muscl_minmod` | $s\min(\lvert a\rvert,\lvert b\rvert)$ | Strong slope reduction |
| `muscl_van_leer` | $2ab/(a+b)$ | Harmonic mean of the gradients |
| `muscl_superbee` | $s\max[\min(2\lvert a\rvert,\lvert b\rvert),\min(\lvert a\rvert,2\lvert b\rvert)]$ | More compressive |
| `muscl_van_albada` | $ab(a+b)/(a^2+b^2)$ | Smooth rational expression away from the gated origin |

The Superbee selection compares **magnitudes**, preserving decreasing as well
as increasing profiles. Van Albada normalizes before squaring to avoid numerical
underflow/overflow. Inactive denominators are made safe before evaluation so
JAX differentiation does not encounter an invalid unused quotient. This leaves
the mathematical kinks of the unsmoothed limiters intact.

`muscl_mc` occurs in retained low-level NumPy helpers but is not a supported
public `Scheme` selector.

## Smooth Van Leer option

`muscl_van_leer_smooth` nondimensionalizes gradients with
$Z_*/L$, using fixed component scales and domain length. For those dimensionless
gradients, define $t=\max(ab,0)$, $s=a+b$ and
$\epsilon=c_\epsilon/N^2$, where
`smooth_limiter_coefficient` is $c_\epsilon>0$. The slope formula is

$$
\mathcal L_\epsilon(a,b)=
\frac{2t^3s}{(t^2+\epsilon^4)(s^2+\epsilon^2)}.
$$

It is a $C^2$ regularization across the axes, returns exactly zero for opposing
gradients and recovers Van Leer as the width vanishes. At finite width it also
damps small smooth slopes; linear profiles are not preserved exactly at that
scale. Fixed normalization avoids introducing nonlocal state dependence into
the Jacobian stencil.

Componentwise slope bounds do not establish positivity, TVD or entropy stability
for the full coupled flow update. Cells, traces and reconstructed states still
pass the model's physical admissibility checks. A failed reconstructed state
rejects a trial; it is not silently repaired by changing the physical state.

The implementation is {func}`~nozzle_hyperbolic.numerics.reconstruction.reconstruct_faces`.
