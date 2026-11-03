# Jacobians, coloring and linear solves

Newton requires the derivative of the **complete discrete residual**, including
nonlinear storage, reconstruction/limiters, fluxes, sources, cell constraints
and algebraic endpoint equations. JAX differentiates this assembled map.
Physical property derivatives therefore come through the selected fluid backend.

## Block structure

With $M=N+2$ point blocks and $V$ variables per block, the coupled unknown vector
contains $n=MV$ scalar entries. If row block $i$ depends only on blocks
$i-r,\ldots,i+r$, its Jacobian has $2r+1$ block diagonals. Individual blocks
are $V\times V$ and can be dense.

The operator combines every declared physical dependency with numerical reach.
Constant reconstruction without the generic filter starts with radius one;
MUSCL or the fourth-difference filter starts with radius two. Source/boundary
dependencies and nonconservative/stabilization/equation policies can enlarge
this. In particular, the two-cell Euler boundary extrapolation declares radius
two, so even a constant Euler reconstruction normally has an augmented
**radius-two** Jacobian. A tridiagonal interior does not imply a tridiagonal
augmented system.

```{figure} ../_static/jacobian_coloring.svg
:alt: Radius-two block Jacobian with three variables per point, fifteen scalar colors, and two nonoverlapping column supports sharing one directional seed.
:class: coloring-diagram
:name: jacobian-coloring

Coloring an illustrative ten-block, three-variable radius-two system. Each
outlined block is $3\times3$; scalar colors repeat every five point blocks.
The highlighted column supports show why two columns can share one JVP seed.
```

Download the {download}`editable SVG <../_static/jacobian_coloring.svg>`.
The diagram's blocks show the declared band, including possible structural
zeros inside it; it does not claim every displayed entry is numerically nonzero.

## Directional seeds and decompression

Number scalar columns in block-major order: $j=iV+v$. The code assigns

$$
s=V(2r+1),\qquad c(j)=j\bmod s,\qquad
n_{\rm colors}=\min(s,n).
$$

For a color $c$, seed vector $v^{(c)}$ has a one at every column of that color
and zero elsewhere. A JAX linearization computes the directional product

$$
y^{(c)}=Jv^{(c)}.
$$

Columns sharing a color have block separation $2r+1$. Their possible row
supports, each extending at most $r$ blocks either way, do not overlap.
Consequently each allowed matrix entry is recovered directly:

$$
J_{ij}=y_i^{(c(j))},\qquad
\left|\left\lfloor i/V\right\rfloor-\left\lfloor j/V\right\rfloor\right|\leq r.
$$

For $V=3$, $r=2$, fifteen directional seeds suffice independently of the number
of cells once the problem is large enough. This is **scalar column coloring**:
five block-color families times three distinct variable colors, rather than
five scalar seeds. Larger models or wider stencils need more colors.

`jax.linearize` forms a reusable linearized map at the current Newton point;
the seed products are batched with `vmap`. Decompression can produce a dense
matrix for comparison or directly assemble block bands for the correction solve.

## Exact local derivatives

Cell storage and the complete independent face-flux maps have known locality.
The operator evaluates their exact local Jacobians once per linearization and
shares them across differentiation seeds. Reconstruction, sources, boundary
laws and endpoint overrides retain their differentiation paths. This changes
derivative evaluation cost while preserving the complete residual derivative.
It is not a lagged Jacobian reused across Newton iterations.

Piecewise limiters and active-set conditions can have mathematical kinks. Safe
inactive arithmetic prevents spurious nonfinite AD values; it does not make
those functions globally smooth. The smooth Van Leer option changes the
reconstruction formula explicitly.

## Linear correction solve

`jacobian_mode="colored", linear_solver="banded"` is the default. The
compressed derivatives populate block bands, then
{func}`~nozzle_hyperbolic.numerics.linear.solve_block_banded` applies block
Gaussian elimination and back substitution to solve $J\Delta z=-H$.
It groups $\max(r,1)$ adjacent point blocks into one superblock. A radius-two
band therefore becomes a block-tridiagonal system with $2V\times2V$ dense
superblocks; an incomplete final group receives identity padding.

For lower, diagonal and upper superblocks $L_i,D_i,U_i$, and right-hand side
$b_i$, the forward block-Thomas recursion is

$$
\overline D_i=D_i-L_iC_{i-1},\qquad
C_i=\overline D_i^{-1}U_i,\qquad
d_i=\overline D_i^{-1}(b_i-L_id_{i-1}),
$$

starting with zero previous factors. The implementation solves for the upper
factor and right-hand side together; it does not form a matrix inverse.
Back substitution gives

$$
\Delta z_i=d_i-C_i\Delta z_{i+1}.
$$

The block-band solver does not perform global pivoting; a singular/ill-conditioned
superblock factorization can still produce a failed correction. Nonfinite corrections
are rejected by the enclosing solve rather than treated as convergence.

`jacobian_mode="dense", linear_solver="dense"` forms the full JAX Jacobian
and uses a dense solve. Colored assembly can also feed the dense solver.
Dense assembly with the banded solver is rejected. Globally coupled boundary
laws require dense/dense because the local band assumption no longer holds.

For fixed $V,r$, band storage and work grow linearly with the number of point
blocks, with dense algebra inside the small blocks. A dense matrix instead
uses quadratic storage and a cubic direct solve in total scalar size.
These are structural costs, not timing guarantees for a particular backend.

## Checking the stencil contract

Coloring is valid only if the declared dependencies contain **all** true ones.
An undeclared distant source or boundary read can make two same-color columns
collide and silently corrupt decompression. `jacobian_difference` compares
colored and independent dense AD. This is the appropriate guard when model
stencils change, together with directional-derivative checks.

The implementation is {mod}`nozzle_hyperbolic.numerics.jacobians` and
{mod}`nozzle_hyperbolic.numerics.local_derivatives`. Boundary elimination adds
the implicit sensitivity described in [boundary solution](boundaries.md).
