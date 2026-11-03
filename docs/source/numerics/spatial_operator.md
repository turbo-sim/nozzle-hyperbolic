# Finite-volume residual

## Mesh and state layout

A fixed mesh contains $N$ cells, $N+1$ face coordinates, cell representatives,
positive volumes $V_i$ and positive face areas $A_f$. Physical cell contents are
$\mathcal Q_i=V_iQ(W_i)$. The full state includes two independent endpoint
traces, so its shape is `(N + 2, V)`, with $V$ state variables per point.
The endpoint coordinates lie at the first and last faces; traces are not
extra cells and have no physical storage volume.

The model may have $T$ transported balances and $C$ local constraints, with
$T+C=V$. Euler nozzle has $(T,C,V)=(3,0,3)$, annular Euler $(4,0,4)$ and
two-fluid $(6,2,8)$. Each endpoint has $V$ algebraic equations, making the full
system square: $(N+2)V$ unknowns and rows.

## Residual sign and assembly order

The common semidiscrete equations are

$$
\frac{d\mathcal Q_i}{dt}+R_i(W)=0,\qquad
R_i=A_{i+1/2}\widehat F_{i+1/2}
-A_{i-1/2}\widehat F_{i-1/2}+N_i-S_i,
$$

$$
g(W_i)=0,\qquad B(W,t)=0.
$$

$S_i$ is already integrated over the control volume. $N_i$ is integrated
nonconservative work, explicitly zero for conservative models. The flux
callbacks return unweighted fluxes: the operator applies face area **once**.
Model-owned frame conversion, such as transverse-to-angular momentum for
annular Euler, occurs before that area weighting.

At each evaluation the operator:

1. Maps iteration variables to reconstruction variables, if a model declares
   such a map, and constructs left/right face states.
2. Evaluates model flux/property bundles and the chosen numerical flux.
3. Computes model nonconservative work from the unfiltered flux when required.
4. Adds selected stabilization/filter fluxes and model endpoint flux overrides.
5. Converts to the transported frame, applies face areas and differences fluxes.
6. Subtracts the integrated physical source and adds nonconservative work.

The two-fluid compatibility pairing specifically needs the raw flux from step
2. Changing its order relative to filtering would change its discrete equations.
`Operator.residual_contributions()` exposes the separate terms, including
equation-policy changes, for inspection.

## Conservation and geometry

Summing the cell rows telescopes the interior face fluxes:

$$
\sum_iR_i=A_{N+1/2}\widehat F_{N+1/2}
-A_{1/2}\widehat F_{1/2}+\sum_iN_i-\sum_iS_i.
$$

Conservation diagnostics use these actual numerical endpoint fluxes and the
same sources/nonconservative work as assembly. They do not substitute a
cell-centered mass flow or an analytical boundary flux. A conservative internal
filter also telescopes; its endpoint flux additions are zero.

Geometry is prepared from the physical shape independently of the solution.
For nozzle Euler, the source $p_i\int_i A'dx$ pairs with face pressure forces.
For annular Euler, the model also supplies centrifugal and wall-torque metrics.
Exact balance for a particular resting state does not establish that every
steady flowing solution is preserved exactly by a given reconstruction/flux.

## Model interface and locality

{mod}`nozzle_hyperbolic.solvers.fvm.contracts` defines structural protocols;
no inheritance is required. A model supplies `cell_conserved`, physical face
data, integrated sources, boundary residuals, admissibility, fixed scales and
dependency declarations. Optional capabilities supply characteristics, finite-jump
Roe data, a path for Osher, cell constraints or model compatibility fluxes.

`face_data` can return conserved state, flux, speed bounds and an opaque
auxiliary property bundle together. This avoids repeated EOS evaluations.
The numerical layer never interprets EOS properties or variable identities.

Storage is cell-local and each face callback acts independently at one face.
Sources and boundaries declare their neighbor reach. These are mathematical
requirements of the [colored Jacobian](jacobians.md), rather than performance
hints. Explicitly time-forced interior source callbacks and periodic domains
are outside the current contract; boundary laws receive physical time.

The main implementation is
{class}`~nozzle_hyperbolic.solvers.fvm.operator.Operator` and fixed geometry is
represented by {class}`~nozzle_hyperbolic.numerics.mesh.Mesh`.
