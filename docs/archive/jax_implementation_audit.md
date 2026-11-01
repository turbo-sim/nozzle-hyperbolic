# JAX, JIT, and automatic differentiation audit

> Historical record of the original branch/structural-refactor review. The
> descriptions of "current" files, pending changes and proposed organization
> below refer to that inspection, not the present package. Use the
> [current architecture](../architecture/ARCHITECTURE.md) and
> [model interface](../model_interface.md) for supported APIs. These findings
> have not all been revalidated or resolved, especially for the deferred
> two-fluid implementation; see the [research backlog](../proposals/research_backlog.md).

Date: 2026-10-26. Baseline: `alberto-branch` at `6fe085c`.

This audit records inherited implementation issues and limitations. No numerical implementation was repaired as part of the file-organization refactor. Source line references below identify the original Alberto files at the recorded commit; function names remain the lookup key after relocation.

## Scope and evidence

The review covered the EulerNozzle finite-volume driver and step factories, NumPy and JAX flux implementations, thermodynamic recovery, and the TwoFluid state, matrices, flux, sources, closures, and step factory. EulerAnnular has no finite-volume implementation on this baseline to audit.

Small probes ran with the available Python interpreter and JAX **0.6.2**, with x64 enabled. They extracted the relevant function ASTs directly from `git show alberto-branch:<path>` and executed the original function bodies in isolated namespaces. Where needed, a smooth analytical ideal-gas property stub supplied the property interface. These probes isolate numerical/JAX behavior from dependency loading, table provenance, and case configuration. They are not full nozzle simulations or validation of the installed `jaxprop` thermodynamic backend.

Successful tracing of a residual, a finite forward-mode Newton Jacobian, reverse differentiation of one step, and differentiation of an entire converged solve are different capabilities. The current repository must not be described as supporting all four for every option.

## Confirmed issues

### JAX-01: Comparison-driven EOS bisection gives incorrect state sensitivities

Original source: `nozzle_hyperbolic/fvm_solver/fvm.py`, `make_jax_step._eos_batch`, lines 1411–1433. Related inlet and outlet bisections are at lines 1461–1515.

The EOS recovery updates pressure brackets through comparisons of the density residual. With constant initial brackets, the returned pressure is locally a selected sequence of bracket midpoints. AD follows those selections and does not differentiate the implicit EOS relation. The inner loop is JIT-compatible, but its derivative is not the intended thermodynamic derivative.

An exact-source probe used the smooth property law $\rho=3.5p/h$, equivalent to an ideal gas with $\gamma=1.4$, and $h=e+p/\rho$. At $\rho=1$ and $e=200000$, the analytical relation is $p=0.4\rho e$:

| Quantity | Result |
|---|---:|
| Pressure from the existing 15-iteration bisection | 80002.75105285645 Pa |
| `jax.grad` pressure derivative with respect to internal energy | 0.0 |
| Centered finite difference, energy perturbation 1000 J/kg | 0.39672811889648435 |
| Analytical derivative | 0.4 |

The probe used `h_lo=1`, `h_hi=1e8`, `p_lo_r=1`, and `p_hi_r=1e6`, which leave the pressure bracket constant near this state. If an initial bound depends on the state, AD can instead propagate a bracket derivative; that still does not establish the correct implicit derivative. The analogous boundary inversions need independent sensitivity checks. No custom derivative or replacement root solver was added.

### JAX-02: Helpers named `_jax` contain tracer-to-Python conversions

Original source: `nozzle_hyperbolic/fvm_solver/fvm.py`, `_bc_inlet_real_jax`, lines 1346–1353, and `_bc_outlet_real_jax`, lines 1356–1365.

These helpers call `float()` on quantities computed from the incoming state. Wrapping `_bc_inlet_real_jax` in `jax.jit` with a dynamic `(3, 4)` state raises **`ConcretizationTypeError`** at the velocity conversion on line 1347, before EOS evaluation. The outlet has the same structural restriction at lines 1358–1361.

The active step factories define separate inner boundary functions, so this result does **not** mean those factories fail at this helper. It establishes that these retained public/module helpers cannot be treated as reusable pure JAX boundary kernels.

### JAX-03: Several limiter options violate the assumed local Jacobian stencil

Original source: `nozzle_hyperbolic/fvm_solver/two_fluid/solver.py`, `make_two_fluid_step._muscl`, lines 1006–1029; Jacobian coloring and stencil assumptions, lines 1612–1656.

The colored Jacobian and block-banded paths assume a residual stencil of radius one or two cells. Some limiter scales depend on domain-wide reductions:

- `va_eps_mode='magnitude'` with `va_mean_stop_gradient=False` differentiates the global mean.
- `va_eps_mode='range'` differentiates the global maximum, minimum, and mean.
- `va_eps_mode='smoothness'` also uses the global mean. The `stop_gradient` branch applies only to `magnitude`.

An exact-source reconstruction probe used 12 cells with smooth, positive primitive states. It differentiated the pressure reconstructed on left face 2 with respect to pressure in distant cells:

| Option | Distant pressure derivative |
|---|---:|
| `magnitude`, `va_mean_stop_gradient=True` | 0.0 |
| `magnitude`, `va_mean_stop_gradient=False`, cell 10 | -8.929870215978643e-7 |
| `range`, cell 11 | -1.2549743024583756e-5 |
| `smoothness`, cell 10 | -8.79248759727128e-7 |

The probe used `_va_rel_eff=0.001`; it establishes nonlocal dependence rather than a case-specific error magnitude. Such dependencies can be dropped or folded into incorrect local entries by the existing coloring/decompression. Agreement is not guaranteed for these option combinations. The usual `magnitude` setting with its default stopped mean avoids this particular nonlocal derivative.

There is also a comment/code mismatch: lines 1023–1024 say the `smoothness` mean contribution is scaled by `1e-12`; line 1026 actually uses `0.1 * _va_mean` on that path. This observation concerns the implemented scale and dependency, not a proposed replacement value.

### JAX-04: EulerNozzle FVS is singular at rest and inconsistent under its eigenvalue floor

Original source: `nozzle_hyperbolic/fvm_solver/fvm.py`, `_jax_F_fvs_stadtke`, especially the flux Jacobian at lines 1279–1285 and `_Rt_vec` at lines 1306–1316.

The kernel solves against the physical-flux Jacobian $K$. At a stationary equal state, $K$ is singular and the solve still executes. Multiplying the resulting projectors by a zero coefficient does not remove the NaNs.

For nonzero but small eigenvalues, `abs(lam_avg) / lam_safe` scales the entire projector. Consequently the positive and negative projectors do not sum to the identity when that coefficient is below one. The numerical flux can then fail the equal-state consistency condition $\hat F(U,U)=F(U)$.

An exact-source probe used the same analytical ideal-gas property law as JAX-01, equal left/right states with $p=100000$ Pa and $h=300000$ J/kg, and the function default `delta_factor=0.1`:

| Velocity | Existing numerical flux | Physical equal-state flux |
|---|---|---|
| 0 m/s | `[nan, nan, nan]` | `[0, 100000, 0]` |
| 1 m/s | `[0.84292814, 100000.84292814, 350000.42146406]` | `[1.16666667, 100001.16666667, 350000.58333333]` |
| 100 m/s | `[116.66666667, 111666.66666667, 35583333.33333333]` | Same to floating-point precision |

This is an inherited numerical defect, not a failure to JIT-compile. It can also invalidate derivatives or Newton steps at affected states. The probe does not establish that every existing configured run enters this regime.

## Confirmed restrictions and deliberate derivative approximations

### JAX-05: Tolerance-controlled Newton loops do not support ordinary reverse-mode AD

Original sources:

- `fvm.py`, `make_jax_stadtke_step.step`, lines 2133–2153.
- `two_fluid/solver.py`, `make_two_fluid_step.step`, lines 2558–2573.
- `fvm.py`, `fvm_euler_1d._jax_run`, lines 2600–2635.

When `newton_tol > 0`, both step implementations use `jax.lax.while_loop`. The single-fluid factory docstring already states that this path is not reverse-differentiable. The outer explicit marching loop also uses a dynamic `while_loop`. A finite forward-mode residual Jacobian does not imply that `jax.grad` can differentiate through these loops. Fixed-count alternatives exist, but those alternatives still require their own derivative validation; changing the loop alone would not resolve the EOS issue above.

### JAX-06: Stopped means deliberately alter derivatives

Original source: `two_fluid/solver.py`, `_muscl`, lines 1012–1026.

The default `va_mean_stop_gradient=True` preserves residual values while suppressing the global-mean derivative. This is a deliberate approximation for Newton sparsity. It means AD of that residual is not the full mathematical derivative of the unstopped numerical formula. A dense-versus-colored comparison can pass because both use the same stopped derivative. It does not replace a directional finite-difference check when solution sensitivities are needed.

### JAX-07: Gates, clipping, and branch transitions need a defined sensitivity policy

Original sources include `two_fluid/sources.py`, `evaporation_weight`, lines 102–118; `two_fluid/matrices.py`, `phase_props`, lines 63–78; and the reconstruction, state-recovery, and projection routines.

The evaporation gate is deliberately binary, including its superheat, volume-fraction, and pressure-cutoff conditions. Its derivative is zero away from the switch and does not represent displacement of the switching surface. Clipping at EOS bounds and volume-fraction floors similarly changes sensitivities outside the admitted interval. These operations can be traceable while remaining nonsmooth or yielding uninformative gradients. They should be recorded as model/numerical choices, not silently replaced during a file move.

`phase_props` clips its property-call inputs but constructs `rho_p` and `rho_h` from thermodynamic identities at that clipped state. Those thermodynamic derivatives are not necessarily the derivative of the complete clipped mapping with respect to an out-of-bounds input. This matters when comparing analytical matrices with AD and is a reason to distinguish admissible interior states from recovery/projection behavior.

### JAX-08: The complete application driver and legacy paths are not pure JAX kernels

Original sources include `fvm.py`, `fvm_euler_1d`, lines 2197 onward; `flux.py`, `state_from_rho_u`, lines 20–40; and the NumPy perfect-gas/property helpers.

The EulerNozzle application driver performs NumPy allocation/conversion, Python scalar conversion, printing, mutable parameter-dictionary updates, and host-controlled acceptance logic. Its `use_jax` condition at line 2547 explicitly excludes the legacy perfect-gas and Roe paths. The NumPy flux/recovery implementation also uses SciPy root finding and Python scalar conversions. These are inherited capability boundaries; moving them into model or infrastructure folders does not make them JIT/AD compatible.

Host configuration and reporting are reasonable outside a differentiable kernel. The future distinction must be explicit: a host runner can call a compiled step without itself being differentiable. The TwoFluid `debug_newton=True` path is intentionally non-JIT and is documented as a debugging facility, not a production defect.

## Related numerical inconsistencies in the explicit JAX path

These source-level observations should also be considered when reviewing the general repository issues:

1. **Transmissive boundary pressure is overwritten.** `make_jax_step.step`, `fvm.py` line 1527, always sets `p[-1] = p_out`, even when `transmissive_bc=True` copied the outgoing state. The interface flux can therefore see a pressure that differs from the ghost state's EOS pressure.
2. **RK stages reuse stale properties and sources.** `make_jax_step.step`, lines 1559–1569, sends the initial `p` and `c` to every RK3/RK4 stage and reuses `S`; it does not refresh stage EOS or boundary values. This is not a standard stage-consistent RK application to the nonlinear semidiscrete residual.
3. **Local pseudo-time is reported as physical time.** `make_jax_step.step`, lines 1535–1543, always uses per-cell `dt_vec`, while `fvm_euler_1d._jax_run`, lines 2611–2635, accumulates scalar `dt_min` into `t_phys` and uses it for a physical-time stopping condition. Local time stepping can accelerate steady convergence, but these updates do not define a general physical transient with one shared time coordinate.
4. **Bisection returns no convergence status.** The explicit recovery and boundary loops use fixed iteration counts and clipping without reporting bracket validity or an EOS residual. JIT execution alone does not establish a valid recovered state.

## Checks that passed and limits of this audit

A separate exact-source TwoFluid flux probe used a smooth ideal-gas stub, equal left/right states `[100000, 10, 10, 0.4, 300000, 100000]`, and `k=0.5`. The flux value, `jax.jacfwd`, and `jax.jacrev` were finite for both `single_branch=False` and `single_branch=True`. Therefore this audit does **not** claim that the zero-slip treatment or branch selection universally fails AD. Other thermodynamic states, zero velocity, near-degenerate phases, and batched higher derivatives were not validated by that probe.

The local custom-JVP implementations linearize each face or cell and apply its local Jacobian to tangents. The source review found no basis to call those local rules intrinsically wrong. Their promises remain conditional on valid property derivatives, valid states, and the global stencil assumptions described above.

## Suggested follow-up, after the structural refactor

- Define supported differentiation targets separately: state, boundary data, geometry, closure parameters, and fluid/table parameters; distinguish forward from reverse mode and one-step from converged-solve sensitivities.
- Add equal-state and stationary-state flux checks before claiming all EulerNozzle kernels are valid on those states.
- Compare EOS recovery sensitivities with analytical identities or finite differences, including boundary recovery.
- Check dense versus colored Jacobians for every limiter, filter, and boundary combination that changes dependency width; include directional finite differences when derivatives have been stopped.
- Give physical transient marching and steady pseudo-time iteration explicit separate contracts.
- Preserve the current implementations until a separate numerical/JAX repair task establishes intended behavior and regression evidence.
