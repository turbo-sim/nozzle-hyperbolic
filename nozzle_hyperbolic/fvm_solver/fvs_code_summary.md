# FVS Implementation Summary (single-phase) — reference for the two-fluid extension

This documents **how the Städtke flux-vector-splitting is actually coded** in
[`fvm.py`](fvm.py), as a starting point for deriving the analogous two-fluid
(two-phase, non-equilibrium) equations. It maps each math object to its code
location and flags every place that must change when the system grows beyond 3
equations. For the full derivation see
[`stadtke_single_phase_theory.md`](stadtke_single_phase_theory.md); this file is
the *implementation* companion.

All flux code lives in `_jax_F_fvs_stadtke(U_L, U_R, fluid, delta_factor)`
([fvm.py:1238](fvm.py)). It is fully vectorised over faces: every array has a
leading `(Nfaces,)` axis and the 3×3 matrices are `(Nfaces, 3, 3)`.

---

## 1. Variable sets (the single most important design choice)

The solver carries **three representations** of the state. Getting the two-fluid
analogue of these three is the first task.

| Symbol | Name | Components (single-phase) | Code |
|--------|------|---------------------------|------|
| `U` | **primitive** (working/unknown vars) | `(p, u, h)` | columns of `U_L`, `U_R` |
| `V` | **conservative** (time-derivative vars) | `(ρ, ρu, ρH − p)` | `_V_from_U` ([fvm.py:1652](fvm.py)) |
| `F` | **physical flux** | `(ρu, ρu² + p, uρH)` | `F_L`, `F_R` ([fvm.py:1334](fvm.py)) |

with total enthalpy `H = h + ½u²`.

- The Newton unknowns are the **primitive** `U = (p, u, h)` per cell. This is
  deliberate: `(p, h)` are the natural EOS inputs, so no inversion is needed to
  call the equation of state.
- `V` is what is actually conserved in time (`∂V/∂t + …`). Note the energy
  variable is `ρH − p = ρ(e + ½u²)` (total energy per volume), **not** `ρE`
  directly — written this way because `H` and `p` are primitives.
- `F` is evaluated from the **actual** left/right states, never from the average
  (the average is used only to build the split matrices).

**Two-fluid implication:** the per-cell state vector grows from 3 to N
(e.g. 6 for a 1-pressure 2-velocity 2-temperature model, or 7 with void
fraction transport). You must decide the primitive set (likely
`α, p, u_g, u_l, h_g, h_l` or similar) and write the analogues of `_V_from_U`
(primitive→conservative) and the physical flux `F`.

---

## 2. EOS interface — the only physics input

Everything else is algebra built on top of these EOS calls. The flux needs, at
the **averaged** face state `(h_a, p_a)`:

```python
st_a  = fluid.get_state(jxp.HmassP_INPUTS, h_a, p_a)   # fvm.py:1257
rho_a = st_a["rho"]      # density
a_a   = st_a["a"]        # speed of sound
G_a   = st_a["G"]        # Grüneisen parameter Γ = (1/ρ)(∂p/∂e)_ρ
rho_p = (1.0 + G_a) / a_a**2       # (∂ρ/∂p)_h     fvm.py:1261
rho_h = -(rho_a * G_a) / a_a**2    # (∂ρ/∂h)_p     fvm.py:1262
```

`rho_p` and `rho_h` are the two thermodynamic derivatives that close the system.
They come from differentiating `ρ(h, p)` and are the bridge between the EOS and
the Jacobian. The sound speed enters only through these.

**Two-fluid implication:** this is where HEM vs. two-fluid diverges most. You
need per-phase densities and their derivatives, plus the closure for how phasic
pressures/velocities relate. The mixture sound speed (and hence the eigenvalues)
is no longer a single EOS output — it comes from the eigenstructure of the
larger system. `get_state` must return the phasic properties; `rho_p`/`rho_h`
generalise to a block of derivatives.

---

## 3. Eigenstructure (hand-coded, not computed numerically)

### Eigenvalues ([fvm.py:1265](fvm.py))
```python
lam1 = u_a + a_a    # acoustic, downstream-running
lam2 = u_a - a_a    # acoustic, upstream-running
lam3 = u_a          # entropy/contact wave
```

### Right eigenvectors `v_k` and left eigenvectors `l_k` ([fvm.py:1272-1277](fvm.py))
```python
v1 = ( ρa,  1,  a)      l1 = ( 1/2ρa,  1/2,  0)
v2 = (-ρa,  1, -a)      l2 = (-1/2ρa,  1/2,  0)
v3 = ( 0,   0,  1)      l3 = (-1/ρ,    0,    1)
```
These satisfy `l_j · v_k = δ_jk` (bi-orthonormal) and are written in **primitive
`(p,u,h)` space**.

**Two-fluid implication:** for the two-fluid model the eigenstructure is the hard
part — it is generally **not available in closed form** and the system can be
non-hyperbolic (complex eigenvalues) without proper interfacial-pressure terms.
This block is what most of the two-fluid literature is about. Either (a) derive
approximate analytic eigenvectors, or (b) compute the eigendecomposition
numerically per face (much more expensive, but the rest of the machinery below
is unchanged).

---

## 4. Flux Jacobian K = ∂F/∂U ([fvm.py:1279-1285](fvm.py))

Built explicitly (in primitive variables) as a `(Nfaces, 3, 3)` matrix:
```python
H_a = h_a + 0.5*u_a**2
K = [[ u·ρ_p,             ρ,                    u·ρ_h            ],
     [ u²·ρ_p + 1,        2ρu,                  u²·ρ_h           ],
     [ u·ρ_p·H,           ρ(H + u²),            u(ρ_h·H + ρ)     ]]
KT = K.swapaxes(-1, -2)   # transpose, used in the left-vector solve
```

This is the derivative of the physical flux `F` w.r.t. the primitive `U`. It is
the object that transforms the splitting from primitive space (where the
eigenvectors are simple) into flux space (where the upwinding happens).

**Two-fluid implication:** straightforward but tedious — differentiate the new
`F(U)` analytically. Grows to N×N. This is mechanical once `F` and the EOS
derivatives are fixed.

---

## 5. The projection matrices R̃_k (rank-1 construction) ([fvm.py:1306-1320](fvm.py))

The core trick. Each wave `k` gets a projection matrix
`R̃_k = K · G_k · K⁻¹ / λ_k`, where `G_k = λ_k v_k ⊗ l_k` is the rank-1 split
matrix in primitive space. Substituting collapses to a **rank-1 outer product**
that avoids ever forming `K⁻¹` densely:

```python
def _Rt_vec(v, l, lam_avg, lam_L, lam_R):
    Kv   = K @ v                          # K v_k          (right side)
    KTl  = solve(KT, l)                   # K⁻ᵀ l_k        (left side)
    ...entropy-fix coeff...
    return coeff * outer(Kv, KTl)         # (Nfaces, 3, 3)
```

So `R̃_k = coeff_k · (K v_k) ⊗ (K⁻ᵀ l_k)`. The only linear solve is
`solve(KT, l)` — one 3×3 system per wave per face.

**Two-fluid implication:** the rank-1 form survives unchanged for any number of
waves, *provided* you have `v_k` and `l_k`. The `solve(KT, l)` becomes N×N. If
the eigenvectors are obtained numerically, you already have `K⁻¹` and can skip
the solve.

---

## 6. Entropy fix — sonic-crossing only ([fvm.py:1306-1316](fvm.py))

Dissipation is added **only** at a face where a wave changes sign from left to
right (an expansion sonic point), using the C¹ Harten form:

```python
sonic    = (lam_L < 0) & (lam_R > 0)                  # wave crosses zero L→R
delta_k  = where(sonic, lam_R - lam_L, delta_global)
lam_safe = where(sonic, (lam_avg² + delta_k²)/(2 delta_k),  # smoothed
                        max(|lam_avg|, delta_global))        # otherwise exact
coeff    = |lam_avg| / lam_safe
```

`delta_global = delta_factor·(|u| + a)` is an optional global floor for
robustness. Crucially, dissipation is **not** added at shocks or in smooth
regions — `coeff = 1` there.

**Two-fluid implication:** same logic per wave. The L/R eigenvalues need a second
set of EOS calls at the actual L/R states ([fvm.py:1288-1293](fvm.py)).

---

## 7. Upwind assembly and the interface flux ([fvm.py:1322-1343](fvm.py))

Split by the **sign of the true (averaged) eigenvalue**, then combine with the
**physical** L/R fluxes:

```python
s_k+    = (lam_k >= 0)                       # 1 if wave runs downstream
R_plus  = Σ_k  s_k+      · R̃_k              # waves carrying info from the left
R_minus = Σ_k (1 - s_k+) · R̃_k              # waves carrying info from the right
F̂ = R_plus @ F_L + R_minus @ F_R            # fvm.py:1342
```

`s_k+` is the hard `where(lam >= 0, ...)` switch. (Noted in `PLAN_shock_capture.md`
as a place a smooth `tanh` could improve the Newton Jacobian near M=1; "Strategy
C". Not yet applied.)

**Two-fluid implication:** identical structure, summed over the larger wave set.

---

## 8. How the flux feeds the residual ([fvm.py:1820-1859](fvm.py))

`F̂` is computed at every face from the MUSCL-reconstructed `U_L, U_R`, then the
conservative residual for interior cell `i` is **area-weighted**:

```python
F_hat = _jax_F_fvs_stadtke(U_L, U_R, fluid, delta_factor)         # (N-1, 3)
fdiv  = A_face[1:]·F_hat[1:] - A_face[:-1]·F_hat[:-1]              # net flux out
R     = V_cell·(V_new - V_old) + dt·fdiv - dt·V_cell·src           # = 0 at steady state
```

The source `src` carries the quasi-1D area term `p·(dA/dx)/A` and wall friction
in the **momentum** component only ([fvm.py:1849](fvm.py)).

**Two-fluid implication:** this is where the **interfacial transfer terms**
(mass/momentum/energy exchange between phases, interfacial pressure work) enter
as additional source terms. The flux-divergence structure is unchanged; the new
physics is mostly in `src`.

---

## 9. Checklist to port to the two-fluid model

In rough order of difficulty:

1. **Define the three variable sets** `U, V, F` (§1) for the chosen two-fluid
   formulation. Write `_V_from_U` and the physical `F`.
2. **Extend the EOS interface** (§2): phasic `ρ, a` and derivatives.
3. **Derive the eigenstructure** (§3) — the genuinely hard step; decide analytic
   vs. numerical. Watch for loss of hyperbolicity → needs interfacial-pressure
   regularisation.
4. **Differentiate the flux** `K = ∂F/∂U` (§4) — mechanical, grows to N×N.
5. **Reuse the rank-1 projection + entropy fix + upwind sum** (§5–7) verbatim,
   just looping over more waves. The `jnp.linalg.solve` and `einsum` calls are
   already shape-generic.
6. **Add interfacial source terms** to `src` (§8).

The machinery in §5–8 is essentially **size-agnostic**; the real two-fluid work
is §1–4 (variables, EOS closure, eigenstructure, Jacobian).
