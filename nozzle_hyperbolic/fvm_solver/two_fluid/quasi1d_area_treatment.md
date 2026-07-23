# Quasi-1D area treatment in the Städtke flux-vector-splitting two-fluid scheme

## Purpose

This note fixes the bookkeeping for cross-sectional area `A(x)` in a **quasi-1D**
implementation of the Städtke hyperbolic two-fluid model. The central question is:

> Should the cross-sectional area `A` multiply the flux inside the
> **non-conservative** term, or only inside the conservative flux?

**Answer:** In the quasi-1D formulation you are implementing, the area multiplies
**only the conservative flux**. The non-conservative (NC) term uses a **plain**
flux gradient, and all remaining geometric effects enter as an **explicit area
source** `D_A`. Do **not** area-weight the NC term *and* add `D_A` — that
double-counts the geometry.

The rest of this note derives why, and reconciles it with Städtke's finite-volume
equation (8.28), which *does* appear to area-weight the NC term.

---

## 1. Notation

- `A = A(x)` — physical cross-sectional area (quasi-1D duct geometry).
- `G = (1/A) dA/dx` — logarithmic area gradient.
- `alpha_k` — void fraction of phase `k` (`k = g, l`).
- `rho_k`, `u_k`, `p`, `s_k` — density, velocity, pressure, entropy.
- Conservative flux (gas momentum row example): `F_g^mom = alpha_g rho_g u_g^2 + alpha_g p`.
- `H^nc` — non-conservative coefficient matrix (acts only on the two momentum rows).
- `[J] = dV/dU` — Jacobian from primitive `U` to conservative `V` variables.

Two distinct "areas" appear in Städtke's book and must not be conflated:

| Symbol | Meaning | Where it lives |
|---|---|---|
| `A(x)` | physical duct cross-section (quasi-1D) | flux weighting + `D_A` source |
| `A_s`, `V_i` | mesh face area / cell volume (multi-D FV measures) | Gauss's theorem in (8.28) |

---

## 2. Exact quasi-1D momentum equation

Start from the exact quasi-1D gas momentum balance (geometry fixed, `A = A(x)` only):

```
d/dt (A alpha_g rho_g u_g) + d/dx (A alpha_g rho_g u_g^2) + A alpha_g dp/dx = A * RHS_g
```

Divide by `A` to return to per-unit-volume conservative variables `V`. The only
awkward term is `alpha_g dp/dx`. Split it, **including the area**, using

```
alpha_g dp/dx = (1/A) d/dx (A alpha_g p)  -  alpha_g p G  -  p d(alpha_g)/dx
```

Substituting gives the honest continuum form:

```
d/dt (alpha_g rho_g u_g)
    + (1/A) d/dx ( A F_g^mom )
    = RHS_g
      + alpha_g p G          <-- wall-pressure source  -> goes into D_A
      + p d(alpha_g)/dx       <-- interfacial NC product -> equals -H^nc dF/dx
```

Read off the three geometric destinations:

1. **`(1/A) d/dx (A F)`** — conservative flux, **area-weighted**.
2. **`alpha_g p G`** — wall-pressure source, component of `D_A`.
3. **`p d(alpha_g)/dx`** — interfacial non-conservative product, `= -H^nc dF/dx`.

**Key fact:** term 3 carries **no factor of `A`**. It is a pure differential
product that looks identical in a straight pipe or a nozzle. Therefore the NC
term must be discretized with a **plain, un-area-weighted** flux difference.
Area-weighting it would double-count the geometry.

---

## 3. Discretization (the scheme to implement)

```
dV_i/dt = - (1 / (A_i dx)) ( A_{i+1/2} F_{i+1/2} - A_{i-1/2} F_{i-1/2} )   # conservative, area-weighted
          - H^nc_i * ( F_{i+1/2} - F_{i-1/2} ) / dx                        # NC, PLAIN gradient
          + D_i
```

with

```
D_i = D_int( sigma_M, F_v, sigma_Q_g, sigma_Q_l )   # interfacial sources
      + D_A                                          # area sources
```

and the area source vector (rows: mass_g, mass_l, mom_g, mom_l, void, entropy...):

```
D_A = [ 0, 0, alpha_g p G, alpha_l p G, -p D_alpha, +p D_alpha, ... ]^T
```

```
D_alpha = alpha_g alpha_l (a0^2 / rho_s) ( rho_l u_l / a_g^2  -  rho_g u_g / a_l^2 ) G
```

(Match the exact row placement of `D_alpha` to your variable ordering — see the
caveat in Section 6.)

---

## 4. Why (8.28) *looks* like it area-weights the NC term

Städtke's finite-volume equation:

```
V_i^{n+1} = V_i^n
    - (dt / V_i) sum_s A_s (F_s)
    - (dt / V_i) sum_s A_s (H^nc_s)(F_s)
    + D_i dt
```

Here `A_s` is the **mesh boundary-segment area** and `V_i` is the **cell volume** —
the Gauss/finite-volume measures for an arbitrary polygon. The scheme is written
for a mesh that **physically resolves** the geometry. Both the conservative flux
and the NC term are turned into face sums by the *same* divergence identity:

```
integral_V dF/dx dV = closed_surface F n_x dS   =>   (dF/dx)_i ~ (1/V_i) sum_s A_s F_s
```

So in the **general multi-D scheme**, area-weighting the NC term is correct —
but that `A_s` is the **mesh face area**, not the physical duct cross-section
`A(x)`. In this partition, `D_i` contains **no geometric source**: all geometry
is carried by `A_s` and `V_i`.

---

## 5. The two self-consistent partitions

Both discretize the *same* continuum equations. They differ only in how the
geometry is split between "flux weighting" and "source". **Pick one. Never mix.**

### Partition 1 — geometry-resolved (Städtke multi-D, literal 8.28)
- Area-weight **both** the conservative flux and the NC term with the **mesh**
  measures `A_s`, `V_i`.
- `D_i` is the fully `[J]`-transformed interfacial source with **no** area term.
- Reducing literal (8.28) to 1D produces an NC contribution
  `~ H^nc ( dF/dx + F G )`. The spurious `H^nc F G` piece is **cancelled by a
  matching counter-term inside the transformed `D_i`** (which is why the book's
  `D` in the primitive form `dU/dt + G dU/dx = D` carries *all* the geometry as
  source and area-weights nothing).

### Partition 2 — quasi-1D (what you are implementing)
- Area-weight **only the conservative flux**, with the **physical** `A_{i±1/2}`.
- Keep the NC gradient **plain** (`~ H^nc dF/dx`).
- Add `D_A` with the **physical** geometric pieces (`alpha_k p G` in momentum,
  plus void/entropy area terms).
- **No counter-term is needed**, because you never generate the `H^nc F G`
  artifact in the first place.

Your momentum treatment is exactly Partition 2 and is **correct**: the plain NC
gradient reproduces `-p d(alpha_g)/dx`, and `alpha_g p G` handles wall pressure
and well-balancing.

**The only bug would be mixing the partitions**, e.g. area-weighting the NC term
*and* adding `alpha p G` (double count), or keeping the plain NC gradient but also
including the Partition-1 counter-term.

---

## 6. Caveat: where does `D_alpha` belong?

In Städtke's Table B.46 (primitive source `D` for quasi-1D), the `D_alpha` term
lives in the **void-fraction row**, not the entropy rows. If your `D_A` places
`∓ p D_alpha` in the **entropy** components, that is only legitimate if it is the
**image of the primitive void source after the `[J] = dV/dU` transformation** into
conservative variables.

- If you built `D_A` by transforming the *entire* Table B.46 vector (interfacial
  **and** area terms together) through `[J]`, it is consistent — proceed.
- If you placed `D_alpha` into the entropy rows **by analogy / by hand**, audit
  this: derive the phasic entropy equation directly,

  ```
  d/dt (alpha_g rho_g s_g) + (1/A) d/dx ( A alpha_g rho_g u_g s_g ) = sigma_g^s
  ```

  which has **no standalone area source** — area-weighting the entropy flux is
  complete. The area effect should surface only through the `[J]` transform, not
  as an independent hand-placed term.

**Action for the implementation:** confirm `D_A` is produced by applying `[J]` to
the full primitive area source, rather than assembling rows individually.

---

## 7. Regression test: stagnant nozzle (well-balancing)

Set `u_g = u_l = 0`, uniform `p`, uniform `alpha_k`, variable `A(x)`. The fluid
must stay exactly at rest. Check the gas momentum row:

- Conservative flux is `F^mom = alpha_g p` (uniform in space).
  Area-weighted difference:
  `- alpha_g p (A_{i+1/2} - A_{i-1/2}) / (A_i dx)  ~  - alpha_g p G`.
- NC term: flux difference of a uniform quantity is **0**.
- `D_A`: `+ alpha_g p G`.

Sum = `- alpha_g p G + 0 + alpha_g p G = 0`. Momentum residual is zero => velocity
stays zero. If your code holds `u = 0` to machine precision on a converging/
diverging duct with uniform `p` and `alpha`, the area partition is consistent.
This is the single most useful unit test to add.

Also verify:
- Setting `A = const` (`G = 0`) recovers your validated straight-pipe results
  exactly (`D_A = 0`, area weights all equal).
- Single-phase limits (`alpha_g -> 1` and `alpha_g -> 0`) reduce to the standard
  quasi-1D Euler nozzle source (`alpha p G` in momentum only).

---

## 8. Summary

- Quasi-1D is "special" because the duct shape is **not** resolved by a mesh; it
  is represented analytically by `A(x)`, which forces the geometry to reappear as
  an explicit source `D_A`.
- Consequence: **area-weight the conservative flux only**, keep the **NC gradient
  plain**, and add `D_A`.
- Städtke's (8.28) area-weights the NC term because its `A_s` is a **mesh face
  measure** (Gauss's theorem), not the physical `A(x)`; in that partition the
  geometry lives entirely in `A_s`/`V_i` and `D_i` carries the cancelling
  counter-term — a different but equivalent bookkeeping.
- **Do not mix the two partitions.** You are correctly using the quasi-1D one.
- Open item to verify: `D_alpha` placement must come from the `[J]` transform of
  the primitive void source, not hand-placement in the entropy rows.
