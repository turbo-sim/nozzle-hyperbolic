# Two-Fluid Source Terms — Implementation Notes

Reference implementation: [`sources.py`](sources.py).
Model: Städtke's 6-equation two-fluid system (gas `g` = vapour, liquid `l`),
one-component (single substance) with phase change. Companion (space-marching)
reference: nozzlex `nozzle_two_fluid_one_component_autonomous`.

All source quantities are **per unit volume**, evaluated **per cell**, and are
scalars. The physical sign convention throughout is:

> `sigma_Mg > 0` ⇒ **evaporation** (liquid → gas).
> `Fv_g > 0` ⇒ interfacial drag accelerates the gas in the `+x` direction.

---

## 1. Pipeline overview

The source enters the conservative residual as `− dt · V_cell · D_VE` (see
`solver.py::_residual`). `D_VE` is a 6-vector aligned with the conservative
variables `V = (m_g, m_l, m_g u_g, m_l u_l, E_g, E_l)`.

There are **two equivalent routes** from physical closures to `D_VE`:

```
                       ┌─ D_s_primitive ──► Φ⁻¹ ──► J ──►  D_VE_conservative   (entropy frame)
 two_fluid_closure ──►─┤
   (physical inputs)    └──────────────────────────────►  D_VE_algebric        (direct, used in runs)
```

- **`D_VE_conservative`** builds the primitive **entropy-frame** source `D_s`
  (Städtke Table B.46), maps it through `Φ⁻¹` (entropy → enthalpy frame) and then
  `J = ∂V/∂U_h` to conservative variables. Pedagogically transparent, matches the
  matrix derivation.
- **`D_VE_algebric`** writes the conservative source **directly** as closed-form
  expressions `D1…D6`. This is the form wired into the demo runs
  (`main_phase_change.py::_sources_fn`). It is algebraically equivalent to the
  conservative route for the mass/momentum/energy content, **but currently omits
  area-change and wall-friction contributions** (see §7).

Both consume the same `TwoFluidClosure` produced by `two_fluid_closure`.

---

## 2. `TwoFluidClosure` — the physical inputs

`two_fluid_closure(U_h, eos_g, eos_l, fluid_g, fluid_l, T_sat, h_sat_g, h_sat_l, …)`
packs the following per-cell scalars (all consumed downstream by `D_s_primitive`
and `D_VE_algebric`):

| field       | meaning                                      | units      |
|-------------|----------------------------------------------|------------|
| `sigma_Mg`  | gas mass source (evaporation rate)           | kg/(m³·s)  |
| `sigma_Qg`  | interfacial heat to gas   (`Qvi`)            | W/m³       |
| `sigma_Ql`  | interfacial heat to liquid (`Qli`)          | W/m³       |
| `Fv_g`      | interfacial drag on gas                      | N/m³       |
| `F_g, F_l`  | external body forces (gravity, …) — 0 here   | N/m³       |
| `Q_g, Q_l`  | wall/external heat — 0 here                  | W/m³       |
| `u_ex`      | velocity of the transferred mass             | m/s        |
| `u_int`     | interface velocity (for interfacial work)    | m/s        |
| `h_ex_g`    | enthalpy carried into the gas phase          | J/kg       |
| `h_ex_l`    | enthalpy carried out of the liquid phase     | J/kg       |
| `dlnA_dx`   | `(1/A) dA/dx` — log-area gradient            | 1/m        |

Inputs `T_sat, h_sat_g, h_sat_l` are supplied **externally** from a saturation
EOS call at the cell pressure `p`, so `two_fluid_closure` stays jit-/AD-friendly
(no secondary EOS solve inside).

---

## 3. Interfacial-structure and transfer closures

Computed inside `two_fluid_closure` (§ the code block after the EOS unpack):

### 3.1 Interfacial area & dispersed diameter
```python
ai, d = interfacial_area(ag, alpha_b=0.3, alpha_d=0.7, Nb, Nd, d_min_b=5e-5)
```
TA-BD model with a nucleation floor. Regimes vs vapour fraction `α_g`:
bubbly (`α_g ≤ 0.3`), linear transition, droplet (`α_g ≥ 0.7`), plus floors.
`d_min_b = 5e-5` matches nozzlex (inert in the bubbly regime).
Returns `ai` [1/m] and characteristic diameter `d` [m].

### 3.2 Drag force
```python
Fv_g = -calculate_drag_force_one_component(ag, al, rg, rl, mu_g, mu_l, u_g, u_l, d, …)
```
Regime-dependent Schiller–Naumann drag. The **sign is flipped** here so that the
returned `Fv_g` is the force **on the gas**; the underlying correlation returns a
magnitude `∝ ¾/d · |Δu| Δu · C_d · α ρ_c`, where the continuous-phase density `ρ_c`
is liquid (bubbly) or gas (droplet). Transport `mu_g, mu_l` come from the extended
EOS lookup. `Fv_l = −Fv_g` downstream.

### 3.3 Heat-transfer coefficients
```python
htc_g = k_g · Nu / d        with Nu = 12          # gas-side, conduction-limited
htc_l = ranz_marshall_htc_two_fluid(liquid_props, |u_g−u_l|, d)   # liquid-continuous
```
`htc_g` is a fixed-Nusselt conduction estimate through the vapour film.
`htc_l` uses **liquid** transport properties (`stl_ex`) as the continuous phase
(bubbly regime) — Ranz–Marshall `Nu = 2 + 0.6 Re^½ Pr^⅓`. Both match nozzlex.

### 3.4 Interfacial heat & mass transfer
```python
Qvi = htc_g · ai · (T_sat − T_g)         # W/m³, interface → gas
Qli = htc_l · ai · (T_sat − T_l)         # W/m³, interface → liquid
sigma_Mg = calculate_mass_transfer_lv(…) = −(Qvi + Qli) / (h_sat_g − h_sat_l)
```
Mass transfer is the **energy-jump closure** (Rane & He): the net interfacial
heat imbalance is released/absorbed as latent heat `h_lv = h_sat_g − h_sat_l`.
This is **continuous in the state and bidirectional by sign**:
`T_l > T_sat ⇒ Qli < 0 ⇒ σ_Mg > 0` (evaporation); `T_l < T_sat ⇒ σ_Mg < 0`
(condensation).

> **Gate note.** `two_fluid_closure` still contains a `jax.lax.cond` on an
> `is_two_phase = (h_l ≥ h_sat_l) & (T_l ≥ T_sat)` flag, but it is currently
> overridden to `is_two_phase = True`, i.e. transfer is **always on**. The gated
> (flashing-only) branch zeroes all transfer once the liquid is subcooled, which
> both freezes the relaxation and — being discontinuous — prevents the implicit
> solver from settling at `T_sat`. Always-on transfer is the physically correct
> choice for a coexisting two-fluid mixture.

---

## 4. Interface kinematics — `u_ex`, `u_int`, `h_ex`

These set **what velocity and enthalpy the transferred mass carries**, and appear
in the momentum and energy source rows. Current implementation:

```python
u_mix2 = (ag·rg·u_g + al·rl·u_l) / (ag·rg + al·rl)   # mass-weighted (computed, not used)
u_ex   = |u_g + u_l| / 2      # velocity of transferred mass
u_int  = u_ex                 # interface velocity (interfacial-work term)
h_ex_g = h_sat_g              # enthalpy INTO the gas   (saturated vapour)
h_ex_l = h_sat_l              # enthalpy OUT of the liquid (saturated liquid)
```

### Rationale (LTE)
Under **local thermodynamic equilibrium** the interface sits on the saturation
line, so mass crossing it is exchanged at the **saturation enthalpy of each
side** — gas side `h_sat_g`, liquid side `h_sat_l` — **independent of transfer
direction**. The latent heat `h_lv = h_sat_g − h_sat_l` is carried by the
interfacial heat `Qvi + Qli`; combined with the mass closure `σ_Mg = −(Qvi+Qli)/h_lv`
this makes the total-energy source conservative:
`Σ D_E = σ_Mg·h_lv + (Qvi+Qli) = 0` (latent heat cancels interfacial heat).

### Choices & caveats (open TODOs in code)
- **`u_ex = |u_g + u_l|/2`** — a simple average of the two phase velocities. The
  absolute value is a guard; the mass-weighted `u_mix2` is computed but **not**
  used. Whether the transferred mass should carry the arithmetic mean, the
  mass-weighted mean, or the donor-phase velocity is model-dependent (TODO: sign
  and definition of `u_ex`/`u_int`).
- **`u_int = u_ex`** — the interface velocity used in the interfacial-work terms
  `Fv·(u_int − u_k)` is taken equal to `u_ex`.
- **`h_ex` direction** — currently fixed to saturation on each side for both
  directions (LTE, as above). The code TODO flags this as a point to revisit if a
  non-equilibrium (donor-cell) enthalpy is ever wanted.

---

## 5. `D_s_primitive` — entropy-frame source (Table B.46)

Returns `D_s = [D_p, D_{u_g}, D_{u_l}, D_α, D_{s_g}, D_{s_l}]`. Key building blocks:

**Entropy production per phase** (Städtke eq. 4.50) — sensible + kinetic +
interfacial-work + heat:
```
T_g σ_g^S = σ_Mg·[(h_ex_g − h_g) + ½(u_ex − u_g)²] + Fv_g·(u_int − u_g) + Q_vi + Q_g
T_l σ_l^S = σ_Ml·[(h_ex_l − h_l) + ½(u_ex − u_l)²] + Fv_l·(u_int − u_l) + Q_li + Q_l
σ_k^S = (T_k σ_k^S) / T_k
```
with `σ_Ml = −σ_Mg`, `Fv_l = −Fv_g`.

**Recurring mass+entropy group:** `M_k = σ_Mk − (T_k ρ_{h,k}/ρ_k) σ_k^S`.

**Rows** (using `CommonQty`, see §8; `k_ρ = k·ρ_mix`):
- **Row 1 (pressure):** `ρ_l a₀²/ρ_s · M_g + ρ_g a₀²/ρ_s · M_l − (area term)`
- **Row 2/3 (phase velocities):** drag + mass-transfer momentum, with the
  `k → k·ρ_mix` virtual-mass correction applied to every bare `k` in B.46
  (Städtke's `k` next to a density is a typo). Body-force terms `F_g, F_l` are
  present but commented out (0 here).
- **Row 4 (void fraction):** `M_g`/`M_l` weighted by `a₀²/ρ_s/a_k²` + area term.
- **Rows 5/6 (entropy):** `σ_g^S/(α_g ρ_g)`, `σ_l^S/(α_l ρ_l)`.

The area terms carry `dlnA_dx` and vanish for the constant-area pipe.

---

## 6. `D_VE` — conservative sources

### 6.1 `D_VE_conservative` (matrix route)
```
D_s  ──Φ⁻¹──►  D_h        (only rows 4,5 change:
                            D_h[4] = D_s[0]/ρ_g + T_g·D_s[4]
                            D_h[5] = D_s[0]/ρ_l + T_l·D_s[5])
D_h  ──J────►  D_VE       (J = ∂V/∂U_h = J_matrix(U_h, eos_g, eos_l))
```
`Φ⁻¹` maps the entropy frame to the enthalpy frame; `J` maps enthalpy-frame
primitives to conservative variables.

### 6.2 `D_VE_algebric` (direct route — used in runs)
Closed-form conservative source. Auxiliary groups:
```
A_common     = k·ρ_mix·(α_g ρ_g u_l + α_l ρ_l u_g) − α_l α_g (ρ_l² − ρ_g²)(u_g − u_l)
density_term = ρ_g ρ_l / rh2 − 1
```
Rows:
```
D1 = σ_Mg                                                            # gas mass
D2 = σ_Ml = −σ_Mg                                                    # liquid mass
D3 = ρ_g ρ_l/rh2 · (Fv_g + σ_Mg·u_ex) + σ_Mg/rh2 · A_common          # gas momentum
D4 = ρ_g ρ_l/rh2 · (Fv_l + σ_Ml·u_ex) + σ_Ml/rh2 · A_common          # liquid momentum
D5 = σ_Mg·(h_ex_g + ½u_ex² + u_ex u_g·density_term + u_g A_common/rh2)
     + Fv_g·(u_int + u_g·density_term) + Q_vi                        # gas energy
D6 = σ_Ml·(h_ex_l + ½u_ex² + u_ex u_l·density_term + u_l A_common/rh2)
     + Fv_l·(u_int + u_l·density_term) + Q_li                        # liquid energy
```
- `D3/D4`: drag + mass-transfer momentum, coupled through the virtual-mass factor
  `rh2` and the slip term `A_common`.
- `D5/D6`: enthalpy of transferred mass (`h_ex`), kinetic (`½u_ex²`), slip/virtual-
  mass work (`density_term`, `A_common`), interfacial drag work (`u_int`), and
  interfacial heat (`Q_vi/Q_li`). The latent-heat/interfacial-heat cancellation of
  §4 holds in `D5 + D6`.

---

## 7. Known limitations / TODOs (from the code)

- **Gate override.** `is_two_phase` is force-set to `True` (transfer always on).
  The dormant gated branch is a flashing-only (HNEM) switch — see §3.4.
- **`D_VE_algebric` omits area-change and wall friction.** Only phase-change,
  drag and interfacial heat are included; geometric (`dlnA_dx`) and wall-shear
  sources are not yet added to the algebraic form (they *are* present in the
  entropy-frame `D_s_primitive` rows). Fine for the constant-area, frictionless
  pipe demo; must be added for nozzles/friction.
- **`u_ex` / `u_int` definition** — arithmetic mean with an `abs()` guard;
  mass-weighted alternative computed but unused. Definition/sign to be confirmed.
- **`h_ex` direction** — fixed to per-side saturation (LTE). Revisit for
  non-equilibrium transfer.
- **Velocity difference in the mass-source denominator** — code TODO questions
  whether a `Δu` term belongs in `calculate_mass_transfer_lv`.

---

## 8. Symbol glossary (`CommonQty`, from `matrices.py`)

| symbol      | definition                                    |
|-------------|-----------------------------------------------|
| `rho_mix`   | `α_g ρ_g + α_l ρ_l`  (mixture density)        |
| `rho_s`     | `α_g ρ_l + α_l ρ_g`  (cross-weighted density) |
| `rh2`       | `ρ_g ρ_l + k ρ_mix²` (virtual-mass modified)  |
| `rg_hat`    | `ρ_g + k ρ_mix`                               |
| `rl_hat`    | `ρ_l + k ρ_mix`                               |
| `a0sq`      | `a₀²`, frozen mixture sound-speed combination |
| `du`        | `u_g − u_l` (slip)                            |
| `k`         | virtual-mass coefficient (`k_vm`, e.g. 0.5)   |
| `k_rho`     | `k ρ_mix = rg_hat − ρ_g`                       |
| `rho_h,k`   | `(∂ρ_k/∂h)_p` from the EOS (`eos_k.rho_h`)    |

Primitive state `U_h = (p, u_g, u_l, α_g, h_g, h_l)`; conservative
`V = (m_g, m_l, m_g u_g, m_l u_l, E_g, E_l)` with `m_k = α_k ρ_k`,
`E_k = m_k e_k + ½ m_k u_k²`.
