# Verification Plan: Two-Fluid FVS Solver Implementation

*Companion to `stadtke_enthalpy_frame_derivation.md` and `stadtke_twofluid_implementation_pipeline.md`. This plan specifies the staged verification sequence: what to test, in what order, against what reference, with what pass criterion. The staging mirrors the implementation order — each stage tests one building block in isolation before integration with the next.*

---

## 0. Approach and conventions

**Verification, not validation.** This plan checks that the *code* correctly implements the *algebra*. Validation against experimental data is a separate exercise (see §9 of the pipeline). The reference for verification is the symbolic and structural identities established in the derivation document.

**Strict tolerance tiers.**

| Tier | Tolerance | Used when |
|------|-----------|-----------|
| Machine | $\lesssim 10^{-14}$ relative | exact algebraic identity (no truncation expected) |
| EOS-bounded | $\sim 10^{-10}$ | calls real-gas tables; bounded by EOS interpolation |
| Discretization | $\sim 10^{-4}$ | spatial/temporal discretization expected |
| Physical | $\sim 10^{-2}$ | benchmark comparisons with reference codes |

A test passes only at its specified tier. A "machine" test failing at $10^{-10}$ indicates a real bug, not floating-point noise.

**Nominal reference state.** Unless specified otherwise, use $p=10^5\,\mathrm{Pa}$, $\alpha_g=0.25$, $u_g=100\,\mathrm{m/s}$, $u_l=50\,\mathrm{m/s}$, $h_g=2.5\times 10^6\,\mathrm{J/kg}$, $h_l=4\times 10^5\,\mathrm{J/kg}$, $\varrho_g=5\,\mathrm{kg/m^3}$, $\varrho_l=960\,\mathrm{kg/m^3}$, $a_g=450\,\mathrm{m/s}$, $a_l=1500\,\mathrm{m/s}$, $T_g=T_l=453\,\mathrm{K}$, $\Gamma_g=0.3$, $\Gamma_l=0.2$, $k=0.5$. This is the state used in the derivation-doc verification ledgers.

**Strict dependency ordering.** No stage advances until the prior stage's tests all pass. A bug in stage 2 surfacing as a failure in stage 5 will be debug hell.

---

## 1. EOS layer

### 1.1 EOS closure identity
For each phase at the reference state, compute $\varrho_p, \varrho_h$ from the same EOS call as $a, \Gamma, \varrho$. Check:
$$\Bigl|\frac{1}{a_i^2} - \varrho_{p,i} - \frac{\varrho_{h,i}}{\varrho_i}\Bigr| < 10^{-12}\cdot\frac{1}{a_i^2}.$$
**Failure means:** $\varrho_p$ and $a$ are coming from separate table lookups. Refactor to a single EOS call returning all four leaves.

### 1.2 Multi-state EOS sweep
Repeat 1.1 at 20 random states spanning the operating range ($p \in [0.5, 10]\,\mathrm{bar}$, $h_g \in [2.0, 3.0]\,\mathrm{MJ/kg}$, $h_l \in [0.2, 0.6]\,\mathrm{MJ/kg}$). Tier: EOS-bounded.

---

## 2. State converters

### 2.1 Forward conversion $U_h \to V$
At the reference state, compute $V$ directly. Compare against hand-evaluated values for all six components. Tier: machine.

### 2.2 Inverse conversion $V \to U_h$
Take the $V$ from 2.1, run the inner Newton, recover $U_h$. Compare against the original. Tier: $10^{-12}$ (Newton-bounded).

### 2.3 Round-trip identity
$U_h \to V \to U_h'$ at 20 random states. Check $\|U_h' - U_h\|/\|U_h\| < 10^{-10}$. Tier: EOS-bounded.

**Failure means:** the inverse Newton isn't converging, or the EOS is being called inconsistently between forward and inverse paths.

---

## 3. Closed-form matrices

### 3.1 $G_h$ eigenvalues
At reference state, compute numerical $\operatorname{eigvals}(G_h)$ via `jnp.linalg.eig`. Compare against analytic $\{u_g, u_l, u_g, u_l, u_{\text{mix}}+a, u_{\text{mix}}-a\}$ using the closed-form $u_{\text{mix}}$ and $a$ from §13.1 of the derivation doc. Pair-match (the two repeated eigenvalues may emerge in any order). Tier: machine.

### 3.2 $J$ determinant structure
Compute $\det J$ at reference state and at $\alpha_g \in \{0.01, 0.5, 0.99\}$. Confirm $\det J \propto \alpha_g^2(1-\alpha_g)^2$ — that is, $\det J(\alpha)/\det J(0.5)$ should match $[\alpha(1-\alpha)/0.25]^2$ to machine precision when other state is held constant. Tier: machine.

### 3.3 $K$ singular structure at zero velocity
Compute $\det K$ at $u_g = 0$ (other state nominal). Should give $0$ to machine precision. Inspect column 5 (the $h_g$ column) — should be entirely zero. Repeat at $u_l = 0$. Tier: machine.

### 3.4 $X = J G_h - K$ structural zeros
Build $X$ at the reference state. Verify:
- Rows 1, 2 are zero (max absolute entry $< 10^{-12}$)
- Columns 5, 6 are zero
- Row 4 is zero
- $X_{4,j} = -X_{3,j}$ for $j=1,2,3$ and $X_{4,4} = -X_{3,4}$ (mixture momentum conservation visible in $X$)
- Row 5: $X_{5,j} - u_g X_{3,j} + p G_{4,j} - p u_g \delta_{j,4} = 0$
- Row 6: $X_{6,j} - u_l X_{4,j} - p G_{4,j} + p u_l \delta_{j,4} = 0$

Tier: machine. **Failure means:** EOS closure not applied consistently, or sign error in one of $J, G_h, K$.

### 3.5 Right eigenvector identity
For each $k \in \{1,\ldots,6\}$, compute $G_h r_k - \lambda_k r_k$ and check max absolute residual $< 10^{-12}\cdot\|r_k\|$. Use the closed-form $r_k$ from §13.2. Tier: machine.

### 3.6 Left eigenvector identity
For each $k$, compute $l_k G_h - \lambda_k l_k$ and check max absolute residual $< 10^{-12}\cdot\|l_k\|$. Tier: machine.

### 3.7 Biorthogonality
Compute $L_h R_h$. Off-diagonal entries should be zero ($< 10^{-12}$ of largest diagonal); diagonal entries $D_k = l_k\cdot r_k$ should be nonzero. Verify $D_5 = D_6 = 1$ exactly (the $T_i\cdot 1/T_i$ cancellation). Tier: machine.

### 3.8 Full reconstruction
Verify $G_h = R_h\,D^{-1}\,\Lambda\,L_h$ entrywise. Tier: machine.

**Note on the $a_1^2$ typo:** if 3.6 fails on row 3 specifically (the $u+a$ acoustic left eigenvector), with the failure concentrated in the $u_l$-momentum column, you've used the first-power $\hat\varrho_l$ instead of squared in $a_1^2$. See §13.1 of derivation doc.

---

## 4. Flux split kernel

### 4.1 Flux-space projector sum
Build $\rho_k = K r_k$ and $\ell_k$ via dual basis (one LU on $\rho$). Verify $\sum_k \widetilde P_k = I$, where $\widetilde P_k = \rho_k \ell_k / D_k$. Tier: $10^{-10}$ (LU-bounded).

### 4.2 Smoothed sign continuity
Sweep one wave speed $\lambda_k$ through zero by varying state (e.g., let one $u_i \to 0$). Verify $s_\epsilon(\lambda_k)$ transitions smoothly: compute $\hat F$ at $\lambda_k = +\epsilon, +\epsilon/2, 0, -\epsilon/2, -\epsilon$ and check that the difference $\|\hat F(+\epsilon/2) - \hat F(-\epsilon/2)\|$ is $O(\epsilon)$, not $O(1)$ (which would indicate a hard switch was used). Tier: discretization.

### 4.3 Single-phase gas limit
At $\alpha_g = 1 - 10^{-6}$, with $u_g = u_l$ and $h_g = h_l$, compute $\hat F$ between two states differing only in the gas variables (liquid fields identical). The gas-row components of $\hat F$ should match single-phase Euler FVS for the gas alone, with the liquid-row components negligible (proportional to $1 - \alpha_g = 10^{-6}$). Tier: $10^{-4}$.

### 4.4 Homogeneous no-slip limit
At $u_g = u_l = u$ and $h_g = h_l = h$, $\alpha_g = 0.5$, compute the eigenvalues of $G_h$. Two acoustic waves should give $u\pm a_{\text{mix}}$ with $a_{\text{mix}}$ matching the no-slip mixture sound speed $\tilde a$ (since $\Delta a^2 \to 0$). The four kinematic waves should all equal $u$. Tier: $10^{-12}$.

---

## 5. Non-conservative correction

### 5.1 $H^{nc}\,\Delta F$ on uniform state
At a spatially uniform state, the flux gradient $\Delta F = 0$. Verify $X K^{-1}\Delta F = 0$ exactly. Tier: machine.

### 5.2 $H^{nc}\,\Delta F$ scaling
On a non-uniform state with a small perturbation $\delta U_h$, verify that the correction scales linearly: $X K^{-1}\Delta F(2\delta) \approx 2\cdot X K^{-1}\Delta F(\delta)$. Tier: $10^{-3}$.

### 5.3 $K$-singular fallback
At a state with $u_g = 10^{-3}\,\mathrm{m/s}$ (near-singular $K$), check that `jnp.linalg.solve(K, dF)` produces a finite result for $\Delta F$ in the range of $K$, OR that the fallback to `jnp.linalg.lstsq` is invoked correctly. Test passes if the energy and momentum residuals remain bounded (no NaN, no order-of-magnitude blow-up). Tier: physical.

---

## 6. Source assembly

### 6.1 $D_s$ row-by-row
At reference state with all physical inputs set to representative nonzero values ($\sigma_g^M = 0.1$, $\sigma_g^Q = 100$, $F^v_g = 50$, $F_l = $ gravity on liquid only, others zero), compute $D_s$ via the pseudocode in Appendix A of the pipeline. Compare each of the 6 entries against hand-evaluated reference values. Tier: $10^{-10}$.

### 6.2 $\Phi^{-1}$ transformation
Take $D_s$ from 6.1 and apply $\Phi^{-1}$. Verify rows 1-4 unchanged; row 5 equals $D_{s,1}/\varrho_g + T_g D_{s,5}$; row 6 analogously. Tier: machine.

### 6.3 $J$ transformation
Take $D_h$ from 6.2 and apply $J$ to get $D_{V_E}$. Verify the result against direct evaluation of $J D_h$. Tier: machine.

### 6.4 Mass conservation (interfacial only)
Set $\sigma_g^M = -\sigma_l^M \neq 0$, all other inputs zero, $\mathrm{d}\ln A/\mathrm{d}x = 0$. Compute $D_{V_E}$. Verify:
- $D_{V_E,1} = \sigma_g^M$ exactly
- $D_{V_E,2} = -\sigma_g^M$ exactly
- $D_{V_E,1} + D_{V_E,2} = 0$ (machine zero)

Tier: machine.

### 6.5 Momentum conservation (interfacial only, $k\to k\varrho$ correction)
Set $F^v_g \neq 0$, $\sigma_g^M \neq 0$, all other zero. Compute $D_{V_E}$. Verify $D_{V_E,3} + D_{V_E,4} = 0$ to machine precision.

**Failure means:** the $k \to k\varrho$ correction in Table B.46 rows 2-3 was not applied. Without it, the residual is ~65% of the momentum scale. With it, machine zero.

Tier: machine (after correction).

### 6.6 Energy conservation (flag, not pass)
Same setup as 6.5. Compute $D_{V_E,5} + D_{V_E,6}$. Record value. Expected: residual $\approx -F^v_g \Delta u$ at the ~1% relative level — this is the **known issue** from the conservation diagnostic. **Do not fail the test on this** — log the residual value and proceed. If the residual is much larger than $F^v_g \Delta u$, that indicates a separate bug.

Tier: bounded by viscous-energy scale.

### 6.7 Geometric source only
Set all interfacial and external inputs to zero, $\mathrm{d}\ln A/\mathrm{d}x \neq 0$. Compute $D_{V_E}$. Verify:
- $D_{V_E,1} = -\alpha_g\varrho_g u_g\,\mathrm{d}\ln A/\mathrm{d}x$ exactly
- $D_{V_E,2} = -\alpha_l\varrho_l u_l\,\mathrm{d}\ln A/\mathrm{d}x$ exactly
- Mixture mass-flux change: $D_{V_E,1} + D_{V_E,2} = -(\alpha_g\varrho_g u_g + \alpha_l\varrho_l u_l)\,\mathrm{d}\ln A/\mathrm{d}x$ (this is the mixture mass-flux through area variation — not a true source, but the geometric contribution)

Tier: machine.

### 6.8 External force / heat
Set $F_l \neq 0$ (gravity), $Q_g \neq 0$ (wall heating), all interfacial zero, $\mathrm{d}\ln A/\mathrm{d}x = 0$. Verify external force appears correctly in momentum rows (gravity acts only on liquid in this test) and external heat appears in gas energy row (proportional to $\alpha_g Q_g$).

Tier: machine.

---

## 7. Full residual function

### 7.1 Stationary uniform state
Construct a spatially uniform state with all sources zero (no interfacial, no external, no area variation). Build the full residual $R(W^n, W^n)$ at time level $n$ on a uniform grid. **Every component of $R$ must be zero to machine precision** at every cell.

This is the most important integration test. **Failure means** one of the following:
- Flux assembly has a sign error (check 4.x tests passed)
- Non-conservative correction is being applied incorrectly (check 5.x)
- Source assembly has a constant offset
- Boundary conditions are corrupting interior cells

Tier: machine.

### 7.2 Stationary balanced state (with sources)
Construct a state with a constant flow $u_g = u_l$ in a uniform-area pipe, $\sigma^M = 0$, $F^v = 0$, but with a constant body force balanced by a pressure gradient (i.e., $\partial_x p = F_l + F_g$). The residual should be zero everywhere — pressure gradient exactly cancels body force.

Tier: $10^{-10}$ (discretization-bounded for the pressure gradient).

### 7.3 Residual continuity in $W$
Pick a state, compute $R(W_0)$. Perturb $W_0$ by a small $\delta W$ in one component at a time, compute $R(W_0 + \delta W)$. Verify continuity: $\|R(W_0+\delta W) - R(W_0)\|/\|\delta W\|$ is bounded as $\|\delta W\| \to 0$. **No spikes**, no jumps. Spikes indicate the smoothed sign function isn't being used somewhere (a hard `where` instead of the Harten fix).

Tier: discretization.

### 7.4 EOS constraint satisfaction
At a state where the conservation residual is zero, verify the EOS constraint residuals $\mathcal C_g, \mathcal C_l$ from the 8-component $W$-residual are also zero. This checks that $W_1, W_2$ (the pressure and void fraction auxiliaries) are consistent with the rest of $W$ via the EOS.

Tier: $10^{-10}$.

---

## 8. Newton iteration

### 8.1 Quadratic convergence near steady state
Start from a small perturbation of the stationary uniform state (7.1). The Newton residual norm should decrease quadratically:
$$\|R^{k+1}\| \lesssim C \|R^k\|^2$$
for some bounded $C$. Plot $\log\|R^{k+1}\|$ vs $\log\|R^k\|$ and verify slope $\approx 2$ over at least three iterations before reaching machine precision.

**Failure means:** Jacobian is not the true Jacobian of the residual (perhaps a manual derivative slipped in instead of `jax.jacfwd`), or the linear solve is being damped/regularized too aggressively.

Tier: physical.

### 8.2 Conditioning across $\alpha_g$ range
Compute $\operatorname{cond}(J_R)$ (the 8×8 Newton Jacobian) at $\alpha_g \in \{0.01, 0.1, 0.5, 0.9, 0.99\}$. Should stay bounded — that's the whole point of the $W$ basis. If conditioning explodes at the extremes, the $W$ choice didn't help as designed and warrants investigation.

Tier: physical.

### 8.3 Conditioning at sonic transition
At a state with one acoustic wave speed close to zero ($|\lambda_3| < 0.1\,a$), compute $\operatorname{cond}(J_R)$. Should remain bounded thanks to the entropy fix. If conditioning blows up here, the entropy fix isn't actually being applied.

Tier: physical.

---

## 9. Physics benchmarks

These are the final validation gates — they test physical correctness against known reductions or reference codes. Pass means the implementation is ready for production.

### 9.1 Single-phase gas reduction
Set $\alpha_g = 1 - 10^{-6}$ everywhere, $u_g = u_l$, $h_g = h_l$ (the liquid is a passenger). Run a steady nozzle problem. Compare the gas-phase profile against the existing single-phase real-gas solver. Pass: $L^2$ error $< 10^{-4}$ relative.

### 9.2 Single-phase liquid reduction
Symmetric. Liquid is the active phase, gas is a passenger.

### 9.3 No-slip homogeneous transient
Initialize with $u_g = u_l$, $h_g = h_l$. Run a transient. Verify $u_g - u_l$ and $h_g - h_l$ stay below $10^{-10}$ for at least 1000 timesteps (no spurious slip or thermal disequilibrium generation).

### 9.4 Stadtke two-phase shock tube
Replicate the test case from Stadtke 1997 paper Section 4.x. Compare shock and rarefaction structures qualitatively, then verify pressure ratio across shock to $\sim 1\%$.

### 9.5 Subsonic nozzle
Standard converging nozzle at low pressure ratio. Verify mass flow rate matches isentropic theory at the gas-only limit and is bounded between gas-only and liquid-only at intermediate void fractions.

### 9.6 Choked nozzle
At critical pressure ratio, verify mass flux matches two-phase critical flow models (Henry-Fauske or similar) to $\sim 5\%$ for the operating void fraction range.

### 9.7 Supersonic nozzle with shock
Converging-diverging nozzle with downstream back-pressure forcing a normal shock in the diverging section. Verify shock location matches Rankine-Hugoniot prediction to $\sim 1$ cell width.

---

## 10. Failure-mode response

If a test fails at its specified tier:

1. **Do not advance to the next stage.** A failure at stage 4 will propagate as inscrutable noise into stage 7.
2. **Re-run all prior stages** to confirm no regression. JIT recompilation or library updates can silently break upstream tests.
3. **Inspect the symbolic identity.** Most failures trace to a sign error or missing factor in one of the closed-form matrices. The verification ledgers in §5, §10, §11.4, §12.6, §13.5 of the derivation doc list what each matrix should satisfy.
4. **Reduce the problem.** If a complex test fails, construct the simplest sub-test that exhibits the failure (e.g., zero out all but one source, set $\Delta u = 0$ to remove slip dependence). This usually localizes the bug to a single function.

---

## 11. Verification ledger summary

| Stage | Building block | Test count | Tier |
|-------|---------------|-----------|------|
| 1 | EOS layer | 2 | EOS |
| 2 | State converters | 3 | EOS / machine |
| 3 | Closed-form matrices | 8 | machine |
| 4 | Flux split kernel | 4 | machine / discretization |
| 5 | Non-conservative correction | 3 | machine / physical |
| 6 | Source assembly | 8 | machine (one flag) |
| 7 | Full residual | 4 | machine / discretization |
| 8 | Newton iteration | 3 | physical |
| 9 | Physics benchmarks | 7 | physical |

Total: 42 verification tests across 9 stages. Stages 1-7 are pure unit/integration testing — they should all pass at machine or near-machine precision before any benchmark run is attempted. Stages 8-9 are physics gates with looser tolerances appropriate to discretization and modeling assumptions.
