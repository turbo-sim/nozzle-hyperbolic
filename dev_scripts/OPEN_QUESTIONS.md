# Open questions — `nozzle_hyperbolic/fvm_solver/two_fluid/solver.py`

Consolidated 2026-08-04 from a repo-wide flag/comment audit. These are
questions that need **your** judgment and probably another probe run, not
things I resolved or should have silently "cleaned up". Each entry: claim →
evidence → recommended action → probe to rerun (all probes already exist,
no new ones needed).

## 1. `nc_form` pairing correctness — UNRESOLVED

`solver.py:822-824` (comment on the non-conservative correction):
> "correctness is a PAIRING property with the area-source content —
> discriminated by the α-contact step-map probe (`debug_nc_contact.py`)."

Three branches exist (`'cell'` = "ours" = default, `'staedtke'` = book eq.
8.38, `'wb'` = well-balanced 2026-07-15). No branch is declared a settled
winner in-code. Production stays on `'cell'`.

**Recommended action:** rerun `debug_nc_contact.py` if/when you want this
settled; until then `'cell'` remains the default and the other two exist only
for that probe.

## 2. Steady-state residual definition — `ssr_mode` is a workaround, not the fix

`solver.py:2210`. The steady-state residual is

    S_ss = t_char * R_ss / (V_cell * |V|)      t_char = dx/max(lam_max)
                                                V_cell = A*dx
         = R_ss / (A * max(lam_max) * |V|)      (the two dx cancel)

then reduced to a scalar by an RMS over all cells and all six rows
(`ssr_mode='all'`, the default, self-labeled legacy in-code) or by a
mass-fraction-weighted RMS (`ssr_mode='mass'`).

**What is right about it.** Dimensionless, so the six rows are comparable at
all (raw they are kg/m3/s vs kg/m2/s2 vs J/m3/s); relative, so no hand-tuned
per-equation constant is needed; scaled by a wave speed, so it is tied to the
physics that sets the timestep.

**What is wrong with it.** The normalisation is the *local, per-phase*
conserved variable `|V|`. A relative residual is right when `V` is a
meaningful magnitude and catastrophic when `V -> 0`: for a ghost at
alpha = 1e-3 the denominator is ~1e-4 of the mixture, so a negligible absolute
residual is reported as an enormous relative one. Measured: ghost rows owned
**99.98%** of the plain SSR2 on a converged N600 shock run, ghost/dominant
ratio **7553x**.

That is not a ghost problem. It is a normalisation problem that the ghost
merely exposes, and `ssr_mode='mass'` cancels the spurious amplification
rather than preventing it. The cost of the workaround: a genuinely diverging
ghost is suppressed too, so the convergence metric cannot warn about it
(`pin_ghost` and a mass-flow conservation check are the current cover).

Two secondary defects:

- **Not mesh-independent.** `R_ss` carries a factor of cell size, so
  `ssr_tol = 1e-8` is a different physical standard on N=420 than on N=600.
  SSR values from different grids are not comparable.
- **`t_char` uses a global `max(lam_max)`**, so cells in slow regions are
  scaled by a transit time much shorter than their own, weighting the fast
  regions more.

**Planned fix, in this order** (order matters — steps 2 and 3 make things
worse if done before step 1):

1. **Reference-scale each row against a mixture-level or global magnitude**
   instead of the local per-phase `|V|` — e.g. gas mass residual over
   `(m_g + m_l)`, momentum over the inlet momentum flux, or a running max over
   the field. A near-massless phase then contributes a near-zero residual,
   which is the honest answer.
2. **Equal weighting across rows becomes correct**, and `ssr_mode` stops
   needing to exist. Keep `'mass'` and `'all'` accepted for reproducing old
   runs.
3. **Reduce with a max over the six equations** rather than an RMS, so the
   worst-converged equation controls the stop instead of hiding behind five
   good ones. RMS over *cells*, max over *equations* — a max over cells too
   would let one shock or sonic cell in a limit cycle veto convergence
   forever, structurally the same failure as the ghost.
4. **Report all six per-equation values every `print_every`.** Worth doing
   regardless of the above: today SSR is one scalar that says nothing about
   *what* is lagging. "E_g limiting at 3e-7 while mass_l is at 2e-9" is
   actionable; "SSR = 3e-7" is not. This would also have made the ghost
   pathology visible immediately instead of requiring a dedicated probe.

**Consequences to plan for.** A max-of-six is always >= the current RMS, so
existing `ssr_tol` values become stricter and every reference case needs a
re-baselined tolerance. Archived NICFD 2026 cases record `ssr_mode: mass` in
their `settings.yaml`; those files need updating if those cases are ever
re-run under a new criterion.

**Stopgap until then:** `ssr_mode='mass'` is the right choice for any case
with a thin ghost, and is what the flashing cases use. Promoting it to the
default is still reasonable, but it is a patch — the real fix is step 1.

## 3. `w_char='auto'` — candidate for promotion

Dated 2026-07-17, certified by `debug_wref_sensitivity.py`, whose own output
recommends: *"'auto' (c·field-max) is safe as the general default."* The
actual kwarg default is still `None` (fixed two-component-calibrated floors).

**Recommended action:** consider promoting `w_char='auto'` to the default;
the certifying probe already argues for it. Rerun
`debug_wref_sensitivity.py` if you want to re-confirm before flipping.

## 4. `eos_rows` (8-row E3-augmented system) — open/unresolved

Self-labeled "prototype, 2026-07-15" at its definition. Fixes a measured
secular liquid-mass drain (−1.5e-4/step, kills ghost-limit runs at ~4700
steps) but `dev_scripts/config_loader.py` default stays `False`, and it's
never used in `main_two_fluid.py`'s default config — only exercised from
`debug_*.py` probes (`debug_banded_e3.py`, `debug_jac_lag.py`,
`debug_outlet_bc.py`, `debug_throat_texture.py`, `debug_wref_sensitivity.py`).

**Recommended action:** this fixes a real bug but is still "prototype" —
worth a decision on what's left before promoting it to default (the audit
didn't find an explicit blocker, just that it never got promoted).

## 5. `eps_ghost_k2` doc/default drift — not a science question, just stale

Inline comment at the filter's use site: *"k2 (default 4): for a 2Δx mode the
4th difference is 4× the 1st difference... k2=1 was ×4 weaker in the shock
zone — held the N600 cycle at ~14% instead of draining it."* The actual kwarg
default (and `config_loader.py:215`) is `eps_ghost_k2=0.0`.

**Recommended action:** one-sentence decision needed — is 0.0 intentional
(filter off by default) or should the default be 4 per the comment's own
recommendation? If 0.0 is intentional, the comment should say so instead of
reading as an un-applied recommendation.

## 6. `solver.py:2116` TODO — highest stakes, core time-marching

Literal comment: *"TODO: check if the time stepping works correctly. Are we
actually computing the correct pipeeline with W. Also we need Hnc and D at
the timestep n+1!"*

This is not a flag choice — it's an open question about the core
time-marching residual itself (whether `W`, the non-conservative term `Hnc`,
and `D` are evaluated at the correct time level). Unlike items 1-5, there's
no existing probe named for this specifically.

**Recommended action:** if/when you're about to seriously refactor the
time-stepping/residual core, do a full, careful, single-sitting read of just
that stepping function before touching it — this TODO is exactly the kind of
thing a partial read risks missing.

## 7. `bc_outlet='characteristic_old'` — not open, just documented here so it isn't mistaken for cleanup fodder

Labeled "legacy" in its own comment, but actively selected by
`dev_scripts/main_two_fluid.py:660` and
`dev_scripts/sensitivity_test/sensitivity_closure.py:76-79` to reproduce old
runs. The modern `'characteristic'` (eigenvector-projection rewrite) agrees
with it to ~0.06% at steady state. **Intentionally retained as an A/B
reference** — not dead code, do not delete without checking those two
callers first.

---

## Already retired (2026-08-04), for reference

Two branches were downgraded from silently-selectable to a hard
`ValueError` in `solver.py`, since both had explicit textual evidence of
being dead (not a judgment call):

- `eos_project='triangular'` — comment said "FAILED certification — do not
  use"; only ever selected by `dev_scripts/debug_projection_tri.py`, which
  exists to document that failure.
- `bc_inlet='stagnation_ramped_soft'` — zero references anywhere in the
  repo, undocumented, contained an inline alternate Newton step explicitly
  marked "BROKE convergence."

Both implementations are kept commented-out in place (not deleted) as the
historical record of what was tried.
