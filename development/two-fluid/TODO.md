# TODO: K-free / smooth port

Record of what was done: [README.md](README.md).

- [x] **1. Grid comparison (200/400 cells).** Done 2026-11-19, results in the README.
      K-free converges in all 18 runs, K-inverse stagnates at 200 for shocked/adapted;
      shock position unchanged; k_free - k_inverse difference is the same size as each
      scheme's own 200 -> 400 change, but not monotonically decreasing.
  - [x] 1b. 800-cell runs (k_inverse, k_free). Shocked: the two approach each other
        (difference halves). Subsonic/adapted: different limits near x ~ 0.093, where
        det K = 0; K-inverse keeps a grid-independent alpha_g bump there, K-free does not.
- [x] **1d. Flashing water (Super Moby Dick), 100 and 400 cells.** Done 2026-11-20
      (`run_flashing.py`, results in the README): K-free converges in all 12 runs,
      K-inverse stagnates in all 12; K-free at 400 within 0.3-0.5 % of the published mass flow.
  - [x] Grid study 100/200/400/800 and K-inverse restart from K-free 400: done
        2026-11-20 (README). Near grid convergence at 800 (about 0.1 % m_dot left), limit
        ~0.9 % above published; K-inverse converges from K-free only in subsonic (agrees
        within 0.04 % m_dot), stagnates for shock/adapted.
  - [ ] Low priority (accepted as is, 2026-11-20): K-free stagnation at 200 cells
        (subsonic, shock). Not a divergence: Newton reaches inner residual 1e-7-1e-9 but
        line search is exhausted (status 3) once CFL > ~3, the controller halves CFL to
        2e-8 and stagnation triggers. Worst cells 76-80 (x 0.46-0.48 m) straddle
        alpha_b = 0.3 of the default `regime_blend="linear"` (slope kink); hypothesis, not
        proven (400/800 also have cells near 0.3 and converge). Closure issue, not K-free.
        Test if needed: rerun with `regime_blend="smoothstep"` as a diagnostic.
- [ ] **1c. Mixture energy not conserved (model-level).** Mixture total enthalpy rises
      3.6-9.8 % inlet -> outlet in both schemes; from the NC energy rows (gas/liquid do not
      cancel, ~2.9-5.8 % of inlet energy flux) and the closure sources (~0.5-0.7 %).
      Same code as alberto_branch, so it affects it too.
      **Hypothesis (user, 2026-11-20):** a property of how the NC terms are written, not a
      bug: the NC (interfacial) forces are built so that they do not increase the phase
      entropies, and as a consequence total energy is not conserved; the imbalance is
      energy that the interface gives to (or takes from) the fluids. To do:
  - [ ] verify that the NC terms leave each phase entropy unchanged (per-phase entropy
        budget of the NC part) and that the NC energy imbalance equals the interface work;
  - [ ] closure-source part (+0.5-0.7 %): the closures themselves conserve energy; the
        imbalance comes from the inviscid (interface-related) terms that the source vector
        on the RHS also carries (user, 2026-11-20). Measured: appears only with closures on
        (closures off: area-source energy rows cancel). Confirm term by term in
        `D_VE_algebric_two_component`;
  - [ ] grid check of the energy gain from the existing 100-800 solutions;
  - [ ] acknowledge it: document the energy non-conservation as a known model property
        (two-fluid notes, and in reported results/papers) with its size per case
        (mixture H +3.6 / +6.8 / +9.8 %, outlet T above inlet).
- [x] **2. Pytest tests.** Done 2026-11-20: `tests/models/two_fluid/test_k_free.py` (13
      tests) and two path-mode tests in `tests/solvers/test_mapped_reconstruction.py`.
  - `tests/models/two_fluid`: default unchanged; uniform state (flux = native, NC = 0);
    NC mass rows zero and momentum rows cancelling; colored = dense Jacobian with
    k_free + MUSCL; config validation; k_free + mass_upwind_cutoff refused.
  - `tests/solvers`: small generic test of the operator `"path"` mode.
- [x] **3. Documentation.** Done 2026-11-20: options, K-free formulas and the energy
      property in `docs/notes/two_fluid_migration.md`; `"path"` mode in
      `docs/notes/model_interface.md`; `docs/notes/nonconservative_terms.md` now points to
      the `src` options; pointer in `development/two_fluid_test_v2/README.md`.
      Pre-existing broken link there: `REPORT.html` (file is
      `REPORT_two_fluid_inverse_free.html`), left unchanged.
- [ ] **4. Optional wiring.** `cases/NICFD2026/scripts/case.py` reading `k_free`/`smooth`
      (default off); optional switch in the demo.
- [ ] **5. Optional evaluation.** Cold start (prototype tested only published-profile
      seeds); fair serial timing k_inverse vs k_free; warm-start throat-bump test
      (as done in twophase-da).
- [ ] **5b. Saving and common initialization (design item).**
  - Save restarts: call `save_restart` (`solvers/fvm/continuation.py`, portable npz, exact
    float64, no overwrite) in `run.py` and `run_flashing.py`, next to the JSON/fields files.
  - Common initialization selector for the two-fluid model, instead of the copies in the
    demo, the NICFD case builder and the development scripts:
    `stagnation` (uniform guess from the inlet stagnation state, the current "cold" start),
    `read_npz` (saved solution on the same or another grid via `load_restart` +
    `prepare_continuation_state`; generic, already exists) and `profile`/`csv`
    (interpolated primitive profile, e.g. published or nozzlex results).
  - Format: npz for restart/initialization (exact; ~2.4x smaller than full-precision CSV,
    e.g. 54 vs 129 kB at 400 cells; default pandas CSV round trip is not bit-exact);
    CSV only as optional export for comparisons and external tools.
  - Optional: a shared results writer (summary + history + fields) for demos, cases and
    development scripts. Package design change, to be scoped separately.
- [ ] **6. Commit** (only when asked): on a branch, production changes (`src`, tests,
      docs) separate from this study folder.
- [ ] **7. Port the alberto_branch source-term capabilities to main.** Bring all the
      source-term / closure capabilities of `nozzle-hyperbolic_alberto_branch` into main
      `src`. To be scoped and verified later.
