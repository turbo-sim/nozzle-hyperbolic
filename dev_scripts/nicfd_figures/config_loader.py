"""
Loads a two-fluid settings YAML and returns a SimpleNamespace cfg object
with the same attributes that the old settings_two_fluid.py exposed.
"""

import types
import pathlib
import yaml


def load_config(yaml_path: pathlib.Path) -> types.SimpleNamespace:
    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = types.SimpleNamespace()

    # ── Fluid ────────────────────────────────────────────────────────────────
    cfg.FLUID_NAME = str(raw["fluid"]["name"])

    # ── EOS tables ───────────────────────────────────────────────────────────
    # Backends that are ANALYTIC (no table): built as FluidPerfectGas, not
    # FluidBicubic.  Ideal as a passive GHOST phase — unbounded (no table edge →
    # no extrapolation blow-up / dt-collapse) and needs no h/p range or table_dir.
    _PG_BACKENDS = ("perfect_gas", "perfectgas", "ideal_gas", "idealgas")

    def _lut(section: str) -> dict:
        s = raw["eos"][section]
        backend = str(s["backend"])
        if backend.lower() in _PG_BACKENDS:
            # Perfect gas: only name + reference state (constants fit at T_ref,p_ref).
            # No h/p bounds needed (analytic, in-range everywhere).  The driver
            # dispatches on `backend` to build jxp.FluidPerfectGas.
            return {
                "backend":    backend,
                "fluid_name": str(s.get("fluid_name", cfg.FLUID_NAME)),
                "T_ref":      float(s.get("T_ref", 300.0)),
                "p_ref":      float(s.get("p_ref", 101_325.0)),
            }
        d = {
            # per-phase fluid_name (falls back to the global fluid.name) so the two
            # phases can be DIFFERENT fluids (e.g. gas=Air, liquid=water).
            "fluid_name":      str(s.get("fluid_name", cfg.FLUID_NAME)),
            "backend":         str(s["backend"]),
            "h_min":           float(s["h_min"]),
            "h_max":           float(s["h_max"]),
            "p_min":           float(s["p_min"]),
            "p_max":           float(s["p_max"]),
            "N_h":             int(s["N_h"]),
            "N_p":             int(s["N_p"]),
            "table_dir":       str(s["table_dir"]),
            "gradient_method": str(s["gradient_method"]),
        }
        # metastable_phase is OPTIONAL: omit it for an ordinary single-phase table
        # (no metastable extension, no saturation props — correct for Air or a
        # purely subcooled liquid in a single-phase validation).
        if s.get("metastable_phase") not in (None, "", "none", "None"):
            d["metastable_phase"] = str(s["metastable_phase"])
        if "N_p_sat" in s:
            d["N_p_sat"] = int(s["N_p_sat"])
        return d

    cfg.LUT_G = _lut("gas")
    cfg.LUT_L = _lut("liquid")

    # ── Interphase-structure closure (TA-BD interfacial area) ─────────────────
    src = raw.get("sources", {})
    cfg.BUBBLE_NUMBER  = float(src.get("bubble_number",  1.0e8))
    cfg.DROPLET_NUMBER = float(src.get("droplet_number", 1.0e8))
    # mass_transfer: True  → single-component with phase change (D_VE_algebric,
    #                        needs saturation props); False → two-component, drag+heat
    #                        only (D_VE_algebric_two_component, no saturation).
    cfg.MASS_TRANSFER  = bool(src.get("mass_transfer", True))

    # ── Grid ─────────────────────────────────────────────────────────────────
    g = raw["grid"]
    cfg.Nx     = int(g["Nx"])
    cfg.x1     = float(g["x1"])
    cfg.x2     = float(g["x2"])
    cfg.RADIUS = float(g.get("radius", 0.005))   # legacy; only used if no geometry block

    # ── Geometry ─────────────────────────────────────────────────────────────
    # Same mechanism as the single-phase main.py: a `geometry:` block names a
    # factory in nozzle_hyperbolic.fvm_solver.geometry (get_pipe,
    # get_symmetric_nozzle, get_lettieri_nozzle, ...) plus its kwargs.  The
    # driver dispatches via getattr(_geom, function)(x, roughness=..., **kwargs),
    # so nozzle geometries reuse the SAME geometry.py — no new file needed.
    #
    # Backward compatible: if there is no `geometry:` block, fall back to a
    # constant-area pipe built from grid.radius (the previous behaviour).
    geo = dict(raw.get("geometry", {}))
    if not geo:
        geo = {"function": "get_pipe", "radius": cfg.RADIUS}
    cfg.GEO_FUNCTION  = str(geo.pop("function", "get_pipe"))
    cfg.GEO_ROUGHNESS = float(geo.pop("roughness", 0.0))
    cfg.GEO_KWARGS    = geo   # remaining keys forwarded verbatim to the factory

    # ── Numerics ─────────────────────────────────────────────────────────────
    n = raw["numerics"]
    cfg.CFL          = float(n["CFL"])
    cfg.NEWTON_ITER  = int(n["newton_iter"])
    cfg.DELTA_FACTOR = float(n["delta_factor"])
    cfg.RECON        = str(n["recon"])
    cfg.K_VM         = float(n["k_vm"])
    cfg.JACOBIAN          = str(n.get("jacobian", "dense"))          # dense | colored
    cfg.JACOBIAN_VALIDATE = bool(n.get("jacobian_validate", False))
    cfg.PROFILE           = bool(n.get("profile", False))            # per-step cost breakdown
    cfg.LINEAR_SOLVE      = str(n.get("linear_solve", "lstsq"))      # lstsq | normal_eq | normal_eq_banded | qr
    # Newton column-scaling floors (8 values, W order p,αg,mg,ug,ml,ul,mg_eg,ml_el),
    # or 'auto' (2026-07-17, RECOMMENDED): per-variable 0.05·max|W| recomputed from
    # the current Newton iterate — reproduces hand floors bit-identically wherever
    # they sit at/below the field scale, adapts through flashing (α 1e-4→0.9), no
    # per-case tuning.  None → solver defaults (two-component calibrated; on a
    # GHOST state they exceed |W| ×20-500 and inflate the SSR floor ×180).
    _wc = n.get("w_char", None)
    cfg.W_CHAR            = ("auto" if isinstance(_wc, str) and _wc == "auto"
                             else [float(v) for v in _wc] if _wc is not None
                             else None)
    # Per-iteration EOS-manifold projection: true (classic 2×2 pair) | false |
    # 'damped' (conditioning-gated pair: λ = s/(s+s_ref) per cell/pass from the
    # 2×2 Jacobian's own conditioning — reduces to pair where well-posed and to
    # false in the ghost/stiff-liquid limit; α-uniform, USE FOR FLASHING) |
    # 'triangular' (FAILED certification 2026-07-15 — any method converging to
    # the exact root inherits dp/dm_g ≈ 1e8 in the ghost limit; do not use).
    _ep = n.get("eos_project", True)
    cfg.EOS_PROJECT       = _ep if isinstance(_ep, str) else bool(_ep)
    # Solver-side void-fraction clip floor.  Default 1e-6 (gas tier).  Flashing
    # runs with a thin liquid-side ghost should match the IC floor (~1e-4..1e-3)
    # so the Newton clip cannot re-thin α below it mid-march (the h_g = E/(αρ)
    # amplifier behind the flashing-onset blow-up).
    cfg.ALPHA_EPS         = float(n.get("alpha_eps", 1e-6))
    # Ghost-vapor saturation pin (nozzlex parity): blend the gas-energy residual
    # toward h_g = h_sat_g(p) where the liquid is NOT superheated (weight
    # 1 − evaporation_weight, complementary to the σ gate — cannot block onset).
    # Requires mass_transfer/use_saturation.  Default off (gas-tier unchanged).
    cfg.PIN_GHOST         = bool(n.get("pin_ghost", False))
    # Ghost-GAS momentum pin (u_g → u_l), α-gated & saturation-free — serves both
    # flashing and Bernoulli ghost-gas cases.  Default off (nothing certified moves).
    cfg.PIN_GHOST_MOMENTUM = bool(n.get("pin_ghost_momentum", False))
    cfg.PROJ_S_REF        = float(n.get("proj_s_ref", 2e-2))          # λ gate scale for 'damped'
    # (λ = s²/(s²+s_ref²); probe-calibrated: two-comp s≈0.36, ghost s≈9e-4)
    # E3-augmented system (2026-07-15, certified ×3): append the 2 EOS closures
    # as residual rows → 8 rows × 8 unknowns least squares, NO per-cell null
    # space → no manifold drift, no projection needed (set eos_project: false),
    # CFL up to 5 certified.  Dense solve path by default (~1.5-2 s/step at
    # N300); eos_rows_banded switches to the O(N) banded twin.
    cfg.EOS_ROWS          = bool(n.get("eos_rows", False))
    # E3-BANDED (new option 2026-07-16): block-banded JᵀJ + super-block Thomas
    # for the eos_rows system — SAME numbers as the dense path, O(N) instead of
    # O(N³).  Keep false until debug_banded_e3.py certifies on your case.
    cfg.EOS_ROWS_BANDED   = bool(n.get("eos_rows_banded", False))
    # JACOBIAN LAGGING (modified Newton, 2026-07-16; E3-banded path only):
    # recompute J every jac_lag-th Newton iteration (residual still every
    # iteration → converged answer unchanged, only the Newton direction lags).
    # jac_lag >= newton_iter = one Jacobian per step (~40 colored JVPs paid
    # once).  1 = certified per-iteration behavior.
    cfg.JAC_LAG           = int(n.get("jac_lag", 1))
    # NC discretization: 'cell' (legacy) | 'wb' (well-balanced face-area-weighted
    # one-sided differences — fixes the α-front-on-taper imbalance, reduces to
    # 'cell' on uniform area; certified 2026-07-15) | 'staedtke' (probe only).
    cfg.NC_FORM           = str(n.get("nc_form", "cell"))
    # Low-Mach acoustic-dissipation floor for the FVS flux: z_ac = clip(Ma, x, 1).
    #   0.0 -> z_ac=Ma  (reduces acoustic/pressure dissipation at low Ma — the
    #                    setting that REMOVES the pressure diffusion and lets the
    #                    low-Mach checkerboard grow),
    #   1.0 -> z_ac=1   (full standard dissipation; checkerboard-free but over-
    #                    dissipative at low Ma).
    cfg.LOWMACH_DISS      = float(n.get("lowmach_diss", 0.0))
    # BOUNDARY-FACE z_ac floor (2026-07-17): pressure/reservoir BCs couple to
    # the interior only through the acoustic jump terms of the first/last face
    # flux — Ma-scaled away when lowmach_diss=0 (standing outlet jump, creeping
    # mass flow).  1.0 = full upwind dissipation on the two boundary faces only
    # (stiff BC enforcement); None/absent = inherit lowmach_diss (no change).
    # Supersonic outlet: no-op (z_ac=1 there anyway, ghost transmissive).
    _lmb = n.get("lowmach_diss_bc", None)
    cfg.LOWMACH_DISS_BC   = float(_lmb) if _lmb is not None else None
    # Non-conservative correction scale (DIAGNOSTIC): 1.0 = on (physical),
    # 0.0 = OFF — used to test whether the H^nc coupling is pumping the
    # void-fraction checkerboard.
    cfg.NC_SCALE          = float(n.get("nc_scale", 1.0))
    # 4th-order (JST) background filter coefficient for the void/alpha 2Δx
    # checkerboard.  0.0 = off.  ~0.05-0.5 typical; 4th-order → smooth solution
    # perturbed only at O(eps_void*dx^3), so sweep it and confirm ε-independence.
    cfg.EPS_VOID          = float(n.get("eps_void", 0.0))
    # LOW-VELOCITY GATE for the JST void filter.  None → ungated (uniform).  When set
    # to a phase-velocity cutoff u0 [m/s], the filter is weighted by
    # ½(1+tanh((u0−u_mat)/Δu)) with u_mat=min(|u_g|,|u_l|), so it acts ONLY in the
    # low-velocity (inlet) region — where the material-wave dissipation vanishes —
    # and leaves the throat/supersonic at full high-order accuracy.  Gates on the
    # VELOCITY, not a Mach (a_mix is the α-dependent Wood speed → poor denominator).
    _evu = n.get("eps_void_ucut", None)
    cfg.EPS_VOID_UCUT   = None if _evu in (None, "", "none", "None") else float(_evu)
    _evw = n.get("eps_void_uwidth", None)                 # None → solver uses u0/3
    cfg.EPS_VOID_UWIDTH = None if _evw in (None, "", "none", "None") else float(_evw)
    # FILTER-STRENGTH scale: 'wood' (default) uses the model's true mixture eigenvalue
    # (dips at α≈0.5 → starves the filter there); 'frozen' uses a non-dipping
    # |u_mix|+max(a_g,a_l) surrogate so the 4th-order damping magnitude stays ~constant
    # in α (fluxes/CFL still use the true dipping speed).  Retune eps_void when switching.
    cfg.EPS_VOID_SCALE = str(n.get("eps_void_scale", "wood")).strip().lower()
    # GHOST-phase 4th-order filter: same conservative JST flux as eps_void but
    # gated per ROW by the phase's own smallness in log10(α) (on below acut, off
    # above — never touches a phase carrying real mass; flashing-safe) and NOT
    # velocity-gated (a ghost needs damping at any speed, incl. the shock layer
    # where the ghost limit-cycle lives).  0.0 = off (bit-identical).
    cfg.EPS_GHOST        = float(n.get("eps_ghost", 0.0))
    cfg.EPS_GHOST_ACUT   = float(n.get("eps_ghost_acut", 3e-3))
    cfg.EPS_GHOST_AWIDTH = float(n.get("eps_ghost_awidth", 0.25))
    # 2nd-order (shock-zone) coefficient multiplier: eps2 = k2·eps_ghost·σ.
    # 4 ≡ equal 2Δx damping across the JST sensor cross-fade.
    # k2 DEFAULT 0.0 (2026-07-18): ANY strong ghost dissipation at the shock digs
    # an α-hole (v3 via the mass row, v4 k2=4 via momentum→continuity: forcing
    # u_ghost smooth across the jump makes the ghost's own continuity pull α to
    # the clip floor → 1/α blow-up).  σ therefore just SWITCHES THE GHOST FILTER
    # OFF at the shock; the bounded shock-cell ghost cycle is handled by
    # ssr_mode: mass + the masked gates instead.
    cfg.EPS_GHOST_K2     = float(n.get("eps_ghost_k2", 0.0))
    # BAND-GATE floor: filter disengages below this α (hole = garbage fields;
    # touching them locks the hole open — let drag heal it).
    cfg.EPS_GHOST_ALO    = float(n.get("eps_ghost_alo", 1e-5))
    # SSR row weighting: 'all' (legacy, every equation votes equally — a ghost
    # phase's shock-layer limit cycle can pin the residual far above tol) or
    # 'mass' (each phase's rows vote with the local mass fraction — the stop
    # criterion follows the phases that actually carry mass; flashing-safe).
    cfg.SSR_MODE = str(n.get("ssr_mode", "all")).strip().lower()
    # ZERO-SLIP FLUX BLEND window [m/s]: the full-branch flux inverts a matrix whose
    # void-wave columns go parallel as du→0 (κ~1/du) → noisy in a ring around every
    # slip sign-change.  Use the well-conditioned zero-slip branch for |du|<lo, the
    # accurate full branch for |du|>hi, C¹ cosine blend between.  Widen on coarser
    # meshes (du changes more per cell) so the blend still spans a few cells.
    cfg.DU_BLEND_LO = float(n.get("du_blend_lo", 1e-2))
    cfg.DU_BLEND_HI = float(n.get("du_blend_hi", 1e-1))

    # ── Time-step control (Städtke-style adaptive halving) ─────────────────────
    tc = raw.get("time_control", {})
    cfg.ADAPTIVE_DT  = bool(tc.get("adaptive_dt", False))  # False = fixed CFL dt
    cfg.MAX_HALVINGS = int(tc.get("max_halvings", 8))
    cfg.DP_MAX       = float(tc.get("dp_max", 0.20))       # max |Δp|/p     per step
    cfg.DH_MAX       = float(tc.get("dh_max", 0.10))       # max |Δh|/|h|   per step
    cfg.DU_MAX       = float(tc.get("du_max", 0.20))       # max |Δu|/(|u|+a) per step
    cfg.DA_MAX       = float(tc.get("da_max", 0.10))       # max |Δα_g|     per step (abs)
    cfg.DT_GROW      = float(tc.get("dt_grow", 1.20))      # dt growth after accept

    # ── Run control ───────────────────────────────────────────────────────────
    r = raw["run"]
    cfg.N_STEPS        = int(r["n_steps"])
    cfg.PRINT_EVERY    = int(r["print_every"])
    cfg.VISUALIZE_IC   = bool(r["visualize_ic"])
    cfg.DEBUG_NEWTON   = bool(r["debug_newton"])
    cfg.SSR_TOL        = float(r.get("ssr_tol",        1e-6))
    _t_end_raw         = r.get("t_end")
    cfg.T_END          = float("inf") if _t_end_raw is None else float(_t_end_raw)
    cfg.PLATEAU_WINDOW = int(r.get("plateau_window",   20))
    cfg.PLATEAU_RATIO  = float(r.get("plateau_ratio",  1.5))

    # ── Initial condition ─────────────────────────────────────────────────────
    ic = raw["ic"]
    cfg.IC_TYPE        = str(ic["type"]).strip().lower()   # uniform | riemann | nozzlex
    cfg.USE_RIEMANN_IC = (cfg.IC_TYPE == "riemann")

    # nozzlex-seeded IC: run the space-marching pipe solve and interpolate its
    # relaxed profile onto the time-marching grid (see nozzlex_ic.py).
    nz = ic.get("nozzlex", {})
    cfg.NZ_T0_IN       = float(nz.get("T0_in",    578.95))   # inlet stagnation T [K]
    cfg.NZ_SOLVER      = str(nz.get("solver",     "ImplicitEuler"))
    cfg.NZ_NSTEPS      = int(nz.get("nsteps",     2000))
    cfg.NZ_N_EVAL      = int(nz.get("n_eval",     200))
    cfg.NZ_MA_IN       = float(nz.get("Ma_in",    0.01))
    cfg.NZ_RTOL        = float(nz.get("rtol",     1e-6))
    cfg.NZ_ATOL        = float(nz.get("atol",     1e-6))

    u = ic.get("uniform", {})
    cfg.P_INIT    = float(u.get("p",          1.0e6))
    cfg.ALPHA_G   = float(u.get("alpha_g",    0.3))
    cfg.U_G_INIT  = float(u.get("u_g",        5.0))
    cfg.U_L_INIT  = float(u.get("u_l",        2.0))
    cfg.H_G_OFFSET = float(u.get("h_g_offset", 0.0))
    cfg.H_L_OFFSET = float(u.get("h_l_offset", 0.0))

    # ── Boundary conditions ───────────────────────────────────────────────────
    bc = raw.get("bc", {})
    cfg.BC_INLET  = str(bc.get("inlet",  "transmissive"))
    cfg.BC_OUTLET = str(bc.get("outlet", "transmissive"))
    cfg.P_OUT     = float(bc.get("p_out", cfg.P_INIT))   # back-pressure for bc.outlet='pressure'
    cfg.P_OUT_RELAX = float(bc.get("p_out_relax", 1.0))  # 1=hard clamp (reflective), <<1=relaxed

    # Stagnation (reservoir) state for bc.inlet='stagnation' — mirrors the
    # single-phase settings_nozzle_validation.yaml `boundary_conditions`, extended
    # for two phases (per-phase stagnation T, composition, slip).  All optional.
    cfg.BC_P0      = float(bc["p0"])            if "p0"            in bc else None
    cfg.BC_T_G0    = float(bc["T_g0"])          if "T_g0"          in bc else None
    cfg.BC_T_L0    = float(bc["T_l0"])          if "T_l0"          in bc else None
    cfg.BC_QUALITY = float(bc["quality"])       if "quality"       in bc else None
    cfg.BC_VOID    = float(bc["void_fraction"]) if "void_fraction" in bc else None
    cfg.BC_SLIP    = float(bc.get("slip", 1.0))  # inlet slip ratio S = u_g/u_l
    # Dominant phase for the stagnation-inlet pressure closure (which phase's
    # isentrope pins the shared static p).  'auto' = mass-fraction rule (density-
    # aware) in the driver; 'gas'/'liquid' force it.
    cfg.DOM_PHASE  = str(bc.get("dom_phase", "auto")).lower()

    rim   = ic.get("riemann", {})
    left  = rim.get("left",  {})
    right = rim.get("right", {})
    cfg.X_DISCONT  = float(rim.get("x_discont", 0.5))
    cfg.P_L        = float(left.get("p",       1.2e6))
    cfg.ALPHA_G_L  = float(left.get("alpha_g", 0.4))
    cfg.U_G_L      = float(left.get("u_g",     5.0))
    cfg.U_L_L      = float(left.get("u_l",     2.0))
    cfg.P_R        = float(right.get("p",       8.0e5))
    cfg.ALPHA_G_R  = float(right.get("alpha_g", 0.6))
    # U_G_R and U_L_R are NOT stored here — they are computed in main
    # from mass-flux continuity (require EOS density lookups).

    # ── Save options (output-directory layout) ─────────────────────────────────
    # repo_main = PARENT results folder (e.g. 'results', 'results_two_phase').
    # repo_run  = run-name PREFIX (e.g. 'two_component_shock', 'smd').  The driver
    # appends the auto part _N{Nx}_{recon}_{timestamp}.  Both OPTIONAL → when empty
    # the driver falls back to the legacy 'results' / 'nozzle_valid_{dom}dom'.
    so = raw.get("save_options") or {}
    cfg.REPO_MAIN = str(so["repo_main"]).strip() if so.get("repo_main") else None
    cfg.REPO_RUN  = str(so["repo_run"]).strip()  if so.get("repo_run")  else None

    return cfg
