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
    cfg.LINEAR_SOLVE      = str(n.get("linear_solve", "lstsq"))      # lstsq | normal_eq
    # Low-Mach acoustic-dissipation floor for the FVS flux: z_ac = clip(Ma, x, 1).
    #   0.0 -> z_ac=Ma  (reduces acoustic/pressure dissipation at low Ma — the
    #                    setting that REMOVES the pressure diffusion and lets the
    #                    low-Mach checkerboard grow),
    #   1.0 -> z_ac=1   (full standard dissipation; checkerboard-free but over-
    #                    dissipative at low Ma).
    cfg.LOWMACH_DISS      = float(n.get("lowmach_diss", 0.0))
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

    return cfg
