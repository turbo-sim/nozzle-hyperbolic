"""
Loads a two-fluid settings YAML and returns a SimpleNamespace cfg object
with the same attributes that the old settings_two_fluid.py exposed.

DEFAULTS CHAIN (2026-10-06)
---------------------------
A settings file names the file it builds on with a top-level `base:` key:

    base: defaults_flashing.yaml      # → its own base: defaults.yaml → base: none

Values are merged from the end of the chain to the case file, so the case file
overrides its defaults files and a family defaults file overrides the common
one.  Rules:

* `base:` paths are relative to the file that names them, else to settings/.
* `base: none` marks a complete file (the end of the chain).
* A file WITHOUT a `base:` key is a legacy file: it builds on
  settings/defaults_legacy.yaml (the defaults that used to be hard-coded here),
  so old settings files and the settings copies in old run folders load exactly
  as before.
* Sections are merged key by key, except `geometry` and `ic`, which a file
  replaces as a whole (their keys depend on geometry.function / ic.type).
* In the fixed sections (sources, grid, numerics, time_control, run, bc) every
  key must be set somewhere in the chain — there are no silent code defaults —
  and, in files that declare `base:`, an unknown key is an error (typo guard).
* `eos`, `geometry`, `ic`, `save_options` and `grid_sequence` are free-form.

load_config() keeps the resolved settings in cfg._settings; print_settings()
shows every value with the file it came from and write_resolved_settings()
saves them as a complete `base: none` file that reproduces the run.
"""

import copy
import difflib
import pathlib
import re
import types
from dataclasses import dataclass, field

import yaml

SETTINGS_DIR = pathlib.Path(__file__).resolve().parent / "settings"
LEGACY_DEFAULTS = SETTINGS_DIR / "defaults_legacy.yaml"

# Keys of the fixed sections, in print order.  Every one must be set in the
# chain (KeyError otherwise); keys not listed are rejected in new-style files.
SCHEMA = {
    "sources": ["bubble_number", "droplet_number", "mass_transfer", "scale",
                "area_source", "p_phase_change_max", "regime_htc"],
    "grid": ["Nx", "x1", "x2", "radius"],
    "numerics": [
        "CFL", "newton_iter", "delta_factor", "recon", "k_vm",
        "va_eps_mode", "va_rel", "va_r0", "va_rm", "va_mean_stop_gradient",
        "jacobian", "linear_solve", "w_char", "eos_project", "alpha_eps",
        "pin_ghost", "pin_ghost_momentum", "pin_mom_alpha", "pin_mom_tau",
        "proj_s_ref", "eos_rows",
        "eos_rows_banded", "jac_lag", "nc_form", "lowmach_diss",
        "lowmach_diss_bc", "nc_scale", "eps_void", "eps_void_ucut",
        "eps_void_uwidth", "eps_void_scale", "eps_void_mass_acut",
        "ghost_mass_upwind_acut", "eps_ghost", "eps_ghost_acut",
        "eps_ghost_awidth", "eps_ghost_k2", "eps_ghost_alo", "ssr_mode",
        "du_blend_lo", "du_blend_hi",
        "fast_wb_flux", "flux_single_branch", "flux_local_jac",
        "cell_local_jac", "thomas", "newton_tol"],
    "time_control": ["adaptive_dt", "max_halvings", "dp_max", "dh_max", "du_max",
                     "da_max", "dt_grow", "ser_beta", "ser_cfl_max",
                     "dt_check_dominant_only", "dt_floor_frac", "floor_stall_steps"],
    "run": ["n_steps", "print_every", "ssr_tol", "t_end", "plateau_window",
            "plateau_ratio", "visualize_ic", "debug_newton", "fast_increments",
            "cpu_affinity"],
    "bc": ["inlet", "outlet", "p_out", "p_out_relax", "p_out_ramp_steps",
           "p_out_ramp_from", "p_out_ramp_shape", "p0", "T_g0", "T_l0",
           "quality", "void_fraction", "slip", "dom_phase"],
}
FREE_FORM = ("fluid", "eos", "geometry", "ic", "save_options", "grid_sequence")
SECTION_ORDER = ("fluid", "eos", "grid", "geometry", "sources", "numerics",
                 "time_control", "run", "ic", "bc", "save_options", "grid_sequence")
REPLACE_WHOLE = ("geometry", "ic")
# Keys that no longer do anything: ignored in legacy files, an error elsewhere.
RETIRED = {"numerics": {"jacobian_validate", "profile"}}
_STR_KEYS = {"cpu_affinity"}                      # strings even when they look numeric
_NUM = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")


@dataclass
class Settings:
    """A resolved settings chain."""
    path: pathlib.Path                 # the case file
    raw: dict                          # merged values (no `base` key)
    chain: list                        # [case file, its base, ..., the `base: none` file]
    source: dict = field(default_factory=dict)   # "section.key" → label of the file that set it
    legacy: bool = False               # chain ends in defaults_legacy.yaml
    overrides: dict = field(default_factory=dict)  # "section.key" → label (env, CLI, ...)
    notes: list = field(default_factory=list)      # warnings collected while loading

    def label(self, p):
        return "case" if p == self.path else _short(p)


def _short(p):
    p = pathlib.Path(p)
    try:
        return str(p.relative_to(SETTINGS_DIR)).replace("\\", "/")
    except ValueError:
        return p.name


def _read_yaml(p):
    with open(p, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    if d is None:
        return {}
    if not isinstance(d, dict):
        raise ValueError(f"{p}: a settings file must be a mapping of sections")
    return d


def _find_base(name, here):
    """`base:` path: relative to the naming file's folder, else to settings/."""
    p = pathlib.Path(str(name))
    if p.is_absolute():
        cands = [p]
    else:
        cands = [here.parent / p, SETTINGS_DIR / p]
    for c in cands:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"{here}: base file {name!r} not found (looked in "
                            + ", ".join(str(c.parent) for c in cands) + ")")


def _leaves(d, prefix=""):
    """Dotted paths of every non-dict value (empty dicts count as values)."""
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            yield from _leaves(v, key + ".")
        else:
            yield key


def _merge(dst, src, label, source, prefix=""):
    """Merge src into dst in place, recording the file label of every leaf."""
    for k, v in src.items():
        key = f"{prefix}{k}"
        top = not prefix
        if top and v is None and isinstance(dst.get(k), dict):
            continue              # `section:` with only comments under it → no change
        if isinstance(v, dict) and isinstance(dst.get(k), dict) and not (
                top and k in REPLACE_WHOLE):
            _merge(dst[k], v, label, source, key + ".")
            continue
        for s in [s for s in source if s == key or s.startswith(key + ".")]:
            del source[s]
        dst[k] = copy.deepcopy(v)
        if isinstance(v, dict) and v:
            for leaf in _leaves(v, key + "."):
                source[leaf] = label
        else:
            source[key] = label


def _check_keys(d, p, strict, notes):
    """Unknown / retired keys of one file of the chain."""
    errors = []
    for sec, val in d.items():
        if sec == "base":
            continue
        if sec not in SCHEMA and sec not in FREE_FORM:
            errors.append(f"unknown section {sec!r}"
                          + _suggest(sec, list(SCHEMA) + list(FREE_FORM)))
            continue
        if sec in SCHEMA and isinstance(val, dict):
            for k in val:
                if k in RETIRED.get(sec, ()):
                    if strict:
                        errors.append(f"{sec}.{k} no longer does anything — remove it")
                elif k not in SCHEMA[sec]:
                    errors.append(f"unknown key {sec}.{k}" + _suggest(k, SCHEMA[sec]))
    if not errors:
        return
    if strict:
        raise ValueError(f"{p}:\n  " + "\n  ".join(errors))
    notes.extend(f"{_short(p)}: {e} (ignored)" for e in errors)


def _suggest(k, options):
    low = {str(o).lower(): str(o) for o in options}
    m = difflib.get_close_matches(str(k).lower(), list(low), n=1)
    return f" (did you mean {low[m[0]]!r}?)" if m else ""


def resolve_settings(yaml_path, overrides=None, override_label="override"):
    """Resolve the `base:` chain of a settings file → Settings.

    overrides: optional {"section.key": value} applied on top of the case file
    (e.g. a grid-sequencing level or an environment variable), recorded under
    `override_label` in the provenance."""
    path = pathlib.Path(yaml_path).resolve()
    chain, files = [], []
    p = path
    while True:
        if p in chain:
            raise ValueError("circular base chain: "
                             + " → ".join(_short(c) for c in chain + [p]))
        d = _read_yaml(p)
        chain.append(p)
        files.append(d)
        if "base" not in d:
            nxt = LEGACY_DEFAULTS            # legacy file
        elif str(d["base"]).strip().lower() in ("none", "null", "") or d["base"] is None:
            break
        else:
            nxt = _find_base(d["base"], p)
        p = nxt

    s = Settings(path=path, raw={}, chain=chain, legacy=LEGACY_DEFAULTS in chain)
    for p, d in zip(chain, files):
        _check_keys(d, p, strict="base" in d, notes=s.notes)
    for p, d in reversed(list(zip(chain, files))):
        _merge(s.raw, {k: v for k, v in d.items() if k != "base"}, s.label(p), s.source)
    # Legacy files may carry keys nothing reads (reported in s.notes): drop them so
    # the resolved settings are a valid strict file.
    for sec in [k for k in s.raw if k not in SCHEMA and k not in FREE_FORM]:
        del s.raw[sec]
    for sec, keys in SCHEMA.items():
        if isinstance(s.raw.get(sec), dict):
            for k in [k for k in s.raw[sec] if k not in keys]:
                del s.raw[sec][k]
    s.source = {k: v for k, v in s.source.items() if _has(s.raw, k)}
    # PyYAML reads 1e-6 / 25e5 / 1.0e12 (no dot or no exponent sign) as STRINGS;
    # the loader converts them with float() anyway.  Store them as numbers so the
    # printed / saved settings are clean.
    for sec in SCHEMA:
        for k, v in (s.raw.get(sec) or {}).items():
            if isinstance(v, str) and k not in _STR_KEYS and _NUM.fullmatch(v.strip()):
                s.raw[sec][k] = float(v)
    for key, val in (overrides or {}).items():
        set_value(s, key, val, override_label)
    return s


def set_value(s, key, value, label="override"):
    """Change one dotted setting of a resolved Settings (recorded as an override)."""
    sec, _, rest = key.partition(".")
    if sec in SCHEMA and rest not in SCHEMA[sec]:
        raise KeyError(f"unknown setting {key!r}" + _suggest(rest, SCHEMA[sec]))
    if sec not in SCHEMA and sec not in FREE_FORM:
        raise KeyError(f"unknown section in {key!r}")
    node, last = s.raw, sec
    for k in rest.split(".") if rest else []:
        node = node.setdefault(last, {})
        last = k
    node[last] = copy.deepcopy(value)
    for leaf in [x for x in s.source if x == key or x.startswith(key + ".")]:
        del s.source[leaf]
    if isinstance(value, dict) and value:
        for leaf in _leaves(value, key + "."):
            s.source[leaf] = label
    else:
        s.source[key] = label
    s.overrides[key] = label


def _has(raw, key):
    node = raw
    for k in key.split("."):
        if not isinstance(node, dict) or k not in node:
            return False
        node = node[k]
    return True


class _Sec(dict):
    """dict whose missing keys raise a KeyError that names the chain."""
    def __init__(self, name, d, settings):
        super().__init__(d or {})
        self._name, self._s = name, settings

    def __missing__(self, k):
        raise KeyError(f"{self._name}.{k} is not set in {_short(self._s.path)} or its "
                       f"defaults ({' → '.join(_short(c) for c in self._s.chain)}); "
                       f"add it to the case file or to a defaults file")


def _none_or(v, conv):
    return None if v in (None, "", "none", "None") else conv(v)


def load_config(yaml_path, overrides=None, override_label="override") -> types.SimpleNamespace:
    s = resolve_settings(yaml_path, overrides, override_label)
    raw = s.raw

    def sec(name):
        return _Sec(name, raw.get(name), s)

    cfg = types.SimpleNamespace()
    cfg._settings = s

    # ── Fluid ────────────────────────────────────────────────────────────────
    cfg.FLUID_NAME = str(sec("fluid")["name"])

    # ── EOS tables ───────────────────────────────────────────────────────────
    # Backends that are ANALYTIC (no table): built as FluidPerfectGas, not
    # FluidBicubic.  Ideal as a passive GHOST phase — unbounded (no table edge →
    # no extrapolation blow-up / dt-collapse) and needs no h/p range or table_dir.
    _PG_BACKENDS = ("perfect_gas", "perfectgas", "ideal_gas", "idealgas")

    def _lut(section: str) -> dict:
        s_ = sec("eos")[section]
        backend = str(s_["backend"])
        if backend.lower() in _PG_BACKENDS:
            # Perfect gas: only name + reference state (constants fit at T_ref,p_ref).
            # No h/p bounds needed (analytic, in-range everywhere).  The driver
            # dispatches on `backend` to build jxp.FluidPerfectGas.
            return {
                "backend":    backend,
                "fluid_name": str(s_.get("fluid_name", cfg.FLUID_NAME)),
                "T_ref":      float(s_.get("T_ref", 300.0)),
                "p_ref":      float(s_.get("p_ref", 101_325.0)),
            }
        d = {
            # per-phase fluid_name (falls back to the global fluid.name) so the two
            # phases can be DIFFERENT fluids (e.g. gas=Air, liquid=water).
            "fluid_name":      str(s_.get("fluid_name", cfg.FLUID_NAME)),
            "backend":         str(s_["backend"]),
            "h_min":           float(s_["h_min"]),
            "h_max":           float(s_["h_max"]),
            "p_min":           float(s_["p_min"]),
            "p_max":           float(s_["p_max"]),
            "N_h":             int(s_["N_h"]),
            "N_p":             int(s_["N_p"]),
            "table_dir":       str(s_["table_dir"]),
            "gradient_method": str(s_["gradient_method"]),
        }
        # metastable_phase is OPTIONAL: omit it for an ordinary single-phase table
        # (no metastable extension, no saturation props — correct for Air or a
        # purely subcooled liquid in a single-phase validation).
        if s_.get("metastable_phase") not in (None, "", "none", "None"):
            d["metastable_phase"] = str(s_["metastable_phase"])
        if "N_p_sat" in s_:
            d["N_p_sat"] = int(s_["N_p_sat"])
        return d

    cfg.LUT_G = _lut("gas")
    cfg.LUT_L = _lut("liquid")

    # ── Interphase-structure closure (TA-BD interfacial area) ─────────────────
    src = sec("sources")
    cfg.BUBBLE_NUMBER  = float(src["bubble_number"])
    cfg.DROPLET_NUMBER = float(src["droplet_number"])
    # mass_transfer: True  → single-component with phase change (D_VE_algebric,
    #                        needs saturation props); False → two-component, drag+heat
    #                        only (D_VE_algebric_two_component, no saturation).
    cfg.MASS_TRANSFER  = bool(src["mass_transfer"])
    # Multiplier of the physical interphase sources (drag, heat, mass transfer);
    # 0 = off (diagnostic).  The SRC_SCALE environment variable overrides it.
    cfg.SRC_SCALE      = float(src["scale"])
    # Area (quasi-1D wall-pressure) source term on/off (diagnostic).
    cfg.AREA_SRC       = bool(src["area_source"])
    # No phase change above this pressure, as a FRACTION of the critical pressure
    # (null = no cut-off).  For supercritical inlets: the saturation table ends at
    # 0.98 p_crit and is clamped above.  Converted to Pa in main_two_fluid.py.
    # .get: resolved settings saved by runs before this key existed (base: none,
    # no defaults to fall back on) still load, with the cut-off off.
    cfg.P_PHASE_CHANGE_MAX = _none_or(src.get("p_phase_change_max"), float)
    # Interphase heat transfer: false = NICFD closure (bubbly-regime HTC at every
    # alpha); true = regime-dependent (bubbly / linear blend / droplet).  .get:
    # resolved settings of runs older than this key load with the NICFD closure.
    cfg.REGIME_HTC = bool(src.get("regime_htc", False))

    # ── Grid ─────────────────────────────────────────────────────────────────
    g = sec("grid")
    cfg.Nx     = int(g["Nx"])
    cfg.x1     = float(g["x1"])
    cfg.x2     = float(g["x2"])
    cfg.RADIUS = _none_or(g["radius"], float)   # legacy; only used if no geometry block

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
        if cfg.RADIUS is None:
            raise KeyError("no geometry block and grid.radius is null — set one of them")
        geo = {"function": "get_pipe", "radius": cfg.RADIUS}
    cfg.GEO_FUNCTION  = str(geo.pop("function", "get_pipe"))
    cfg.GEO_ROUGHNESS = float(geo.pop("roughness", 0.0))
    cfg.GEO_KWARGS    = geo   # remaining keys forwarded verbatim to the factory

    # ── Numerics ─────────────────────────────────────────────────────────────
    n = sec("numerics")
    cfg.CFL          = float(n["CFL"])
    cfg.NEWTON_ITER  = int(n["newton_iter"])
    cfg.DELTA_FACTOR = float(n["delta_factor"])
    cfg.RECON        = str(n["recon"])
    # van Albada eps normalisation: 'magnitude' (original) | 'range' | 'smoothness';
    # va_rel null → mode-calibrated default.
    cfg.VA_EPS_MODE  = str(n["va_eps_mode"])
    cfg.VA_REL       = _none_or(n["va_rel"], float)
    cfg.VA_R0        = float(n["va_r0"])   # smoothness-gate knee
    cfg.VA_RM        = int(n["va_rm"])     # smoothness-gate exponent
    # 'magnitude' only: stop_gradient on the eps domain mean (same residual,
    # colored Jacobian stays exact for thin ghosts); false = earlier code path.
    cfg.VA_MEAN_STOP_GRADIENT = bool(n["va_mean_stop_gradient"])
    cfg.K_VM         = float(n["k_vm"])
    cfg.JACOBIAN          = str(n["jacobian"])          # dense | colored
    cfg.LINEAR_SOLVE      = str(n["linear_solve"])      # lstsq | normal_eq | normal_eq_banded | qr
    # Newton column-scaling floors (8 values, W order p,αg,mg,ug,ml,ul,mg_eg,ml_el),
    # or 'auto' (2026-07-17, RECOMMENDED): per-variable 0.05·max|W| recomputed from
    # the current Newton iterate — reproduces hand floors bit-identically wherever
    # they sit at/below the field scale, adapts through flashing (α 1e-4→0.9), no
    # per-case tuning.  None → solver defaults (two-component calibrated; on a
    # GHOST state they exceed |W| ×20-500 and inflate the SSR floor ×180).
    _wc = n["w_char"]
    cfg.W_CHAR            = ("auto" if isinstance(_wc, str) and _wc == "auto"
                             else [float(v) for v in _wc] if _wc is not None
                             else None)
    # Per-iteration EOS-manifold projection: true (classic 2×2 pair) | false |
    # 'damped' (conditioning-gated pair: λ = s/(s+s_ref) per cell/pass from the
    # 2×2 Jacobian's own conditioning — reduces to pair where well-posed and to
    # false in the ghost/stiff-liquid limit; α-uniform, USE FOR FLASHING) |
    # 'triangular' (FAILED certification 2026-07-15 — any method converging to
    # the exact root inherits dp/dm_g ≈ 1e8 in the ghost limit; do not use).
    _ep = n["eos_project"]
    cfg.EOS_PROJECT       = _ep if isinstance(_ep, str) else bool(_ep)
    # Solver-side void-fraction clip floor.  Flashing runs with a thin liquid-side
    # ghost should match the IC floor (~1e-4..1e-3) so the Newton clip cannot
    # re-thin α below it mid-march (the h_g = E/(αρ) amplifier behind the
    # flashing-onset blow-up).
    cfg.ALPHA_EPS         = float(n["alpha_eps"])
    # Ghost-vapor saturation pin (nozzlex parity): blend the gas-energy residual
    # toward h_g = h_sat_g(p) where the liquid is NOT superheated (weight
    # 1 − evaporation_weight, complementary to the σ gate — cannot block onset).
    # Requires mass_transfer/use_saturation.
    cfg.PIN_GHOST         = bool(n["pin_ghost"])
    # Ghost-GAS momentum pin (u_g → u_l), α-gated & saturation-free — serves both
    # flashing and Bernoulli ghost-gas cases.
    cfg.PIN_GHOST_MOMENTUM = bool(n["pin_ghost_momentum"])
    # Thin-ghost switches (2026-10-08, solver docstring "THIN-GHOST SWITCHES");
    # null = previous code path.  pin_mom_alpha: α of half release (null =
    # 10·alpha_eps); pin_mom_tau [s]: dt-consistent pin (null = algebraic blend).
    cfg.PIN_MOM_ALPHA = None if n["pin_mom_alpha"] is None else float(n["pin_mom_alpha"])
    cfg.PIN_MOM_TAU   = None if n["pin_mom_tau"] is None else float(n["pin_mom_tau"])
    cfg.PROJ_S_REF        = float(n["proj_s_ref"])          # λ gate scale for 'damped'
    # (λ = s²/(s²+s_ref²); probe-calibrated: two-comp s≈0.36, ghost s≈9e-4)
    # E3-augmented system (2026-07-15, certified ×3): append the 2 EOS closures
    # as residual rows → 8 rows × 8 unknowns least squares, NO per-cell null
    # space → no manifold drift, no projection needed (set eos_project: false),
    # CFL up to 5 certified.  Dense solve path by default (~1.5-2 s/step at
    # N300); eos_rows_banded switches to the O(N) banded twin.
    cfg.EOS_ROWS          = bool(n["eos_rows"])
    # E3-BANDED (2026-07-16): block-banded JᵀJ + super-block Thomas for the
    # eos_rows system — SAME numbers as the dense path, O(N) instead of O(N³).
    cfg.EOS_ROWS_BANDED   = bool(n["eos_rows_banded"])
    # JACOBIAN LAGGING (modified Newton, 2026-07-16; E3-banded path only):
    # recompute J every jac_lag-th Newton iteration (residual still every
    # iteration → converged answer unchanged, only the Newton direction lags).
    # jac_lag >= newton_iter = one Jacobian per step (~40 colored JVPs paid
    # once).  1 = certified per-iteration behavior.
    cfg.JAC_LAG           = int(n["jac_lag"])
    # NC discretization: 'cell' (legacy) | 'wb' (well-balanced face-area-weighted
    # one-sided differences — fixes the α-front-on-taper imbalance, reduces to
    # 'cell' on uniform area; certified 2026-07-15) | 'staedtke' (probe only).
    cfg.NC_FORM           = str(n["nc_form"])
    # Low-Mach acoustic-dissipation floor for the FVS flux: z_ac = clip(Ma, x, 1).
    #   0.0 -> z_ac=Ma  (reduces acoustic/pressure dissipation at low Ma — the
    #                    setting that REMOVES the pressure diffusion and lets the
    #                    low-Mach checkerboard grow),
    #   1.0 -> z_ac=1   (full standard dissipation; checkerboard-free but over-
    #                    dissipative at low Ma).
    cfg.LOWMACH_DISS      = float(n["lowmach_diss"])
    # BOUNDARY-FACE z_ac floor (2026-07-17): pressure/reservoir BCs couple to
    # the interior only through the acoustic jump terms of the first/last face
    # flux — Ma-scaled away when lowmach_diss=0 (standing outlet jump, creeping
    # mass flow).  1.0 = full upwind dissipation on the two boundary faces only
    # (stiff BC enforcement); null = inherit lowmach_diss (no change).
    # Supersonic outlet: no-op (z_ac=1 there anyway, ghost transmissive).
    cfg.LOWMACH_DISS_BC   = _none_or(n["lowmach_diss_bc"], float)
    # Non-conservative correction scale (DIAGNOSTIC): 1.0 = on (physical),
    # 0.0 = OFF — used to test whether the H^nc coupling is pumping the
    # void-fraction checkerboard.
    cfg.NC_SCALE          = float(n["nc_scale"])
    # 4th-order (JST) background filter coefficient for the void/alpha 2Δx
    # checkerboard.  0.0 = off.  ~0.05-0.5 typical; 4th-order → smooth solution
    # perturbed only at O(eps_void*dx^3), so sweep it and confirm ε-independence.
    cfg.EPS_VOID          = float(n["eps_void"])
    # LOW-VELOCITY GATE for the JST void filter.  null → ungated (uniform).  When set
    # to a phase-velocity cutoff u0 [m/s], the filter is weighted by
    # ½(1+tanh((u0−u_mat)/Δu)) with u_mat=min(|u_g|,|u_l|), so it acts ONLY in the
    # low-velocity (inlet) region — where the material-wave dissipation vanishes —
    # and leaves the throat/supersonic at full high-order accuracy.  Gates on the
    # VELOCITY, not a Mach (a_mix is the α-dependent Wood speed → poor denominator).
    cfg.EPS_VOID_UCUT   = _none_or(n["eps_void_ucut"], float)
    cfg.EPS_VOID_UWIDTH = _none_or(n["eps_void_uwidth"], float)   # null → solver uses u0/3
    # FILTER-STRENGTH scale: 'wood' uses the model's true mixture eigenvalue
    # (dips at α≈0.5 → starves the filter there); 'frozen' uses a non-dipping
    # |u_mix|+max(a_g,a_l) surrogate so the 4th-order damping magnitude stays ~constant
    # in α (fluxes/CFL still use the true dipping speed).  Retune eps_void when switching.
    cfg.EPS_VOID_SCALE = str(n["eps_void_scale"]).strip().lower()
    # GHOST-phase 4th-order filter: same conservative JST flux as eps_void but
    # gated per ROW by the phase's own smallness in log10(α) (on below acut, off
    # above — never touches a phase carrying real mass; flashing-safe) and NOT
    # velocity-gated (a ghost needs damping at any speed, incl. the shock layer
    # where the ghost limit-cycle lives).  0.0 = off (bit-identical).
    cfg.EPS_GHOST        = float(n["eps_ghost"])
    cfg.EPS_GHOST_ACUT   = float(n["eps_ghost_acut"])
    cfg.EPS_GHOST_AWIDTH = float(n["eps_ghost_awidth"])
    # 2nd-order (shock-zone) coefficient multiplier: eps2 = k2·eps_ghost·σ.
    # 4 ≡ equal 2Δx damping across the JST sensor cross-fade.
    # k2 = 0.0 (2026-07-18): ANY strong ghost dissipation at the shock digs
    # an α-hole (v3 via the mass row, v4 k2=4 via momentum→continuity: forcing
    # u_ghost smooth across the jump makes the ghost's own continuity pull α to
    # the clip floor → 1/α blow-up).  σ therefore just SWITCHES THE GHOST FILTER
    # OFF at the shock; the bounded shock-cell ghost cycle is handled by
    # ssr_mode: mass + the masked gates instead.
    cfg.EPS_GHOST_K2     = float(n["eps_ghost_k2"])
    # BAND-GATE floor: filter disengages below this α (hole = garbage fields;
    # touching them locks the hole open — let drag heal it).
    cfg.EPS_GHOST_ALO    = float(n["eps_ghost_alo"])
    # Thin-ghost switches (null = previous code path): ghost mass row out of the
    # eps_void filter below ~acut; upwind ghost mass flux at faces touching α < ~acut.
    cfg.EPS_VOID_MASS_ACUT = (None if n["eps_void_mass_acut"] is None
                              else float(n["eps_void_mass_acut"]))
    cfg.GHOST_MASS_UPWIND_ACUT = (None if n["ghost_mass_upwind_acut"] is None
                                  else float(n["ghost_mass_upwind_acut"]))
    # SSR row weighting: 'all' (legacy, every equation votes equally — a ghost
    # phase's shock-layer limit cycle can pin the residual far above tol) or
    # 'mass' (each phase's rows vote with the local mass fraction — the stop
    # criterion follows the phases that actually carry mass; flashing-safe).
    cfg.SSR_MODE = str(n["ssr_mode"]).strip().lower()
    # ZERO-SLIP FLUX BLEND window [m/s]: the full-branch flux inverts a matrix whose
    # void-wave columns go parallel as du→0 (κ~1/du) → noisy in a ring around every
    # slip sign-change.  Use the well-conditioned zero-slip branch for |du|<lo, the
    # accurate full branch for |du|>hi, C¹ cosine blend between.  Widen on coarser
    # meshes (du changes more per cell) so the blend still spans a few cells.
    cfg.DU_BLEND_LO = float(n["du_blend_lo"])
    cfg.DU_BLEND_HI = float(n["du_blend_hi"])
    # ── SPEED-UP SWITCHES (2026-10-05) ─ false/orig/0 = the previous code path ──
    # The first five do not change the result (residual identical to ~1e-15, an
    # N600 restart reproduced a 3,410-step reference run to 1e-13 in every field);
    # see make_two_fluid_step's docstring and SPEEDUP_REPORT.md.
    cfg.FAST_WB_FLUX       = bool(n["fast_wb_flux"])        # wb: F_c = F(U) directly
    cfg.FLUX_SINGLE_BRANCH = bool(n["flux_single_branch"])  # slip branches evaluated once
    cfg.FLUX_LOCAL_JAC     = bool(n["flux_local_jac"])      # per-face 6×12 local Jacobian
    cfg.CELL_LOCAL_JAC     = bool(n["cell_local_jac"])      # per-cell local Jacobians
    cfg.THOMAS             = str(n["thomas"]).strip().lower()  # orig | chol
    # Newton early exit: stop when max|dW_scaled| < newton_tol (0 = always newton_iter).
    # Per-step results change only at ~newton_tol² (1e-8 recommended).
    cfg.NEWTON_TOL         = float(n["newton_tol"])

    # ── Time-step control (Städtke-style adaptive halving) ─────────────────────
    tc = sec("time_control")
    cfg.ADAPTIVE_DT  = bool(tc["adaptive_dt"])   # False = fixed CFL dt
    cfg.MAX_HALVINGS = int(tc["max_halvings"])
    cfg.DP_MAX       = float(tc["dp_max"])       # max |Δp|/p     per step
    cfg.DH_MAX       = float(tc["dh_max"])       # max |Δh|/|h|   per step
    cfg.DU_MAX       = float(tc["du_max"])       # max |Δu|/(|u|+a) per step
    cfg.DA_MAX       = float(tc["da_max"])       # max |Δα_g|     per step (abs)
    cfg.DT_GROW      = float(tc["dt_grow"])      # dt growth after accept
    # SER CFL ramp (2026-10-05, adaptive_dt only): CFL_eff = CFL·clip((SSR_1/SSR)^ser_beta,
    # 1, ser_cfl_max/CFL), SSR_1 = SSR after the first step.  Pseudo-transient
    # continuation: small dt in the transient, near-Newton once converging.  Changes
    # the PATH, not the steady equations (SSR uses dt=1, dV=0).  0 = off (fixed CFL).
    cfg.SER_BETA     = float(tc["ser_beta"])
    cfg.SER_CFL_MAX  = float(tc["ser_cfl_max"])
    # Base the adaptive-dt increment check on the DOMINANT phase only, ignoring
    # the near-massless GHOST's u/h (which, with no drag, oscillate freely and
    # can trigger halving / dt-collapse even when the dominant phase is fine).
    cfg.DOM_ONLY_DT       = bool(tc["dt_check_dominant_only"])
    # Keep-marching controller: a step that violates the increment thresholds is
    # taken with dt clamped to dt_floor_frac × the CFL dt instead of stopping the
    # run; the run stops only on NaN/Inf or after floor_stall_steps consecutive
    # floored steps without SSR progress.
    cfg.DT_FLOOR_FRAC     = float(tc["dt_floor_frac"])
    cfg.FLOOR_STALL_STEPS = int(tc["floor_stall_steps"])

    # ── Run control ───────────────────────────────────────────────────────────
    r = sec("run")
    cfg.N_STEPS        = int(r["n_steps"])
    cfg.PRINT_EVERY    = int(r["print_every"])
    cfg.VISUALIZE_IC   = bool(r["visualize_ic"])
    cfg.DEBUG_NEWTON   = bool(r["debug_newton"])
    cfg.SSR_TOL        = float(r["ssr_tol"])
    _t_end_raw         = r["t_end"]
    cfg.T_END          = float("inf") if _t_end_raw is None else float(_t_end_raw)
    cfg.PLATEAU_WINDOW = int(r["plateau_window"])
    cfg.PLATEAU_RATIO  = float(r["plateau_ratio"])
    # jit the driver's per-step increment check (eager: ~16 ms/step of op-by-op
    # dispatch; jitted: ~0.1 ms).  Same numbers.
    cfg.FAST_INCREMENTS = bool(r["fast_increments"])
    # Pin the process to these logical CPUs, e.g. "0-15" (P-cores on the i7-13850HX
    # laptop; E-cores 16-27 are ~1.9× slower and Windows may schedule the run
    # there).  null = no pinning.
    cfg.CPU_AFFINITY = _none_or(r["cpu_affinity"], str)

    # ── Initial condition (free-form: keys depend on ic.type) ─────────────────
    ic = sec("ic")
    cfg.IC_TYPE        = str(ic["type"]).strip().lower()

    # nozzlex-seeded IC: run the space-marching pipe solve and interpolate its
    # relaxed profile onto the time-marching grid (see initial_conditions.py).
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
    bc = sec("bc")
    cfg.BC_INLET  = str(bc["inlet"])
    cfg.BC_OUTLET = str(bc["outlet"])
    # back-pressure for the pressure/characteristic outlets; null → ic.uniform.p
    cfg.P_OUT     = _none_or(bc["p_out"], float)
    if cfg.P_OUT is None:
        cfg.P_OUT = cfg.P_INIT
    cfg.P_OUT_RELAX = float(bc["p_out_relax"])  # 1=hard clamp (reflective), <<1=relaxed
    # RAMPED START-UP (2026-10-06): lower the outlet back-pressure linearly from
    # p_out_ramp_from to p_out over the first p_out_ramp_steps steps ("opening the
    # valve" from a stagnation IC, where p_out = p0 is already steady).  The SSR
    # stop / plateau checks and the SER reference are suspended until the ramp ends.
    # 0 = off.  p_out_ramp_from: null = the IC's pressure in the last interior cell
    # (= p0 for ic.type stagnation).
    cfg.P_OUT_RAMP_STEPS = int(bc["p_out_ramp_steps"])
    cfg.P_OUT_RAMP_FROM  = _none_or(bc["p_out_ramp_from"], float)
    # 'linear' (equal Δp per step) | 'log' (equal p ratio per step: spends more of
    # the ramp at low pressure, where choking/flashing develop).
    cfg.P_OUT_RAMP_SHAPE = str(bc["p_out_ramp_shape"]).strip().lower()

    # Stagnation (reservoir) state for bc.inlet='stagnation' — mirrors the
    # single-phase settings_nozzle_validation.yaml `boundary_conditions`, extended
    # for two phases (per-phase stagnation T, composition, slip).  null = unused
    # (T_l0 null → T_g0; give void_fraction OR quality).
    cfg.BC_P0      = _none_or(bc["p0"], float)
    cfg.BC_T_G0    = _none_or(bc["T_g0"], float)
    cfg.BC_T_L0    = _none_or(bc["T_l0"], float)
    cfg.BC_QUALITY = _none_or(bc["quality"], float)
    cfg.BC_VOID    = _none_or(bc["void_fraction"], float)
    cfg.BC_SLIP    = float(bc["slip"])  # inlet slip ratio S = u_g/u_l
    # Dominant phase for the stagnation-inlet pressure closure (which phase's
    # isentrope pins the shared static p).  'auto' = mass-fraction rule (density-
    # aware) in the driver; 'gas'/'liquid' force it.
    cfg.DOM_PHASE  = str(bc["dom_phase"]).lower()

    # ── Save options (output-directory layout) ─────────────────────────────────
    # repo_main = PARENT results folder (e.g. 'results', 'results_two_phase').
    # repo_run  = run-name PREFIX (e.g. 'two_component_shock', 'smd').  The driver
    # appends the auto part _N{Nx}_{recon}_{timestamp}.  Both OPTIONAL → when empty
    # the driver falls back to the legacy 'results' / 'nozzle_valid_{dom}dom'.
    so = raw.get("save_options") or {}
    cfg.REPO_MAIN = str(so["repo_main"]).strip() if so.get("repo_main") else None
    cfg.REPO_RUN  = str(so["repo_run"]).strip()  if so.get("repo_run")  else None

    return cfg


def resolved_dict(yaml_path, overrides=None, override_label="override"):
    """Merged settings of a file as a plain dict with `base: none` first — a
    complete settings file once dumped (yaml.safe_dump(..., sort_keys=False))."""
    s = resolve_settings(yaml_path, overrides, override_label)
    return {"base": "none", **_ordered(s.raw)}


# ═════════════════════════════════════════════════════════════════════════════
# Showing / saving the resolved settings
# ═════════════════════════════════════════════════════════════════════════════

def _ordered(raw):
    """Sections in SECTION_ORDER, fixed-section keys in SCHEMA order."""
    out = {}
    for sec in list(SECTION_ORDER) + [k for k in raw if k not in SECTION_ORDER]:
        if sec not in raw:
            continue
        v = raw[sec]
        if sec in SCHEMA and isinstance(v, dict):
            v = {k: v[k] for k in SCHEMA[sec] + [k for k in v if k not in SCHEMA[sec]]
                 if k in v}
        out[sec] = v
    return out


def _scalar(v):
    s = yaml.safe_dump(v, default_flow_style=True, width=10**9, allow_unicode=True)
    s = s.strip()
    if s.endswith("\n..."):
        s = s[:-4].rstrip()
    elif s.endswith("..."):
        s = s[:-3].rstrip()
    return s


def _emit(d, settings, prefix, indent, lines, comments):
    pad = "  " * indent
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            lines.append(f"{pad}{_scalar(k)}:")
            _emit(v, settings, key + ".", indent + 1, lines, comments)
        else:
            text = f"{pad}{_scalar(k)}: {_scalar(v)}"
            src = settings.source.get(key, "") if comments else ""
            lines.append(f"{text:<48s} # {src}" if src else text)


def format_settings(settings):
    """Every resolved setting with the file it came from (YAML text)."""
    lines = []
    _emit(_ordered(settings.raw), settings, "", 0, lines, comments=True)
    return "\n".join(lines)


def print_settings(cfg):
    s = cfg._settings
    print("── Settings " + "─" * 66)
    print(f"  case file : {s.path}")
    print(f"  chain     : {' → '.join(_short(c) for c in s.chain)}")
    for key, label in s.overrides.items():
        print(f"  override  : {key} = {_scalar(_get(s.raw, key))}  ({label})")
    for note in s.notes:
        print(f"  WARNING   : {note}")
    print("  (# case = set in the case file; otherwise the defaults file that set it)")
    for line in format_settings(s).splitlines():
        print("  " + line)
    print("─" * 78)


def _get(raw, key):
    node = raw
    for k in key.split("."):
        node = node[k]
    return node


def write_resolved_settings(cfg, path, header=()):
    """Save the resolved settings as a complete `base: none` file; loading it
    gives the same cfg (checked here)."""
    s = cfg._settings
    lines = [
        "# RESOLVED SETTINGS — every value used by the run (base: none).",
        f"# case file : {s.path}",
        f"# chain     : {' → '.join(_short(c) for c in s.chain)}",
        *[f"# override  : {k} ({lbl})" for k, lbl in s.overrides.items()],
        *[f"# {h}" for h in header],
        "# The comment after each value names the file that set it.",
        "base: none",
    ]
    _emit(_ordered(s.raw), s, "", 0, lines, comments=True)
    path = pathlib.Path(path)
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    again = resolve_settings(path).raw
    if _ordered(again) != _ordered(s.raw):
        raise RuntimeError(f"{path}: resolved settings do not round-trip")
    return path
