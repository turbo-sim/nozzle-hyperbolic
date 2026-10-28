"""
Settings loader - R1233zd(E) flashing expansion in the Elliot nozzle.
=====================================================================

Reads the case YAML and exposes it as module-level constants, mirroring how
`dev_scripts/main_two_fluid.py` uses `config_loader.load_config`.  The loader is
local to this folder so the case stays self-contained.

    python run_simulation.py                            # the default case (A)
    python run_simulation.py settings_quality_5pct.yaml # positional argument
    SETTINGS_YAML=other.yaml python run_simulation.py   # or the env var

Imported by `build_tables.py`, `patch_conductivity.py`, `isentrope.py` and
`run_simulation.py`, all of which just read the constants below.

The `bc:` block takes the same keys as the other cases - p0 shared by both
phases, a per-phase T_g0/T_l0, slip, and EITHER quality or void_fraction (the
other is derived from the slip relation).

Some keys accept a keyword instead of a literal, where the literal would be
fragile:

  eos.*.p_max: critical           -> the CoolProp critical pressure
  bc.p0:       saturation         -> p_sat(T_l0)
  numerics.alpha_eps: match_void  -> tied to the inlet void fraction

and the reservoir branch is chosen per phase rather than assumed: an ordinary
(p,T) flash off the saturation line, the phase's own saturation call ON it
(where a (p,T) flash is ambiguous), plus the ghost-vapour fallback when a
subcooled reservoir would put h_g below the vapour table floor.
"""

import os
import pathlib
import sys

import yaml
import CoolProp.CoolProp as CP

# ---------------------------------------------------------------------------
# Paths - every output stays inside this case folder
# ---------------------------------------------------------------------------
CASE_DIR    = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = CASE_DIR / "results"

# Settings file, in order of precedence:
#   1. first positional CLI argument   python run_simulation.py my.yaml
#   2. SETTINGS_YAML environment var   SETTINGS_YAML=my.yaml python run_simulation.py
#   3. the default case
# A bare name is resolved inside this case folder; an absolute or relative path
# is honoured as given.
DEFAULT_SETTINGS = "settings_saturated_liquid.yaml"


def _settings_from_argv():
    """First positional argument that looks like a settings file, or None.

    Skips option-style tokens so a future `--flag` cannot be mistaken for a
    filename, and ignores anything that is not a .yaml/.yml.
    """
    for arg in sys.argv[1:]:
        if arg.startswith("-"):
            continue
        if arg.lower().endswith((".yaml", ".yml")):
            return arg
        raise SystemExit(
            f"ERROR: unrecognised argument {arg!r}.\n"
            f"  usage: python {pathlib.Path(sys.argv[0]).name} [settings.yaml]\n"
            f"  (or set SETTINGS_YAML=<file>)")
    return None


_argv_settings = _settings_from_argv()
SETTINGS_FILE = _argv_settings or os.environ.get("SETTINGS_YAML", DEFAULT_SETTINGS)
SETTINGS_SOURCE = ("command line" if _argv_settings
                   else "SETTINGS_YAML" if os.environ.get("SETTINGS_YAML")
                   else "default")

SETTINGS_PATH = CASE_DIR / SETTINGS_FILE
if not SETTINGS_PATH.exists():
    SETTINGS_PATH = pathlib.Path(SETTINGS_FILE)      # absolute / relative path
if not SETTINGS_PATH.exists():
    _avail = sorted(p.name for p in CASE_DIR.glob("settings_*.yaml"))
    raise FileNotFoundError(
        f"settings file not found: {SETTINGS_FILE}  (from {SETTINGS_SOURCE})\n"
        f"  looked in {CASE_DIR} and as a literal path\n"
        f"  available in this case folder: {', '.join(_avail) or '(none)'}")

with open(SETTINGS_PATH, encoding="utf-8") as _f:
    RAW = yaml.safe_load(_f)


def _req(section: str, key: str):
    """Fetch a required key with a message that names the file."""
    try:
        return RAW[section][key]
    except KeyError:
        raise KeyError(f"{SETTINGS_PATH.name}: missing '{section}.{key}'") from None


# ---------------------------------------------------------------------------
# Fluid
# ---------------------------------------------------------------------------
FLUID_NAME = str(_req("fluid", "name"))
BACKEND    = str(RAW["eos"]["liquid"].get("backend", "HEOS"))

P_CRIT   = float(CP.PropsSI("pcrit",   FLUID_NAME))
T_CRIT   = float(CP.PropsSI("Tcrit",   FLUID_NAME))
RHO_CRIT = float(CP.PropsSI("rhocrit", FLUID_NAME))
T_TRIPLE = float(CP.PropsSI("Ttriple", FLUID_NAME))

# ---------------------------------------------------------------------------
# EOS tables
# ---------------------------------------------------------------------------
def _pressure(v):
    """Resolve a pressure entry; the string 'critical' means p_crit."""
    if isinstance(v, str):
        if v.strip().lower() in ("critical", "p_crit", "pcrit"):
            return P_CRIT
        raise ValueError(f"{SETTINGS_PATH.name}: unknown pressure keyword {v!r} "
                         f"(use a number or 'critical')")
    return float(v)


def _lut(section: str) -> dict:
    s = RAW["eos"][section]
    table_dir = pathlib.Path(str(s.get("table_dir", "fluid_tables")))
    if not table_dir.is_absolute():
        table_dir = CASE_DIR / table_dir
    d = dict(
        fluid_name      = str(s.get("fluid_name", FLUID_NAME)),
        backend         = str(s["backend"]),
        h_min           = float(s["h_min"]),
        h_max           = float(s["h_max"]),
        p_min           = _pressure(s["p_min"]),
        p_max           = _pressure(s["p_max"]),
        N_h             = int(s["N_h"]),
        N_p             = int(s["N_p"]),
        table_dir       = str(table_dir),
        gradient_method = str(s.get("gradient_method", "forward")),
    )
    if s.get("metastable_phase") not in (None, "", "none", "None"):
        d["metastable_phase"] = str(s["metastable_phase"])
    if "N_p_sat" in s:
        d["N_p_sat"] = int(s["N_p_sat"])
    return d


_LUT_L, _LUT_G = _lut("liquid"), _lut("gas")

TABLE_DIR = pathlib.Path(_LUT_L["table_dir"])
P_TABLE_MIN, P_TABLE_MAX = _LUT_L["p_min"], _LUT_L["p_max"]
H_LIQ_MIN, H_LIQ_MAX = _LUT_L["h_min"], _LUT_L["h_max"]
H_VAP_MIN, H_VAP_MAX = _LUT_G["h_min"], _LUT_G["h_max"]
N_H, N_P = _LUT_L["N_h"], _LUT_L["N_p"]
N_P_SAT  = _LUT_L.get("N_p_sat", 150)
GRADIENT_METHOD = _LUT_L["gradient_method"]

if (_LUT_G["N_h"], _LUT_G["N_p"]) != (N_H, N_P):
    raise ValueError(f"{SETTINGS_PATH.name}: the two eos tables must share N_h/N_p")


def lut_liquid() -> dict:
    """FluidBicubic kwargs for the metastable-LIQUID table (dominant phase)."""
    return dict(_LUT_L)


def lut_vapor() -> dict:
    """FluidBicubic kwargs for the metastable-VAPOUR table (ghost phase)."""
    return dict(_LUT_G)


def build_fluids(verbose: bool = True):
    """Build (or load) both FluidBicubic tables.

    jaxprop is imported lazily so this module stays importable from a plain
    CoolProp-only interpreter.
    """
    import jaxprop as jxp

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"EOS tables in: {TABLE_DIR}")
    fluid_l = jxp.FluidBicubic(**lut_liquid())
    fluid_g = jxp.FluidBicubic(**lut_vapor())
    return fluid_g, fluid_l


# ---------------------------------------------------------------------------
# Reservoir (stagnation) state
# ---------------------------------------------------------------------------
_BC = RAW.get("bc", {})

T_G0 = float(_BC["T_g0"])
T_L0 = float(_BC["T_l0"])

_p0_raw = _BC["p0"]
if isinstance(_p0_raw, str):
    if _p0_raw.strip().lower() not in ("saturation", "sat", "p_sat"):
        raise ValueError(f"{SETTINGS_PATH.name}: unknown bc.p0 {_p0_raw!r} "
                         f"(use a number in Pa or 'saturation')")
    P0 = float(CP.PropsSI("P", "T", T_L0, "Q", 0, FLUID_NAME))
else:
    P0 = float(_p0_raw)

# Backwards-compatible alias: several scripts and the summary print T0.
T0 = T_L0
T0_CELSIUS = T_L0 - 273.15

T_SAT_P0 = float(CP.PropsSI("T", "P", P0, "Q", 0, FLUID_NAME))
_T_SAT_P0 = T_SAT_P0
_SAT_TOL_K = 1.0e-3          # within this of T_sat(p0) counts as "on saturation"


def _dT_label(T):
    """'saturated' / 'subcooled by X K' / 'superheated by X K' at p0."""
    d = T - T_SAT_P0
    if abs(d) <= _SAT_TOL_K:
        return "saturated"
    return f"subcooled by {-d:.2f} K" if d < 0 else f"superheated by {d:.2f} K"


SUBCOOLING_L = T_SAT_P0 - T_L0      # >0 = subcooled liquid
SUPERHEAT_G  = T_G0 - T_SAT_P0      # >0 = superheated vapour


def _sat_state(q):
    return [float(CP.PropsSI(k, "P", P0, "Q", q, FLUID_NAME)) for k in ("H", "S", "D")]


def _metastable_liquid(p, T_target):
    """(h, s, rho) of a SUPERHEATED (metastable) liquid at (p, T_target).

    A superheated liquid has no equilibrium state, so CoolProp's ordinary flash
    cannot reach it - it must be found on the metastable continuation.  This
    marches h upward from saturation with jaxprop's metastable evaluator (the
    same one the EOS tables are generated with, so the reservoir is consistent
    with the table the solver reads) until T reaches T_target.

    Raises if the SPINODAL is hit first: past it no liquid state exists at all,
    so the requested superheat is not merely hard to compute, it is impossible.
    """
    import jaxprop as jxp
    import numpy as _np

    fl = jxp.Fluid(FLUID_NAME, BACKEND)
    h = float(CP.PropsSI("H", "P", p, "Q", 0, FLUID_NAME))
    rho = float(CP.PropsSI("D", "P", p, "Q", 0, FLUID_NAME))
    T = T_SAT_P0
    dh = 200.0                      # J/kg per march step
    h_prev, T_prev = h, T

    def _state(h_val, guess):
        return fl.get_state_metastable(
            prop_1="p", prop_1_value=float(p),
            prop_2="h", prop_2_value=float(h_val),
            rhoT_guess=_np.array(guess))

    while T < T_target:
        try:
            st = _state(h + dh, (rho, T))
            if float(st["isothermal_bulk_modulus"]) <= 0.0:
                raise RuntimeError("spinodal")
        except Exception:
            raise ValueError(
                f"{SETTINGS_PATH.name}: a liquid superheated by "
                f"{T_target - T_SAT_P0:.2f} K does not exist at p0 = "
                f"{p/1e5:.4f} bar.\n"
                f"  The SPINODAL is reached at only "
                f"{T - T_SAT_P0:.2f} K of superheat (T/Tc = {T_SAT_P0/T_CRIT:.4f}; "
                f"the metastable region narrows to nothing at the critical point).\n"
                f"  Past the spinodal no liquid state exists, so this reservoir is "
                f"not merely hard to converge - it is unphysical.\n"
                f"  Lower p0 (a colder reservoir has a much wider metastable "
                f"region) or ask for less superheat.") from None
        h_prev, T_prev = h, T
        h, rho, T = h + dh, float(st["density"]), float(st["temperature"])

    # Land exactly on T_target by interpolating in h, then re-evaluate there.
    if T > T_prev:
        h = h_prev + (h - h_prev) * (T_target - T_prev) / (T - T_prev)
    st = _state(h, (rho, T))
    return (float(st["enthalpy"]), float(st["entropy"]), float(st["density"]))


def _reservoir_phase(T, phase):
    """(h, s, rho, branch_label) of one phase at the reservoir.

    A (p,T) flash only defines the requested phase where that phase actually
    EXISTS: liquid for T <= T_sat(p0), vapour for T >= T_sat(p0).  Outside its
    own region the flash silently returns the OTHER branch - for a subcooled
    reservoir it hands back the liquid state for the gas phase, which then goes
    into the solver as the ghost and produces a garbage vapour state.  So the
    branch is selected on existence, not on an enthalpy threshold.

    (main_two_fluid.py guards this with `h_g < vapour table h_min`, which happens
    to work for water - floor 2200 kJ/kg, liquid ~1300 - but NOT here, where the
    vapour floor is 250 kJ/kg and the subcooled liquid sits at ~395.)

    ON the saturation line both branches are valid roots, so the phase is again
    taken from its own saturation call.
    """
    q_self = 0.0 if phase == "liquid" else 1.0
    on_sat = abs(T - T_SAT_P0) <= _SAT_TOL_K
    exists = (T <= T_SAT_P0) if phase == "liquid" else (T >= T_SAT_P0)

    if on_sat:
        return (*_sat_state(q_self),
                f"saturated (Q={q_self:g}) - reservoir is ON the saturation line, "
                f"where a (p,T) flash is ambiguous")
    if not exists:
        if phase == "liquid":
            # A superheated liquid IS a real (metastable) state, and it is the
            # whole point of a flashing model - so take the metastable branch
            # rather than silently substituting saturated liquid.
            return (*_metastable_liquid(P0, T),
                    f"METASTABLE liquid, superheated by {T - T_SAT_P0:.2f} K "
                    f"(no equilibrium state exists; taken on the metastable "
                    f"continuation, consistent with the EOS table)")
        # Ghost vapour: a subcooled vapour would be metastable too, but the
        # established convention here (and in main_two_fluid.py) is to seed the
        # ghost on the saturation line.
        return (*_sat_state(q_self),
                f"saturated (Q=1) - reservoir is subcooled, so no vapour exists "
                f"at (p0,T_g0); ghost taken at saturation")
    vals = [float(CP.PropsSI(k, "P", P0, "T", T, FLUID_NAME)) for k in ("H", "S", "D")]
    label = "subcooled liquid" if phase == "liquid" else "superheated vapour"
    return (*vals, f"(p,T) flash - {label}")


H_L0, S_L0, RHO_L0, RESERVOIR_BRANCH_L = _reservoir_phase(T_L0, "liquid")
H_G0, S_G0, RHO_G0, RESERVOIR_BRANCH_G = _reservoir_phase(T_G0, "gas")

# Belt-and-braces: whatever branch was chosen, each phase's enthalpy must land
# inside its own table or the solver starts from a clamped, meaningless state.
for _nm, _h, _lo, _hi in (("h_l0", H_L0, H_LIQ_MIN, H_LIQ_MAX),
                          ("h_g0", H_G0, H_VAP_MIN, H_VAP_MAX)):
    if not (_lo < _h < _hi):
        raise ValueError(
            f"{SETTINGS_PATH.name}: reservoir {_nm} = {_h/1e3:.2f} kJ/kg is outside "
            f"its table [{_lo/1e3:.0f}, {_hi/1e3:.0f}] kJ/kg.\n"
            f"  Widen the eos bounds and re-run build_tables.py, or move the "
            f"reservoir state.")

P_OUT = float(_BC["p_out"])

# ---------------------------------------------------------------------------
# Grid and geometry
# ---------------------------------------------------------------------------
NX = int(_req("grid", "Nx"))
X1 = float(_req("grid", "x1"))
X2 = float(_req("grid", "x2"))

_GEO = dict(RAW.get("geometry", {}) or {})
GEO_FUNCTION  = str(_GEO.pop("function", "get_pipe"))
GEO_ROUGHNESS = float(_GEO.pop("roughness", 0.0))
GEO_KWARGS    = _GEO                      # anything left is forwarded verbatim

# ---------------------------------------------------------------------------
# Inlet composition
# ---------------------------------------------------------------------------
SLIP      = float(_BC.get("slip", 1.0))
DOM_PHASE = str(_BC.get("dom_phase", "auto")).lower()

# Give EITHER quality (gas MASS fraction) or void_fraction (gas VOLUME
# fraction); the other follows from the slip relation
#     alpha = 1 / (1 + ((1-x)/x) * (rho_g/rho_l) * S)
_q_in = _BC.get("quality", None)
_a_in = _BC.get("void_fraction", None)
if _q_in is not None and _a_in is not None:
    raise ValueError(f"{SETTINGS_PATH.name}: set bc.quality OR bc.void_fraction, "
                     f"not both (they are related by the slip relation)")
if _q_in is not None:
    QUALITY = float(_q_in)
    if not (0.0 < QUALITY < 1.0):
        raise ValueError(f"{SETTINGS_PATH.name}: bc.quality must be in (0,1), "
                         f"got {QUALITY}")
    VOID_FRACTION = 1.0 / (1.0 + (1.0 - QUALITY) / QUALITY
                           * (RHO_G0 / RHO_L0) * SLIP)
    COMPOSITION_SOURCE = "quality given, void_fraction derived"
elif _a_in is not None:
    VOID_FRACTION = float(_a_in)
    if not (0.0 < VOID_FRACTION < 1.0):
        raise ValueError(f"{SETTINGS_PATH.name}: bc.void_fraction must be in "
                         f"(0,1), got {VOID_FRACTION}")
    _beta = (1.0 / VOID_FRACTION - 1.0) / ((RHO_G0 / RHO_L0) * SLIP)
    QUALITY = 1.0 / (1.0 + _beta)
    COMPOSITION_SOURCE = "void_fraction given, quality derived"
else:
    raise ValueError(f"{SETTINGS_PATH.name}: bc.inlet='stagnation' needs "
                     f"bc.void_fraction or bc.quality")

# Dominant phase for the inlet pressure closure.  'auto' is the MASS-fraction
# rule: alpha is volume-weighted, but pressure/entropy dominance is mass-weighted,
# so a thin-but-heavy liquid must still close the pressure.
if DOM_PHASE in ("gas", "liquid"):
    DOM, DOM_SOURCE = DOM_PHASE, f"forced (bc.dom_phase={DOM_PHASE})"
else:
    DOM = "gas" if QUALITY >= 0.5 else "liquid"
    DOM_SOURCE = f"auto: mass fraction x = {QUALITY:.3e}"

# ---------------------------------------------------------------------------
# Interphase closure
# ---------------------------------------------------------------------------
_SRC = RAW.get("sources", {})
BUBBLE_NUMBER  = float(_SRC.get("bubble_number",  1.0e8))
DROPLET_NUMBER = float(_SRC.get("droplet_number", 1.0e8))
MASS_TRANSFER  = bool(_SRC.get("mass_transfer", True))

# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------
_N = RAW["numerics"]
CFL          = float(_N["CFL"])
NEWTON_ITER  = int(_N["newton_iter"])
DELTA_FACTOR = float(_N["delta_factor"])
RECON        = str(_N["recon"])
K_VM         = float(_N["k_vm"])
JACOBIAN     = str(_N.get("jacobian", "dense"))
LINEAR_SOLVE = str(_N.get("linear_solve", "lstsq"))

EOS_ROWS        = bool(_N.get("eos_rows", False))
EOS_ROWS_BANDED = bool(_N.get("eos_rows_banded", False))
JAC_LAG         = int(_N.get("jac_lag", 1))
_ep = _N.get("eos_project", True)
EOS_PROJECT     = _ep if isinstance(_ep, str) else bool(_ep)
PROJ_S_REF      = float(_N.get("proj_s_ref", 2e-2))

NC_FORM  = str(_N.get("nc_form", "cell"))
NC_SCALE = float(_N.get("nc_scale", 1.0))

LOWMACH_DISS    = float(_N.get("lowmach_diss", 0.0))
_lmb = _N.get("lowmach_diss_bc", None)
LOWMACH_DISS_BC = None if _lmb is None else float(_lmb)


def _opt_float(v):
    return None if v in (None, "", "none", "None") else float(v)


EPS_VOID        = float(_N.get("eps_void", 0.0))
EPS_VOID_UCUT   = _opt_float(_N.get("eps_void_ucut"))
EPS_VOID_UWIDTH = _opt_float(_N.get("eps_void_uwidth"))
EPS_VOID_SCALE  = str(_N.get("eps_void_scale", "wood")).strip().lower()

EPS_GHOST        = float(_N.get("eps_ghost", 0.0))
EPS_GHOST_ACUT   = float(_N.get("eps_ghost_acut", 3e-3))
EPS_GHOST_AWIDTH = float(_N.get("eps_ghost_awidth", 0.25))
EPS_GHOST_K2     = float(_N.get("eps_ghost_k2", 0.0))
EPS_GHOST_ALO    = float(_N.get("eps_ghost_alo", 1e-5))

_wc = _N.get("w_char", None)
W_CHAR = ("auto" if isinstance(_wc, str) and _wc == "auto"
          else [float(v) for v in _wc] if _wc is not None else None)

PIN_GHOST          = bool(_N.get("pin_ghost", False))
PIN_GHOST_MOMENTUM = bool(_N.get("pin_ghost_momentum", False))

# alpha_eps: 'match_void' ties the solver clip floor to the inlet void fraction.
# They MUST agree - see the YAML comment for the measured failure when they don't.
_ae = _N.get("alpha_eps", 1e-6)
if isinstance(_ae, str):
    if _ae.strip().lower() != "match_void":
        raise ValueError(f"{SETTINGS_PATH.name}: unknown alpha_eps {_ae!r} "
                         f"(use a number or 'match_void')")
    # 'match_void' only makes sense for a THIN GHOST inlet, where the inlet void
    # fraction genuinely is the floor.  For a real two-phase inlet it would set
    # the clip floor to the bulk void fraction, so alpha could never fall below
    # it ANYWHERE in the nozzle - which is not a floor, it is a straitjacket.
    _MATCH_VOID_MAX = 1.0e-2
    if VOID_FRACTION > _MATCH_VOID_MAX:
        raise ValueError(
            f"{SETTINGS_PATH.name}: numerics.alpha_eps='match_void' with an inlet "
            f"void fraction of {VOID_FRACTION:.4g} would set the solver's clip "
            f"floor to {VOID_FRACTION:.4g}.\n"
            f"  That is a real two-phase inlet, not a thin ghost, so alpha could "
            f"never drop below {VOID_FRACTION:.4g} anywhere in the nozzle.\n"
            f"  Set an explicit small floor instead, e.g. 'alpha_eps: 1.0e-3'.")
    ALPHA_EPS = VOID_FRACTION
else:
    ALPHA_EPS = float(_ae)

SSR_MODE    = str(_N.get("ssr_mode", "all")).strip().lower()
DU_BLEND_LO = float(_N.get("du_blend_lo", 1e-2))
DU_BLEND_HI = float(_N.get("du_blend_hi", 1e-1))

# ---------------------------------------------------------------------------
# Boundary / initial conditions
# ---------------------------------------------------------------------------
BC_INLET    = str(_BC.get("inlet", "transmissive"))
BC_OUTLET   = str(_BC.get("outlet", "transmissive"))
P_OUT_RELAX = float(_BC.get("p_out_relax", 1.0))
IC_TYPE     = str(RAW["ic"]["type"]).strip().lower()

# ---------------------------------------------------------------------------
# Back-pressure continuation
# ---------------------------------------------------------------------------
_CONT = RAW.get("continuation", {}) or {}
SSR_TOL_STAGE = float(_CONT.get("ssr_tol_stage", 1.0e-5))
N_STEPS_STAGE = int(_CONT.get("n_steps_stage", 2000))

# `stages` lists the INTERMEDIATE back-pressures to pass through; the run always
# ends at bc.p_out, which is appended automatically.  So bc.p_out is the single
# place the target is set - changing it never requires editing the stage list.
#   omitted / null / []      -> [p_out]                  (straight to the target)
#   [10e5, 5e5], p_out=2e5   -> [10e5, 5e5, 2e5]
#   [10e5, 2e5], p_out=2e5   -> [10e5, 2e5]              (no duplicate)
_stages_raw = _CONT.get("stages") or []
P_OUT_STAGES = [float(v) for v in _stages_raw]
if not P_OUT_STAGES or abs(P_OUT_STAGES[-1] - P_OUT) > 1e-9 * max(abs(P_OUT), 1.0):
    P_OUT_STAGES.append(P_OUT)

_bad = [q for q in P_OUT_STAGES[:-1] if q <= P_OUT]
if _bad:
    raise ValueError(
        f"{SETTINGS_PATH.name}: continuation.stages are INTERMEDIATE "
        f"back-pressures and must all be above bc.p_out = {P_OUT/1e5:.4g} bar; "
        f"got {[q/1e5 for q in _bad]} bar.  Remove them, or lower bc.p_out.")
if any(b >= a for a, b in zip(P_OUT_STAGES, P_OUT_STAGES[1:])):
    raise ValueError(
        f"{SETTINGS_PATH.name}: continuation.stages must be strictly "
        f"DESCENDING (each stage is warm-started from the previous one); got "
        f"{[q/1e5 for q in P_OUT_STAGES]} bar.")

# ---------------------------------------------------------------------------
# Time-step control
# ---------------------------------------------------------------------------
_TC = RAW.get("time_control", {})
ADAPTIVE_DT  = bool(_TC.get("adaptive_dt", False))
MAX_HALVINGS = int(_TC.get("max_halvings", 8))
DP_MAX       = float(_TC.get("dp_max", 0.20))
DH_MAX       = float(_TC.get("dh_max", 0.10))
DU_MAX       = float(_TC.get("du_max", 0.20))
DA_MAX       = float(_TC.get("da_max", 0.10))
DT_GROW      = float(_TC.get("dt_grow", 1.20))
DT_FLOOR_FRAC     = float(_TC.get("dt_floor_frac", 1.0e-3))
FLOOR_STALL_STEPS = int(_TC.get("floor_stall_steps", 500))

# ---------------------------------------------------------------------------
# Run control
# ---------------------------------------------------------------------------
_R = RAW["run"]
N_STEPS        = int(_R["n_steps"])
PRINT_EVERY    = int(_R["print_every"])
SSR_TOL        = float(_R.get("ssr_tol", 1e-6))
_t_end         = _R.get("t_end")
T_END          = float("inf") if _t_end is None else float(_t_end)
PLATEAU_WINDOW = int(_R.get("plateau_window", 20))
PLATEAU_RATIO  = float(_R.get("plateau_ratio", 1.5))
VISUALIZE_IC   = bool(_R.get("visualize_ic", False))
DEBUG_NEWTON   = bool(_R.get("debug_newton", False))
SRC_SCALE      = float(_R.get("src_scale", 1.0))
AREA_SRC       = bool(_R.get("area_src", True))
DOM_ONLY_DT    = bool(_R.get("dom_only_dt", True))

# ---------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------
_SO = RAW.get("save_options") or {}
REPO_MAIN = str(_SO.get("repo_main", "results")).strip()
REPO_RUN  = str(_SO.get("repo_run", "run")).strip()
RESULTS_DIR = CASE_DIR / REPO_MAIN


def summary() -> str:
    """Human-readable echo of the case definition."""
    return "\n".join([
        "R1233zd(E) flashing expansion - Elliot nozzle, six-equation TFM",
        f"  settings file    {SETTINGS_PATH.name}   (from {SETTINGS_SOURCE})",
        f"  fluid            {FLUID_NAME}  (CoolProp canonical: "
        f"{CP.get_fluid_param_string(FLUID_NAME, 'name')})",
        f"  critical point   p_c = {P_CRIT/1e5:.4f} bar   T_c = {T_CRIT-273.15:.2f} degC",
        f"  reservoir        p0 = {P0/1e5:.4f} bar  (p0/p_c = {P0/P_CRIT:.4f})   "
        f"T_sat(p0) = {T_SAT_P0-273.15:.2f} degC",
        f"                   T_g0 = {T_G0-273.15:.2f} degC ({_dT_label(T_G0)})   "
        f"T_l0 = {T_L0-273.15:.2f} degC ({_dT_label(T_L0)})",
        f"                   liquid: {RESERVOIR_BRANCH_L}",
        f"                     h_l0 = {H_L0/1e3:.3f} kJ/kg   s_l0 = {S_L0/1e3:.5f} kJ/kg/K"
        f"   rho_l0 = {RHO_L0:.3f} kg/m3",
        f"                   vapour: {RESERVOIR_BRANCH_G}",
        f"                     h_g0 = {H_G0/1e3:.3f} kJ/kg   s_g0 = {S_G0/1e3:.5f} kJ/kg/K"
        f"   rho_g0 = {RHO_G0:.3f} kg/m3",
        f"  back-pressure    p_out  = {P_OUT/1e5:.4f} bar   (p0/p_out = {P0/P_OUT:.2f})",
        f"                   relax  = {P_OUT_RELAX:g}   stages = "
        + ", ".join(f"{q/1e5:.2f}" for q in P_OUT_STAGES) + " bar",
        f"  composition      alpha_g = {VOID_FRACTION:.4e}   quality = {QUALITY:.4e}"
        f"   slip = {SLIP:g}",
        f"                   [{COMPOSITION_SOURCE}]   alpha_eps = {ALPHA_EPS:.1e}",
        f"                   dominant phase = {DOM}  [{DOM_SOURCE}]",
        f"  closure          Nb = {BUBBLE_NUMBER:.2e} /m3   mass_transfer = {MASS_TRANSFER}",
        f"  grid             Nx = {NX}   x = [{X1}, {X2}] m   geometry = {GEO_FUNCTION}",
        f"  liquid table     h [{H_LIQ_MIN/1e3:.0f}, {H_LIQ_MAX/1e3:.0f}] kJ/kg   "
        f"p [{P_TABLE_MIN/1e5:.3f}, {P_TABLE_MAX/1e5:.4f}] bar   {N_H}x{N_P}",
        f"  vapour table     h [{H_VAP_MIN/1e3:.0f}, {H_VAP_MAX/1e3:.0f}] kJ/kg   "
        f"p [{P_TABLE_MIN/1e5:.3f}, {P_TABLE_MAX/1e5:.4f}] bar   {N_H}x{N_P}",
    ])


if __name__ == "__main__":
    print(summary())
