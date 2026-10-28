# %%
"""
Two-fluid FVS solver.

To run different cases, create your own settings/settings_your_name.yaml
and set it in SETTINGS_FILE (or pass it via the SETTINGS_YAML environment variable).
A settings file starts from a defaults file (`base: defaults_flashing.yaml`, ...;
see config_loader.py) and lists only what differs.  Every run prints all the
resolved settings and saves them in its folder, with run_info.yaml.

Run from the repository root:
    poetry run python -m development.alberto.two_fluid.main_two_fluid
"""

import os
import sys
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=true")
import jax
jax.config.update("jax_enable_x64", True)

import pathlib
_cache_dir = pathlib.Path(__file__).resolve().parents[3] / ".jax_cache"
_cache_dir.mkdir(exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(_cache_dir))

import time as _time
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jaxprop as jxp
import nozzle_hyperbolic.fvm_solver.geometry as _geom
from nozzle_hyperbolic.fvm_solver.two_fluid import (
    make_two_fluid_step,
    W_from_U_h_batch,
    U_h_from_W_batch,
    phase_props,
    common_quantities,
)

from development.alberto.two_fluid.config_loader import load_config, print_settings, write_resolved_settings
from development.alberto.two_fluid.initial_conditions import build_initial_condition, ic_source_file
from development.alberto.support.run_info import write_run_info
from nozzle_hyperbolic.fvm_solver.two_fluid.sources import (
    two_fluid_closure,
    D_VE_algebric,
    D_VE_algebric_two_component, 
    D_VE_area,
    get_interfacial_area,
)

# Settings file for this test (two-fluid format: gas=Air, liquid=water ghost,
# mirroring the single-phase settings_nozzle_validation.yaml).
# Settings files live in development/alberto/two_fluid/settings/ (paths relative to this module).
# SETTINGS_FILE = "settings/settings_two_fluid_nozzle_validation_two_component_test_static.yaml"
SETTINGS_FILE = "settings/settings_flashing_smd_supersonic_test.yaml"
SETTINGS_FILE = "settings/settings_flashing_nitrogen_simoneau.yaml"
SETTINGS_YAML = os.environ.get("SETTINGS_YAML",SETTINGS_FILE)
# SETTINGS_YAML = os.environ.get("SETTINGS_YAML","settings/settings_two_fluid_liquid_bernoulli_symmetric.yaml")
_here = pathlib.Path(__file__).parent
# The settings file names its defaults with `base:` (config_loader.py).  The
# SRC_SCALE environment variable (debug_ghost_pump convention) overrides
# sources.scale; the override is printed and saved with the run.
_env_over = ({"sources.scale": float(os.environ["SRC_SCALE"])}
             if "SRC_SCALE" in os.environ else None)
cfg   = load_config(_here / SETTINGS_YAML, overrides=_env_over,
                    override_label="env SRC_SCALE")
# Interfacial area law (sources.interfacial_area / interfacial_area_params) — the
# same one two_fluid_closure uses; built here so a bad name fails before the run.
_area_law = get_interfacial_area(cfg.INTERFACIAL_AREA, cfg.INTERFACIAL_AREA_PARAMS)


def _pin_cpus(spec):
    """Restrict this process to the logical CPUs in `spec` ("0-15", "0,2,4-7").
    On hybrid laptops (P+E cores) Windows may otherwise schedule the run on the
    ~2× slower E-cores.  Applied before any JAX computation."""
    cpus = set()
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        cpus.update(range(int(a), int(b or a) + 1))
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        k32.SetProcessAffinityMask.restype = wintypes.BOOL
        ok = k32.SetProcessAffinityMask(k32.GetCurrentProcess(),
                                        sum(1 << c for c in cpus))
        if not ok:
            raise OSError(f"SetProcessAffinityMask failed (error {ctypes.get_last_error()})")
    else:
        os.sched_setaffinity(0, cpus)
    print(f"CPU affinity: pinned to logical CPUs {spec}")


if cfg.CPU_AFFINITY is not None:
    _pin_cpus(cfg.CPU_AFFINITY)

SRC_SCALE = cfg.SRC_SCALE          # sources.scale (env SRC_SCALE overrides)
AREA_SRC  = cfg.AREA_SRC           # sources.area_source
# DOM_ONLY_DT (time_control.dt_check_dominant_only): base the adaptive-dt
#   increment check on the DOMINANT phase only, ignoring the near-massless
#   GHOST's u_l/h_l (which, with no drag, oscillate freely and can trigger
#   halving / dt-collapse even when the dominant phase is fine).
DOM_ONLY_DT = cfg.DOM_ONLY_DT

# ── Keep-marching controller (steady-state mode) ──────────────────────────────
# We care about the STEADY state, so a step that violates the per-step increment
# thresholds (dp/du/dh/da) must NOT kill the run.  Instead of the old hard stop
# at a near-zero dt floor, we clamp dt to a PHYSICAL floor (a fraction of the CFL
# limit) and keep integrating: a stiff transient (flashing front) then marches
# through and dt recovers.  We bail ONLY on (a) NaN/Inf, or (b) a genuine STALL —
# many consecutive floored steps with no SSR progress (a standing checkerboard
# won't be cured by marching, so we stop and report it instead of spinning).
DT_FLOOR_FRAC     = cfg.DT_FLOOR_FRAC      # floor dt = this * CFL dt_cap (per step)
FLOOR_STALL_STEPS = cfg.FLOOR_STALL_STEPS  # consecutive floored steps w/ no SSR drop → bail

# Dominant phase (for plotting / dom-phase diagnostics AND the stagnation-inlet
# pressure closure).  PROVISIONAL here from the volume fraction (for the run_name
# label only); reassigned authoritatively below from the MASS fraction once the
# stagnation densities are known (α is volume-weighted, but pressure/entropy
# dominance is mass-weighted — see bc.dom_phase).  bc.dom_phase='gas'/'liquid'
# forces it; 'auto' (default) uses the mass-fraction rule.
_void_hint = (cfg.BC_VOID if cfg.BC_VOID is not None else
              (cfg.BC_QUALITY if cfg.BC_QUALITY is not None else 0.999999))
if cfg.DOM_PHASE in ("gas", "liquid"):
    dom = cfg.DOM_PHASE
else:
    dom = "gas" if _void_hint >= 0.5 else "liquid"

# ── Output directory ──────────────────────────────────────────────────────────
import shutil
import datetime as _dt

_stamp    = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
# save_options.repo_run = name PREFIX (default legacy 'nozzle_valid_{dom}dom');
# save_options.repo_main = parent folder (default legacy 'results').  The auto
# suffix _N{Nx}_{recon}_{timestamp} is always appended here.
_prefix   = cfg.REPO_RUN  or f"nozzle_valid_{dom}dom"
_repo     = cfg.REPO_MAIN or "results"
run_name  = f"{_prefix}_N{cfg.Nx}_{cfg.RECON}_{_stamp}"
out_dir   = _here.parent / "outputs" / _repo / run_name
out_dir.mkdir(parents=True, exist_ok=True)
# Settings saved with the run: <case name>.yaml = the RESOLVED settings (every
# value, base: none) — rerun with SETTINGS_YAML=<that file>; inputs/ = the case
# file and its defaults chain exactly as written.
_settings_out = write_resolved_settings(
    cfg, out_dir / pathlib.Path(SETTINGS_YAML).name,
    header=[f"run: {run_name}"])
(out_dir / "inputs").mkdir(exist_ok=True)
_inputs_missing = []
for _p in cfg._settings.chain:
    try:
        shutil.copy(_p, out_dir / "inputs" / _p.name)
    except OSError:          # e.g. Windows 260-character path limit in a deep folder
        _inputs_missing.append(_p.name)

# ── Tee terminal output into the run folder (stdout + stderr) ─────────────────
# Everything printed from here on (settings echo, march log, diagnostics,
# tracebacks) is also saved to <out_dir>/run_log.txt for post-mortems.
class _Tee:
    """Write to both a file and the original stream."""
    def __init__(self, stream, filepath):
        self.stream = stream
        self.file = open(filepath, "a", encoding="utf-8")

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()

_log_path = out_dir / "run_log.txt"
sys.stdout = _Tee(sys.stdout, _log_path)
sys.stderr = _Tee(sys.stderr, _log_path)
print(f"Output directory: {out_dir.resolve()}")
print(f"Terminal log:     {_log_path.name}")
print_settings(cfg)
print(f"Resolved settings saved: {_settings_out.name}  (case file + defaults in inputs/)")
if _inputs_missing:
    print(f"  WARNING: could not copy {', '.join(_inputs_missing)} into inputs/ "
          f"(path too long?) — the resolved settings file is complete anyway")

# ── EOS tables ────────────────────────────────────────────────────────────────
print("Building / loading EOS tables...")
_PG_BACKENDS = ("perfect_gas", "perfectgas", "ideal_gas", "idealgas")

def _build_fluid(lut):
    """FluidBicubic from a table LUT, or FluidPerfectGas for an analytic backend.
    A perfect-gas phase is unbounded (no table edge) → an explosion-proof ghost."""
    if str(lut.get("backend", "")).lower() in _PG_BACKENDS:
        fp = jxp.FluidPerfectGas(lut["fluid_name"],
                                 T_ref=lut["T_ref"], p_ref=lut["p_ref"])
        print(f"  {lut['fluid_name']}: PERFECT GAS "
              f"(T_ref={lut['T_ref']:.1f} K, p_ref={lut['p_ref']/1e5:.2f} bar)")
        return fp, True
    return jxp.FluidBicubic(**lut), False

fluid_g, _pg_g = _build_fluid(cfg.LUT_G)
fluid_l, _pg_l = _build_fluid(cfg.LUT_L)

fluid_gas_name = cfg.LUT_G["fluid_name"]
fluid_liquid_name = cfg.LUT_L["fluid_name"]

# Reproducibility record: code version + local changes (git_diff.patch), software
# versions, machine / power source, environment, checksums of the input files.
_inputs = {}
for _ph, _fl in (("eos_table_gas", fluid_g), ("eos_table_liquid", fluid_l)):
    if getattr(_fl, "table_name", None):
        _inputs[_ph] = pathlib.Path(_fl.table_dir) / f"{_fl.table_name}.pkl"
_ic_file = ic_source_file(cfg._settings.raw["ic"], _here.parent)
if _ic_file is not None:
    _inputs["ic_source"] = _ic_file
print(f"Run record: {write_run_info(out_dir, cfg, resolved_file=_settings_out, input_files=_inputs).name}")

# Reservoir-state (PT) evaluator per phase.  For a perfect gas, the phase object
# itself does PT analytically (and self-consistently → its s0 matches the solver's
# gas isentrope); for a real fluid use the FluidJAX inverse call as before.
fluid_g_inverse_call = fluid_g if _pg_g else jxp.FluidJAX(fluid_gas_name)
fluid_l_inverse_call = fluid_l if _pg_l else jxp.FluidJAX(fluid_liquid_name)
print("  Done.")

# ── Grid ─────────────────────────────────────────────────────────────────────
x  = np.linspace(cfg.x1, cfg.x2, cfg.Nx)
dx = float(x[1] - x[0])

_geo_fn = getattr(_geom, cfg.GEO_FUNCTION)
geo     = _geo_fn(x, roughness=cfg.GEO_ROUGHNESS, **cfg.GEO_KWARGS)
A     = np.asarray(geo.A,    dtype=float)
dAdx  = np.array(geo.dAdx, dtype=float)      # copy — asarray gives a read-only view
A_face = 0.5 * (A[:-1] + A[1:])
# ── WELL-BALANCED area gradient (2026-07-13 root-cause fix) ───────────────────
# The flux divergence uses FACE areas, Δ(A_f·F̂); the wall-pressure source used
# the ANALYTIC cell-centred geo.dAdx.  Discretely dAdx_i·dx ≠ (A_{i+1/2}−A_{i-1/2})
# wherever A(x) is curved, so a SMOOTH state can never zero the steady residual —
# the solver absorbs the mismatch as a 2Δx checkerboard in the soft void/slip
# subspace (the persistent throat checkerboard; fixed point, not instability —
# proven by the pipe control + step-map probe, debug_step_map.py).  Using the
# face-consistent gradient makes α·p·ΔA_f cancel the smooth-flow flux divergence
# EXACTLY: checkerboard fixed point 2.44 → 0.09 m/s du-osc (probe-certified).
dAdx[1:-1] = (A_face[1:] - A_face[:-1]) / dx

print(f"Grid: Nx={cfg.Nx}  dx={dx:.4f} m  geometry={cfg.GEO_FUNCTION}  "
      f"A: {A.min():.4e}..{A.max():.4e} m^2")
_G_max = float(np.max(np.abs(dAdx / A)))
print(f"  max|(1/A)dA/dx| = {_G_max:.3e} 1/m"
      + ("   WARNING: ~constant area → area source is a NO-OP; use a nozzle geometry!"
         if _G_max < 1e-6 else "   (nozzle — area source active)"))


# ── Shared state plotter — used for BOTH the initial condition and the final ──
# solution so the IC is drawn "as an initial solution".
def _plot_state(U_h_arr, title, out_png):
    U  = np.asarray(U_h_arr)
    p, ag = U[:, 0], U[:, 3]
    if dom == "gas":
        domname, u_d, h_d, fl = "gas", U[:, 1], U[:, 4], fluid_g
    else:
        domname, u_d, h_d, fl = "liquid", U[:, 2], U[:, 5], fluid_l
    T_d = np.asarray(fl.get_state(
        jxp.HmassP_INPUTS, jnp.array(h_d), jnp.array(p))["temperature"])

    M = np.empty(U.shape[0], dtype=float)
    area_inter = np.empty(U.shape[0], dtype=float)
    for i in range(U.shape[0]):
        p_i = float(U[i, 0])
        ag_i = float(U[i, 3])
        u_g_i = float(U[i, 1])
        u_l_i = float(U[i, 2])
        h_g_i = float(U[i, 4])
        h_l_i = float(U[i, 5])

        eos_g_i = phase_props(fluid_g, p_i, h_g_i)
        eos_l_i = phase_props(fluid_l, p_i, h_l_i)
        cq_i = common_quantities(
            jnp.array([p_i, u_g_i, u_l_i, ag_i, h_g_i, h_l_i]),
            eos_g_i,
            eos_l_i,
            cfg.K_VM,
        )

        rg_i = float(eos_g_i.rho)
        rl_i = float(eos_l_i.rho)
        rh2_i = float(cq_i.rh2)
        rh_i = np.sqrt(rh2_i)
        k_rho_mix_i = float(cq_i.rg_hat - rg_i)
        rho_hat_s_i = float(cq_i.rho_s + k_rho_mix_i)
        a_tilde_sq_i = (rho_hat_s_i / float(cq_i.rho_s)) * (rg_i * rl_i / rh2_i) * float(cq_i.a0sq)
        delta_a_sq_i = (
            ag_i * (1.0 - ag_i) * rg_i * rl_i * float(cq_i.rg_hat) * float(cq_i.rl_hat)
            / (rh2_i**2) * float(cq_i.du) ** 2
        )
        a_sq_i = a_tilde_sq_i - delta_a_sq_i
        a_i = np.sqrt(max(float(a_sq_i), 0.0))

        hat_ag_i = ag_i * rg_i * float(cq_i.rl_hat) / rh2_i
        hat_al_i = (1.0 - ag_i) * rl_i * float(cq_i.rg_hat) / rh2_i
        u_mix_i = hat_ag_i * u_g_i + hat_al_i * u_l_i
        M[i] = u_mix_i / a_i if a_i > 0.0 else np.nan
        # M[i] = a_sq_i

        # Interfacial area calculations
        area_inter[i], _ = _area_law(ag_i, Nb=cfg.BUBBLE_NUMBER, Nd=cfg.DROPLET_NUMBER)

    du = U[:, 1] - U[:, 2]                       # slip velocity u_g - u_l
    fig, ax = plt.subplots(4, 2, figsize=(5, 14), sharex=True)
    ax = ax.flatten()
    fig.suptitle(title, fontsize=9)
    ax[0].plot(x, A, color="black", lw=1.0, marker="o", markersize=2)
    ax[0].set_ylabel(r"Area $A$ [m$^2$]")
    ax[1].plot(x, p / 1e5, color="purple", lw=1.0, marker="o", markersize=3)
    ax[1].set_ylabel("Pressure [bar]")
    ax[2].plot(x, U[:, 1], color="red", lw=1.0, marker="o", markersize=3, label="gas")
    ax[2].plot(x, U[:, 2], color="blue", lw=1.0, marker="o", markersize=3, label="liquid")
    ax[2].set_ylabel(f"{domname} velocity [m/s]")
    # Slip du = u_g - u_l with a zero line: alpha spikes are hypothesised to sit at
    # the du sign-changes (zero-crossings).  A clean 2Δx sawtooth here = odd-even
    # decoupling in the relative-velocity mode.
    ax[3].axhline(0.0, color="grey", lw=0.8, ls="--")
    ax[3].plot(x, du, color="teal", lw=1.0, marker="o", markersize=3)
    ax[3].set_ylabel(r"slip $u_g - u_l$ [m/s]")
    ax[4].plot(x, T_d, color="darkorange", lw=1.0, marker="o", markersize=3)
    ax[4].set_ylabel(f"{domname} T [K]")
    ax[5].plot(x, ag, color="green", lw=1.0, marker="o", markersize=3)
    ax[5].set_ylabel(r"$\alpha_g$")
    ax[6].plot(x, M, color="blue", lw=1.0, marker="o", markersize=3)
    ax[6].set_ylabel("Mach number")
    ax[6].set_xlabel("x [m]")
    ax[7].plot(x, area_inter, color="green", lw=1.0, marker=".", markersize=3)
    ax[7].set_ylabel(r"$a_i$ [1/m]")
    ax[7].set_xlabel(r"$\alpha_g$ [-]")
    ax[2].legend()
    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {out_png}")
    return fig

# ── Stagnation (reservoir) inlet state from the YAML bc block ──────────────────
P0   = cfg.BC_P0   if cfg.BC_P0   is not None else cfg.P_INIT
if cfg.BC_T_G0 is None:
    raise ValueError("bc.T_g0 (gas stagnation temperature) is required")
T_G0 = cfg.BC_T_G0
T_L0 = cfg.BC_T_L0 if cfg.BC_T_L0 is not None else T_G0
SLIP = cfg.BC_SLIP


_stl0 = fluid_l_inverse_call.get_state(jxp.PT_INPUTS, jnp.array([P0]), jnp.array([T_L0]))
_stg0 = fluid_g_inverse_call.get_state(jxp.PT_INPUTS, jnp.array([P0]), jnp.array([T_G0]))
# _stl0 = _stg0

H_g0   = float(_stg0["enthalpy"][0])          # gas stagnation enthalpy (u=0)
s_g0   = float(_stg0["s"][0])                 # gas stagnation entropy
H_l0   = float(_stl0["enthalpy"][0])          # liquid stagnation enthalpy
s_l0   = float(_stl0["s"][0])                 # liquid stagnation entropy
rho_g0 = float(_stg0["density"][0]);  rho_l0 = float(_stl0["density"][0])

# ── Ghost-vapor reservoir fix (FLASHING) ──────────────────────────────────────
# For a SUBCOOLED reservoir (flashing: T0 < T_sat(p0)) the equilibrium PT call
# returns the LIQUID branch for the GAS phase too — so H_g0 comes back as the
# liquid enthalpy, BELOW the metastable-vapor table floor (h_min).  h_g then
# clamps to the floor → garbage vapor state → negative sound speed → NaN.
# The nozzlex demo takes the ghost vapor at SATURATION at p0 instead
# (fluid_vapor.get_state_saturation(p0)); do the same when the PT gas state
# falls outside the vapor table.  Gas-DOMINANT cases (superheated T_g0) give
# H_g0 above h_min → untouched.
#
# SUPERCRITICAL reservoir (p0 above the saturation table, which ends at
# 0.98 p_crit): the PT gas state is the same supercritical fluid as the liquid,
# while the ghost pin drives h_g to h_sat_g(p) — clamped at the table top — in
# every cell.  So a GHOST vapor (liquid-dominant flashing) takes the same
# saturated-vapor state at the inlet: get_state_saturation(p0) returns the
# values at the table top, consistent with the pin.
_hg_min = float(getattr(fluid_g, "h_min", -np.inf))
_lp_sat = getattr(fluid_g, "logP_sat_vals", None)
_p_sat_top = float(np.exp(_lp_sat[-1])) if _lp_sat is not None else np.inf
_ghost_supercrit = (cfg.MASS_TRANSFER and dom == "liquid" and P0 >= _p_sat_top)
if (H_g0 < _hg_min or _ghost_supercrit) and not _pg_g:
    # get_state_saturation returns CANONICAL names only (entropy, not the 's' alias).
    _satg0 = fluid_g.get_state_saturation(jnp.array([P0]))
    H_g0   = float(_satg0["enthalpy"][0])
    s_g0   = float(_satg0["entropy"][0])
    rho_g0 = float(_satg0["density"][0])
    if _ghost_supercrit:
        _why = (f"p0={P0/1e5:.2f} bar is above the saturation table "
                f"(top {_p_sat_top/1e5:.2f} bar = 0.98 p_crit)")
        _at = f"the table top ({_p_sat_top/1e5:.2f} bar)"
    else:
        _why = (f"PT gas state h={float(_stg0['enthalpy'][0])/1e3:.1f} kJ/kg < vapor "
                f"table floor {_hg_min/1e3:.1f} kJ/kg (subcooled reservoir)")
        _at = "p0"
    print(f"  [ghost-vapor fix] {_why}\n"
          f"                    → using SATURATED VAPOR at {_at}: H_g0={H_g0/1e3:.1f} kJ/kg "
          f"s_g0={s_g0:.2f} J/kg/K  rho_g0={rho_g0:.1f} kg/m3")

# Composition: void_fraction OR quality (derive the other via the slip relation
#   α = 1/(1 + ((1−x)/x)(ρ_g/ρ_l)S) ).
# TODO: add the possiblity to have both quality and void fracction specified. THe other quantities are derived from these
if cfg.BC_QUALITY is not None:
    QUALITY = cfg.BC_QUALITY
    ALPHA_G = 1.0 / (1.0 + (1.0 - QUALITY) / QUALITY * (rho_g0 / rho_l0) * SLIP)
elif cfg.BC_VOID is not None:
    ALPHA_G = cfg.BC_VOID
    _beta   = (1.0 / ALPHA_G - 1.0) / ((rho_g0 / rho_l0) * SLIP)
    QUALITY = 1.0 / (1.0 + _beta)
else:
    raise ValueError("bc.inlet='stagnation' needs bc.void_fraction or bc.quality")

# Authoritative dominant phase for the pressure closure: MASS fraction.
if cfg.DOM_PHASE in ("gas", "liquid"):
    dom = cfg.DOM_PHASE
    _dom_src = f"forced (bc.dom_phase={cfg.DOM_PHASE})"
else:
    dom = "gas" if QUALITY >= 0.5 else "liquid"
    _dom_src = f"auto: mass fraction x={QUALITY:.3e}"

print(f"\nStagnation inlet:  p0={P0/1e5:.2f} bar  T_g0={T_G0:.1f}  T_l0={T_L0:.1f} K")
print(f"  H_g0={H_g0/1e3:.1f} kJ/kg  s_g0={s_g0:.2f}  H_l0={H_l0/1e3:.1f} kJ/kg  s_l0={s_l0:.2f} J/kg/K")
print(f"  alpha_g={ALPHA_G:.6f}  quality={QUALITY:.6f}  slip={SLIP:.3f}")
print(f"  dominant phase = {dom}  [{_dom_src}]  -> p closed on {dom} isentrope")
print(f"  SRC_SCALE={SRC_SCALE} (physical sources)   AREA_SRC={AREA_SRC} (area term under test)")

# ── Phase-change pressure cut-off (sources.p_phase_change_max) ─────────────────
# No phase change at p >= P_MT_MAX (fraction of p_crit → Pa).  Passed to the
# closure (σ, interphase heat) and to the solver (ghost pin) — same value.
# Check: the liquid can only superheat where h_sat_l(p) < H_l0 (adiabatic flow,
# h_l <= H_l0), so above the highest such p no phase change is possible anyway;
# a cut-off below it would suppress real flashing.
P_MT_MAX = None
if cfg.P_PHASE_CHANGE_MAX is not None and cfg.MASS_TRANSFER:
    _cp = getattr(fluid_l, "critical_point", None)
    if _cp is not None:
        _p_crit = float(_cp["pressure"])
    else:
        import CoolProp.CoolProp as _CP
        _p_crit = float(_CP.PropsSI("pcrit", fluid_liquid_name))
    P_MT_MAX = cfg.P_PHASE_CHANGE_MAX * _p_crit
    print(f"  phase-change cut-off: no phase change at p >= {P_MT_MAX/1e5:.2f} bar "
          f"({cfg.P_PHASE_CHANGE_MAX:g} p_crit, p_crit = {_p_crit/1e5:.2f} bar)")
    if dom == "liquid" and _lp_sat is not None:
        _pg = np.exp(np.linspace(float(_lp_sat[0]), float(_lp_sat[-1]), 2000))
        _hsl = np.asarray(fluid_l.get_state_saturation(jnp.asarray(_pg))["enthalpy"])
        _ok = np.flatnonzero(_hsl < H_l0)
        if _ok.size:
            _p_on = float(_pg[_ok[-1]])
            print(f"    flashing possible only below {_p_on/1e5:.2f} bar "
                  f"(h_sat_l = H_l0 = {H_l0/1e3:.1f} kJ/kg)"
                  + ("" if _p_on < P_MT_MAX else
                     "   WARNING: cut-off is BELOW this - it suppresses real flashing"))

# ── Reservoir isentrope for the stagnation-inlet pressure closure ─────────────
# The solver needs h_dom(p) ON the dominant-phase isentrope s_dom(p,h)=s_dom0.
# The bicubic table's reverse (p,s)→h root-find DIVERGES for the LIQUID isentrope
# (at low p, s_l0 lands in the two-phase region → no metastable-liquid root).  So
# we build the table here with the robust FluidJAX/CoolProp reverse call (finite,
# monotone h(p) everywhere) and pass the arrays to make_two_fluid_step.  Same p
# grid the solver would use (fluid_g table bounds) so the interp is consistent.
_dom_inv = fluid_g_inverse_call if dom == "gas" else fluid_l_inverse_call
_s_dom0  = s_g0 if dom == "gas" else s_l0
_iso_p   = np.linspace(float(getattr(fluid_g, "p_min", 1e4)) * 1.01,
                       float(getattr(fluid_g, "p_max", 300e5)) * 0.99, 400)
_iso_h   = np.asarray(_dom_inv.get_state(
    jxp.PSmass_INPUTS, jnp.asarray(_iso_p),
    jnp.full(_iso_p.size, jnp.float64(_s_dom0)))["h"])
print(f"  isentrope (FluidJAX, {dom}): h_iso {_iso_h.min()/1e3:.1f}..{_iso_h.max()/1e3:.1f} "
      f"kJ/kg over p {_iso_p.min()/1e5:.2f}..{_iso_p.max()/1e5:.1f} bar  "
      f"(monotone={bool(np.all(np.diff(_iso_h) > 0))})")

# ── Initial condition (derived from stagnation) ───────────────────────────────
# Expand the gas ISENTROPICALLY from stagnation to an inlet-static guess (≈ p_out
# for subsonic), giving the inlet static enthalpy/velocity; then nozzle_simple
# continuity  u = mdot/(ρA)  (u peaks at the throat), enthalpy from H0 = h+u²/2,
# incompressible-Bernoulli pressure dip.  Ghost phase co-flows via the slip ratio.
_p_in    = float(P0 * 0.99)  #  float(min(cfg.P_OUT, P0 * 0.999))
_stg_in  = fluid_g.get_state(jxp.PSmass_INPUTS, jnp.array([_p_in]), jnp.array([s_g0]))
h_g_0    = float(_stg_in["enthalpy"][0])
rho_g_in = float(_stg_in["density"][0])
_a_in    = float(_stg_in["speed_of_sound"][0])
u_g_in   = float(np.sqrt(max(2.0 * (H_g0 - h_g_0), 0.0)))
u_l_in   = u_g_in / SLIP
h_l_0    = H_l0 - 0.5 * u_l_in**2
print(f"  IC inlet static: p={_p_in/1e5:.2f} bar  h_g={h_g_0/1e3:.1f} kJ/kg  "
      f"u_g={u_g_in:.1f} m/s  a={_a_in:.1f}  (M_in={u_g_in/_a_in:.3f})")

_ic_block = cfg._settings.raw["ic"]
IC_MODE = _ic_block["type"]
# All initial conditions live in initial_conditions.py (table of ic.type values at
# the top of that file).
U_h_0 = build_initial_condition(
    _ic_block, x=x, A=A, p0=P0, T_g0=T_G0, T_l0=T_L0,
    alpha_g=ALPHA_G, slip=SLIP, H_g0=H_g0, H_l0=H_l0, p_out=cfg.P_OUT,
    fluid_g=fluid_g, fluid_l=fluid_l, dom=dom,
    inlet_static=(_p_in, u_g_in, u_l_in, h_g_0, h_l_0), base_dir=_here.parent,
)

# ── Visualize the initial condition (cfg.VISUALIZE_IC / run.visualize_ic) ──────
if getattr(cfg, "VISUALIZE_IC", False):
    _Tg_ic = np.asarray(fluid_g.get_state(
        jxp.HmassP_INPUTS, jnp.array(U_h_0[:, 4]), jnp.array(U_h_0[:, 0]))["temperature"])
    print("\n── Initial condition (per-field range) ────────────────────────────")
    print(f"  p     [bar]:  {U_h_0[:,0].min()/1e5:.3f} .. {U_h_0[:,0].max()/1e5:.3f}")
    print(f"  u_g   [m/s]:  {U_h_0[:,1].min():.2f} .. {U_h_0[:,1].max():.2f}")
    print(f"  u_l   [m/s]:  {U_h_0[:,2].min():.2f} .. {U_h_0[:,2].max():.2f}")
    print(f"  alpha_g    :  {U_h_0[:,3].min():.6f} .. {U_h_0[:,3].max():.6f}")
    print(f"  h_g [kJ/kg]:  {U_h_0[:,4].min()/1e3:.2f} .. {U_h_0[:,4].max()/1e3:.2f}")
    print(f"  h_l [kJ/kg]:  {U_h_0[:,5].min()/1e3:.2f} .. {U_h_0[:,5].max()/1e3:.2f}")
    print(f"  T_g   [K]  :  {_Tg_ic.min():.1f} .. {_Tg_ic.max():.1f}")
    # Draw the IC in the SAME format as the final solution (it IS an initial solution).
    _plot_state(U_h_0, f"INITIAL condition ({IC_MODE}, dominant={dom})  "
                       f"{cfg.GEO_FUNCTION}  N={cfg.Nx}",
                out_dir / "initial_condition.png")
    # Show the IC BEFORE solving and block — close the window to start the sim.
    print("  >> Close the initial-condition window to start the simulation...")
    plt.show()

W_0 = W_from_U_h_batch(jnp.array(U_h_0), fluid_g, fluid_l)
print(f"W_0 shape: {W_0.shape}   dtype: {W_0.dtype}")

U_h_check = np.array(U_h_from_W_batch(W_0))
max_err = np.max(np.abs(U_h_check - U_h_0) / (np.abs(U_h_0) + 1e-300))
print(f"Round-trip error  U_h -> W -> U_h:  max relative = {max_err:.2e}  (expected ~1e-15)")

# ── Source-term closure — area source ONLY (physical sources scaled by SRC_SCALE)
from typing import NamedTuple
class _AreaClosure(NamedTuple):
    """Minimal closure carrying only the log-area gradient for D_VE_area."""
    dlnA_dx: jnp.ndarray


def _sources_fn(u_h, eos_g, eos_l, cq, dlnA, sat_g, sat_l):
    D = jnp.zeros(6)
    if SRC_SCALE != 0.0:
        # One closure, flag-aware: two_fluid_closure computes drag + heat (+ mass
        # transfer only if cfg.MASS_TRANSFER).  Then pick the matching ALGEBRAIC
        # assembly — full (with mass rows) vs two-component (drag+heat, no mass).
        closure = two_fluid_closure(
            u_h, eos_g, eos_l, fluid_g, fluid_l,
            T_sat   = sat_g["temperature"],
            h_sat_g = sat_g["enthalpy"],
            h_sat_l = sat_l["enthalpy"],
            dlnA_dx = dlnA,
            Nb      = cfg.BUBBLE_NUMBER,
            Nd      = cfg.DROPLET_NUMBER,
            mass_transfer = cfg.MASS_TRANSFER,
            p_mt_max = P_MT_MAX,
            regime_htc = cfg.REGIME_HTC,
            area_law = cfg.INTERFACIAL_AREA,
            area_params = cfg.INTERFACIAL_AREA_PARAMS,
        )
        if cfg.MASS_TRANSFER:
            D = D + SRC_SCALE * D_VE_algebric(u_h, cq, closure, eos_g, eos_l, cfg.K_VM)
        else:
            D = D + SRC_SCALE * D_VE_algebric_two_component(u_h, cq, closure, eos_g, eos_l, cfg.K_VM)
    if AREA_SRC:
        # D_VE_area only needs dlnA_dx from the closure — build a minimal one.
        D = D + D_VE_area(u_h, cq, _AreaClosure(dlnA_dx=dlnA), eos_g, eos_l)
    return D

# ── Inlet BC conditions ───────────────────────────────────────────────────────
# H_g0, s_g0, H_l0, QUALITY, ALPHA_G, SLIP all computed above from the stagnation
# state.  U_h_inlet (used by the legacy subsonic_inflow/static_state inlets) carries
# the IC inlet-static estimate; the 'stagnation' inlet ignores it.
alpha_g0  = ALPHA_G
U_h_inlet = jnp.array([_p_in, u_g_in, u_l_in, ALPHA_G, h_g_0, h_l_0])
print(f"\nInlet BC: {cfg.BC_INLET}   Outlet BC: {cfg.BC_OUTLET}"
      + (f"  (p_out={cfg.P_OUT/1e5:.2f} bar)"
         if cfg.BC_OUTLET in ("pressure", "characteristic", "characteristic_old") else "")
      + (f"  H_g0={H_g0/1e3:.1f} kJ/kg  s_g0={s_g0:.1f}  quality={QUALITY:.6f}  slip={SLIP:.3f}"
         if cfg.BC_INLET == "stagnation" else ""))

PROJECT = cfg.EOS_PROJECT          # from YAML numerics.eos_project (default True)
MONITOR = False
print(f"[Diagnostic]  PROJECT={PROJECT}  MONITOR={MONITOR}  alpha_eps={cfg.ALPHA_EPS:.1e}"
      + f"  pin_ghost={cfg.PIN_GHOST}  pin_ghost_mom={cfg.PIN_GHOST_MOMENTUM}"
      + (f"  w_char={cfg.W_CHAR}" if cfg.W_CHAR is not None else ""))

# ── Solver factory ────────────────────────────────────────────────────────────
print(f"[Speed-up]  fast_wb_flux={cfg.FAST_WB_FLUX}  flux_single_branch={cfg.FLUX_SINGLE_BRANCH}"
      f"  flux_local_jac={cfg.FLUX_LOCAL_JAC}  cell_local_jac={cfg.CELL_LOCAL_JAC}"
      f"  thomas={cfg.THOMAS}  newton_tol={cfg.NEWTON_TOL:g}"
      f"  ser_beta={cfg.SER_BETA:g}" + (f" (CFL<={cfg.SER_CFL_MAX:g})" if cfg.SER_BETA > 0 else "")
      + f"  fast_increments={cfg.FAST_INCREMENTS}  cpu_affinity={cfg.CPU_AFFINITY}")
print(f"Building solver (first JIT compilation may take a minute)... jacobian={cfg.JACOBIAN}")
_ret = make_two_fluid_step(
    fluid_g, fluid_l,
    A, A_face, dAdx, dx,
    CFL           = cfg.CFL,
    k             = cfg.K_VM,
    delta_factor  = cfg.DELTA_FACTOR,
    newton_iter   = cfg.NEWTON_ITER,
    recon         = cfg.RECON,
    va_eps_mode   = cfg.VA_EPS_MODE,
    va_rel        = cfg.VA_REL,
    va_r0         = cfg.VA_R0,
    va_rm         = cfg.VA_RM,
    va_mean_stop_gradient = cfg.VA_MEAN_STOP_GRADIENT,
    sources_fn    = _sources_fn,
    lowmach_diss  = cfg.LOWMACH_DISS,
    lowmach_diss_bc = cfg.LOWMACH_DISS_BC,
    nc_scale      = cfg.NC_SCALE,
    eps_void      = cfg.EPS_VOID,
    eps_void_ucut   = cfg.EPS_VOID_UCUT,
    eps_void_uwidth = cfg.EPS_VOID_UWIDTH,
    eps_void_scale  = cfg.EPS_VOID_SCALE,
    eps_void_mass_acut     = cfg.EPS_VOID_MASS_ACUT,
    ghost_mass_upwind_acut = cfg.GHOST_MASS_UPWIND_ACUT,
    eps_ghost        = cfg.EPS_GHOST,
    eps_ghost_acut   = cfg.EPS_GHOST_ACUT,
    eps_ghost_awidth = cfg.EPS_GHOST_AWIDTH,
    eps_ghost_k2     = cfg.EPS_GHOST_K2,
    eps_ghost_alo    = cfg.EPS_GHOST_ALO,
    ssr_mode         = cfg.SSR_MODE,
    du_blend_lo     = cfg.DU_BLEND_LO,
    du_blend_hi     = cfg.DU_BLEND_HI,
    debug_newton  = cfg.DEBUG_NEWTON,
    jacobian_mode = cfg.JACOBIAN,
    linear_solve  = cfg.LINEAR_SOLVE,
    bc_inlet      = cfg.BC_INLET,
    bc_outlet     = cfg.BC_OUTLET,
    H_g0          = H_g0,
    s_g0          = s_g0,
    H_l0          = H_l0,
    s_l0          = s_l0,
    dom_phase     = dom,
    iso_h         = _iso_h,
    iso_p         = _iso_p,
    alpha_g0      = alpha_g0,
    quality       = QUALITY,
    slip          = SLIP,
    U_h_inlet     = U_h_inlet,
    p_out         = cfg.P_OUT,
    p_out_relax   = cfg.P_OUT_RELAX,
    eos_project   = PROJECT,
    proj_s_ref    = cfg.PROJ_S_REF,
    eos_rows      = cfg.EOS_ROWS,
    eos_rows_banded = cfg.EOS_ROWS_BANDED,
    jac_lag       = cfg.JAC_LAG,
    nc_form       = cfg.NC_FORM,
    use_saturation = cfg.MASS_TRANSFER,    # only the mass-transfer closure reads
                                           # saturation; two-component/area-only skip it
    w_char        = cfg.W_CHAR,
    alpha_eps     = cfg.ALPHA_EPS,
    pin_ghost     = cfg.PIN_GHOST,
    p_mt_max      = P_MT_MAX,
    pin_ghost_momentum = cfg.PIN_GHOST_MOMENTUM,
    pin_mom_alpha      = cfg.PIN_MOM_ALPHA,
    pin_mom_tau        = cfg.PIN_MOM_TAU,
    fast_wb_flux       = cfg.FAST_WB_FLUX,
    flux_single_branch = cfg.FLUX_SINGLE_BRANCH,
    flux_local_jac     = cfg.FLUX_LOCAL_JAC,
    cell_local_jac     = cfg.CELL_LOCAL_JAC,
    newton_tol         = cfg.NEWTON_TOL,
    thomas             = cfg.THOMAS,
    return_debug  = False,
)
step, compute_dt = _ret

# ── Ramped start-up (bc.p_out_ramp_steps > 0) ─────────────────────────────────
# The outlet back-pressure is passed to step() at RUNTIME and lowered linearly
# from _ramp_from to cfg.P_OUT over the first _ramp_n steps; afterwards it stays
# at cfg.P_OUT (still passed at runtime → one compilation for the whole run).
# Off (0) → step(W, dt) exactly as before.
_ramp_n    = cfg.P_OUT_RAMP_STEPS
_ramp_from = (cfg.P_OUT_RAMP_FROM if cfg.P_OUT_RAMP_FROM is not None
              else float(U_h_0[-2, 0]))
_p_out_now = _ramp_from if _ramp_n > 0 else cfg.P_OUT
if _ramp_n > 0:
    if cfg.P_OUT_RAMP_SHAPE not in ("linear", "log"):
        raise ValueError(f"bc.p_out_ramp_shape must be 'linear' or 'log', got {cfg.P_OUT_RAMP_SHAPE!r}")
    print(f"Ramped start-up ({cfg.P_OUT_RAMP_SHAPE}): p_out {_ramp_from/1e5:.3f} → "
          f"{cfg.P_OUT/1e5:.3f} bar over {_ramp_n} steps "
          f"(convergence checks suspended until the ramp ends)")


def _step(W, dt_):
    if _ramp_n > 0:
        return step(W, dt_, jnp.float64(_p_out_now))
    return step(W, dt_)


print("  Warming up compute_dt / step (triggers full compilation)...")
dt0 = float(compute_dt(W_0))
W_test, ssr_test = _step(W_0, jnp.float64(dt0))
print(f"  dt = {dt0:.4e} s   SSR after first step: {float(ssr_test):.4e}   Compilation done.")

# ── Diagnostic monitor ────────────────────────────────────────────────────────
# A perfect-gas LUT carries no h/p bounds (analytic, unbounded) → use ±inf so the
# monitor's out-of-range flag never trips for that phase.
_HG_LO, _HG_HI = cfg.LUT_G.get("h_min", -np.inf), cfg.LUT_G.get("h_max", np.inf)
_HL_LO, _HL_HI = cfg.LUT_L.get("h_min", -np.inf), cfg.LUT_L.get("h_max", np.inf)
_P_LO,  _P_HI  = cfg.LUT_G.get("p_min", -np.inf), cfg.LUT_G.get("p_max", np.inf)

def _monitor(W, n):
    U  = np.array(U_h_from_W_batch(W))
    p, hg, hl, ag = U[:, 0], U[:, 4], U[:, 5], U[:, 3]
    Wn = np.array(W); mg = Wn[:, 2]
    rg = np.array(fluid_g.get_state(jxp.HmassP_INPUTS, jnp.array(hg), jnp.array(p))["density"])
    res = np.abs(mg - ag * rg) / (np.abs(mg) + 1e-300)
    def _f(v, lo, hi): return "!" if (v.min() <= lo or v.max() >= hi) else " "
    print(f"  [mon] n={n:5d}  "
          f"p[{p.min():.3e},{p.max():.3e}]{_f(p,_P_LO,_P_HI)} "
          f"ug[{U[:,1].min():.2f},{U[:,1].max():.2f}] "
          f"ag[{ag.min():.5f},{ag.max():.5f}]  relEOS={res.max():.2e}", flush=True)

# ── Time-stepping loop (adaptive dt, mirrors main_phase_change.py) ─────────────
W_n       = W_0
t_phys    = 0.0
n         = 0
ssr       = float("inf")
_win      = cfg.PLATEAU_WINDOW
_ssr_hist = [float("inf")] * _win
_stop_reason = f"max steps ({cfg.N_STEPS}) reached"

_adaptive = cfg.ADAPTIVE_DT
dt        = float(compute_dt(W_0))
_floor_streak  = 0                 # consecutive floored (over-threshold) steps
_ssr_at_streak = float("inf")      # SSR when the current floor streak began

print(f"\nRunning (max {cfg.N_STEPS} steps  SSR_tol={cfg.SSR_TOL:.1e})")
print(f"{'Step':>6}  {'t [s]':>12}  {'dt [s]':>12}  {'SSR':>12}")
print("-" * 50)


@jax.jit
def _increments_core(W_old, W_new, a_char):
    """Jitted twin of the eager checks below (run.fast_increments): same
    per-field relative increments, one dispatch instead of ~40 eager ops."""
    Uo = U_h_from_W_batch(W_old)[1:-1]
    Un = U_h_from_W_batch(W_new)[1:-1]
    bad = jnp.any(jnp.isnan(Un)) | jnp.any(jnp.isinf(Un))
    den = jnp.stack([jnp.abs(Uo[:, 0]) + 1e-10,
                     jnp.abs(Uo[:, 1]) + a_char + 1e-10,
                     jnp.abs(Uo[:, 2]) + a_char + 1e-10,
                     jnp.ones_like(Uo[:, 3]),
                     jnp.abs(Uo[:, 4]) + 1e-10,
                     jnp.abs(Uo[:, 5]) + 1e-10], axis=1)
    return bad, jnp.max(jnp.abs(Un - Uo) / den, axis=0)


def _increments(W_old, W_new, a_char):
    if cfg.FAST_INCREMENTS:
        _bad, _d = jax.device_get(_increments_core(W_old, W_new, a_char))
        bad = bool(_bad)
        dp, dug, dul, dag, dhg, dhl = (float(v) for v in _d)
    else:
        Uo = U_h_from_W_batch(W_old)[1:-1]
        Un = U_h_from_W_batch(W_new)[1:-1]
        bad = bool(jnp.any(jnp.isnan(Un)) | jnp.any(jnp.isinf(Un)))
        dp  = float(jnp.max(jnp.abs(Un[:, 0] - Uo[:, 0]) / (jnp.abs(Uo[:, 0]) + 1e-10)))
        dug = float(jnp.max(jnp.abs(Un[:, 1] - Uo[:, 1]) / (jnp.abs(Uo[:, 1]) + a_char + 1e-10)))
        dul = float(jnp.max(jnp.abs(Un[:, 2] - Uo[:, 2]) / (jnp.abs(Uo[:, 2]) + a_char + 1e-10)))
        dag = float(jnp.max(jnp.abs(Un[:, 3] - Uo[:, 3])))
        dhg = float(jnp.max(jnp.abs(Un[:, 4] - Uo[:, 4]) / (jnp.abs(Uo[:, 4]) + 1e-10)))
        dhl = float(jnp.max(jnp.abs(Un[:, 5] - Uo[:, 5]) / (jnp.abs(Uo[:, 5]) + 1e-10)))
    # DOM_ONLY_DT: gate dt on the DOMINANT phase only (exclude the ghost's u/h).
    if DOM_ONLY_DT:
        du = dug if dom == "gas" else dul
        dh = dhg if dom == "gas" else dhl
    else:
        du = max(dug, dul);  dh = max(dhg, dhl)
    return bad, dp, du, dh, dag, (dug, dul, dhg, dhl)


_ser_ref  = None                   # SER: SSR after the first step
_cfl_mult = 1.0                    # SER: current CFL multiplier (1 = plain CFL)

_t0 = _time.perf_counter()
try:
    while n < cfg.N_STEPS:
        halvings = 0
        _ramping = n < _ramp_n                     # still opening the outlet valve
        if _ramp_n > 0:
            _frac = min(1.0, (n + 1) / _ramp_n)
            if cfg.P_OUT_RAMP_SHAPE == "log":
                _p_out_now = _ramp_from * (cfg.P_OUT / _ramp_from) ** _frac
            else:
                _p_out_now = _ramp_from + (cfg.P_OUT - _ramp_from) * _frac
        if _adaptive:
            dt_cap   = float(compute_dt(W_n))
            a_char   = cfg.CFL * dx / max(dt_cap, 1e-300)
            dt_floor = DT_FLOOR_FRAC * dt_cap            # physical floor (per step)
            # ── SER CFL ramp (time_control.ser_beta > 0) ─────────────────────
            # Grow the CFL cap as the SSR falls (pseudo-transient continuation);
            # dt still grows at most dt_grow per step and the increment checks /
            # halvings below still apply.  a_char and dt_floor keep the plain CFL.
            if cfg.SER_BETA > 0.0 and np.isfinite(ssr) and not _ramping:
                if _ser_ref is None:
                    _ser_ref = ssr
                _cfl_mult = min(cfg.SER_CFL_MAX / cfg.CFL,
                                max(1.0, (_ser_ref / max(ssr, 1e-300)) ** cfg.SER_BETA))
                dt_cap   = dt_cap * _cfl_mult
            dt       = max(min(dt * cfg.DT_GROW, dt_cap), dt_floor)
            _per = (0, 0, 0, 0)
            bad = ok = floored = False
            for halvings in range(cfg.MAX_HALVINGS + 1):
                W_try, ssr_j = _step(W_n, jnp.float64(dt))
                jax.block_until_ready(W_try)
                bad, dp, du, dh, da, _per = _increments(W_n, W_try, a_char)
                ok = (not bad) and dp < cfg.DP_MAX and dh < cfg.DH_MAX \
                    and du < cfg.DU_MAX and da < cfg.DA_MAX
                if ok:
                    break
                # Halve toward the floor.  At the floor (or out of halvings) stop:
                # NaN → bail below; finite-but-over-threshold → accept & keep going.
                if dt <= dt_floor * (1.0 + 1e-9) or halvings >= cfg.MAX_HALVINGS:
                    floored = True
                    break
                dt = max(dt * 0.5, dt_floor)
            # NaN/Inf is never a usable state — bail (marching can't fix it).
            if bad:
                _stop_reason = f"NaN/Inf in state at step {n + 1} (dt={dt:.2e})"
                break
            # Track the floor streak: consecutive floored (over-threshold) steps.
            if floored and not ok:
                if _floor_streak == 0:
                    _ssr_at_streak = ssr
                _floor_streak += 1
            else:
                _floor_streak = 0
            # When halving/flooring, show which field/phase blew the threshold —
            # dug/dhg = dominant gas, dul/dhl = ghost.  A FLOORED tag means the step
            # was accepted over-threshold (steady-state keep-marching).
            if halvings > 0 or floored:
                _dug, _dul, _dhg, _dhl = _per
                _tag = f"FLOORED x{_floor_streak}" if floored else f"halve x{halvings}"
                print(f"        [{_tag}] dp={dp:.2f} da={da:.3f} | "
                      f"dug={_dug:.2f} dul={_dul:.2f}  dhg={_dhg:.3f} dhl={_dhl:.3f}"
                      f"  ({'GHOST-gated' if max(_dul,_dhl) > max(_dug,_dhg) else 'gas-gated'})",
                      flush=True)
            W_n = W_try
            ssr = (max(float(ssr_j), cfg.SSR_TOL * 10)
                   if floored else float(ssr_j))
            # Bail ONLY on a genuine stall: a long floor streak with no SSR progress
            # (a standing checkerboard — marching won't cure it, so stop & report).
            if _floor_streak >= FLOOR_STALL_STEPS and ssr >= _ssr_at_streak * 0.99:
                _stop_reason = (f"floored {_floor_streak} steps with no SSR progress "
                                f"({_ssr_at_streak:.2e} → {ssr:.2e})")
                break
        else:
            dt         = float(compute_dt(W_n))
            W_n, ssr_j = _step(W_n, jnp.float64(dt))
            ssr        = float(ssr_j)
            jax.block_until_ready(W_n)

        t_phys += dt
        n      += 1
        _ssr_hist = _ssr_hist[1:] + [ssr]

        if MONITOR and (n <= 100 or n % cfg.PRINT_EVERY == 0):
            _monitor(W_n, n)

        if n <= 10 or n % cfg.PRINT_EVERY == 0 or halvings > 0:
            _el = _time.perf_counter() - _t0
            _hv = (f"  [{halvings} halving(s)]" if (_adaptive and halvings > 0) else "")
            _cfl_txt = (f"  CFL_eff={cfg.CFL * _cfl_mult:.1f}"
                        if (_adaptive and cfg.SER_BETA > 0.0) else "")
            _ramp_txt = f"  p_out={_p_out_now/1e5:.3f} bar" if _ramping else ""
            print(f"{n:>6}  {t_phys:>12.4e}  {dt:>12.4e}  {ssr:>12.4e}  ({_el:.1f}s){_cfl_txt}{_ramp_txt}{_hv}",
                  flush=True)

        # No convergence verdict mid-ramp: at the start of a stagnation ramp the
        # state IS steady for the current (high) back-pressure, so SSR is tiny.
        if ssr < cfg.SSR_TOL and not _ramping:
            _stop_reason = f"SSR={ssr:.3e} < tol={cfg.SSR_TOL:.1e}"
            break
        if n >= _win and not _ramping:
            _hmax = max(_ssr_hist); _hmin = min(_ssr_hist) + 1e-300
            if _hmax / _hmin < cfg.PLATEAU_RATIO:
                _stop_reason = f"SSR plateau ({_hmin:.3e}-{_hmax:.3e})"
                break
        if t_phys >= cfg.T_END:
            _stop_reason = f"t_end reached"
            break
except KeyboardInterrupt:
    _stop_reason = "interrupted by user"

_elapsed = _time.perf_counter() - _t0
print(f"\nStopped: {_stop_reason}")
print(f"  steps={n}  t={t_phys:.3e} s  SSR={ssr:.3e}  wall={_elapsed:.1f} s")

# ── Post-processing ───────────────────────────────────────────────────────────
U_h_f   = np.array(U_h_from_W_batch(W_n))
p_f     = U_h_f[:, 0]
u_g_f   = U_h_f[:, 1]
u_l_f   = U_h_f[:, 2]
alpha_g = U_h_f[:, 3]
h_g_f   = U_h_f[:, 4]
h_l_f   = U_h_f[:, 5]

T_g_f = np.array(fluid_g.get_state(jxp.HmassP_INPUTS, jnp.array(h_g_f), jnp.array(p_f))["temperature"])
T_l_f = np.array(fluid_l.get_state(jxp.HmassP_INPUTS, jnp.array(h_l_f), jnp.array(p_f))["temperature"])
rho_g_f = np.array(fluid_g.get_state(jxp.HmassP_INPUTS, jnp.array(h_g_f), jnp.array(p_f))["density"])
rho_l_f = np.array(fluid_l.get_state(jxp.HmassP_INPUTS, jnp.array(h_l_f), jnp.array(p_f))["density"])

# dominant phase for the single-phase comparison (dom set from the YAML composition)
if dom == "gas":
    u_dom, T_dom, rho_dom, h_dom = u_g_f, T_g_f, rho_g_f, h_g_f
else:
    u_dom, T_dom, rho_dom, h_dom = u_l_f, T_l_f, rho_l_f, h_l_f

# Per-phase mass-flux conservation (should be flat along the nozzle).  Split gas
# vs ghost: if the GAS flux is flat but the mixture/ghost is not, the ghost is the
# problem (its near-massless continuity is ill-constrained), not the main flow.
from nozzle_hyperbolic.fvm_solver.two_fluid import V_from_W
V_f    = np.array(jax.vmap(V_from_W)(W_n))
mdot_g = V_f[:, 0] * u_g_f * A                 # gas    mass flux  α_g ρ_g u_g A
mdot_l = V_f[:, 1] * u_l_f * A                 # liquid mass flux  α_l ρ_l u_l A
mdot   = mdot_g + mdot_l                        # mixture
print("\n── Single-phase-limit diagnostics ─────────────────────────────────")
print(f"  ghost alpha:  min={alpha_g.min():.6f}  max={alpha_g.max():.6f}  "
      f"(dominant phase = {dom})")
def _flux_rms(md):
    mm = np.mean(md[1:-1]); return mm, np.sqrt(np.mean((md[1:-1]-mm)**2))/(abs(mm)+1e-300)
_mmg, _rg = _flux_rms(mdot_g)
_mml, _rl = _flux_rms(mdot_l)
mm,  rms  = _flux_rms(mdot)
print(f"  GAS   mass flux  mean={_mmg:.4e} kg/s   RMS variation={_rg*100:.3f} %  <- dominant")
print(f"  ghost mass flux  mean={_mml:.4e} kg/s   RMS variation={_rl*100:.3f} %")
print(f"  total mass flux  mean={mm:.4e} kg/s   RMS variation={rms*100:.3f} %  "
      f"(→0 = quasi-1D mass conserved)")
print(f"  {dom} p: {p_f.min()/1e5:.2f}..{p_f.max()/1e5:.2f} bar   "
      f"u: {u_dom.min():.1f}..{u_dom.max():.1f} m/s   "
      f"T: {T_dom.min():.1f}..{T_dom.max():.1f} K")

# Stagnation-enthalpy conservation — the sharpest area-source check.  For steady
# adiabatic frictionless nozzle flow H0 = h + u^2/2 is CONSTANT along x.  If the
# area source (convective energy + the alpha*p*G / p*Dalpha terms) is wrong, H0
# drifts through the throat.  Report the spread relative to the inlet value.
H0_dom = h_dom + 0.5 * u_dom**2
H0_in  = H_g0 if dom == "gas" else H_l0     # reservoir stagnation enthalpy
dH0    = (H0_dom[1:-1] - H0_in) / (abs(H0_in) + 1e-300)
print(f"  stagnation enthalpy H0={H0_in/1e3:.2f} kJ/kg  "
      f"drift along x: max|ΔH0|/H0 = {np.max(np.abs(dH0)):.2e}  "
      f"(→0 = adiabatic nozzle energy conserved ⇒ area energy source correct)")

# ── Checkerboard (odd-even) diagnostic ────────────────────────────────────────
# The 2Δx sawtooth mode is invisible to a 3-point stencil, so a low-Mach density
# FVS lets it persist (pressure–velocity decoupling).  Measure each field's
# deviation from its local 3-point average: osc[i] = f[i] - ½(f[i-1]+f[i+1]).
# A large value on p / u_g (relative to the field's smooth range) = checkerboard.
def _oddeven(f):
    o = np.abs(f[1:-1] - 0.5 * (f[:-2] + f[2:]))
    rng = (f.max() - f.min()) + 1e-300
    return o.max(), o.max() / rng          # absolute, and relative to smooth range
print("\n── Checkerboard (odd-even 2Δx) amplitude ──────────────────────────")
for _nm, _f in [("p    ", p_f), ("u_g  ", u_g_f), ("u_l  ", u_l_f),
                ("du   ", u_g_f - u_l_f), ("alpha", alpha_g), ("h_g  ", h_g_f)]:
    _amp, _rel = _oddeven(_f)
    _flag = "  <-- CHECKERBOARD" if _rel > 0.02 else ""
    print(f"  {_nm}:  max 2Δx osc = {_amp:.3e}   ({_rel*100:.2f}% of range){_flag}")

# ── Mass-transfer source diagnostic (σ_Mg at flashing onset) ──────────────────
if cfg.MASS_TRANSFER and SRC_SCALE != 0.0:
    def _mt_cell(u_h):
        eos_g_c = phase_props(fluid_g, u_h[0], u_h[4])
        eos_l_c = phase_props(fluid_l, u_h[0], u_h[5])
        sat_g_c = fluid_g.get_state_saturation(u_h[0])
        sat_l_c = fluid_l.get_state_saturation(u_h[0])
        cl = two_fluid_closure(
            u_h, eos_g_c, eos_l_c, fluid_g, fluid_l,
            T_sat   = sat_g_c["temperature"],
            h_sat_g = sat_g_c["enthalpy"],
            h_sat_l = sat_l_c["enthalpy"],
            Nb      = cfg.BUBBLE_NUMBER,
            Nd      = cfg.DROPLET_NUMBER,
            mass_transfer = True,
            p_mt_max = P_MT_MAX,
            regime_htc = cfg.REGIME_HTC,
            area_law = cfg.INTERFACIAL_AREA,
            area_params = cfg.INTERFACIAL_AREA_PARAMS,
        )
        return (cl.sigma_Mg, cl.sigma_Qg, cl.sigma_Ql,
                sat_g_c["temperature"], eos_l_c.T)

    _sMg, _sQg, _sQl, _Tsat, _Tl = (np.array(v) for v in
                                    jax.vmap(_mt_cell)(jnp.array(U_h_f)))
    _superheat = _Tl - _Tsat                       # >0 = liquid superheated → σ_Mg on
    _on = np.flatnonzero(_sMg > 0.0)
    print("\n── Mass-transfer source (σ_Mg) diagnostic ─────────────────────────")
    if _on.size:
        print(f"  flashing onset: x={x[_on[0]]:.4f} m (cell {_on[0]})  "
              f"active in {_on.size}/{len(x)} cells")
        print(f"  σ_Mg: max={_sMg.max():.3e} kg/(m³s) at x={x[np.argmax(_sMg)]:.4f} m  "
              f"min={_sMg.min():.3e}")
        _amp, _rel = _oddeven(_sMg)
        print(f"  σ_Mg 2Δx osc = {_amp:.3e}  ({_rel*100:.2f}% of range)"
              + ("  <-- SOURCE CHECKERBOARD (source is the pump)" if _rel > 0.02
                 else "  (smooth → amplifier is the thin ghost, not the source)"))
    else:
        print("  σ_Mg = 0 everywhere (liquid nowhere superheated on the final state)")
    print(f"  liquid superheat T_l - T_sat: min={_superheat.min():.2f} K  "
          f"max={_superheat.max():.2f} K  "
          f"(crosses 0 at x≈{x[np.argmin(np.abs(_superheat))]:.4f} m)")

    _figS, _axS = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    _axS[0].plot(x, _sMg, 'o-', ms=2.5, lw=0.8)
    _axS[0].set_ylabel(r"$\sigma_{Mg}$ [kg/(m$^3$s)]")
    _axS[0].set_title("Mass-transfer source on the final state")
    _axS[1].plot(x, _superheat, 'o-', ms=2.5, lw=0.8)
    _axS[1].axhline(0.0, color='k', lw=0.6)
    _axS[1].set_ylabel(r"$T_l - T_{sat}$ [K]")
    _axS[2].plot(x, _sQg, 'o-', ms=2.5, lw=0.8, label=r"$\sigma_{Qg}$")
    _axS[2].plot(x, _sQl, 'o-', ms=2.5, lw=0.8, label=r"$\sigma_{Ql}$")
    _axS[2].set_ylabel(r"interfacial heat [W/m$^3$]")
    _axS[2].set_xlabel("x [m]")
    _axS[2].legend()
    for _a in _axS:
        _a.grid(alpha=0.3)
    _figS.tight_layout()
    _figS.savefig(out_dir / "mass_transfer_source.png", dpi=150)
    print(f"Source plot saved → {out_dir / 'mass_transfer_source.png'}")

# ── CSV — FULL two-phase profile (for overlay vs the nozzlex reference) ────────
# Both phases so slip (u_g vs u_l) and both temperatures can be compared.
csv_header = "x,A,p,alpha_g,u_g,u_l,T_g,T_l,h_g,h_l,rho_g,rho_l"
csv_data   = np.column_stack([x, A, p_f, alpha_g, u_g_f, u_l_f,
                              T_g_f, T_l_f, h_g_f, h_l_f, rho_g_f, rho_l_f])
sol_path = out_dir / "solution.csv"
np.savetxt(sol_path, csv_data, delimiter=",", header=csv_header, comments="")
print(f"Solution saved → {sol_path}")

# ── Plot: FINAL solution — same format as the initial condition ───────────────
_plot_state(U_h_f, f"FINAL solution (dominant={dom})  "
                   f"{cfg.GEO_FUNCTION}  N={cfg.Nx}  steps={n}",
            out_dir / "single_phase_limit.png")
plt.show()

# %%
