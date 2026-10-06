"""
sensitivity_study.py — AD closure-parameter sensitivity study, one command.

    poetry run python sensitivity_test/sensitivity_study.py             # everything from the yaml
    poetry run python sensitivity_test/sensitivity_study.py --nx 400    # override the mesh only
    poetry run python sensitivity_test/sensitivity_study.py --seed      # start from ic.run
    poetry run python sensitivity_test/sensitivity_study.py --no-seed   # force a cold start

The initial condition follows the yaml's `ic.type` by default; --seed/--no-seed
override it. A seed is never a second simulation -- it is a starting state for
the one solve -- but a cold start is the honest default and is what runs unless
the yaml or the switch asks otherwise.

Runs ONE simulation, differentiates the converged state IN MEMORY (never through a
csv round-trip), and writes solution + sensitivity products to
`sensitivity_test/results/<tag>/`.

What this script is
-------------------
Phases 0-2 and 5-7 of the hand-off, honestly scoped to a single solve.

The solve reaches steady state by implicit time marching — thousands of adaptive
steps.  Differentiating that trajectory would mean unrolling every one of them,
which is infeasible, so this is Option B (implicit differentiation): at
convergence R(W; theta) = 0, and differentiating the CONSTRAINT rather than the
path gives

    dW/dtheta = -(dR/dW)^-1 (dR/dtheta)

with dR/dW the Jacobian Newton already assembles.  The primal is untouched; the
tangent is exactly the implicit-function-theorem derivative and carries no
iteration-count noise.  Cost is one linear solve of a system the code already
solves several times per timestep, plus one forward-AD column per parameter.

What this script is NOT
-----------------------
The hand-off's Phase 3 gate (central FD in log space at three step sizes for
every parameter x QoI) needs 2*n_param*3 *converged solves*.  At one simulation
that is out of reach, so this runs in the equivalent of `--no-gate` and the
report carries the loud banner the hand-off requires.

What IS affordable, and is run unconditionally, is the same comparison one level
down: AD vs central FD of the RESIDUAL, dR/dtheta, which costs function
evaluations rather than solves.  That covers the failure modes the gate is
actually aimed at — dead-branch NaN, a stray stop_gradient, a parameter
disconnected from the physics, a regime switch crossed between stencil points —
everything except an error in the linear solve itself, which the rcond sweep
covers separately.  It is a weaker claim than the full gate and is labelled as
such wherever it is reported.

DGSM (§7.5) and the mesh study (§8) are likewise multi-solve and are out of
scope by construction; the report says so rather than omitting them silently.

Parameters
----------
The closure constants are NOT reachable through the installed library: `Nu_d` is
the hard-coded `Nu_g = 120` at sources.py:260, and the Schiller-Naumann drag and
Ranz-Marshall Nusselt carry no multiplier at all.  Rather than edit the library,
this script owns a theta-parametrized copy of `two_fluid_closure`'s
mass-transfer branch (§CLOSURE).  `check_closure_parity` verifies it reproduces
the library's source vector to machine precision at theta_0 before anything is
differentiated — without that check the study would be measuring this file's
transcription rather than the solver.
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import io
import json
import os
import pathlib
import subprocess
import sys
import time as _time
from typing import NamedTuple

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))          # dev_scripts/ — for config_loader

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp
import yaml as _yaml

from config_loader import load_config, resolved_dict
import nozzle_hyperbolic as hyp
import nozzle_hyperbolic.fvm_solver.geometry as _geom
from nozzle_hyperbolic.fvm_solver.two_fluid import (
    make_two_fluid_step, W_from_U_h_batch, U_h_from_W_batch)
from nozzle_hyperbolic.fvm_solver.two_fluid.sources import (
    two_fluid_closure, evaporation_weight, TwoFluidClosure,
    D_VE_algebric, D_VE_algebric_two_component, D_VE_area,
    interfacial_area, calculate_drag_force_one_component,
    calculate_mass_transfer_lv, ranz_marshall_htc_two_fluid)

_PG = ("perfect_gas", "perfectgas", "ideal_gas", "idealgas")

# Library constants this script must mirror exactly.  If any of these drifts in
# sources.py, check_closure_parity fails loudly rather than silently measuring
# the wrong model.
LIB_ALPHA_B, LIB_ALPHA_D = 0.3, 0.7        # two_fluid_closure defaults
LIB_D_MIN_B = 5e-6                         # sources.py:238 call site
LIB_GATE_DT, LIB_GATE_DH = 1e-3, 0.5       # two_fluid_closure signature
LIB_NU_G = 120.0                           # sources.py:260 — the hard-coded one


# =============================================================================
# §PARAMS — theta declaration, log-parameterization, priors
# =============================================================================
class Param(NamedTuple):
    name: str
    value: float          # baseline theta_0 (physical units, NOT log)
    sigma: float          # prior std-dev of ln(theta); None => 1-D sweep, not a bar
    units: str
    kind: str             # "physical" | "numerical"
    provenance: str
    enters: str


# Prior sigmas are the hand-off §1 table.  Provenance is recorded because without
# experimental data the prior is not one input among several — it is doing ALL the
# work in the ranking, and two people with different priors get different answers
# and are both correct.  Anything tagged "author judgement" is exactly that.
PARAMS = [
    Param("N_b", None, 2.3, "1/m^3", "physical",       # <- sources.bubble_number
          "hand-off §1: ~1 decade 1-sigma. Author judgement; no measurement.",
          "interfacial_area -> a_i, d -> drag + both HTCs"),
    Param("N_d", None, 2.3, "1/m^3", "physical",       # <- sources.droplet_number
          "hand-off §1: ~1 decade 1-sigma. Author judgement; no measurement.",
          "interfacial_area (droplet branch only)"),
    Param("Nu_d", LIB_NU_G, 0.7, "-", "physical",
          "hand-off §1: ~factor 2. Currently HARD-CODED as Nu_g=120 at "
          "sources.py:260 with the Ranz-Marshall call commented out above it.",
          "htc_g = k_g*Nu_d/d -> interfacial heat to gas -> sigma_Mg"),
    Param("Nu_c", 1.0, 0.7, "- (multiplier)", "physical",
          "hand-off §1: ~factor 2. Multiplier on Ranz-Marshall "
          "Nu = 2 + 0.6 Re^0.5 Pr^(1/3); no multiplier exists in the library.",
          "htc_l -> interfacial heat to liquid -> sigma_Mg"),
    Param("C_D", 1.0, 0.7, "- (multiplier)", "physical",
          "hand-off §1: ~factor 2. Multiplier on Schiller-Naumann "
          "24/Re (1 + 0.15 Re^0.687); no multiplier exists in the library.",
          "Fv_g -> interfacial drag -> slip"),
    Param("C_vm", None, 1.1, "-", "physical",          # <- numerics.k_vm
          "hand-off §1: ~factor 2-5. Read from numerics.k_vm.",
          "flux eigenstructure (factory k) AND D_VE_algebric"),
]

QOI_LABELS = {
    "mdot":        ("mass flow rate", "kg/s"),
    "x_exit":      ("exit flowing quality", "-"),
    "slip_exit":   ("exit slip ratio u_g/u_l", "-"),
    "p_exit":      ("exit pressure", "Pa"),
    "x_front":     ("front position (alpha_g = 0.5)", "m"),
    "delta_p":     ("integral pressure thickness", "m"),
    "L_relax":     ("implied thermal relaxation length", "m"),
    "max_dpdx":    ("smoothed max |dp/dx|  (DIAGNOSTIC ONLY)", "Pa/m"),
}
# Established this session: AD and FD field sensitivities agree over the whole
# nozzle EXCEPT the last ~8 cells at the outflow, where AD violates the
# x-independence that steady mass conservation forces on d(mdot)/d(theta).  Any
# QoI evaluated at the exit plane samples exactly that region.
EXIT_PLANE_QOIS = {"x_exit", "slip_exit", "p_exit"}


# =============================================================================
# §CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="settings/settings_flashing_smd_space_marching_comparison.yaml",
                   help="settings yaml, relative to dev_scripts/ (default: flashing SMD)")
    p.add_argument("--nx", default=None,
                   help="cells; overrides grid.Nx. A COMMA-SEPARATED LIST runs a "
                        "mesh-continuation ladder, e.g. --nx 100,200,400: each rung "
                        "is solved from the previous rung's solution and only the "
                        "LAST is differentiated.")
    p.add_argument("--ladder-ssr", type=float, default=1e-6,
                   help="SSR target for the intermediate rungs (default 1e-6). Only "
                        "the final mesh is driven to run.ssr_tol; the earlier ones "
                        "exist to give the next mesh a good seed, not to be analysed.")
    p.add_argument("--ssr-tol", type=float, default=None,
                   help="override run.ssr_tol")
    p.add_argument("--max-steps", type=int, default=None, help="override run.n_steps")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--seed", dest="seed", action="store_true", default=None,
                   help="force seeding from ic.run in the yaml (interpolated in x)")
    g.add_argument("--no-seed", dest="seed", action="store_false",
                   help="force a cold start, ignoring ic.run")
    p.add_argument("--dt-factor", type=float, default=1e8,
                   help="steady-Jacobian dt = this * dt_CFL (suppresses the transient term)")
    p.add_argument("--lib-closure", action="store_true",
                   help="DEBUG: march with the library's two_fluid_closure instead "
                        "of the theta-parametrized copy. A/B for isolating whether a "
                        "divergence comes from the closure transcription.")
    p.add_argument("--rcond-sweep", action="store_true",
                   help="add the SVD-based pseudo-inverse-cut sweep (O(n^3), slow at Nx=400)")
    p.add_argument("--bc-outlet", default=None, help="override bc.outlet")
    p.add_argument("--tag", default=None, help="output subfolder name")
    p.add_argument("--out", default=None, help="output root (default sensitivity_test/results)")
    return p.parse_args()


ARGS = parse_args()
_t_start = _time.perf_counter()


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(_HERE), stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:                                          # noqa: BLE001
        return "unknown"


def banner(txt, ch="="):
    print("\n" + ch * 78 + f"\n{txt}\n" + ch * 78, flush=True)


# =============================================================================
# §SETUP — config, fluids, grid, boundary conditions, initial condition
# =============================================================================
_cfg_path = _HERE.parent / ARGS.config
RAW = resolved_dict(_cfg_path)       # the settings file with its `base:` defaults merged in


def _as_ladder(v):
    """Parse a mesh spec into a list of ints. Accepts 400, [100,200,400],
    '100,200,400' and '100 200 400'."""
    if isinstance(v, (list, tuple)):
        return [int(i) for i in v]
    if isinstance(v, str):
        return [int(t) for t in v.replace(",", " ").split()]
    return [int(v)]


# grid.Nx may be a LIST in the yaml, but config_loader does int(g["Nx"]) and would
# raise on one.  Hand it a scalar (the final mesh) via a temporary copy of the
# yaml written beside the original, so any path resolved relative to the config's
# directory still resolves the same way.
_raw_nx = RAW.get("grid", {}).get("Nx", 100)
_NX_SRC = "yaml"
LADDER = _as_ladder(_raw_nx)
if ARGS.nx is not None:
    LADDER, _NX_SRC = _as_ladder(ARGS.nx), "--nx"
if len(LADDER) < 1 or any(i < 4 for i in LADDER):
    sys.exit(f"invalid mesh spec {LADDER!r}")

if isinstance(_raw_nx, (list, tuple)):
    import copy as _copy, tempfile as _tf
    _patched = _copy.deepcopy(RAW)
    _patched["grid"]["Nx"] = LADDER[-1]
    _fd, _tmp = _tf.mkstemp(suffix=".yaml", prefix="_nxladder_", dir=str(_cfg_path.parent))
    os.close(_fd)
    _tmp = pathlib.Path(_tmp)
    _tmp.write_text(_yaml.safe_dump(_patched, sort_keys=False), encoding="utf-8")
    try:
        cfg = load_config(_tmp)
    finally:
        _tmp.unlink(missing_ok=True)
else:
    cfg = load_config(_cfg_path)
cfg.Nx = LADDER[-1]
if ARGS.ssr_tol is not None:
    cfg.SSR_TOL = float(ARGS.ssr_tol)
if ARGS.max_steps is not None:
    cfg.N_STEPS = int(ARGS.max_steps)
if ARGS.bc_outlet is not None:
    cfg.BC_OUTLET = ARGS.bc_outlet

# The yaml owns the initial condition; the switch only overrides it.  Seeding is
# NOT a second simulation — it is where the single solve starts — but a cold
# start is the default because a seed taken from a different geometry or a
# different operating point silently lengthens the transient instead of
# shortening it.
if ARGS.seed is None:
    IC_MODE, IC_SRC = RAW["ic"]["type"], "yaml ic.type"
elif ARGS.seed:
    IC_MODE, IC_SRC = "restart", "--seed"
else:
    IC_MODE, IC_SRC = "stagnation", "--no-seed"

CASE = _cfg_path.stem
TAG = ARGS.tag or f"{RAW.get('save_options', {}).get('repo_run', CASE)}_N{LADDER[-1]}"
OUT_ROOT = pathlib.Path(ARGS.out) if ARGS.out else (_HERE / "results")
OUT = OUT_ROOT / TAG
OUT.mkdir(parents=True, exist_ok=True)

banner(f"AD CLOSURE-SENSITIVITY STUDY — {CASE}   Nx={cfg.Nx}")
print(f"  config      : {_cfg_path}")
print(f"  output      : {OUT}")
print(f"  mass_transfer={cfg.MASS_TRANSFER}  bc_inlet={cfg.BC_INLET}  "
      f"bc_outlet={cfg.BC_OUTLET}")
print(f"  mesh={' -> '.join(str(i) for i in LADDER)} (from {_NX_SRC})"
      + (f"   [continuation ladder: only N={LADDER[-1]} is differentiated]"
         if len(LADDER) > 1 else "")
      + f"\n  ssr_tol={cfg.SSR_TOL:.1e}"
      + (f" (intermediate rungs {ARGS.ladder_ssr:.1e})" if len(LADDER) > 1 else "")
      + f"  max_steps={cfg.N_STEPS}  git={_git_commit()}")
print(f"  initial condition: {IC_MODE}  (from {IC_SRC})"
      + ("  [ONE simulation, cold start - no seed]" if IC_MODE != "restart"
         else "  [seeded from ic.run - still ONE simulation]"))
if not cfg.MASS_TRANSFER:
    print("  NOTE: mass_transfer=false — the mass-transfer closures cannot be "
          "exercised in this case.")


def _build_fluid(lut):
    """Build a fluid, with `table_dir` resolved against the CONFIG's directory.

    The yaml writes `table_dir: fluid_tables`, a RELATIVE path, which jaxprop
    resolves against the process working directory.  Running this script from
    sensitivity_test/ therefore looked for a different table store than running
    main_two_fluid.py from dev_scripts/ -- and since the cached filename encodes
    only the grid size (`90x90`) and never the contents, the two stores silently
    held DIFFERENT tables under the same name.  That produced a NaN on the first
    timestep for one cwd and a healthy solve for the other, on identical inputs.
    Anchoring to the config's directory makes the choice of tables a property of
    the configuration rather than of where you happened to be standing.
    """
    lut = dict(lut)
    if "table_dir" in lut and lut["table_dir"] is not None:
        td = pathlib.Path(lut["table_dir"])
        if not td.is_absolute():
            td = (_cfg_path.parent / td).resolve()
        lut["table_dir"] = str(td)
    if str(lut.get("backend", "")).lower() in _PG:
        return jxp.FluidPerfectGas(lut["fluid_name"], T_ref=lut["T_ref"],
                                   p_ref=lut["p_ref"]), True
    return jxp.FluidBicubic(**lut), False


print(f"\nBuilding EOS tables...  table_dir="
      f"{cfg.LUT_L.get('table_dir', '(none)')} -> resolved against {_cfg_path.parent}")
fluid_g, _pg_g = _build_fluid(cfg.LUT_G)
fluid_l, _pg_l = _build_fluid(cfg.LUT_L)
fl_g_inv = fluid_g if _pg_g else jxp.FluidJAX(cfg.LUT_G["fluid_name"])
fl_l_inv = fluid_l if _pg_l else jxp.FluidJAX(cfg.LUT_L["fluid_name"])

# ── Grid + WELL-BALANCED area gradient ───────────────────────────────────────
# dAdx from the FACE areas, not the analytic derivative: the analytic form leaves
# a discrete imbalance between the quasi-1D area source and the area-weighted
# flux divergence, which forces a 2*dx void-fraction mode.  One line, and it is
# load-bearing.
def setup_grid(nx):
    """(Re)build every grid-dependent global for a mesh of nx cells.

    build() reads A / A_face / dAdx / dx as globals at CALL time, so a
    continuation ladder just calls this before each rung and everything
    downstream — the solver factory, the Jacobian, the QoIs — picks up the new
    mesh with no further plumbing.
    """
    global x, dx, geo, A, dAdx, A_face, A_j, I_THROAT, N, Nc
    cfg.Nx = int(nx)
    x = np.linspace(cfg.x1, cfg.x2, cfg.Nx)
    dx = float(x[1] - x[0])
    geo = getattr(_geom, cfg.GEO_FUNCTION)(x, roughness=cfg.GEO_ROUGHNESS,
                                           **cfg.GEO_KWARGS)
    A = np.asarray(geo.A, dtype=float)
    dAdx = np.array(geo.dAdx, dtype=float)
    A_face = 0.5 * (A[:-1] + A[1:])
    dAdx[1:-1] = (A_face[1:] - A_face[:-1]) / dx
    A_j = jnp.asarray(A)
    I_THROAT = int(np.argmin(A))
    N, Nc = cfg.Nx, cfg.Nx - 2


setup_grid(LADDER[0])

# ── Stagnation inlet ─────────────────────────────────────────────────────────
P0 = cfg.BC_P0 if cfg.BC_P0 is not None else cfg.P_INIT
T_G0 = cfg.BC_T_G0
T_L0 = cfg.BC_T_L0 if cfg.BC_T_L0 is not None else T_G0
SLIP = float(cfg.BC_SLIP)

_stg0 = fl_g_inv.get_state(jxp.PT_INPUTS, jnp.array([P0]), jnp.array([T_G0]))
_stl0 = fl_l_inv.get_state(jxp.PT_INPUTS, jnp.array([P0]), jnp.array([T_L0]))
H_g0, s_g0 = float(_stg0["enthalpy"][0]), float(_stg0["s"][0])
H_l0, s_l0 = float(_stl0["enthalpy"][0]), float(_stl0["s"][0])
rho_g0, rho_l0 = float(_stg0["density"][0]), float(_stl0["density"][0])



# Ghost-vapor reservoir fix: for a SUBCOOLED (flashing) reservoir the equilibrium
# PT call returns the LIQUID branch for the gas phase, below the vapor table
# floor.  Take saturated vapor at p0 instead.
_hg_min = float(getattr(fluid_g, "h_min", -np.inf))
if H_g0 < _hg_min and not _pg_g:
    _sat0 = fluid_g.get_state_saturation(jnp.array([P0]))
    H_g0, s_g0 = float(_sat0["enthalpy"][0]), float(_sat0["entropy"][0])
    rho_g0 = float(_sat0["density"][0])
    print(f"  [ghost-vapor fix] saturated vapor at p0: H_g0={H_g0/1e3:.1f} kJ/kg")


def check_eos_tables(rel_tol=0.02):
    """Cross-check each interpolated table against the direct EOS evaluator.

    A cached bicubic table is keyed on grid size alone (`water_meta_liquid_90x90`
    says nothing about ranges or contents), so a stale or corrupt file loads
    silently and only announces itself as a NaN several seconds into the march --
    where it is indistinguishable from a physics or numerics problem.  Measured
    case: a corrupt liquid table differed from the good one in 181 arrays,
    density by up to 471 kg/m^3, and killed the solve on step 1.

    Probe points are chosen to be unambiguously SINGLE PHASE, where the
    metastable table and the equilibrium evaluator must agree:
      * liquid at the reservoir state (subcooled by ~19 K at 120 bar),
      * gas nudged above h_sat_g into the superheated region, since sitting
        exactly on the saturation line risks the equilibrium call returning a
        two-phase mixture density and failing for the wrong reason.

    Checksum-free by design: it stays valid when a table is legitimately rebuilt
    with different ranges, and still catches one that is simply wrong.
    """
    probes = []
    if not _pg_l:
        probes.append(("liquid", fluid_l, fl_l_inv, H_l0, P0))
    if not _pg_g:
        h_hi = float(getattr(fluid_g, "h_max", H_g0 * 1.2))
        probes.append(("gas", fluid_g, fl_g_inv,
                       min(H_g0 + 5.0e4, 0.98 * h_hi), P0))
    bad = []
    for tag, tab, ref, h_p, p_p in probes:
        try:
            rho_t = float(tab.get_state(jxp.HmassP_INPUTS, jnp.array([h_p]),
                                        jnp.array([p_p]))["density"][0])
            rho_r = float(ref.get_state(jxp.HmassP_INPUTS, jnp.array([h_p]),
                                        jnp.array([p_p]))["density"][0])
        except Exception as exc:                                   # noqa: BLE001
            bad.append(f"  {tag}: lookup FAILED ({type(exc).__name__}: {exc})")
            continue
        err = abs(rho_t - rho_r) / max(abs(rho_r), 1e-300)
        ok = np.isfinite(rho_t) and np.isfinite(rho_r) and err <= rel_tol
        print(f"  [eos check] {tag:<6} h={h_p/1e3:8.1f} kJ/kg  p={p_p/1e5:7.2f} bar  "
              f"rho: table={rho_t:10.4f}  direct={rho_r:10.4f}  "
              f"rel={err*100:7.3f} %  {'ok' if ok else 'MISMATCH'}")
        if not ok:
            bad.append(f"  {tag}: table rho={rho_t:.4f} vs direct rho={rho_r:.4f} "
                       f"({err*100:.2f} % off) at p={p_p:.6g} Pa, h={h_p:.6g} J/kg")
    if bad:
        sys.exit(
            "\nEOS TABLE CHECK FAILED -- an interpolated table disagrees with the\n"
            "direct evaluator at a single-phase probe point, where they must match.\n"
            "The table is stale or corrupt and the solve would NaN.\n\n"
            + "\n".join(bad)
            + f"\n\ntable_dir: {cfg.LUT_L.get('table_dir')}\n"
            "Delete the offending .pkl there and let it rebuild, or point\n"
            "eos.*.table_dir at a known-good store. NOTE: a rebuild may reproduce\n"
            "the same bad table -- the liquid-table generator in jaxprop has been\n"
            "seen to emit a corrupt 90x90 table deterministically.")


check_eos_tables()

if cfg.BC_QUALITY is not None:
    QUALITY = float(cfg.BC_QUALITY)
    ALPHA_G = 1.0 / (1.0 + (1.0 - QUALITY) / QUALITY * (rho_g0 / rho_l0) * SLIP)
elif cfg.BC_VOID is not None:
    ALPHA_G = float(cfg.BC_VOID)
    QUALITY = 1.0 / (1.0 + (1.0 / ALPHA_G - 1.0) / ((rho_g0 / rho_l0) * SLIP))
else:
    QUALITY = ALPHA_G = None

dom = cfg.DOM_PHASE if cfg.DOM_PHASE in ("gas", "liquid") else (
    "gas" if (QUALITY is not None and QUALITY >= 0.5) else "liquid")

_iso_h = _iso_p = None
if cfg.BC_INLET in ("stagnation", "stagnation_ramped"):
    _dom_inv = fl_g_inv if dom == "gas" else fl_l_inv
    _s_dom0 = s_g0 if dom == "gas" else s_l0
    _iso_p = np.linspace(float(getattr(fluid_g, "p_min", 1e4)) * 1.01,
                         float(getattr(fluid_g, "p_max", 300e5)) * 0.99, 400)
    _iso_h = np.asarray(_dom_inv.get_state(
        jxp.PSmass_INPUTS, jnp.asarray(_iso_p),
        jnp.full(_iso_p.size, jnp.float64(_s_dom0)))["h"])

# ── Initial condition ────────────────────────────────────────────────────────
_p_in = float(P0 * 0.99)
_stg_in = fluid_g.get_state(jxp.PSmass_INPUTS, jnp.array([_p_in]), jnp.array([s_g0]))
h_g_0 = float(_stg_in["enthalpy"][0])
u_g_in = float(np.sqrt(max(2.0 * (H_g0 - h_g_0), 0.0)))
u_l_in = u_g_in / SLIP
h_l_0 = H_l0 - 0.5 * u_l_in**2

def _interp_onto_grid(xs, cols, label):
    """Linear interpolation of a stored profile onto the CURRENT grid."""
    Uh = np.column_stack([np.interp(x, xs, c) for c in cols])
    if not np.isfinite(Uh).all():
        sys.exit(f"seed profile contains NaN/Inf: {label}")
    if abs(xs[0] - cfg.x1) > 1e-9 or abs(xs[-1] - cfg.x2) > 1e-9:
        print(f"  ** seed domain [{xs[0]:.4g}, {xs[-1]:.4g}] m does not match this "
              f"config's [{cfg.x1:.4g}, {cfg.x2:.4g}] m — the seed is a DIFFERENT "
              f"nozzle and the transient will not be short. **")
    return Uh


def make_ic(prev_csv=None):
    """Initial condition for the CURRENT grid.

    `prev_csv` is the previous ladder rung's solution.csv.  When present it wins
    over ic.type: continuation is the whole point of the ladder, and the rungs
    below the last exist only to hand the next mesh a converged seed.
    """
    if prev_csv is not None:
        D = np.genfromtxt(prev_csv, delimiter=",", names=True)
        Uh = _interp_onto_grid(np.asarray(D["x"]),
                               [D[c] for c in ("p", "u_g", "u_l", "alpha_g",
                                               "h_g", "h_l")], str(prev_csv))
        print(f"IC (ladder): {prev_csv.name}  {D['p'].size} -> {cfg.Nx} cells")
        return Uh

    if IC_MODE == "restart":
        run_rel = RAW.get("ic", {}).get("run")
        if run_rel is None:
            sys.exit("Seeding was requested but the yaml has no `ic.run:` entry. "
                     "Add one, or run with --no-seed for a cold start.")
        rst = pathlib.Path(run_rel)
        if not rst.is_absolute():
            rst = _HERE.parent / rst
        if rst.is_dir():
            rst = rst / "solution.csv"
        D = np.genfromtxt(rst, delimiter=",", names=True)
        # Read the source x COLUMN — never synthesize it from cfg.x1/cfg.x2.  Runs
        # of the same case exist on different domain lengths, and assuming the
        # current extent silently squashes one onto the other, producing a
        # plausible-looking IC that is a solution of nothing.
        Uh = _interp_onto_grid(np.asarray(D["x"]),
                               [D[c] for c in ("p", "u_g", "u_l", "alpha_g",
                                               "h_g", "h_l")], str(rst))
        print(f"IC (restart): {rst.parent.name}  {D['p'].size} -> {cfg.Nx} cells")
        return Uh

    if IC_MODE == "stagnation":
        print(f"IC (stagnation cold start): p={P0/1e5:.1f} bar  alpha_g={ALPHA_G:.2e}")
        return np.tile([P0, 1e-6, 1e-6, ALPHA_G, H_g0, H_l0], (cfg.Nx, 1)).astype(float)

    if IC_MODE == "simple":
        u = 5.0
        print(f"IC (simple cold start): p ramp {P0/1e5:.1f}->{cfg.P_OUT/1e5:.2f} bar")
        return np.stack([np.linspace(P0, cfg.P_OUT, cfg.Nx),
                         np.full(cfg.Nx, u * SLIP), np.full(cfg.Nx, u),
                         np.full(cfg.Nx, ALPHA_G),
                         np.full(cfg.Nx, H_g0 - 0.5 * (u * SLIP)**2),
                         np.full(cfg.Nx, H_l0 - 0.5 * u**2)], axis=1)

    print(f"IC (uniform): p={_p_in/1e5:.1f} bar")
    return np.tile([_p_in, u_g_in, u_l_in, ALPHA_G, h_g_0, h_l_0],
                   (cfg.Nx, 1)).astype(float)


# =============================================================================
# §CLOSURE — theta-parametrized copy of two_fluid_closure's mass-transfer branch
# =============================================================================
class _AreaClosure(NamedTuple):
    dlnA_dx: jnp.ndarray


_D_VE = D_VE_algebric if cfg.MASS_TRANSFER else D_VE_algebric_two_component


def _transport(p, h_g, h_l):
    """Extended EOS for mu / k / cp, clipped exactly as the library clips."""
    p_a = jnp.expand_dims(p, 0)
    p_a = jnp.clip(p_a, getattr(fluid_l, "p_min", 1e2), getattr(fluid_l, "p_max", 1e9))
    p_a = jnp.clip(p_a, getattr(fluid_g, "p_min", 1e2), getattr(fluid_g, "p_max", 1e9))
    hg = jnp.clip(jnp.expand_dims(h_g, 0), getattr(fluid_g, "h_min", 1e3),
                  getattr(fluid_g, "h_max", 5e6))
    hl = jnp.clip(jnp.expand_dims(h_l, 0), getattr(fluid_l, "h_min", 1e3),
                  getattr(fluid_l, "h_max", 5e6))
    return (fluid_g.get_state(jxp.HmassP_INPUTS, hg, p_a),
            fluid_l.get_state(jxp.HmassP_INPUTS, hl, p_a))


def theta_closure(U_h, eos_g, eos_l, T_sat, h_sat_g, h_sat_l, dlnA_dx, th):
    """Line-for-line mirror of two_fluid_closure(mass_transfer=True) with the
    closure constants lifted out of the body into `th`.

    Verified against the library by check_closure_parity() before use.  Returns
    the closure plus the diagnostics the report needs (a_i, d, htc, the
    evaporation-gate weight, and the thermal relaxation time).
    """
    p, u_g, u_l, ag, h_g, h_l = U_h
    al = 1.0 - ag
    rg, T_g = eos_g.rho, eos_g.T
    rl, T_l = eos_l.rho, eos_l.T

    w_mt = evaporation_weight(h_l, T_l, h_sat_l, T_sat, ag,
                              gate_dT=LIB_GATE_DT, gate_dh=LIB_GATE_DH)
    stg, stl = _transport(p, h_g, h_l)

    ai, d = interfacial_area(ag, alpha_b=LIB_ALPHA_B, alpha_d=LIB_ALPHA_D,
                             Nb=th["N_b"], Nd=th["N_d"], d_min_b=LIB_D_MIN_B)
    mu_g, k_g = stg["viscosity"][0], stg["conductivity"][0]
    mu_l = stl["viscosity"][0]

    Fv_g = -th["C_D"] * calculate_drag_force_one_component(
        ag, al, rg, rl, mu_g, mu_l, u_g, u_l, d,
        alpha_b=LIB_ALPHA_B, alpha_d=LIB_ALPHA_D)

    du_abs = jnp.abs(u_g - u_l)
    htc_g = (k_g * th["Nu_d"]) / d
    htc_l = th["Nu_c"] * ranz_marshall_htc_two_fluid(
        {"rho": stl["rho"][0], "mu": stl["mu"][0],
         "k": stl["k"][0], "cp": stl["cp"][0]}, du_abs, d)

    # Evaluate ungated, then scale ALL THREE exchanges by the SAME weight: that
    # keeps the interface energy balance sigma_Mg = -(Qvi+Qli)/(h_sat_g-h_sat_l)
    # exact at every gate level.  Scaling sigma alone would create interfacial
    # energy from nothing.
    _Qvi = htc_g * ai * (T_sat - T_g)
    _Qli = htc_l * ai * (T_sat - T_l)
    _sMg = calculate_mass_transfer_lv(T_l, T_g, T_sat, htc_l, htc_g, ai,
                                      h_sat_l, h_sat_g, h_l, h_g, _Qvi, _Qli)
    Qvi, Qli, sigma_Mg = w_mt * _Qvi, w_mt * _Qli, w_mt * _sMg

    u_mix2 = (ag * rg * u_g + al * rl * u_l) / (ag * rg + al * rl)
    u_ex = jax.lax.cond(sigma_Mg >= 0.0, lambda _: u_l, lambda _: u_g, operand=None)

    # Thermal relaxation time of the LIQUID (continuous) side — the physically
    # meaningful steepness scale, and the anchor for the §7.1 hand-check.
    tau = (al * rl * stl["cp"][0]) / jnp.maximum(htc_l * ai, 1e-30)

    clo = TwoFluidClosure(
        sigma_Mg=sigma_Mg, sigma_Qg=Qvi, sigma_Ql=Qli, Fv_g=Fv_g,
        F_g=0.0, F_l=0.0, Q_g=0.0, Q_l=0.0,
        u_ex=u_ex, u_int=u_mix2, h_ex_g=h_sat_g, h_ex_l=h_sat_l, dlnA_dx=dlnA_dx)
    return clo, dict(ai=ai, d=d, htc_g=htc_g, htc_l=htc_l, w_mt=w_mt,
                     sigma_Mg=sigma_Mg, tau=tau, u_mix=u_mix2)


def make_sources_fn(th, use_theta=True):
    """The driver's source closure, with the constants taken from `th`.

    `th` is a dict of TRACED scalars here, so jacfwd w.r.t. it flows through the
    interfacial area, the dispersed length scale, drag, and both heat-transfer
    coefficients — which is what lets this work without touching the library.
    """
    def _sources_fn(u_h, eos_g, eos_l, cq, dlnA, sat_g, sat_l):
        Dv = jnp.zeros(6)
        if use_theta and cfg.MASS_TRANSFER:
            clo, _ = theta_closure(u_h, eos_g, eos_l, sat_g["temperature"],
                                   sat_g["enthalpy"], sat_l["enthalpy"], dlnA, th)
        else:
            clo = two_fluid_closure(
                u_h, eos_g, eos_l, fluid_g, fluid_l,
                T_sat=sat_g["temperature"], h_sat_g=sat_g["enthalpy"],
                h_sat_l=sat_l["enthalpy"], dlnA_dx=dlnA,
                Nb=th["N_b"], Nd=th["N_d"], mass_transfer=cfg.MASS_TRANSFER)
        Dv = Dv + _D_VE(u_h, cq, clo, eos_g, eos_l, th["C_vm"])
        Dv = Dv + D_VE_area(u_h, cq, _AreaClosure(dlnA_dx=dlnA), eos_g, eos_l)
        return Dv
    return _sources_fn


def build(th, use_theta=True, debug=True):
    """Rebuild the solver with theta-dependent sources.

    The factory itself never touches `th` beyond forwarding it, so `th` stays a
    tracer inside _sources_fn (and the flux's virtual-mass coefficient) only.
    """
    return make_two_fluid_step(
        fluid_g, fluid_l, A, A_face, dAdx, dx,
        CFL=cfg.CFL, k=th["C_vm"], delta_factor=cfg.DELTA_FACTOR,
        newton_iter=cfg.NEWTON_ITER, recon=cfg.RECON,
        sources_fn=make_sources_fn(th, use_theta),
        lowmach_diss=cfg.LOWMACH_DISS, lowmach_diss_bc=cfg.LOWMACH_DISS_BC,
        nc_scale=cfg.NC_SCALE, eps_void=cfg.EPS_VOID,
        eps_void_ucut=cfg.EPS_VOID_UCUT, eps_void_uwidth=cfg.EPS_VOID_UWIDTH,
        eps_void_scale=cfg.EPS_VOID_SCALE, eps_ghost=cfg.EPS_GHOST,
        eps_ghost_acut=cfg.EPS_GHOST_ACUT, eps_ghost_awidth=cfg.EPS_GHOST_AWIDTH,
        eps_ghost_k2=cfg.EPS_GHOST_K2, eps_ghost_alo=cfg.EPS_GHOST_ALO,
        ssr_mode=cfg.SSR_MODE, du_blend_lo=cfg.DU_BLEND_LO,
        du_blend_hi=cfg.DU_BLEND_HI, jacobian_mode=cfg.JACOBIAN,
        debug_newton=cfg.DEBUG_NEWTON,
        linear_solve=cfg.LINEAR_SOLVE, bc_inlet=cfg.BC_INLET,
        bc_outlet=cfg.BC_OUTLET, H_g0=H_g0, s_g0=s_g0, H_l0=H_l0, s_l0=s_l0,
        dom_phase=dom, iso_h=_iso_h, iso_p=_iso_p, alpha_g0=ALPHA_G,
        quality=QUALITY, slip=SLIP,
        U_h_inlet=jnp.asarray(U_h_0[0]), p_out=cfg.P_OUT,
        p_out_relax=cfg.P_OUT_RELAX, eos_project=cfg.EOS_PROJECT,
        proj_s_ref=cfg.PROJ_S_REF, eos_rows=cfg.EOS_ROWS,
        eos_rows_banded=cfg.EOS_ROWS_BANDED, jac_lag=cfg.JAC_LAG,
        nc_form=cfg.NC_FORM, use_saturation=cfg.MASS_TRANSFER,
        w_char=cfg.W_CHAR, alpha_eps=cfg.ALPHA_EPS, pin_ghost=cfg.PIN_GHOST,
        pin_ghost_momentum=cfg.PIN_GHOST_MOMENTUM, return_debug=debug)


# theta_0.  Every parameter the YAML owns is read from it; the rest are constants
# this script or the library owns (Nu_d is sources.py's hard-coded 120, Nu_c and
# C_D are multipliers with no yaml entry at all).  A parameter whose value is
# restated here instead of read would silently ignore the config -- keep the
# None-means-from-yaml convention.
_FROM_YAML = {"N_b": float(cfg.BUBBLE_NUMBER),
              "N_d": float(cfg.DROPLET_NUMBER),
              "C_vm": float(cfg.K_VM)}
PARAMS = [p._replace(value=_FROM_YAML.get(p.name, p.value)) for p in PARAMS]
assert all(p.value is not None for p in PARAMS), "theta_0 missing a value"
# PYTHON floats, not jnp scalars.  THETA0 is what the MARCH is built with, and
# main_two_fluid.py passes plain floats (notably k=cfg.K_VM into the flux
# eigenstructure).  A jnp scalar is strongly typed where a Python float is weakly
# typed, so it can change dtype promotion inside the factory and make this
# script's solve differ from the driver's on the same yaml.  The Jacobian path is
# unaffected: theta_vec() builds tracers from LOGTHETA0 independently.
THETA0 = {p.name: float(p.value) for p in PARAMS}
PNAMES = [p.name for p in PARAMS]
SIGMA = np.array([p.sigma for p in PARAMS], float)


# =============================================================================
# §PARITY — the script's closure must BE the library's closure at theta_0
# =============================================================================
def check_closure_parity(W):
    """Compare the theta-parametrized source vector against the library's, cell
    by cell, at theta_0.  Without this the study measures this file rather than
    the solver."""
    if not cfg.MASS_TRANSFER:
        return 0.0, "skipped (mass_transfer=false: the library branch is used directly)"
    _, _, res_t, _, _, _ = build(THETA0, use_theta=True)
    _, _, res_l, _, _, _ = build(THETA0, use_theta=False)
    W_int = W[1:-1].reshape(-1)
    W_full = W.reshape(-1)
    dt = jnp.float64(1.0)
    r_t = np.asarray(res_t(W_int, W_full, dt))
    r_l = np.asarray(res_l(W_int, W_full, dt))
    den = max(np.abs(r_l).max(), 1e-300)
    rel = float(np.abs(r_t - r_l).max() / den)
    return rel, ("PASS" if rel < 1e-12 else "FAIL")


# =============================================================================
# §SOLVE — the one simulation
# =============================================================================
# Mirrors main_two_fluid.py.  DOM_ONLY_DT excludes the GHOST phase's u/h from the
# dt gate: the ghost carries negligible mass but swings hard, and gating on it
# pins the march on the dt floor forever (measured: dt stuck at 3.1e-9, SSR ~1
# after 400 steps from an already-converged restart).
DOM_ONLY_DT = True
DT_FLOOR_FRAC = 1.0e-3
FLOOR_STALL_STEPS = 500
# Convergence is what controls sensitivity quality, not cell count: a measured
# comparison on this solver gave -0.0565 at 1.56% FD error from an SSR 1e-8 state
# and -0.0671 at 19.6% from an SSR 2.9e-6 state of the SAME case.
SSR_TRUSTED = 1e-7


def _increments(W_old, W_new, a_char):
    Uo = U_h_from_W_batch(W_old)[1:-1]
    Un = U_h_from_W_batch(W_new)[1:-1]
    bad = bool(jnp.any(jnp.isnan(Un)) | jnp.any(jnp.isinf(Un)))
    rel = lambda nn, oo, sc=0.0: float(
        jnp.max(jnp.abs(nn - oo) / (jnp.abs(oo) + sc + 1e-10)))
    dp = rel(Un[:, 0], Uo[:, 0])
    dug, dul = rel(Un[:, 1], Uo[:, 1], a_char), rel(Un[:, 2], Uo[:, 2], a_char)
    dhg, dhl = rel(Un[:, 4], Uo[:, 4]), rel(Un[:, 5], Uo[:, 5])
    if DOM_ONLY_DT:
        du = dug if dom == "gas" else dul
        dh = dhg if dom == "gas" else dhl
    else:
        du, dh = max(dug, dul), max(dhg, dhl)
    da = float(jnp.max(jnp.abs(Un[:, 3] - Uo[:, 3])))
    return bad, dp, du, dh, da


def march(step, compute_dt, W_0, ssr_tol):
    """Implicit time-march to steady state on the CURRENT grid."""
    W_n, ssr, n, t_phys = W_0, float("inf"), 0, 0.0
    dt = float(compute_dt(W_0))
    hist = [float("inf")] * cfg.PLATEAU_WINDOW
    floor_streak, ssr_at_streak = 0, float("inf")
    stop_reason = f"max steps ({cfg.N_STEPS}) reached"
    t0 = _time.perf_counter()
    print(f"\nRunning (max {cfg.N_STEPS} steps, SSR_tol={ssr_tol:.1e})")
    print(f"{'Step':>7} {'t [s]':>12} {'dt [s]':>12} {'SSR':>12} {'wall':>9}")
    try:
        while n < cfg.N_STEPS:
            if cfg.ADAPTIVE_DT:
                dt_cap = float(compute_dt(W_n))
                a_char = cfg.CFL * dx / max(dt_cap, 1e-300)
                dt_floor = DT_FLOOR_FRAC * dt_cap
                dt = max(min(dt * cfg.DT_GROW, dt_cap), dt_floor)
                bad = ok = floored = False
                for _h in range(cfg.MAX_HALVINGS + 1):
                    W_try, ssr_j = step(W_n, jnp.float64(dt))
                    jax.block_until_ready(W_try)
                    bad, dp, du, dh, da = _increments(W_n, W_try, a_char)
                    ok = ((not bad) and dp < cfg.DP_MAX and dh < cfg.DH_MAX
                          and du < cfg.DU_MAX and da < cfg.DA_MAX)
                    if ok:
                        break
                    # At the floor (or out of halvings): a NaN bails, a
                    # finite-but-over-threshold step is ACCEPTED and marching
                    # continues -- the steady-state keep-marching policy.
                    if dt <= dt_floor * (1 + 1e-9) or _h >= cfg.MAX_HALVINGS:
                        floored = True
                        break
                    dt = max(dt * 0.5, dt_floor)
                if bad:
                    stop_reason = f"NaN/Inf in state at step {n+1}"
                    break
                if floored and not ok:
                    if floor_streak == 0:
                        ssr_at_streak = ssr
                    floor_streak += 1
                else:
                    floor_streak = 0
                W_n = W_try
                ssr = max(float(ssr_j), ssr_tol * 10) if floored else float(ssr_j)
                # Bail only on a GENUINE stall: a long floor streak with no SSR
                # progress is a standing checkerboard; marching will not cure it.
                if floor_streak >= FLOOR_STALL_STEPS and ssr >= ssr_at_streak * 0.99:
                    stop_reason = (f"floored {floor_streak} steps with no SSR "
                                   f"progress ({ssr_at_streak:.2e} -> {ssr:.2e})")
                    break
            else:
                dt = float(compute_dt(W_n))
                W_n, ssr_j = step(W_n, jnp.float64(dt))
                ssr = float(ssr_j)
                jax.block_until_ready(W_n)
            t_phys += dt
            n += 1
            hist = hist[1:] + [ssr]
            if n <= 10 or n % cfg.PRINT_EVERY == 0:
                print(f"{n:>7} {t_phys:>12.4e} {dt:>12.4e} {ssr:>12.4e} "
                      f"{_time.perf_counter()-t0:>8.1f}s", flush=True)
            if ssr < ssr_tol:
                stop_reason = f"SSR={ssr:.3e} < tol={ssr_tol:.1e}"
                break
            if (n >= cfg.PLATEAU_WINDOW
                    and max(hist) / (min(hist) + 1e-300) < cfg.PLATEAU_RATIO):
                stop_reason = f"SSR plateau ({min(hist):.3e}-{max(hist):.3e})"
                break
            if t_phys >= cfg.T_END:
                stop_reason = "t_end reached"
                break
    except KeyboardInterrupt:
        stop_reason = "interrupted by user"
    return W_n, ssr, n, t_phys, stop_reason, _time.perf_counter() - t0


def write_solution(path, U):
    """State plus derived EOS quantities.

    The `x` column is what a later restart or the next ladder rung interpolates
    from, so it is always written -- synthesizing it from cfg.x1/cfg.x2 on read
    is exactly the bug that silently squashes one domain length onto another.
    """
    hg, hl, pp = jnp.asarray(U[:, 4]), jnp.asarray(U[:, 5]), jnp.asarray(U[:, 0])
    sg = fluid_g.get_state(jxp.HmassP_INPUTS, hg, pp)
    sl = fluid_l.get_state(jxp.HmassP_INPUTS, hl, pp)
    np.savetxt(path, np.column_stack(
        [x, A, U[:, 0], U[:, 1], U[:, 2], U[:, 3], U[:, 4], U[:, 5],
         np.asarray(sg["temperature"]), np.asarray(sl["temperature"]),
         np.asarray(sg["density"]), np.asarray(sl["density"])]),
        delimiter=",", comments="",
        header="x,A,p,u_g,u_l,alpha_g,h_g,h_l,T_g,T_l,rho_g,rho_l")


# -- The mesh-continuation ladder --------------------------------------------
# Each rung is solved starting from the previous rung's solution.csv.  The
# earlier rungs are NOT analysed: they exist to hand the next, finer mesh a
# converged seed, which is far cheaper and far more robust than cold-starting
# the fine mesh directly.  Only LADDER[-1] is differentiated, and the output
# folder is named for it.
banner(f"SOLVE -- mesh ladder {' -> '.join(str(i) for i in LADDER)}"
       if len(LADDER) > 1 else "SOLVE -- one simulation")
LADDER_LOG, _prev_csv = [], None
for _k, _nx in enumerate(LADDER):
    _last = (_k == len(LADDER) - 1)
    _tol = cfg.SSR_TOL if _last else max(cfg.SSR_TOL, ARGS.ladder_ssr)
    if len(LADDER) > 1:
        banner(f"rung {_k+1}/{len(LADDER)}:  N = {_nx}"
               + ("   [FINAL -- this is the one differentiated]" if _last
                  else f"   [seed for N={LADDER[_k+1]}, not analysed]"), ch="-")
    setup_grid(_nx)
    U_h_0 = make_ic(_prev_csv)
    W_0 = W_from_U_h_batch(jnp.asarray(U_h_0), fluid_g, fluid_l)
    print("Building solver (first JIT compilation may take a minute)...")
    step, compute_dt, residual, jac_fn, _jd, DBG = build(
        THETA0, use_theta=not ARGS.lib_closure)
    W_STAR, ssr, n, t_phys, stop_reason, WALL_SOLVE = march(
        step, compute_dt, W_0, _tol)
    print(f"\nStopped: {stop_reason}")
    print(f"  steps={n}  t={t_phys:.3e} s  SSR={ssr:.3e}  wall={WALL_SOLVE:.1f} s")
    U_STAR = np.asarray(U_h_from_W_batch(W_STAR))
    if not np.isfinite(U_STAR).all() or not np.isfinite(ssr) or n == 0:
        sys.exit(f"\nrung N={_nx} made no usable progress "
                 f"(steps={n}, SSR={ssr:.3e}, {stop_reason}).\n"
                 f"The ladder stops here rather than seeding the next mesh from a\n"
                 f"state that never converged. Try a finer first rung (a very\n"
                 f"coarse mesh can blow up on a cold start), a smaller CFL, or a\n"
                 f"seed via ic.run + --seed.")
    _csv = OUT / (f"solution_N{_nx}.csv" if len(LADDER) > 1 else "solution.csv")
    write_solution(_csv, U_STAR)
    LADDER_LOG.append({"Nx": _nx, "ssr": ssr, "ssr_target": _tol, "steps": n,
                       "wall_seconds": WALL_SOLVE, "stop_reason": stop_reason,
                       "solution": _csv.name, "analysed": _last})
    _prev_csv = _csv

if len(LADDER) > 1:                    # the analysed mesh, also under the plain name
    write_solution(OUT / "solution.csv", U_STAR)
    print("\nladder summary:")
    for r in LADDER_LOG:
        print(f"  N={r['Nx']:<5} SSR={r['ssr']:.3e} (target {r['ssr_target']:.0e})"
              f"  {r['steps']:>6} steps  {r['wall_seconds']:>7.1f} s"
              f"  {'<- differentiated' if r['analysed'] else ''}")

CONVERGED = ssr < SSR_TRUSTED
if not CONVERGED:
    print(f"\n  ** WARNING: SSR={ssr:.2e} is above {SSR_TRUSTED:.0e}. Sensitivities "
          f"from this state carry the iteration error, and the elasticities below "
          f"may be wrong by tens of percent. **")

U_STAR = np.asarray(U_h_from_W_batch(W_STAR))
W_int0 = W_STAR[1:-1].reshape(-1)
W_full = W_STAR.reshape(-1)
ag_int = U_STAR[1:-1, 3]

print(f"  solution -> {OUT / 'solution.csv'}")


# =============================================================================
# §JACOBIAN — field sensitivity dW/dln(theta) by the implicit function theorem
# =============================================================================
banner("FIELD JACOBIAN — implicit differentiation at the converged state")

PARITY_REL, PARITY_MSG = check_closure_parity(W_STAR)
print(f"[parity] theta-closure vs library residual: rel={PARITY_REL:.3e}  {PARITY_MSG}")
if PARITY_MSG == "FAIL":
    sys.exit("Closure parity FAILED — this script's closure is not the library's. "
             "Every number below would describe the wrong model.")

dt_cfl = float(compute_dt(W_STAR))
DT_SS = ARGS.dt_factor * dt_cfl
print(f"  dt_CFL={dt_cfl:.3e}  dt_steady={DT_SS:.3e} (x{ARGS.dt_factor:g})")


def theta_vec(th_arr):
    """Rebuild the theta dict from a flat LOG-parameter array.

    Log-parameterization is not cosmetic: dJ/dln(theta) = theta * dJ/d(theta),
    which normalized by J gives the elasticity — dimensionless, and comparable
    across parameters whose raw magnitudes differ by twelve decades.  Comparing
    raw dJ/dtheta across different units is the standard way this study produces
    a confidently wrong ranking.
    """
    return {nm: jnp.exp(th_arr[i]) for i, nm in enumerate(PNAMES)}


LOGTHETA0 = jnp.array([np.log(p.value) for p in PARAMS])


def R_full_of(lth, W_int_flat, W_full_flat, dtv):
    """Square Newton residual: with eos_rows=True the system is
    concat([balance (6/cell), eos (2/cell)]) against 8 unknowns per cell.

    Using the balance rows alone leaves 6*Nc equations for 8*Nc unknowns; the
    min-norm dW/dtheta is then arbitrary in a 2*Nc-dimensional null space and the
    'sensitivities' are truncation noise that swings and changes sign across an
    rcond sweep.  This cost hours to find once; do not 'simplify' it away.
    """
    *_, res, _jf, _jd, d = build(theta_vec(lth))
    return jnp.concatenate([res(W_int_flat, W_full_flat, dtv),
                            d["eos_residual"](W_int_flat)])


print("  forward-AD dR/dln(theta) (rebuilds the solver under trace)...")
R_TH = np.asarray(jax.jacfwd(R_full_of, argnums=0)(
    LOGTHETA0, W_int0, W_full, jnp.float64(DT_SS)))          # (8Nc, n_param)
_n6 = 6 * Nc
print(f"    shape={R_TH.shape}   balance rows |dR|={np.abs(R_TH[:_n6]).max():.3e}  "
      f"EOS rows={np.abs(R_TH[_n6:]).max():.3e} (must be 0: no theta in them)")

REACH = {}
for j, nm in enumerate(PNAMES):
    col = R_TH[:, j]
    if np.any(~np.isfinite(col)):
        REACH[nm] = "NaN/Inf in dR/dtheta"
    elif np.max(np.abs(col)) == 0.0:
        REACH[nm] = "inert (never enters this regime)"
    else:
        REACH[nm] = "active"
    print(f"    {nm:>6}: |dR/dln{nm}|_inf = {np.max(np.abs(col)):.4e}   {REACH[nm]}")
ACTIVE = [nm for nm in PNAMES if REACH[nm] == "active"]
if not ACTIVE:
    sys.exit("No parameter reaches the physics — check mass_transfer and the closure.")

# ── Scaled square solve.  Column scaling by w_ref plus row equilibration mirrors
#    the Newton body: both are exact for a square system (they change
#    conditioning, not the answer), and without them cond(J) ~ 1e16 sits at the
#    edge of float64 and the solve is meaningless.
w_ref = np.maximum(np.abs(np.asarray(W_int0)), np.asarray(DBG["w_char"]))
J = np.concatenate([np.asarray(jac_fn(W_int0, W_full, jnp.float64(DT_SS))),
                    np.asarray(DBG["eos_jac_dense"](W_int0))], axis=0)
J_sc = J * w_ref[None, :]
r_sc = 1.0 / (np.max(np.abs(J_sc), axis=1) + 1e-300)
J_sc, R_sc = J_sc * r_sc[:, None], R_TH * r_sc[:, None]
print(f"  system: {J_sc.shape[0]} x {J_sc.shape[1]} "
      f"({'square' if J_sc.shape[0] == J_sc.shape[1] else 'RECTANGULAR'})")

_t = _time.perf_counter()
dW = (-np.linalg.solve(J_sc, R_sc)) * w_ref[:, None]          # (8Nc, n_param)
print(f"  LU solve: {_time.perf_counter()-_t:.1f} s")

RCOND_ROWS, SVALS = [], None
if ARGS.rcond_sweep:
    print("  rcond sweep (SVD)...")
    Us, SVALS, Vt = np.linalg.svd(J_sc, full_matrices=False)
    for rc in (1e-6, 1e-8, 1e-10, 1e-12, 1e-14):
        keep = SVALS > rc * SVALS[0]
        y = -(Vt[keep].T @ ((Us[:, keep].T @ R_sc) / SVALS[keep, None]))
        RCOND_ROWS.append((rc, int(keep.sum()), y * w_ref[:, None]))
    print(f"    cond={SVALS[0]/SVALS[-1]:.3e}")


# =============================================================================
# §GATE — AD vs central FD of the RESIDUAL (no re-solve)
# =============================================================================
banner("VALIDATION — AD vs central FD of dR/dln(theta)  [residual level]")
print("This is NOT the hand-off's Phase 3 gate: that needs 2*n_param*3 CONVERGED")
print("SOLVES and is unreachable at one simulation.  It validates the theta-path")
print("through the closure — dead-branch NaN, stray stop_gradient, disconnected")
print("parameter, branch crossed between stencil points — but not the linear solve.")
print(f"\n{'param':>7} {'h':>8} {'|AD|inf':>13} {'|FD|inf':>13} {'rel.diff':>11}")
GATE_ROWS = []
for j, nm in enumerate(PNAMES):
    if REACH[nm] != "active":
        GATE_ROWS.append((nm, None, None, None, None, REACH[nm]))
        print(f"{nm:>7} {'-':>8} {'-':>13} {'-':>13} {'-':>11}   {REACH[nm]}")
        continue
    for h in (1e-2, 1e-3, 1e-4):
        ep = np.zeros(len(PNAMES)); ep[j] = h
        rp = np.asarray(R_full_of(LOGTHETA0 + jnp.asarray(ep), W_int0, W_full,
                                  jnp.float64(DT_SS)))
        rm = np.asarray(R_full_of(LOGTHETA0 - jnp.asarray(ep), W_int0, W_full,
                                  jnp.float64(DT_SS)))
        fd = (rp - rm) / (2 * h)
        ad = R_TH[:, j]
        den = max(np.abs(ad).max(), 1e-300)
        rel = float(np.abs(fd - ad).max() / den)
        GATE_ROWS.append((nm, h, float(np.abs(ad).max()), float(np.abs(fd).max()),
                          rel, "ok" if rel < 1e-3 else "CHECK"))
        print(f"{nm:>7} {h:>8.0e} {np.abs(ad).max():>13.5e} "
              f"{np.abs(fd).max():>13.5e} {rel*100:>10.4f}%")
print("\nPass = the relative difference falls as h shrinks until round-off. A value")
print("that does NOT fall means a branch was crossed between the stencil points.")


# =============================================================================
# §QOI — pure functions of W, differentiated by JVP along the sensitivity
# =============================================================================
banner("QUANTITIES OF INTEREST")
_x_int = x[1:-1]
_A_int = A_j[1:-1]
_dx_int = jnp.asarray(np.gradient(_x_int))

# The front index is chosen ONCE from the baseline state, then the crossing is
# located by linear interpolation between the two bracketing cells.  The
# derivative flows through the two alpha values; the index itself is frozen, so
# this is a valid LOCAL sensitivity that becomes a kink only if the front moves a
# full cell.  argmax would have given zero gradient almost everywhere.
_cross = np.flatnonzero((ag_int[:-1] < 0.5) & (ag_int[1:] >= 0.5))
I_FRONT = int(_cross[0]) if _cross.size else None
if I_FRONT is None:
    print("  NOTE: alpha_g never crosses 0.5 "
          f"(range {ag_int.min():.4f}..{ag_int.max():.4f}) — x_front unavailable.")


def qois(W_flat):
    Wm = W_flat.reshape(Nc, 8)
    p_, ag_, mg_, ug_, ml_, ul_ = (Wm[:, 0], Wm[:, 1], Wm[:, 2],
                                   Wm[:, 3], Wm[:, 4], Wm[:, 5])
    mdot_x = (mg_ * ug_ + ml_ * ul_) * _A_int
    dpdx = jnp.gradient(p_) / _dx_int
    out = {
        "mdot": jnp.mean(mdot_x),                       # mean station: the exit
                                                        # cell is contaminated
        "x_exit": mg_[-1] * ug_[-1] / (mg_[-1] * ug_[-1] + ml_[-1] * ul_[-1]),
        "slip_exit": ug_[-1] / ul_[-1],
        "p_exit": p_[-1],
        # Smooth, mesh-convergent steepness that does not require resolving the
        # front — unlike max|dp/dx|, which JST dissipation sets.
        "delta_p": (p_[0] - p_[-1])**2 / jnp.sum(dpdx**2 * _dx_int),
        "max_dpdx": jax.scipy.special.logsumexp(jnp.abs(dpdx)) ,
    }
    if I_FRONT is not None:
        a0, a1 = ag_[I_FRONT], ag_[I_FRONT + 1]
        out["x_front"] = (_x_int[I_FRONT] + (0.5 - a0)
                          * (_x_int[I_FRONT + 1] - _x_int[I_FRONT]) / (a1 - a0))
    return out


# L_relax = u_mix * tau at the throat, from the closure itself.  Analytically
# differentiable and the anchor for the §7.1 hand-check.
def _relax_length(lth):
    th = theta_vec(lth)
    Uh = jnp.asarray(U_STAR[I_THROAT])
    p_, ag_, h_g, h_l = Uh[0], Uh[3], Uh[4], Uh[5]
    stg, stl = _transport(p_, h_g, h_l)
    ai, d = interfacial_area(ag_, alpha_b=LIB_ALPHA_B, alpha_d=LIB_ALPHA_D,
                             Nb=th["N_b"], Nd=th["N_d"], d_min_b=LIB_D_MIN_B)
    htc_l = th["Nu_c"] * ranz_marshall_htc_two_fluid(
        {"rho": stl["rho"][0], "mu": stl["mu"][0], "k": stl["k"][0],
         "cp": stl["cp"][0]}, jnp.abs(Uh[1] - Uh[2]), d)
    tau = ((1 - ag_) * stl["rho"][0] * stl["cp"][0]) / jnp.maximum(htc_l * ai, 1e-30)
    return 0.5 * (Uh[1] + Uh[2]) * tau


Q0 = {k: float(v) for k, v in qois(W_int0).items()}
Q0["L_relax"] = float(_relax_length(LOGTHETA0))
QNAMES = [k for k in QOI_LABELS if k in Q0]

# d(QoI)/d(ln theta_j): one JVP per parameter along the sensitivity direction.
S = np.zeros((len(QNAMES), len(PNAMES)))            # elasticities
DQ = np.zeros((len(QNAMES), len(PNAMES)))           # absolute derivatives
for j in range(len(PNAMES)):
    _, tang = jax.jvp(qois, (W_int0,), (jnp.asarray(dW[:, j]),))
    for i, q in enumerate(QNAMES):
        if q == "L_relax":
            continue
        DQ[i, j] = float(tang[q])
_iL = QNAMES.index("L_relax")
DQ[_iL, :] = np.asarray(jax.jacfwd(_relax_length)(LOGTHETA0))
for i, q in enumerate(QNAMES):
    S[i, :] = DQ[i, :] / (Q0[q] if Q0[q] != 0 else np.nan)

print(f"{'QoI':<12} {'value':>14}   " + " ".join(f"{n:>10}" for n in PNAMES))
for i, q in enumerate(QNAMES):
    # Mask on the ELASTICITY, not on residual reachability: L_relax depends on
    # theta directly, so it stays sensitive to parameters that cannot move the
    # solution field (gate shut) and must not be reported as n/a.
    print(f"{q:<12} {Q0[q]:>14.6e}   "
          + " ".join(f"{S[i,j]:>10.3e}" if S[i, j] != 0.0 and np.isfinite(S[i, j])
                     else f"{'n/a':>10}" for j in range(len(PNAMES))))

# ── Variance budget (linearized Sobol) + SVD model reduction ─────────────────
VAR = np.zeros_like(S)
for i in range(len(QNAMES)):
    c = (S[i, :] * SIGMA)**2
    VAR[i, :] = c / c.sum() if c.sum() > 0 else 0.0
SIG_LNJ = np.sqrt(np.nansum((S * SIGMA[None, :])**2, axis=1))

# Rows normalized by local state magnitude so the mix of pressure, velocity and
# enthalpy rows is dimensionless before the SVD.  Small singular values are then
# parameter combinations that do not move the solution ANYWHERE — with no data
# this is model reduction, not identifiability.
_dU_scaled = (dW / w_ref[:, None]) * SIGMA[None, :]
_Usv, SV, VT = np.linalg.svd(_dU_scaled, full_matrices=False)

# ── Gamma == 0 detector (hand-off §9) ────────────────────────────────────────
GAMMA_NOTE = ""
if cfg.MASS_TRANSFER:
    _T_l_int = np.asarray(fluid_l.get_state(
        jxp.HmassP_INPUTS, jnp.asarray(U_STAR[1:-1, 5]),
        jnp.asarray(U_STAR[1:-1, 0]))["temperature"])
    _sat = fluid_l.get_state_saturation(jnp.asarray(U_STAR[1:-1, 0]))
    _w = np.asarray(evaporation_weight(
        jnp.asarray(U_STAR[1:-1, 5]), jnp.asarray(_T_l_int),
        jnp.asarray(_sat["enthalpy"]), jnp.asarray(_sat["temperature"]),
        jnp.asarray(ag_int), gate_dT=LIB_GATE_DT, gate_dh=LIB_GATE_DH))
    _frac = float(np.mean(_w > 0))
    if _frac == 0.0:
        GAMMA_NOTE = ("The evaporation gate is CLOSED in every cell, so Gamma == 0 "
                      "on this state and every mass-transfer closure has exactly "
                      "zero sensitivity. That is correct behaviour, not evidence "
                      "that these parameters are unimportant.")
    else:
        GAMMA_NOTE = f"Evaporation gate open in {_frac*100:.1f}% of cells."
    print(f"\n[Gamma] {GAMMA_NOTE}")

_cens = (int((ag_int <= LIB_ALPHA_B).sum()),
         int(((ag_int > LIB_ALPHA_B) & (ag_int < LIB_ALPHA_D)).sum()),
         int((ag_int >= LIB_ALPHA_D).sum()))
REGIME_NOTE = (f"alpha_g in {ag_int.min():.4f}..{ag_int.max():.4f}; of {Nc} interior "
               f"cells {_cens[0]} bubbly, {_cens[1]} transition, {_cens[2]} droplet.")
print(f"[regime] {REGIME_NOTE}")


# =============================================================================
# §OUTPUT
# =============================================================================
banner("OUTPUTS")
np.savez_compressed(OUT / "jacobian.npz", dW_dlntheta=dW, w_ref=w_ref,
                    param_names=np.array(PNAMES), sigma=SIGMA, x=x, A=A,
                    U_star=U_STAR, singular_values=SV, right_vectors=VT,
                    ssr=ssr, nx=cfg.Nx)

with open(OUT / "sensitivity.csv", "w", encoding="utf-8") as f:
    f.write("qoi,value,units,parameter,theta0,sigma_ln,"
            "dQ_dlntheta,elasticity,elasticity_x_sigma,variance_fraction,status\n")
    for i, q in enumerate(QNAMES):
        for j, nm in enumerate(PNAMES):
            f.write(f"{q},{Q0[q]:.9e},{QOI_LABELS[q][1]},{nm},{PARAMS[j].value:.9e},"
                    f"{SIGMA[j]:.4f},{DQ[i,j]:.9e},{S[i,j]:.9e},"
                    f"{S[i,j]*SIGMA[j]:.9e},{VAR[i,j]:.6f},{REACH[nm]}\n")

META = {
    "generated": _dt.datetime.now().isoformat(timespec="seconds"),
    "git_commit": _git_commit(),
    "config": str(_cfg_path), "case": CASE,
    "differentiation": "Option B — implicit function theorem at the converged state",
    "validation_gate": "NOT RUN (single-simulation budget). Residual-level AD-vs-FD only.",
    "mesh": {"Nx": cfg.Nx, "x1": cfg.x1, "x2": cfg.x2, "dx": dx,
             "geometry": cfg.GEO_FUNCTION},
    "solve": {"steps": n, "t_phys": t_phys, "ssr": ssr, "ssr_tol": cfg.SSR_TOL,
              "stop_reason": stop_reason, "wall_seconds": WALL_SOLVE,
              "converged_for_sensitivity": bool(CONVERGED),
              "ssr_trusted_threshold": SSR_TRUSTED,
              "ladder": LADDER, "ladder_rungs": LADDER_LOG,
              "ic_mode": IC_MODE, "ic_source": IC_SRC,
              "ic_seed_run": (str(_rst) if IC_MODE == "restart" else None)},
    "numerics": {"dt_cfl": dt_cfl, "dt_steady": DT_SS,
                 "dt_factor": ARGS.dt_factor, "eos_rows": bool(cfg.EOS_ROWS),
                 "bc_inlet": cfg.BC_INLET, "bc_outlet": cfg.BC_OUTLET},
    "closure_parity": {"relative": PARITY_REL, "verdict": PARITY_MSG},
    "theta": {p.name: {"value": p.value, "units": p.units,
                       "sigma_ln": p.sigma, "kind": p.kind,
                       "provenance": p.provenance, "enters": p.enters,
                       "status": REACH[p.name]} for p in PARAMS},
    "qoi": {q: {"label": QOI_LABELS[q][0], "units": QOI_LABELS[q][1],
                "value": Q0[q], "sigma_ln_propagated": float(SIG_LNJ[i]),
                "exit_plane_caveat": q in EXIT_PLANE_QOIS}
            for i, q in enumerate(QNAMES)},
    "notes": {"gamma": GAMMA_NOTE, "regime": REGIME_NOTE},
    "wall_seconds_total": _time.perf_counter() - _t_start,
}
(OUT / "run_metadata.json").write_text(json.dumps(META, indent=2), encoding="utf-8")

# ── Figures + HTML report ────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

hyp.set_plot_options(fontsize=10, grid=False)
matplotlib.rcParams["savefig.dpi"] = 140

# One fixed colour per PARAMETER across every figure — identity must not move
# when a figure shows a different subset.
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
       "#4a3aa7", "#e34948"]
PCOL = {nm: PAL[i % len(PAL)] for i, nm in enumerate(PNAMES)}
DASH = [(None, None), (5, 2), (1.4, 1.4), (7, 2, 1.4, 2), (4, 1.4, 1.4, 1.4), (2.4, 1.6)]
PDASH = {nm: DASH[i % len(DASH)] for i, nm in enumerate(PNAMES)}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
POS, NEG = "#2a78d6", "#d03b3b"


def _b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _active_for(i):
    """Parameter indices with a non-zero elasticity for QoI i, ranked.

    Reachability is decided PER QoI, not once from dR/dtheta.  L_relax is a
    property of the closure evaluated at a station, so it depends on theta
    directly and stays sensitive even where the evaporation gate is shut and the
    parameter cannot move the solution field at all.  Filtering it by the
    residual-level reach would hide a real, exactly-known elasticity.
    """
    v = np.nan_to_num(S[i, :] * SIGMA)
    return [j for j in np.argsort(-np.abs(v)) if S[i, j] != 0.0 and np.isfinite(S[i, j])]


def fig_tornado(i):
    """Signed elasticity x prior sigma, sorted by magnitude. Signed matters: that
    one parameter moves the front upstream while another moves it downstream is
    half the physical content."""
    vals = S[i, :] * SIGMA
    order = _active_for(i)
    fig, ax = plt.subplots(figsize=(7.2, 0.45 * max(len(order), 3) + 1.4))
    if not order:
        ax.text(0.5, 0.5, "no active parameter for this QoI", ha="center",
                va="center", transform=ax.transAxes, color=MUTED)
        ax.axis("off")
        return fig
    _span = max(np.abs([vals[j] for j in order]).max(), 1e-30)
    ypos = np.arange(len(order))[::-1]
    ax.barh(ypos, [vals[j] for j in order], height=0.6,
            color=[POS if vals[j] >= 0 else NEG for j in order])
    ax.set_yticks(ypos, [PNAMES[j] for j in order])
    for yp, j in zip(ypos, order):
        ax.text(vals[j] + (0.02 if vals[j] >= 0 else -0.02) * _span,
                yp, f"{VAR[i,j]*100:.0f}%", va="center",
                ha="left" if vals[j] >= 0 else "right", fontsize=9, color=INK2)
    ax.axvline(0, color="#c3c2b7", lw=1)
    ax.set_xlabel(r"signed elasticity $\times$ prior $\sigma$   [-]")
    ax.margins(x=0.18)
    return fig


def fig_pareto(i):
    vals = VAR[i, :] * 100
    order = _active_for(i)
    vals = np.nan_to_num(vals)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(range(len(order)), [vals[j] for j in order],
           color=[PCOL[PNAMES[j]] for j in order], width=0.62)
    ax.set_xticks(range(len(order)), [PNAMES[j] for j in order])
    # Bars and the cumulative line are both percentages, so they share ONE axis.
    # A twin y-scale here would be pure decoration and would invite reading the
    # line against the wrong ticks.
    ax.set_ylabel("% of $\\sigma^2_{\\ln J}$   (bars and cumulative)")
    cum = np.cumsum([vals[j] for j in order])
    ax.plot(range(len(order)), cum, color=INK, lw=1.4, marker="o", ms=4,
            label="cumulative")
    ax.axhline(90, color=MUTED, lw=0.9, ls=(0, (3, 3)))
    ax.text(len(order) - 0.5, 91, "90% cut ", fontsize=8, color=MUTED, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(loc="center right", fontsize=8)
    return fig


def fig_spatial(i, top=4):
    """d(p)/d(ln theta) against x, beneath the baseline pressure on a shared
    axis. This is the figure that yields physics rather than ranking: it shows
    whether a parameter acts upstream, at the throat, or in the relaxation
    zone."""
    order = _active_for(i)[:top]
    dp_span = max(abs(U_STAR[:, 0].max() - U_STAR[:, 0].min()), 1e-30)
    fig, axs = plt.subplots(2, 1, figsize=(7.6, 5.4), sharex=True,
                            gridspec_kw=dict(height_ratios=[1, 1.7], hspace=0.1))
    axs[0].plot(x, U_STAR[:, 0] / 1e5, color=INK, lw=1.4)
    axs[0].set_ylabel("$p$ [bar]")
    axs[1].axhline(0, color="#c3c2b7", lw=1)
    for j in order:
        dp = dW[:, j].reshape(Nc, 8)[:, 0]
        axs[1].plot(_x_int, dp / dp_span, color=PCOL[PNAMES[j]],
                    dashes=PDASH[PNAMES[j]], lw=1.6, label=PNAMES[j])
    axs[1].set_ylabel(r"$\partial p/\partial\ln\theta \;/\; \Delta p$  [-]")
    axs[1].set_xlabel("$x$ [m]")
    axs[1].legend(loc="best", fontsize=8)
    for a in axs:
        a.axvline(x[I_THROAT], color=MUTED, lw=0.9, ls=(0, (3, 3)))
        if I_FRONT is not None:
            a.axvline(_x_int[I_FRONT], color="#4a3aa7", lw=0.9, ls=(0, (1.5, 1.5)))
        a.set_xlim(x[0], x[-1])
    return fig


def fig_svd():
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.8),
                            gridspec_kw=dict(width_ratios=[1, 1.25]))
    axs[0].semilogy(np.arange(1, SV.size + 1), SV / SV[0], color=INK,
                    lw=1.4, marker="o", ms=5)
    axs[0].set_xlabel("index"); axs[0].set_ylabel(r"$\sigma_i/\sigma_1$ [-]")
    axs[0].set_title("singular spectrum of the $\\sigma$-scaled field Jacobian",
                     fontsize=9, color=INK2)
    k = min(3, VT.shape[0])
    M = VT[:k, :]
    lim = np.abs(M).max()
    im = axs[1].imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    axs[1].set_xticks(range(len(PNAMES)), PNAMES, rotation=30, ha="right")
    axs[1].set_yticks(range(k), [f"v{i+1}" for i in range(k)])
    axs[1].set_title("leading right singular vectors (parameter loadings)",
                     fontsize=9, color=INK2)
    for a in range(k):
        for b in range(len(PNAMES)):
            axs[1].text(b, a, f"{M[a,b]:+.2f}", ha="center", va="center",
                        fontsize=8, color=INK if abs(M[a, b]) < 0.6 * lim else "white")
    fig.colorbar(im, ax=axs[1], fraction=0.03)
    return fig


print("  rendering figures...")
FIGS = {f"tornado_{q}": _b64(fig_tornado(i)) for i, q in enumerate(QNAMES)}
FIGS.update({f"pareto_{q}": _b64(fig_pareto(i)) for i, q in enumerate(QNAMES)})
FIGS.update({f"spatial_{q}": _b64(fig_spatial(i)) for i, q in enumerate(QNAMES)})
FIGS["svd"] = _b64(fig_svd())


def _headline(i, q):
    order = _active_for(i)
    if not order:
        return "No parameter is active for this QoI in this regime."
    cum, k = 0.0, 0
    for k, j in enumerate(order, 1):
        cum += VAR[i, j]
        if cum >= 0.9:
            break
    return (f"<b>{QOI_LABELS[q][0]}</b>: &sigma; = &plusmn;{SIG_LNJ[i]*100:.0f}% "
            f"(1&sigma;, propagated from priors). {VAR[i,order[0]]*100:.0f}% of that "
            f"variance comes from <code>{PNAMES[order[0]]}</code>; {k} parameter"
            f"{'s' if k > 1 else ''} explain {cum*100:.0f}%.")


def _tbl(rows, head):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


_sec = []
for i, q in enumerate(QNAMES):
    warn = ""
    if q in EXIT_PLANE_QOIS:
        warn = ("<div class='warn'>Exit-plane QoI. On this solver AD and central FD "
                "field sensitivities agree over the whole nozzle <em>except</em> the "
                "last few cells at the outflow, where AD violates the x-independence "
                "that steady mass conservation forces on d(mdot)/d&theta;. This QoI "
                "samples exactly that region — treat its elasticities as indicative.</div>")
    if q == "max_dpdx":
        warn += ("<div class='warn'>Diagnostic only. If the front is smeared over a "
                 "few cells by JST dissipation this number is set by the scheme, and "
                 "its derivative measures how the parameters push the front through a "
                 "fixed dissipation stencil. Never rank on it.</div>")
    _sec.append(f"""
<section><h2>QoI — {QOI_LABELS[q][0]}</h2>
<p class="headline">{_headline(i, q)}</p>
<p class="meta">value = {Q0[q]:.6g} {QOI_LABELS[q][1]}</p>{warn}
<h3>Tornado — signed elasticity &times; prior &sigma;</h3>
<img src="data:image/png;base64,{FIGS[f'tornado_{q}']}">
<h3>Pareto variance budget</h3>
<img src="data:image/png;base64,{FIGS[f'pareto_{q}']}">
<h3>Spatial sensitivity profile</h3>
<img src="data:image/png;base64,{FIGS[f'spatial_{q}']}">
<p class="meta">Dashed grey = throat. Dotted violet = &alpha;<sub>g</sub>=0.5 front.</p>
</section>""")

_gate_tbl = _tbl([(r[0], "—" if r[1] is None else f"{r[1]:.0e}",
                   "—" if r[2] is None else f"{r[2]:.4e}",
                   "—" if r[3] is None else f"{r[3]:.4e}",
                   "—" if r[4] is None else f"{r[4]*100:.4f}%", r[5])
                  for r in GATE_ROWS],
                 ["parameter", "h", "|AD dR/dlnθ|∞", "|FD|∞", "rel. diff", "status"])
_prior_tbl = _tbl([(p.name, f"{p.value:.4g}", p.units, f"{p.sigma:.2f}",
                    REACH[p.name], p.enters, p.provenance) for p in PARAMS],
                  ["parameter", "θ₀", "units", "σ(ln θ)", "status", "enters via",
                   "provenance"])
_deg = []
for a in range(len(PNAMES)):
    for b in range(a + 1, len(PNAMES)):
        if REACH[PNAMES[a]] != "active" or REACH[PNAMES[b]] != "active":
            continue
        ca, cb = _dU_scaled[:, a], _dU_scaled[:, b]
        na, nb = np.linalg.norm(ca), np.linalg.norm(cb)
        if na > 0 and nb > 0:
            cosang = float(abs(ca @ cb) / (na * nb))
            if cosang > 0.98:
                _deg.append((PNAMES[a], PNAMES[b], f"{cosang:.4f}"))
_deg_html = (_tbl(_deg, ["parameter A", "parameter B", "|cos angle|"]) if _deg
             else "<p>No pair of active parameters has near-parallel columns "
                  "(|cos| &gt; 0.98); each is an independent direction.</p>")

HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Closure sensitivity — {CASE} N{cfg.Nx}</title><style>
body{{font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;max-width:1000px;
margin:2rem auto;padding:0 1.2rem;color:#0b0b0b;background:#fcfcfb}}
h1{{font-size:1.6rem;margin-bottom:.2rem}} h2{{font-size:1.25rem;margin-top:2.4rem;
border-bottom:2px solid #e1e0d9;padding-bottom:.3rem}} h3{{font-size:1rem;color:#52514e;
margin-top:1.5rem}} img{{max-width:100%;height:auto;display:block;margin:.6rem 0}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:.6rem 0;
display:block;overflow-x:auto}} th,td{{border:1px solid #e1e0d9;padding:5px 8px;
text-align:left}} th{{background:#f0efec}} code{{background:#f0efec;padding:1px 4px;
border-radius:3px;font-size:.9em}}
.headline{{font-size:1.05rem;background:#f0efec;padding:.8rem 1rem;border-radius:6px;
border-left:4px solid #2a78d6}}
.meta{{color:#52514e;font-size:13px}}
.banner{{background:#fdf0e8;border:2px solid #eb6834;border-radius:6px;padding:1rem 1.2rem;
margin:1.2rem 0}}
.warn{{background:#fdf6e8;border-left:4px solid #eda100;padding:.6rem .9rem;
margin:.7rem 0;font-size:14px}}
</style></head><body>
<h1>AD closure-parameter sensitivity study</h1>
<p class="meta">{CASE} &middot; Nx={cfg.Nx} &middot; git {META['git_commit']} &middot;
{META['generated']} &middot; solve {WALL_SOLVE:.0f} s, {n} steps, SSR {ssr:.2e}</p>

<div class="banner"><b>VALIDATION GATE NOT RUN.</b> The hand-off's Phase&nbsp;3 gate
requires 2&times;n<sub>param</sub>&times;3 <em>converged solves</em>; this study has a
one-simulation budget, so it runs in the equivalent of <code>--no-gate</code> and
<b>no ranking below has passed the gate</b>. What <em>was</em> run is the same
comparison one level down — AD versus central FD of the residual
&part;R/&part;ln&theta; (Appendix&nbsp;A) — which costs function evaluations rather
than solves. That covers a dead branch, a stray <code>stop_gradient</code>, a
disconnected parameter, and a switch crossed between stencil points; it does
<em>not</em> cover an error in the linear solve. DGSM (&sect;7.5) and the mesh
study (&sect;8) are multi-solve and were not run.
{"" if CONVERGED else f"<br><br><b>Additionally: the solve stopped at SSR {ssr:.2e}, above the {SSR_TRUSTED:.0e} threshold at which these sensitivities have been observed to be reliable. On this solver an SSR 1e-8 state gave an elasticity within 1.6% of finite differences while an SSR 2.9e-6 state of the same case was off by 19.6%. Treat every number here as provisional.</b>"}
</div>

<section><h2>How to read this report</h2>
<p>An <b>elasticity</b> S = &part;ln&nbsp;J/&part;ln&nbsp;&theta; is dimensionless:
S = 0.5 means a 1% rise in the parameter moves the quantity by 0.5%. Bars are
ranked by <b>|S|&nbsp;&times;&nbsp;&sigma;</b>, not |S|, because a parameter known
to a decade matters more than an equally elastic one known to 10%. With no
experimental data <b>the prior &sigma; is doing all the work</b> — two people with
different priors get different rankings and both are correct, so Appendix&nbsp;B
records where every &sigma; came from. Everything here is <b>local</b>: a gradient
at &theta;<sub>0</sub>, extrapolated across a prior box that is decades wide.</p>
<p class="meta">{REGIME_NOTE}{(" " + GAMMA_NOTE) if GAMMA_NOTE else ""}</p></section>

{"".join(_sec)}

<section><h2>Model reduction (SVD)</h2>
<p>Columns of the field Jacobian scaled by the prior &sigma;, rows normalized by
local state magnitude. Without data this is <b>not</b> identifiability: a small
singular value means that parameter combination does not move the solution
<em>anywhere in the nozzle</em>, so the dimension can be deleted from the model.</p>
<img src="data:image/png;base64,{FIGS['svd']}">
<h3>Near-degenerate pairs</h3>{_deg_html}</section>

<section><h2>Appendix A — residual-level AD vs FD</h2>
<p>Central difference in log space at h &isin; {{1e-2, 1e-3, 1e-4}} against
forward-mode AD, both of &part;R/&part;ln&theta;. Pass = the relative difference
falls with h until round-off. One that does <em>not</em> fall means a branch was
crossed between the stencil points.</p>{_gate_tbl}
<p class="meta">Closure parity (this script's &theta;-parametrized closure vs the
library's, at &theta;<sub>0</sub>): rel = {PARITY_REL:.3e} &mdash; {PARITY_MSG}.</p></section>

<section><h2>Appendix B — priors and provenance</h2>{_prior_tbl}
<p class="meta">Parameters marked <code>inert</code> never enter the physics in this
regime, so their sensitivity is exactly zero as a structural fact, not a small
number. <code>Nu_d</code> is the hard-coded <code>Nu_g = 120</code> at
sources.py:260; <code>Nu_c</code> and <code>C_D</code> are multipliers this script
introduces because the library exposes no such knob.</p></section>

<section><h2>Appendix C — what was not run</h2><ul>
<li><b>Phase 3 FD gate</b> — needs converged re-solves; out of budget.</li>
<li><b>DGSM global sweep</b> — needs a Latin-hypercube of solves. This matters
here: the prior on N<sub>b</sub> is ~1 decade, the closure switches branch on
&alpha;<sub>g</sub>, and a parameter can be inert at baseline and dominant a
decade away.</li>
<li><b>Mesh convergence of the sensitivities</b> — needs a second and third mesh.
Sensitivities converge more slowly than the state.</li>
<li><b>Numerical parameters</b> (k<sub>2</sub>, k<sub>4</sub>, &epsilon;<sub>Harten</sub>)
— <code>eps_void</code> sits behind a Python-level branch in the solver factory and
is not traceable without editing the library.</li></ul></section>

<section><h2>Appendix D — solver diagnostics</h2>
<p class="meta">dt<sub>CFL</sub> = {dt_cfl:.3e} s, steady-Jacobian dt =
{DT_SS:.3e} s ({ARGS.dt_factor:g}&times;). Newton system
{J_sc.shape[0]}&times;{J_sc.shape[1]}, eos_rows={bool(cfg.EOS_ROWS)}.
{f"Scaled cond = {SVALS[0]/SVALS[-1]:.3e}." if SVALS is not None else
 "Run with --rcond-sweep for the conditioning and pseudo-inverse-cut sweep."}
Stop reason: {stop_reason}.</p></section>

</body></html>"""
(OUT / "report.html").write_text(HTML, encoding="utf-8")

for f in ("solution.csv", "jacobian.npz", "sensitivity.csv",
          "run_metadata.json", "report.html"):
    print(f"  {f}")
print(f"\nAll outputs -> {OUT}")
print(f"Total wall: {_time.perf_counter()-_t_start:.1f} s")
