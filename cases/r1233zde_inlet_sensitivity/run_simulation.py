"""
R1233zd(E) flashing expansion in the Elliot nozzle - six-equation two-fluid model.
=================================================================================

Stand-alone driver for this case.  It imports only the installed
`nozzle_hyperbolic` / `jaxprop` packages plus the local `case_config.py`; nothing
outside this folder is read or modified.

    python build_tables.py        # once - builds + verifies the EOS tables
    python run_simulation.py      # the simulation

Physics
-------
SATURATED LIQUID R1233zd(E) at 155 degC (p0 = p_sat = 29.83 bar) is expanded to
2 bar through the Elliot converging-diverging nozzle.  The inlet carries liquid
only; the vapour is a thin passive GHOST (alpha_g = 1e-3) seeded on its own
saturation line.  As the pressure falls the liquid becomes superheated
(metastable) and FLASHES, so the ghost grows into a real second phase.  Single
component, interphase mass transfer ON.

Numerics
--------
Staedtke six-equation single-pressure model in the enthalpy-primitive frame,
flux-vector splitting with a Harten entropy fix, implicit Newton time-stepping
towards steady state.  The settings in case_config.py are the flashing-certified
stack: E3-augmented banded system, well-balanced non-conservative term,
well-balanced area gradient, JST void filter, and both ghost pins.

Outputs (results/<run name>/)
-----------------------------
    run_log.txt                terminal transcript
    solution.csv               full two-phase profile
    final_solution.png         p, u, slip, T, alpha, Mach
    initial_condition.png      the IC drawn in the same format
    mass_transfer_source.png   flashing source + liquid superheat
"""

import os
import sys
import shutil
import pathlib
import datetime as _dt
import time as _time

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=true")

import jax
jax.config.update("jax_enable_x64", True)

_HERE = pathlib.Path(__file__).resolve().parent
_cache_dir = _HERE / ".jax_cache"
_cache_dir.mkdir(exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(_cache_dir))

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jaxprop as jxp
import nozzle_hyperbolic.fvm_solver.geometry as _geom
from nozzle_hyperbolic.fvm_solver.two_fluid import (
    make_two_fluid_step,
    W_from_U_h_batch,
    U_h_from_W_batch,
    V_from_W,
    phase_props,
    common_quantities,
)
from nozzle_hyperbolic.fvm_solver.two_fluid.sources import (
    two_fluid_closure,
    D_VE_algebric,
    D_VE_area,
)

sys.path.insert(0, str(_HERE))
import case_config as cfg
import CoolProp.CoolProp as _CP


def CP_sat_liquid_enthalpy(p):
    """h of saturated liquid at p [Pa] - used by the 'stagnation_saturated' IC."""
    return float(_CP.PropsSI("H", "P", float(p), "Q", 0, cfg.FLUID_NAME))


# ---------------------------------------------------------------------------
# Output directory + terminal tee
# ---------------------------------------------------------------------------
_stamp   = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = (f"{cfg.REPO_RUN}_to_{cfg.P_OUT/1e5:.0f}bar_N{cfg.NX}_"
            f"{cfg.RECON}_{_stamp}")
out_dir  = cfg.RESULTS_DIR / run_name
out_dir.mkdir(parents=True, exist_ok=True)
# Provenance: the settings file that produced this run, copied verbatim.
shutil.copy(cfg.SETTINGS_PATH, out_dir / cfg.SETTINGS_PATH.name)


class _Tee:
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

    def isatty(self):
        # tqdm (table generation) probes this; a tee is never a terminal.
        return False


_log_path = out_dir / "run_log.txt"
sys.stdout = _Tee(sys.stdout, _log_path)
sys.stderr = _Tee(sys.stderr, _log_path)

print(cfg.summary())
print(f"\nOutput directory: {out_dir}")

# ---------------------------------------------------------------------------
# EOS tables
# ---------------------------------------------------------------------------
print("\nLoading EOS tables ...")
_missing = [n for n in (f"{cfg.FLUID_NAME}_meta_liquid_{cfg.N_H}x{cfg.N_P}.pkl",
                        f"{cfg.FLUID_NAME}_meta_vapor_{cfg.N_H}x{cfg.N_P}.pkl")
            if not (cfg.TABLE_DIR / n).exists()]
if _missing:
    print(f"  NOTE: {', '.join(_missing)} not found - they will be generated now "
          f"(tens of minutes).  Run `python build_tables.py` first to build AND "
          f"verify them.")
fluid_g, fluid_l = cfg.build_fluids()

# Guard: CoolProp has no thermal-conductivity model for R1233zd(E), and the
# interfacial-HTC closure reads conductivity for both phases.  An unpatched
# table would make every heat/mass-transfer source NaN on the first step.
for _nm, _fl in (("liquid", fluid_l), ("vapour", fluid_g)):
    if not np.isfinite(np.asarray(_fl.table["conductivity"]["value"])).all():
        raise SystemExit(
            f"ERROR: the {_nm} table has non-finite thermal conductivity.\n"
            f"       CoolProp has no conductivity model for {cfg.FLUID_NAME}; the\n"
            f"       interfacial heat-transfer closure needs it and would produce\n"
            f"       NaN on the first step.  Run `python patch_conductivity.py`\n"
            f"       (or `python build_tables.py`, which does it automatically).")

print("  done.")

# ---------------------------------------------------------------------------
# Grid and geometry
# ---------------------------------------------------------------------------
x  = np.linspace(cfg.X1, cfg.X2, cfg.NX)
dx = float(x[1] - x[0])

_geo_fn = getattr(_geom, cfg.GEO_FUNCTION)
geo     = _geo_fn(x, roughness=cfg.GEO_ROUGHNESS, **cfg.GEO_KWARGS)
A      = np.asarray(geo.A, dtype=float)
dAdx   = np.array(geo.dAdx, dtype=float)     # copy: asarray gives a read-only view
A_face = 0.5 * (A[:-1] + A[1:])

# WELL-BALANCED area gradient.  The flux divergence uses FACE areas; using the
# analytic cell-centred dA/dx leaves a discrete imbalance wherever A(x) is curved
# that the solver absorbs as a 2*dx checkerboard in the void/slip subspace.  The
# face-consistent gradient makes the wall-pressure source cancel the smooth-flow
# flux divergence exactly.
dAdx[1:-1] = (A_face[1:] - A_face[:-1]) / dx

_i_throat = int(np.argmin(A))
print(f"\nGrid: Nx={cfg.NX}  dx={dx*1e3:.4f} mm  geometry={cfg.GEO_FUNCTION}")
print(f"  A: {A.min():.4e} .. {A.max():.4e} m2   "
      f"r: {np.sqrt(A.min()/np.pi)*1e3:.3f} .. {np.sqrt(A.max()/np.pi)*1e3:.3f} mm")
print(f"  throat at x = {x[_i_throat]:.5f} m (cell {_i_throat})   "
      f"A_in/A_th = {A[0]/A[_i_throat]:.3f}   A_exit/A_th = {A[-1]/A[_i_throat]:.3f}")

# ---------------------------------------------------------------------------
# Reservoir state - per-phase (p0, T) from the settings file
# ---------------------------------------------------------------------------
# case_config picks the right branch per phase: an ordinary (p,T) flash off the
# saturation line, the phase's own saturation call ON it (where a (p,T) flash is
# ambiguous), plus the ghost-vapour fallback when a subcooled reservoir would put
# h_g below the vapour table floor.  It reports which branch it used.
P0, H_L0, S_L0, H_G0, S_G0 = cfg.P0, cfg.H_L0, cfg.S_L0, cfg.H_G0, cfg.S_G0
RHO_L0, RHO_G0 = cfg.RHO_L0, cfg.RHO_G0
SLIP    = cfg.SLIP
ALPHA_G = cfg.VOID_FRACTION
QUALITY = cfg.QUALITY
dom, _dom_src = cfg.DOM, cfg.DOM_SOURCE

print(f"\nStagnation inlet:")
print(f"  p0 = {P0/1e5:.4f} bar   T_g0 = {cfg.T_G0:.2f} K   T_l0 = {cfg.T_L0:.2f} K"
      f"   (T_sat(p0) = {cfg.T_SAT_P0:.2f} K)")
print(f"  liquid: {cfg.RESERVOIR_BRANCH_L}")
print(f"          h_l0 = {H_L0/1e3:.2f} kJ/kg  s_l0 = {S_L0:.2f} J/kg/K  "
      f"rho_l0 = {RHO_L0:.2f} kg/m3")
print(f"  vapour: {cfg.RESERVOIR_BRANCH_G}")
print(f"          h_g0 = {H_G0/1e3:.2f} kJ/kg  s_g0 = {S_G0:.2f} J/kg/K  "
      f"rho_g0 = {RHO_G0:.2f} kg/m3")
print(f"  alpha_g = {ALPHA_G:.6e}   quality = {QUALITY:.6e}   slip = {SLIP:.3f}"
      f"   [{cfg.COMPOSITION_SOURCE}]")
print(f"  dominant phase = {dom}  [{_dom_src}]  ->  p closed on the {dom} isentrope")

# Guard: both reservoir enthalpies must be inside their tables.
for _nm, _h, _lo, _hi in (("h_l0", H_L0, cfg.H_LIQ_MIN, cfg.H_LIQ_MAX),
                          ("h_g0", H_G0, cfg.H_VAP_MIN, cfg.H_VAP_MAX)):
    if not (_lo < _h < _hi):
        raise SystemExit(
            f"ERROR: reservoir {_nm} = {_h/1e3:.2f} kJ/kg is OUTSIDE its table "
            f"[{_lo/1e3:.0f}, {_hi/1e3:.0f}] kJ/kg.  Widen the bounds in "
            f"case_config.py and re-run build_tables.py.")

# ---------------------------------------------------------------------------
# Reservoir isentrope for the stagnation-inlet pressure closure
# ---------------------------------------------------------------------------
# The inlet BC needs h_dom(p) along s_dom = s_dom0.
#
# NOT CoolProp's PSmass call: that returns the EQUILIBRIUM state, so once the
# isentrope crosses the saturation line it yields a two-phase MIXTURE enthalpy,
# which the solver would then use as the single-phase LIQUID enthalpy.  With a
# reservoir sitting exactly ON the saturation line (this case) that happens at
# the first infinitesimal pressure drop.  The correct object for a two-fluid
# model is the liquid's own metastable single-phase isentrope, integrated from
#     dh/dp = 1/rho(p,h)   at constant s
# with the solver's own table, so the closure is EOS-consistent and monotone by
# construction.  Validated against CoolProp in the subcooled window p > p0,
# where the equilibrium call does return the true liquid branch: agreement is
# exact to the printed precision (see isentrope.py --self-test).
from isentrope import metastable_isentrope

_iso_p = np.linspace(cfg.P_TABLE_MIN * 1.01, cfg.P_TABLE_MAX * 0.99, 400)
if dom == "gas":
    _iso_h = metastable_isentrope(fluid_g, _iso_p, P0, H_G0)
else:
    _iso_h = metastable_isentrope(fluid_l, _iso_p, P0, H_L0)
_mono = bool(np.all(np.diff(_iso_h) > 0))
print(f"  isentrope ({dom}): h = {_iso_h.min()/1e3:.2f} .. {_iso_h.max()/1e3:.2f} kJ/kg "
      f"over p = {_iso_p.min()/1e5:.3f} .. {_iso_p.max()/1e5:.3f} bar  (monotone={_mono})")
if not _mono:
    print("  WARNING: inlet isentrope is not monotone in p - the stagnation "
          "pressure closure may pick the wrong root.")

# ---------------------------------------------------------------------------
# Initial condition
# ---------------------------------------------------------------------------
#   'stagnation'            uniform field at rest at the reservoir state.
#   'stagnation_saturated'  same, but the LIQUID is seeded at h_l,sat(p0) rather
#                           than at the reservoir enthalpy.
#
# The second exists for a METASTABLE (superheated) reservoir.  With 'stagnation'
# the whole column starts at rest already superheated, so the flashing source is
# large in every cell at t = 0 - an explosive initial condition that has nothing
# to do with the steady state being sought.  (A saturated reservoir escapes this
# by construction: zero superheat at t=0 means sigma_Mg = 0.)  Measured on the
# +10 K metastable case at Nb = 1e15: 'stagnation' made SSR GROW 1.6e2 -> 2.6e4.
# Seeding the interior at saturation removes the t=0 source; the inlet BC still
# imposes the metastable reservoir, so the superheat develops as flow is drawn
# in, and the STEADY STATE is unchanged (an IC cannot move a fixed point).
if cfg.IC_TYPE in ("stagnation_saturated", "stagnation_sat"):
    _h_l_seed = float(CP_sat_liquid_enthalpy(P0))
    _tag = (f"stagnation_saturated: liquid seeded at h_l,sat(p0) = "
            f"{_h_l_seed/1e3:.2f} kJ/kg instead of the reservoir "
            f"{H_L0/1e3:.2f} kJ/kg (removes the t=0 flashing source)")
elif cfg.IC_TYPE == "stagnation":
    _h_l_seed = H_L0
    _tag = "stagnation: uniform field at rest at the reservoir state"
else:
    raise SystemExit(f"ERROR: unknown ic.type = {cfg.IC_TYPE!r} "
                     f"(use 'stagnation' or 'stagnation_saturated')")

U_h_0 = np.tile([P0, 1e-6, 1e-6, ALPHA_G, H_G0, _h_l_seed],
                (cfg.NX, 1)).astype(float)
print(f"\nIC ({_tag})")
print(f"   uniform p = {P0/1e5:.3f} bar, fluid at rest, alpha_g = {ALPHA_G:.1e}")


# ---------------------------------------------------------------------------
# Shared state plotter - used for BOTH the IC and the final solution
# ---------------------------------------------------------------------------
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
    for i in range(U.shape[0]):
        p_i, u_g_i, u_l_i = float(U[i, 0]), float(U[i, 1]), float(U[i, 2])
        ag_i, h_g_i, h_l_i = float(U[i, 3]), float(U[i, 4]), float(U[i, 5])
        eos_g_i = phase_props(fluid_g, p_i, h_g_i)
        eos_l_i = phase_props(fluid_l, p_i, h_l_i)
        cq_i = common_quantities(
            jnp.array([p_i, u_g_i, u_l_i, ag_i, h_g_i, h_l_i]),
            eos_g_i, eos_l_i, cfg.K_VM)
        rg_i, rl_i = float(eos_g_i.rho), float(eos_l_i.rho)
        rh2_i = float(cq_i.rh2)
        k_rho_mix_i = float(cq_i.rg_hat) - rg_i
        rho_hat_s_i = float(cq_i.rho_s) + k_rho_mix_i
        a_tilde_sq_i = (rho_hat_s_i / float(cq_i.rho_s)) * (rg_i * rl_i / rh2_i) \
            * float(cq_i.a0sq)
        delta_a_sq_i = (ag_i * (1.0 - ag_i) * rg_i * rl_i
                        * float(cq_i.rg_hat) * float(cq_i.rl_hat)
                        / (rh2_i ** 2) * float(cq_i.du) ** 2)
        a_i = np.sqrt(max(float(a_tilde_sq_i - delta_a_sq_i), 0.0))
        hat_ag_i = ag_i * rg_i * float(cq_i.rl_hat) / rh2_i
        hat_al_i = (1.0 - ag_i) * rl_i * float(cq_i.rg_hat) / rh2_i
        u_mix_i = hat_ag_i * u_g_i + hat_al_i * u_l_i
        M[i] = u_mix_i / a_i if a_i > 0.0 else np.nan

    du = U[:, 1] - U[:, 2]
    fig, ax = plt.subplots(7, 1, figsize=(6, 15), sharex=True)
    fig.suptitle(title, fontsize=9)
    ax[0].plot(x, A * 1e6, color="black", lw=1.0)
    ax[0].set_ylabel(r"Area $A$ [mm$^2$]")
    ax[1].plot(x, p / 1e5, color="purple", lw=1.0, marker="o", markersize=2)
    ax[1].axhline(cfg.P_OUT / 1e5, color="grey", ls="--", lw=0.8, label="$p_{out}$")
    ax[1].set_ylabel("Pressure [bar]")
    ax[1].legend(fontsize=7)
    ax[2].plot(x, U[:, 1], color="red", lw=1.0, marker="o", markersize=2, label="gas")
    ax[2].plot(x, U[:, 2], color="blue", lw=1.0, marker="o", markersize=2, label="liquid")
    ax[2].set_ylabel("velocity [m/s]")
    ax[2].legend(fontsize=7)
    ax[3].axhline(0.0, color="grey", lw=0.8, ls="--")
    ax[3].plot(x, du, color="teal", lw=1.0, marker="o", markersize=2)
    ax[3].set_ylabel(r"slip $u_g - u_l$ [m/s]")
    ax[4].plot(x, T_d - 273.15, color="darkorange", lw=1.0, marker="o", markersize=2)
    ax[4].set_ylabel(f"{domname} T [degC]")
    ax[5].plot(x, ag, color="green", lw=1.0, marker="o", markersize=2)
    ax[5].set_ylabel(r"$\alpha_g$")
    ax[6].plot(x, M, color="blue", lw=1.0, marker="o", markersize=2)
    ax[6].axhline(1.0, color="grey", ls="--", lw=0.8)
    ax[6].set_ylabel("Mach number")
    ax[6].set_xlabel("x [m]")
    for a_ in ax:
        a_.grid(alpha=0.3)
        a_.axvline(x[_i_throat], color="grey", lw=0.6, ls=":")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot saved: {out_png.name}")


_plot_state(U_h_0, f"INITIAL condition (stagnation, dominant={dom})  "
                   f"{cfg.GEO_FUNCTION}  N={cfg.NX}",
            out_dir / "initial_condition.png")

# ---------------------------------------------------------------------------
# Conservative state + round-trip check
# ---------------------------------------------------------------------------
W_0 = W_from_U_h_batch(jnp.array(U_h_0), fluid_g, fluid_l)
U_h_check = np.array(U_h_from_W_batch(W_0))
_rt = np.max(np.abs(U_h_check - U_h_0) / (np.abs(U_h_0) + 1e-300))
print(f"Round-trip U_h -> W -> U_h: max relative error = {_rt:.2e}  (expect ~1e-15)")


# ---------------------------------------------------------------------------
# Source terms: interphase closure (drag + heat + FLASHING mass transfer) + area
# ---------------------------------------------------------------------------
from typing import NamedTuple


class _AreaClosure(NamedTuple):
    dlnA_dx: jnp.ndarray


def _sources_fn(u_h, eos_g, eos_l, cq, dlnA, sat_g, sat_l):
    D = jnp.zeros(6)
    if cfg.SRC_SCALE != 0.0:
        closure = two_fluid_closure(
            u_h, eos_g, eos_l, fluid_g, fluid_l,
            T_sat   = sat_g["temperature"],
            h_sat_g = sat_g["enthalpy"],
            h_sat_l = sat_l["enthalpy"],
            dlnA_dx = dlnA,
            Nb      = cfg.BUBBLE_NUMBER,
            Nd      = cfg.DROPLET_NUMBER,
            mass_transfer = cfg.MASS_TRANSFER,
        )
        D = D + cfg.SRC_SCALE * D_VE_algebric(u_h, cq, closure, eos_g, eos_l, cfg.K_VM)
    if cfg.AREA_SRC:
        D = D + D_VE_area(u_h, cq, _AreaClosure(dlnA_dx=dlnA), eos_g, eos_l)
    return D


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
_p_in = float(P0 * 0.99)
U_h_inlet = jnp.array([_p_in, 1e-6, 1e-6, ALPHA_G, H_G0, H_L0])


def build_solver(p_out, p_out_relax):
    """Build (step, compute_dt) for one back-pressure.

    p_out and p_out_relax are captured as compile-time constants inside the
    jitted step, so a continuation stage costs one retrace.
    """
    return make_two_fluid_step(
        fluid_g, fluid_l,
        A, A_face, dAdx, dx,
        CFL             = cfg.CFL,
        k               = cfg.K_VM,
        delta_factor    = cfg.DELTA_FACTOR,
        newton_iter     = cfg.NEWTON_ITER,
        recon           = cfg.RECON,
        sources_fn      = _sources_fn,
        lowmach_diss    = cfg.LOWMACH_DISS,
        lowmach_diss_bc = cfg.LOWMACH_DISS_BC,
        nc_scale        = cfg.NC_SCALE,
        eps_void        = cfg.EPS_VOID,
        eps_void_ucut   = cfg.EPS_VOID_UCUT,
        eps_void_uwidth = cfg.EPS_VOID_UWIDTH,
        eps_void_scale  = cfg.EPS_VOID_SCALE,
        eps_ghost        = cfg.EPS_GHOST,
        eps_ghost_acut   = cfg.EPS_GHOST_ACUT,
        eps_ghost_awidth = cfg.EPS_GHOST_AWIDTH,
        eps_ghost_k2     = cfg.EPS_GHOST_K2,
        eps_ghost_alo    = cfg.EPS_GHOST_ALO,
        ssr_mode         = cfg.SSR_MODE,
        du_blend_lo      = cfg.DU_BLEND_LO,
        du_blend_hi      = cfg.DU_BLEND_HI,
        debug_newton     = cfg.DEBUG_NEWTON,
        jacobian_mode    = cfg.JACOBIAN,
        linear_solve     = cfg.LINEAR_SOLVE,
        bc_inlet         = cfg.BC_INLET,
        bc_outlet        = cfg.BC_OUTLET,
        H_g0             = H_G0,
        s_g0             = S_G0,
        H_l0             = H_L0,
        s_l0             = S_L0,
        dom_phase        = dom,
        iso_h            = _iso_h,
        iso_p            = _iso_p,
        alpha_g0         = ALPHA_G,
        quality          = QUALITY,
        slip             = SLIP,
        U_h_inlet        = U_h_inlet,
        p_out            = p_out,
        p_out_relax      = p_out_relax,
        eos_project      = cfg.EOS_PROJECT,
        proj_s_ref       = cfg.PROJ_S_REF,
        eos_rows         = cfg.EOS_ROWS,
        eos_rows_banded  = cfg.EOS_ROWS_BANDED,
        jac_lag          = cfg.JAC_LAG,
        nc_form          = cfg.NC_FORM,
        use_saturation   = cfg.MASS_TRANSFER,
        w_char           = cfg.W_CHAR,
        alpha_eps        = cfg.ALPHA_EPS,
        pin_ghost        = cfg.PIN_GHOST,
        pin_ghost_momentum = cfg.PIN_GHOST_MOMENTUM,
        return_debug     = False,
    )



# ---------------------------------------------------------------------------
# Table-bounds monitor - the check that matters for a metastable-table run
# ---------------------------------------------------------------------------
def _table_bounds_report(U):
    """Count cells whose (p, h) has left either EOS table."""
    p, hg, hl = U[:, 0], U[:, 4], U[:, 5]
    msgs = []
    n_p_lo = int(np.sum(p < cfg.P_TABLE_MIN))
    n_p_hi = int(np.sum(p > cfg.P_TABLE_MAX))
    if n_p_lo:
        msgs.append(f"p < p_min in {n_p_lo} cells (min {p.min()/1e5:.4f} bar)")
    if n_p_hi:
        msgs.append(f"p > p_max in {n_p_hi} cells (max {p.max()/1e5:.4f} bar)")
    for nm, h, lo, hi in (("h_g", hg, cfg.H_VAP_MIN, cfg.H_VAP_MAX),
                          ("h_l", hl, cfg.H_LIQ_MIN, cfg.H_LIQ_MAX)):
        n_lo, n_hi = int(np.sum(h < lo)), int(np.sum(h > hi))
        if n_lo:
            msgs.append(f"{nm} < h_min in {n_lo} cells (min {h.min()/1e3:.2f} kJ/kg)")
        if n_hi:
            msgs.append(f"{nm} > h_max in {n_hi} cells (max {h.max()/1e3:.2f} kJ/kg)")
    return msgs


# ---------------------------------------------------------------------------
# Time marching
# ---------------------------------------------------------------------------
def _increments(W_old, W_new, a_char):
    Uo = U_h_from_W_batch(W_old)[1:-1]
    Un = U_h_from_W_batch(W_new)[1:-1]
    bad = bool(jnp.any(jnp.isnan(Un)) | jnp.any(jnp.isinf(Un)))
    dp  = float(jnp.max(jnp.abs(Un[:, 0] - Uo[:, 0]) / (jnp.abs(Uo[:, 0]) + 1e-10)))
    dug = float(jnp.max(jnp.abs(Un[:, 1] - Uo[:, 1]) / (jnp.abs(Uo[:, 1]) + a_char + 1e-10)))
    dul = float(jnp.max(jnp.abs(Un[:, 2] - Uo[:, 2]) / (jnp.abs(Uo[:, 2]) + a_char + 1e-10)))
    dag = float(jnp.max(jnp.abs(Un[:, 3] - Uo[:, 3])))
    dhg = float(jnp.max(jnp.abs(Un[:, 4] - Uo[:, 4]) / (jnp.abs(Uo[:, 4]) + 1e-10)))
    dhl = float(jnp.max(jnp.abs(Un[:, 5] - Uo[:, 5]) / (jnp.abs(Uo[:, 5]) + 1e-10)))
    if cfg.DOM_ONLY_DT:
        du = dug if dom == "gas" else dul
        dh = dhg if dom == "gas" else dhl
    else:
        du, dh = max(dug, dul), max(dhg, dhl)
    return bad, dp, du, dh, dag, (dug, dul, dhg, dhl)


def march(W_start, step, compute_dt, n_steps, ssr_tol, label=""):
    """March one back-pressure stage towards steady state.

    Returns (W_last_good, ssr, n_done, stop_reason, t_phys).  W_last_good is
    always a finite state: on a NaN the pre-step state is returned, so a failed
    stage still yields something to post-process and restart from.
    """
    W_n    = W_start
    W_good = W_start
    t_phys = 0.0
    n      = 0
    ssr    = float("inf")
    _win      = cfg.PLATEAU_WINDOW
    _ssr_hist = [float("inf")] * _win
    _stop_reason = f"max steps ({n_steps}) reached"
    dt = float(compute_dt(W_start))
    _floor_streak, _ssr_at_streak = 0, float("inf")
    _oob_warned = False

    print(f"\nRunning {label}(max {n_steps} steps, SSR_tol = {ssr_tol:.1e})")
    print(f"{'Step':>6}  {'t [s]':>12}  {'dt [s]':>12}  {'SSR':>12}")
    print("-" * 60)

    _t0 = _time.perf_counter()
    try:
      while n < n_steps:
          halvings = 0
          if cfg.ADAPTIVE_DT:
              dt_cap   = float(compute_dt(W_n))
              a_char   = cfg.CFL * dx / max(dt_cap, 1e-300)
              dt_floor = cfg.DT_FLOOR_FRAC * dt_cap
              dt       = max(min(dt * cfg.DT_GROW, dt_cap), dt_floor)
              _per = (0, 0, 0, 0)
              bad = ok = floored = False
              for halvings in range(cfg.MAX_HALVINGS + 1):
                  W_try, ssr_j = step(W_n, jnp.float64(dt))
                  jax.block_until_ready(W_try)
                  bad, dp, du, dh, da, _per = _increments(W_n, W_try, a_char)
                  ok = (not bad) and dp < cfg.DP_MAX and dh < cfg.DH_MAX \
                      and du < cfg.DU_MAX and da < cfg.DA_MAX
                  if ok:
                      break
                  if dt <= dt_floor * (1.0 + 1e-9) or halvings >= cfg.MAX_HALVINGS:
                      floored = True
                      break
                  dt = max(dt * 0.5, dt_floor)
              if bad:
                  _stop_reason = f"NaN/Inf in the state at step {n+1} (dt={dt:.2e})"
                  break
              if floored and not ok:
                  if _floor_streak == 0:
                      _ssr_at_streak = ssr
                  _floor_streak += 1
              else:
                  _floor_streak = 0
              if halvings > 0 or floored:
                  _dug, _dul, _dhg, _dhl = _per
                  _tag = (f"FLOORED x{_floor_streak}" if floored
                          else f"halve x{halvings}")
                  print(f"        [{_tag}] dp={dp:.2f} da={da:.3f} | "
                        f"dug={_dug:.2f} dul={_dul:.2f} dhg={_dhg:.3f} dhl={_dhl:.3f}"
                        f"  ({'GHOST-gated' if max(_dul,_dhl) > max(_dug,_dhg) else 'liquid-gated'})",
                        flush=True)
              W_good = W_n
              W_n = W_try
              ssr = (max(float(ssr_j), ssr_tol * 10) if floored else float(ssr_j))
              if _floor_streak >= cfg.FLOOR_STALL_STEPS and ssr >= _ssr_at_streak * 0.99:
                  _stop_reason = (f"floored {_floor_streak} steps with no SSR progress "
                                  f"({_ssr_at_streak:.2e} -> {ssr:.2e})")
                  break
          else:
              dt = float(compute_dt(W_n))
              W_n, ssr_j = step(W_n, jnp.float64(dt))
              ssr = float(ssr_j)
              jax.block_until_ready(W_n)

          t_phys += dt
          n += 1
          _ssr_hist = _ssr_hist[1:] + [ssr]

          if n <= 10 or n % cfg.PRINT_EVERY == 0 or halvings > 0:
              _el = _time.perf_counter() - _t0
              _hv = f"  [{halvings} halving(s)]" if (cfg.ADAPTIVE_DT and halvings > 0) else ""
              print(f"{n:>6}  {t_phys:>12.4e}  {dt:>12.4e}  {ssr:>12.4e}  ({_el:.1f}s){_hv}",
                    flush=True)

          if n % cfg.PRINT_EVERY == 0:
              _msgs = _table_bounds_report(np.array(U_h_from_W_batch(W_n)))
              if _msgs and not _oob_warned:
                  print("        [TABLE BOUNDS] " + "; ".join(_msgs), flush=True)
                  print("        (states outside the table are CLAMPED to the edge "
                        "- treat the result there as unreliable)", flush=True)
                  _oob_warned = True

          if ssr < ssr_tol:
              _stop_reason = f"SSR = {ssr:.3e} < tol = {ssr_tol:.1e}"
              break
          if n >= _win:
              _hmax, _hmin = max(_ssr_hist), min(_ssr_hist) + 1e-300
              if _hmax / _hmin < cfg.PLATEAU_RATIO:
                  _stop_reason = f"SSR plateau ({_hmin:.3e} - {_hmax:.3e})"
                  break
          if t_phys >= cfg.T_END:
              _stop_reason = "t_end reached"
              break
    except KeyboardInterrupt:
        _stop_reason = "interrupted by user"

    _elapsed = _time.perf_counter() - _t0
    print(f"\nStopped: {_stop_reason}")
    print(f"  steps = {n}   t = {t_phys:.3e} s   SSR = {ssr:.3e}   wall = {_elapsed:.1f} s")
    W_out = W_good if "NaN" in _stop_reason else W_n
    return W_out, ssr, n, _stop_reason, t_phys


# ---------------------------------------------------------------------------
# Back-pressure continuation
# ---------------------------------------------------------------------------
# A single transient from the stagnation IC straight to p_out cannot work here:
# the liquid must FLASH to shed enough enthalpy to stay on the live side of the
# spinodal as p falls (see the README), and flashing is finite-rate.  Marching
# each intermediate back-pressure to STEADY STATE gives the mass-transfer source
# unlimited time to do that work, so each stage starts from a liquid that has
# already flashed and sits well clear of the spinodal.
#
# Each stage costs one retrace (p_out is a compile-time constant in the jitted
# step), which is the price of not modifying the solver package.
_stages = list(cfg.P_OUT_STAGES)
print(f"\n{'='*70}")
print(f"BACK-PRESSURE CONTINUATION: {len(_stages)} stage(s)  "
      + " -> ".join(f"{q/1e5:.2f}" for q in _stages) + " bar")
print(f"{'='*70}")

W_n = W_0
_hist = []
for _k, _p_stage in enumerate(_stages):
    _last = (_k == len(_stages) - 1)
    _tol  = cfg.SSR_TOL if _last else cfg.SSR_TOL_STAGE
    _nmax = cfg.N_STEPS if _last else cfg.N_STEPS_STAGE

    # Spinodal margin check on the state entering this stage: the liquid must be
    # able to reach p_stage without crossing the spinodal.
    _Uin = np.array(U_h_from_W_batch(W_n))
    _hl_max = float(_Uin[:, 5].max())
    print(f"\n{'-'*70}")
    print(f"STAGE {_k+1}/{len(_stages)}:  p_out = {_p_stage/1e5:.3f} bar"
          f"   (entering h_l max = {_hl_max/1e3:.2f} kJ/kg)")
    print(f"{'-'*70}")
    print(f"Building solver for this stage (retrace) ... ", flush=True)

    step, compute_dt = build_solver(_p_stage, cfg.P_OUT_RELAX)
    _dt0 = float(compute_dt(W_n))
    print(f"  compiled.  dt_cap = {_dt0:.4e} s")

    W_n, _ssr, _n, _reason, _t = march(
        W_n, step, compute_dt, _nmax, _tol,
        label=f"stage {_k+1} (p_out={_p_stage/1e5:.2f} bar) ")
    _hist.append((_p_stage, _ssr, _n, _reason))

    # Checkpoint every stage so a later failure still leaves usable output.
    _Uk = np.array(U_h_from_W_batch(W_n))
    np.savetxt(out_dir / f"stage{_k+1:02d}_p{_p_stage/1e5:.2f}bar.csv",
               np.column_stack([x, A, _Uk[:, 0], _Uk[:, 3], _Uk[:, 1],
                                _Uk[:, 2], _Uk[:, 4], _Uk[:, 5]]),
               delimiter=",", comments="",
               header="x,A,p,alpha_g,u_g,u_l,h_g,h_l")

    if "NaN" in _reason:
        print(f"\n  STAGE {_k+1} FAILED ({_reason}).")
        print(f"  Continuation stops here; the last good state is kept and "
              f"post-processed below.")
        break

print(f"\n{'='*70}")
print("CONTINUATION SUMMARY")
for _p_stage, _ssr, _n, _reason in _hist:
    print(f"  p_out = {_p_stage/1e5:7.3f} bar   steps = {_n:6d}   "
          f"SSR = {_ssr:.3e}   {_reason}")
print(f"{'='*70}")

ssr, n = (_hist[-1][1], _hist[-1][2]) if _hist else (float("nan"), 0)

# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
U_h_f = np.array(U_h_from_W_batch(W_n))
p_f, u_g_f, u_l_f = U_h_f[:, 0], U_h_f[:, 1], U_h_f[:, 2]
alpha_g, h_g_f, h_l_f = U_h_f[:, 3], U_h_f[:, 4], U_h_f[:, 5]

_stg = fluid_g.get_state(jxp.HmassP_INPUTS, jnp.array(h_g_f), jnp.array(p_f))
_stl = fluid_l.get_state(jxp.HmassP_INPUTS, jnp.array(h_l_f), jnp.array(p_f))
T_g_f, rho_g_f = np.array(_stg["temperature"]), np.array(_stg["density"])
T_l_f, rho_l_f = np.array(_stl["temperature"]), np.array(_stl["density"])

u_dom, T_dom, h_dom = ((u_g_f, T_g_f, h_g_f) if dom == "gas"
                       else (u_l_f, T_l_f, h_l_f))

print("\n-- Table-bounds check on the FINAL state ---------------------------")
_msgs = _table_bounds_report(U_h_f)
if _msgs:
    for m in _msgs:
        print(f"  OUT OF TABLE: {m}")
    print("  -> the solution is not trustworthy where this happens; widen the "
          "table bounds in case_config.py and rebuild.")
else:
    print(f"  all {cfg.NX} cells inside both tables "
          f"(p {p_f.min()/1e5:.4f}..{p_f.max()/1e5:.4f} bar, "
          f"h_g {h_g_f.min()/1e3:.1f}..{h_g_f.max()/1e3:.1f}, "
          f"h_l {h_l_f.min()/1e3:.1f}..{h_l_f.max()/1e3:.1f} kJ/kg)")

V_f    = np.array(jax.vmap(V_from_W)(W_n))
mdot_g = V_f[:, 0] * u_g_f * A
mdot_l = V_f[:, 1] * u_l_f * A
mdot   = mdot_g + mdot_l


def _flux_rms(md):
    mm = np.mean(md[1:-1])
    return mm, np.sqrt(np.mean((md[1:-1] - mm) ** 2)) / (abs(mm) + 1e-300)


print("\n-- Flow diagnostics ------------------------------------------------")
print(f"  void fraction alpha_g: {alpha_g.min():.6f} .. {alpha_g.max():.6f}  "
      f"(dominant phase = {dom})")
_mmg, _rg = _flux_rms(mdot_g)
_mml, _rl = _flux_rms(mdot_l)
mm, rms = _flux_rms(mdot)
print(f"  vapour mass flux  mean = {_mmg:.4e} kg/s   RMS variation = {_rg*100:.3f} %")
print(f"  liquid mass flux  mean = {_mml:.4e} kg/s   RMS variation = {_rl*100:.3f} %")
print(f"  TOTAL  mass flux  mean = {mm:.4e} kg/s   RMS variation = {rms*100:.3f} %"
      f"   (-> 0 = quasi-1D mass conserved)")
print(f"  p: {p_f.min()/1e5:.3f} .. {p_f.max()/1e5:.3f} bar   "
      f"p_exit = {p_f[-1]/1e5:.4f} bar  (target p_out = {cfg.P_OUT/1e5:.3f} bar)")

# Choking / exit regime.  If the exit is supersonic the characteristic outlet is
# transmissive by construction and the back-pressure is NOT felt inside the
# nozzle - so p_exit != p_out is the correct answer, not a BC failure.
_M = np.empty(cfg.NX)
for _i in range(cfg.NX):
    _eg = phase_props(fluid_g, float(p_f[_i]), float(h_g_f[_i]))
    _el = phase_props(fluid_l, float(p_f[_i]), float(h_l_f[_i]))
    _cq = common_quantities(jnp.array([p_f[_i], u_g_f[_i], u_l_f[_i],
                                       alpha_g[_i], h_g_f[_i], h_l_f[_i]]),
                            _eg, _el, cfg.K_VM)
    _rg, _rl, _rh2 = float(_eg.rho), float(_el.rho), float(_cq.rh2)
    _k = float(_cq.rg_hat) - _rg
    _a2 = ((float(_cq.rho_s) + _k) / float(_cq.rho_s)) * (_rg * _rl / _rh2) * float(_cq.a0sq)
    _da2 = (alpha_g[_i] * (1.0 - alpha_g[_i]) * _rg * _rl * float(_cq.rg_hat)
            * float(_cq.rl_hat) / _rh2 ** 2 * float(_cq.du) ** 2)
    _a = np.sqrt(max(_a2 - _da2, 0.0))
    _hag = alpha_g[_i] * _rg * float(_cq.rl_hat) / _rh2
    _hal = (1.0 - alpha_g[_i]) * _rl * float(_cq.rg_hat) / _rh2
    _M[_i] = (_hag * u_g_f[_i] + _hal * u_l_f[_i]) / _a if _a > 0 else np.nan
_isup = np.flatnonzero(_M >= 1.0)
print(f"  mixture Mach: {np.nanmin(_M):.4f} .. {np.nanmax(_M):.4f}   "
      f"exit M = {_M[-1]:.4f}")
if _isup.size:
    print(f"  CHOKED: sonic point at cell {_isup[0]} (x = {x[_isup[0]]:.4f} m); "
          f"geometric throat is cell {_i_throat} (x = {x[_i_throat]:.4f} m)")
    print(f"    the sonic point sits DOWNSTREAM of the area throat - thermal "
          f"choking from the flashing mass/heat source.")
if _M[-1] > 1.0:
    print(f"  Exit is SUPERSONIC -> the characteristic outlet is transmissive and "
          f"p_out is not felt;")
    print(f"    the nozzle is UNDER-EXPANDED (p_exit = {p_f[-1]/1e5:.3f} bar > "
          f"p_out = {cfg.P_OUT/1e5:.3f} bar); the rest of the expansion would "
          f"occur in a plume downstream.")
print(f"  u_liquid: {u_l_f.min():.2f} .. {u_l_f.max():.2f} m/s   "
      f"u_vapour: {u_g_f.min():.2f} .. {u_g_f.max():.2f} m/s")
print(f"  T_liquid: {T_l_f.min()-273.15:.2f} .. {T_l_f.max()-273.15:.2f} degC   "
      f"T_vapour: {T_g_f.min()-273.15:.2f} .. {T_g_f.max()-273.15:.2f} degC")

# Energy check.  Under mass transfer a SINGLE phase's h + u^2/2 is NOT conserved
# (the phases exchange enthalpy), so the per-phase quantity is meaningless here -
# it drifts by tens of percent on a healthy flashing solution.  The conserved
# quantity is the MIXTURE stagnation enthalpy, mass-flux-weighted over both
# phases; that is what is reported.
_H0_mix = (mdot_g * (h_g_f + 0.5 * u_g_f ** 2)
           + mdot_l * (h_l_f + 0.5 * u_l_f ** 2)) / mdot
_dH0 = np.max(np.abs(_H0_mix[1:-1] - _H0_mix[0])) / (abs(_H0_mix[0]) + 1e-300)
print(f"  MIXTURE stagnation enthalpy H0 = {_H0_mix[0]/1e3:.3f} kJ/kg   "
      f"max drift = {_dH0:.2e}  (-> 0 = energy conserved)")
_H0_dom = (h_dom + 0.5 * u_dom ** 2)
print(f"  (per-phase {dom} h+u^2/2 spans {_H0_dom.min()/1e3:.1f}..{_H0_dom.max()/1e3:.1f} "
      f"kJ/kg - NOT conserved under mass transfer, shown for reference only)")


def _oddeven(f):
    o = np.abs(f[1:-1] - 0.5 * (f[:-2] + f[2:]))
    return o.max(), o.max() / ((f.max() - f.min()) + 1e-300)


print("\n-- Checkerboard (odd-even 2*dx) amplitude --------------------------")
for _nm, _f in [("p     ", p_f), ("u_g   ", u_g_f), ("u_l   ", u_l_f),
                ("du    ", u_g_f - u_l_f), ("alpha ", alpha_g), ("h_g   ", h_g_f)]:
    _amp, _rel = _oddeven(_f)
    _flag = "  <-- CHECKERBOARD" if _rel > 0.02 else ""
    print(f"  {_nm}: max 2dx osc = {_amp:.3e}  ({_rel*100:.2f} % of range){_flag}")

# -- flashing source diagnostic ---------------------------------------------
if cfg.MASS_TRANSFER and cfg.SRC_SCALE != 0.0:
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
        )
        return (cl.sigma_Mg, cl.sigma_Qg, cl.sigma_Ql,
                sat_g_c["temperature"], eos_l_c.T)

    _sMg, _sQg, _sQl, _Tsat, _Tl = (np.array(v) for v in
                                    jax.vmap(_mt_cell)(jnp.array(U_h_f)))
    _superheat = _Tl - _Tsat
    _on = np.flatnonzero(_sMg > 0.0)
    print("\n-- Flashing (mass-transfer) diagnostic -----------------------------")
    if _on.size:
        print(f"  flashing onset at x = {x[_on[0]]:.4f} m (cell {_on[0]}, "
              f"{'upstream of' if _on[0] < _i_throat else 'at/after'} the throat); "
              f"active in {_on.size}/{cfg.NX} cells")
        print(f"  sigma_Mg: max = {_sMg.max():.3e} kg/(m3 s) at "
              f"x = {x[int(np.argmax(_sMg))]:.4f} m")
        _amp, _rel = _oddeven(_sMg)
        print(f"  sigma_Mg 2dx osc = {_amp:.3e}  ({_rel*100:.2f} % of range)")
    else:
        print("  sigma_Mg = 0 everywhere (the liquid is nowhere superheated)")
    print(f"  liquid superheat T_l - T_sat: {_superheat.min():.2f} .. "
          f"{_superheat.max():.2f} K")

    figS, axS = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axS[0].plot(x, _sMg, "o-", ms=2.5, lw=0.8)
    axS[0].set_ylabel(r"$\sigma_{Mg}$ [kg/(m$^3$s)]")
    axS[0].set_title("Flashing mass-transfer source on the final state")
    axS[1].plot(x, _superheat, "o-", ms=2.5, lw=0.8)
    axS[1].axhline(0.0, color="k", lw=0.6)
    axS[1].set_ylabel(r"$T_l - T_{sat}$ [K]")
    axS[2].plot(x, _sQg, "o-", ms=2.5, lw=0.8, label=r"$\sigma_{Qg}$")
    axS[2].plot(x, _sQl, "o-", ms=2.5, lw=0.8, label=r"$\sigma_{Ql}$")
    axS[2].set_ylabel(r"interfacial heat [W/m$^3$]")
    axS[2].set_xlabel("x [m]")
    axS[2].legend()
    for a_ in axS:
        a_.grid(alpha=0.3)
        a_.axvline(x[_i_throat], color="grey", lw=0.6, ls=":")
    figS.tight_layout()
    figS.savefig(out_dir / "mass_transfer_source.png", dpi=150)
    plt.close(figS)
    print(f"  plot saved: mass_transfer_source.png")

# -- CSV + final plot --------------------------------------------------------
csv_header = "x,A,p,alpha_g,u_g,u_l,T_g,T_l,h_g,h_l,rho_g,rho_l"
csv_data = np.column_stack([x, A, p_f, alpha_g, u_g_f, u_l_f,
                            T_g_f, T_l_f, h_g_f, h_l_f, rho_g_f, rho_l_f])
np.savetxt(out_dir / "solution.csv", csv_data, delimiter=",",
           header=csv_header, comments="")
print(f"\nSolution saved -> {out_dir / 'solution.csv'}")

_plot_state(U_h_f, f"FINAL solution (dominant={dom})  {cfg.GEO_FUNCTION}  "
                   f"N={cfg.NX}  steps={n}  SSR={ssr:.2e}",
            out_dir / "final_solution.png")
print(f"\nAll outputs in: {out_dir}")
