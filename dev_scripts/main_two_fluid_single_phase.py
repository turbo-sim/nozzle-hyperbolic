# %%
"""
Two-fluid FVS solver — SINGLE-PHASE-LIMIT verification (ghost second phase).

Purpose
-------
Verify the new quasi-1D AREA-CHANGE source term (sources.D_VE_area) by collapsing
one phase to a "ghost" (alpha ~ 1e-4) so the two-fluid model must reduce to the
ordinary single-phase quasi-1D Euler flow in a nozzle.  The dominant phase should
then match the single-phase Städtke solver (dev_scripts/main.py) run with the SAME
fluid, geometry, and boundary conditions.

Why this isolates the area source
---------------------------------
  * GHOST collapses the second phase → no real two-phase coupling.
  * SRC_SCALE = 0.0 turns OFF the physical interphase sources (drag / heat / mass
    transfer in D_VE_algebric), so the ONLY source acting is the area term.
  * AREA_SRC = True enables D_VE_area = [0, 0, ag*p*G, al*p*G, -p*Da, +p*Da],
    with G = (1/A) dA/dx.  In the single-phase limit ag→1 this is exactly the
    single-phase wall-pressure source p*G (fvm.py), and Da→0 so the energy terms
    vanish.  A constant-area pipe gives G=0 → the source is a no-op, so this test
    REQUIRES a nozzle geometry (set the `geometry:` block in the YAML).

Expected result
---------------
Dominant-phase p, u, (rho, T) profiles along the nozzle overlay the single-phase
solver to plotting accuracy.  Ghost phase is passive (its alpha stays ~floor).

Run from dev_scripts/:
    python main_two_fluid_single_phase.py
"""

import os
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=true")
import jax
jax.config.update("jax_enable_x64", True)

import pathlib
_cache_dir = pathlib.Path(__file__).parent.parent / ".jax_cache"
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

import yaml as _yaml
from config_loader import load_config
from nozzle_hyperbolic.fvm_solver.two_fluid.sources import (
    two_fluid_closure,
    D_VE_algebric,
    D_VE_algebric_two_component,
    D_VE_area,
)

# Settings file for this test (two-fluid format: gas=Air, liquid=water ghost,
# mirroring the single-phase settings_nozzle_validation.yaml).
# SETTINGS_YAML = "settings_two_fluid_nozzle_validation.yaml"
SETTINGS_YAML = "settings_two_fluid_nozzle_validation_two_comonent.yaml"
# SETTINGS_YAML = "settings_two_fluid_liquid_bernoulli.yaml"
_here = pathlib.Path(__file__).parent
cfg   = load_config(_here / SETTINGS_YAML)
with open(_here / SETTINGS_YAML, encoding="utf-8") as _f:
    _raw = _yaml.safe_load(_f)
T_IN = float(_raw["bc"]["T_g0"])   # inlet static T (driver builds h from p,T)

# ── Single-phase-limit test knobs ─────────────────────────────────────────────
# SRC_SCALE : scales the interphase source (D_VE_algebric with mass transfer, or
#   D_VE_algebric_two_component = drag+heat only when mass_transfer=False in the
#   YAML).  Keep 1.0 here: with mass_transfer=False the drag+heat SLAVES the ghost
#   (u_l→u_g, T_l→T_g), dumping the area source's +p·D_α interfacial work into the
#   gas so h_l can't blow up.  Redistributes between phases only ⇒ the 99.999%
#   dominant phase (and the area-source validation) is unaffected.
# AREA_SRC  : the quasi-1D area-change source (D_VE_area) — the term under test.
SRC_SCALE = 1.0
AREA_SRC  = True
# DOM_ONLY_DT: base the adaptive-dt increment check on the DOMINANT phase only,
#   ignoring the near-massless GHOST's u_l/h_l (which, with no drag, oscillate
#   freely and can trigger halving / dt-collapse even when the dominant phase is
#   fine).  Tests whether the ghost — not the main flow — gates the timestep.
DOM_ONLY_DT = True

# Dominant phase (for plotting / dom-phase diagnostics) from the YAML composition.
_void_hint = (cfg.BC_VOID if cfg.BC_VOID is not None else
              (cfg.BC_QUALITY if cfg.BC_QUALITY is not None else 0.999999))
dom = "gas" if _void_hint >= 0.5 else "liquid"

# ── Output directory ──────────────────────────────────────────────────────────
import shutil
import datetime as _dt

_stamp    = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name  = f"nozzle_valid_{dom}dom_N{cfg.Nx}_{cfg.RECON}_{_stamp}"
out_dir   = _here / "results" / run_name
out_dir.mkdir(parents=True, exist_ok=True)
shutil.copy(_here / SETTINGS_YAML, out_dir / SETTINGS_YAML)
print(f"Output directory: {out_dir.resolve()}")

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

fluid_gas_name = _raw["eos"]["gas"]["fluid_name"]
fluid_liquid_name = _raw["eos"]["liquid"]["fluid_name"]

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
dAdx  = np.asarray(geo.dAdx, dtype=float)
A_face = 0.5 * (A[:-1] + A[1:])

print(f"Grid: Nx={cfg.Nx}  dx={dx:.4f} m  geometry={cfg.GEO_FUNCTION}  "
      f"A: {A.min():.4e}..{A.max():.4e} m^2")
_G_max = float(np.max(np.abs(dAdx / A)))
print(f"  max|(1/A)dA/dx| = {_G_max:.3e} 1/m"
      + ("   WARNING: ~constant area → area source is a NO-OP; use a nozzle geometry!"
         if _G_max < 1e-6 else "   (nozzle — area source active)"))


# ── Shared state plotter — used for BOTH the initial condition and the final ──
# solution so the IC is drawn "as an initial solution" (identical panels/format).
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

    du = U[:, 1] - U[:, 2]                       # slip velocity u_g - u_l
    fig, ax = plt.subplots(7, 1, figsize=(5, 14), sharex=True)
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
    ax[2].legend()
    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {out_png}")
    return fig

# ── Stagnation (reservoir) inlet state from the YAML bc block ──────────────────
# Mirrors the single-phase settings_nozzle_validation.yaml `boundary_conditions`,
# extended for two phases: p0 (shared), per-phase stagnation T_g0/T_l0, composition
# (void_fraction OR quality) and slip.  H_g0/s_g0/H_l0 come from the reservoir
# state at rest (u=0); the inlet velocity/mass flux emerge (so the nozzle chokes).
P0   = cfg.BC_P0   if cfg.BC_P0   is not None else cfg.P_INIT
T_G0 = cfg.BC_T_G0 if cfg.BC_T_G0 is not None else T_IN
T_L0 = cfg.BC_T_L0 if cfg.BC_T_L0 is not None else T_G0
SLIP = cfg.BC_SLIP


_stl0 = fluid_l_inverse_call.get_state(jxp.PT_INPUTS, jnp.array([P0]), jnp.array([T_L0]))
_stg0 = fluid_g_inverse_call.get_state(jxp.PT_INPUTS, jnp.array([P0]), jnp.array([T_G0]))
# _stl0 = _stg0
# print("Second call is not working, change if change the fluid!!!")
# input("Now press enter to continue: \n \n")
H_g0   = float(_stg0["enthalpy"][0])          # gas stagnation enthalpy (u=0)
s_g0   = float(_stg0["s"][0])                 # gas stagnation entropy
H_l0   = float(_stl0["enthalpy"][0])          # liquid stagnation enthalpy
rho_g0 = float(_stg0["density"][0]);  rho_l0 = float(_stl0["density"][0])

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

print(f"\nStagnation inlet:  p0={P0/1e5:.2f} bar  T_g0={T_G0:.1f}  T_l0={T_L0:.1f} K")
print(f"  H_g0={H_g0/1e3:.1f} kJ/kg  s_g0={s_g0:.2f} J/kg/K  H_l0={H_l0/1e3:.1f} kJ/kg")
print(f"  alpha_g={ALPHA_G:.6f}  quality={QUALITY:.6f}  slip={SLIP:.3f}  (dominant={dom})")
print(f"  SRC_SCALE={SRC_SCALE} (physical sources)   AREA_SRC={AREA_SRC} (area term under test)")

# ── Initial condition (derived from stagnation) ───────────────────────────────
# Expand the gas ISENTROPICALLY from stagnation to an inlet-static guess (≈ p_out
# for subsonic), giving the inlet static enthalpy/velocity; then nozzle_simple
# continuity  u = mdot/(ρA)  (u peaks at the throat), enthalpy from H0 = h+u²/2,
# incompressible-Bernoulli pressure dip.  Ghost phase co-flows via the slip ratio.
_p_in    = float(P0 * 0.99)  #  float(min(cfg.P_OUT, P0 * 0.999))
p_0_out = cfg.P_OUT
_stg_in  = fluid_g.get_state(jxp.PSmass_INPUTS, jnp.array([_p_in]), jnp.array([s_g0]))
h_g_0    = float(_stg_in["enthalpy"][0])
rho_g_in = float(_stg_in["density"][0])
_a_in    = float(_stg_in["speed_of_sound"][0])
u_g_in   = float(np.sqrt(max(2.0 * (H_g0 - h_g_0), 0.0)))
u_l_in   = u_g_in / SLIP
h_l_0    = H_l0 - 0.5 * u_l_in**2
print(f"  IC inlet static: p={_p_in/1e5:.2f} bar  h_g={h_g_0/1e3:.1f} kJ/kg  "
      f"u_g={u_g_in:.1f} m/s  a={_a_in:.1f}  (M_in={u_g_in/_a_in:.3f})")

IC_MODE = _raw["ic"]["type"]


if IC_MODE == "uniform":
    U_h_0 = np.tile(
        [_p_in, u_g_in, u_l_in, ALPHA_G, h_g_0, h_l_0], (cfg.Nx, 1),
    ).astype(float)
    print(f"IC (uniform):  p={_p_in/1e5:.1f} bar  alpha_g={ALPHA_G:.6f}"
          f"  u_g={u_g_in:.1f} m/s  u_l={u_l_in:.1f} m/s")
    
elif IC_MODE == "stagnation":
    U_h_0 = np.tile(
        [P0, 1e-6, 1e-6, ALPHA_G, H_g0, H_l0], (cfg.Nx, 1),
    ).astype(float)
    print(f"IC (uniform):  p={_p_in/1e5:.1f} bar  alpha_g={ALPHA_G:.6f}"
          f"  u_g={u_g_in:.1f} m/s  u_l={u_l_in:.1f} m/s")
    
elif IC_MODE == "simple":
    # ── Robust "cold-start" IC (no nozzlex needed) ────────────────────────────
    # A GENTLE, SUBSONIC, slip-consistent guess that lets the time-marching build
    # the (transonic) flow itself.  Converges in more iterations than the nozzlex
    # seed, but is robust and self-contained.
    #
    # Why NOT per-phase continuity (the previous attempt): computing u_g from GAS
    # continuity (ρ_g drops ~10× along the nozzle) and u_l from LIQUID continuity
    # (ρ_l≈const) gives two UNRELATED velocity fields — u_g→585, u_l→264 at the
    # throat, du swinging −2→+320 m/s — which (a) destroys the imposed slip and
    # (b) is wildly mixture-SUPERSONIC (Wood a_mix ~tens of m/s at α≈0.39), with a
    # velocity CAP kink.  The first implicit step then blows up (NaN).  See probe.
    #
    # Robust recipe instead:
    #   * pressure: mild linear ramp P0 → p_out (matches the stagnation inlet and
    #     the back-pressure outlet at the two ends → minimal startup shock),
    #   * composition: uniform α_g = ALPHA_G,
    #   * velocities: SMALL and SLIP-CONSISTENT (u_g = S·u_l, so du≈0) — do not
    #     pre-build the throat spike; the transient accelerates the flow,
    #   * enthalpies: stagnation H_g0/H_l0 minus the (tiny) dynamic head → always
    #     valid EOS states.
    u_seed_l  = 5.0                                   # m/s — gentle, safely subsonic
    p_cells   = np.linspace(P0, p_0_out, cfg.Nx)      # inlet→outlet pressure ramp
    ag        = np.full(cfg.Nx, ALPHA_G)
    u_cells_l = np.full(cfg.Nx, u_seed_l)
    u_cells_g = np.full(cfg.Nx, u_seed_l * SLIP)      # impose slip → du starts ~0
    h_cells_g = np.full(cfg.Nx, H_g0 - 0.5 * u_cells_g**2)
    h_cells_l = np.full(cfg.Nx, H_l0 - 0.5 * u_cells_l**2)

    U_h_0 = np.stack([p_cells, u_cells_g, u_cells_l, ag, h_cells_g, h_cells_l], axis=1)

    print(f"IC (simple cold-start):  p ramp {P0/1e5:.1f}→{p_0_out/1e5:.2f} bar  "
          f"alpha_g={ALPHA_G:.4f}  u_g={u_cells_g[0]:.1f}  u_l={u_cells_l[0]:.1f} m/s "
          f"(slip={SLIP:.2f}, du≈0)")
elif IC_MODE == "nozzlex":
    # Seed the transient from a SAVED nozzlex space-marching solution (pickle).
    # The pickle carries a converged, physically self-consistent two-fluid
    # profile (drag/heat/area balanced, correct slip) on the SAME FluidBicubic
    # tables — so the interior starts on the slow manifold instead of a steep,
    # slip-inconsistent guess.  Phase ordering is auto-detected from densities
    # (nozzlex idx-1 = liquid in the Elliot two-component pickle).
    from nozzlex_ic import ic_from_nozzlex_pickle
    _pkl_rel = _raw["ic"].get("pickle")
    if _pkl_rel is None:
        raise ValueError("ic.type='nozzlex' requires ic.pickle: <path to solution pickle>")
    _pkl = (_here / _pkl_rel)
    if not _pkl.exists():
        _pkl = pathlib.Path(_pkl_rel)          # allow an absolute / cwd path too
    U_h_0, _nz_prof = ic_from_nozzlex_pickle(str(_pkl), x,
                                             phase1=_raw["ic"].get("phase1", "auto"))
    print(f"IC (nozzlex pickle):  {_pkl.name}")
    print(f"  phase1={_nz_prof['phase1']}  (nozzlex idx-1)   profile x:"
          f" {_nz_prof['x'].min():.4f}..{_nz_prof['x'].max():.4f} m")
    print(f"  p {U_h_0[:,0].min()/1e5:.2f}..{U_h_0[:,0].max()/1e5:.2f} bar"
          f"  u_g {U_h_0[:,1].min():.1f}..{U_h_0[:,1].max():.1f}"
          f"  u_l {U_h_0[:,2].min():.1f}..{U_h_0[:,2].max():.1f} m/s"
          f"  alpha_g {U_h_0[:,3].min():.3f}..{U_h_0[:,3].max():.3f}")
elif IC_MODE == "bernoulli":
    # ── Analytic incompressible (Bernoulli) cold-start — no nozzlex needed ─────
    # For a nearly incompressible DOMINANT phase the exact steady quasi-1D flow is
    #   u_dom = u_in·A_in/A(x)   (mass),   p = p0 − ½ρ_dom u_dom²   (Bernoulli),
    # valid to O(Ma²).  This same profile is the ANALYTIC REFERENCE the ghost-limit
    # solution must reproduce.  The ghost co-flows (slip), constant α_g, isothermal
    # at T0 (h from PT → always in-table).  Serves the low-speed-liquid / flashing
    # inlet regime.  `ic.u_in` sets the inlet dominant-phase velocity (default 1).
    from bernoulli_ic import bernoulli_ic
    _u_in = float(_raw["ic"].get("u_in", 1.0))
    U_h_0, _bern = bernoulli_ic(
        x, A, p0=P0, T0_g=T_G0, T0_l=T_L0, u_in=_u_in,
        alpha_g=ALPHA_G, slip=SLIP,
        fluid_g=fluid_g, fluid_l=fluid_l, dom=dom,
    )
    print(f"IC (bernoulli, dom={dom}):  p0={P0/1e5:.2f} bar  u_in={_u_in:.2f} m/s  "
          f"rho_dom={_bern['rho_dom']:.2f} kg/m³  Ma_in={_bern['Ma_in']:.4f}")
    print(f"  p {U_h_0[:,0].min()/1e5:.3f}..{U_h_0[:,0].max()/1e5:.3f} bar"
          f"  u_dom {_bern['u_dom'].min():.2f}..{_bern['u_dom'].max():.2f} m/s"
          f"  alpha_g={ALPHA_G:.6f}  slip={SLIP:.2f}")
    if _bern["p_floor_hit"]:
        print("  WARNING: Bernoulli pressure hit the floor (p≤p_floor) — the "
              "throat dip is too deep; lower ic.u_in or widen the throat.")
else:
    raise ValueError("Initial condition specified is not available. Select among the availables")

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
      + (f"  (p_out={cfg.P_OUT/1e5:.2f} bar)" if cfg.BC_OUTLET in ("pressure", "characteristic") else "")
      + (f"  H_g0={H_g0/1e3:.1f} kJ/kg  s_g0={s_g0:.1f}  quality={QUALITY:.6f}  slip={SLIP:.3f}"
         if cfg.BC_INLET == "stagnation" else ""))

PROJECT = False
MONITOR = False
print(f"[Diagnostic]  PROJECT={PROJECT}  MONITOR={MONITOR}")

# ── Solver factory ────────────────────────────────────────────────────────────
print(f"Building solver (first JIT compilation may take a minute)... jacobian={cfg.JACOBIAN}")
_ret = make_two_fluid_step(
    fluid_g, fluid_l,
    A, A_face, dAdx, dx,
    CFL           = cfg.CFL,
    k             = cfg.K_VM,
    delta_factor  = cfg.DELTA_FACTOR,
    newton_iter   = cfg.NEWTON_ITER,
    recon         = cfg.RECON,
    sources_fn    = _sources_fn,
    lowmach_diss  = cfg.LOWMACH_DISS,
    nc_scale      = cfg.NC_SCALE,
    eps_void      = cfg.EPS_VOID,
    eps_void_ucut   = cfg.EPS_VOID_UCUT,
    eps_void_uwidth = cfg.EPS_VOID_UWIDTH,
    eps_void_scale  = cfg.EPS_VOID_SCALE,
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
    alpha_g0      = alpha_g0,
    quality       = QUALITY,
    slip          = SLIP,
    U_h_inlet     = U_h_inlet,
    p_out         = cfg.P_OUT,
    p_out_relax   = cfg.P_OUT_RELAX,
    eos_project   = PROJECT,
    use_saturation = cfg.MASS_TRANSFER,    # only the mass-transfer closure reads
                                           # saturation; two-component/area-only skip it
    return_debug  = False,
)
step, compute_dt = _ret

print("  Warming up compute_dt / step (triggers full compilation)...")
dt0 = float(compute_dt(W_0))
W_test, ssr_test = step(W_0, jnp.float64(dt0))
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
_DT_MIN   = dx / 1e12
dt        = float(compute_dt(W_0))

print(f"\nRunning (max {cfg.N_STEPS} steps  SSR_tol={cfg.SSR_TOL:.1e})")
print(f"{'Step':>6}  {'t [s]':>12}  {'dt [s]':>12}  {'SSR':>12}")
print("-" * 50)


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
    # DOM_ONLY_DT: gate dt on the DOMINANT phase only (exclude the ghost's u/h).
    if DOM_ONLY_DT:
        du = dug if dom == "gas" else dul
        dh = dhg if dom == "gas" else dhl
    else:
        du = max(dug, dul);  dh = max(dhg, dhl)
    return bad, dp, du, dh, dag, (dug, dul, dhg, dhl)


_t0 = _time.perf_counter()
try:
    while n < cfg.N_STEPS:
        halvings = 0
        if _adaptive:
            dt_cap = float(compute_dt(W_n))
            a_char = cfg.CFL * dx / max(dt_cap, 1e-300)
            dt     = min(dt * cfg.DT_GROW, dt_cap)
            if dt < _DT_MIN:
                _stop_reason = f"dt collapsed to {dt:.2e}"
                break
            _per = (0, 0, 0, 0)
            for halvings in range(cfg.MAX_HALVINGS + 1):
                W_try, ssr_j = step(W_n, jnp.float64(dt))
                jax.block_until_ready(W_try)
                bad, dp, du, dh, da, _per = _increments(W_n, W_try, a_char)
                ok = (not bad) and dp < cfg.DP_MAX and dh < cfg.DH_MAX \
                    and du < cfg.DU_MAX and da < cfg.DA_MAX
                if ok:
                    break
                if halvings < cfg.MAX_HALVINGS:
                    dt *= 0.5
            # When halving, show which field/phase blew the threshold — dug/dhg =
            # dominant gas, dul/dhl = ghost.  If dul/dhl dominate, the GHOST gates dt.
            if halvings > 0:
                _dug, _dul, _dhg, _dhl = _per
                print(f"        [halve x{halvings}] dp={dp:.2f} da={da:.3f} | "
                      f"dug={_dug:.2f} dul={_dul:.2f}  dhg={_dhg:.3f} dhl={_dhl:.3f}"
                      f"  ({'GHOST-gated' if max(_dul,_dhl) > max(_dug,_dhg) else 'gas-gated'})",
                      flush=True)
            W_n = W_try
            ssr = (max(float(ssr_j), cfg.SSR_TOL * 10)
                   if halvings >= cfg.MAX_HALVINGS else float(ssr_j))
        else:
            dt         = float(compute_dt(W_n))
            W_n, ssr_j = step(W_n, jnp.float64(dt))
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
            print(f"{n:>6}  {t_phys:>12.4e}  {dt:>12.4e}  {ssr:>12.4e}  ({_el:.1f}s){_hv}",
                  flush=True)

        if ssr < cfg.SSR_TOL:
            _stop_reason = f"SSR={ssr:.3e} < tol={cfg.SSR_TOL:.1e}"
            break
        if n >= _win:
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
