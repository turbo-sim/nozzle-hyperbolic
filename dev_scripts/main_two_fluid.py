# %%
"""
Two-fluid FVS solver — pipe test case.

Loads settings from settings_two_fluid.py, builds EOS tables, runs the
implicit Newton time-stepping loop, and plots the primitive variables.

Run from dev_scripts/:
    python main_two_fluid.py
or interactively cell-by-cell in VS Code (# %% cells).
"""

import os
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=true")
import jax
jax.config.update("jax_enable_x64", True)

import pathlib
_cache_dir = pathlib.Path(__file__).parent.parent / ".jax_cache"
_cache_dir.mkdir(exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(_cache_dir))

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jaxprop as jxp
import nozzle_hyperbolic.fvm_solver.geometry as _geom
from nozzle_hyperbolic.fvm_solver.two_fluid import (
    make_two_fluid_step,
    W_from_U_h_batch,
    U_h_from_W_batch,
)

import settings_two_fluid as cfg

# ── EOS tables ────────────────────────────────────────────────────────────────
# fluid_g covers the vapor region; fluid_l covers the liquid region.
# First run builds tables and caches them to table_dir; subsequent runs load.
print("Building / loading EOS tables...")
fluid_g = jxp.FluidBicubic(**cfg.LUT_G)
fluid_l = jxp.FluidBicubic(**cfg.LUT_L)
print("  Done.")

# ── Grid ─────────────────────────────────────────────────────────────────────
x  = np.linspace(cfg.x1, cfg.x2, cfg.Nx)
dx = float(x[1] - x[0])

geo  = _geom.get_pipe(x, radius=cfg.RADIUS)
A    = np.asarray(geo.A,    dtype=float)   # (N,) cell-centre areas
dAdx = np.asarray(geo.dAdx, dtype=float)   # (N,) dA/dx = 0 for pipe

# Face areas: arithmetic mean of adjacent cell-centre areas
A_face = 0.5 * (A[:-1] + A[1:])           # (N-1,)

print(f"Grid: Nx={cfg.Nx}  dx={dx:.4f} m  A={A[0]:.4e} m²  (pipe r={cfg.RADIUS} m)")

# ── Initial condition ─────────────────────────────────────────────────────────
# Get saturation enthalpies from the tables at P_INIT.
# fluid_g is the vapor table → get_state_saturation returns sat. vapor props.
# fluid_l is the liquid table → get_state_saturation returns sat. liquid props.
if cfg.USE_RIEMANN_IC:
    # ── Riemann IC ────────────────────────────────────────────────────────────
    # Saturation enthalpies at left and right pressures
    _p_L = jnp.array(cfg.P_L);  _p_R = jnp.array(cfg.P_R)
    h_g_L = float(fluid_g.get_state_saturation(_p_L)["enthalpy"])
    h_l_L = float(fluid_l.get_state_saturation(_p_L)["enthalpy"])
    h_g_R = float(fluid_g.get_state_saturation(_p_R)["enthalpy"])
    h_l_R = float(fluid_l.get_state_saturation(_p_R)["enthalpy"])

    # Densities at each saturation state (via EOS tables)
    _rg_L = float(fluid_g.get_state(jxp.HmassP_INPUTS, jnp.array([h_g_L]), jnp.array([cfg.P_L]))["rho"][0])
    _rl_L = float(fluid_l.get_state(jxp.HmassP_INPUTS, jnp.array([h_l_L]), jnp.array([cfg.P_L]))["rho"][0])
    _rg_R = float(fluid_g.get_state(jxp.HmassP_INPUTS, jnp.array([h_g_R]), jnp.array([cfg.P_R]))["rho"][0])
    _rl_R = float(fluid_l.get_state(jxp.HmassP_INPUTS, jnp.array([h_l_R]), jnp.array([cfg.P_R]))["rho"][0])

    _al_L = 1.0 - cfg.ALPHA_G_L
    _al_R = 1.0 - cfg.ALPHA_G_R

    # Left mass fluxes  G = α ρ u  [kg/(m² s)]
    _G_g = cfg.ALPHA_G_L * _rg_L * cfg.U_G_L
    _G_l = _al_L         * _rl_L * cfg.U_L_L

    # Right velocities computed from continuity  u = G / (α ρ)
    _U_G_R = _G_g / (cfg.ALPHA_G_R * _rg_R)
    _U_L_R = _G_l / (_al_R         * _rl_R)

    state_L = np.array([cfg.P_L, cfg.U_G_L, cfg.U_L_L, cfg.ALPHA_G_L, h_g_L, h_l_L])
    state_R = np.array([cfg.P_R, _U_G_R,    _U_L_R,    cfg.ALPHA_G_R, h_g_R, h_l_R])
    U_h_0   = np.where(x[:, None] < cfg.X_DISCONT, state_L, state_R).astype(float)

    print(
        f"Riemann IC  x_discont={cfg.X_DISCONT} m\n"
        f"  L: p={cfg.P_L/1e5:.1f} bar  α_g={cfg.ALPHA_G_L:.2f}"
        f"  u_g={cfg.U_G_L:.2f}  u_l={cfg.U_L_L:.2f} m/s"
        f"  ρ_g={_rg_L:.3f}  ρ_l={_rl_L:.1f} kg/m³\n"
        f"  R: p={cfg.P_R/1e5:.1f} bar  α_g={cfg.ALPHA_G_R:.2f}"
        f"  u_g={_U_G_R:.2f}  u_l={_U_L_R:.2f} m/s  (from continuity)"
        f"  ρ_g={_rg_R:.3f}  ρ_l={_rl_R:.1f} kg/m³\n"
        f"  Mass fluxes: G_g={_G_g:.3f}  G_l={_G_l:.1f} kg/(m²s)"
    )

else:
    # ── Uniform IC ────────────────────────────────────────────────────────────
    _p0 = jnp.array(cfg.P_INIT)
    _sat_g = fluid_g.get_state_saturation(_p0)
    _sat_l = fluid_l.get_state_saturation(_p0)
    h_g_sat = float(_sat_g["enthalpy"])
    h_l_sat = float(_sat_l["enthalpy"])
    T_sat_0 = float(_sat_g["temperature"])
    h_g_0   = h_g_sat + cfg.H_G_OFFSET
    h_l_0   = h_l_sat + cfg.H_L_OFFSET

    print(
        f"Saturation at {cfg.P_INIT/1e5:.1f} bar:  T_sat={T_sat_0:.1f} K"
        f"  h_g_sat={h_g_sat/1e3:.1f} kJ/kg  h_l_sat={h_l_sat/1e3:.1f} kJ/kg"
    )
    if cfg.H_G_OFFSET != 0.0 or cfg.H_L_OFFSET != 0.0:
        print(
            f"  Enthalpy offsets:  Δh_g={cfg.H_G_OFFSET/1e3:+.1f} kJ/kg"
            f"  Δh_l={cfg.H_L_OFFSET/1e3:+.1f} kJ/kg"
            f"  → h_g={h_g_0/1e3:.1f}  h_l={h_l_0/1e3:.1f} kJ/kg"
        )
    U_h_0 = np.tile(
        [cfg.P_INIT, cfg.U_G_INIT, cfg.U_L_INIT, cfg.ALPHA_G, h_g_0, h_l_0],
        (cfg.Nx, 1),
    ).astype(float)
    print(
        f"IC: p={cfg.P_INIT/1e5:.1f} bar  α_g={cfg.ALPHA_G:.2f}"
        f"  u_g={cfg.U_G_INIT:.1f} m/s  u_l={cfg.U_L_INIT:.1f} m/s"
    )

# Convert U_h → W (one EOS call per phase to get ρ_g, ρ_l)
W_0 = W_from_U_h_batch(jnp.array(U_h_0), fluid_g, fluid_l)
print(f"W_0 shape: {W_0.shape}   dtype: {W_0.dtype}")

# Sanity check: round-trip W → U_h should recover the initial state
U_h_check = np.array(U_h_from_W_batch(W_0))
max_err = np.max(np.abs(U_h_check - U_h_0) / (np.abs(U_h_0) + 1e-300))
print(f"Round-trip error  U_h → W → U_h:  max relative = {max_err:.2e}  (expected ~1e-15)")

# ── Initial condition plot ────────────────────────────────────────────────────
if cfg.VISUALIZE_IC:
    _ic = np.array(U_h_0)
    _panels_ic = [
        (r"p [bar]",         _ic[:, 0] / 1e5),
        (r"$\alpha_g$ [-]",  _ic[:, 3]),
        (r"$u_g$ [m/s]",     _ic[:, 1]),
        (r"$u_l$ [m/s]",     _ic[:, 2]),
        (r"$h_g$ [kJ/kg]",   _ic[:, 4] / 1e3),
        (r"$h_l$ [kJ/kg]",   _ic[:, 5] / 1e3),
    ]
    _fig_ic, _axes_ic = plt.subplots(2, 3, figsize=(14, 7))
    _fig_ic.suptitle(
        f"Initial condition — Water at {cfg.P_INIT/1e5:.0f} bar"
        f"  α_g={cfg.ALPHA_G:.2f}  N={cfg.Nx}"
    )
    for _ax, (_ylabel, _data) in zip(_axes_ic.flat, _panels_ic):
        _ax.plot(x[1:-1], _data[1:-1], "bo-", lw=0.8, ms=4, mfc="w", label="Interior")
        _ax.plot(x[[0, -1]], _data[[0, -1]], "r*", ms=9, zorder=5, label="Ghost (BC)")
        _ax.set_xlabel("x [m]")
        _ax.set_ylabel(_ylabel)
        _ax.legend(fontsize="x-small")
    plt.tight_layout()
    plt.show()

# ── Solver factory ────────────────────────────────────────────────────────────
print("\nBuilding solver (first JIT compilation may take a minute)...")
step, compute_dt = make_two_fluid_step(
    fluid_g, fluid_l,
    A, A_face, dAdx, dx,
    CFL          = cfg.CFL,
    k            = cfg.K_VM,
    delta_factor = cfg.DELTA_FACTOR,
    newton_iter  = cfg.NEWTON_ITER,
    recon        = cfg.RECON,
    debug_newton = cfg.DEBUG_NEWTON,
)

# Warm up: trigger JIT compilation on the first call
print("  Warming up compute_dt...")
dt0 = float(compute_dt(W_0))
print(f"  dt = {dt0:.4e} s")
print("  Warming up step (this triggers full Newton + Jacobian compilation)...")
W_test, ssr_test = step(W_0, jnp.float64(dt0))
print(f"  SSR after first step: {float(ssr_test):.4e}")
print("  Compilation done.")

# ── Time-stepping loop ────────────────────────────────────────────────────────
import time as _time

W_n      = W_0
t_phys   = 0.0
n        = 0
ssr      = float("inf")
_win     = cfg.PLATEAU_WINDOW
_ssr_hist = [float("inf")] * _win   # rolling SSR window for plateau detection
_stop_reason = f"max steps ({cfg.N_STEPS}) reached"

print(f"\nRunning (max {cfg.N_STEPS} steps  SSR_tol={cfg.SSR_TOL:.1e}"
      + (f"  t_end={cfg.T_END:.3e} s" if cfg.T_END < float("inf") else "") + ")")
print(f"{'Step':>6}  {'t [s]':>12}  {'dt [s]':>12}  {'SSR':>12}")
print("-" * 50)

_t0 = _time.perf_counter()
try:
    while n < cfg.N_STEPS:
        dt       = float(compute_dt(W_n))
        W_n, ssr_j = step(W_n, jnp.float64(dt))
        ssr      = float(ssr_j)
        t_phys  += dt
        n       += 1
        _ssr_hist = _ssr_hist[1:] + [ssr]

        if n <= 10 or n % cfg.PRINT_EVERY == 0:
            _el = _time.perf_counter() - _t0
            print(f"{n:>6}  {t_phys:>12.4e}  {dt:>12.4e}  {ssr:>12.4e}"
                  f"  ({_el:.1f} s)", flush=True)

        # ── Termination checks ────────────────────────────────────────────
        if ssr < cfg.SSR_TOL:
            _stop_reason = f"SSR={ssr:.3e} < tol={cfg.SSR_TOL:.1e}"
            break

        if n >= _win:
            _hmax = max(_ssr_hist);  _hmin = min(_ssr_hist) + 1e-300
            if _hmax / _hmin < cfg.PLATEAU_RATIO:
                _stop_reason = (f"SSR plateau ({_hmin:.3e}–{_hmax:.3e}"
                                f"  ratio={_hmax/_hmin:.2f})")
                break

        if t_phys >= cfg.T_END:
            _stop_reason = f"t_end={cfg.T_END:.3e} s reached"
            break

except KeyboardInterrupt:
    _stop_reason = f"interrupted by user"

_elapsed = _time.perf_counter() - _t0
print(f"\nStopped: {_stop_reason}")
print(f"  steps={n}  t={t_phys:.3e} s  SSR={ssr:.3e}  wall={_elapsed:.1f} s")

# ── Post-processing ───────────────────────────────────────────────────────────
U_h_f = np.array(U_h_from_W_batch(W_n))

p_f     = U_h_f[:, 0]
u_g_f   = U_h_f[:, 1]
u_l_f   = U_h_f[:, 2]
alpha_g = U_h_f[:, 3]
h_g_f   = U_h_f[:, 4]
h_l_f   = U_h_f[:, 5]

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
_ic_label = (
    f"Riemann  pL={cfg.P_L/1e5:.0f}/pR={cfg.P_R/1e5:.0f} bar"
    if cfg.USE_RIEMANN_IC
    else f"Uniform  p={cfg.P_INIT/1e5:.0f} bar  α_g={cfg.ALPHA_G:.2f}"
)
fig.suptitle(f"Two-fluid pipe — Water  {_ic_label}  N={cfg.Nx}  steps={n}")

panels = [
    (axes[0, 0], "p [bar]",       p_f / 1e5),
    (axes[0, 1], r"$\alpha_g$ [-]", alpha_g),
    (axes[0, 2], "$u_g$ [m/s]",   u_g_f),
    (axes[1, 0], "$u_l$ [m/s]",   u_l_f),
    (axes[1, 1], "$h_g$ [kJ/kg]", h_g_f / 1e3),
    (axes[1, 2], "$h_l$ [kJ/kg]", h_l_f / 1e3),
]
for ax, ylabel, data in panels:
    ax.plot(x[1:-1], data[1:-1], "go-", lw=0.8, ms=4, mfc="w", label="Interior")
    ax.plot(x[[0, -1]], data[[0, -1]], "r*", ms=9, zorder=5, label="Ghost (BC)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize="x-small")

plt.tight_layout()
out_path = pathlib.Path(__file__).parent / "two_fluid_pipe_result.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Plot saved → {out_path}")

# ── Diagnostics ───────────────────────────────────────────────────────────────
# For a uniform stationary state with no sources the steady-state residual
# should be zero (modulo floating-point). A non-zero residual here means a bug
# in fluxes, nc correction, or area-source term.
from nozzle_hyperbolic.fvm_solver.two_fluid import V_from_W
V_f = np.array(jax.vmap(V_from_W)(W_n))

m_g  = V_f[:, 0]
m_l  = V_f[:, 1]
mdot_g = m_g * u_g_f * A
mdot_l = m_l * u_l_f * A

print("\n── Conservative-variable diagnostics (interior cells) ──────────────")
for label, mdot in [("gas  ", mdot_g), ("liquid", mdot_l)]:
    mm = np.mean(mdot[1:-1])
    rms = np.sqrt(np.mean((mdot[1:-1] - mm)**2)) / (abs(mm) + 1e-300)
    print(f"  Mass flux {label}  mean={mm:.4e} kg/s   RMS variation={rms*100:.3f} %")
print("────────────────────────────────────────────────────────────────────")
