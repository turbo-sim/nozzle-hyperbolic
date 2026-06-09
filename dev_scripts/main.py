# %%
import os
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=true")
import jax
jax.config.update("jax_enable_x64", True)

# Persist XLA compiled kernels across runs — first run compiles once, all
# subsequent runs (same settings, same Nx) load from cache and skip compilation.
# The cache is keyed on the computation graph + hardware, so it is safe to share.
import pathlib
_cache_dir = pathlib.Path(__file__).parent.parent / ".jax_cache"
_cache_dir.mkdir(exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(_cache_dir))
import yaml
import numpy as np
import matplotlib.pyplot as plt
import nozzle_hyperbolic as hyp
import jaxprop as jxp

hyp.set_plot_options(grid=False)

## Read settings
with open("settings_shock_tube.yaml", 'r') as file:
# with open("settings.yaml", 'r') as file:
# with open("settings_nozzle_validation.yaml", 'r') as file:
# with open("settings_subsonic_pipe.yaml", 'r') as file:
    case_data = yaml.safe_load(file)

hyp.print_dict(case_data)

## Output directory
_geo_fn   = case_data.get('geometry', {}).get('function', 'nozzle')
run_name = (
    f"{_geo_fn}_"
    f"{case_data['numerical_settings']['flux_method']}_"
    f"N{case_data['numerical_settings']['Nx']}"
)
out_dir  = pathlib.Path(run_name)
out_dir.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {out_dir.resolve()}")

## Fluid
perfect_gas = case_data["fluid"]["perfect_gas"]
fluid_name  = case_data["fluid"]["fluid_name"]
use_lut     = case_data["fluid"].get("use_lookup_table", True)
lut_cfg     = case_data["fluid"].get("lut", {})

if perfect_gas:
    fluid = jxp.Fluid(name=fluid_name)
elif use_lut:
    fluid = jxp.FluidBicubic(
        fluid_name = fluid_name,
        backend    = lut_cfg.get("backend",   "HEOS"),
        h_min      = float(lut_cfg.get("h_min",     50e3)),
        h_max      = float(lut_cfg.get("h_max",     850e3)),
        p_min      = float(lut_cfg.get("p_min",     1e4)),
        p_max      = float(lut_cfg.get("p_max",     300e5)),
        N_h        = int(lut_cfg.get("N_h",          200)),
        N_p        = int(lut_cfg.get("N_p",          200)),
        table_dir  = lut_cfg.get("table_dir",  "fluid_tables"),
    )
else:
    fluid = jxp.Fluid(name=fluid_name)   # direct HEOS, no table

## Grid
Nx = case_data["numerical_settings"]["Nx"]
x1 = case_data["numerical_settings"]["x1"]
x2 = case_data["numerical_settings"]["x2"]
x  = np.linspace(x1, x2, Nx)
dx = x[1] - x[0]

## Time
t1 = case_data["numerical_settings"]["t1"]
t2 = case_data["numerical_settings"]["t2"]
t  = np.array([t1, t2])

CFL       = case_data["numerical_settings"]["CFL"]
residuals = float(case_data["numerical_settings"]["residuals"])

## Boundary conditions
p_stagnation  = float(case_data["boundary_conditions"]["p_stagnation"])
T_stagnation  = float(case_data["boundary_conditions"]["T_stagnation"])
pT_stagnation = np.array([p_stagnation, T_stagnation])
p_out         = float(case_data["boundary_conditions"]["p_out"])

## Initial conditions (perfect gas: cp/cv needed for post-processing)
if perfect_gas:
    cp, cv         = hyp.fvm_solver.specific_heat2()
    rho_stagnation = p_stagnation / ((cp - cv) * T_stagnation)

## Geometry
import nozzle_hyperbolic.fvm_solver.geometry as _geom

geo_cfg     = case_data["geometry"].copy()
geo_fn_name = geo_cfg.pop("function")
roughness   = float(geo_cfg.pop("roughness", 0.0))
geo_fn      = getattr(_geom, geo_fn_name)
geo         = geo_fn(x, roughness=roughness,
                     **{k: v for k, v in geo_cfg.items()})

A_nodes     = np.asarray(geo.A)
D_hydraulic = np.asarray(geo.D_h)
nozzle_type = geo_fn_name.replace("get_", "")  # e.g. "shock_tube", "pipe"
r           = geo.D_h / 2.0   # radius / half-height for overlay plots
h           = r

## Parameters dict
parameters = {
    "perfect_gas":    perfect_gas,
    "fluid_name":     fluid_name,
    "fluid":          fluid,
    "residuals":      residuals,
    "max_iterations": case_data["numerical_settings"]["max_iterations"],
    "geometry":       geo,
    "A_nodes":        A_nodes,   # kept for nozzle_exact2 and legacy callers
    "dx":             dx,
    "CFL":            CFL,
    "pT_stagnation":  pT_stagnation,
    "p_out":          p_out,
    "N_eq":           3,
    "space_method":   case_data["numerical_settings"]["flux_method"],
    "time_method":    case_data["numerical_settings"]["time_discretization"],
    "reconstruction":  case_data["numerical_settings"].get("reconstruction", "constant"),
    "initialization":  case_data["numerical_settings"].get("initialization", "simple"),
    "delta_factor":    float(case_data["numerical_settings"].get("delta_factor", 0.1)),
    "newton_iter":     int(case_data["numerical_settings"].get("newton_iter", 20)),
    "warmup_cfg":      case_data["numerical_settings"].get("warmup", None),
    "nozzle_type":     nozzle_type,
    "wall_friction":   bool(case_data["numerical_settings"].get("wall_friction", True)),
}
if perfect_gas:
    parameters["gamma"] = cp / cv
if "shock_tube" in nozzle_type:
    parameters["shock_tube_cfg"] = case_data["shock_tube"]

## Initial condition
print("Computing initialization...")
if "shock_tube" in nozzle_type:
    Q_init = hyp.fvm_solver.shock_tube_Q_initialization(x, A_nodes, parameters)
else:
    Q_init = hyp.fvm_solver.isentropic_initialization(x, A_nodes, parameters)
parameters["Q_o"] = lambda _: Q_init
print("Initialization done.")

# ## Plot initial solution + area
# if perfect_gas:
#     rho0 = Q_init[0]
#     u0   = Q_init[1] / Q_init[0]
#     p0   = (cp / cv - 1) * (Q_init[2] - 0.5 * Q_init[1]**2 / Q_init[0])
# else:
#     rho0  = Q_init[0]
#     u0    = Q_init[1] / Q_init[0]
#     e0    = Q_init[2] / Q_init[0] - 0.5 * u0**2
#     _st0  = hyp.fvm_solver.state_from_rho_u_batch(fluid, rho0, e0)
#     p0    = np.asarray(_st0.pressure, dtype=float)

# fig_init, axes = plt.subplots(2, 2, figsize=(10, 6))
# fig_init.suptitle("Initial solution")

# ax = axes[0, 0]
# ax.plot(x, rho0)
# ax.set_ylabel(r"$\rho$ [kg/m³]")

# ax = axes[0, 1]
# ax.plot(x, u0)
# ax.set_ylabel("u [m/s]")

# ax = axes[1, 0]
# ax.plot(x, p0 / 1e5)
# ax.set_ylabel("p [bar]")

# ax = axes[1, 1]
# ax.plot(x, A_nodes * 1e4)
# ax.plot(x, h * 1e3, "k--", label="r [mm]")
# ax.set_ylabel(r"A [cm²]")
# ax.legend()

# for ax in axes.flat:
#     ax.set_xlabel("x [m]")

# plt.tight_layout()
# plt.show()

## Solve
Q, t = hyp.fvm_solver.fvm_euler_1d(x, t, parameters)
print(f"{parameters['space_method']} computed")

## Post-processing
if perfect_gas:
    density     = Q[0, :, -1]
    velocity    = Q[1, :, -1] / Q[0, :, -1]
    pressure    = (cp / cv - 1) * (Q[2, :, -1] - 0.5 * Q[1, :, -1]**2 / Q[0, :, -1])
    sound_speed = np.sqrt((cp / cv) * pressure / density)
    Ma          = velocity / sound_speed
else:
    density         = Q[0, :, -1]
    velocity        = Q[1, :, -1] / Q[0, :, -1]
    internal_energy = Q[2, :, -1] / Q[0, :, -1] - 0.5 * velocity**2

    _states     = hyp.fvm_solver.state_from_rho_u_batch(fluid, density, internal_energy)
    pressure    = np.asarray(_states.pressure,    dtype=float)
    temperature = np.asarray(_states.T,           dtype=float)
    sound_speed = np.asarray(_states.speed_sound, dtype=float)
    s           = np.asarray(_states.entropy,     dtype=float)
    Qvap        = np.asarray(_states.Q,           dtype=float)
    Z           = np.asarray(_states.Z,           dtype=float)
    Ma          = velocity / sound_speed

## Mass-flux diagnostic  (ρ u A = const for steady inviscid 1D flow)
mdot = density * velocity * A_nodes
mdot_mean = np.mean(mdot)
mdot_rms  = np.sqrt(np.mean((mdot - mdot_mean)**2)) / (abs(mdot_mean) + 1e-300)
print(f"Mass flux:  mean = {mdot_mean:.4e} kg/s  |  RMS variation = {mdot_rms*100:.3f} %")
if mdot_rms > 0.01:
    print("  WARNING: mass flux not conserved (>1 %). Check solver or post-processing.")

## Save solution as CSV
if perfect_gas:
    csv_header = "x,rho,u,p,Ma"
    csv_data   = np.column_stack([x, density, velocity, pressure, Ma])
else:
    csv_header = "x,rho,u,p,T,c,Ma,s,Qvap,Z"
    csv_data   = np.column_stack([x, density, velocity, pressure,
                                   temperature, sound_speed, Ma, s, Qvap, Z])
sol_path = out_dir / "solution.csv"
np.savetxt(sol_path, csv_data, delimiter=",", header=csv_header, comments="")
print(f"Solution saved → {sol_path}")

## Plots
def _add_nozzle_overlay(ax_main, x, r, x2, show_ylabel=False):
    ax2 = ax_main.twinx()
    ax2.plot(x, r, 'k--', linewidth=1.1)
    ax2.set_ylim([0, x2])
    if show_ylabel:
        ax2.set_ylabel('r [m]', color='k')

if perfect_gas:
    total_bc   = np.array([T_stagnation, p_stagnation])
    M_exact, p_exact, u_exact, rho_exact, i_shock = hyp.fvm_solver.nozzle_exact2(x, parameters, total_bc)
    A_cr_index = np.argmin(A_nodes)
    legend_kw  = dict(fontsize='x-small', handlelength=1.2, handletextpad=0.4, borderpad=0.4, frameon=True)

    cases = [
        (r'$\rho / \rho_0$',
         density / rho_stagnation,
         rho_exact[:, 0] / rho_stagnation,
         rho_exact[:, 1] / rho_stagnation,
         rho_exact[:, 2] / rho_stagnation if i_shock is not None else None),
        (r'$p / p_0$',
         pressure / p_stagnation,
         p_exact[:, 0] / p_stagnation,
         p_exact[:, 1] / p_stagnation,
         p_exact[:, 2] / p_stagnation if i_shock is not None else None),
        ('M',
         Ma,
         M_exact[:, 0],
         M_exact[:, 1],
         M_exact[:, 2] if i_shock is not None else None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"Numerical solution – {parameters['space_method']}")
    for k, (ax, (ylabel, num, exact_sub, exact_sup, exact_shock)) in enumerate(zip(axes, cases)):
        ax.set_xlabel('x [m]')
        ax.set_ylabel(ylabel)
        ax.plot(x, num, 'go-', lw=0.75, ms=4, mfc='w', label='Numerical')
        ax.plot(x, exact_sub, 'r--', lw=0.75, label='Exact – subsonic')
        ax.plot(x[A_cr_index:], exact_sup[A_cr_index:], 'b--', lw=0.75, label='Exact – supersonic')
        if exact_shock is not None:
            ax.plot(x[i_shock - 1:], exact_shock[i_shock - 1:], 'k-', lw=0.75, label='Exact – shock')
        ax.legend(loc='upper left', **legend_kw)
        _add_nozzle_overlay(ax, x, r, x2, show_ylabel=(k == len(cases) - 1))

else:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle(f"Numerical solution – {parameters['space_method']}")
    pairs = [
        (axes[0, 0], r'$\rho$ [kg/m³]',  density),
        (axes[0, 1], r'$p$ [bar]',        pressure / 1e5),
        (axes[0, 2], r'$T$ [K]',          temperature),
        (axes[1, 0], 'M',                  Ma),
        (axes[1, 1], r'$s$ [J/(kg·K)]',   s),
        (axes[1, 2], r'$Q_\mathrm{vap}$', Qvap),
    ]
    for k, (ax, ylabel, data) in enumerate(pairs):
        ax.plot(x, data, 'go-', lw=0.75, ms=4, mfc='w')
        ax.set_xlabel('x [m]')
        ax.set_ylabel(ylabel)
        _add_nozzle_overlay(ax, x, r, x2, show_ylabel=(k % 3 == 2))

    fluid = jxp.Fluid(fluid_name)
    try:
        fig_ph, ax_ph = fluid.plot_phase_diagram()
        ax_ph.plot(s, temperature, "o")
    except (AttributeError, Exception):
        # Single-phase fluid (e.g. Air) has no saturation line — plain T-s plot
        fig_ph, ax_ph = plt.subplots(figsize=(6, 5))
        ax_ph.scatter(s, temperature, s=10)
        ax_ph.set_xlabel(r'$s$ [J/(kg·K)]')
        ax_ph.set_ylabel(r'$T$ [K]')
        ax_ph.set_title(f"{fluid_name} — T-s diagram")

fig.tight_layout()
fig.savefig(out_dir / "solution.png", dpi=150, bbox_inches='tight')
if not perfect_gas:
    fig_ph.tight_layout()
    fig_ph.savefig(out_dir / "phase_diagram.png", dpi=150, bbox_inches='tight')
plt.show()

## Symmetric nozzle: compare with perfect-gas exact solution
# Works even for real-gas fluid because Air at p0~2 bar, T0~300 K satisfies Z≈1.
if "symmetric_nozzle" in nozzle_type and not perfect_gas:
    _fluid_pg  = jxp.Fluid(fluid_name)
    _st_in     = _fluid_pg.get_state(jxp.PT_INPUTS, p_stagnation, T_stagnation)
    gamma_eff  = float(_st_in.cpmass / _st_in.cvmass)
    rho_stag   = float(_st_in.rhomass)
    params_ex  = {"gamma": gamma_eff, "p_out": p_out, "A_nodes": A_nodes}
    total_bc_n = np.array([T_stagnation, p_stagnation])
    M_exact, p_exact, u_exact, rho_exact, i_shock = \
        hyp.fvm_solver.nozzle_exact2(x, params_ex, total_bc_n)
    A_cr_index = np.argmin(A_nodes)
    legend_kw  = dict(fontsize='x-small', handlelength=1.2, handletextpad=0.4,
                      borderpad=0.4, frameon=True)
    cases_ex = [
        (r'$\rho\,/\,\rho_0$',
         density / rho_stag,
         rho_exact[:, 0] / rho_stag,
         rho_exact[:, 1] / rho_stag,
         rho_exact[:, 2] / rho_stag if i_shock is not None else None),
        (r'$p\,/\,p_0$',
         pressure / p_stagnation,
         p_exact[:, 0] / p_stagnation,
         p_exact[:, 1] / p_stagnation,
         p_exact[:, 2] / p_stagnation if i_shock is not None else None),
        ('M',
         Ma,
         M_exact[:, 0],
         M_exact[:, 1],
         M_exact[:, 2] if i_shock is not None else None),
    ]
    fig_ex, axes_ex = plt.subplots(1, 3, figsize=(14, 4))
    fig_ex.suptitle(f"FVS vs exact  (γ_eff={gamma_eff:.3f})  —  {nozzle_type},  N={Nx}")
    for k, (ax, (ylabel, num, ex_sub, ex_sup, ex_sh)) in enumerate(zip(axes_ex, cases_ex)):
        ax.set_xlabel('x [m]');  ax.set_ylabel(ylabel)
        ax.plot(x, ex_sub, 'r--', lw=0.75, label='Exact – subsonic')
        ax.plot(x, num,  'g-',  lw=1.75, ms=4, mfc='w', label='FVS')
        ax.plot(x[A_cr_index:], ex_sup[A_cr_index:], 'b--', lw=0.75, label='Exact – supersonic')
        if ex_sh is not None:
            ax.plot(x[i_shock - 1:], ex_sh[i_shock - 1:], 'k-', lw=0.75, label='Exact – shock')
        # ax.legend(loc='center', **legend_kw)
        _add_nozzle_overlay(ax, x, r, x2, show_ylabel=(k == len(cases_ex) - 1))
    fig_ex.tight_layout()
    fig_ex.savefig(out_dir / "exact_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()

    # Determine which exact branch to compare against
    if i_shock is not None:
        col = 2
    elif Ma[-1] > 1.0:
        col = 1
    else:
        col = 0
    print(f"\n  L1 / L2 errors vs exact nozzle solution (γ_eff={gamma_eff:.3f}):")
    for label, num_arr, ex_arr, unit in [
        ("density ", density[1:-1],  rho_exact[1:-1, col], "kg/m³"),
        ("pressure", pressure[1:-1], p_exact[1:-1,   col], "Pa"),
        ("Mach    ", Ma[1:-1],       M_exact[1:-1,   col], "-"),
    ]:
        diff = num_arr - ex_arr
        print(f"  {label}  L1={np.mean(np.abs(diff)):.4e}  "
              f"L2={np.sqrt(np.mean(diff**2)):.4e}  [{unit}]")

## Shock tube: compare with exact Sod solution
if "shock_tube" in nozzle_type:
    from scipy.optimize import brentq as _brentq_sod
    cfg_st   = case_data["shock_tube"]
    x_d      = float(cfg_st.get("x_diaphragm", 0.5 * (x1 + x2)))
    p_L_st   = float(cfg_st["p_L"]);  T_L_st = float(cfg_st["T_L"]);  u_L_st = float(cfg_st.get("u_L", 1.0))
    p_R_st   = float(cfg_st["p_R"]);  T_R_st = float(cfg_st["T_R"]);  u_R_st = float(cfg_st.get("u_R", 1.0))

    # Approximate ideal-gas sound speeds for exact solution
    st_L_ex  = jxp.Fluid(fluid_name).get_state(jxp.PT_INPUTS, p_L_st, T_L_st)
    st_R_ex  = jxp.Fluid(fluid_name).get_state(jxp.PT_INPUTS, p_R_st, T_R_st)
    rho_L_ex = float(st_L_ex.rhomass);  rho_R_ex = float(st_R_ex.rhomass)
    gam_ex   = float(st_L_ex.cpmass / st_L_ex.cvmass)
    cL_ex    = float(st_L_ex.speed_sound);  cR_ex = float(st_R_ex.speed_sound)

    AL = 2 / ((gam_ex + 1) * rho_L_ex);  BL = (gam_ex - 1) / (gam_ex + 1) * p_L_st
    AR = 2 / ((gam_ex + 1) * rho_R_ex);  BR = (gam_ex - 1) / (gam_ex + 1) * p_R_st

    def _fK(p, pK, cK, AK, BK):
        return ((p - pK) * np.sqrt(AK / (p + BK)) if p > pK
                else 2 * cK / (gam_ex - 1) * ((p / pK) ** ((gam_ex - 1) / (2 * gam_ex)) - 1))

    p_star = _brentq_sod(lambda p: _fK(p, p_L_st, cL_ex, AL, BL)
                                  + _fK(p, p_R_st, cR_ex, AR, BR)
                                  + (u_R_st - u_L_st), 1e-6 * p_L_st, 10 * p_L_st)
    u_star = (0.5 * (u_L_st + u_R_st)
              + 0.5 * (_fK(p_star, p_R_st, cR_ex, AR, BR) - _fK(p_star, p_L_st, cL_ex, AL, BL)))
    rhoL_star = (rho_L_ex * (p_star / p_L_st + (gam_ex-1)/(gam_ex+1))
                 / ((gam_ex-1)/(gam_ex+1) * p_star / p_L_st + 1)) if p_star > p_L_st \
                else rho_L_ex * (p_star / p_L_st) ** (1 / gam_ex)
    rhoR_star = (rho_R_ex * (p_star / p_R_st + (gam_ex-1)/(gam_ex+1))
                 / ((gam_ex-1)/(gam_ex+1) * p_star / p_R_st + 1)) if p_star > p_R_st \
                else rho_R_ex * (p_star / p_R_st) ** (1 / gam_ex)
    cL_star = np.sqrt(gam_ex * p_star / rhoL_star)
    S_HL = u_L_st - cL_ex;  S_TL = u_star - cL_star
    S_R  = u_R_st + cR_ex * np.sqrt((gam_ex+1)/(2*gam_ex)*p_star/p_R_st + (gam_ex-1)/(2*gam_ex))

    t_phys = float(t[-1])   # elapsed physical time
    xi_s = (x - x_d) / t_phys if t_phys > 0 else np.zeros_like(x)
    gm1 = gam_ex - 1.0;  gp1 = gam_ex + 1.0
    # rarefaction fan: u, p, rho as functions of xi_s
    u_fan   = 2/gp1 * (cL_ex + gm1/2*u_L_st + xi_s)
    c_fan   = cL_ex + gm1/2*(u_L_st - u_fan)
    p_fan   = p_L_st * (c_fan/cL_ex)**(2*gam_ex/gm1)
    rho_fan = gam_ex * p_fan / c_fan**2
    rho_ex = np.where(xi_s < S_HL, rho_L_ex,
             np.where(xi_s < S_TL, rho_fan,
             np.where(xi_s < u_star, rhoL_star,
             np.where(xi_s < S_R,   rhoR_star, rho_R_ex))))
    u_ex   = np.where(xi_s < S_HL, u_L_st,
             np.where(xi_s < S_TL, u_fan,
             np.where(xi_s < S_R,  u_star, u_R_st)))
    p_ex   = np.where(xi_s < S_HL, p_L_st,
             np.where(xi_s < S_TL, p_fan,
             np.where(xi_s < S_R,  p_star, p_R_st)))

    fig_sod, axes_sod = plt.subplots(1, 3, figsize=(13, 4))
    fig_sod.suptitle(f"Sod shock tube — {parameters['space_method']},  "
                     f"N={Nx},  t={t_phys:.3e} s")
    for ax, ylabel, num, ex in zip(
            axes_sod,
            [r"$\rho$ [kg/m³]", r"$u$ [m/s]", r"$p$ [Pa]"],
            [density, velocity, pressure],
            [rho_ex, u_ex, p_ex]):
        ax.plot(x, ex,  "k-",  lw=1.5, label="Exact")
        ax.plot(x, num, "r--", lw=1.0, label="FVM")
        ax.set_xlabel("x [m]");  ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
    plt.tight_layout()
    fig_sod.savefig(out_dir / "sod_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()

    print("\n  L1 / L2 errors (interior cells, vs exact Sod):")
    for label, num, ex, unit in [
        ("density ", density[1:-1],  rho_ex[1:-1], "kg/m³"),
        ("velocity", velocity[1:-1], u_ex[1:-1],   "m/s"),
        ("pressure", pressure[1:-1], p_ex[1:-1],   "Pa"),
    ]:
        diff = num - ex
        print(f"  {label}  L1={np.mean(np.abs(diff)):.4e}  "
              f"L2={np.sqrt(np.mean(diff**2)):.4e}  [{unit}]")

## Diagnostics
min_index  = np.argmin(A_nodes)
m_dot      = density * velocity * A_nodes          # mass flux [kg/s]
# Exclude BC cells (index 0 and -1): they are set by the boundary conditions,
# not by flux conservation, so their mass flow is not constrained to match
# the interior. Using only interior cells gives the true solver conservation error.
m_dot_int  = m_dot[1:-1]
m_dot_mean = np.mean(m_dot_int)
m_dot_dev  = np.abs(m_dot - m_dot_mean) / np.abs(m_dot_mean) * 100  # % deviation

if perfect_gas:
    gamma = cp / cv
    h_field = gamma / (gamma - 1) * pressure / density
else:
    h_field = internal_energy + pressure / density

H_field   = h_field + 0.5 * velocity**2            # total enthalpy [J/kg]
H_int     = H_field[1:-1]
H_mean    = np.mean(H_int)
H_dev     = np.abs(H_field - H_mean) / np.abs(H_mean) * 100  # % deviation

if "shock_tube" in nozzle_type:
    # Conservation diagnostics assume steady-state flow — not meaningful for a
    # transient shock tube where mass flux and enthalpy vary across the domain.
    print("=" * 55)
    print("  Shock tube — steady-state conservation diagnostics")
    print("  skipped (transient case: mass flux and H not uniform)")
    print("=" * 55)
else:
    print("=" * 55)
    print("  Spatial conservation residuals  (interior cells)")
    print("=" * 55)
    print(f"  Mass flux  — mean: {m_dot_mean:.6g} kg/s")
    print(f"               max deviation: {np.max(m_dot_dev[1:-1]):.3f} %")
    print(f"  Tot. enth. — mean: {H_mean:.6g} J/kg")
    print(f"               max deviation: {np.max(H_dev[1:-1]):.3f} %")
    if not perfect_gas:
        ds = s[-1] - s[0]
        print(f"  Entropy    — inlet: {s[0]:.6g}  outlet: {s[-1]:.6g}  Δs = {ds:+.4g} J/(kg·K)")
    print("=" * 55)
    print(f"  Mass flow rate (outlet): {m_dot[-1]:.6g} kg/s")
    print(f"  Pressure at throat: {pressure[min_index]:.6g} Pa,  outlet: {pressure[-1]:.6g} Pa")
    print("=" * 55)

    fig_res, axes_res = plt.subplots(1, 2, figsize=(10, 4))
    fig_res.suptitle("Spatial conservation residuals (interior cells only)")

    ax = axes_res[0]
    ax.plot(x[1:-1], m_dot_dev[1:-1])
    ax.set_xlabel("x [m]")
    ax.set_ylabel(r"$|\dot{m} - \bar{\dot{m}}| / \bar{\dot{m}}$ [%]")
    ax.set_title("Mass flux deviation")

    ax = axes_res[1]
    ax.plot(x[1:-1], H_dev[1:-1])
    ax.set_xlabel("x [m]")
    ax.set_ylabel(r"$|H - \bar{H}| / \bar{H}$ [%]")
    ax.set_title("Total enthalpy deviation")

    plt.tight_layout()
    fig_res.savefig(out_dir / "residuals.png", dpi=150, bbox_inches='tight')
    plt.show()
