# %%
import jax
jax.config.update("jax_enable_x64", True)
import yaml
import numpy as np
import matplotlib.pyplot as plt
import nozzle_hyperbolic as hyp
import jaxprop as jxp

hyp.set_plot_options(grid=False)

## Read settings
with open("settings.yaml", 'r') as file:
    case_data = yaml.safe_load(file)

hyp.print_dict(case_data)

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

## Nozzle geometry
roughness    = case_data["nozzle_parameters"]["roughness"]
nozzle_type  = case_data["nozzle_parameters"]["type"].lower()

file_path  = "Lettieri_nozzle.xlsx"
sheet_name = "Sheet1"
_, _, spline = hyp.fit_nozzle_shape(file_path, sheet_name, degree=10)

r = np.array([spline(xi) if xi >= -0.03201 else 0.006365 for xi in x])
h = r  # half-height (planar) or radius (axisymmetric)

if nozzle_type == "planar":
    width       = float(case_data["nozzle_parameters"]["width"])
    A_nodes     = 2 * r * width
    D_hydraulic = 4 * A_nodes / (2 * (width + 2 * r))
elif nozzle_type == "axisymmetric":
    A_nodes     = np.pi * r**2
    D_hydraulic = 2 * r
else:
    raise ValueError(f"Unknown nozzle type '{nozzle_type}'. Choose 'planar' or 'axisymmetric'.")

## Parameters dict
parameters = {
    "perfect_gas":    perfect_gas,
    "fluid_name":     fluid_name,
    "fluid":          fluid,  # INJECT ENGINE CONTEXT DOWNSTREAM
    "residuals":      residuals,
    "max_iterations": case_data["numerical_settings"]["max_iterations"],
    "A_nodes":        A_nodes,
    "D_hydraulic":    D_hydraulic,
    "roughness":      roughness,
    "dx":             dx,
    "CFL":            CFL,
    "pT_stagnation":  pT_stagnation,
    "p_out":          p_out,
    "N_eq":           3,
    "space_method":   case_data["numerical_settings"]["flux_method"],
    "time_method":    case_data["numerical_settings"]["time_discretization"],
    "reconstruction": case_data["numerical_settings"].get("reconstruction", "constant"),
    "warmup_cfg":     case_data["numerical_settings"].get("warmup", None),
}
if perfect_gas:
    parameters["gamma"] = cp / cv

## Isentropic initialization
print("Computing isentropic initialization...")
Q_init = hyp.fvm_solver.isentropic_initialization(x, A_nodes, parameters)
parameters["Q_o"] = lambda _: Q_init
print("Initialization done.")

## Plot initial solution + area
if perfect_gas:
    rho0 = Q_init[0]
    u0   = Q_init[1] / Q_init[0]
    p0   = (cp / cv - 1) * (Q_init[2] - 0.5 * Q_init[1]**2 / Q_init[0])
else:
    rho0  = Q_init[0]
    u0    = Q_init[1] / Q_init[0]
    e0    = Q_init[2] / Q_init[0] - 0.5 * u0**2
    _st0  = hyp.fvm_solver.state_from_rho_u_batch(fluid, rho0, e0)
    p0    = np.asarray(_st0.pressure, dtype=float)

fig_init, axes = plt.subplots(2, 2, figsize=(10, 6))
fig_init.suptitle("Initial solution")

ax = axes[0, 0]
ax.plot(x, rho0)
ax.set_ylabel(r"$\rho$ [kg/m³]")

ax = axes[0, 1]
ax.plot(x, u0)
ax.set_ylabel("u [m/s]")

ax = axes[1, 0]
ax.plot(x, p0 / 1e5)
ax.set_ylabel("p [bar]")

ax = axes[1, 1]
ax.plot(x, A_nodes * 1e4)
ax.plot(x, h * 1e3, "k--", label="r [mm]")
ax.set_ylabel(r"A [cm²]")
ax.legend()

for ax in axes.flat:
    ax.set_xlabel("x [m]")

plt.tight_layout()
plt.show()

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
    fig_ph, ax_ph = fluid.plot_phase_diagram()
    ax_ph.plot(s, temperature, "o")

plt.tight_layout()
plt.show()

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
plt.show()