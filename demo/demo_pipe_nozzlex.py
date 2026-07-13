import time
import jax
import jax.numpy as jnp
import diffrax as dfx

import optimistix as optx
import jaxprop as jxp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from matplotlib import gridspec

jxp.set_plot_options(grid=False)

from nozzlex.two_fluid.nozzle_model_solver import (
    NozzleParams,
    BVPSettings,
    IVPSettings,
    replace_param,
)

from nozzlex.two_fluid.nozzle_model_core import (
    get_pipe,
    nozzle_two_fluid_one_component_autonomous,
    get_speed_of_sound_mixture,
)

# -----------------------------------------------------------------------------
# Main API — uses the lax.cond version for implicit solver compatibility
# -----------------------------------------------------------------------------
def transonic_nozzle_one_component_lax(
    params_model: NozzleParams,
    params_bvp: BVPSettings,
    params_ivp: IVPSettings,
):
    """
    Pipe / nozzle simulation using the lax.cond formulation,
    compatible with implicit ODE solvers (e.g. Kvaerno5).
    """

    fluid_vapor = params_model.fluid1
    fluid_liquid = params_model.fluid2

    print(params_model.T01_in)

    state_vapor_in = fluid_vapor.get_state_saturation(params_model.p0_in)
    state_liquid_in = fluid_liquid.get_state_saturation(params_model.p0_in)
    hin_l = state_liquid_in["enthalpy"]
    hin_v = state_vapor_in["enthalpy"]

    # Start liquid slightly subcooled (metastable)
    state_liquid_in = fluid_liquid.get_state(jxp.HmassP_INPUTS, hin_l + 100e3, params_model.p0_in)

    print(state_liquid_in["T"])

    p_in, rho_l_in, rho_v_in, h_l_in, h_v_in = (
        state_liquid_in["pressure"],
        state_liquid_in["density"],
        state_vapor_in["density"],
        state_liquid_in["enthalpy"],
        state_vapor_in["enthalpy"],
    )

    jax.debug.print("p_in:{} | rho_l_in:{} | h_l_in:{}", p_in, rho_l_in, h_l_in)

    alpha_v_in = 0.2
    alpha_l_in = 1.0 - alpha_v_in

    uv_in = params_model.u1_in
    ul_in = params_model.u2_in

    y_inlet = jnp.array([1e-9, alpha_v_in, alpha_l_in, rho_v_in, rho_l_in, uv_in, ul_in, p_in, h_v_in, h_l_in])

    # --- Fixed row/column scales for A_mat (equilibration from inlet A_mat) ---
    params_model = replace_param(params_model, "row_scales", jnp.ones(9))
    params_model = replace_param(params_model, "col_scales", jnp.ones(9))
    params_model = replace_param(params_model, "det_ref", jnp.float64(1.0))
    out_inlet = nozzle_two_fluid_one_component_autonomous(0.0, y_inlet, params_model)
    A_mat_inlet = out_inlet["A_mat"]
    col_scales_in = jnp.maximum(jnp.max(jnp.abs(A_mat_inlet), axis=0), 1e-30)
    A_col_scaled  = A_mat_inlet / col_scales_in[None, :]
    row_scales_in = jnp.maximum(jnp.max(jnp.abs(A_col_scaled), axis=1), 1e-30)
    params_model = replace_param(params_model, "row_scales", row_scales_in)
    params_model = replace_param(params_model, "col_scales", col_scales_in)
    out_inlet_scaled = nozzle_two_fluid_one_component_autonomous(0.0, y_inlet, params_model)
    det_ref_val = jnp.maximum(jnp.abs(out_inlet_scaled["D"]), 1e-10)
    params_model = replace_param(params_model, "det_ref", det_ref_val)

    # --- State-variable scaling ---
    # Reference scales so that Y_hat = Y / Y_ref is O(1) for all components.
    # This dramatically improves the Jacobian conditioning for the implicit solver.
    #   Y = [x,      αv,  ρv,       ρl,       u,          p,    hv,      hl     ]
    Y_ref = jnp.array([
        params_model.length,   # x ~ L
        1.0,                   # αv already O(1)
        1.0,                   # αl already O(1)
        rho_v_in,              # ρv ~ initial vapor density
        rho_l_in,              # ρl ~ initial liquid density
        uv_in,                 # u  ~ initial velocity
        ul_in,                 # u  ~ initial velocity
        p_in,                  # p  ~ initial pressure
        h_v_in,                # hv ~ initial vapor enthalpy
        h_l_in,                # hl ~ initial liquid enthalpy
    ])

    # --- ODE wrappers with scaling ---
    def ode_full_lax(t, y_hat, args):
        """Evaluate full ODE output from scaled state."""
        y_phys = y_hat * Y_ref
        return nozzle_two_fluid_one_component_autonomous(t, y_phys, args)

    def eval_ode_rhs_scaled(t, y_hat, args):
        """RHS in scaled coordinates: dY_hat/dτ = f(Y_hat * Y_ref) / Y_ref."""
        out = ode_full_lax(t, y_hat, args)
        return out["rhs_autonomous"] / Y_ref

    y_inlet_hat = y_inlet / Y_ref

    t0, t1 = 1e-9, 1e9
    solver = jxp.make_diffrax_solver(params_ivp.solver_name)
    adjoint = jxp.make_diffrax_adjoint(params_ivp.adjoint_name)
    term = dfx.ODETerm(eval_ode_rhs_scaled)
    ctrl = dfx.PIDController(rtol=params_ivp.rtol, atol=params_ivp.atol)

    def eval_end_of_domain_event_scaled(t, y, args, **kwargs):
        y_phys = y * Y_ref
        x = y_phys[0]
        L = args.length
        out = nozzle_two_fluid_one_component_autonomous(0.0, y_phys, args)
        Ma_mix = jnp.min(out.get("Ma_mix"))
        M = 150.0 - Ma_mix
        jax.debug.print("x={x}, L-x={leng},M={D}", x=x, leng=L - x, D=out.get("Ma_mix"))
        conditions = jnp.stack([ 1 - x / L, M])
        return jnp.min(conditions)

    event = dfx.Event(
        cond_fn=eval_end_of_domain_event_scaled,
        root_finder=optx.Newton(rtol=1e-6, atol=1e-6),
    )

    saveat = dfx.SaveAt(t1=True, dense=True, fn=ode_full_lax)
    sol1 = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=1e-9,
        y0=y_inlet_hat,
        args=params_model,
        stepsize_controller=ctrl,
        adjoint=adjoint,
        saveat=saveat,
        event=event,
        max_steps=nsteps,
        # throw=False,
    )

    return sol1, Y_ref, params_model




# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    nsteps = 1000
    case = "a"

    table_dir = "fluid_tables"
    fluid_name = "water"
    metastable_phase = ["liquid", "vapor"]
    backend = "HEOS"
    hl_table = [1000e3, 1800e3]  # J/kg
    pl_table = [5e6, 13e6]       # Pa
    hv_table = [2200e3, 3400e3]
    pv_table = [5e6, 13e6]       # Pa
    N_point_liquid_table = 100
    N_point_vapor_table = 100
    N_p_sat = 150
    gradient_method = "forward"

    P_0_in = 1.2e7

    T_in_a = 578.95
    p_out_a = 6.76e6

    T_in_b = 592.55
    p_out_b = 7.19e6

    if case == "a":
        T_0_in = T_in_a
    else:
        T_0_in = T_in_b

    fluid_liquid = jxp.FluidBicubic(
        fluid_name=fluid_name,
        backend="HEOS",
        h_min=hl_table[0],
        h_max=hl_table[1],
        p_min=pl_table[0],
        p_max=pl_table[1],
        N_h=N_point_liquid_table,
        N_p=N_point_liquid_table,
        table_dir=table_dir,
        gradient_method=gradient_method,
        metastable_phase=metastable_phase[0],
        N_p_sat=N_p_sat,
    )

    fluid_vapor = jxp.FluidBicubic(
        fluid_name=fluid_name,
        backend="HEOS",
        h_min=hv_table[0],
        h_max=hv_table[1],
        p_min=pv_table[0],
        p_max=pv_table[1],
        N_h=N_point_vapor_table,
        N_p=N_point_vapor_table,
        table_dir=table_dir,
        gradient_method=gradient_method,
        metastable_phase=metastable_phase[1],
        N_p_sat=N_p_sat,
    )

    params_model = NozzleParams(
        p0_in=P_0_in,
        T01_in=T_0_in,
        T02_in=T_0_in,
        length=1.0,
        roughness=0e-6,
        Ma_low=0.95,
        Ma_high=1.05,
        heat_transfer=0.0,
        wall_friction=0.0,
        fluid1=fluid_vapor,
        fluid2=fluid_liquid,
        geometry=get_pipe,
        bubble_number=1e10,
        droplet_number=1e8,
        u1_in= 30.01,
        u2_in= 30.0,
    )

    params_bvp = BVPSettings(
        solve_mode="mach_crit",
        num_points=60, 
        rtol=1e-8,
        atol=1e-8,
        max_steps=500,
        jac_mode="bwd",
        verbose=False,
        method="Bisection",
        warmup_method="Dogleg",
        warmup_steps=0,
    )

    params_ivp = IVPSettings(
        solver_name="Kvaerno5",  # Tsit5 | Kvaerno3 | Dopri5 | ImplicitEuler
        adjoint_name="DirectAdjoint",
        number_of_points=60,
        rtol=1e-6,
        atol=1e-6,
    )

    # Solve the problem
    print("\n" + "-" * 60)
    print("Evaluating transonic solution (lax.cond + Kvaerno5)")
    print("-" * 60)
    input_array = jnp.array([0.01]) # 0.17353044358840114
    colors = plt.cm.magma(jnp.linspace(0.2, 0.8, len(input_array)))
    solution_list = []
    out_list = []
    for i, p0_in in enumerate(input_array):
        t0 = time.perf_counter()
        params_model = replace_param(params_model, "Ma_in", p0_in)
        sol, Y_ref, params_model = transonic_nozzle_one_component_lax(params_model, params_bvp, params_ivp)
        dt_ms = (time.perf_counter() - t0) * 1e3
        out = sol.ys
        # Post-process: evaluate dense output on a uniform grid
        # Use a for-loop instead of vmap to avoid OOM (lax.cond doubles memory with vmap)
        ts = jnp.linspace(sol.t0, sol.ts[0], 100)
        out2_hat = [sol.evaluate(t) for t in ts]
        # Unscale from solver coordinates back to physical coordinates
        out2 = [y_hat * Y_ref for y_hat in out2_hat]
        sol2 = [
            nozzle_two_fluid_one_component_autonomous(0.0, y, params_model)
            for y in out2
        ]
        keys = sol2[0].keys()
        out = {
            k: jnp.stack([d[k] for d in sol2])
            for k in keys if jnp.isscalar(sol2[0][k]) or sol2[0][k].shape == ()
        }

        mdot_error = (out["m_dot"].max() - out["m_dot"].min()) / out["m_dot"][0]

        print(
            f"Solution {i} | Solver status {sol.result} | "
            f"steps {int(sol.stats['num_steps']):3d} | "
            f"mdot error {mdot_error:0.2e} | "
        )

        solution_list.append(sol)
        out_list.append(out)

    # Create the figure
    fig = plt.figure(figsize=(5, 10))
    gs = gridspec.GridSpec(5, 1, height_ratios=[3, 3, 3, 3, 3])
    xg = out_list[0]["x"]
    rg = out_list[0]["diameter"] / 2.0
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    ax4 = fig.add_subplot(gs[4], sharex=ax0)
    axs = [ax0, ax1, ax2, ax3, ax4]

    # --- row 1: pressure (bar) ---
    axs[0].set_ylabel("Pressure (bar)")
    for color, val, out in zip(colors, input_array, out_list):
        x = out["x"]
        axs[0].plot(x, out["p"] * 1e-5, linestyle="--", color=color)
        axs[0].plot(
            x,
            out["p"] * 1e-5,
            linestyle="-",
            color=color,
            marker="o",
            markersize="3",
            label=rf"$p_{{0,\mathrm{{in}}}} = {val:0.3f}$",
        )

    print(f"Last pressure:{out["p"][-1]}")

    # --- row 2: speed of sound ---
    axs[1].set_ylabel("Velocity (m/s)")
    for color, val, out in zip(colors, input_array, out_list):
        axs[1].plot(
            out["x"],
            out["v1"],
            color=color,
            marker="o",
            markersize="3",
        )

        axs[1].plot(
            out["x"],
            out["v2"],
            color=color,
            marker="^",
            markersize="3",
        )
    # --- row 3: mass source term ---
    axs[2].set_ylabel("Mass source term")
    for color, val, out in zip(colors, input_array, out_list):
        axs[2].plot(out["x"], out["m_dot_lv"], color=color, markersize=3, marker="o")

    # --- row 4: temperature ---
    axs[3].set_ylabel("Temperature")
    for color, val, out in zip(colors, input_array, out_list):
        axs[3].plot(out["x"], out["T1"], color="red", label="vap")
        axs[3].plot(out["x"], out["T2"], color="blue", label="liq")

    # --- row 5: volume fraction ---
    axs[4].set_ylabel("Vol. Fraction")
    for color, val, out in zip(colors, input_array, out_list):
        axs[4].plot(out["x"], out["alpha1"], markersize=3, marker="o", color="red")
        axs[4].plot(out["x"], out["alpha2"], markersize=3, marker="o", color="blue")

    # --- nozzle geometry ---
    fig2 = plt.figure(figsize=(5, 10))
    plt.fill_between(xg, -rg, +rg, color="lightgrey")
    x_closed = jnp.concatenate([xg, xg[::-1], xg[:1]])
    y_closed = jnp.concatenate([rg, -rg[::-1], rg[:1]])
    plt.plot(x_closed, y_closed, "k", linewidth=1.2)

    # --- p-h diagram ---
    fluid_ph = jxp.Fluid(fluid_name, backend)
    fig_ph, ax_ph = fluid_ph.plot_phase_diagram(
        x_prop="h", y_prop="p", x_scale="linear", y_scale="log",
        plot_critical_point=True, plot_quality_isolines=True,
    )
    for color, val, out in zip(colors, input_array, out_list):
        ax_ph.plot(out["h1"], out["p"], "-o", color=color, markersize=3, label="vapor (h1)")
        ax_ph.plot(out["h2"], out["p"], "-s", color=color, markersize=3, label="liquid (h2)")
    ax_ph.set_xlabel("Enthalpy (kJ/kg)")
    ax_ph.set_ylabel("Pressure (Pa)")
    ax_ph.legend(fontsize=7)
    fig_ph.tight_layout()

    plt.show()
