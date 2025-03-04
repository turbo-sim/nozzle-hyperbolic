import numpy as np
import math
from .flux import *
import time
import nozzle_hyperbolic as hyp
import barotropy as bpy
 
def euler_eigenvalues(Q, fluid, parameters):
    # Extract variables from the state vector Q
    rho = Q[0, :]        # Density
    rhou = Q[1, :]       # Momentum (rho * u)
    p = parameters['p']  # Pressure, pre-computed before calling this function
    # Compute velocity
    u = rhou / rho       # Velocity
 
    c = np.zeros_like(rho)  # Speed of sound array
    # Loop through each cell to calculate speed of sound using CoolProp
    for i in range(len(rho)):        
        # Set the fluid state using density and internal energy
        state = fluid.get_state(bpy.fluid_properties.DmassP_INPUTS, rho[i], p[i])
 
        # Retrieve speed of sound for this state
        c[i] = state.speed_sound
 
    # Calculate the eigenvalues
    a = np.array([u - c, u, u + c])
 
    return a
 
def euler_eigenvalues_perfect(Q, gamma):
    # Extract variables from the state vector Q
    rho = Q[0, :]        # Density
    rhou = Q[1, :]       # Momentum (rho * u)
    rhoE = Q[2, :]       # Total energy
 
    # Compute velocity, pressure, and speed of sound
    u = rhou / rho       # Velocity
    p = (gamma - 1) * (rhoE - 0.5 * (rhou ** 2) / rho)  # Pressure
    c = np.sqrt(gamma * p / rho)  # Speed of sound
 
    # Return the eigenvalues (u-c, u, u+c)
    a = np.array([u - c, u, u + c])
 
    return a
 
 
def euler_flux(Q, p):
    # Extract variables from the state vector Q
    rho = Q[0, :]        # Density
    rhou = Q[1, :]       # Momentum (rho * u)
    rhoE = Q[2, :]       # Total energy
   
    # Compute velocity and check consistency
    u = rhou / rho       # Velocity
 
    # Compute the flux vector
    f = np.array([
        rhou,                    # Flux for mass conservation
        rhou * u + p,            # Flux for momentum conservation
        (rhoE + p) * u           # Flux for energy conservation
    ])
 
    return f
 
def euler_flux_perfect(Q, gamma):
    # Extract variables from the state vector Q
    rho = Q[0, :]        # Density
    rhou = Q[1, :]       # Momentum (rho * u)
    rhoE = Q[2, :]       # Total energy
 
    # Compute velocity and pressure
    u = rhou / rho       # Velocity
    p = (gamma - 1) * (rhoE - 0.5 * (rhou ** 2) / rho)  # Pressure
 
    # Compute the flux vector
    f = np.array([
        rhou,                    # Flux for mass conservation
        rhou * u + p,            # Flux for momentum conservation
        (rhoE + p) * u           # Flux for energy conservation
    ])
 
    return f
 
def fvm_euler_1d(x, t, parameters):
    # Extend the grid to ghost cells
    dx_const = parameters['dx']
    A_nodes = parameters['A_nodes']
    # rx = parameters['r_function']
    roughness = parameters['roughness']
    D_hydraulic = parameters['D_hydraulic']
 
    # A_nodes = rx(x)**2 * np.pi
    Nj = len(x)
    # Initialize the time vector
    t_end = t[-1]
    t = np.zeros(1000)
    dt = np.sqrt(np.finfo(float).eps)
    parameters['dt'] = dt
 
    # Initialize the solution
    N = parameters['N_eq']  # Number of equations of the system
    Q = np.zeros((N, Nj, 1000))  # Preallocate space for the solution
    Q_o = parameters['Q_o']  # Load the initial condition
    # Q[:, :, 0] = Q_o (x) # Set the initial condition
    Q[:, :, 0] = Q_o[:,:] # Set the initial condition

 
    # Initialize boundary conditions
    p_stagnation = parameters['pT_stagnation'][0]
    T_stagnation = parameters['pT_stagnation'][1]
    p_out = parameters['p_out']
    perfect_gas = parameters["perfect_gas"]
 
    if perfect_gas:
        cp, cv = hyp.fvm_solver.specific_heat2()
        R_const = cp-cv
        gamma = cp/cv
    else:
        fluid_name = parameters['fluid_name']
        fluid = bpy.fluid_properties.Fluid(fluid_name)
        stagnation_state = fluid.get_state(bpy.fluid_properties.PT_INPUTS, p_stagnation, T_stagnation)
        s0 = stagnation_state.s
        h0 = stagnation_state.h
        parameters["s0"] = s0
        s_critical = fluid.critical_point.s
 
    rhoT_guess_metastable = [stagnation_state["rho"], stagnation_state["T"]]
    rhoT_guess_equilibrium = [stagnation_state["rho"], stagnation_state["T"]]
    phase_change = "condensation"
    Q_onset = 0.97
    Q_width = 0.05
    parameters["rhoT_guess_metastable"] = rhoT_guess_metastable
    parameters["rhoT_guess_equilibrium"] = rhoT_guess_equilibrium
 
    # Define threshold for steady state
    steady_state_threshold = parameters["residuals"]
    max_iterations = parameters["max_iterations"]
    n = 0
    warmup_iterations = 50
    try:
        while t[n] <= t_end and n < max_iterations:
            if n + 1 >= Q.shape[2]:  # Expand Q and t if the current allocation is exhausted
                Q = np.concatenate((Q, np.zeros((N, Nj, 100))), axis=2)
                t = np.concatenate((t, np.zeros(100)))
            
            # Extrapolate linearly the inlet velocity
            u_in = 2 * Q[1, 1, n] / Q[0, 1, n] - Q[1, 2, n] / Q[0, 2, n]
            # u_in = Q[1, 1, n] / Q[0, 1, n]
    
            if perfect_gas:
                T_static = T_stagnation - 0.5*u_in**2*(gamma-1)/(gamma*R_const)
                M_in = np.sqrt(2*((T_stagnation/T_static)-1)/(gamma-1))
                p_static = p_stagnation / (1 + (gamma - 1) / 2 * M_in**2)**(gamma / (gamma - 1))
                rho_static = p_static / (R_const * T_static)
    
                # Inlet boundary condition
                Q[0, 0, n] = rho_static
                Q[1, 0, n] = rho_static * u_in
                Q[2, 0, n] = p_static / (gamma - 1) + 0.5 * rho_static * u_in**2
    
                # Update density (Q[0, :, :]) and momentum (Q[1, :, :]) at the outlet
                Q[0, -1, n] = 2 * Q[0, -2, n] * A_nodes[-2] / A_nodes[-1] - Q[0, -3, n] * A_nodes[-3] / A_nodes[-1]
                Q[1, -1, n] = 2 * Q[1, -2, n] * A_nodes[-2] / A_nodes[-1] - Q[1, -3, n] * A_nodes[-3] / A_nodes[-1]
                # Q[0, -1, n] = Q[0, -2, n]
                # Q[1, -1, n] = Q[1, -2, n]
                u_outlet = Q[1, -1, n]/Q[0, -1, n]
                rho_outlet = Q[0, -1, n]
                # p_outlet = (gamma - 1) * (Q[2, -1, n] - 0.5 * Q[1, -1, n]**2 / Q[0, -1, n])
                c_outlet = np.sqrt(gamma*p_out/rho_outlet)
                Ma_outlet = u_outlet/c_outlet
                Q[2, -1, n] = p_out / (gamma - 1) + 0.5 * Q[1, -1, n]**2 / Q[0, -1, n]
    
                # # Check if the flow is subsonic or supersonic at the outlet
                # if Ma_outlet < 1:
                #     # Subsonic flow: impose outlet pressure condition
                #     Q[2, -1, n] = p_out / (gamma - 1) + 0.5 * Q[1, -1, n]**2 / Q[0, -1, n]
                # else:
                #     # Supersonic flow: outflow boundary condition
                #     Q[2, -1, n] = 2 * Q[2, -2, n] * A_nodes[-2] / A_nodes[-1] - Q[2, -3, n] * A_nodes[-3] / A_nodes[-1]
    
                # Compute source term
                p = (gamma - 1) * (Q[2, :, n] - 0.5 * Q[1, :, n]**2 / Q[0, :, n])
                tau_w = np.zeros(Q.shape[1])
    
            else:
                #Compute the inlet Mach number and static inlet conditions
                h = h0 - 0.5*u_in**2
                static_state = fluid.get_state(bpy.fluid_properties.HmassSmass_INPUTS, h, s0)
                rho_static = static_state.rhomass
                e_inlet = static_state.umass
                # a_static = static_state.speed_sound
                # M_in = u_in / a_static
    
                # Inlet boundary condition
                Q[0, 0, n] = rho_static
                Q[1, 0, n] = rho_static * u_in
                Q[2, 0, n] = static_state.umass * rho_static + 0.5 * rho_static * u_in**2  # Total energy (internal + kinetic)
    
                # Outlet boundary condition
                Q[0, -1, n] = 2 * Q[0, -2, n] * A_nodes[-2] / A_nodes[-1] - Q[0, -3, n] * A_nodes[-3] / A_nodes[-1]
                Q[1, -1, n] = 2 * Q[1, -2, n] * A_nodes[-2] / A_nodes[-1] - Q[1, -3, n] * A_nodes[-3] / A_nodes[-1]      
    
                # Calculate outlet Mach number
                u_outlet = Q[1, -1, n] / Q[0, -1, n]  # Velocity at the outlet
                rho_outlet = Q[0, -1, n]              # Density at the outlet
                e_outlet = (Q[2, -1, n] - 0.5 * rho_outlet * u_outlet**2)/rho_outlet
    
                # outlet_state = fluid.get_state(bpy.fluid_properties.DmassUmass_INPUTS, rho_outlet, e_outlet)
                # a_outlet = outlet_state.speed_sound       # Speed of sound at the outlet
                # Ma_out = u_outlet / a_outlet              # Mach number at the outlet
    
                p = np.zeros(Q.shape[1])  # Initialize pressure array
                c = np.zeros(Q.shape[1])  # Initialize pressure array
                tau_w = np.zeros(Q.shape[1])
                LM_vec = np.zeros(Q.shape[1])
                v_quality = np.zeros(Q.shape[1])
    
                LM_count =0
                count = 0
                for j in range(Q.shape[1]):
                    rho = Q[0, j, n]  # Density at cell j
                    u = Q[1, j, n] / rho  # Velocity at cell j
    
                    try:
                        # # Calculate specific internal energy (from total energy minus kinetic energy)
                        e_internal = Q[2, j, n]/ rho - 0.5 * u**2
                        # # Set the fluid state using density and internal energy
                        # internal_state = fluid.get_state(bpy.fluid_properties.DmassSmass_INPUTS, rho, s0)
    
                        internal_state = fluid.get_state(bpy.fluid_properties.DmassUmass_INPUTS, rho, e_internal)
                        # props_blend, props_eq, props_meta = fluid.get_state_blending(
                        #     prop_1="rho",
                        #     prop_1_value=rho,
                        #     prop_2="u",
                        #     prop_2_value=e_internal,
                        #     rhoT_guess_equilibrium=rhoT_guess_equilibrium,
                        #     rhoT_guess_metastable=rhoT_guess_metastable,
                        #     phase_change=phase_change,
                        #     blending_variable="Q",
                        #     blending_onset=Q_onset,
                        #     blending_width=Q_width,
                        #     generalize_quality=True,
                        #     supersaturation=True,
                        #     solver_tolerance=1e-9,
                        #     print_convergence=False,
                        #     solver_algorithm="lm",
                        # )
    
                        # # Update initial guess
                        # rhoT_guess_metastable = [props_meta["rho"], props_meta["T"]]
                        # rhoT_guess_equilibrium = [props_eq["rho"], props_eq["T"]]
        
                    except Exception as e:
                        print(f"Warning: {e} at node {x[j]}, density is ", rho)
                        #  e_internal = e_back_up
                        #  rho = rho_back_up
                        #  internal_state = fluid.get_state(bpy.fluid_properties.DmassUmass_INPUTS, rho, e_internal)
    
                    # Retrieve the pressure and speed of sound from the state
                    p[j] = internal_state.p
                    c[j] = internal_state.speed_sound
                    # p[j] = props_blend.p
                    # c[j] = props_blend.speed_sound
    
                    v_quality[j] = internal_state.Q
    
                    s_critical = fluid.critical_point.s
    
                    if 0 < v_quality[j] < 1 and n > warmup_iterations:
                        G_tot = u*rho
                        # print(s_critical)
                        # print(internal_state.s)
    
                        if internal_state.s > s_critical: # Condensation
                            # G_v = G_tot*v_quality[j]
    
    
                            # temporary_state = fluid.get_state(bpy.fluid_properties.PQ_INPUTS, p[j], 1)  
                            # mu_v = temporary_state.mu
                            # rho_v = temporary_state.rho
                            # G_v = u*rho_v
                            # temporary_state = fluid.get_state(bpy.fluid_properties.PQ_INPUTS, p[j], 0)  
                            # mu_L = temporary_state.mu
                            # rho_L = temporary_state.rho
                            # Re_v = G_v*D_hydraulic[j]/mu_v
                            # Cf_v = Haaland(Re_v, roughness, D_hydraulic[j])
                            # tau_v = Cf_v*(G_v**2)/(rho_v * 8)

                            # Re_L = G_tot*D_hydraulic[j]/mu_L
                            # Cf_L = Haaland(Re_L, roughness, D_hydraulic[j])
                            # tau_L = Cf_L*((G_tot*(1 - v_quality[j]))**2)/(rho_L * 8) 
                            
                            # # # Richardson
                            # # LM = richardson(v_quality[j], rho_L, rho_v)

                            # # Gronnerud
                            # LM = LM_gronnerud(rho_L, rho_v, mu_L, mu_v, v_quality[j], G_tot, D_hydraulic[j])
                            # # LM_vec[j] = LM
                            # tau_w[j] = tau_L * LM

                            # G_tot = u*rho
                            # G_L = G_tot*(1-v_quality[j])
                            # temporary_state = fluid.get_state(bpy.fluid_properties.PQ_INPUTS, p[j], 0)  
                            # mu_L = temporary_state.mu
                            # rho_L = temporary_state.rho
                            # temporary_state = fluid.get_state(bpy.fluid_properties.PQ_INPUTS, p[j], 1)  
                            # mu_v = temporary_state.mu
                            # rho_v = temporary_state.rho
 
                            # Re_L = G_tot*D_hydraulic[j]/mu_L
                            # Cf_L = Haaland(Re_L, roughness, D_hydraulic[j])
                            # tau_L = Cf_L*G_L**2/(rho_L*8)
                            # LM = LM_gronnerud(rho_L, rho_v, mu_L, mu_v, v_quality[j], G_tot, D_hydraulic[j])
                            # # LM = richardson(v_quality[j], rho_L, rho_v)
                            # tau_w[j] = LM * tau_L * 4
                            
                            # G_v = G_tot*v_quality[j]
                            # temporary_state = fluid.get_state(bpy.fluid_properties.PQ_INPUTS, p[j], 1)  
                            # mu_v = temporary_state.mu
                            # rho_v = temporary_state.rho
                            # temporary_state = fluid.get_state(bpy.fluid_properties.PQ_INPUTS, p[j], 0)  
                            # mu_L = temporary_state.mu
                            # rho_L = temporary_state.rho
                            # phi_L, phi_V, tau_L, tau_V = LM_condensation(Re, rho_L, mu_L, rho_v, mu_v, D_hydraulic[j], u, v_quality[j], G_tot)
                            # tau_w[j] = 2.5*tau_V*phi_V

                            # HEM Purdue
                            ts_liquid = fluid.get_state(bpy.fluid_properties.PQ_INPUTS, p[j], 0)  
                            mu_L = ts_liquid.mu
                            rho_L = ts_liquid.rho
                            ts_vapour = fluid.get_state(bpy.fluid_properties.PQ_INPUTS, p[j], 1) 
                            mu_v = ts_vapour.mu
                            rho_v = ts_vapour.rho
                            viscosity = viscosity_Beattie(mu_L, mu_v, rho_L, rho_v, v_quality[j])
                            Re = rho*u*D_hydraulic[j]/viscosity
                            Cf = Haaland(Re, roughness, D_hydraulic[j])
                            # Cf = 4 * f_HEM_purdue(Re)
                            tau_w[j] = Cf*rho*u**2 / 8

                            # # HEM Purdue
                            # ts_liquid = fluid.get_state(bpy.fluid_properties.PQ_INPUTS, p[j], 0)  
                            # mu_L = ts_liquid.mu
                            # rho_L = ts_liquid.rho
                            # ts_vapour = fluid.get_state(bpy.fluid_properties.PQ_INPUTS, p[j], 1)
                            # mu_v = ts_vapour.mu
                            # rho_v = ts_vapour.rho
                            # viscosity = viscosity_Beattie(mu_L, mu_v, rho_L, rho_v, v_quality[j])
                            # Re = rho*u*D_hydraulic[j]/viscosity
                            # # Cf Haaland
                            # # Cf = Haaland(Re, roughness, D_hydraulic[j])
                            # # Cf Purdue
                            # Cf = 4 * f_HEM_purdue(Re)
    
                            # tau_w[j] = Cf*rho*u**2/8
    
                        else: # Flashing
                            print("Error")
                            # G_tot = u*rho
                            # G_L = G_tot*(1-v_quality[j])
                            # temporary_state = fluid.get_state(bpy.fluid_properties.PQ_INPUTS, p[j], 0)  
                            # mu_L = temporary_state.mu
                            # rho_L = temporary_state.rho
                            # temporary_state = fluid.get_state(bpy.fluid_properties.PQ_INPUTS, p[j], 1)  
                            # mu_v = temporary_state.mu
                            # rho_v = temporary_state.rho

                            G_v = G_tot*v_quality[j]
                            temporary_state = fluid.get_state(bpy.fluid_properties.PQ_INPUTS, p[j], 1)  
                            mu_v = temporary_state.mu
                            rho_v = temporary_state.rho
                            temporary_state = fluid.get_state(bpy.fluid_properties.PQ_INPUTS, p[j], 0)  
                            mu_L = temporary_state.mu
                            rho_L = temporary_state.rho
                            phi_L, phi_V, tau_L, tau_V = LM_condensation(Re, rho_L, mu_L, rho_v, mu_v, D_hydraulic[j], u, v_quality[j], G_tot)
                            tau_w[j] = 0.25*tau_L*phi_L
    
                            # Re_L = G_L*D_hydraulic[j]/mu_L
                            # Cf_L = Haaland(Re_L, roughness, D_hydraulic[j])
                            # tau_L = Cf_L*G_L**2/(rho_L * 8)
                            # # LM = richardson(v_quality[j], rho_L, rho_v)
                            # LM = LM_gronnerud(rho_L, rho_v, mu_L, mu_v, v_quality[j], G_tot, D_hydraulic[j])
                            # tau_w[j] = LM * tau_L

                            # # HEM Purdue
                            # ts_liquid = fluid.get_state(bpy.fluid_properties.PQ_INPUTS, p[j], 0)  
                            # mu_L = ts_liquid.mu
                            # rho_L = ts_liquid.rho
                            # ts_vapour = fluid.get_state(bpy.fluid_properties.PQ_INPUTS, p[j], 1) 
                            # mu_v = ts_vapour.mu
                            # viscosity = viscosity_Beattie(mu_L, mu_v, rho_L, rho_v, v_quality[j])
                            # Re = rho*u*D_hydraulic[j]/viscosity
                            # # Cf Haaland
                            # Cf = Haaland(Re, roughness, D_hydraulic[j])
                            # # Cf Purdue
                            # # Cf = f_HEM_purdue(Re)

                            # tau_w[j] = Cf*rho*u**2 / 8 
    
                    else:
                        # viscosity = internal_state.mu
                        # Re = rho*u*D_hydraulic[j]/viscosity
                        # Cf = Haaland(Re, roughness, D_hydraulic[j])
                        # tau_w[j] = Cf*rho*u**2 / 8

                        viscosity = internal_state.mu
                        Re = rho*u*D_hydraulic[j]/viscosity
                        Cf = Haaland(Re, roughness, D_hydraulic[j])
                        tau_w[j] = Cf*rho*u**2/8
    
                # # Compute the average tau_w and print it (TurboExpo 2025)
                # if count > 0:
                #     tau_w_avg = LM_count/ count
                #     print("Average tau_w:", tau_w_avg)
                #     print("max", np.max(LM_vec))
                # else:
                #     print("No valid tau_w values found for averaging.")
    
                # Q[2, -1, n] = 2 * Q[2, -2, n] * A_nodes[-2] / A_nodes[-1] - Q[2, -3, n] * A_nodes[-3] / A_nodes[-1]  
                outlet_state = fluid.get_state(bpy.fluid_properties.DmassP_INPUTS, rho_outlet, p_out)
                e_outlet = outlet_state.umass  # Internal energy per unit mass
                Q[2, -1, n] = e_outlet * rho_outlet + 0.5 * rho_outlet * u_outlet**2  # Total energy
                p[-1] = p_out
    
                # # Subsonic Outlet Boundary Condition
                # if Ma_out < 1:
                #     # Set outlet pressure (subsonic flow assumes we control the pressure at the outlet)
                #     outlet_state = fluid.get_state(bpy.fluid_properties.DmassP_INPUTS, rho_outlet, p_out)
                #     e_outlet = outlet_state.umass  # Internal energy per unit mass
                #     Q[2, -1, n] = e_outlet * rho_outlet + 0.5 * rho_outlet * u_outlet**2  # Total energy
                #     p[-1] = p_out
                # else:
                #     # For supersonic flow, simply extrapolate values from the interior
                #     Q[2, -1, n] = 2 * Q[2, -2, n] * A_nodes[-2] / A_nodes[-1] - Q[2, -3, n] * A_nodes[-3] / A_nodes[-1]  
    
    
            parameters['p'] = p
            # parameters['c'] = c
    
            # Compute source term
            S = np.zeros((3, Nj))
            S[1, 1:Nj-1] = p[1:Nj-1] * (A_nodes[2:] - A_nodes[:-2]) / (2 * dx_const) - tau_w[1:Nj-1]*4*A_nodes[1:Nj-1]/D_hydraulic[1:Nj-1]
            S[1, 0] = p[0] * (A_nodes[1] - A_nodes[0]) / dx_const - tau_w[0]*4*A_nodes[0]/D_hydraulic[0]
            S[1, -1] = p[-1] * (A_nodes[-1] - A_nodes[-2]) / dx_const - tau_w[-1]*4*A_nodes[-1]/D_hydraulic[-1]
            # S[1, 1:Nj-1] = p[1:Nj-1] * (A_nodes[2:] - A_nodes[:-2]) / (2 * dx_const)
            # S[1, 0] = p[0] * (A_nodes[1] - A_nodes[0]) / dx_const
            # S[1, -1] = p[-1] * (A_nodes[-1] - A_nodes[-2]) / dx_const
    
            # Use the desired time integration method
            if parameters['time_method'] == 'euler':
                flux = R(Q[:, :, n], t[n], x, parameters)  # flux = (F_{i+1/2} - F_{i-1/2}) / dx
                Q[:, :, n+1] = Q[:, :, n] + (dt / A_nodes) * flux + (dt / A_nodes) * S
                # residuals = (Q[:, :, n+1]-Q[:, :, n]) - flux*(dt/A_nodes) - S*(dt/A_nodes)
                # residuals = (Q[:, :, n+1]-Q[:, :, n])/dt - flux*(1/A_nodes) - S*(1/A_nodes)
                # L2_norm_residuals = np.sqrt(np.sum(residuals**2, axis=1) / residuals.shape[1])
                # print(L2_norm_residuals) # Flux residual
    
            elif parameters['time_method'] == 'RK3':
                Q1 = Q[:, :, n] + (dt / A_nodes) * R(Q[:, :, n], t[n], x, parameters)
                Q2 = 3/4 * Q[:, :, n] + 1/4 * Q1 + (dt / A_nodes) / 4 * R(Q1, t[n] + dt/4, x, parameters)
                Q[:, :, n+1] = 1/3 * Q[:, :, n] + 2/3 * Q2 + 2/3 * (dt / A_nodes) * R(Q2, t[n] + dt/3, x, parameters) + (dt / A_nodes) * S
    
            elif parameters['time_method'] == 'RK4':
                k1 = R(Q[:, :, n], t[n], x, parameters)
                k2 = R(Q[:, :, n] + k1 * dt / 2, t[n] + dt / 2, x, parameters)
                k3 = R(Q[:, :, n] + k2 * dt / 2, t[n] + dt / 2, x, parameters)
                k4 = R(Q[:, :, n] + k3 * dt, t[n] + dt, x, parameters)
                Q[:, :, n+1] = Q[:, :, n] + (1/6 * (k1 + 2*k2 + 2*k3 + k4)) * (dt / A_nodes) + (dt / A_nodes) * S
    
            # Calculate the maximum change between iterations
            if n > warmup_iterations:
                # infinite norm
                # delta_Q = np.max(np.abs(Q[:, 1:-1, n+1] - Q[:, 1:-1, n]) / np.abs(Q[:, 1:-1, n]))
    
                # L2 norm
                # Find the index of the minimum value in A_bodes
                # skip_index = np.argmin(A_nodes)
                # start = max(0, skip_index - 50)
                # end = min(Q.shape[1], skip_index + 50 + 1)  # +1 to include the upper bound
                # Q_new= np.concatenate((Q[:, :start, :], Q[:, end:, :]), axis=1)
                # delta_Q = np.abs(Q_new[:, 1:-1, n+1] - Q_new[:, 1:-1, n]) / (np.abs(Q_new[:, 1:-1, n]))
    
                delta_Q = np.abs(Q[:, 1:-1, n+1] - Q[:, 1:-1, n]) / (np.abs(Q[:, 1:-1, n]))
                L2_norm = np.sqrt(np.sum(delta_Q**2, axis=1) / delta_Q.shape[1])
                # print(L2_norm)
                delta_Q = np.max(L2_norm)
    
                if n % 25 == 0:
                    print(f"Iteration {n}: delta_Q = {delta_Q:0.4e}")
                    print(f"Residuals: {L2_norm}")
                    # print(Ma_out)
                    # print(p[-1])
                    print(min(p))
                    np.savez(f"rusanov_N{Nj}_pout_{p_out}_HEM_haal_3roug.npz", h=h, x=x, u=u, p=p, rho=rho)
                    # print(p[np.argmin(A_nodes)])
                    # print(rho_outlet*A_nodes[-1]*u_outlet)
    
                # Check for steady state after warm-up
                if delta_Q < steady_state_threshold:
                    print(f"Steady-state reached after {t[n]} s")
                    break
    
            # Adjust the time step based on the CFL condition for interior cells
            lambda_max = np.max(np.abs(parameters['eigenvalues_function'](Q[:, 2:Nj-2, n])), axis=0)
            # x_face = x[1:Nj-2] + (x[2:Nj-1] - x[1:Nj-2]) / 2
            # dx = x_face[1:] - x_face[:-1]
            dt = parameters['CFL'] / np.max(lambda_max / dx_const)
            parameters['dt'] = dt
    
            # Update the time
            t[n+1] = t[n] + dt
            n += 1

            # Save results at each timestep
            # np.savez(f"rusanov_N{Nj}_B13_CFL09_fHEM_purdue.npz",v=u, h=h, x=x, p=p, rho=rho)
    
    
        # Discard the solution in the ghost cells
        Q = Q[:, :, :n]
        t = t[:n]
        # np.savez("lm_param_hem_b596.npz", LM_vec=LM_vec, x=x)
    
    except KeyboardInterrupt:
        print("Interrupted! Saving final Q before exit...")
        Q = Q[:, :, :n]
        t = t[:n]
        np.savez(f"Q_N{Nj}_{p_out}_HEM_haal_3roug.npz", Q=Q)
    

    return Q, t
 
 
def R(Q, _, x, parameters):
    N, Nj = Q.shape
 
    # Compute the right-hand side using a constant cell reconstruction
    if parameters['reconstruction'] == 'constant':
        Q_L = Q[:, :-1]  # Left face
        Q_R = Q[:, 1:]   # Right face
        F = F_numeric(Q_L, Q_R, parameters)
 
        x_face = x[:-1] + (x[1:] - x[:-1]) / 2
        dx = np.ones((N, 1)) * (x_face[1:] - x_face[:-1])
 
        RHS = np.zeros((N, Nj))
        RHS[:, 1:Nj-1] = -(F[:, 1:] - F[:, :-1]) / dx
 
    return RHS
 
def F_numeric(Q_L, Q_R, parameters):
    perfect_gas = parameters["perfect_gas"]
    if perfect_gas:
        if parameters['space_method'] == 'roe':
            return F_Roe_perfect(Q_L, Q_R, parameters)
        elif parameters['space_method'] == 'rusanov':
            return F_Rusanov_perfect(Q_L, Q_R, parameters)
        elif parameters['space_method'] == 'HLLC':
            return F_HLLC_perfect(Q_L, Q_R, parameters)
        elif parameters['space_method'] == 'HLL':
            return F_HLL_perfect(Q_L, Q_R, parameters)
        elif parameters['space_method'] == 'LF':
            return F_LaxFriedrichs_perfect(Q_L, Q_R, parameters)
        else:
            raise ValueError("Choose a valid space method")
    else:
        if parameters['space_method'] == 'roe':
            return F_Roe(Q_L, Q_R, parameters)
        elif parameters['space_method'] == 'rusanov':
            return F_Rusanov(Q_L, Q_R, parameters)
        elif parameters['space_method'] == 'HLLC':
            return F_HLLC(Q_L, Q_R, parameters)
        elif parameters['space_method'] == 'HLL':
            return F_HLL(Q_L, Q_R, parameters)
        elif parameters['space_method'] == 'LF':
            return F_LaxFriedrichs(Q_L, Q_R, parameters)
        else:
            raise ValueError("Choose a valid space method")
       
 
# def minmod(a, b):
#     """Minmod limiter function for the MUSCL approach."""
#     return np.sign(a) * np.maximum(0, np.minimum(np.abs(a), np.sign(a) * b))
 
 
# def R(Q, _, x, parameters):
#     N, Nj = Q.shape
 
#     # Compute the right-hand side using a constant cell reconstruction
#     if parameters['reconstruction'] == 'constant':
#         # Calculate differences between neighboring cells
#         dq_left = Q[:, 1:] - Q[:, :-1]  # Slope between left and center
#         dq_right = Q[:, 2:] - Q[:, 1:-1]  # Slope between center and right
#         dq_limited = minmod(dq_left[:, :-1], dq_right)  # Apply minmod limiter
 
#         # Define Q_L and Q_R with limited slopes, and keep boundary conditions
#         Q_L = np.zeros((N, Nj - 1))  # Left state on each face
#         Q_R = np.zeros((N, Nj - 1))  # Right state on each face
#         Q_L[:, 1:] = Q[:, 1:-1] - 0.5 * dq_limited  # Internal left state
#         Q_R[:, :-1] = Q[:, 1:-1] + 0.5 * dq_limited  # Internal right state
#         Q_L[:, 0] = Q[:, 0]  # Boundary
#         Q_R[:, -1] = Q[:, -1]  # Boundary
 
#         # Compute fluxes
#         F = F_numeric(Q_L, Q_R, parameters)
 
#         # Compute cell face positions and dx values
#         x_face = x[:-1] + (x[1:] - x[:-1]) / 2
#         dx = np.ones((N, 1)) * (x_face[1:] - x_face[:-1])
 
#         # Compute right-hand side (RHS)
#         RHS = np.zeros((N, Nj))
#         RHS[:, 1:Nj-1] = -(F[:, 1:] - F[:, :-1]) / dx
 
#     return RHS
 
 
def Haaland(Re, epsilon, D):
    """
    Calculate the Darcy friction factor using the Haaland equation.
 
    Parameters:
        Re (float): Reynolds number (dimensionless).
        epsilon (float): Pipe roughness height (m).
        D (float): Pipe diameter (m).
 
    Returns:
        float: Darcy friction factor (dimensionless).
    """
 
    # Calculate terms
    term1 = (epsilon / (3.7 * D))**1.11
    term2 = 6.9 / Re
    log_argument = term1 + term2
 
    # Handle invalid logarithm argument
    if log_argument <= 0:
        return 0  # Friction factor is set to 0 if the argument is invalid
   
    # Compute friction factor
    f = (-1.81 * math.log10(log_argument))**-2
    return f
 
def LM_gronnerud(rho_l, rho_v, mu_l, mu_v, x, G_l, d, g=9.81):
    """
    Calculate the Lockhart-Martinelli parameter (Phi_LM) based on the given parameters.
   
    Parameters:
    rho_l (float): Liquid density (kg/m^3)
    rho_v (float): Vapor density (kg/m^3)
    mu_l (float): Liquid viscosity (Pa·s or kg/m·s)
    mu_v (float): Vapor viscosity (Pa·s or kg/m·s)
    x (float): Void fraction (dimensionless)
    g_l (float): Liquid mass flux (kg/m^2/s)
    g (float): Gravitational acceleration (m/s^2)
    d (float): Pipe diameter (m)
   
    Returns:
    float: Phi_LM (dimensionless)
    """
    # Step 1: Calculate Fr_L
    Fr_l = G_l**2 / (g * d * rho_l**2)
   
    # Step 2: Determine f_FR based on Fr_L
    if Fr_l >= 1:
        f_fr = 1
    else:
        f_fr = Fr_l**0.3 + 0.0055 * (math.log(1 / Fr_l))**2
   
    # Step 3: Calculate (dp/dz)_Fr
    dp_dz_fr = f_fr * (x + 4 * (x**1.8 - x**10 * f_fr**0.5))
   
    # Step 4: Calculate Phi_LM^2
    phi_lm_squared = 1 + dp_dz_fr * (((rho_l / rho_v) / (mu_l / mu_v)**0.25) - 1)
   
    return phi_lm_squared  # Return Phi_LM^2
 
def richardson(x, rho_L, rho_v):
 
    # gamma = 1/(1+((1-x)/x)*(rho_v/rho_L))
    gamma = rho_L * x / (rho_L * x + rho_v * (1 - x))
    phi_lm_squared = (1-gamma)**(-1.75)
   
    return phi_lm_squared

def LM_condensation(Re, rho_L, mu_L, rho_V, mu_V, D_h, u, x, G):
 
    Re_L = rho_L*u*D_h/mu_L
    Re_V = rho_V*u*D_h/mu_V
 
    if Re < 2300:
        f_L = 64/Re_L
        f_V = 64/Re_V
    else:
        f_L = 0.316/Re_L**0.25
        f_V = 0.184/Re_V**0.2         # To check if it is Re_G or Re_V
   
    X = (f_L*rho_V*(1-x)**2)/(rho_L*f_V*x**2)
 
    phi_L = 1 + 20/X + 1/X**2
    phi_V = 1 + 20*X + X**2
 
    tau_L = (f_L*G**2*(1-x)**2)/(8*rho_L)
    tau_V = f_V*G**2*x**2/(8*rho_V)
 
    return phi_L, phi_V, tau_L, tau_V

def f_HEM_purdue(Re):
    if Re < 2000:
        f = 16 / Re
    elif 200 <= Re < 20000:
        f = 0.079 * (Re ** -0.25)
    else:
        f = 0.046 * (Re ** -0.2)
        
    return f


def viscosity_Beattie(mu_L, mu_v, rho_L, rho_v, x):
    # Calculate the mixture viscosity with Beattie formula
    # x: vapour quality
    # y: void fraction
    y = rho_L * x / (rho_L * x + rho_v * (1 - x))
    mu = mu_L * (1 - y) * (1 + 2.5 * y) + mu_v * y
    
    return mu

def viscosity_MacAdams(mu_L, mu_v, x):
    mu = (x/mu_v + (1-x)/mu_L)**(-1)
    return mu
