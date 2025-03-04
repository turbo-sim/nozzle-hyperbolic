# %%
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nozzle_hyperbolic as hyp
import barotropy as bpy



hyp.set_plot_options(grid=False)
 
# Read YAML file
with open("settings.yaml", 'r') as file:
    case_data = yaml.safe_load(file)
 
# print(data)
hyp.print_dict(case_data)
 
## Define fluid
perfect_gas = case_data["fluid"]["perfect_gas"]
fluid_name = case_data["fluid"]["fluid_name"]
fluid = bpy.fluid_properties.Fluid(fluid_name)
 
## Define numerical setting
Nx = case_data["numerical_settings"]["Nx"]  # Number of cells
x1 = case_data["numerical_settings"]["x1"]    # Left boundary
x2 = case_data["numerical_settings"]["x2"]   # Right boundary
x = np.linspace(x1, x2, Nx)  # x grid vector
# x = x1 + (x2 - x1) * x
dx = x[1] - x[0]
residuals = float(case_data["numerical_settings"]["residuals"])
 
# Define the simulation time
t1 = case_data["numerical_settings"]["t1"]   # Initial time
t2 = case_data["numerical_settings"]["t2"]  # Final time
t = np.array([t1, t2])  # Time vector
 
# Define the CFL condition
CFL = case_data["numerical_settings"]["CFL"]  # Maximum allowed local Courant number
 
## Define boundary conditions
# Inlet
p_stagnation = float(case_data["boundary_conditions"]["p_stagnation"])
T_stagnation = float(case_data["boundary_conditions"]["T_stagnation"])
pT_stagnation = np.array([p_stagnation, T_stagnation])
# Outlet
p_out = float(case_data["boundary_conditions"]["p_out"])
 
## Define initial conditions
p_o = p_stagnation  # bar
T_o = T_stagnation  # K
u_o = 0
 
if perfect_gas:
   cp, cv = hyp.fvm_solver.specific_heat2()
   rho_o = p_o / ((cp-cv) * T_o)
   rho_stagnation = p_stagnation /((cp-cv)*T_stagnation)
   Q_o = lambda x: np.array([rho_o * np.ones_like(x), rho_o * u_o * np.ones_like(x),
                          (p_o / ((cp/cv) - 1) + 0.5 * rho_o * u_o**2) * np.ones_like(x)])
 
else:
    # # Uniform across the nozzle
    # initial_state = fluid.get_state(bpy.fluid_properties.PT_INPUTS, p_o, T_o)
    # rho_initial = initial_state.rho
    # u_internal_initial = initial_state.umass
    # s0 = initial_state.s
    # # parameters['s0'] = s0# Pressure, pre-computed before calling this function
    # total_energy = u_internal_initial + 0.5 * u_o**2
    # Q_o = lambda x: np.array([rho_initial * np.ones_like(x), rho_initial * u_o * np.ones_like(x), rho_initial * total_energy * np.ones_like(x)])
 
    # # Define phase change process
    # if s0 > fluid.critical_point.s:
    #     phase_change = "condensation"
    #     Q_onset = 0.95
    #     Q_width = 0.01
    # else:
    #     phase_change = "evaporation"
    #     Q_onset = 0.01
    #     Q_width = 0.01
 
    # parameters["phase_change"] = phase_change
    # parameters["Q_onset"] = Q_onset
    # parameters["Q_width"] = Q_width
 
    # # Linear across the nozzle
    # def p(x, pin, pout, x0, L):
    #     return pin - (pin - pout) * (x - x0) / (L - x0)
    # stagnation_state = fluid.get_state(bpy.fluid_properties.PT_INPUTS, p_stagnation, T_stagnation)
    # s_o = stagnation_state.s
    # h_o = stagnation_state.h
    # p_o = [p(x_elem, p_stagnation, p_out, x[0], x[-1]) for x_elem in x]
    # rho_initials = np.zeros(len(p_o))
    # u_o = np.zeros(len(p_o))
    # total_energies = np.zeros(len(p_o))
 
    # for i, p in enumerate(p_o):
    #     # Set the state for the given pressure and temperature
    #     print(p,i)
    #     initial_state = fluid.get_state(bpy.fluid_properties.PSmass_INPUTS, p, s_o)
    #     # Extract properties from the state
    #     rho_initial = initial_state.rho
    #     h_internal = initial_state.h
    #     u_internal_initial = initial_state.umass
    #     u_o[i] = np.sqrt(2*(h_o-h_internal))
    #     total_energy = u_internal_initial + 0.5 * u_o[i]**2
    #     # Store the values in the arrays
    #     rho_initials[i] = rho_initial
    #     total_energies[i] = total_energy
 
    # # plt.figure(figsize=(6, 5))
    # # plt.plot(x, u_o)
    # # plt.figure(figsize=(6, 5))
    # # plt.plot(x, p_o)
    # # plt.figure(figsize=(6, 5))
    # # plt.plot(x, rho_initials)
    # # plt.show()
 
    # Q_o = [rho_initials, rho_initials*u_o, rho_initials*total_energies] # To use when linear initialization is chosen
    # Q_o = np.array(Q_o)
 
    # Linear across the nozzle
    # File names

    files = "Q_N1400_5960000.0_HEM_haal_2roug.npz"

    f = np.load(files)
    # vector = f['Q'][0,0,:]
    # print(f['Q'])

    # i = 0
    # for column in vector:
    #     if column == 0:
    #         index = i-1
    #         break
    #     i = i+1

    # print("Last solution:",f['Q'][:,:,index])

 
    # rho_initials = f['Q'][0,:,:]
    # rho_u_initial = f['Q'][1,:,:]
    # rho_e_initial = f['Q'][2,:,:]
 
 
    # plt.figure(figsize=(6, 5))
    # plt.plot(x, rho_initials)
    # plt.figure(figsize=(6, 5))
    # plt.plot(x, rho_u_initial)    
    # plt.figure(figsize=(6, 5))
    # plt.plot(x, rho_e_initial)
    # # plt.show()
 
    Q_o = f['Q'][:,:,-1] # To use when linear initialization is chosen
 
## Define the nozzle
L = x2
roughness = case_data["nozzle_parameters"]["roughness"]
 
# # Polynomial interpolation for nozzle shape
# file_path = "Spinelli_nozzles.xlsx"  # Relative path to the common folder
# sheet_name = "nozzle Ma1.5"  # Adjust as necessary
# degree = 12  # Degree of the polynomial
# coefficients, polynomial, spline = hyp.fit_nozzle_shape(file_path, sheet_name, degree)  # Fit nozzle shape
 
# def nozzle_shape(x):
#     if x < -0.03201:
#         return 0.006365
#     else:
#         return spline(x)
 
# h = np.array([polynomial(x_elem) for x_elem in x])
# depth = 1
# A_nodes = 2*h*depth
 
# file_path = "Lettieri_nozzle.xlsx"  # Relative path to the common folder
# sheet_name = "Sheet1"  # Adjust as necessary
# degree = 10  # Degree of the polynomial
# coefficients, polynomial, spline = hyp.fit_nozzle_shape(file_path, sheet_name, degree)  # Fit nozzle shape
 
# def nozzle_shape(x):
#     if x < -0.03201:
#         return 0.006365
#     else:
#         return spline(x)
 
# h = np.array([spline(x_elem) for x_elem in x])
# width = 6.48e-3
# A_nodes = 2*h*width
 
# ## Calculate hydraulic diameter for planar nozzles
# D_hydraulic = np.zeros_like(x)
# D_hydraulic[:] = 4*A_nodes[:]/(2*(width+2*h[:]))
# print(roughness/D_hydraulic)
# plt.plot(x, roughness/D_hydraulic)
# plt.show()
 
# # Linear interpolation for nozzle shape
# R = [0.011261088563105, 0.0017775, 0.0017775, 0.016312752388279] # Simoneau paper
# L = [0, 0.07965, (0.07965+0.01135), (0.07965+0.01135+0.22)]
# radius_function = hyp.define_radius_function(L, R)
# r = np.zeros_like(x)
# for j in range(len(r)):
#     r[j] = radius_function(x[j])
# depth = 1
# A_nodes = r*depth
# A_nodes = r**2*np.pi
 
# # Double linear interpolation for Barana's nozzle
# file_path = "Barana_Conference.xlsx"  # Relative path to the common folder
# sheet_name = "Sheet1"  # Adjust as necessary
# coefficients, polynomial, spline = hyp.fit_nozzle_shape(file_path, sheet_name, 8)
# data = pd.read_excel(file_path, sheet_name=sheet_name)
 
m1 = -(0.005-0.0002)/0.015
m2 = 0.0702/8.38
 
# Piecewise nozzle shape
def nozzle_shape(x):
    if x <= 0:
        return 0.0002 + m1*x
    else:
        return 0.0002+m2*(x)
 
r = np.array([nozzle_shape(x_elem) for x_elem in x]) # For Berana actually it is h (height)
h=r
width = 1e-3
A_nodes = 2*h*width
 
## Calculate hydraulic diameter for planar nozzles
D_hydraulic = np.zeros_like(x)
perimeter = 2*(width+2*h)
D_hydraulic = 4 * A_nodes / perimeter
 
# file_path = "Lettieri_nozzle.xlsx"  # Relative path to the common folder
# sheet_name = "Sheet1"  # Adjust as necessary
 
# # Fit the nozzle shape using the shared function
# coefficients, polynomial, spline, data = hyp.fit_nozzle_shape(file_path, sheet_name, 6)
 
# r = spline(x)
 
# # r = rx(x)
# # A_nodes = r**2 * np.pi
# width = 1
# A_nodes = 2*r*width
# parameters = {'A_nodes': A_nodes}
# parameters['dx'] = dx
# perimeter = 2*r*np.pi
# D_hydraulic = 4 * A_nodes / perimeter
 
# plt.plot(x, r)
# plt.plot(data.x, data.y, "o", markersize=2)
# plt.show()
 
# Linea+Polynomial interpolation for Lettieri´s nozzle
# file_path = "lettieri.xlsx"  # Relative path to the common folder
# sheet_name = ""  # Adjust as necessary
# degree = 12  # Degree of the polynomial
# coefficients, polynomial, spline = hyp.fit_nozzle_shape(file_path, sheet_name, degree)
# h_in = 0.00635
# h_spline = lambda x: spline(x)
 
# # Piecewise nozzle shape
# def nozzle_shape(x):
#     if x < -0.032:
#         return h_in
#     else:
#         return h_spline(x)
 
# h = np.array([polynomial(x_elem) for x_elem in x])
# A_nodes = h
 
# plt.plot(x, h, "-")
# plt.show()
 
## Group up the problem parameters
parameters = {}
parameters["residuals"] = residuals
parameters["perfect_gas"] = perfect_gas
parameters['D_hydraulic'] = D_hydraulic
parameters['max_iterations'] = case_data["numerical_settings"]["max_iterations"]
parameters['residuals'] = case_data["numerical_settings"]["residuals"]
parameters['roughness'] = roughness
parameters['pT_stagnation'] = pT_stagnation
parameters['p_out'] = p_out
parameters['dx'] = dx
# parameters['depth'] = depth
parameters['A_nodes'] = A_nodes
# parameters['R_const'] = R_const
parameters['fluid_name'] = fluid_name
parameters['N_eq'] = 3  # Number of equations of the hyperbolic system
parameters['CFL'] = CFL  # Maximum allowed Courant number
# parameters['r_function'] = polynomial  # Function describing the radius along nozzle axis
parameters['Q_o'] = Q_o  # Initial condition function
parameters['time_method'] = case_data["numerical_settings"]["time_discretization"]  # Choose: 'euler', 'RK3', or 'RK4'
parameters['reconstruction'] = 'constant'  # Choose: 'constant' or 'linear'
if perfect_gas:
    parameters["gamma"] = (cp/cv)
    parameters['eigenvalues_function'] = lambda q: hyp.fvm_solver.euler_eigenvalues_perfect(q, (cp/cv))  # Eigenvalues of the Euler equations
    parameters['flux_function'] = lambda q: hyp.fvm_solver.euler_flux_perfect(q, (cp/cv))  # Flux function of the Euler equations
    parameters['roe_function'] = lambda q_L, q_R: hyp.fvm_solver.euler_roe_matrix_perfect(q_L, q_R, (cp/cv))  # Roe matrix
else:
    parameters['eigenvalues_function'] = lambda q: hyp.fvm_solver.euler_eigenvalues(q, fluid, parameters)  # Eigenvalues of the Euler equations
    parameters['flux_function'] = lambda q, p: hyp.fvm_solver.euler_flux(q, p)  # Flux function of the Euler equations
    parameters['roe_function'] = lambda q_L, q_R, p_L, p_R, parameters: hyp.fvm_solver.euler_roe_matrix(q_L, q_R, p_L, p_R, parameters)  # Roe matrix
   
# Compute the numerical solution of the problem
parameters['space_method'] = case_data["numerical_settings"]["flux_method"]  
Q, t = hyp.fvm_solver.fvm_euler_1d(x, t, parameters)
print(f"{parameters['space_method']} computed")
 
# Retrieve the thermodynamic quantities
if perfect_gas:
    density = Q[0, :, -1]  # Density at the last time step
    velocity = Q[1, :, -1] / Q[0, :, -1]  # Velocity at the last time step
    internal_energy = (Q[2, :, -1] / Q[0, :, -1]) - 0.5 * (Q[1, :, -1] / Q[0, :, -1])**2  # Specific internal energy at last time step_rusanov = np.zeros_like(d_rusanov)  # Initialize pressure array for the last time step
    pressure = ((cp/cv) - 1) * (Q[2, :, -1] - 0.5 * Q[1, :, -1]**2 / Q[0, :, -1])
    sound_speed = np.sqrt((cp/cv)*pressure/density)
    Ma = velocity/sound_speed
else:
    density = Q[0, :, -1]  # Density at the last time step
    velocity = Q[1, :, -1] / Q[0, :, -1]  # Velocity at the last time step
    internal_energy = (Q[2, :, -1] / Q[0, :, -1]) - 0.5 * (Q[1, :, -1] / Q[0, :, -1])**2  # Specific internal energy at last time step_rusanov = np.zeros_like(d_rusanov)  # Initialize pressure array for the last time step
   
    sound_speed = np.zeros_like(density)
    pressure = np.zeros_like(density)
    s = np.zeros_like(density)
    Q = np.zeros_like(density)
    Z = np.zeros_like(density)
    Ma = np.zeros_like(density)
 
    # Loop through each cell to set the state and retrieve pressure and other quantities at the last time step
    for j in range(len(density)):
        # Set the fluid state using density and specific internal energy at the last time step
        state = fluid.get_state(bpy.fluid_properties.DmassUmass_INPUTS, density[j], internal_energy[j])
        pressure[j] = state.p  # Access 'p' from the FluidState dictionary-like object
        sound_speed[j] = state.speed_sound
        s[j] = state.s
        Q[j] = state.Q
        Z[j] = state.Z
        # props_blend, props_eq, props_meta = fluid.get_state_blending(
        #     prop_1="rho",
        #     prop_1_value=d_rusanov[j],
        #     prop_2="u",
        #     prop_2_value=e_rusanov[j],
        #     rhoT_guess_equilibrium=rhoT_guess_equilibrium,
        #     rhoT_guess_metastable=rhoT_guess_metastable,
        #     phase_change=phase_change,
        #     blending_variable="Q",
        #     blending_onset=Q_onset,
        #     blending_width=Q_width,
        #     generalize_quality=True,
        #     supersaturation=True,
        #     print_convergence=False,
        #     solver_algorithm="lm",
        # )
        # # Update initial guess
        # rhoT_guess_metastable = [props_meta["rho"], props_meta["T"]]
        # rhoT_guess_equilibrium = [props_eq["rho"], props_eq["T"]]  
        # p_rusanov[j] = props_blend.p  # Access 'p' from the FluidState dictionary-like object
        # a_sound[j] = props_blend.speed_sound
        # s[j] = props_blend.s
        # Q[j] = props_blend.Q
        # T[j] = props_blend.T
 
        Ma[j] = velocity[j]/sound_speed[j]
 
## plotting density, pressure, Mach
if perfect_gas:  # this include also the analytic solution
    total_bc = np.array([T_stagnation, p_stagnation])
    M_exact, p_exact, u_exact, rho_exact, i_shock = hyp.fvm_solver.nozzle_exact2(x, parameters, total_bc)
    A_cr_index = np.argmin(A_nodes)
   
    # Plot the Density Profile
    plt.figure(figsize=(6, 5))
    plt.xlabel('x/L')
    plt.ylabel(r'$\rho / \rho_0$')
    plt.plot(x, density/rho_stagnation, 'go-', linewidth=0.75, markersize=4, markerfacecolor='w', label=f'Numerical Solution - {parameters['space_method']}')
    plt.plot(x, rho_exact[:, 0]/rho_stagnation, 'r--', linewidth=0.75, markersize=4, markerfacecolor='w', label='Exact solution - Subsonic')
    plt.plot(x[A_cr_index:], rho_exact[A_cr_index:, 1]/rho_stagnation, 'b--', linewidth=0.75, markersize=4, markerfacecolor='w', label='Exact solution - Supersonic')
    if i_shock is not None:
        plt.plot(x[i_shock-1:], rho_exact[i_shock-1:, 2]/rho_stagnation, 'k-', linewidth=0.75, markersize=2, markerfacecolor='w', label='Exact solution - Shock wave')
        plt.legend(
            loc='center left',
            # bbox_to_anchor=(0.0, 0.35),  # Adjust the y value (0.4) to move the legend downward
            fontsize='small',
            handlelength=1.5,
            handletextpad=0.5,
            borderpad=0.5,
            frameon=True
        )
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(x, r, 'k--', linewidth=1.1, label='Nozzle cross section')
    ax2.set_ylabel('r / L')
    ax2.set_ylim([0, L])
 
    # Plot the Pressure Profile
    plt.figure(figsize=(6, 5))
    plt.xlabel('x/L')
    plt.ylabel(r'P / P_T')
    plt.plot(x, pressure/p_stagnation, 'go-', linewidth=0.75, markersize=4, markerfacecolor='w', label=f'Numerical Solution - {parameters['space_method']}')
    plt.plot(x, p_exact[:, 0]/p_stagnation, 'r--', linewidth=0.75, markersize=4, markerfacecolor='w', label='Exact solution - Subsonic')
    plt.plot(x[A_cr_index:], p_exact[A_cr_index:, 1]/p_stagnation, 'b--', linewidth=0.75, markersize=4, markerfacecolor='w', label='Exact solution - Supersonic')
    if i_shock is not None:
        plt.plot(x[i_shock-1:], p_exact[i_shock-1:, 2]/p_stagnation, 'k-', linewidth=0.75, markersize=2, markerfacecolor='w', label='Exact solution - Shock wave')
        plt.legend(
            loc='center left',
            # bbox_to_anchor=(0.0, 0.35),  # Adjust the y value (0.4) to move the legend downward
            fontsize='small',
            handlelength=1.5,
            handletextpad=0.5,
            borderpad=0.5,
            frameon=True
        )
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(x, r, 'k--', linewidth=1.1, label='Nozzle cross section')
    ax2.set_ylabel('r / L')
    ax2.set_ylim([0, L])
 
 
    # Plot the Mach Profile
    plt.figure(figsize=(6, 5))
    plt.xlabel('x / L')
    plt.ylabel('M')
    plt.plot(x, Ma, 'go-', linewidth=0.75, markersize=4, markerfacecolor='w', label=f'Numerical Solution - {parameters['space_method']}')
    plt.plot(x, M_exact[:, 0], 'r--', linewidth=0.75, markersize=4, markerfacecolor='w', label='Exact solution - Subsonic')
    plt.plot(x[A_cr_index:], M_exact[A_cr_index:, 1], 'b--', linewidth=0.75, markersize=4, markerfacecolor='w', label='Exact solution - Supersonic')
    if i_shock is not None:
        plt.plot(x[i_shock-1:], M_exact[i_shock-1:, 2], 'k-', linewidth=0.75, markersize=2, markerfacecolor='w', label='Exact solution - Shock wave')
        plt.legend(
            loc='upper left',
            handlelength=1,           # Shorter line markers in the legend
            handletextpad=0.3,        # Less space between line markers and text
            borderpad=0.4,            # Reduce padding inside the legend box
            frameon=True              # Keep the frame around the legend
        )
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(x, r, 'k--', linewidth=1.1, label='Nozzle cross section')
    ax2.set_ylabel('r / L')
    ax2.set_ylim([0, L])
 
else:
   
    # Plot the Density Profile
    plt.figure(figsize=(6, 5))
    plt.xlabel('x/L')
    plt.ylabel(r'$\rho / \rho_0$')
    plt.plot(x, density, 'go-', linewidth=0.75, markersize=4, markerfacecolor='w', label=f'Numerical Solution - {parameters['space_method']}')
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(x, r, 'k--', linewidth=1.1, label='Nozzle cross section')
    ax2.set_ylabel('r / L')
    ax2.set_ylim([0, L])
 
    # Plot the Pressure Profile
    plt.figure(figsize=(6, 5))
    plt.xlabel('x/L')
    plt.ylabel(r'P / P_T')
    plt.plot(x, pressure, 'go-', linewidth=0.75, markersize=4, markerfacecolor='w', label=f'Numerical Solution - {parameters['space_method']}')
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(x, r, 'k--', linewidth=1.1, label='Nozzle cross section')
    ax2.set_ylabel('r / L')
    ax2.set_ylim([0, L])
 
    # Plot the Mach Profile
    plt.figure(figsize=(6, 5))
    plt.xlabel('x / L')
    plt.ylabel('M')
    plt.plot(x, Ma, 'go-', linewidth=0.75, markersize=4, markerfacecolor='w', label=f'Numerical Solution - {parameters['space_method']}')
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(x, r, 'k--', linewidth=1.1, label='Nozzle cross section')
    ax2.set_ylabel('r / L')
    ax2.set_ylim([0, L])

plt.grid(False)
plt.tight_layout()
# print("Exact pressures:", p_exact[np.argmin(r), 1], p_exact[-1, 1])
# print("Exact flow rate:", A_nodes[-1]*rho_exact[-1, 1]*u_exact[-1, 1])
  
np.savez(f"rusanov_N{Nx}_pout_{p_out}_HEM_haal_3roug.npz",v=velocity, h=h, x=x, M=Ma, p=pressure, rho=density)
# np.savez("rusanov_N260_pin90_tin37_po.ut48.2_rough3.npz", h=h, x=x, e=e_rusanov, a=a_sound, p=p_rusanov, rho=d_rusanov, u=u_rusanov, Z=Z)
mass_flow = density[-1]*velocity[-1]*A_nodes[-1]
print(mass_flow)
np.savez(f"rusanov_N{Nx}_pout_{p_out}_massflow_HEM_haal_3roug.npz", mass_flow=mass_flow)
print("Numerical pressures:", min(pressure), pressure[-1])
print("Numerical flow rate:", A_nodes[-1]*density[-1]*velocity[-1])
plt.show()
