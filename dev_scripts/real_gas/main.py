import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nozzle_hyperbolic as hyp
import barotropy as bpy

hyp.set_plot_options(grid=False)

# Read YAML file
with open("euler_adapted_expansion.yaml", 'r') as file:
    case_data = yaml.safe_load(file)

# print(data)
hyp.print_dict(case_data)

# Define the computational problem
# Define fluid
fluid_name = "CO2"
fluid = bpy.fluid_properties.Fluid(fluid_name)
# Define the computational problem
# Define the grid
Nx = case_data["numerical_settings"]["Nx"]  # Number of cells
x1 = case_data["numerical_settings"]["x1"]    # Left boundary
x2 = case_data["numerical_settings"]["x2"]   # Right boundary

x = np.linspace(0, 1, Nx)  # x grid vector
x = x1 + (x2 - x1) * x
dx = x[1] - x[0]

# Define the simulation time
t1 = case_data["numerical_settings"]["t1"]   # Initial time
t2 = case_data["numerical_settings"]["t2"]  # Final time
t = np.array([t1, t2])  # Time vector

# Define the CFL condition
CFL = case_data["numerical_settings"]["CFL"]  # Maximum allowed local Courant number

# Define nozzle area
L = x2

# # Polynomial interpolation for nozzle shape 
# file_path = "Lettieri_nozzle.xlsx"  # Relative path to the common folder
# sheet_name = "Sheet1"  # Adjust as necessary
# degree = 10  # Degree of the polynomial
# coefficients, polynomial, spline = hyp.fit_nozzle_shape(file_path, sheet_name, degree)  # Fit nozzle shape
# hx = spline
# h = spline(x)
# depth = 1
# A_nodes = 2*h*depth

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

# Double linear interpolation for Barana's nozzle
file_path = "Barana_Conference.xlsx"  # Relative path to the common folder
sheet_name = "Sheet1"  # Adjust as necessary
coefficients, polynomial, spline = hyp.fit_nozzle_shape(file_path, sheet_name, 8)

# # Extract data for plotting
# data = pd.read_excel(file_path, sheet_name=sheet_name)

# m1 = -(5-0.12)/27.35
# m2 = (0.195-0.12)/56.15

# # Piecewise nozzle shape
# def nozzle_shape(x):
#     if x <= 25:
#         return 5 + m1*x
#     elif x > 35:
#         return 0.12+m2*(x-27.35)
#     else:
#         return spline(x)
        
h = np.array([spline(x_elem) for x_elem in x]) 
A_nodes = 2*h
        
# # Linea+Polynomial interpolation for Lettieri´s nozzle
# file_path = "Lettieri_nozzle.xlsx"  # Relative path to the common folder
# sheet_name = "Sheet1"  # Adjust as necessary
# degree = 7  # Degree of the polynomial
# coefficients, polynomial, spline = hyp.fit_nozzle_shape(file_path, sheet_name, degree)
# h_in = 0.00635
# h_spline = lambda x: spline(x)

# # Piecewise nozzle shape
# def nozzle_shape(x):
#     if x < -0.032:
#         return h_in
#     else:
#         return h_spline(x)

# h = np.array([nozzle_shape(x_elem) for x_elem in x])
# A_nodes = h

plt.plot(x, h, "-")
plt.show()

# ##
# parameters = {'A_nodes': A_nodes}
# parameters['dx'] = dx

# Define boundary conditions
# Inlet
p_stagnation = float(case_data["boundary_conditions"]["p_stagnation"])
T_stagnation = float(case_data["boundary_conditions"]["T_stagnation"])
pT_stagnation = np.array([p_stagnation, T_stagnation])
# Outlet
p_out = float(case_data["boundary_conditions"]["p_out"])

## Define the initial conditions, uniform across the nozzle
p_o = p_stagnation  # bar
T_o = T_stagnation  # K
initial_state = fluid.get_state(bpy.fluid_properties.PT_INPUTS, p_o, T_o)
rho_initial = initial_state.rho
u_internal_initial = initial_state.umass
s0 = initial_state.s 
# parameters['s0'] = s0# Pressure, pre-computed before calling this function
u_o = 0
total_energy = u_internal_initial + 0.5 * u_o**2
Q_o = lambda x: np.array([rho_initial * np.ones_like(x), rho_initial * u_o * np.ones_like(x), rho_initial * total_energy * np.ones_like(x)])

# # Define phase change process
# if s0 > fluid.critical_point.s:
#     phase_change = "condensation"
#     Q_onset = 0.95
#     Q_width = 0.01
# else:
#     phase_change = "evaporation"
#     Q_onset = 0.01
#     Q_width = 0.01

parameters = {}
# parameters["phase_change"] = phase_change
# parameters["Q_onset"] = Q_onset
# parameters["Q_width"] = Q_width

## Define the initial conditions, linear across the nozzle
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

# plt.figure(figsize=(6, 5))
# plt.plot(x, u_o)
# plt.figure(figsize=(6, 5))
# plt.plot(x, p_o)
# plt.figure(figsize=(6, 5))
# plt.plot(x, rho_initials)
# plt.show()

# Q_o = [rho_initials, rho_initials*u_o, rho_initials*total_energies]
# Q_o = np.array(Q_o)

##

# Group up the problem parameters
parameters['pT_stagnation'] = pT_stagnation
parameters['p_out'] = p_out
parameters['dx'] = dx
# parameters['depth'] = depth
parameters['A_nodes'] = A_nodes
# parameters['R_const'] = R_const
parameters['fluid_name'] = fluid_name
parameters['N_eq'] = 3  # Number of equations of the hyperbolic system
parameters['CFL'] = CFL  # Maximum allowed Courant number
parameters['r_function'] = h  # Function describing the radius along nozzle axis
parameters['Q_o'] = Q_o  # Initial condition function
parameters['time_method'] = 'euler'  # Choose: 'euler', 'RK3', or 'RK4'
parameters['reconstruction'] = 'constant'  # Choose: 'constant' or 'linear'
parameters['flux_function'] = lambda q, p: hyp.real_gas.euler_flux(q, p)  # Flux function of the Euler equations
parameters['eigenvalues_function'] = lambda q: hyp.real_gas.euler_eigenvalues(q, fluid, parameters)  # Eigenvalues of the Euler equations
parameters['roe_function'] = lambda q_L, q_R, p_L, p_R: hyp.real_gas.euler_roe_matrix(q_L, q_R, p_L, p_R, fluid)  # Roe matrix

# Compute the numerical solution of the problem
parameters['space_method'] = 'rusanov'  
Q, t = hyp.real_gas.fvm_euler_1d(x, t, parameters)
d_rusanov = Q[0, :, -1]  # Density at the last time step
u_rusanov = Q[1, :, -1] / Q[0, :, -1]  # Velocity at the last time step
e_rusanov = (Q[2, :, -1] / Q[0, :, -1]) - 0.5 * (Q[1, :, -1] / Q[0, :, -1])**2  # Specific internal energy at last time step
p_rusanov = np.zeros_like(d_rusanov)  # Initialize pressure array for the last time step
a_sound = np.zeros_like(d_rusanov)
s = np.zeros_like(d_rusanov)
Q = np.zeros_like(d_rusanov)
T = np.zeros_like(d_rusanov)

# rhoT_guess_metastable = [initial_state["rho"], initial_state["T"]]
# rhoT_guess_equilibrium = [initial_state["rho"], initial_state["T"]]
stagnation_state = fluid.get_state(bpy.fluid_properties.PT_INPUTS, p_stagnation, T_stagnation)
s0 = stagnation_state.s

# Loop through each cell to set the state and retrieve pressure at the last time step
for j in range(len(d_rusanov)):
    # Set the fluid state using density and specific internal energy at the last time step
    state = fluid.get_state(bpy.fluid_properties.DmassUmass_INPUTS, d_rusanov[j], e_rusanov[j])

    p_rusanov[j] = state.p  # Access 'p' from the FluidState dictionary-like object
    a_sound[j] = state.speed_sound
    s[j] = state.s
    Q[j] = state.Q
    T[j] = state.T

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

print(f"{parameters['space_method']} computed")

ax0 = plt.gca()
ax0.set_xlabel("Entropy (J/kg/K)")
ax0.set_ylabel("Temperature (K)")

# Plot phase diagram
fluid.plot_phase_diagram(
    "s",
    "T",
    axes=ax0,
    plot_critical_point=True,
    plot_quality_isolines=True,
    plot_pseudocritical_line=False,
    # plot_spinodal_line=True,
)

ax0.plot(s, T, marker="o", color="purple")

# Plot the entropy Profile
# plt.figure(figsize=(6, 5))
# plt.title('Entropy Profile')
# plt.xlabel('Position (x)')
# plt.ylabel('entropy')
# plt.plot(x, s, 'go-', linewidth=0.75, markersize=4, markerfacecolor='w', label='Numerical Solution - Roe')
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
# plt.grid(True)

liq_mass_frac = 1 - Q
# Plot the Density Profile
plt.figure(figsize=(6, 5))
plt.xlabel('Position (x)')
plt.ylabel('Vapour quality')
plt.plot(x, liq_mass_frac, 'go-', linewidth=0.75, markersize=4, markerfacecolor='w', label='Numerical Solution - Roe')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
plt.grid(True)

# Add the second y-axis for the nozzle area
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(x, h, 'k--', linewidth=1.1, label='Nozzle')
ax2.set_ylabel('Nozzle Radius [m]')
# ax2.set_ylim([0, L])

# Plot the Velocity Profile
plt.figure(figsize=(8, 6))
plt.xlabel('Position (x)')
plt.ylabel('Sound velocity')
plt.plot(x, a_sound, 'go-', linewidth=0.75, markersize=4, markerfacecolor='w', label='Numerical Solution - Roe')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
plt.grid(True)

# Add the second y-axis for the nozzle area
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(x, h, 'k--', linewidth=1.1, label='Nozzle cross section')
ax2.set_ylabel('Nozzle Radius [m]')
# ax2.set_ylim([0, L])


## Zhang experimental results

x_exp = [
    0, 0.01, 0.02, 0.03, 0.04, 0.05, 
    0.06, 0.07, 0.08, 0.09, 0.1, 
    0.11, 0.12, 0.13, 0.14, 0.15, 
    0.16, 0.17, 0.18, 0.19
]

## Case 1
# pressure_kPa = np.array([55.22, 51.07, 47.85, 45.82, 42.77, 40.72, 38.8, 36.43, 34.82, 
#                          33.23, 31.5, 29.95, 28.7, 27.29, 26.28, 24.99, 24.03, 24.22, 
#                          24.26, 24.3])

## Case 2
# pressure_kPa = np.array([39.95, 36.88, 34.57, 33.11, 30.89, 29.4, 28.06, 26.31, 25, 24.03, 
#               22.81, 21.68, 21.02, 20.78, 20.78, 20.93, 20.76, 20, 19.24, 18.5])

## Case 3
# pressure_kPa = np.array([59.81, 55.27, 51.78, 49.55, 46.22, 44.01, 41.86, 39.35, 38.04,
#                38.71, 39.25, 39.38, 37.56, 36.06, 34.36, 33.23, 31.79, 30.42,
#                29.32, 28.11])

# ## Case 4

# pressure_Pa = pressure_kPa * 1e3

# Plot the Pressure Profile
plt.figure(figsize=(8, 6))
plt.title('Pressure Profile')
plt.xlabel('Position (x)')
plt.ylabel('Pressure')
plt.plot(x, p_rusanov, '-', linewidth=0.75, markersize=4, markerfacecolor='w', label='Numerical Solution - Roe')
# plt.plot(x_exp, pressure_Pa, 'ro')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
plt.grid(True)

# Add the second y-axis for the nozzle area
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(x, h, 'k--', linewidth=1.1, label='Nozzle cross section')
ax2.set_ylabel('Nozzle Area [m²]')
# ax2.set_ylim([0, L])

plt.tight_layout()
plt.show()

# np.savez("Case5_rusanov_N180.npz", h=h, x=x, e=e_rusanov, a=a_sound, p=p_rusanov, rho=d_rusanov, u=u_rusanov, Q=Q)
mass_flow = d_rusanov[-1]*u_rusanov[-1]*A_nodes[-1]
print(mass_flow)