import yaml
import numpy as np
import matplotlib.pyplot as plt
# from fvm import *
# import CoolProp.CoolProp as CP
# import fluid_properties
# from fit_nozzle_shape import fit_nozzle_shape
# from functions import print_dict

import nozzle_hyperbolic as hyp

hyp.set_plot_options(grid=False)


# Read YAML file
with open("euler_adapted_expansion.yaml", 'r') as file:
    case_data = yaml.safe_load(file)


# print(data)
hyp.print_dict(case_data)


# Define the computational problem
# Define fluid
# fluid_name = "nitrogen"
fluid_name = "MDM"
fluid = hyp.fluid_properties.Fluid(fluid_name)

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

# # Define the properties of the gas
# R_const = 287.00  # Gas constant of air
# gamma = 1.4  # Heat capacity ratio of air

# Define nozzle area
r0 = case_data["nozzle_geometry"]["r0"]
rt = case_data["nozzle_geometry"]["rt"]
L = x2
# r = lambda x: (r0 - rt) * (x - L / 2)**2 + rt
# A_nodes = rx(x)**2 * np.pi

file_path = "nozzles points.xlsx"  # Relative path to the common folder
sheet_name = "nozzle Ma2"  # Adjust as necessary
degree = 12  # Degree of the polynomial

# Fit the nozzle shape using the shared function
coefficients, polynomial = hyp.fit_nozzle_shape(file_path, sheet_name, degree)  # Fit nozzle shape

hx = polynomial
h = polynomial(x)
depth = 0.5
A_nodes = 2*h*depth
parameters = {'A_nodes': A_nodes}
parameters['dx'] = dx
# plt.plot(x, A_nodes)
# plt.show()

# Define boundary conditions
# Inlet
p_stagnation = float(case_data["boundary_conditions"]["p_stagnation"])
T_stagnation = float(case_data["boundary_conditions"]["T_stagnation"])
pT_stagnation = np.array([p_stagnation, T_stagnation])

# Outlet
p_out = float(case_data["boundary_conditions"]["p_out"])

# Define the initial conditions
p_o = p_stagnation  # bar
T_o = T_stagnation  # K
initial_state = fluid.set_state(hyp.fluid_properties.PT_INPUTS, p_o, T_o)
rho_initial = initial_state.rho
u_internal_initial = initial_state.umass 
u_o = 0
total_energy = u_internal_initial + 0.5 * u_o**2

# Uniform initial condition across the nozzle
Q_o = lambda x: np.array([rho_initial * np.ones_like(x), rho_initial * u_o * np.ones_like(x), rho_initial * total_energy * np.ones_like(x)])

# Group up the problem parameters
parameters = {}
parameters['pT_stagnation'] = pT_stagnation
parameters['p_out'] = p_out
parameters['dx'] = dx
parameters['depth'] = depth
parameters['A_nodes'] = A_nodes
# parameters['R_const'] = R_const
parameters['fluid_name'] = fluid_name
parameters['N_eq'] = 3  # Number of equations of the hyperbolic system
parameters['CFL'] = CFL  # Maximum allowed Courant number
parameters['r_function'] = hx  # Function describing the radius along nozzle axis
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

# Loop through each cell to set the state and retrieve pressure at the last time step
for j in range(len(d_rusanov)):
    # Set the fluid state using density and specific internal energy at the last time step
    state = fluid.set_state(hyp.DmassUmass_INPUTS, d_rusanov[j], e_rusanov[j])
    p_rusanov[j] = state.p  # Access 'p' from the FluidState dictionary-like object

print("Rusanov computed")

# Re-initialize A_nodes length for the next flux function
A_nodes = 2*h*depth
parameters['A_nodes'] = A_nodes

# Compute exact solution 
total_bc = np.array([T_stagnation, p_stagnation])

# Plot the Density Profile
plt.figure(figsize=(6, 5))
plt.title('Density Profile')
plt.xlabel('Position (x)')
plt.ylabel('Density')
plt.plot(x, h, 'go-', linewidth=0.75, markersize=4, markerfacecolor='w', label='Numerical Solution - Roe')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
plt.grid(True)

# Add the second y-axis for the nozzle area
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(x, h, 'k--', linewidth=1.1, label='Nozzle')
ax2.set_ylabel('Nozzle Radius [m]')
ax2.set_ylim([0, L])

# Plot the Velocity Profile
plt.figure(figsize=(8, 6))
plt.title('Velocity Profile')
plt.xlabel('Position (x)')
plt.ylabel('Velocity')
plt.plot(x, u_rusanov, 'go-', linewidth=0.75, markersize=4, markerfacecolor='w', label='Numerical Solution - Roe')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
plt.grid(True)

# Add the second y-axis for the nozzle area
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(x, h, 'k--', linewidth=1.1, label='Nozzle cross section')
ax2.set_ylabel('Nozzle Radius [m]')
ax2.set_ylim([0, L])

x_exp = [8.24, 10.27, 12.30, 14.33]
p_exp = [0.84, 0.62, 0.37, 0.28]
# Plot the Pressure Profile
plt.figure(figsize=(8, 6))
plt.title('Pressure Profile')
plt.xlabel('Position (x)')
plt.ylabel('Pressure')
plt.plot(x, p_rusanov/p_stagnation, 'go-', linewidth=0.75, markersize=4, markerfacecolor='w', label='Numerical Solution - Roe')
plt.plot(x_exp, p_exp, 'ro')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
plt.grid(True)

# Add the second y-axis for the nozzle area
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(x, h, 'k--', linewidth=1.1, label='Nozzle cross section')
ax2.set_ylabel('Nozzle Area [m²]')
ax2.set_ylim([0, L])

plt.tight_layout()
plt.show()

