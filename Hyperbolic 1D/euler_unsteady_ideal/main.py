import yaml
import numpy as np
import matplotlib.pyplot as plt
from fvm import *
from exact_solution import *
from fit_nozzle_shape import fit_nozzle_shape

from functions import print_dict, set_plot_options

set_plot_options(grid=False)

# Read YAML file
with open("euler_adapted_expansion.yaml", 'r') as file:
    case_data = yaml.safe_load(file)


# print(data)
print_dict(case_data)


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

# Define the properties of the gas
R = 287.00  # Gas constant of air
gamma = 1.4  # Heat capacity ratio of air

# Define nozzle area
r0 = case_data["nozzle_geometry"]["r0"]
rt = case_data["nozzle_geometry"]["rt"]
L = x2
rx = lambda x: (4 * (r0 - rt) / L**2) * (x - L / 2)**2 + rt
# Path to the Excel file in the common folder

file_path = "nozzles points.xlsx"  # Relative path to the common folder
sheet_name = "nozzle Ma1.5"  # Adjust as necessary
degree = 12  # Degree of the polynomial

# Fit the nozzle shape using the shared function
# coefficients, polynomial = fit_nozzle_shape(file_path, sheet_name, degree)  # Fit nozzle shape

# r = polynomial(x)
r = rx(x)
A_nodes = r**2 * np.pi
parameters = {'A_nodes': A_nodes}
parameters['dx'] = dx
plt.plot(x, A_nodes)

# Define boundary conditions
# Inlet
p_stagnation = float(case_data["boundary_conditions"]["p_stagnation"])
T_stagnation = float(case_data["boundary_conditions"]["T_stagnation"])
pT_stagnation = np.array([p_stagnation, T_stagnation])
parameters['pT_stagnation'] = pT_stagnation

# Outlet
p_out = float(case_data["boundary_conditions"]["p_out"])
parameters['p_out'] = p_out

# Define the initial conditions
p_o = p_stagnation  # bar
T_o = 500  # K
rho_o = p_o / (R * T_o)
u_o = 0

# Uniform initial condition across the nozzle
Q_o = lambda x: np.array([rho_o * np.ones_like(x), rho_o * u_o * np.ones_like(x),
                          (p_o / (gamma - 1) + 0.5 * rho_o * u_o**2) * np.ones_like(x)])

# Group up the problem parameters
parameters['gamma'] = gamma
parameters['N_eq'] = 3  # Number of equations of the hyperbolic system
parameters['CFL'] = CFL  # Maximum allowed Courant number
# parameters['r_function'] = polynomial  # Function describing the radius along nozzle axis
parameters['r_function'] = rx  # Function describing the radius along nozzle axis
parameters['Q_o'] = Q_o  # Initial condition function
parameters['time_method'] = 'RK3'  # Choose: 'euler', 'RK3', or 'RK4'
parameters['reconstruction'] = 'constant'  # Choose: 'constant' or 'linear'
parameters['flux_function'] = lambda q: euler_flux(q, gamma)  # Flux function of the Euler equations
parameters['eigenvalues_function'] = lambda q: euler_eigenvalues(q, gamma)  # Eigenvalues of the Euler equations
parameters['roe_function'] = lambda q_L, q_R: euler_roe_matrix(q_L, q_R, gamma)  # Roe matrix

# Compute the numerical solution of the problem
parameters['space_method'] = 'rusanov'  
Q, t = fvm_euler_1d(x, t, parameters)
d_roe = Q[0, :, :]
u_roe = Q[1, :, :] / Q[0, :, :]
p_roe = (gamma - 1) * (Q[2, :, :] - 0.5 * Q[1, :, :]**2 / Q[0, :, :])
print("Rusanov computed")

# Compute exact solution 
total_bc = np.array([T_stagnation, p_stagnation])
M, p, u, rho, i_shock = nozzle_exact2(x, parameters, total_bc)

A_cr_index = np.argmin(A_nodes)

# Plot the Density Profile
plt.figure(figsize=(6, 5))
plt.xlabel('Axis position [m]')
plt.ylabel('Density [kg/m^3]')
plt.plot(x[::2,], d_roe[::2, -1], 'go-', linewidth=0.75, markersize=4, markerfacecolor='w', label='Numerical Solution - Roe')
plt.plot(x, rho[:, 0], 'r--', linewidth=0.75, markersize=4, markerfacecolor='w', label='Exact solution - Subsonic')
plt.plot(x[A_cr_index:], rho[A_cr_index:, 1], 'b--', linewidth=0.75, markersize=4, markerfacecolor='w', label='Exact solution - Supersonic')
if i_shock is not None:
   plt.plot(x[i_shock-1:], rho[i_shock-1:, 2], 'k-', linewidth=0.75, markersize=2, markerfacecolor='w', label='Exact solution - Shock wave') 
plt.legend(
    loc='center left',
    bbox_to_anchor=(0.0, 0.35),  # Adjust the y value (0.4) to move the legend downward
    fontsize='small',
    handlelength=1.5,
    handletextpad=0.5,
    borderpad=0.5,
    frameon=True
)
plt.grid(True)

# Add the second y-axis for the nozzle area
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(x, r, 'k--', linewidth=1.1, label='Nozzle cross section')
ax2.set_ylabel('Nozzle Radius [m]')
ax2.set_ylim([0, L])
plt.tight_layout()


# Plot the Velocity Profile
plt.figure(figsize=(6, 5))
plt.xlabel('Axis position [m]')
plt.ylabel('Velocity [m/s]')
plt.plot(x[::2], u_roe[::2, -1], 'go-', linewidth=0.75, markersize=4, markerfacecolor='w', label='Numerical Solution - Roe')
plt.plot(x, u[:, 0], 'r--', linewidth=0.75, markersize=4, markerfacecolor='w', label='Exact solution - Subsonic')
plt.plot(x[A_cr_index:], u[A_cr_index:, 1], 'b--', linewidth=0.75, markersize=4, markerfacecolor='w', label='Exact solution - Supersonic')
if i_shock is not None:
   plt.plot(x[i_shock-1:], u[i_shock-1:, 2], 'k-', linewidth=0.75, markersize=2, markerfacecolor='w', label='Exact solution - Shock wave') 
plt.legend(
    loc='upper left',
    handlelength=1,           # Shorter line markers in the legend
    handletextpad=0.3,        # Less space between line markers and text
    borderpad=0.4,            # Reduce padding inside the legend box
    frameon=True              # Keep the frame around the legend
)
plt.grid(True)

# Add the second y-axis for the nozzle area
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(x, r, 'k--', linewidth=1.1, label='Nozzle cross section')
ax2.set_ylabel('Nozzle Radius [m]')
ax2.set_ylim([0, L])
plt.tight_layout()


plt.figure(figsize=(6, 5))
plt.xlabel('Axis position [m]')
plt.ylabel('Pressure [bar]')
plt.plot(x[::9], p_roe[::9, -1] / 1e5, 'go-', linewidth=0.75, markersize=4, markerfacecolor='w', label='Numerical Solution - Roe')
plt.plot(x, p[:, 0] / 1e5, 'r--', linewidth=0.75, label='Exact solution - Subsonic')
plt.plot(x[A_cr_index:], p[A_cr_index:,1] / 1e5, 'b--', linewidth=0.75, label='Exact solution - Supersonic')
if i_shock is not None:
    plt.plot(x[i_shock - 1:], p[i_shock - 1:, 2] / 1e5, 'k-', linewidth=0.75, markersize=2, label='Exact solution - Shock wave')

plt.legend(
    loc='center left',
    bbox_to_anchor=(0.0, 0.35),  # Adjust the y value (0.4) to move the legend downward
    fontsize='small',
    handlelength=1.5,
    handletextpad=0.5,
    borderpad=0.5,
    frameon=True
)
plt.grid(True)

ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(x, r, 'k--', linewidth=1.1, label='Nozzle cross section')
ax2.set_ylabel('Nozzle Radius [m]', labelpad=10)  # Adjust labelpad to move it away from the frame
ax2.set_ylim([0, L])
plt.tight_layout()
plt.show()

# A_critical = np.pi * r[len(x) // 2]**2
# mass_flow= d_roe[len(x)//2, -1]*u_roe[len(x)//2, -1]*A_critical

# np.savez("hll_supersonicd3.npz", x=x, p1=p_roe, rho1=d_roe, u1=u_roe, i_shock=i_shock)
# np.savez("exact_supersonicd3.npz", r=r, L=L, x=x, p=p, rho=rho, u=u)

cv = R / (gamma - 1)
# entropy_generation = cv * (np.log(p_roe[-1, -1] / d_roe[-1, -1]**gamma) - np.log(p_roe[1, -1] / d_roe[1, -1]**gamma))
# entropy_generation_exact = cv * (np.log(p[-1, -1] / rho[-1, -1]**gamma) - np.log(p[1, -1] / rho[1, -1]**gamma))

# print(entropy_generation)
# print(entropy_generation_exact)
# p_roe[0,-1]