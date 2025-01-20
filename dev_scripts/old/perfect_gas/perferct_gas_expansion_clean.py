# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from barotropy import properties as props
# import CoolProp.CoolProp as cp
import scipy.optimize as spo
import scipy.interpolate as spi
from nozzle_hyperbolic.fvm_solver import perfect_gas_props as pgp

# Input parameters
# fluid_name = 'Air'  # Working gas
T0 = 500.0  # Total temperature at inlet (K)
P0 = 7e5  # Total pressure at inlet (Pa)
p_out = 6.46999e5 # Static pressure at exit (Pa)
n = 200 # number of division of x

# Area distribution of the nozzle
nozzle_file = 'nozzles points.xlsx'
nozzle_data = pd.read_excel(nozzle_file)
area_distribution = np.pi * np.array(nozzle_data['y']) ** 2
A_in, A_out = area_distribution[0], area_distribution[-1]  
x, y = nozzle_data['x'], nozzle_data['y']


# old_area = np.load('roe_subsonic.npz')
# x = old_area['x']
# y = old_area['r']


degree = 8  # Degree of the polynomial
coefficients = np.polyfit(x, y, degree)
polynomial = np.poly1d(coefficients)
# polynomial = spi.CubicSpline(x,y)
# polynomial = spi.CubicSpline(x,y)

x = np.linspace(0, 1, n)
y = polynomial(x)



area_distribution = np.pi * (y ** 2)

# %%

plt.figure('Nozzle interpolation')
plt.plot(nozzle_data['x'], nozzle_data['y'], marker = 'o', linestyle = '')
plt.plot(x,y)
# plt.show()

# fluid.update(cp.PT_INPUTS, P0, T0)
prop = pgp("PT_INPUTS", P0, T0)
s0 = prop['s']
h0 = prop['h']
rho0 = prop['d']
R = 287.0  # Specific gas constant [J/(kg K)]
gamma = 1.4  # Ratio of specific heats

# Calculate the mass-flow rate
def mass_balance(velocity_inlet):
    # Inlet mass-flow
    h_in = h0 - 0.5 * velocity_inlet ** 2
    h_in = h_in.item()
    prop = pgp("HmassSmass_INPUTS",h_in, s0)
    rho_in = prop['d']
    mass_in = rho_in * velocity_inlet * A_in

    # Outlet mass-flow
    prop = pgp("PSmass_INPUTS",p_out, s0)
    rho_out = prop['d']
    h_out = prop['h']
    velocity_outlet = np.sqrt(2 * (h0 - h_out))
    mass_out = rho_out * velocity_outlet * A_out

    return mass_out - mass_in

velocity_problem = spo.root_scalar(mass_balance, method='newton', x0=10)

# Calculate the mass-flow rate
V_in = velocity_problem.root
h_in = h0 - 0.5 * V_in ** 2
h_in = h_in.item()
prop = pgp("HmassSmass_INPUTS",h_in, s0)
rho_in = prop['d']
mass = rho_in * V_in * A_in # Mass flow rate

# Calcilate the pressure and density profile by imposing mass-flow rate at each section
rho_mass = []
p_mass = []
M_mass = []

for A in area_distribution:
    
    # local mass-flow
    def loc_mass_flow_balance(velocity_local):
        h_loc = h0 - 0.5 * velocity_local ** 2
        h_loc = h_loc.item()
        prop = pgp("HmassSmass_INPUTS",h_loc, s0)
        rho_loc = prop['d']
        mass_loc = rho_loc * velocity_local * A
        return mass - mass_loc
    
    velocity = spo.root_scalar(loc_mass_flow_balance, method='newton', x0=10)

    # Calculate the mass-flow rate
    V_loc = velocity.root
    # print(V_loc)
    h_loc = h0 - 0.5 * V_loc ** 2
    h_loc = h_loc.item()
    prop = pgp("HmassSmass_INPUTS",h_loc, s0)
    rho_mass.append(prop['d'])
    p_mass.append(prop['p'])
    M_mass.append(V_loc.item()/prop['a'])

# print(p_mass)

plt.figure('Pressure mass')
plt.plot(x,np.array(p_mass)/P0, marker= 'o')
plt.ylim([0.5, 1.1])

plt.figure('Density mass')
plt.plot(x,np.array(rho_mass)/rho0, marker= 'o')
plt.ylim([0.5, 1.1])

plt.figure('Mach mass')
plt.plot(x,np.array(M_mass), marker= 'o')
# plt.ylim([0.5, 1.1])

# p, rho, M = p_mass, rho_mass, M_mass

# np.savez("C:\\Users\\alber\\OneDrive - Danmarks Tekniske Universitet\\Desktop\\prefect_gas_6_55_new_points.npz", x=x, p=p_mass, rho=rho_mass, M=M_mass)
# data = np.load('C:\\Users\\alber\\OneDrive - Danmarks Tekniske Universitet\\Desktop\\prefect_gas.npz')
# print(data['x'])

plt.show()

# raise Exception('End debug')
# %%
