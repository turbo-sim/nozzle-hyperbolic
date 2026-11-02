# Testing friction factors from Haaland and Purdue uni paper
# %%
import nozzle_hyperbolic.fvm_solver as hyp
import matplotlib.pyplot as plt
import numpy as np
import barotropy as bpy

Re = np.linspace(1000, 100000, 100000)

def andrea_f_gas(Re):
    if Re < 2300:
        f = 64/Re

    else:
        f = 0.184 * Re ** (-0.2)
    
    return f
    


f_haaland = []
f_purdue = []
f_andrea = []

for r in Re:
    f_H = hyp.Haaland(r, 0, 1)
    f_P = hyp.f_HEM_purdue(r)
    f_A = andrea_f_gas(r)

    f_haaland += [f_H]
    f_purdue += [f_P*4]
    f_andrea += [f_A]

# %%
# plt.plot(Re, f_haaland, label = "Haaland")
# plt.plot(Re, f_purdue, label = "Purdue")
# plt.plot(Re, f_andrea, label = "Lockhart-Martinelli gas")

import barotropy as bpy
 
fluid = bpy.fluid_properties.Fluid("CO2")
p0 = 90e5
T0 = 318

state = fluid.get_state(bpy.fluid_properties.PT_INPUTS, p0, T0)
s0 = state.s
Tcrit = fluid.critical_point.T


state = state = fluid.get_state(bpy.fluid_properties.SmassT_INPUTS, s0, Tcrit*0.99)
T_max = state.T
T_min = T_max*0.8

temperatures = np.linspace(T_min, T_max, 200)

vis_Bett = []
vis_MacA = []
vis_L = []
vis_V = []

for t in temperatures:
    sL = fluid.get_state(bpy.fluid_properties.QT_INPUTS, 0, t)
    sV = fluid.get_state(bpy.fluid_properties.QT_INPUTS, 1, t)
    rho_L = sL.rho
    rho_V = sV.rho
    mu_L = sL.mu
    mu_V = sV.mu
    
    sq = fluid.get_state(bpy.fluid_properties.SmassT_INPUTS, s0, t)
    x = sq.Q

    mu_bett = hyp.viscosity_Beattie(mu_L, mu_V, rho_L, rho_V, x)
    mu_ma = hyp.viscosity_MacAdams(mu_L, mu_V, x)
    vis_Bett.append(mu_bett)
    vis_MacA.append(mu_ma)
    vis_L.append(mu_L)
    vis_V.append(mu_V)

# plt.plot(temperatures,vis_Bett,label = 'Bettie')
# plt.plot(temperatures,vis_MacA,label = 'MacA')
# # plt.plot(temperatures,vis_L,label = 'L')
# plt.plot(temperatures,vis_V,label = 'V')
# plt.legend()
# plt.show()



# %%


# %%
import numpy as np
data = np.load("lm_param_hem_b596.npz")
LM = data["LM_vec"]
x = np.linspace(0,1,len(LM))
plt.plot(x,LM)
plt.show()

# %%
