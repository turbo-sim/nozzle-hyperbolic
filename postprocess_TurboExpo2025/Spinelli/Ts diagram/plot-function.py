# %%
 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from barotropy import properties as props
from barotropy import graphics
import matplotlib.colors as mcolors
import matplotlib as mpl
import CoolProp.CoolProp as cp
 
import barotropy as bpy
 
# Define plot settings
SHOW_FIG = True
SAVE_FIG = True
 
# Create fluid object
fluid_name = "MDM"
fluid = cp.AbstractState("HEOS",fluid_name) # "BICUBIC&HEOS"
 
 
# Spinelli 1
data1 = pd.DataFrame({'case_name': ['Case 1'],
        'T_in':[541.0],
        'p_in':[9.20],
        'p_out':[9.20*0.3]})
 
# Spinelli 2
data2 = pd.DataFrame({'case_name': ['Case 2'],
        'T_in':[512.0],
        'p_in':[4.59],
        'p_out':[4.59*0.3]})
 
# Spinelli 3
data3 = pd.DataFrame({'case_name': ['Case 3'],
        'T_in':[537.0],
        'p_in':[0.84],
        'p_out':[0.84*0.3]})
 
 
# All data collected
case_data = pd.concat([data1,data2,data3], ignore_index=True)
print(case_data)
 
 
 
var_x = "s"
var_y = "T"
 
fig, ax = plt.subplots(figsize=(6.0, 5.0))
mpl.rcParams['font.family'] = 'Times New Roman'
# fig.axes(fontname='Times New Roman')
sze = 14 # font size
wdt = 1.5
wdt_case = 2
 
excel_data = "MDM_sat_tables.xlsx"
data = pd.read_excel(excel_data)
 
linestyles = ['-', '-', '-', '-']
 
# Apply the mapping if it exists; otherwise, use the variable name
ax.set_xlabel(props.LABEL_MAPPING.get(var_x, var_x), fontname='Times New Roman',fontsize=sze)
ax.set_ylabel(props.LABEL_MAPPING.get(var_y, var_y), fontname='Times New Roman',fontsize=sze)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontname('Times New Roman')
    label.set_fontsize(sze-2)
 
# Plot phase diagram for the first subplot
ax.plot(data['sl']*1e3,data['T'],color='black',linewidth=wdt,linestyle = '-')
ax.plot(data['sv']*1e3,data['T'],color='black',linewidth=wdt,linestyle = '-')
 
T = np.ones(2)*data['T'].iloc[-1]
s = np.array([data['sl'].iloc[-1],data['sv'].iloc[-1]])
ax.plot(s*1e3,T,color='black',linewidth=wdt,linestyle = '-')
 
### Create Z contour
# Create a set of points in the saturated gas region
T_min = 310 + 273.15
T_max = 200 + 273.15
 
s_min = 560
s_max = 900
 
temperatures = np.linspace(T_min, T_max, 500)
entropies = np.linspace(s_min, s_max, 500)
 
X, Y = np.meshgrid(entropies, temperatures)
 
points = np.vstack([X.ravel(), Y.ravel()]).T
X, Y = X.ravel(), Y.ravel()
 
# %%
 
# filtered_points = []
# Z = []
# for s, T in points:
#     try:
#         fluid.update(cp.SmassT_INPUTS,s,T)
#         phase = fluid.Q()
#         # print(phase)
#         if not 0 < phase < 1:  # Keep only gas phase points
#             fluid.update(cp.SmassT_INPUTS,s,T)
#             z = fluid.compressibility_factor()
#             if 0.4 <= z <= 1:
#                 Z.append(z)
#                 filtered_points.append((s, T))
#     except ValueError:
#         # Skip invalid states (e.g., outside the valid range for the fluid)
#         continue
 
# # print(filtered_points)
 
# # %%
 
# # Convert the filtered points back to arrays
# filtered_points = np.array(filtered_points)
# X_filtered, Y_filtered = filtered_points[:, 0], filtered_points[:, 1]
 
# # ax.plot(X_filtered,Y_filtered,color='black',linewidth=wdt,marker = 'o')
 
# # %%
# levels = np.linspace(0.4,1.0,13)
# Z = np.array(Z)
# X_filtered = X_filtered.reshape(-1, 1)  # Assuming it corresponds to a 2D grid
# Y_filtered = Y_filtered.reshape(-1, 1)
 
# Reshape Z to match the 2D grid dimensions
# X_reshaped, Y_reshaped = np.meshgrid(np.unique(X_filtered), np.unique(Y_filtered))
# Z_reshaped = Z.reshape(X_reshaped.shape)
 
# contour = ax.contourf(X_reshaped,Y_reshaped, Z_reshaped, levels = levels, cmap='viridis')
# cbar = plt.colorbar(contour)
# cbar.set_label('Contour Value')
 
X, Y = np.meshgrid(entropies, temperatures)
 
# Initialize Z as a 2D array
Z = np.zeros_like(X)
 
# Compute Z for the full grid
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        try:
            fluid.update(cp.SmassT_INPUTS, X[i, j], Y[i, j])
            Z[i, j] = fluid.compressibility_factor()
        except ValueError:
            Z[i, j] = np.nan
            print(Z[i,j])
 
       
levels = np.linspace(0,1.0,30) # 13
contour = ax.contourf(X, Y, Z, levels=levels, cmap='viridis')
plt.colorbar(contour)
 
 
 
 
 
plt.tight_layout(pad=1)
 
# Plot Lettieri
for i, row in case_data.iterrows():
    print(f"Running case {row['case_name']}")
    # total preessure in Pa
    p_in = float(row["p_in"])*1e5
 
    # Static pressure in Pa
    p_out = float(row["p_out"])*1e5
    # p_out = p_in/1.5
 
    # Total temperature in K
    T_in = row["T_in"]
 
    # Create barotropic model object
    fluid.update(cp.PT_INPUTS,p_in, T_in)
    s_in = fluid.smass()
 
    entr = np.ones(2)*s_in
 
    fluid.update(cp.PSmass_INPUTS,p_out,s_in)
    T_out = fluid.T()
 
    temp = [T_in, T_out]
    print(entr,temp)
 
    # Plot the calculated cases
    ax.plot(
        entr,
        temp,
        color=graphics.COLORS_MATLAB[i],
        linewidth=wdt_case,
        linestyle = '-',
        label = row['case_name'],
    )
    ax.plot(
        entr,
        temp,
        color=graphics.COLORS_MATLAB[i],
        marker = 'o',
        # label = row['case_name'],
    )
 
 
 
plt.ylim([495, 563])
plt.xlim([617, 820])
plt.legend()
plt.show()