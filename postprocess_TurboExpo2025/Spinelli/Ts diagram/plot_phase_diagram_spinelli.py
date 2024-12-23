
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

# Define plot settings
SHOW_FIG = True
SAVE_FIG = False

# Create fluid object
fluid_name = "MDM"
fluid = cp.AbstractState("REFPROP",fluid_name) # "BICUBIC&HEOS"

fluid.update(cp.QT_INPUTS, 1,fluid.T_critical())
T_crit = fluid.T_critical()
s_crit = fluid.smass()
# %%

# Spinelli 1
data1 = pd.DataFrame({'case_name': ['Case S1'],
        'T_in':[541.0],
        'p_in':[9.20],
        'p_out':[9.20*0.3]})

# Spinelli 2
data2 = pd.DataFrame({'case_name': ['Case S2'],
        'T_in':[512.0],
        'p_in':[4.59],
        'p_out':[4.59*0.3]})

# Spinelli 3
data3 = pd.DataFrame({'case_name': ['Case S3'],
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
wdt = 1.2
wdt_case = 1.5

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
ax.plot(data['sl']*1e3/s_crit,data['T']/T_crit,color='black',linewidth=wdt,linestyle = '-')
ax.plot(data['sv']*1e3/s_crit,data['T']/T_crit,color='black',linewidth=wdt,linestyle = '-')

T = np.ones(2)*data['T'].iloc[-1]/T_crit
s = np.array([data['sl'].iloc[-1]*1e3,data['sv'].iloc[-1]*1e3])/s_crit
ax.plot(s,T,color='black',linewidth=wdt,linestyle = '-')

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
            # print(Z[i,j]) 

        
step_size = 0.05
levels = np.linspace(0.4, 1.0, int((1 - 0.4) / step_size))
levels = np.linspace(0.4, 1.0, 13)
colormaps = ['viridis', # 0
        'RdYlBu_r', # 1
        'inferno', # 2
        'hsv', # 3
        'turbo', # 4
        'twilight', # 5
        'twilight_shifted' # 6
        ]
contour = ax.contourf(X/s_crit, Y/T_crit, Z, levels=levels, cmap=colormaps[0])
plt.colorbar(contour,  label=r'$Z$ (-)',)

plt.tight_layout(pad=1)

colors = ['coral', 'red', 'darkred']
markers = ['o',  '^', 's']

# Plot Lettieri
for i, row in case_data.iterrows():
    # print(f"Running case {row['case_name']}")
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

    temp = np.array([T_in, T_out])
    # print(entr,temp)

    ax.plot(
        entr/s_crit,
        temp/T_crit,
        color=colors[i],
        marker = markers[i],
        linestyle = '--',
        label = row['case_name'],
    )


x = np.concatenate([data['sl'],data['sv'][::-1]])*1e3/s_crit
y = np.concatenate([data['T'],data['T'][::-1]])/T_crit
ax.fill(x, y, color='white', alpha=1)


fluid.update(cp.QT_INPUTS, 1,fluid.T_critical())
xp = [705,        800, data['sl'].iloc[0]*1e3, data['sl'].iloc[0]*1e3 ]
yp = [566,   600, 600,                     data['T'].iloc[0]]
x = np.concatenate([data['sl']*1e3,xp])/s_crit
y = np.concatenate([data['T'],yp])/T_crit
ax.fill(x, y, color='white', alpha=1)

ax.plot(1,1, color='black', marker= 'o',markerfacecolor='none', markersize = 5)



plt.ylim([(273.15 + 220)/T_crit, 580/T_crit])
plt.xlim([550/s_crit, 850/s_crit])
ax.grid(False)
ax.legend(fontsize = sze-2,
        loc= 'upper left',
        prop={'family': 'Times New Roman'})
ax.legend(fontsize = sze-2,
        loc= 'upper left',)
ax.set_xlabel(r'$s/s_{\text{cr}}$ (-)', fontname='Times New Roman', fontsize=14)
ax.set_ylabel(r'$T/T_{\text{cr}}$ (-)', fontname='Times New Roman', fontsize=14)
if SAVE_FIG:
    plt.savefig("mdm_phase_diagram.pdf", format='pdf', bbox_inches='tight')
plt.show()


