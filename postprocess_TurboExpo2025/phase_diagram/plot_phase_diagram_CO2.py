
# %%

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from barotropy import properties as props
from barotropy import graphics
import matplotlib.colors as mcolors
import matplotlib as mpl
from CoolProp.CoolProp import PropsSI
from matplotlib import font_manager


import barotropy as bpy

# Define plot settings
SHOW_FIG = True
SAVE_FIG = True

# Create folder to save results
DIR_OUT = "./output_phase_diagram"
save_figures = True
if not os.path.isdir(DIR_OUT):
    os.makedirs(DIR_OUT)

# Create fluid object
fluid_name = "CO2"
fluid = bpy.Fluid(name=fluid_name, backend="HEOS")

# Get critical properties
T_crit = PropsSI('Tcrit', fluid_name)  # Critical temperature
P_crit = PropsSI('Pcrit', fluid_name)  # Critical pressure
s_crit = fluid.critical_point.s
plt.rcParams['font.family'] = 'Times New Roman'

# Lettieri 1
data1 = pd.DataFrame({'case_name': ['Lettieri et al. - Case L1'],
        'T_in':[314.67],
        'p_in':[58.69],
        'p_out':[16]})

# Lettieri 5
data2 = pd.DataFrame({'case_name': ['Lettieri et al. - Case L5'],
        'T_in':[313.88],
        'p_in':[84.74],
        'p_out':[26]})

### Berana cases
case_summ = "cases_summary.xlsx"
case_table = pd.read_excel(case_summ)


# Flashing
case_index = 1
row = case_table[case_table["Case"] == case_index].squeeze()
data31 = pd.DataFrame({'case_name': ['Berana et al. - Case B'],
        'T_in':[row["T_0_in"]+273.15],
        'p_in':[row["P_0_in"]/1e5],
        'p_out':[row["P_out"]/1e5]})

case_index = 2
row = case_table[case_table["Case"] == case_index].squeeze()
data32 = pd.DataFrame({'case_name': ['Berana et al. - Case B'],
        'T_in':[row["T_0_in"]+273.15],
        'p_in':[row["P_0_in"]/1e5],
        'p_out':[row["P_out"]/1e5]})

case_index = 3
row = case_table[case_table["Case"] == case_index].squeeze()
data33 = pd.DataFrame({'case_name': ['Berana et al. - Case B'],
        'T_in':[row["T_0_in"]+273.15],
        'p_in':[row["P_0_in"]/1e5],
        'p_out':[row["P_out"]/1e5]})

# Condensation
case_index = 12
row = case_table[case_table["Case"] == case_index].squeeze()
data41 = pd.DataFrame({'case_name': ['Berana et al. - Case A'],
        'T_in':[row["T_0_in"]+273.15],
        'p_in':[row["P_0_in"]/1e5],
        'p_out':[row["P_out"]/1e5]})

case_index = 13
row = case_table[case_table["Case"] == case_index].squeeze()
data42 = pd.DataFrame({'case_name': ['Berana et al. - Case A'],
        'T_in':[row["T_0_in"]+273.15],
        'p_in':[row["P_0_in"]/1e5],
        'p_out':[row["P_out"]/1e5]})

case_index = 14
row = case_table[case_table["Case"] == case_index].squeeze()
data43 = pd.DataFrame({'case_name': ['Berana et al. - Case A'],
        'T_in':[row["T_0_in"]+273.15],
        'p_in':[row["P_0_in"]/1e5],
        'p_out':[row["P_out"]/1e5]})

# All data collected
data = pd.concat([data1,data2,data31,data32,data33,data41,data42,data43], ignore_index=True)
print(data)

# %%

# raise Exception('End Debug')

# ax = fluid.plot_phase_diagram(
#     "s", "T", plot_spinodal_line=True, plot_quality_isolines=True
# )

var_x = "s"
var_y = "T"


fig, ax = plt.subplots(figsize=(6.0, 5.0))
ax.set_xlabel(r's/s$_{cr}$ (-)')
ax.set_ylabel(r'T/T$_{cr}$ (-)')

ax.set_xlabel(props.LABEL_MAPPING.get(var_x, var_x), fontname='Times New Roman',fontsize=14)
ax.set_ylabel(props.LABEL_MAPPING.get(var_y, var_y), fontname='Times New Roman',fontsize=14)


# Increase the thickness of the axes' spines (frame)
spine_thickness = 1  # Set this value to increase the thickness of the frame
for spine in ax.spines.values():
    spine.set_linewidth(spine_thickness)

# mpl.rcParams['font.family'] = 'Times New Roman'
# fig.axes(fontname='Times New Roman')
sze = 14 # font size
wdt = 1.5
wdt_case = 2


# Parula color map
parula = [(0.2514, 0.2748, 0.8786),
               (0.5209, 0.6945, 0.4698),
               (0.7712, 0.9157, 0.3375),
               (0.9932, 0.9062, 0.1439)]

linestyles = ['--', '--', '--', '--']


# Apply the mapping if it exists; otherwise, use the variable name
# ax.set_xlabel(props.LABEL_MAPPING.get(var_x, var_x), fontname='Times New Roman',fontsize=sze)
# ax.set_ylabel(props.LABEL_MAPPING.get(var_y, var_y), fontname='Times New Roman',fontsize=sze)
# for label in ax.get_xticklabels() + ax.get_yticklabels():
#     label.set_fontname('Times New Roman')
#     label.set_fontsize(sze-2) 

# Plot phase diagram for the first subplot
ax = fluid.plot_phase_diagram(
    var_x,
    var_y,
    axes=ax,
    plot_critical_point=False,
    plot_saturation_line=False,
    plot_spinodal_line=False,
    plot_quality_isolines=True,
    N=50,
)

# Compute saturation lines
sat_liq, sat_vap = props.compute_saturation_line(fluid, N=100)
plt.plot(np.array(sat_liq['smass'])/s_crit,
    np.array(sat_liq['T'])/T_crit,
    color='black',
    linewidth=wdt,
    # label = 'Saturation line'
    )
plt.plot(np.array(sat_vap['smass'])/s_crit,
    np.array(sat_vap['T'])/T_crit,
    color='black',
    linewidth=wdt,)


plt.tight_layout(pad=1)

## Plot the critical isobar
 
# Generate a range of temperatures (from triple point to critical temperature)
T_min = PropsSI('Tmin', fluid_name)+10  # Minimum temperature
T_max = 2.5 * T_crit
temperatures = np.linspace(T_min, T_max, 1000)  # Temperature range (K)
 
# Arrays to store entropy values for the critical isobar
entropies = []
 
# Calculate entropy for the critical isobar
for T in temperatures:
    # try:
    entropies.append(PropsSI('S', 'T', T, 'P', P_crit, fluid_name))  # Liquid entropy
 
 
# ax.plot(entropies,temperatures,color='darkblue', linestyle='-',linewidth=wdt,label='Critical pressure isobar')
plt.plot(np.array(entropies)/s_crit,np.array(temperatures)/T_crit,color='black', linestyle='-.',linewidth=wdt, label = "Critical isobar")

# Generate spinodal lines
spinodal_liq, spinodal_vap = props.compute_spinodal_line(fluid, N=100)

# Print spinodal lines
plt.plot(np.array(spinodal_liq['smass'])/s_crit,
    np.array(spinodal_liq['T'])/T_crit,
    color='black',
    linewidth=wdt,
    label = 'Spinodal line',
    linestyle=':')
plt.plot(np.array(spinodal_vap['smass'])/s_crit,
    np.array(spinodal_vap['T'])/T_crit,
    color='black',
    linewidth=wdt,
    linestyle=':')

sat_liq["smass"] = sat_liq['smass'][20:]
sat_liq["T"] = sat_liq['T'][20:]
x = np.concatenate([sat_liq['smass'][::-1],spinodal_liq['smass']])
y = np.concatenate([sat_liq['T'][::-1],spinodal_liq['T']])
ax.fill(x/s_crit, y/T_crit, color='skyblue', alpha=0.5, label='Superheated liquid')

sat_vap["smass"] = sat_vap['smass'][20:]
sat_vap["T"] = sat_vap['T'][20:]
x = np.concatenate([sat_vap['smass'][::-1],spinodal_vap['smass']])
y = np.concatenate([sat_vap['T'][::-1],spinodal_vap['T']])
ax.fill(x/s_crit, y/T_crit, color='red', alpha=0.2, label='Supersaturated vapor')


# Plot Lettieri
for i, row in data.iterrows():
    print(f"Running case {row['case_name']}")
    # total preessure in Pa
    p_in = float(row["p_in"])*1e5

    # Static pressure in Pa
    p_out = float(row["p_out"])*1e5
    # p_out = p_in/1.5

    # Total temperature in K
    T_in = row["T_in"]

    # Create barotropic model object
    model = bpy.BarotropicModel(
        fluid_name=fluid_name,
        T_in=T_in,
        p_in=p_in,
        p_out=p_out,
        efficiency=1,
        calculation_type="equilibrium",
        HEOS_solver="hybr",
        ODE_solver="LSODA",
        ODE_tolerance=1e-9,
        polynomial_degree=5,
        polynomial_format="horner",
        output_dir=DIR_OUT,
        polynomial_variables=["density", "viscosity", "speed_sound"],
    )

    # Evaluate barotropic model and export polynomial expressions
    model.solve()
    if i<2:
        # Plot the calculated cases
        plt.plot(
            model.states['s']/s_crit,
            model.states['T']/T_crit,
            color=graphics.COLORS_MATLAB[i],
            # color = parula[i],
            linewidth=wdt_case,
            linestyle = linestyles[i],
            label = row['case_name'],

        )
    plt.plot(
        model.states['s'][0]/s_crit,
        model.states['T'][0]/T_crit,
        color=graphics.COLORS_MATLAB[i],
        # color = parula[i],
        linewidth=wdt_case,
        marker="o",
        markerfacecolor=graphics.COLORS_MATLAB[i],
    )
    plt.plot(
        model.states['s'][-1]/s_crit,
        model.states['T'][-1]/T_crit,
        color=graphics.COLORS_MATLAB[i],
        # color = parula[i],
        linewidth=wdt_case,
        marker="o",
        markerfacecolor=graphics.COLORS_MATLAB[i],
    )

    if i > 1:
        break

# Start counter
count = 1

data_berana = pd.concat([data31,data32,data33,data41,data42,data43], ignore_index=True)
print(data_berana)

markers = ['o', 'o', 'o', 'o', 'o', 'o']

# Plot Berana
for i, row in data_berana.iterrows():
    print(f"Running case {row['case_name']}")
    # total preessure in Pa
    p_in = float(row["p_in"])*1e5

    # Static pressure in Pa
    p_out = float(row["p_out"])*1e5
    # p_out = p_in/1.5

    # Total temperature in K
    T_in = row["T_in"]

    # Create barotropic model object
    model = bpy.BarotropicModel(
        fluid_name=fluid_name,
        T_in=T_in,
        p_in=p_in,
        p_out=p_out,
        efficiency=1,
        calculation_type="equilibrium",
        HEOS_solver="hybr",
        ODE_solver="LSODA",
        ODE_tolerance=1e-9,
        polynomial_degree=5,
        polynomial_format="horner",
        output_dir=DIR_OUT,
        polynomial_variables=["density", "viscosity", "speed_sound"],
    )

    # Evaluate barotropic model and export polynomial expressions
    model.solve()

    # Plot the calculated cases
    
    # Plot the line
    if i in [0, 3]:
        count = count + 1
        plt.plot(
            model.states['s']/s_crit,
            model.states['T']/T_crit,
            color=graphics.COLORS_MATLAB[count],
            # color = parula[i],
            linewidth=wdt_case,
            linestyle = linestyles[count],
        )
        plt.plot(
            model.states['s'][0]/s_crit,
            model.states['T'][0]/T_crit,
            color=graphics.COLORS_MATLAB[count],
            # color = parula[i],
            linewidth=wdt_case,
            marker="o",
            markerfacecolor=graphics.COLORS_MATLAB[count],
        )

    plt.plot(
        model.states['s'][-1]/s_crit,
        model.states['T'][-1]/T_crit,
        color=graphics.COLORS_MATLAB[count],
        # color = parula[i],
        linewidth=wdt_case,
        marker=markers[i],
        markerfacecolor=graphics.COLORS_MATLAB[count],
        # label = row['case_name'],
    )

plt.plot([], [], linestyle="--", color="purple", linewidth=2, label="Berana et al. - Case B1")
plt.plot([], [], linestyle="--", color="#E5A900", linewidth=2, label="Berana et al. - Case B2")

plt.plot(s_crit/s_crit,T_crit/T_crit, marker="o", color="k", markerfacecolor="w")


# Show figures
plt.xlim([0.85, 1.35])
plt.ylim([0.79, 1.15])
ax.grid(False)


font_prop = font_manager.FontProperties(family='Times New Roman', size=12)
# Get the current legend handles and labels
handles, labels = plt.gca().get_legend_handles_labels()

# Reorder handles and labels (swap columns)
new_order = [4, 5, 6, 7, 0, 1, 2, 3]  # Adjust based on your desired order
handles = [handles[i] for i in new_order]
labels = [labels[i] for i in new_order]

# Reapply the legend with the reordered items
plt.legend(
    handles, labels,
    loc='upper left',
    ncol=2,
    handlelength=1.5,
    handletextpad=0.3,
    borderpad=0.3,
    frameon=True,
    columnspacing=1.5,
    labelspacing=0.4,
    prop=font_prop
)

# ax.legend(fontsize = sze-2,
#         loc= 'upper left',
#         prop={'family': 'Times New Roman'})
# ax.legend(fontsize = sze-2,
#         loc= 'upper left',)
# Set tick label font size and font family

ax.tick_params(axis='both', labelsize=12)
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname('Times New Roman')

ax.set_xlabel(r'$s/s_{\text{cr}}$ (-)', fontname='Times New Roman', fontsize=14)
ax.set_ylabel(r'$T/T_{\text{cr}}$ (-)', fontname='Times New Roman', fontsize=14)


plt.show()


# raise Exception('end debug')