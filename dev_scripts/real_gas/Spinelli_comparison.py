# Comparison with experimental results of Spinelli for real gas validation

import matplotlib.pyplot as plt
import numpy as np
import nozzle_hyperbolic as hyp
import pandas as pd


hyp.set_plot_options(grid=False)
data1 = np.load("rusanov_Ma15_Z063.npz")
data2 =  np.load("rusanov_Ma15_Z098.npz")
data3 = np.load("rusanov_Ma2_Z065.npz")
data4 = np.load("rusanov_Ma2_Z098.npz")
data5 = np.load("rusanov_Ma15_Z08.npz")

x1, u1, p1, rho1 = data1["x"], data1["u"], data1["p"], data1["rho"]
h1, x2, u2, p2, rho2 = data2["h"], data2["x"], data2["u"], data2["p"], data2["rho"]
h2, x3, u3, p3, rho3 = data3["h"], data3["x"], data3["u"], data3["p"], data3["rho"]
x4, u4, p4, rho4 = data4["x"], data4["u"], data4["p"], data4["rho"]
x5, u5, p5, rho5 = data5["x"], data5["u"], data5["p"], data5["rho"]

p_stagnationH_m15 = 9.2e5 # Ma1.5 Z0.63
p_stagnationH_m2 = 9.02e5 # Ma2 Z0.65
p_stagnationL_m15 = 0.84e5 # Ma1.5 Z0.98
p_stagnationL_m2 = 0.8e5  # Ma2 Z0.98
p_stagnation_5 = 4.59e5

# Ma = u1/a_sound

exp_data_m15_z63 = r"C:\Users\ancio\OneDrive - Danmarks Tekniske Universitet\Documents\Python_Scripts\nozzle-hyperbolic\validation_cases\Spinelli2019\profiles\case_m15_z63.xlsx"
exp_data_m15_z63 = pd.read_excel(exp_data_m15_z63)

exp_data_m15_z98 = r"C:\Users\ancio\OneDrive - Danmarks Tekniske Universitet\Documents\Python_Scripts\nozzle-hyperbolic\validation_cases\Spinelli2019\profiles\case_m15_z98.xlsx"
exp_data_m15_z98 = pd.read_excel(exp_data_m15_z98)

exp_data_m2_z65 = r"C:\Users\ancio\OneDrive - Danmarks Tekniske Universitet\Documents\Python_Scripts\nozzle-hyperbolic\validation_cases\Spinelli2019\profiles\case_m20_z65.xlsx"
exp_data_m2_z65 = pd.read_excel(exp_data_m2_z65)

exp_data_m2_z98 = r"C:\Users\ancio\OneDrive - Danmarks Tekniske Universitet\Documents\Python_Scripts\nozzle-hyperbolic\validation_cases\Spinelli2019\profiles\case_m20_z98.xlsx"
exp_data_m2_z98 = pd.read_excel(exp_data_m2_z98)

exp_trial = r"C:\Users\ancio\OneDrive - Danmarks Tekniske Universitet\Documents\Python_Scripts\nozzle-hyperbolic\validation_cases\Spinelli2019\profiles\case_m15_z80.xlsx"
exp_trial = pd.read_excel(exp_trial)

plt.figure(figsize=(6, 5))
plt.xlabel(r'$x/L$')
plt.ylabel(r'$P / P_T$')
plt.plot(x1, p1/p_stagnationH_m15, 'b-', color='purple', linewidth = 1.1, label='Z 0.63 - Numerical')
plt.plot(x5, p5/p_stagnation_5, 'g-', color='green', linewidth = 1.1, label='Z 0.81 - Numerical')
plt.plot(x2, p2/p_stagnationL_m15, 'r-', color='orange', linewidth = 1.1, label='Z 0.98 - Numerical')

plt.plot(exp_data_m15_z63['Position'], exp_data_m15_z63['pressure_ratio'], '^', color='purple', markersize=6, markerfacecolor='w', label='Z 0.63 - Experimental')
plt.plot(exp_trial['Position'], exp_trial['pressure_ratio'], '^', color='green', markersize=6, markerfacecolor='w', label='Z 0.81 - Experimental')
plt.plot(exp_data_m15_z98['Position'], exp_data_m15_z98['pressure_ratio'], '^', color='orange', markersize=6, markerfacecolor='w', label='Z 0.98 - Experimental')

ax1 = plt.gca()
ax1.set_ylim([0.1, 1.2])

# Add the second y-axis for the nozzle area
plt.grid(False)
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(x2, h1, 'k-', linewidth=1, alpha=0.8, label='Nozzle height')
ax2.set_ylabel(r'$h / L$', labelpad=10)  # Adjust labelpad to move it away from the frame
ax2.set_ylim([0, 50])
ax2.fill_between(x2, 0, h1, color='lightblue', alpha=0.3)
plt.tight_layout()

# Get handles and labels from both axes
handles1, labels1 = ax1.get_legend_handles_labels()  # From primary axis
handles2, labels2 = ax2.get_legend_handles_labels()  # From secondary axis

# Combine handles and labels
handles = handles1 + handles2
labels = labels1 + labels2

plt.legend(
    handles, labels,
    loc='center left',
    bbox_to_anchor=(0.03, 0.5),  # Custom positioning
    fontsize='x-small',        # Custom font size
    handlelength=1,            # Shorten the line markers
    handletextpad=0.3,         # Reduce space between markers and text
    borderpad=0.3,             # Decrease padding inside the legend box
    frameon=True               # Keep the frame around the legend
)


# Plot Mach2

plt.figure(figsize=(6, 5))
plt.xlabel(r'$x/L$')
plt.ylabel(r'$P / P_T$')
plt.plot(x3, p3/p_stagnationH_m2, '-', color = 'purple', linewidth = 1.1, label='Z 0.65 - Numerical')
plt.plot(exp_data_m2_z65['Position'], exp_data_m2_z65['pressure_ratio'], 'o', color='purple', markersize=5, markerfacecolor='w', label='Z 0.65 - Experimental')
plt.plot(x4, p4/p_stagnationL_m2, '-', color = 'orange', linewidth = 1.1, label='Z 0.98 - Numerical')
plt.plot(exp_data_m2_z98['Position'], exp_data_m2_z98['pressure_ratio'], 'o', color='orange', markersize=5, markerfacecolor='w', label='Z 0.98 - Experimental')

ax1 = plt.gca()
ax1.set_ylim([0, 1.1])

# Add the second y-axis for the nozzle area
plt.grid(False)
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(x3, h2, 'k-', linewidth=1, alpha=0.8, label='Nozzle height')
ax2.set_ylabel(r'$h / L$', labelpad=10)  # Adjust labelpad to move it away from the frame
ax2.set_ylim([0, 50])
ax2.fill_between(x3, 0, h2, color='lightblue', alpha=0.3)
plt.tight_layout()

# Get handles and labels from both axes
handles1, labels1 = ax1.get_legend_handles_labels()  # From primary axis
handles2, labels2 = ax2.get_legend_handles_labels()  # From secondary axis

# Combine handles and labels
handles = handles1 + handles2
labels = labels1 + labels2

plt.legend(
    handles, labels,
    loc='center left',
    bbox_to_anchor=(0.03, 0.5),  # Custom positioning
    fontsize='x-small',        # Custom font size
    handlelength=1,            # Shorten the line markers
    handletextpad=0.3,         # Reduce space between markers and text
    borderpad=0.3,             # Decrease padding inside the legend box
    frameon=True               # Keep the frame around the legend
)

# p_stagnation = 4.58e5
# pos = [8.67, 10.60, 12.47, 14.37]
# pres = [3.912e5, 3.017e5, 1.825e5, 1.478e5]
# pos = np.array(pos)
# pres = np.array(pres)
# plt.figure(figsize=(6, 5))
# plt.xlabel(r'$x/L$')
# plt.ylabel(r'$P / P_T$')
# plt.plot(x5, p5, 'b-', linewidth = 1.1, label='')
# plt.plot(pos, pres, 'o', color='purple', markersize=5, markerfacecolor='w', label='Ma1.5 Z 0.63 - Experimental')
# # plt.plot(x2, p2/p_stagnationL_m15, 'r-', linewidth = 1.1, label='Ma1.5 Z 0.98 - Numerical')
# # plt.plot(exp_data_m15_z98['Position'], exp_data_m15_z98['pressure_ratio'], 'o', color='green', markersize=5, markerfacecolor='w', label='Ma1.5 Z 0.98 - Experimental')


plt.show()
