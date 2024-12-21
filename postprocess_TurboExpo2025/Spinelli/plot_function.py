# Comparison with experimental results of Spinelli for real gas validation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import nozzle_hyperbolic as hyp
import pandas as pd
import barotropy as bpy

# fluid_name = "MDM"
# fluid = bpy.fluid_properties.Fluid(fluid_name)

hyp.set_plot_options(grid=False)
data = np.load("isentropic.npz")
data1 = np.load("rusanov_Ma15_Z063.npz")
data2 =  np.load("rusanov_Ma15_Z098.npz")
data3 = np.load("rusanov_Ma2_Z065.npz")
data4 = np.load("rusanov_Ma2_Z098.npz")
data5 = np.load("rusanov_Ma15_Z08.npz")

r, x, u, p, rho = data["r"], data["x"], data["u"], data["p"], data["rho"]
x1, u1, p1, rho1 = data1["x"], data1["u"], data1["p"], data1["rho"]
h1, x2, u2, p2, rho2 = data2["h"], data2["x"], data2["u"], data2["p"], data2["rho"]
h2, x3, u3, p3, rho3 = data3["h"], data3["x"], data3["u"], data3["p"], data3["rho"]
x4, u4, p4, rho4 = data4["x"], data4["u"], data4["p"], data4["rho"]
x5, u5, p5, rho5 = data5["x"], data5["u"], data5["p"], data5["rho"]
print("Number of elements:", len(x3))
# # Loop through each cell to compute compressibility factor Z
# compressibility_Z63 = np.zeros_like(x1)
# for j in range(len(x1)):
#     # Set the fluid state using density and specific internal energy at the last time step
#     state = fluid.get_state(bpy.DmassP_INPUTS, rho1[j], p1[j])
#     compressibility_Z63[j] = state.compressibility_factor

# compressibility_Z98 = np.zeros_like(x2)
# for j in range(len(x2)):
#     # Set the fluid state using density and specific internal energy at the last time step
#     state = fluid.get_state(bpy.DmassP_INPUTS, rho2[j], p2[j])
#     compressibility_Z98[j] = state.compressibility_factor

# compressibility_Z80 = np.zeros_like(x5)
# for j in range(len(x5)):
#     # Set the fluid state using density and specific internal energy at the last time step
#     state = fluid.get_state(bpy.DmassP_INPUTS, rho5[j], p5[j])
#     compressibility_Z80[j] = state.compressibility_factor


p_stagnationH_m15 = 9.2e5 # Ma1.5 Z0.63
p_stagnationH_m2 = 9.02e5 # Ma2 Z0.65
p_stagnationL_m15 = 0.84e5 # Ma1.5 Z0.98
p_stagnationL_m2 = 0.8e5  # Ma2 Z0.98
p_stagnation_5 = 4.59e5

min_index = np.argmin(h1)
x_of_h_min = 10.177035175879398
closest_index1 = np.argmin(np.abs(x1 - x_of_h_min))
closest_index5 = np.argmin(np.abs(x5 - x_of_h_min))

# print(f"The length of x1 is: {x2[min_index]}")
# print(f"The length of x1 is: {x1[closest_index1]}")
# print(f"The length of x1 is: {x5[closest_index5]}")

# Ma = u1/a_sound
p_cr_1 = p1[closest_index1]
p_cr_2 = p2[min_index]
p_cr_5 = p5[closest_index5]

exp_data_m15_z63 = r"C:\Users\ancio\OneDrive - Danmarks Tekniske Universitet\Documents\Python_Scripts\nozzle-hyperbolic\validation_cases\Spinelli2019\profiles\case_m15_z63.xlsx"
exp_data_m15_z63 = pd.read_excel(exp_data_m15_z63)
y_err_063 = np.array([0.006666676, 0.006666676, 0, 0]) 

exp_data_m15_z98 = r"C:\Users\ancio\OneDrive - Danmarks Tekniske Universitet\Documents\Python_Scripts\nozzle-hyperbolic\validation_cases\Spinelli2019\profiles\case_m15_z98.xlsx"
exp_data_m15_z98 = pd.read_excel(exp_data_m15_z98)
y_err_098 = np.array([0.041111172, 0.041111172, 0.015556757, 0.012221062]) 

exp_data_m2_z65 = r"C:\Users\ancio\OneDrive - Danmarks Tekniske Universitet\Documents\Python_Scripts\nozzle-hyperbolic\validation_cases\Spinelli2019\profiles\case_m20_z65.xlsx"
exp_data_m2_z65 = pd.read_excel(exp_data_m2_z65)

exp_data_m2_z98 = r"C:\Users\ancio\OneDrive - Danmarks Tekniske Universitet\Documents\Python_Scripts\nozzle-hyperbolic\validation_cases\Spinelli2019\profiles\case_m20_z98.xlsx"
exp_data_m2_z98 = pd.read_excel(exp_data_m2_z98)

exp_data_m2_z80 = r"C:\Users\ancio\OneDrive - Danmarks Tekniske Universitet\Documents\Python_Scripts\nozzle-hyperbolic\validation_cases\Spinelli2019\profiles\case_m15_z80.xlsx"
exp_data_m2_z80 = pd.read_excel(exp_data_m2_z80)
y_err_081 = np.array([0.006666676, 0.006666676, 0, 0]) 

Z_638098 = r"C:\Users\ancio\OneDrive - Danmarks Tekniske Universitet\Documents\Python_Scripts\nozzle-hyperbolic\validation_cases\Spinelli2019\profiles\Z_case_m15_z638098.xlsx"
Z_638098 = pd.read_excel(Z_638098)
Zy_err_063 = np.array([0.015590551, 0.014173228, 0.011811024, 0.010866142])
Zy_err_081 = np.array([0.012283465, 0.010866142, 0.008031496, 0.008031496])
Zy_err_098 = np.array([0.052440945, 0.059055118, 0.017480315, 0.030708661])
min_index0 = np.argmin(r)

plt.figure(figsize=(6, 5))
# plt.xlabel(r'Normalized axis position x/H (-)')
plt.ylabel(r'$P/P_0$ (-)')
# plt.plot(x, p/9.2e5, 'r-', linewidth = 1.1, label='Z 0.63 - Numerical')
plt.plot(x1, p1/p_stagnationH_m15, '-', color='coral', linewidth = 1.4, label=r'Case S1 - Num.')
plt.plot(x5, p5/p_stagnation_5, '-', color='red', linewidth = 1.4, label=r'Case S2 - Num.')
plt.plot(x2, p2/p_stagnationL_m15, '-', color='darkred', linewidth = 1.4, label= r'Case S3 - Num.')

plt.errorbar(exp_data_m15_z63['Position'], exp_data_m15_z63['pressure_ratio'], 
             yerr=y_err_063, fmt='o', color='coral', capsize=4, elinewidth=0.8, 
             markerfacecolor='coral', markeredgecolor='black', markeredgewidth=1, 
             label=r'Case S1 - Exp.')

plt.errorbar(exp_data_m2_z80['Position'], exp_data_m2_z80['pressure_ratio'], 
             yerr=y_err_081, fmt='o', color='red', capsize=4, elinewidth=0.8, 
             markerfacecolor='red', markeredgecolor='black', markeredgewidth=1, 
             label=r'Case S2 - Exp.')

plt.errorbar(exp_data_m15_z98['Position'], exp_data_m15_z98['pressure_ratio'], 
             yerr=y_err_098, fmt='o', color='darkred', capsize=4, elinewidth=0.8, 
             markerfacecolor='darkred', markeredgecolor='black', markeredgewidth=1, 
             label=r'Case S3 - Exp.')
ax1 = plt.gca()

ax1.set_ylim([0, 1.15])
ax1.set_xlim([-0.5, 15])

# Add the second y-axis for the nozzle area
plt.grid(False)
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(x, r, 'k-', linewidth=1, alpha=0.8)
ax2.set_ylabel(r'Normalized nozzle height h/L (-)', labelpad=10)  # Adjust labelpad to move it away from the frame
ax2.set_ylim([0, x1[-1]*3])
ax2.fill_between(x, 0, r, color='lightblue', alpha=0.3)
plt.tight_layout()

# Get handles and labels from both axes
handles1, labels1 = ax1.get_legend_handles_labels()  # From primary axis
handles2, labels2 = ax2.get_legend_handles_labels()  # From secondary axis

# Combine handles and labels
handles = handles1 + handles2
labels = labels1 + labels2

# Create a combined legend with your custom settings
plt.legend(
    handles, labels,
    loc='center left',
    # bbox_to_anchor=(0.03, 0.48),   # Custom positioning
    # fontsize='x-small',        # Custom font size
    handlelength=1,            # Shorten the line markers
    handletextpad=0.3,         # Reduce space between markers and text
    borderpad=0.3,             # Decrease padding inside the legend box
    frameon=True               # Keep the frame around the legend
)

# Create inset axes 1
ax_inset = inset_axes(ax1, width=0.5, height=1, loc='upper right', bbox_to_anchor=(0.83, 0.97), bbox_transform=ax1.transAxes)
plt.plot(x1, p1/p_stagnationH_m15, '-', color='coral', linewidth = 1.4, label='Z 0.63 - Numerical')
plt.plot(x5, p5/p_stagnation_5, '-', color='red', linewidth = 1.4, label='Z 0.81 - Numerical')
plt.plot(x2, p2/p_stagnationL_m15, '-', color='darkred', linewidth = 1.4, label='Z 0.98 - Numerical')

plt.errorbar(exp_data_m15_z63['Position'], exp_data_m15_z63['pressure_ratio'], 
             yerr=y_err_063, fmt='o', color='coral', capsize=4, elinewidth=0.8, 
             markerfacecolor='coral', markeredgecolor='black', markeredgewidth=1, 
             label='Z 0.63 - Experimental')

plt.errorbar(exp_data_m2_z80['Position'], exp_data_m2_z80['pressure_ratio'], 
             yerr=y_err_081, fmt='o', color='red', capsize=4, elinewidth=0.8, 
             markerfacecolor='red', markeredgecolor='black', markeredgewidth=1, 
             label='Z 0.81 - Experimental')

plt.errorbar(exp_data_m15_z98['Position'], exp_data_m15_z98['pressure_ratio'], 
             yerr=y_err_098, fmt='o', color='darkred', capsize=4, elinewidth=0.8, 
             markerfacecolor='darkred', markeredgecolor='black', markeredgewidth=1, 
             label='Z 0.98 - Experimental')

# Define zoomed-in range
ax_inset.set_xlim(5.88, 6.6)  # Specify x-range
ax_inset.set_ylim(0.89, 1)  # Specify y-range

x_zoom_min, x_zoom_max = 5.88, 6.6  # x-range for zoom
y_zoom_min, y_zoom_max = 0.89, 1  # y-range for zoom

ax_inset.set_xlim(x_zoom_min, x_zoom_max)
ax_inset.set_ylim(y_zoom_min, y_zoom_max)

# Set color for the frame (spines) of the inset axes
for spine in ax_inset.spines.values():
    spine.set_edgecolor('black')  # Change border color to red
    spine.set_linewidth(2)  # Change border thickness

# Mark the zoomed-in region with a rectangle using the same coordinates
rect = patches.Rectangle((x_zoom_min, y_zoom_min), x_zoom_max - x_zoom_min, y_zoom_max - y_zoom_min, 
                         linewidth=2, edgecolor='black', facecolor='none', linestyle='-')

ax1.add_patch(rect)

# Create inset axes 2
ax_inset = inset_axes(ax1, width=0.5, height=1.1, loc='upper right', bbox_to_anchor=(0.98, 0.74), bbox_transform=ax1.transAxes)
plt.plot(x1, p1/p_stagnationH_m15, '-', color='coral', linewidth = 1.4, label='Z 0.63 - Numerical')
plt.plot(x5, p5/p_stagnation_5, '-', color='red', linewidth = 1.4, label='Z 0.81 - Numerical')
plt.plot(x2, p2/p_stagnationL_m15, '-', color='darkred', linewidth = 1.4, label='Z 0.98 - Numerical')

plt.errorbar(exp_data_m15_z63['Position'], exp_data_m15_z63['pressure_ratio'], 
             yerr=y_err_063, fmt='o', color='coral', capsize=4, elinewidth=0.8, 
             markerfacecolor='coral', markeredgecolor='black', markeredgewidth=1, 
             label='Z 0.63 - Experimental')

plt.errorbar(exp_data_m2_z80['Position'], exp_data_m2_z80['pressure_ratio'], 
             yerr=y_err_081, fmt='o', color='red', capsize=4, elinewidth=0.8, 
             markerfacecolor='red', markeredgecolor='black', markeredgewidth=1, 
             label='Z 0.81 - Experimental')

plt.errorbar(exp_data_m15_z98['Position'], exp_data_m15_z98['pressure_ratio'], 
             yerr=y_err_098, fmt='o', color='darkred', capsize=4, elinewidth=0.8, 
             markerfacecolor='darkred', markeredgecolor='black', markeredgewidth=1, 
             label='Z 0.98 - Experimental')

x_zoom_min, x_zoom_max = 7.885, 8.533  # x-range for zoom
y_zoom_min, y_zoom_max = 0.785, 0.895  # y-range for zoom

ax_inset.set_xlim(x_zoom_min, x_zoom_max)
ax_inset.set_ylim(y_zoom_min, y_zoom_max)

# Set color for the frame (spines) of the inset axes
for spine in ax_inset.spines.values():
    spine.set_edgecolor('dimgray')  # Change border color to red
    spine.set_linewidth(2)  # Change border thickness

# Mark the zoomed-in region with a rectangle using the same coordinates
rect = patches.Rectangle((x_zoom_min, y_zoom_min), x_zoom_max - x_zoom_min, y_zoom_max - y_zoom_min, 
                         linewidth=2, edgecolor='dimgray', facecolor='none', linestyle='-')

ax1.add_patch(rect)

# Create inset axes 3
ax_inset = inset_axes(ax1, width=0.7, height=0.8, loc='upper right', bbox_to_anchor=(0.61, 0.58), bbox_transform=ax1.transAxes)
plt.plot(x1, p1/p_stagnationH_m15, '-', color='coral', linewidth = 1.4, label='Z 0.63 - Numerical')
plt.plot(x5, p5/p_stagnation_5, '-', color='red', linewidth = 1.4, label='Z 0.81 - Numerical')
plt.plot(x2, p2/p_stagnationL_m15, '-', color='darkred', linewidth = 1.4, label='Z 0.98 - Numerical')

plt.errorbar(exp_data_m15_z63['Position'], exp_data_m15_z63['pressure_ratio'], 
             yerr=y_err_063, fmt='o', color='coral', capsize=4, elinewidth=0.8, 
             markerfacecolor='coral', markeredgecolor='black', markeredgewidth=1, 
             label='Z 0.63 - Experimental')

plt.errorbar(exp_data_m2_z80['Position'], exp_data_m2_z80['pressure_ratio'], 
             yerr=y_err_081, fmt='o', color='red', capsize=4, elinewidth=0.8, 
             markerfacecolor='red', markeredgecolor='black', markeredgewidth=1, 
             label='Z 0.81 - Experimental')

plt.errorbar(exp_data_m15_z98['Position'], exp_data_m15_z98['pressure_ratio'], 
             yerr=y_err_098, fmt='o', color='darkred', capsize=4, elinewidth=0.8, 
             markerfacecolor='darkred', markeredgecolor='black', markeredgewidth=1, 
             label='Z 0.98 - Experimental')
x_zoom_min, x_zoom_max = 10, 10.58  # x-range for zoom
y_zoom_min, y_zoom_max = 0.595, 0.7  # y-range for zoom

ax_inset.set_xlim(x_zoom_min, x_zoom_max)
ax_inset.set_ylim(y_zoom_min, y_zoom_max)

# Set color for the frame (spines) of the inset axes
for spine in ax_inset.spines.values():
    spine.set_edgecolor('darkgray')  # Change border color to red
    spine.set_linewidth(2)  # Change border thickness

# Mark the zoomed-in region with a rectangle using the same coordinates
rect = patches.Rectangle((x_zoom_min, y_zoom_min), x_zoom_max - x_zoom_min, y_zoom_max - y_zoom_min, 
                         linewidth=2, edgecolor='darkgray', facecolor='none', linestyle='-')

ax1.add_patch(rect)

# Create inset axes 4
ax_inset = inset_axes(ax1, width=0.7, height=0.8, loc='upper right', bbox_to_anchor=(0.79, 0.3), bbox_transform=ax1.transAxes)
plt.plot(x1, p1/p_stagnationH_m15, '-', color='coral', linewidth = 1.4, label='Z 0.63 - Numerical')
plt.plot(x5, p5/p_stagnation_5, '-', color='red', linewidth = 1.4, label='Z 0.81 - Numerical')
plt.plot(x2, p2/p_stagnationL_m15, '-', color='darkred', linewidth = 1.4, label='Z 0.98 - Numerical')

plt.errorbar(exp_data_m15_z63['Position'], exp_data_m15_z63['pressure_ratio'], 
             yerr=y_err_063, fmt='o', color='coral', capsize=4, elinewidth=0.8, 
             markerfacecolor='coral', markeredgecolor='black', markeredgewidth=1, 
             label='Z 0.63 - Experimental')

plt.errorbar(exp_data_m2_z80['Position'], exp_data_m2_z80['pressure_ratio'], 
             yerr=y_err_081, fmt='o', color='red', capsize=4, elinewidth=0.8, 
             markerfacecolor='red', markeredgecolor='black', markeredgewidth=1, 
             label='Z 0.81 - Experimental')

plt.errorbar(exp_data_m15_z98['Position'], exp_data_m15_z98['pressure_ratio'], 
             yerr=y_err_098, fmt='o', color='darkred', capsize=4, elinewidth=0.8, 
             markerfacecolor='darkred', markeredgecolor='black', markeredgewidth=1, 
             label='Z 0.98 - Experimental')

x_zoom_min, x_zoom_max = 12.0, 12.6  # x-range for zoom
y_zoom_min, y_zoom_max = 0.342, 0.445  # y-range for zoom

ax_inset.set_xlim(x_zoom_min, x_zoom_max)
ax_inset.set_ylim(y_zoom_min, y_zoom_max)

# Set color for the frame (spines) of the inset axes
for spine in ax_inset.spines.values():
    spine.set_edgecolor('lightgray')  # Change border color to red
    spine.set_linewidth(2)  # Change border thickness

# Mark the zoomed-in region with a rectangle using the same coordinates
rect = patches.Rectangle((x_zoom_min, y_zoom_min), x_zoom_max - x_zoom_min, y_zoom_max - y_zoom_min, 
                         linewidth=2, edgecolor='lightgray', facecolor='none', linestyle='-')

ax1.add_patch(rect)


## plot compressibility Z

plt.figure(figsize=(6, 2.5))

# # Plot the lines
# plt.plot(x1, compressibility_Z63, '-', color='coral', linewidth=1.4)
# plt.plot(x2, compressibility_Z98, '-', color='darkred', linewidth=1.4)
# plt.plot(x5, compressibility_Z80, '-', color='red', linewidth=1.4)

# Plot the error bars
plt.errorbar(Z_638098['Position'], Z_638098['Z63'], yerr=Zy_err_063, fmt='o', 
             color='coral', capsize=4, elinewidth=0.8, markeredgecolor='black', markerfacecolor='coral')
plt.errorbar(Z_638098['Position'], Z_638098['Z98'], yerr=Zy_err_098, fmt='o', 
             color='darkred', capsize=4, elinewidth=0.8, markeredgecolor='black', markerfacecolor='darkred')
plt.errorbar(Z_638098['Position'], Z_638098['Z80'], yerr=Zy_err_081, fmt='o', 
             color='red', capsize=4, elinewidth=0.8, markeredgecolor='black', markerfacecolor='red')

# Get current axes
ax1 = plt.gca()

# Set axis limits
ax1.set_ylim([0.6, 1.05])
ax1.set_xlim([-0.5, 15])

# Set labels
plt.xlabel(r'Normalized axis position x/H (-)')
plt.ylabel(r'Compressibility $Z$ (-)')
plt.tight_layout(pad=1)  # Add extra padding to avoid clipping
plt.subplots_adjust(right=0.89)

plt.show()


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


