import matplotlib.pyplot as plt
import numpy as np
import nozzle_hyperbolic as hyp
import pandas as pd
import matplotlib.patches as patches



hyp.set_plot_options(grid=False)

# Plot function1 for showing shock at position 25 50 75 100
# yaml file: euler_adapted_expansion


# Load the data from the files
# data01 = np.load("Results/Case1_rusanov_N200_friction5um.npz")
data02 = np.load("Results/Case5_HLL_N180_friction0um.npz")
data03 = np.load("Results/Case1_rusanov_N180.npz")
data04 = np.load("Results/Case1_HLL_N200_friction10um.npz")
data05 = np.load("Results/Case5_rusanov_N180.npz")
# data06 = np.load("Results/Alberto/rusanov_lettieri_N150_pin59_tin314_pout17_rough5_gronnerud.npz")

# h, x1, u1, p1, rho1, Q1, a1 = data01["h"], data01["x"], data01["u"], data01["p"], data01["rho"], data01["Q"], data01["a"]
x2, u2, p2, rho2, Q2, a2 = data02["x"], data02["u"], data02["p"], data02["rho"], data02["Q"], data02["a"]
x3, u3, p3, rho3, Q3, a3 = data03["x"], data03["u"], data03["p"], data03["rho"], data03["Q"], data03["a"]
x4, u4, p4, rho4, a4 = data04["x"], data04["u"], data04["p"], data04["rho"], data04["a"]
x5, p5 = data05["x"], data05["p"]
# x6, p6 = data06["x"], data06["p"]

exp_case1 = 'Case1.xlsx'
exp_case1 = pd.read_excel(exp_case1)

CFD_case1 = 'Case1_CFD.xlsx'
CFD_case1 = pd.read_excel(CFD_case1)

exp_case5 = 'Case5.xlsx'
exp_case5 = pd.read_excel(exp_case5)

CFD_case5 = 'Case5_CFD.xlsx'
CFD_case5 = pd.read_excel(CFD_case5)

# min_index = np.argmin(h)

## Case 1
x_nucleation_case1 = 3.748

m3 = u3[-1] * rho3[-1] * 0.002006
# m4 = u4[-1] * rho4[-1] * 0.002006
print(m3)
# print(m4)
plt.figure(figsize=(6, 4))
ax1 = plt.gca()
ax1.set_ylim([15, 60])
ax1.set_xlim([-32, 60])
# ax1.set_xlabel(r'Position along nozzle axis (mm)')
ax1.set_ylabel(r'Static pressure (bar)')
# ax1.plot(x4*1e3, p4/1e5, '-', color='orange', linewidth=1.8, label="1D HEM smooth wall")
# plt.plot(x1*1e3, p1/1e5, '-', color='blue', linewidth=1.8, markerfacecolor='white', label="1D HEM k$_s$ = 5 ($\\mu$m)")
# plt.plot(x2*1e3, p2/1e5, '-', color='crimson', linewidth=1.8, label="1D HEM k$_s$ = 15 ($\\mu$m)")
# Main plot for ax1
ax2 = ax1.twinx()
ax2.plot(x3*1e3, (1-Q3), '-', linewidth=2, color="darkcyan", label=' 1D HEM - Liquid mass fraction')
ax2.plot(CFD_case1['position frac'], CFD_case1['liquid_fraction'], '--', linewidth=2, color="blue", label='Romei et al. - Liquid mass fraction')

# position frac	liquid_fraction

# Set labels, limits, and colors for ax2
ax2.set_ylabel(r'Liquid mass fraction (-)')  
ax2.set_ylim([0, 0.4])
y2_color = "darkcyan"
ax2.spines['right'].set_color(y2_color)
ax2.tick_params(axis='y', colors=y2_color)
ax2.yaxis.label.set_color(y2_color)

# Main plot for ax1 (plot after ax2)
ax1.plot(x3*1e3, p3/1e5, '-', color='crimson', linewidth=2, label="1D HEM - Rusanov")
ax1.plot(x4*1e3, p4/1e5, '-', color='orange', linewidth=2, label="1D HEM - HLL")
ax1.plot(CFD_case1['Position p'], CFD_case1['Pressure'], '--', color='black', linewidth=2, label='Romei et al. (2021)')
ax1.plot(exp_case1['Position'], exp_case1['Pressure'], 's', color='black', markersize=6, markerfacecolor='k', label='Lettieri et al. (2017)')

# Add vertical line to ax1
plt.axvline(x=x_nucleation_case1, color='darkorange', linestyle='-', linewidth=3, ymin=0, ymax=0.037, label="Lettieri et al. - Condensation onset")

# Combine legends from both axes
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
handles = handles1 + handles2
labels = labels1 + labels2

# Custom combined legend
plt.legend(
    handles, labels,
    loc='upper right',
    # bbox_to_anchor=(0.99, 0.99),  # Custom positioning
    handlelength=1.7,
    handletextpad=0.3,
    borderpad=0.3,
    frameon=True
)

rect = patches.Rectangle(
    (-27, 25),  # Bottom-left corner of the rectangle (x, y), moved down by 1 unit
    15.8,  # Width
    4,   # Height
    linewidth=1.3,
    edgecolor='black',
    facecolor='white'
)
ax1.add_patch(rect)

# Calculate the new center of the rectangle
rect_center_x = -27 + 15.8 / 2  # x + width/2
rect_center_y = 25 + 4 / 2   # y + height/2, adjusted for the new y position

# Add centered text
ax1.text(
    rect_center_x, rect_center_y,  # Use the updated center coordinates
    "Case L1",
    fontsize=15,
    verticalalignment='center',
    horizontalalignment='center'
)

# Display the plot
plt.tight_layout()

# # ---- Create Standalone Legend Figure ----
# fig_legend = plt.figure(figsize=(4, 2))  # Adjust figure size
# ax_legend = fig_legend.add_subplot(111)
# ax_legend.axis('off')  # No axes for the legend-only figure

# # Transfer legend to the new figure
# legend = ax_legend.legend(
#     handles, labels,
#     loc='center',
#     handlelength=1.5,
#     handletextpad=0.5,
#     borderpad=0.5,
#     frameon=True
# )

# plt.tight_layout()


## Case 5
x_nucleation_case5 = -4.675

plt.figure(figsize=(6, 4.35))
ax1 = plt.gca()
ax1.set_ylim([25, 85])
ax1.set_xlim([-32, 60])
plt.xlabel(r'Position along nozzle axis (mm)')
ax1.set_ylabel(r'Static pressure (bar)')
plt.plot(x5*1e3, p5/1e5, '-', color='crimson', linewidth=2, markerfacecolor='white', label="1D HEM smooth wall")
plt.plot(x2*1e3, p2/1e5, '-', color='orange', linewidth=2, markerfacecolor='white', label="1D HEM smooth wall")

# plt.plot(x4, p4/1e5, '-', color='orange', linewidth=1.5, label="friction")
plt.plot(CFD_case5['Position p'], CFD_case5['Pressure'], '--', color='black', linewidth=2, label='Romei et al. (2021)')
plt.plot(exp_case5['Position'], exp_case5['Pressure'], 's', color='black', markersize=6, markerfacecolor='k', label='Lettieri et al. (2017)')

# Add the second y-axis for the vapour quality area
plt.grid(False)
ax2 = ax1.twinx()
ax2.plot(x2*1e3, (1-Q2), '-', linewidth=2, color="darkcyan", label='Liquid mass fraction ')
ax2.plot(CFD_case5['position frac'], CFD_case5['liquid_fraction'], '--', linewidth=2, color="blue", label='Liquid mass fraction')

ax2.set_ylabel(r'Liquid mass fraction (-)')  # Adjust labelpad to move it away from the frame
ax2.set_ylim([0, 0.4])
y2_color = "darkcyan"
ax2.spines['right'].set_color(y2_color)  # Change the spine color
ax2.tick_params(axis='y', colors=y2_color)  # Change the tick color
ax2.yaxis.label.set_color(y2_color)  # Change the label color
plt.tight_layout()
plt.axvline(x=x_nucleation_case5, color='darkorange', linestyle='-', linewidth=3, ymin=0, ymax=0.038, label="Experimental condensation onset")

# Get handles and labels from both axes
handles1, labels1 = ax1.get_legend_handles_labels()  # From primary axis
handles2, labels2 = ax2.get_legend_handles_labels()  # From secondary axis
# Combine handles and labels
handles = handles1 + handles2
labels = labels1 + labels2

# Create a combined legend with your custom settings
# plt.legend(
#     handles, labels,
#     loc='upper right',
#     bbox_to_anchor=(0.98, 0.98),  # Custom positioning
#     # fontsize='x-small',        # Custom font size
#     handlelength=1,            # Shorten the line markers
#     handletextpad=0.3,         # Reduce space between markers and text
#     borderpad=0.3,             # Decrease padding inside the legend box
#     frameon=True               # Keep the frame around the legend
# )


rect = patches.Rectangle(
    (-27, 38.5),  # Bottom-left corner of the rectangle (x, y), moved down by 1 unit
    16,  # Width
    4*1.28,   # Height
    linewidth=1.3,
    edgecolor='black',
    facecolor='white'
)
ax1.add_patch(rect)

# Calculate the new center of the rectangle
rect_center_x = -27 + 16 / 2  # x + width/2
rect_center_y = 38.5 + 4*1.26 / 2   # y + height/2, adjusted for the new y position

# Add centered text
ax1.text(
    rect_center_x, rect_center_y,  # Use the updated center coordinates
    "Case L5",
    fontsize=15,
    verticalalignment='center',
    horizontalalignment='center'
)

plt.show()

