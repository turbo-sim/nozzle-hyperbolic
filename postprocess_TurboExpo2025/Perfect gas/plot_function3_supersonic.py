# Plot function 2: compare different solvers with shock at two positions¨
# yaml file: Supersonic_comparison

import matplotlib.pyplot as plt
import numpy as np
import nozzle_hyperbolic as hyp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

hyp.set_plot_options(grid=False)

# Load the data from the files
data01 = np.load("Results/exact_supersonic.npz")
# data02 = np.load("exact_supersonic.npz")
# data03 = np.load("exact_supersonic.npz")
# data1 = np.load("Results/rusanov_supersonic.npz")
data2 = np.load("Results/roe_supersonic.npz")
data3 = np.load("Results/hll_supersonic.npz")
data4 = np.load("Results/rusanov_supersonic.npz")
data5 = np.load("Results/roe_subsonic.npz")
data6 = np.load("Results/hll_subsonic.npz")
data7 = np.load("Results/rusanov_subsonic.npz")
data8 = np.load("Results/prefect_gas.npz")
# data9 = np.load("Results/hll_supersonic.npz")

L, r01, x01, u01, p01, rho01 = data01["L"], data01["r"], data01["x"], data01["u"], data01["p"], data01["rho"]
# r02, x02, u02, p02, rho02 = data02["r"], data02["x"], data02["u"], data02["p"], data02["rho"]
# r03, x03, u03, p03, rho03 = data03["r"], data03["x"], data03["u"], data03["p"], data03["rho"]

# x, u1, p1, rho1 = data1["x"], data1["u1"], data1["p1"], data1["rho1"]
x2, u2, p2, rho2 = data2["x"], data2["u1"], data2["p1"], data2["rho1"]
x3, u3, p3, rho3= data3["x"], data3["u1"], data3["p1"], data3["rho1"]
x4, u4, p4, rho4 = data4["x"], data4["u1"], data4["p1"], data4["rho1"]
x5, u5, p5, rho5 = data5["x"], data5["u1"], data5["p1"], data5["rho1"]
x6, u6, p6, rho6 = data6["x"], data6["u1"], data6["p1"], data6["rho1"]
x7, u7, p7, rho7 = data7["x"], data7["u1"], data7["p1"], data7["rho1"]
x, M, p, rho = data8["x"], data8["M"], data8["p"], data8["rho"]
# x, u9, p9, rho9 = data9["x"], data9["u1"], data9["p1"], data9["rho1"]
n_points = 20
n_points0 = 300


# Generate indices for each vector
indices_x2 = np.linspace(1, len(x2) - 1, n_points, dtype=int)
indices_x3 = np.linspace(0, len(x3) - 1, n_points, dtype=int)
indices_x4 = np.linspace(0, len(x4) - 1, n_points, dtype=int)
indices_x5 = np.linspace(1, len(x5) - 1, n_points, dtype=int)
indices_x6 = np.linspace(0, len(x6) - 1, n_points, dtype=int)
indices_x7 = np.linspace(0, len(x7) - 1, n_points, dtype=int)

indices_x02 = np.linspace(1, len(x2) - 1, n_points0, dtype=int)
indices_x03 = np.linspace(0, len(x3) - 1, n_points0, dtype=int)
indices_x04 = np.linspace(0, len(x4) - 1, n_points0, dtype=int)
indices_x05 = np.linspace(1, len(x5) - 1, n_points0, dtype=int)
indices_x06 = np.linspace(0, len(x6) - 1, n_points0, dtype=int)
indices_x07 = np.linspace(0, len(x7) - 1, n_points0, dtype=int)


# Add offsets to stagger points
indices_x3 = np.clip(indices_x3 + 15, 0, len(x3) - 1)  # Offset by 1, ensure it doesn't exceed bounds
indices_x4 = np.clip(indices_x4 + 30, 0, len(x4) - 1)  # Offset by 2, ensure it doesn't exceed bounds
indices_x5 = np.clip(indices_x5 + 45, 0, len(x5) - 1)  # Offset by 1, ensure it doesn't exceed bounds
indices_x6 = np.clip(indices_x6 + 60, 0, len(x6) - 1)  # Offset by 2, ensure it doesn't exceed bounds
indices_x7 = np.clip(indices_x7 + 75, 0, len(x7) - 1)  # Offset by 2, ensure it doesn't exceed bounds

min_index = np.argmin(r01)

R = 287
gamma = 1.4
T_stagnation = 500
p_stagnation = 7e5
rho_stagnation = p_stagnation/(R*T_stagnation)

a_sound_roe_sup = np.sqrt(gamma*p2/rho2)
Ma_roe_sup = u2/a_sound_roe_sup
a_sound_hll_sup = np.sqrt(gamma*p3/rho3)
Ma_hllc_sup = u3/a_sound_hll_sup
a_sound_rus_sup = np.sqrt(gamma*p4/rho4)
Ma_rus_sup = u4/a_sound_rus_sup

a_sound_roe_sub = np.sqrt(gamma*p5/rho5)
Ma_roe_sub = u5/a_sound_roe_sub
a_sound_hll_sub = np.sqrt(gamma*p6/rho6)
Ma_hllc_sub = u6/a_sound_hll_sub
a_sound_rus_sub = np.sqrt(gamma*p7/rho7)
Ma_rus_sub = u7/a_sound_rus_sub

a_sound_exact = np.sqrt(gamma*p01/rho01)
Ma_exact = u01/a_sound_exact
print(p01[-1, 0])
print(p5[0,-1]/(287*rho5[0,-1]))
print(r01[-1])

# print(Ma_exact[-1,1])
# print((2*r01[-1])**2/(2*r01[min_index])**2)

plt.figure(figsize=(6, 5))
plt.xlabel(r'Normalized axis position x/L (-)', fontsize=18)  # Increase xlabel size
plt.ylabel(r'$P/P_0$ (-)', fontsize=18)  # Increase ylabel size
# Increase tick label sizes
plt.plot(x01[:min_index], p01[:min_index, 0] / p_stagnation, '--', color='k', linewidth=1.6)
# plt.plot(x01[min_index:], p01[min_index:, 1] / p_stagnation, '--', color='k', linewidth=1.6)
# plt.plot(x01[min_index:], p01[min_index:,0] / p_stagnation, '--', color='k', linewidth=1.6, label='Exact solution')
# plt.plot(x2[indices_x2], p2[indices_x2, -1] / p_stagnation, 's', color='orange', linewidth=0.75, markersize=4, markerfacecolor='w', label='Roe supersonic')
# plt.plot(x3[indices_x3], p3[indices_x3, -1] / p_stagnation, '^', color='green', linewidth=0.75, markersize=4, markerfacecolor='w', label='HLL supersonic')
# plt.plot(x4[indices_x4], p4[indices_x4, -1] / p_stagnation, 's', color='purple', linewidth=0.75, markersize=4, markerfacecolor='w', label='Rusanov supersonic')
# plt.plot(x5[indices_x5], p5[indices_x5, -1] / p_stagnation, '^', color='orange', linewidth=0.75, markersize=4, markerfacecolor='w', label='Roe subsonic')
# plt.plot(x6[indices_x6], p6[indices_x6, -1] / p_stagnation, '*', color='green', linewidth=0.75, markersize=4, markerfacecolor='w', label='HLL subsonic')
# plt.plot(x7[indices_x7], p7[indices_x7, -1] / p_stagnation, 's', color='purple', linewidth=0.75, markersize=4, markerfacecolor='w', label='Rusanov subsonic')
plt.plot(x2[indices_x02], p2[indices_x02, -1] / p_stagnation, '-', color='orange', linewidth=1.6, markersize=4, markerfacecolor='w', label='Roe supersonic')
plt.plot(x3[indices_x03], p3[indices_x03, -1] / p_stagnation, '-.', color='red', linewidth=1.6, markersize=4, markerfacecolor='w', label='HLL supersonic')
plt.plot(x4[indices_x04], p4[indices_x04, -1] / p_stagnation, '--', color='darkred', linewidth=1.6, markersize=4, markerfacecolor='w', label='Rusanov supersonic')
plt.plot(x5[indices_x05], p5[indices_x05, -1] / p_stagnation, '-', color='mediumvioletred', linewidth=1.6, markersize=4, markerfacecolor='w', label='Roe subsonic')
plt.plot(x6[indices_x06], p6[indices_x06, -1] / p_stagnation, '-.', color='purple', linewidth=1.6, markersize=4, markerfacecolor='w', label='HLL subsonic')
plt.plot(x7[indices_x07], p7[indices_x07, -1] / p_stagnation, '--', color='indigo', linewidth=1.6, markersize=4, markerfacecolor='w', label='Rusanov subsonic')
plt.plot(x, p / 7e5, '--', color='black', linewidth=1.6, markersize=4, markerfacecolor='w', label='Rusanov supersonic')

# Add the second y-axis for the nozzle area
plt.grid(False)
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(x01, r01, 'k-', linewidth=1, alpha=0.8)
ax2.set_ylabel(r'Normalized nozzle radius r/L (-)', fontsize=18, labelpad=10)  # Adjust labelpad to move it away from the frame
ax1.set_ylim([0.01, 1.1])
ax2.set_ylim([0, L*3])
ax2.fill_between(x01, 0, r01, color='lightblue', alpha=0.3)

# Get handles and labels from both axes
handles1, labels1 = ax1.get_legend_handles_labels()  # From primary axis
handles2, labels2 = ax2.get_legend_handles_labels()  # From secondary axis

# Combine handles and labels
handles = handles1 + handles2
labels = labels1 + labels2

# # Create a combined legend with your custom settings
# plt.legend(
#     handles, labels,
#     loc='center left',
#     # bbox_to_anchor=(0.03, 0.48),   # Custom positioning
#     # fontsize='x-small',        # Custom font size
#     handlelength=1,            # Shorten the line markers
#     handletextpad=0.3,         # Reduce space between markers and text
#     borderpad=0.3,             # Decrease padding inside the legend box
#     frameon=True,               # Keep the frame around the legend
#     fontsize=14,  # Replace 12 with your desired font size
# )
plt.tight_layout()

# Create inset axes
ax_inset = inset_axes(ax1, width=0.7, height=0.7, loc='center right', bbox_to_anchor=(0.97, 0.47), bbox_transform=ax1.transAxes)

# Define zoomed-in range
ax_inset.set_xlim(0.998, 1)  # Specify x-range
ax_inset.set_ylim(0.10925, 0.111)  # Specify y-range
plt.plot(x01[min_index:], p01[min_index:, 1] / p_stagnation, '--', color='blue', linewidth=1.6)
plt.plot(x2[indices_x02], p2[indices_x02, -1] / p_stagnation, '-', color='orange', linewidth=1.6, markersize=4, markerfacecolor='w', label='Roe supersonic')
plt.plot(x3[indices_x03], p3[indices_x03, -1] / p_stagnation, '-.', color='red', linewidth=1.6, markersize=4, markerfacecolor='w', label='HLL supersonic')
plt.plot(x4[indices_x04], p4[indices_x04, -1] / p_stagnation, '--', color='darkred', linewidth=1.6, markersize=4, markerfacecolor='w', label='Rusanov supersonic')
ax_inset.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.3f}'))
ax_inset.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.3f}'))

# Add rectangle and lines to indicate zoomed region
mark_inset(ax1, ax_inset, loc1=3, loc2=4, fc="none", ec="0.5")


# Create inset axes
ax_inset = inset_axes(ax1, width=0.7, height=0.7, loc='upper right', bbox_to_anchor=(0.58, 0.48), bbox_transform=ax1.transAxes)

# Define zoomed-in range
ax_inset.set_xlim(0.65, 0.68)  # Specify x-range
ax_inset.set_ylim(0.69, 0.774)  # Specify y-range
plt.plot(x5[indices_x05], p5[indices_x05, -1] / p_stagnation, '-', color='mediumvioletred', linewidth=1.6, markersize=4, markerfacecolor='w', label='Roe subsonic')
plt.plot(x6[indices_x06], p6[indices_x06, -1] / p_stagnation, '-.', color='purple', linewidth=1.6, markersize=4, markerfacecolor='w', label='HLL subsonic')
plt.plot(x7[indices_x07], p7[indices_x07, -1] / p_stagnation, '--', color='indigo', linewidth=1.6, markersize=4, markerfacecolor='w', label='Rusanov subsonic')
plt.plot(x, p / 7e5, '--', color='black', linewidth=1.6, markersize=4, markerfacecolor='w', label='Rusanov supersonic')
ax_inset.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.2f}'))
ax_inset.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.2f}'))

# Add rectangle and lines to indicate zoomed region
mark_inset(ax1, ax_inset, loc1=1, loc2=2, fc="none", ec="0.5")

# PLot density
plt.figure(figsize=(6, 5))
plt.xlabel(r'Normalized axis position x/L (-)', fontsize=18)
plt.ylabel(r'$\rho/\rho_0$ (-)', fontsize=18)
plt.plot(x01[:min_index], rho01[:min_index, 0] / rho_stagnation, '--', color='b', linewidth=1.6, label='Exact supersonic')
plt.plot(x, rho / rho_stagnation, '--', color='black', linewidth=1.6, markersize=4, markerfacecolor='w', label='Exact subsonic')
# plt.plot(x01[min_index:], rho01[min_index:, 1] / rho_stagnation, '--', color='k', linewidth=1.6)
# plt.plot(x01[min_index:], rho01[min_index:,0] / rho_stagnation, '--', color='k', linewidth=1.6, label='Exact solution')
# plt.plot(x2[indices_x2], p2[indices_x2, -1] / p_stagnation, 's', color='orange', linewidth=0.75, markersize=4, markerfacecolor='w', label='Roe supersonic')
# plt.plot(x3[indices_x3], p3[indices_x3, -1] / p_stagnation, '^', color='green', linewidth=0.75, markersize=4, markerfacecolor='w', label='HLL supersonic')
# plt.plot(x4[indices_x4], p4[indices_x4, -1] / p_stagnation, 's', color='purple', linewidth=0.75, markersize=4, markerfacecolor='w', label='Rusanov supersonic')
# plt.plot(x5[indices_x5], p5[indices_x5, -1] / p_stagnation, '^', color='orange', linewidth=0.75, markersize=4, markerfacecolor='w', label='Roe subsonic')
# plt.plot(x6[indices_x6], p6[indices_x6, -1] / p_stagnation, '*', color='green', linewidth=0.75, markersize=4, markerfacecolor='w', label='HLL subsonic')
# plt.plot(x7[indices_x7], p7[indices_x7, -1] / p_stagnation, 's', color='purple', linewidth=0.75, markersize=4, markerfacecolor='w', label='Rusanov subsonic')
plt.plot(x2[indices_x02], rho2[indices_x02, -1] / rho_stagnation, '-', color='orange', linewidth=1.6, markersize=4, markerfacecolor='w', label='Roe supersonic')
plt.plot(x3[indices_x03], rho3[indices_x03, -1] / rho_stagnation, '-.', color='red', linewidth=1.6, markersize=4, markerfacecolor='w', label='HLL supersonic')
plt.plot(x4[indices_x04], rho4[indices_x04, -1] / rho_stagnation, '--', color='darkred', linewidth=1.6, markersize=4, markerfacecolor='w', label='Rusanov supersonic')
plt.plot(x5[indices_x05], rho5[indices_x05, -1] / rho_stagnation, '-', color='mediumvioletred', linewidth=1.6, markersize=4, markerfacecolor='w', label='Roe subsonic')
plt.plot(x6[indices_x06], rho6[indices_x06, -1] / rho_stagnation, '-.', color='purple', linewidth=1.6, markersize=4, markerfacecolor='w', label='HLL subsonic')
plt.plot(x7[indices_x07], rho7[indices_x07, -1] / rho_stagnation, '--', color='indigo', linewidth=1.6, markersize=4, markerfacecolor='w', label='Rusanov subsonic')

# Add the second y-axis for the nozzle area
plt.grid(False)
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(x01, r01, 'k-', linewidth=1, alpha=0.8)
ax2.set_ylabel(r'Normalized nozzle radius r/L (-)', fontsize=18, labelpad=10)  # Adjust labelpad to move it away from the frame
ax1.set_ylim([0.08, 1.1])
ax2.set_ylim([0, L*3])
ax2.fill_between(x01, 0, r01, color='lightblue', alpha=0.3)

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
    frameon=True,               # Keep the frame around the legend
    fontsize=14,  # Replace 12 with your desired font size
)
plt.tight_layout()

ax_inset = inset_axes(ax1, width=0.7, height=0.7, loc='center right', bbox_to_anchor=(0.98, 0.55), bbox_transform=ax1.transAxes)

# Define zoomed-in range
ax_inset.set_xlim(0.998, 1)  # Specify x-range
ax_inset.set_ylim(0.2053, 0.2095)  # Specify y-range
plt.plot(x01[min_index:], rho01[min_index:, 1] / rho_stagnation, '--', color='blue', linewidth=1.6)
plt.plot(x2[indices_x02], rho2[indices_x02, -1] / rho_stagnation, '-', color='orange', linewidth=1.6, markersize=4, markerfacecolor='w', label='Roe supersonic')
plt.plot(x3[indices_x03], rho3[indices_x03, -1] / rho_stagnation, '-.', color='red', linewidth=1.6, markersize=4, markerfacecolor='w', label='HLL supersonic')
plt.plot(x4[indices_x04], rho4[indices_x04, -1] / rho_stagnation, '--', color='darkred', linewidth=1.6, markersize=4, markerfacecolor='w', label='Rusanov supersonic')
ax_inset.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.3f}'))
ax_inset.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.3f}'))

# Add rectangle and lines to indicate zoomed region
mark_inset(ax1, ax_inset, loc1=3, loc2=4, fc="none", ec="0.5")


# Create inset axes
ax_inset = inset_axes(ax1, width=0.63, height=0.63, loc='upper right', bbox_to_anchor=(0.72, 0.27), bbox_transform=ax1.transAxes)

# Define zoomed-in range
ax_inset.set_xlim(0.64, 0.69)  # Specify x-range
ax_inset.set_ylim(0.768, 0.84)  # Specify y-range
plt.plot(x5[indices_x05], rho5[indices_x05, -1] / rho_stagnation, '-', color='mediumvioletred', linewidth=1.6, markersize=4, markerfacecolor='w', label='Roe subsonic')
plt.plot(x6[indices_x06], rho6[indices_x06, -1] / rho_stagnation, '-.', color='purple', linewidth=1.6, markersize=4, markerfacecolor='w', label='HLL subsonic')
plt.plot(x7[indices_x07], rho7[indices_x07, -1] / rho_stagnation, '--', color='indigo', linewidth=1.6, markersize=4, markerfacecolor='w', label='Rusanov subsonic')
plt.plot(x, rho / rho_stagnation, '--', color='black', linewidth=1.6, markersize=4, markerfacecolor='w', label='Exact subsonic')
ax_inset.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.2f}'))
ax_inset.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.2f}'))

# Add rectangle and lines to indicate zoomed region
mark_inset(ax1, ax_inset, loc1=1, loc2=2, fc="none", ec="0.5")




# PLot Mach
plt.figure(figsize=(6, 5))
plt.xlabel(r'Normalized axis position x/L (-)', fontsize=18)
plt.ylabel(r'Mach number (-)', fontsize=18)
plt.plot(x01[:min_index], Ma_exact[:min_index, 0], '--', color='k', linewidth=1.6)
# plt.plot(x01[min_index:], Ma_exact[min_index:, 1], '--', color='k', linewidth=1.6)
# plt.plot(x01[min_index:], Ma_exact[min_index:,0], '--', color='k', linewidth=1.6, label='Exact solution')
# plt.plot(x2[indices_x2], p2[indices_x2, -1] / p_stagnation, 's', color='orange', linewidth=0.75, markersize=4, markerfacecolor='w', label='Roe supersonic')
# plt.plot(x3[indices_x3], p3[indices_x3, -1] / p_stagnation, '^', color='green', linewidth=0.75, markersize=4, markerfacecolor='w', label='HLL supersonic')
# plt.plot(x4[indices_x4], p4[indices_x4, -1] / p_stagnation, 's', color='purple', linewidth=0.75, markersize=4, markerfacecolor='w', label='Rusanov supersonic')
# plt.plot(x5[indices_x5], p5[indices_x5, -1] / p_stagnation, '^', color='orange', linewidth=0.75, markersize=4, markerfacecolor='w', label='Roe subsonic')
# plt.plot(x6[indices_x6], p6[indices_x6, -1] / p_stagnation, '*', color='green', linewidth=0.75, markersize=4, markerfacecolor='w', label='HLL subsonic')
# plt.plot(x7[indices_x7], p7[indices_x7, -1] / p_stagnation, 's', color='purple', linewidth=0.75, markersize=4, markerfacecolor='w', label='Rusanov subsonic')
plt.plot(x2[indices_x02], Ma_roe_sup[indices_x02, -1], '-', color='orange', linewidth=1.6, markersize=4, markerfacecolor='w', label='Roe supersonic')
plt.plot(x3[indices_x03], Ma_hllc_sup[indices_x03, -1], '-.', color='red', linewidth=1.6, markersize=4, markerfacecolor='w', label='HLL supersonic')
plt.plot(x4[indices_x04], Ma_rus_sup[indices_x04, -1], '--', color='darkred', linewidth=1.6, markersize=4, markerfacecolor='w', label='Rusanov supersonic')
plt.plot(x5, Ma_roe_sub[:, -1], '-', color='mediumvioletred', linewidth=1.6, markersize=4, markerfacecolor='w', label='Roe subsonic')
plt.plot(x6[indices_x06], Ma_hllc_sub[indices_x06, -1], '-.', color='darkviolet', linewidth=1.6, markersize=4, markerfacecolor='w', label='HLL subsonic')
plt.plot(x7[indices_x07], Ma_rus_sub[indices_x07, -1], '--', color='purple', linewidth=1.6, markersize=4, markerfacecolor='w', label='Rusanov subsonic')
plt.plot(x, M, '--', color='black', linewidth=1.6, markersize=4, markerfacecolor='w', label='Subsonic exact')

# Add the second y-axis for the nozzle area
plt.grid(False)
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(x01, r01, 'k-', linewidth=1, alpha=0.8)
ax2.set_ylabel(r'Normalized nozzle radius r/L (-)', fontsize=18, labelpad=10)  # Adjust labelpad to move it away from the frame
ax1.set_ylim([-0.4, 2.4])
ax2.set_ylim([0, L*3])
ax2.fill_between(x01, 0, r01, color='lightblue', alpha=0.3)

# Get handles and labels from both axes
handles1, labels1 = ax1.get_legend_handles_labels()  # From primary axis
handles2, labels2 = ax2.get_legend_handles_labels()  # From secondary axis

# Combine handles and labels
handles = handles1 + handles2
labels = labels1 + labels2

# Create a combined legend with your custom settings
# plt.legend(
#     handles, labels,
#     loc='upper left',
#     bbox_to_anchor=(0.01, 0.98),   # Custom positioning
#     # fontsize='x-small',        # Custom font size
#     handlelength=1,            # Shorten the line markers
#     handletextpad=0.3,         # Reduce space between markers and text
#     borderpad=0.3,             # Decrease padding inside the legend box
#     frameon=True               # Keep the frame around the legend
# )
plt.tight_layout()


ax_inset = inset_axes(ax1, width=0.7, height=0.7, loc='upper right', bbox_to_anchor=(0.6, 0.75), bbox_transform=ax1.transAxes)

# Define zoomed-in range
ax_inset.set_xlim(0.650, 0.685)  # Specify x-range
ax_inset.set_ylim(0.61, 0.745)  # Specify y-range
plt.plot(x01[min_index:], Ma_exact[min_index:,0], '--', color='k', linewidth=1, label='Exact solution')
plt.plot(x5, Ma_roe_sub[:, -1], '-', color='mediumvioletred', linewidth=1.6, markersize=4, markerfacecolor='w', label='Roe subsonic')
plt.plot(x6[indices_x06], Ma_hllc_sub[indices_x06, -1], '-.', color='purple', linewidth=1.6, markersize=4, markerfacecolor='w', label='HLL subsonic')
plt.plot(x7[indices_x07], Ma_rus_sub[indices_x07, -1], '--', color='indigo', linewidth=1.6, markersize=4, markerfacecolor='w', label='Rusanov subsonic')
plt.plot(x, M, '--', color='black', linewidth=1.6, markersize=4, markerfacecolor='w', label='Subsonic exact')
# Add rectangle and lines to indicate zoomed region
mark_inset(ax1, ax_inset, loc1=3, loc2=4, fc="none", ec="0.5")
ax_inset.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.2f}'))
ax_inset.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.2f}'))

# Create inset axes
ax_inset = inset_axes(ax1, width=0.7, height=0.7, loc='center right', bbox_to_anchor=(0.965, 0.55), bbox_transform=ax1.transAxes)

# Define zoomed-in range
ax_inset.set_xlim(0.9960, 0.9985)  # Specify x-range
ax_inset.set_ylim(2.0792, 2.098)  # Specify y-range
plt.plot(x01[:], Ma_exact[:, 1], '--', color='blue', linewidth=1.8, label='Exact solution')
plt.plot(x2[indices_x02], Ma_roe_sup[indices_x02, -1], '-', color='orange', linewidth=1.6, markersize=4, markerfacecolor='w', label='Roe supersonic')
plt.plot(x3[indices_x03], Ma_hllc_sup[indices_x03, -1], '-.', color='red', linewidth=1.6, markersize=4, markerfacecolor='w', label='HLL supersonic')
plt.plot(x4[indices_x04], Ma_rus_sup[indices_x04, -1], '--', color='darkred', linewidth=1.6, markersize=4, markerfacecolor='w', label='Rusanov supersonic')
# Add rectangle and lines to indicate zoomed region
mark_inset(ax1, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")
ax_inset.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.3f}'))
ax_inset.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.2f}'))

plt.show()