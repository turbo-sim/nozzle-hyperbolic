# Plot function 2: compare different solvers with shock at two positions¨
# yaml file: flux_solvers_comparison

import matplotlib.pyplot as plt
import numpy as np
import nozzle_hyperbolic as hyp

hyp.set_plot_options(grid=False)

# Plot function1 for showing shock at position 25 50 75 100
# yaml file: euler_adapted_expansion

# Load the data from the files
data01 = np.load("exact1.npz")
# data02 = np.load("exact2.npz")
data1 = np.load("rusanov.npz")
# data2 = np.load("rusanov2.npz")
data3 = np.load("roe.npz")
# data4 = np.load("roe2.npz")
data5 = np.load("HLL.npz")
# data6 = np.load("HLL2.npz")

R = 287
gamma = 1.4
T_stagnation = 500
p_stagnation = 7e5
rho_stagnation = p_stagnation/(R*T_stagnation)

L, r, x, u01, p01, rho01 = data01["L"], data01["r"], data01["x"], data01["u"], data01["p"], data01["rho"]
# u02, p02, rho02 = data02["u"], data02["p"], data02["rho"]
x, u1, p1, rho1, i_shock1 = data1["x"], data1["u1"], data1["p1"], data1["rho1"], data1["i_shock"]
# x, u2, p2, rho2, i_shock2 = data2["x"], data2["u1"], data2["p1"], data2["rho1"], data2["i_shock"]
x, u3, p3, rho3, i_shock2 = data3["x"], data3["u1"], data3["p1"], data3["rho1"], data3["i_shock"]
# x, u4, p4, rho4, i_shock4 = data4["x"], data4["u1"], data4["p1"], data4["rho1"], data4["i_shock"]
x, u5, p5, rho5, i_shock5 = data5["x"], data5["u1"], data5["p1"], data5["rho1"], data5["i_shock"]
# x, u6, p6, rho6, i_shock6 = data6["x"], data6["u1"], data6["p1"], data6["rho1"], data6["i_shock"]

a_sound_rusanov = np.sqrt(gamma*p1/rho1)
Ma_rus = u1/a_sound_rusanov
a_sound_roe = np.sqrt(gamma*p3/rho3)
Ma_roe = u3/a_sound_roe
a_sound_hllc = np.sqrt(gamma*p5/rho5)
Ma_hllc = u5/a_sound_hllc

min_index = np.argmin(r)

plt.figure(figsize=(6, 5))
plt.xlabel(r'$x/L$')
plt.ylabel(r'$P / P_T$')
# plt.plot(x, p01[:, 0]/ 1e5, 'r--', linewidth=1, markersize=4, markerfacecolor='w', label='Exact solution - Subsonic')
# plt.plot(x[min_index:], p01[min_index:, 1]/ 1e5, 'b--', linewidth=0.75, markersize=4, markerfacecolor='w', label='Exact solution - Supersonic')
# plt.plot(x[i_shock1- 1:], p01[i_shock1 - 1:, 2] / 1e5, 'k-', linewidth=1, markersize=2, label='Exact solution - Shock wave')
# plt.plot(x[i_shock2- 1:], p02[i_shock2 - 1:, 2] / 1e5, 'k-', linewidth=1, markersize=2, label='Exact solution - Shock wave')

plt.plot(x[:], p1[:, -1] / p_stagnation, '-', color='purple', linewidth=1.1, markerfacecolor='w')
plt.plot(x[:], p3[:, -1] / p_stagnation, '-', color='orange', linewidth=1.1, markerfacecolor='w')
plt.plot(x[:], p5[:, -1] / p_stagnation, '-', color='green', linewidth=1.1, markerfacecolor='w')
plt.plot(x[1::30], p1[1::30, -1] / p_stagnation, 'o', color='purple', markersize=5, markerfacecolor='w', label='Numerical solution Rusanov')
plt.plot(x[10::30], p3[10::30, -1] / p_stagnation, 'o', color='orange', markersize=5, markerfacecolor='w', label='Numerical solution Roe')
plt.plot(x[20::30], p5[20::30, -1] / p_stagnation, 'o', color='green', markersize=5, markerfacecolor='w', label='Numerical solution HLL')

# plt.plot(x[12::12], p2[12::12, -1] / 1e5, 's', color='purple', linewidth=0.75, markersize=4, markerfacecolor='w', label='SW2 - Rusanov')
# plt.plot(x[16::12], p4[16::12, -1] / 1e5, '^', color='orange', linewidth=0.75, markersize=4, markerfacecolor='w', label='SW2 - Roe')
# plt.plot(x[20::12], p6[20::12, -1] / 1e5, '*', color='green', linewidth=0.75, markersize=4, markerfacecolor='w', label='SW2 - HLL')

# Add the second y-axis for the nozzle area
plt.grid(False)
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(x, r, 'k-', linewidth=1, alpha=0.8, label='Nozzle radius')
ax2.set_ylabel(r'$r / L$', labelpad=10)  # Adjust labelpad to move it away from the frame
ax1.set_ylim([0.05, 1.1])
ax2.set_ylim([0, L*3])
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
    bbox_to_anchor=(0.05, 0.5),  # Custom positioning
    fontsize='x-small',        # Custom font size
    handlelength=1,            # Shorten the line markers
    handletextpad=0.3,         # Reduce space between markers and text
    borderpad=0.3,             # Decrease padding inside the legend box
    frameon=True               # Keep the frame around the legend
)


## plot Mach

plt.figure(figsize=(6, 5))
plt.xlabel(r'$x/L$')
plt.ylabel(r'$M$')
# plt.plot(x, u01[:, 0], 'r--', linewidth=1, markersize=4, markerfacecolor='w', label='Exact solution - Subsonic')
# plt.plot(x[min_index:], u01[min_index:, 1], 'b--', linewidth=1, markersize=4, markerfacecolor='w', label='Exact solution - Supersonic')
# plt.plot(x[i_shock1- 1:], u01[i_shock1 - 1:, 2], 'k-', linewidth=1, markersize=2, label='Exact solution - Shock wave')
# plt.plot(x[i_shock2- 1:], u02[i_shock2 - 1:, 2], 'k-', linewidth=1, markersize=2, label='Exact solution - Shock wave')

plt.plot(x[:], Ma_rus[:, -1], '-', color='purple', linewidth=1.1, markersize=5, markerfacecolor='w')
plt.plot(x[:], Ma_roe[:, -1], '-', color='orange', linewidth=1.1, markersize=5, markerfacecolor='w')
plt.plot(x[:], Ma_hllc[:, -1], '-', color='green', linewidth=1.1, markersize=5, markerfacecolor='w')
plt.plot(x[1::30], Ma_rus[1::30, -1], '^', color='purple', markersize=5, markerfacecolor='w', label='Numerical solution Rusanov')
plt.plot(x[10::30], Ma_roe[10::30, -1], '^', color='orange', markersize=5, markerfacecolor='w', label='Numerical solution Roe')
plt.plot(x[20::30], Ma_hllc[20::30, -1], '^', color='green', markersize=5, markerfacecolor='w', label='Numerical solution HLL')

# plt.plot(x[12::12], u2[12::12, -1], 's', color='purple', linewidth=0.75, markersize=4, markerfacecolor='w', label='SW2 - Rusanov')
# plt.plot(x[16::12], u4[16::12, -1], '^', color='orange', linewidth=0.75, markersize=4, markerfacecolor='w', label='SW2 - Roe')
# plt.plot(x[20::12], u6[20::12, -1], '*', color='green', linewidth=0.75, markersize=4, markerfacecolor='w', label='SW2 - HLL')

# Add the second y-axis for the nozzle area
plt.grid(False)
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(x, r, 'k-', linewidth=1, alpha=0.7, label='Nozzle radius')
ax2.set_ylabel(r'$r / L$', labelpad=10)  # Adjust labelpad to move it away from the frame
ax1.set_ylim([-0.35, 2.1])
ax2.set_ylim([0, L*3])
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
    loc='upper left',
    fontsize='x-small',          # Use 'x-small' for a smaller font size
    bbox_to_anchor=(0.05, 0.95),  # Keep this to position the legend downward
    handlelength=1,            # Shorten the line markers
    handletextpad=0.3,         # Reduce space between markers and text
    borderpad=0.3,             # Decrease padding inside the legend box
    frameon=True               # Keep the frame around the legend
)


## plot density

plt.figure(figsize=(6, 5))
plt.xlabel(r'$x/L$')
plt.ylabel(r'$\rho / \rho_0$')
# plt.plot(x, rho01[:, 0]/rho_stagnation, 'r--', linewidth=1, markersize=4, markerfacecolor='w', label='Exact solution - Subsonic')
# plt.plot(x[min_index:], rho01[min_index:, 1]/rho_stagnation, 'b--', linewidth=1, markersize=4, markerfacecolor='w', label='Exact solution - Supersonic')
# plt.plot(x[i_shock1- 1:], rho01[i_shock1 - 1:, 2]/rho_stagnation, 'k-', linewidth=1, markersize=2, label='Exact solution - Shock wave')
# plt.plot(x[i_shock2- 1:], rho02[i_shock2 - 1:, 2], 'k-', linewidth=1, markersize=2, label='Exact solution - Shock wave')

plt.plot(x[:], rho1[:, -1]/rho_stagnation, '-', color='purple', linewidth=1.1)
plt.plot(x[:], rho3[:, -1]/rho_stagnation, '-', color='orange', linewidth=1.1)
plt.plot(x[:], rho5[:, -1]/rho_stagnation, '-', color='green', linewidth=1.1)
plt.plot(x[1::30], rho1[1::30, -1]/rho_stagnation, '*', color='purple', markersize=5, markerfacecolor='w', label='Numerical solution Rusanov')
plt.plot(x[10::30], rho3[10::30, -1]/rho_stagnation, '*', color='orange', markersize=5, markerfacecolor='w', label='Numerical solution Roe')
plt.plot(x[20::30], rho5[20::30, -1]/rho_stagnation, '*', color='green', markersize=5, markerfacecolor='w', label='Numerical solution HLL')

# plt.plot(x[12::12], rho2[12::12, -1], 's', color='purple', linewidth=0.75, markersize=4, markerfacecolor='w', label='SW2 - Rusanov')
# plt.plot(x[16::12], rho4[16::12, -1], '^', color='orange', linewidth=0.75, markersize=4, markerfacecolor='w', label='SW2 - Roe')
# plt.plot(x[20::12], rho6[20::12, -1], '*', color='green', linewidth=0.75, markersize=4, markerfacecolor='w', label='SW2 - HLL')

# Add the second y-axis for the nozzle area
plt.grid(False)
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(x, r, 'k-', linewidth=1, alpha=0.7, label='Nozzle radius')
ax2.set_ylabel(r'$r / L$', labelpad=10)  # Adjust labelpad to move it away from the frame
ax1.set_ylim([0.1, 1.1])
ax2.set_ylim([0, L*3])
ax2.fill_between(x, 0, r, color='lightblue', alpha=0.3)
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
    bbox_to_anchor=(0.05, 0.5),  # Custom positioning
    fontsize='x-small',        # Custom font size
    handlelength=1,            # Shorten the line markers
    handletextpad=0.3,         # Reduce space between markers and text
    borderpad=0.3,             # Decrease padding inside the legend box
    frameon=True               # Keep the frame around the legend
)


plt.show()