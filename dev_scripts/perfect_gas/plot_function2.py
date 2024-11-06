# Plot function 2: compare different solvers with shock at two positions¨
# yaml file: flux_solvers_comparison

import matplotlib.pyplot as plt
import numpy as np
from functions import set_plot_options

# Plot function1 for showing shock at position 25 50 75 100
# yaml file: euler_adapted_expansion

set_plot_options(grid=False)

# Load the data from the files
data01 = np.load("exact1.npz")
data02 = np.load("exact2.npz")
data1 = np.load("rusanov1.npz")
data2 = np.load("rusanov2.npz")
data3 = np.load("roe1.npz")
data4 = np.load("roe2.npz")
data5 = np.load("HLLC1.npz")
data6 = np.load("HLLC2.npz")


L, r, x, u01, p01, rho01 = data01["L"], data01["r"], data01["x"], data01["u"], data01["p"], data01["rho"]
u02, p02, rho02 = data02["u"], data02["p"], data02["rho"]
x, u1, p1, rho1, i_shock1 = data1["x"], data1["u1"], data1["p1"], data1["rho1"], data1["i_shock"]
x, u2, p2, rho2, i_shock2 = data2["x"], data2["u1"], data2["p1"], data2["rho1"], data2["i_shock"]
x, u3, p3, rho3, i_shock2 = data3["x"], data3["u1"], data3["p1"], data3["rho1"], data3["i_shock"]
x, u4, p4, rho4, i_shock4 = data4["x"], data4["u1"], data4["p1"], data4["rho1"], data4["i_shock"]
x, u5, p5, rho5, i_shock5 = data5["x"], data5["u1"], data5["p1"], data5["rho1"], data5["i_shock"]
x, u6, p6, rho6, i_shock6 = data6["x"], data6["u1"], data6["p1"], data6["rho1"], data6["i_shock"]


plt.figure(figsize=(6, 5))
plt.xlabel('Axis position [m]')
plt.ylabel('Pressure [bar]')
plt.plot(x, p01[:, 0]/ 1e5, 'r--', linewidth=1, markersize=4, markerfacecolor='w', label='Exact solution - Subsonic')
plt.plot(x[len(p01)//2:], p01[len(p01)//2:, 1]/ 1e5, 'b--', linewidth=0.75, markersize=4, markerfacecolor='w', label='Exact solution - Supersonic')
plt.plot(x[i_shock1- 1:], p01[i_shock1 - 1:, 2] / 1e5, 'k-', linewidth=1, markersize=2, label='Exact solution - Shock wave')
plt.plot(x[i_shock2- 1:], p02[i_shock2 - 1:, 2] / 1e5, 'k-', linewidth=1, markersize=2, label='Exact solution - Shock wave')

plt.plot(x[1::12], p1[1::12, -1] / 1e5, 'o', color='purple', linewidth=0.75, markersize=4, markerfacecolor='w', label='SW1 - Rusanov')
plt.plot(x[4::12], p3[4::12, -1] / 1e5, 's', color='orange', linewidth=0.75, markersize=4, markerfacecolor='w', label='SW1 - Roe')
plt.plot(x[8::12], p5[8::12, -1] / 1e5, '^', color='green', linewidth=0.75, markersize=4, markerfacecolor='w', label='SW1 - HLL')

plt.plot(x[12::12], p2[12::12, -1] / 1e5, 's', color='purple', linewidth=0.75, markersize=4, markerfacecolor='w', label='SW2 - Rusanov')
plt.plot(x[16::12], p4[16::12, -1] / 1e5, '^', color='orange', linewidth=0.75, markersize=4, markerfacecolor='w', label='SW2 - Roe')
plt.plot(x[20::12], p6[20::12, -1] / 1e5, '*', color='green', linewidth=0.75, markersize=4, markerfacecolor='w', label='SW2 - HLL')

# Add the second y-axis for the nozzle area
plt.grid(True)
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(x, r, 'k--', linewidth=1.1, label='Nozzle radius')
ax2.set_ylabel('Nozzle Radius [m]')
ax2.set_ylim([0, L])
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
    bbox_to_anchor=(0, 0.53),  # Custom positioning
    fontsize='x-small',        # Custom font size
    handlelength=1,            # Shorten the line markers
    handletextpad=0.3,         # Reduce space between markers and text
    borderpad=0.3,             # Decrease padding inside the legend box
    frameon=True               # Keep the frame around the legend
)


## plot velocity

plt.figure(figsize=(6, 5))
plt.xlabel('Axis position [m]')
plt.ylabel('Velocity [m/s]')
plt.plot(x, u01[:, 0], 'r--', linewidth=1, markersize=4, markerfacecolor='w', label='Exact solution - Subsonic')
plt.plot(x[len(u01)//2:], u01[len(u01)//2:, 1], 'b--', linewidth=1, markersize=4, markerfacecolor='w', label='Exact solution - Supersonic')
plt.plot(x[i_shock1- 1:], u01[i_shock1 - 1:, 2], 'k-', linewidth=1, markersize=2, label='Exact solution - Shock wave')
plt.plot(x[i_shock2- 1:], u02[i_shock2 - 1:, 2], 'k-', linewidth=1, markersize=2, label='Exact solution - Shock wave')

plt.plot(x[1::12], u1[1::12, -1], 'o', color='purple', linewidth=0.75, markersize=4, markerfacecolor='w', label='SW1 - Rusanov')
plt.plot(x[4::12], u3[4::12, -1], 's', color='orange', linewidth=0.75, markersize=4, markerfacecolor='w', label='SW1 - Roe')
plt.plot(x[8::12], u5[8::12, -1], '^', color='green', linewidth=0.75, markersize=4, markerfacecolor='w', label='SW1 - HLL')

plt.plot(x[12::12], u2[12::12, -1], 's', color='purple', linewidth=0.75, markersize=4, markerfacecolor='w', label='SW2 - Rusanov')
plt.plot(x[16::12], u4[16::12, -1], '^', color='orange', linewidth=0.75, markersize=4, markerfacecolor='w', label='SW2 - Roe')
plt.plot(x[20::12], u6[20::12, -1], '*', color='green', linewidth=0.75, markersize=4, markerfacecolor='w', label='SW2 - HLL')

# Add the second y-axis for the nozzle area
plt.grid(True)
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(x, r, 'k--', linewidth=1.1, label='Nozzle radius')
ax2.set_ylabel('Nozzle Radius [m]')
ax2.set_ylim([0, L])
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
    bbox_to_anchor=(0.01, 0.99),  # Keep this to position the legend downward
    handlelength=1,            # Shorten the line markers
    handletextpad=0.3,         # Reduce space between markers and text
    borderpad=0.3,             # Decrease padding inside the legend box
    frameon=True               # Keep the frame around the legend
)


## plot density

plt.figure(figsize=(6, 5))
plt.xlabel('Axis position [m]')
plt.ylabel('Density [kg/m$^3$]')
plt.plot(x, rho01[:, 0], 'r--', linewidth=1, markersize=4, markerfacecolor='w', label='Exact solution - Subsonic')
plt.plot(x[len(rho01)//2:], rho01[len(rho01)//2:, 1], 'b--', linewidth=1, markersize=4, markerfacecolor='w', label='Exact solution - Supersonic')
plt.plot(x[i_shock1- 1:], rho01[i_shock1 - 1:, 2], 'k-', linewidth=1, markersize=2, label='Exact solution - Shock wave')
plt.plot(x[i_shock2- 1:], rho02[i_shock2 - 1:, 2], 'k-', linewidth=1, markersize=2, label='Exact solution - Shock wave')

plt.plot(x[1::12], rho1[1::12, -1], 'o', color='purple', linewidth=0.75, markersize=4, markerfacecolor='w', label='SW1 - Rusanov')
plt.plot(x[4::12], rho3[4::12, -1], 's', color='orange', linewidth=0.75, markersize=4, markerfacecolor='w', label='SW1 - Roe')
plt.plot(x[8::12], rho5[8::12, -1], '^', color='green', linewidth=0.75, markersize=4, markerfacecolor='w', label='SW1 - HLL')

plt.plot(x[12::12], rho2[12::12, -1], 's', color='purple', linewidth=0.75, markersize=4, markerfacecolor='w', label='SW2 - Rusanov')
plt.plot(x[16::12], rho4[16::12, -1], '^', color='orange', linewidth=0.75, markersize=4, markerfacecolor='w', label='SW2 - Roe')
plt.plot(x[20::12], rho6[20::12, -1], '*', color='green', linewidth=0.75, markersize=4, markerfacecolor='w', label='SW2 - HLL')

# Add the second y-axis for the nozzle area
plt.grid(True)
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(x, r, 'k--', linewidth=1.1, label='Nozzle radius')
ax2.set_ylabel('Nozzle Radius [m]')
ax2.set_ylim([0, L])
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
    bbox_to_anchor=(0.01, 0.48),  # Custom positioning
    fontsize='x-small',        # Custom font size
    handlelength=1,            # Shorten the line markers
    handletextpad=0.3,         # Reduce space between markers and text
    borderpad=0.3,             # Decrease padding inside the legend box
    frameon=True               # Keep the frame around the legend
)


plt.show()