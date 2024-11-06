# Plot function 2: compare different solvers with shock at two positions¨
# yaml file: Supersonic_comparison

import matplotlib.pyplot as plt
import numpy as np
from functions import set_plot_options


set_plot_options(grid=False)

# Load the data from the files
data01 = np.load("exact_supersonicd1.npz")
data02 = np.load("exact_supersonicd2.npz")
data03 = np.load("exact_supersonicd3.npz")
data1 = np.load("rusanov_supersonicd1.npz")
data2 = np.load("roe_supersonicd1.npz")
data3 = np.load("hll_supersonicd1.npz")
data4 = np.load("rusanov_supersonicd2.npz")
data5 = np.load("roe_supersonicd2.npz")
data6 = np.load("hll_supersonicd2.npz")
data7 = np.load("rusanov_supersonicd3.npz")
data8 = np.load("roe_supersonicd3.npz")
data9 = np.load("hll_supersonicd3.npz")

L, r01, x01, u01, p01, rho01 = data01["L"], data01["r"], data01["x"], data01["u"], data01["p"], data01["rho"]
r02, x02, u02, p02, rho02 = data02["r"], data02["x"], data02["u"], data02["p"], data02["rho"]
r03, x03, u03, p03, rho03 = data03["r"], data03["x"], data03["u"], data03["p"], data03["rho"]

x, u1, p1, rho1 = data1["x"], data1["u1"], data1["p1"], data1["rho1"]
x, u2, p2, rho2 = data2["x"], data2["u1"], data2["p1"], data2["rho1"]
x, u3, p3, rho3= data3["x"], data3["u1"], data3["p1"], data3["rho1"]
x, u4, p4, rho4 = data4["x"], data4["u1"], data4["p1"], data4["rho1"]
x, u5, p5, rho5 = data5["x"], data5["u1"], data5["p1"], data5["rho1"]
x, u6, p6, rho6 = data6["x"], data6["u1"], data6["p1"], data6["rho1"]
x, u7, p7, rho7 = data7["x"], data7["u1"], data7["p1"], data7["rho1"]
x, u8, p8, rho8 = data8["x"], data8["u1"], data8["p1"], data8["rho1"]
x, u9, p9, rho9 = data9["x"], data9["u1"], data9["p1"], data9["rho1"]

plt.figure(figsize=(6, 5))
plt.xlabel('Axis position [m]')
plt.ylabel('Pressure [bar]')
plt.plot(x[:len(p01) // 2], p01[:len(p01) // 2, 0] / 1e5, color='green', linewidth=1)
plt.plot(x[len(x) // 2:], p01[len(p01) // 2:, 1] / 1e5, color='green', linewidth=1, label='Exact solution - Supersonic')
plt.plot(x[1::12], p1[1::12, -1] / 1e5, 'o', color='purple', linewidth=0.75, markersize=4, markerfacecolor='w', label='Rusanov')
plt.plot(x[6::12], p2[6::12, -1] / 1e5, 's', color='orange', linewidth=0.75, markersize=4, markerfacecolor='w', label='Roe')
plt.plot(x[9::12], p3[9::12, -1] / 1e5, '^', color='green', linewidth=0.75, markersize=4, markerfacecolor='w', label='HLL')

plt.plot(x[:len(p02) // 2], p02[:len(p02) // 2, 0] / 1e5, 'b', linewidth=1)
plt.plot(x[len(x) // 2:], p02[len(p02) // 2:, 1] / 1e5,'b', linewidth=1, label='Exact solution - Supersonic')
plt.plot(x[1::12], p4[1::12, -1] / 1e5, 's', color='purple', linewidth=0.75, markersize=4, markerfacecolor='w', label='Rusanov')
plt.plot(x[6::12], p5[6::12, -1] / 1e5, '^', color='orange', linewidth=0.75, markersize=4, markerfacecolor='w', label='Roe')
plt.plot(x[9::12], p6[9::12, -1] / 1e5, '*', color='green', linewidth=0.75, markersize=4, markerfacecolor='w', label='HLL')

plt.plot(x[:len(p03) // 2], p03[:len(p03) // 2, 0] / 1e5, 'k', linewidth=1)
plt.plot(x[len(x) // 2:], p03[len(p03) // 2:, 1] / 1e5,'k', linewidth=1, label='Exact solution - Supersonic')
plt.plot(x[1::12], p7[1::12, -1] / 1e5, 's', color='purple', linewidth=0.75, markersize=4, markerfacecolor='w', label='Rusanov')
plt.plot(x[6::12], p8[6::12, -1] / 1e5, '^', color='orange', linewidth=0.75, markersize=4, markerfacecolor='w', label='Roe')
plt.plot(x[9::12], p9[9::12, -1] / 1e5, '*', color='green', linewidth=0.75, markersize=4, markerfacecolor='w', label='HLL')

# Add the second y-axis for the nozzle area
plt.grid(True)
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(x, r01, '--', color='green', linewidth=1.1, label='Nozzle 1')
ax2.plot(x, r02, 'b--', linewidth=1.1, label='Nozzle 2')
ax2.plot(x, r03, 'k--', linewidth=1.1, label='Nozzle 3')
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
    loc='upper right',
    bbox_to_anchor=(1.01, 1.01),  # Custom positioning
    fontsize='x-small',        # Custom font size
    handlelength=1,            # Shorten the line markers
    handletextpad=0.3,         # Reduce space between markers and text
    borderpad=0.3,             # Decrease padding inside the legend box
    frameon=True               # Keep the frame around the legend
)


plt.show()