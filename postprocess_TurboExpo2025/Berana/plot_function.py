import matplotlib.pyplot as plt
import numpy as np
import nozzle_hyperbolic as hyp
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.patches as patches


hyp.set_plot_options(grid=False)

# Plot function1 for showing shock at position 25 50 75 100
# yaml file: euler_adapted_expansion


# Load the data from the files
# data01 = np.load("Results/rusanov_N230_pin90_tin45_pout59.6_rough3.npz")
data01 = np.load("Results/rusanov_N250_pin90_tin37_pout51.4_rough001.npz")
data02= np.load("Results/rusanov_N240_pin90_tin37_pout48.2_rough001_gronn.npz")
data03 = np.load("Results/rusanov_N240_pin90_tin37_pout42.6_rough001_gronn.npz")
data04 = np.load("Results/rusanov_N230_pin90_tin45_pout59.6_rough3.npz")
data05 = np.load("Results/rusanov_N240_pin90_tin45_pout56.9_rough3.npz")
data06 = np.load("Results/rusanov_N240_pin90_tin45_pout50.1_rough3.npz")

# data04 = np.load("Results/rus_pin90_T45_pout50.npz")

h, x1, u1, p1, rho1, a1 = data01["h"], data01["x"], data01["u"], data01["p"], data01["rho"], data01["a"]
x2, u2, p2, rho2 = data02["x"], data02["u"], data02["p"], data02["rho"]
x3, u3, p3, rho3, a3 = data03["x"], data03["u"], data03["p"], data03["rho"], data03["a"]
x4, u4, p4, rho4, a4 = data04["x"], data04["u"], data04["p"], data04["rho"], data04["a"]
x5, u5, p5, rho5, a5= data05["x"], data05["u"], data05["p"], data05["rho"], data05["a"]
x6, p6, a6, u6= data06["x"], data06["p"], data06["a"], data06["u"]

Ma = u6/a6
print(Ma)
# case1 = 'pin90_T45.xlsx'
# case1_1 = pd.read_excel(case1, sheet_name="pin90_T45_pout59.6")
# case1_2 = pd.read_excel(case1, sheet_name="pin90_T45_pout56.9")
# case1_3 = pd.read_excel(case1, sheet_name="pin90_T45_pout50.1")
# case1_4 = pd.read_excel(case1, sheet_name="pin90_T45_pout41.2")
case1 = 'pin90_T45.xlsx'
case1_1 = pd.read_excel(case1, sheet_name="pin90_T45_pout59.6")
case1_2 = pd.read_excel(case1, sheet_name="pin90_T45_pout56.9")
case1_3 = pd.read_excel(case1, sheet_name="pin90_T45_pout50.1")
# case1_4 = pd.read_excel(case1, sheet_name="pin90_T45_pout41.2")


## Case 1
x1 = x1*1e3-15
x2 = x2*1e3-15
x3 = x3*1e3-15
x4 = x4*1e3-15
x5 = x5*1e3-15
x6 = x6*1e3-15


plt.figure(figsize=(6, 3.83))
ax1 = plt.gca()
ax1.set_ylim([30, 95])
ax1.set_xlim([-15, 8.5])
ax1.set_xlabel(r'Position along nozzle axis (mm)')
ax1.set_ylabel(r'Static pressure (bar)')

indices_x6 = np.linspace(0, len(x6) - 1, 70, dtype=int)
indices_x3 = np.linspace(0, len(x3) - 1, 100, dtype=int)
indices_x2 = np.linspace(0, len(x2) - 1, 100, dtype=int)
indices_x1 = np.linspace(0, len(x1) - 1, 100, dtype=int)

# Plot only the specific indices for p1, p2, and p3
plt.plot(x6[indices_x3], p6[indices_x3]/1e5, '-', color='black', linewidth=2, label=r"Case B1-1")
plt.plot(x5[indices_x2], p5[indices_x2]/1e5, '-', color='mediumvioletred', linewidth=2, markerfacecolor='white', label=r"Case B1-2")
plt.plot(case1_2['position'], case1_2['pressure']*1e1, 's', markersize=5, markerfacecolor='mediumvioletred', markeredgecolor='black', label=r"")
plt.plot(x4, p4/1e5, '-', color='orange', linewidth=2, markerfacecolor='white', label=r"Case B1-3")
plt.plot(case1_1['position'], case1_1['pressure']*1e1, 's', markersize=5, markerfacecolor='orange', markeredgecolor='black', label=r"")
plt.plot(case1_3['position'], case1_3['pressure']*1e1, 's', markersize=5, markerfacecolor='black', markeredgecolor='black', label=r"Berana (2009)")

# plt.plot(case1_3['position'], case1_3['pressure']*1e1, 's', markersize=5, markerfacecolor='black', markeredgecolor='black', label='Berana (2009)')
# plt.plot(case1_2['position'], case1_2['pressure']*1e1, 's', markersize=5, markerfacecolor='mediumvioletred', markeredgecolor='black', label='')
# plt.plot(case1_1['position'], case1_1['pressure']*1e1, 's', markersize=5, markerfacecolor='orange', markeredgecolor='black', label='')

# plt.plot(case1_4['position'], case1_4['pressure'], 's', color='black', markersize=6, markerfacecolor='k', label='')


# Get handles and labels from both axes
handles1, labels1 = ax1.get_legend_handles_labels()  # From primary axis

# Create a combined legend with your custom settings
plt.legend(
    handles1, labels1,
    loc='upper right',
    # bbox_to_anchor=(0.05, 0.5),  # Custom positioning
    # fontsize='x-small',        # Custom font size
    handlelength=1.5,
    handletextpad=0.3,
    borderpad=0.3,
    frameon=True,
    columnspacing=1.5,
    labelspacing=0.4,
)

plt.tight_layout()
# Create inset axes
ax_inset = inset_axes(ax1, width=2, height=2, loc='center left', bbox_to_anchor=(0.13, 0.47), bbox_transform=ax1.transAxes)


# Define zoomed-in range
x_zoom_min, x_zoom_max = 2.7, 7.55  # x-range for zoom
y_zoom_min, y_zoom_max = 38, 60.1  # y-range for zoom

ax_inset.set_xlim(x_zoom_min, x_zoom_max)
ax_inset.set_ylim(y_zoom_min, y_zoom_max)

# Mark the zoomed-in region with a rectangle using the same coordinates
rect = patches.Rectangle((x_zoom_min, y_zoom_min), x_zoom_max - x_zoom_min, y_zoom_max - y_zoom_min, 
                         linewidth=1, edgecolor='black', facecolor='none', linestyle='-')

ax1.add_patch(rect)

plt.plot(x6[indices_x3], p6[indices_x3]/1e5, '-', color='black', linewidth=2, label=r"Case B2-1")
plt.plot(x5[indices_x2], p5[indices_x2]/1e5, '-', color='mediumvioletred', linewidth=2, markerfacecolor='white', label=r"Case B2-2")
plt.plot(case1_2['position'], case1_2['pressure']*1e1, 's', markersize=5, markerfacecolor='mediumvioletred', markeredgecolor='black', label=r"")
plt.plot(x4, p4/1e5, '-', color='orange', linewidth=2, markerfacecolor='white', label=r"Case B2-3")
plt.plot(case1_1['position'], case1_1['pressure']*1e1, 's', markersize=5, markerfacecolor='orange', markeredgecolor='black', label=r"")
plt.plot(case1_3['position'], case1_3['pressure']*1e1, 's', markersize=5, markerfacecolor='black', markeredgecolor='black', label=r"Berana (2009)")

# # Load the data from the files
# data01 = np.load("Results/rusanov_N230_pin90_tin37_pout59.6_rough3.npz")
data01 = np.load("Results/rusanov_N240_pin90_tin37_pout51.4_rough3.npz")
data02= np.load("Results/rusanov_N240_pin90_tin37_pout48.2_rough3.npz")
data03 = np.load("Results/rusanov_N200_pin90_tin37_pout42.6_rough3.npz")
data04 = np.load("Results/rusanov_N250_pin90_tin37_pout51.4_rough001.npz")
data05 = np.load("Results/rusanov_N240_pin90_tin37_pout48.2_rough001_gronn.npz")
data06 = np.load("Results/rusanov_N240_pin90_tin37_pout42.6_rough001_gronn.npz")


# # data04 = np.load("Results/rus_pin90_T45_pout50.npz")

h, x1, u1, p1, rho1, a1 = data01["h"], data01["x"], data01["u"], data01["p"], data01["rho"], data01["a"]
x2, u2, p2, rho2 = data02["x"], data02["u"], data02["p"], data02["rho"]
x3, u3, p3, rho3, a3 = data03["x"], data03["u"], data03["p"], data03["rho"], data03["a"]
x4, u4, p4, rho4, a4 = data04["x"], data04["u"], data04["p"], data04["rho"], data04["a"]
x5, u5, p5, rho5, a5= data05["x"], data05["u"], data05["p"], data05["rho"], data05["a"]
x6, p6 = data06["x"], data06["p"]


case1 = 'pin90_T37.xlsx'
case1_1 = pd.read_excel(case1, sheet_name="pin90_T37_pout51.4")
case1_2 = pd.read_excel(case1, sheet_name="pin90_T37_pout48.2")
case1_3 = pd.read_excel(case1, sheet_name="pin90_T37_pout42.6")
# case1_4 = pd.read_excel(case1, sheet_name="pin90_T45_pout41.2")

# Ma1 = u1/a1

# Ma3 = u3/a3
## Case 1
x1 = x1*1e3-15
x2 = x2*1e3-15
x3 = x3*1e3-15
x4 = x4*1e3-15
x5 = x5*1e3-15
x6 = x6*1e3-15

indices_x6 = np.linspace(0, len(x6) - 1, 70, dtype=int)
indices_x3 = np.linspace(0, len(x3) - 1, 70, dtype=int)
indices_x2 = np.linspace(0, len(x2) - 1, 100, dtype=int)
indices_x1 = np.linspace(0, len(x1) - 1, 95, dtype=int)
indices_x5 = np.linspace(0, len(x5) - 1, 100, dtype=int)
indices_x4 = np.linspace(0, len(x4) - 1, 95, dtype=int)


plt.figure(figsize=(6, 3.5))
ax1 = plt.gca()
# ax1.set_ylim([30, 95])
# ax1.set_xlim([-15, 8.5])
# ax1.set_xlabel(r'Position along nozzle axis (mm)')
ax1.set_ylabel(r'Static pressure (bar)')
plt.plot(x3[indices_x3], p3[indices_x3]/1e5, '-', color='black', linewidth=2, label=r"Case B2-1, $\tau_{\text{w}} = 3 \; \tau_{\text{L}}$")
plt.plot(x6[indices_x6], p6[indices_x6]/1e5, '--', color='black', linewidth=2, label=r"Case B2-1, $\tau_{\text{w}} = \Phi^2_{\text{LM}} \; \tau_{\text{L}}$")

plt.plot(x2[indices_x2], p2[indices_x2]/1e5, '-', color='mediumvioletred', linewidth=2, markerfacecolor='white', label=r"Case B2-2, $\tau_{\text{w}} = 3 \; \tau_{\text{L}}$")
plt.plot(x5[indices_x5], p5[indices_x2]/1e5, '--', color='mediumvioletred', linewidth=2, markerfacecolor='white', label=r"Case B2-2, $\tau_{\text{w}} = \Phi^2_{\text{LM}} \; \tau_{\text{L}}$")

plt.plot(x1[indices_x1], p1[indices_x1]/1e5, '-', color='orange', linewidth=2, markerfacecolor='white', label=r"Case B2-3, $\tau_{\text{w}} = 3 \; \tau_{\text{L}}$")
plt.plot(x4[indices_x4], p4[indices_x4]/1e5, '--', color='orange', linewidth=2, markerfacecolor='white', label=r"Case B2-3, $\tau_{\text{w}} = \Phi^2_{\text{LM}} \; \tau_{\text{L}}$")

# plt.plot(x4, p4/1e5, '--', color='orange', linewidth=2, markerfacecolor='white', label=r"P$_{out}$ = 59.6 bar, k$_s$ = 3 µm")


# plt.plot(case1_1['position'], case1_1['pressure']*1e1, 's', color='black', markersize=4, markerfacecolor='k', label='Nakagawa (2008)')
# plt.plot(case1_2['position'], case1_2['pressure']*1e1, 'o', color='purple', markersize=4, markerfacecolor='purple', label='')
# plt.plot(case1_3['position'], case1_3['pressure']*1e1, '^', color='orange', markersize=4, markerfacecolor='orange', label='')
# plt.plot(case1_4['position'], case1_4['pressure'], 's', color='black', markersize=6, markerfacecolor='k', label='')
ax1.set_ylim([30, 95])
ax1.set_xlim([-15, 8.5])

# Get handles and labels from both axes
handles1, labels1 = ax1.get_legend_handles_labels()  # From primary axis

# Create a combined legend with your custom settings
plt.legend(
    handles1, labels1,
    loc='center left',
    bbox_to_anchor=(0.05, 0.5),  # Custom positioning
    # fontsize='x-small',        # Custom font size
    handlelength=1.5,
    handletextpad=0.3,
    borderpad=0.3,
    frameon=True,
    columnspacing=1.5,
    labelspacing=0.4,
)
plt.tight_layout()

plt.show()
