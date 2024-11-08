import matplotlib.pyplot as plt
import numpy as np
import nozzle_hyperbolic as hyp

hyp.set_plot_options(grid=False)

# Plot function1 for showing shock at position 25 50 75 100
# yaml file: euler_adapted_expansion


# Load the data from the files
data01 = np.load("exact25.npz")
data02 = np.load("exact50.npz")
data03 = np.load("exact75.npz")
# data04 = np.load("exact100.npz")
data1 = np.load("roe25.npz")
data2 = np.load("roe50.npz")
data3 = np.load("roe75.npz")
# data4 = np.load("simulation100.npz")

L, r, x, u025, p025, rho025 = data01["L"], data01["r"], data01["x"], data01["u"], data01["p"], data01["rho"]
u050, p050, rho050 = data02["u"], data02["p"], data02["rho"]
u075, p075, rho075 = data03["u"], data03["p"], data03["rho"]
# u0100, p0100, rho0100 = data04["u"], data04["p"], data04["rho"]
x, u25, p25, rho25, i_shock25 = data1["x"], data1["u1"], data1["p1"], data1["rho1"], data1["i_shock"]
x, u50, p50, rho50, i_shock50 = data2["x"], data2["u1"], data2["p1"], data2["rho1"], data2["i_shock"]
x, u75, p75, rho75, i_shock75 = data3["x"], data3["u1"], data3["p1"], data3["rho1"], data3["i_shock"]
# x, u100, p100, rho100, i_shock100 = data4["x"], data4["u1"], data4["p1"], data4["rho1"], data4["i_shock"]

A_critical = np.pi * r[len(x) // 2]**2
# mass_flow_roe25 = rho25[len(x)//2, -1]*u25[len(x)//2, -1]*A_critical
# mass_flow_roe50 = rho50[len(x)//2, -1]*u50[len(x)//2, -1]*A_critical
# mass_flow_roe75 = rho75[len(x)//2, -1]*u75[len(x)//2, -1]*A_critical
# mass_flow_roe100 = rho100[len(x)//2, -1]*u100[len(x)//2, -1]*A_critical

mass_flow_exact = rho025[len(x)//2, -1]*u025[len(x)//2, -1]*A_critical
R = 287
gamma = 1.4
cv = R / (gamma - 1)
entropy_generation = cv * (np.log(p25[-1, -1] / rho25[-1, -1]**gamma) - np.log(p25[1, -1] / rho25[1, -1]**gamma))
entropy_generation_exact = cv * (np.log(p075[-1, -1] / rho075[-1, -1]**gamma) - np.log(p075[1, -1] / rho075[1, -1]**gamma))

print(entropy_generation)
print(entropy_generation_exact)

R = 287
gamma = 1.4
T_stagnation = 500
p_stagnation = 7e5
rho_stagnation = p_stagnation/(R*T_stagnation)

a_sound25 = np.sqrt(gamma*p25/rho25)
a_sound025 = np.sqrt(gamma*p025/rho025)
Ma_25 = u25/a_sound25
Ma_025 = u025/a_sound025
a_sound50 = np.sqrt(gamma*p50/rho50)
a_sound050 = np.sqrt(gamma*p050/rho050)
Ma_50 = u50/a_sound50
Ma_050 = u050/a_sound050
a_sound75 = np.sqrt(gamma*p75/rho75)
a_sound075 = np.sqrt(gamma*p075/rho075)
Ma_75 = u75/a_sound75
Ma_075 = u075/a_sound075


min_index = np.argmin(r)

plt.figure(figsize=(6, 5))
plt.xlabel(r'$x/L$')
plt.ylabel(r'$M$')
plt.plot(x[:], Ma_25[:, -1], '-', color='purple', linewidth=1)
plt.plot(x[i_shock25-1:], Ma_025[i_shock25-1:, 2], 'k-', linewidth=1) 

plt.plot(x[:], Ma_50[:, -1], '-', color='orange', linewidth=1)
plt.plot(x[i_shock50-1:], Ma_050[i_shock50-1:, 2], 'k-', linewidth=1) 

plt.plot(x[:], Ma_75[:, -1], '-', color='green', linewidth=1)
plt.plot(x[i_shock75-1:], Ma_075[i_shock75-1:, 2], 'k-', linewidth=1, markersize=2, markerfacecolor='w', label='SW1, SW2, SW3 - Exact solution') 

# plt.plot(x[9::12], u100[9::12, -1], '*', color='darkcyan', linewidth=0.75, markersize=5, markerfacecolor='w', label='Numerical Solution 100% ')
# plt.plot(x[i_shock100-1:], u0100[i_shock100-1:, 2], '-',  color='darkcyan', linewidth=1, markersize=2, markerfacecolor='w', label='Exact solution - Shock 100%') 
# plt.plot(x, Ma_025[:, 0], 'r--', linewidth=1, markersize=4, markerfacecolor='w', label='Exact solution - Subsonic')
plt.plot(x[min_index:], Ma_025[min_index:, 1], 'b--', linewidth=1, markersize=4, markerfacecolor='w', label='Adapted supersonic exact solution')
plt.plot(x[1::12], Ma_25[1::12, -1], 'o', color='purple', markersize=4, markerfacecolor='w', label='SW1 - Numerical Solution')
plt.plot(x[4::12], Ma_50[4::12, -1], 'o', color='orange', markersize=4, markerfacecolor='w', label='SW2 - Numerical Solution')
plt.plot(x[8::12], Ma_75[8::12, -1], 'o', color='green', markersize=4, markerfacecolor='w', label='SW3 - Numerical Solution ')

# Add the second y-axis for the nozzle area
plt.grid(False)
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(x, r, 'k-', linewidth=1, alpha=0.8, label='Nozzle radius')
ax2.set_ylabel(r'$r / L$', labelpad=10)  # Adjust labelpad to move it away from the frame
ax1.set_ylim([-0.35, 2.25])
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
    bbox_to_anchor=(0.01, 0.99),  # Keep this to position the legend downward
    handlelength=1,            # Shorten the line markers
    handletextpad=0.3,         # Reduce space between markers and text
    borderpad=0.3,             # Decrease padding inside the legend box
    frameon=True               # Keep the frame around the legend
)


# Plot the Density Profile
plt.figure(figsize=(6, 5))
plt.xlabel(r'$x/L$')
plt.ylabel(r'$\rho / \rho_0$')
plt.plot(x[:], rho25[:, -1]/rho_stagnation, '-', color='purple', linewidth=1)
plt.plot(x[i_shock25-1:], rho025[i_shock25-1:, 2]/rho_stagnation, 'k-', linewidth=1, markersize=2, markerfacecolor='w')

plt.plot(x[:], rho50[:, -1]/rho_stagnation, '-', color='orange', linewidth=1)
plt.plot(x[i_shock50-1:], rho050[i_shock50-1:, 2]/rho_stagnation, 'k-', linewidth=1, markersize=2, markerfacecolor='w') 

plt.plot(x[:], rho75[:, -1]/rho_stagnation, '-', color='green', linewidth=1)
plt.plot(x[i_shock75-1:], rho075[i_shock75-1:, 2]/rho_stagnation, 'k-', linewidth=1, markersize=2, markerfacecolor='w', label='SW1, SW2, SW3 - Exact solution')
# plt.plot(x[9::12], rho100[9::12, -1], '*', color='darkcyan', linewidth=0.75, markersize=5, markerfacecolor='w', label='Numerical Solution 100%')
# plt.plot(x[i_shock100-1:], rho0100[i_shock100-1:, 2], '-',  color='darkcyan', linewidth=1, markersize=2, markerfacecolor='w', label='Exact solution - Shock 100%')  
# plt.plot(x, rho025[:, 0]/rho_stagnation, 'r--', linewidth=1, markersize=4, markerfacecolor='w', label='Exact solution - Subsonic')
plt.plot(x[min_index:], rho025[min_index:, 1]/rho_stagnation, 'b--', linewidth=1, markersize=4, markerfacecolor='w', label='Supersonic Adapted Exact solution ')
plt.plot(x[1::12], rho25[1::12, -1]/rho_stagnation, '^', color='purple', markersize=5, markerfacecolor='w', label='SW1 - Numerical Solution')
plt.plot(x[4::12], rho50[4::12, -1]/rho_stagnation, '^', color='orange', markersize=5, markerfacecolor='w', label='SW2 - Numerical Solution')
plt.plot(x[8::12], rho75[8::12, -1]/rho_stagnation, '^', color='green', markersize=5, markerfacecolor='w', label='SW3 - Numerical Solution')

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
    bbox_to_anchor=(0.01, 0.48),  # Custom positioning
    fontsize='x-small',        # Custom font size
    handlelength=1,            # Shorten the line markers
    handletextpad=0.3,         # Reduce space between markers and text
    borderpad=0.3,             # Decrease padding inside the legend box
    frameon=True               # Keep the frame around the legend
)



plt.figure(figsize=(6, 5))
plt.xlabel(r'$x/L$')
plt.ylabel(r'$P / P_T$')
plt.plot(x[:], p75[:, -1]/ p_stagnation, '-', color='green', linewidth=1, markerfacecolor='white')
plt.plot(x[i_shock75-1:], p075[i_shock75-1:, 2]/ p_stagnation, 'k-', linewidth=1, label='SW1, SW2, SW3 - Exact solution') 
plt.plot(x[:], p50[:, -1]/ p_stagnation, '-', color='orange', linewidth=1)
plt.plot(x[i_shock50-1:], p050[i_shock50-1:, 2]/ p_stagnation, 'k-', linewidth=1) 
plt.plot(x[:], p25[:, -1]/ p_stagnation, '-', color='purple', linewidth=1)
plt.plot(x[i_shock25-1:], p025[i_shock25-1:, 2]/ p_stagnation, 'k-', linewidth=1)
 
# plt.plot(x[9::12], p100[9::12, -1]/ 1e5, '*', color='darkcyan', linewidth=0.75, markersize=5, markerfacecolor='w', label='Numerical Solution 100%')
# plt.plot(x[i_shock100-1:], p0100[i_shock100-1:, 2]/ 1e5, '-',  color='darkcyan', linewidth=1, markersize=2, markerfacecolor='w', label='Exact solution - Shock 100%') 
# plt.plot(x, p025[:, 0]/ p_stagnation, 'r--', linewidth=1, markersize=4, markerfacecolor='w', label='Exact solution - Subsonic')
plt.plot(x[min_index:], p025[min_index:, 1]/ p_stagnation, 'b--', linewidth=1, markersize=4, markerfacecolor='w', label='Exact solution - Supersonic')
plt.plot(x[1::12], p25[1::12, -1]/ p_stagnation, 's', color='purple', markersize=4, markerfacecolor='white', label='SW1 - Numerical Solution')
plt.plot(x[4::12], p50[4::12, -1]/ p_stagnation, 's', color='orange', markersize=4, markerfacecolor='white', label='SW2 - Numerical Solution')
plt.plot(x[8::12], p75[8::12, -1]/ p_stagnation, 's', color='green', markersize=4, markerfacecolor='white', label='SW3 - Numerical Solution')

# Add the second y-axis for the nozzle area
plt.grid(False)
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(x, r, 'k-', linewidth=1, alpha=0.8, label='Nozzle radius')
ax2.set_ylabel(r'$r / L$', labelpad=10)  # Adjust labelpad to move it away from the frame
ax1.set_ylim([0.01, 1.1])
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
    bbox_to_anchor=(0.01, 0.48),  # Custom positioning
    fontsize='x-small',        # Custom font size
    handlelength=1,            # Shorten the line markers
    handletextpad=0.3,         # Reduce space between markers and text
    borderpad=0.3,             # Decrease padding inside the legend box
    frameon=True               # Keep the frame around the legend
)

plt.show()
