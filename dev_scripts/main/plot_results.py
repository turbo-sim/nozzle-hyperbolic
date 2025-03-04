# %%
import numpy as np
import matplotlib.pyplot as plt
import barotropy as bpy
import pandas as pd 
 
# fluid = bpy.fluid_properties.Fluid("CO2")
 
# # File names
# files = [
#     "results/rusanov_N100_B13_CFL09_fHEM_purdue.npz",
#     "results/rusanov_N200_B13_CFL09_fHEM_purdue.npz",
#     "results/rusanov_N400_B13_CFL09_fHEM_purdue.npz",
#     "results/rusanov_N700_B13_CFL09_fHEM_purdue.npz",
#     "results/rusanov_N800_B13_CFL09_fHEM_purdue.npz",
#     "results/rusanov_N1400_B13_CFL09_fHEM_purdue.npz",
#     "results/rusanov_N1600_B13_CFL09_fHEM_purdue.npz",

# ]

 
# # Dictionary to store data
# data = {}
 
# # Read the files and extract required variables
# for file in files:
#     try:
#         npzfile = np.load(file)
#         h = npzfile['h']
#         x = npzfile['x']
#         M = npzfile['M']
#         p = npzfile['p']
#         rho = npzfile['rho']
       
#         data[file] = {'x': x, 'M': M, 'p': p, 'rho': rho}
#         state = fluid.get_state(bpy.fluid_properties.DmassP_INPUTS, rho[-1], p[-1])
#         c = state.speed_sound
 
#         print(f"Minimum p for {file}: {min(p)}")
#         print(f"Flow rate for {file}: {h[-1]*2*1e-3*rho[-1]*M[-1]*c}")
 
#     except Exception as e:
#         print(f"Error reading {file}: {e}")
 
# # Plotting
# fig, axes = plt.subplots(3, 1, figsize=(8, 12))
# labels = ["p", "M", "rho"]
# colors = ['b', 'r', 'g']
 
# for i, key in enumerate(["p", "M", "rho"]):
#     for file, values in data.items():
#         axes[i].plot(values['x'], values[key], label=f"Rusanov {len(values['x'])}")
#     axes[i].set_xlabel("x")
#     axes[i].set_ylabel(labels[i])
#     axes[i].legend()
#     axes[i].grid()
 
# plt.tight_layout()
# plt.show()


# fevhrdg

# File names
files = [
    "results/rusanov_N400_B13_CFL09_LM_1_5.npz",
    "results/rusanov_N400_B13_CFL09_LM_3.npz",
    "results/rusanov_N400_B13_CFL09_LM_4.npz",
    "results/rusanov_N400_B13_CFL09_LM_6.npz",
    # "rusanov_N1400_pout_5690000.0_Grunn_4.npz",
    # "rusanov_N1400_pout_5960000.0_con_2.npz"
]


mesh_size = [400, 400, 400, 400,]

x1 = -0.015
x2 = 0.00838
Nx = 400 
x = np.linspace(x1, x2, Nx)
 
# Dictionary to store data
# data = {}
# i = 0
# P_e = []

# LM_val = [1.5, 3, 4, 6]
# fig, axes = plt.subplots(1, 1, figsize=(8, 6))

# for file in files:
#     npzfile = np.load(file)
#     p = npzfile['p']

#     # Nx = mesh_size[i]  # Number of cells
#     # x = np.linspace(x1, x2, Nx)

#     plt.plot(x, p,
#              label = f"LM {LM_val[i]}",
#             #  label= f'P$_e$ {P_e}',
#     )

#     i = i+1

# # Read the files and extract required variables
# for file in files:
#     try:
#         npzfile = np.load(file)
#         p = npzfile['p']
       
#         data[file] = {'p': p}
 
#         print(f"Minimum p for {file}: {min(p)}")
#         print(f"Flow rate for {file}: {h[-1]*2*1e-3*rho[-1]*u[-1]}")
 
#     except Exception as e:
#         print(f"Error reading {file}: {e}")

# # Plotting
# fig, axes = plt.subplots(1, 1, figsize=(8, 12))
# labels = ["p", "rho"]
# colors = ['b', 'r', 'g']

# LM_val = [1.5, 3, 4, 6]

# for i, key in enumerate(["p"]):
#     for file, values in data.items():
#         ## Define numerical setting
#         Nx = mesh_size[i]  # Number of cells
#         x = np.linspace(x1, x2, Nx)
#         axes[i].plot(x, values[key])
#     axes[i].set_xlabel("x")
#     axes[i].set_ylabel(labels[i])
#     axes[i].legend()
#     axes[i].grid()
 
# plt.tight_layout()
# plt.legend()
# plt.show()


# fig, axs = plt.subplots(1, 1, figsize=(7, 4))
plt.plot(figsize=(7, 4))
# axs[0].title("P exit, 59.6")

# plt.xlim([-0.015, 0.00838])

case1 = 'C:\\Users\\alber\\OneDrive - Danmarks Tekniske Universitet\\TurboExpo\\Berana\\pin90_T45.xlsx'

case1_1 = pd.read_excel(case1, sheet_name="pin90_T45_pout59.6")
case1_2 = pd.read_excel(case1, sheet_name="pin90_T45_pout56.9")
case1_3 = pd.read_excel(case1, sheet_name="pin90_T45_pout50.1")
plt.plot(case1_1['position']*1e-3, case1_1['pressure']*1e6, 's', markersize=5, markerfacecolor='orange', markeredgecolor='black', label=r"Experimental")

casep = 'petruccelli_berana_3D.xlsx'
case_p = pd.read_excel(casep)
plt.plot(case_p['position'], case_p['p_ratio']*1e6*9, '-', color='blue',  label=r"Petruccelli 3D")

casep = 'petruccelli_berana_2D.xlsx'
case_p = pd.read_excel(casep)
plt.plot(case_p['position'], case_p['p_ratio']*1e6*9, '-', color='black',  label=r"Petruccelli 2D")


# File names
files = [
    "rusanov_N1400_pout_5960000.0_HEM_haal_0roug.npz",
    "rusanov_N1400_pout_5960000.0_HEM_haal_1roug.npz",
    "rusanov_N1400_pout_5960000.0_HEM_haal_2roug.npz",
    "rusanov_N1400_pout_5960000.0_HEM_haal_3roug.npz",
    # "results\\rusanov_N1600_B13_CFL09_fHEM_purdue.npz"
    # "rusanov_N1400_pout_5960000.0_con_2_5.npz",
    # "rusanov_N1400_pout_5960000.0_con_3.npz",
    # "rusanov_N1400_pout_5960000.0_con_4.npz"
]

x1 = -0.015
x2 = 0.00838


# Dictionary to store data
data = {}
i = 0
# P_e = []

LM_val = [2, 2.5, 3, 4]
LM_val = [2.5]

rough = ['smooth',r"1 $\mu m$", r"2 $\mu m$", r"3 $\mu m$"]


for file in files:
    npzfile = np.load(file)
    p = npzfile['p']
    x = np.linspace(x1, x2, len(p))

    plt.plot(x, p,
            label= f'{rough[i]}',
    )

    i = i+1


plt.xlim([-0.015, 0.00838])
plt.tight_layout()
plt.legend()
plt.show()


# ###########################################
# ############### Second Plot ###############
# ###########################################

# # axs[1].title("P exit, 59.6")
# axs[1].plot(case1_2['position']*1e-3, case1_2['pressure']*1e6, 's', markersize=5, markerfacecolor='orange', markeredgecolor='black', label=r"Experimental")


# # File names
# files = [
#     "rusanov_N1400_pout_5690000.0_con_2_5.npz",
# ]

# x1 = -0.015
# x2 = 0.00838
# Nx = 1400
# x = np.linspace(x1, x2, Nx)

# # Dictionary to store data
# data = {}
# i = 0
# # P_e = []

# LM_val = [2.5, 2.5, 3, 4]

# for file in files:
#     npzfile = np.load(file)
#     p = npzfile['p']

#     axs[1].plot(x, p,
#             label= f'LM {LM_val[i]}',
#     )

#     i = i+1

# # plt.xlim([-0.015, 0.00838])


# ###########################################
# ############### Third Plot ################
# ###########################################

# # axs[1].title("P exit, 59.6")
# axs[2].plot(case1_3['position']*1e-3, case1_3['pressure']*1e6, 's', markersize=5, markerfacecolor='orange', markeredgecolor='black', label=r"Experimental")


# # File names
# files = [
#     "rusanov_N1400_pout_5010000.0_con_2_5.npz",
# ]

# x1 = -0.015
# x2 = 0.00838
# Nx = 1400
# x = np.linspace(x1, x2, Nx)

# # Dictionary to store data
# data = {}
# i = 0
# # P_e = []

# LM_val = [2.5, 2.5, 3, 4]

# for file in files:
#     npzfile = np.load(file)
#     p = npzfile['p']

#     axs[2].plot(x, p,
#             label= f'LM {LM_val[i]}',
#     )

#     i = i+1

# axs[2].legend()
# axs[0].legend()

# # plt.xlim([-0.015, 0.00838])
# plt.tight_layout()
# axs[1].legend()



# case1 = 'C:\\Users\\alber\\OneDrive - Danmarks Tekniske Universitet\\TurboExpo\\Berana\\pin90_T37.xlsx'
# case1_1 = pd.read_excel(case1, sheet_name="pin90_T37_pout51.4")
# case1_2 = pd.read_excel(case1, sheet_name="pin90_T37_pout48.2")
# case1_3 = pd.read_excel(case1, sheet_name="pin90_T37_pout42.6")



# # axs[1].title("P exit, 59.6")
# # axs[3].plot(case1_3['position']*1e-3, case1_3['pressure']*1e6, 's', markersize=5, markerfacecolor='orange', markeredgecolor='black', label=r"Experimental")


# # File names
# files = [
#     "rusanov_N1200_pout_4260000.0_flash_0_25.npz",
# ]

# x1 = -0.015
# x2 = 0.00838
# Nx = 1200
# x = np.linspace(x1, x2, Nx)

# # Dictionary to store data
# data = {}
# i = 0
# # P_e = []

# LM_val = [0.25, 2.5, 3, 4]

# for file in files:
#     npzfile = np.load(file)
#     p = npzfile['M']

#     axs[3].plot(x, p,
#             label= f'LM {LM_val[i]}',
#     )

#     i = i+1

# axs[2].legend()
# axs[0].legend()
# axs[3].legend()

# plt.xlim([-0.015, 0.00838])
# plt.tight_layout()
# plt.legend()
# plt.show()
