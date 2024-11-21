# %%

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import barotropy as bpy


CASE_INDEX = [1, 4, 9, 15, 18, 24, 31, 34, 38]
CASES_NUMBER = [
    "1-3",
    "4-8",
    "9-14",
    "15-17",
    "18-23",
    "24-30",
    "31-33",
    "34-37",
    "38-43",
]

# Define plot settings
bpy.print_package_info()
bpy.set_plot_options()
SHOW_FIG = True
SAVE_FIG = True

# Create folder to save results
DIR_OUT = "./output"
save_figures = True
if not os.path.isdir(DIR_OUT):
    os.makedirs(DIR_OUT)

# Read the case summary file
EXCEL_DATA_SUMMARY = "cases_summary.xlsx"
EXCEL_DATA_PROFILES = "pressure_profile_summary_converted.xlsx"
data_summary = pd.read_excel(EXCEL_DATA_SUMMARY)
data_profiles = pd.read_excel(EXCEL_DATA_PROFILES)

# Create fluid object
fluid_name = data_summary["fluidname"].values[0]
fluid = bpy.Fluid(name=fluid_name, backend="HEOS")
ax = fluid.plot_phase_diagram(
    "s", "T", plot_spinodal_line=True, plot_quality_isolines=True
)

COLORS_PYTHON = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

# Define case parameters
dT_subcooling = 5
polytropic_efficiency = 1.00
calculation_type = "equilibrium"
q_onset = 0.2  # use the liquid line till quality of 5%
q_transition = 0.05  # transient from metastable liquid to equilibrium zone

i = 0

for case in CASE_INDEX:
    print(f"Running case {case}")
    case_data = data_summary[data_summary["Case"] == case].squeeze()

    # total preessure in Pa
    p_in = float(case_data["P_0_in"])

    # Static pressure in Pa
    p_out = case_data["P_out"]
    # p_out = p_in/1.5

    # Total temperature in K
    T_in = case_data["T_0_in"] + 273.15

    # Create barotropic model object
    model = bpy.BarotropicModel(
        fluid_name=fluid_name,
        T_in=T_in,
        p_in=p_in,
        p_out=p_out,
        efficiency=polytropic_efficiency,
        calculation_type=calculation_type,
        blending_onset=q_onset,
        blending_width=q_transition,
        HEOS_solver="hybr",
        ODE_solver="LSODA",
        ODE_tolerance=1e-9,
        polynomial_degree=5,
        polynomial_format="horner",
        output_dir=DIR_OUT,
        polynomial_variables=["density", "viscosity", "speed_sound"],
    )

    # Evaluate barotropic model and export polynomial expressions
    model.solve()

    colors = COLORS_PYTHON[i]

    a = ax.plot(model.states["s"], model.states["T"], label=f"Cases {CASES_NUMBER[i]}", color = colors)
    i = i + 1

# plt.legend()
# plt.grid(False)
# plt.savefig("cases_summary_Ts_diagram.png")
# plt.show()


