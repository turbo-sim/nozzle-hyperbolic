# %% Import
import pandas as pd
import numpy as np
from CoolProp.CoolProp import PropsSI

# Sample data
data = pd.read_excel("cases_summary.xlsx")
pressure_data = pd.read_excel("pressure_profile_summary_original.xlsx")
fluidname = data['fluidname'].values[0]

pressure_data['pressure'] = pressure_data['pressure']*10**6

# %% compute the convergin part
# Define the functions
def compute_converging_T(pressure, s_in):
    temperature = round(PropsSI('T', 'P', pressure, 'S', s_in, fluidname),4)
    return temperature

def compute_diverging_T(pressure):
    temperature = round(PropsSI('T','P',pressure,'Q',0,fluidname),4)
    return temperature

# Define a custom function that applies different calculations based on conditions
def calculate(row):
    if row['type'] == "converging":
        # PropsSI(output, input1, value1, input2, value2, fluid)
        inlet_pressure = data.loc[data['Case'] == row['case'], 'P_0_in'].values[0]
        inlet_temperature = data.loc[data['Case'] == row['case'], 'T_0_in'].values[0] + 273.15
        s_in = PropsSI('S', 'P', inlet_pressure, 'T', inlet_temperature, fluidname)
        return compute_converging_T(row['pressure'], s_in)
    elif row['type'] == "diverging":
        return compute_diverging_T(row['pressure'])

# Apply the custom function row-wise to create column 'temperature'
pressure_data['temperature'] = pressure_data.apply(calculate, axis=1)

with pd.ExcelWriter("pressure_profile_summary_converted.xlsx") as writer:
    pressure_data.to_excel(writer, sheet_name="Sheet1", index=False)

print(pressure_data)
