import os
import pandas as pd
import matplotlib.pyplot as plt


# Experimental Data
main_file = "cases_summary.xlsx"
data = pd.read_excel(main_file)
colours = ['k', 'g', 'y', 'r', 'b']
plot_variable = 'pressure'

plt.figure(num=1, figsize=(6, 5))
plt.figure(num=2, figsize=(6, 5))

for i, row in data.iterrows():
    file_pressure = row["file_pressure"]
    file_compressability = row["file_compressability"]

    pressure_data = pd.read_excel(os.path.join("profiles",file_pressure))
    compressability_data = pd.read_excel(os.path.join("profiles",file_compressability))

    # Plot Pressure
    plt.figure(1)
    plt.plot(pressure_data['Position'], pressure_data['pressure_ratio'], marker='o', linestyle='-', color=colours[i], label=f'Case {row['Case']}')
    
    plt.figure(2)
    plt.plot(compressability_data['Position'], compressability_data['z'], marker='o', linestyle='-', color=colours[i], label=f'Case {row['Case']}')
    

plt.figure(1)
plt.xlabel('Position')
plt.ylabel('P/Pt')
plt.title(f"Pressure Plot")
plt.grid(True)
plt.legend()

plt.figure(2)
plt.xlabel('Position')
plt.ylabel('Z')
plt.title(f"Compressability Plot")
plt.grid(True)
plt.legend()

plt.show()
