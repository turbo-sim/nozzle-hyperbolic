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
    filename = row["file_pressure"]
    pressure_data = pd.read_excel(os.path.join("profiles",filename))

    # Plot Pressure
    if row['Nozzle'] == "M1.5":
        plt.figure(1)

    else:
        plt.figure(2)


    plt.plot(pressure_data['Position'], pressure_data['pressure_ratio'], marker='o', linestyle='', color=colours[i], label=f'Case {row['Nozzle']}, $z_t$={row["Zt"]}')
    


plt.figure(1)
plt.xlabel('Position')
plt.ylabel('P/Pt')
plt.title(f"Pressure Plot")
plt.legend()

plt.figure(2)
plt.xlabel('Position')
plt.ylabel('P/Pt')
plt.title(f"Pressure Plot")
plt.legend()


plt.show()
