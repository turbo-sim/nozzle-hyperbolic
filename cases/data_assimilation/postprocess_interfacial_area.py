"""
plotting the interfacial area
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config_loader import load_config
from nozzle_hyperbolic.fvm_solver.source_two_fluid import interfacial_area

run_folder = "results/thick_N400_thick_N400_muscl_van_albada_20261014_172944"
results = pd.read_csv(os.path.join(run_folder , "solution.csv"))

ag = np.asarray(results["alpha_g"])
x = np.asarray(results["x"])

a_i = np.empty(ag.shape[0], dtype=float)


# NB: IT IS IMPORTANT TO USE THE CORRECT INTERFACIAL AREA CORRELATION!
for i in range(a_i.shape[0]):
    a_i[i], _ = interfacial_area(ag[i])

plt.figure(1)
plt.plot(x, a_i, label=r"$a_i$")

plt.figure(2)
plt.plot(x, ag, label=r"$\alpha_g$")

plt.show()
