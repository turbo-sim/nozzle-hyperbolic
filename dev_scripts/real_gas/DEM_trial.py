import numpy as np
import matplotlib.pyplot as plt
import nozzle_hyperbolic as hyp
import barotropy as bpy

data01 = np.load("rusanov_N200_pin90_tin37_pout51_rough0.npz")

e, h, x, u, p, rho, a = data01["e"], data01["h"], data01["x"], data01["u"], data01["p"], data01["rho"], data01["a"]

phase_change = "condensation"
Q_onset = 0.5
Q_width = 0.01
p = np.zeros_like(rho)  # Initialize pressure array for the last time step
a_sound = np.zeros_like(rho)
s = np.zeros_like(rho)
Q = np.zeros_like(rho)
T = np.zeros_like(rho)

fluid_name = "CO2"
fluid = bpy.fluid_properties.Fluid(fluid_name)
p_stagnation = 58.96e5
T_stagnation = 314.67
stagnation_state = fluid.get_state(bpy.fluid_properties.PT_INPUTS, p_stagnation, T_stagnation)

rhoT_guess_metastable = [stagnation_state["rho"], stagnation_state["T"]]
rhoT_guess_equilibrium = [stagnation_state["rho"], stagnation_state["T"]]

# Loop through each cell to set the state and retrieve pressure at the last time step
for j in range(len(rho)):
    # Set the fluid state using density and specific internal energy at the last time step
    # state = fluid.get_state(bpy.fluid_properties.DmassUmass_INPUTS, rho[j], e[j])
    # p[j] = state.p  # Access 'p' from the FluidState dictionary-like object
    # a_sound[j] = state.speed_sound
    # s[j] = state.s
    # Q[j] = state.Q
    # T[j] = state.T

    props_blend, props_eq, props_meta = fluid.get_state_blending(
        prop_1="rho",
        prop_1_value=rho[j],
        prop_2="u",
        prop_2_value=e[j],
        rhoT_guess_equilibrium=rhoT_guess_equilibrium,
        rhoT_guess_metastable=rhoT_guess_metastable,
        phase_change=phase_change,
        blending_variable="Q",
        blending_onset=Q_onset,
        blending_width=Q_width,
        generalize_quality=True,
        supersaturation=True,
        print_convergence=False,
        solver_algorithm="lm",
    )

    # Update initial guess
    rhoT_guess_metastable = [props_meta["rho"], props_meta["T"]]
    rhoT_guess_equilibrium = [props_eq["rho"], props_eq["T"]]   
    p[j] = props_blend.p  # Access 'p' from the FluidState dictionary-like object
    a_sound[j] = props_blend.speed_sound
    s[j] = props_blend.s
    Q[j] = props_blend.Q
    T[j] = props_blend.T

plt.figure(figsize=(6, 5))
plt.grid(True)
ax1 = plt.gca()
plt.xlabel(r'$x$ (mm)')
plt.ylabel(r'$P$ (bar)')
plt.figure(figsize=(6, 5))
plt.plot(x, p, '-', color='darkorange', linewidth=1.8, markerfacecolor='white', label="rusanov")

# plt.figure(figsize=(6, 5))
# plt.plot(x, Q, '-', color='darkorange', linewidth=1.8, markerfacecolor='white', label="rusanov")

plt.show()

np.savez("trial.npz", x=x, a=a_sound, p=p, Q=Q)
