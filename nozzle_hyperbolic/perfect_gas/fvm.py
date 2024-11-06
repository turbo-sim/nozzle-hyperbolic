import numpy as np
from .flux import *
import matplotlib.pyplot as plt

def euler_eigenvalues(Q, gamma):
    # Extract variables from the state vector Q
    rho = Q[0, :]        # Density
    rhou = Q[1, :]       # Momentum (rho * u)
    rhoE = Q[2, :]       # Total energy

    # Compute velocity, pressure, and speed of sound
    u = rhou / rho       # Velocity
    p = (gamma - 1) * (rhoE - 0.5 * (rhou ** 2) / rho)  # Pressure
    c = np.sqrt(gamma * p / rho)  # Speed of sound

    # Return the eigenvalues (u-c, u, u+c)
    a = np.array([u - c, u, u + c])

    return a

def euler_flux(Q, gamma):
    # Extract variables from the state vector Q
    rho = Q[0, :]        # Density
    rhou = Q[1, :]       # Momentum (rho * u)
    rhoE = Q[2, :]       # Total energy

    # Compute velocity and pressure
    u = rhou / rho       # Velocity
    p = (gamma - 1) * (rhoE - 0.5 * (rhou ** 2) / rho)  # Pressure

    # Compute the flux vector
    f = np.array([
        rhou,                    # Flux for mass conservation
        rhou * u + p,            # Flux for momentum conservation
        (rhoE + p) * u           # Flux for energy conservation
    ])

    return f

def fvm_euler_1d(x, t, parameters):
    # Extend the grid to ghost cells
    dx = parameters['dx']
    rx = parameters['r_function']
    A_nodes = rx(x)**2 * np.pi

    parameters['A_nodes'] = A_nodes
    Nj = len(x)

    # Initialize the time vector
    t_end = t[-1]
    t = np.zeros(1000)
    dt = np.sqrt(np.finfo(float).eps)
    parameters['dt'] = dt

    # Initialize the solution
    N = parameters['N_eq']  # Number of equations of the system
    Q = np.zeros((N, Nj, 1000))  # Preallocate space for the solution
    Q_o = parameters['Q_o']  # Load the initial condition
    Q[:, :, 0] = Q_o(x)  # Set the initial condition

    # Initialize boundary conditions
    p_stagnation = parameters['pT_stagnation'][0]
    T_stagnation = parameters['pT_stagnation'][1]
    p_out = parameters['p_out']
    gamma = 1.4
    R_const = 287

    # Define threshold for steady state
    steady_state_threshold = 1e-6
    max_iterations = 10000000
    n = 0
    warmup_iterations = 100

    while t[n] <= t_end and n < max_iterations:
        if n + 1 >= Q.shape[2]:  # Expand Q and t if the current allocation is exhausted
            Q = np.concatenate((Q, np.zeros((N, Nj, 100))), axis=2)
            t = np.concatenate((t, np.zeros(100)))
            
        # Extrapolate linearly the inlet velocity
        # u_in = 2 * Q[1, 1, n] / Q[0, 1, n] - Q[1, 2, n] / Q[0, 2, n]
        u_in = Q[1, 1, n] / Q[0, 1, n]


        # Compute the inlet Mach number and static inlet conditions
        # T_static = T_stagnation / (1 + (gamma - 1) / 2 * (u_in / np.sqrt(gamma * R_const * T_stagnation))**2)
        T_static = T_stagnation - 0.5*u_in**2*(gamma-1)/(gamma*R_const)
        # a_in = np.sqrt(gamma * R_const * T_static)
        # M_in = u_in / a_in
        M_in = np.sqrt(2*((T_stagnation/T_static)-1)/(gamma-1))
        p_static = p_stagnation / (1 + (gamma - 1) / 2 * M_in**2)**(gamma / (gamma - 1))
        rho_static = p_static / (R_const * T_static)

        # Inlet boundary condition
        Q[0, 0, n] = rho_static
        Q[1, 0, n] = rho_static * u_in
        Q[2, 0, n] = p_static / (gamma - 1) + 0.5 * rho_static * u_in**2

        # Update density (Q[0, :, :]) and momentum (Q[1, :, :]) at the outlet
        # Q[0, -1, n] = 2 * Q[0, -2, n] * A_nodes[-2] / A_nodes[-1] - Q[0, -3, n] * A_nodes[-3] / A_nodes[-1]
        # Q[1, -1, n] = 2 * Q[1, -2, n] * A_nodes[-2] / A_nodes[-1] - Q[1, -3, n] * A_nodes[-3] / A_nodes[-1]
        Q[0, -1, n] = Q[0, -2, n]
        Q[1, -1, n] = Q[1, -2, n] 
        u_outlet = Q[1, -1, n]/Q[0, -1, n]
        rho_outlet = Q[0, -1, n]
        p_outlet = (gamma - 1) * (Q[2, -1, n] - 0.5 * Q[1, -1, n]**2 / Q[0, -1, n])
        c_outlet = np.sqrt(gamma*p_outlet/rho_outlet)
        Ma_outlet = u_outlet/c_outlet

        # Calculate internal pressure in the last cell before the boundary
        p_internal = (gamma - 1) * (Q[2, -2, n] - 0.5 * Q[1, -2, n]**2 / Q[0, -2, n])
        Q[2, -1, n] = p_out / (gamma - 1) + 0.5 * Q[1, -1, n]**2 / Q[0, -1, n]

        # # Check if the flow is subsonic or supersonic at the outlet
        # if (p_out - p_internal) > 0:
        #     try:
        #         # Subsonic flow: impose outlet pressure condition
        #         Q[2, -1, n] = p_out / (gamma - 1) + 0.5 * Q[1, -1, n]**2 / Q[0, -1, n]
        #     except ValueError as e:
        #         Q[2, -1, n] = 2 * Q[2, -2, n] * A_nodes[-2] / A_nodes[-1] - Q[2, -3, n] * A_nodes[-3] / A_nodes[-1]
        # else:
        #     # Supersonic flow: outflow boundary condition
        #     Q[2, -1, n] = 2 * Q[2, -2, n] * A_nodes[-2] / A_nodes[-1] - Q[2, -3, n] * A_nodes[-3] / A_nodes[-1]

        # Compute source term
        p = (gamma - 1) * (Q[2, :, n] - 0.5 * Q[1, :, n]**2 / Q[0, :, n])
        S = np.zeros((3, Nj))
        S[1, 1:Nj-1] = p[1:Nj-1] * (A_nodes[2:] - A_nodes[:-2]) / (2 * dx)
        S[1, 0] = p[0] * (A_nodes[1] - A_nodes[0]) / dx
        S[1, -1] = p[-1] * (A_nodes[-1] - A_nodes[-2]) / dx

        # Use the desired time integration method
        if parameters['time_method'] == 'euler':
            Q[:, :, n+1] = Q[:, :, n] + (dt / A_nodes) * R(Q[:, :, n], t[n], x, parameters) + (dt / A_nodes) * S

        elif parameters['time_method'] == 'RK3':
            Q1 = Q[:, :, n] + (dt / A_nodes) * R(Q[:, :, n], t[n], x, parameters)
            Q2 = 3/4 * Q[:, :, n] + 1/4 * Q1 + (dt / A_nodes) / 4 * R(Q1, t[n] + dt/4, x, parameters)
            Q[:, :, n+1] = 1/3 * Q[:, :, n] + 2/3 * Q2 + 2/3 * (dt / A_nodes) * R(Q2, t[n] + dt/3, x, parameters) + (dt / A_nodes) * S

        elif parameters['time_method'] == 'RK4':
            k1 = R(Q[:, :, n], t[n], x, parameters)
            k2 = R(Q[:, :, n] + k1 * dt / 2, t[n] + dt / 2, x, parameters)
            k3 = R(Q[:, :, n] + k2 * dt / 2, t[n] + dt / 2, x, parameters)
            k4 = R(Q[:, :, n] + k3 * dt, t[n] + dt, x, parameters)
            Q[:, :, n+1] = Q[:, :, n] + (1/6 * (k1 + 2*k2 + 2*k3 + k4)) * (dt / A_nodes) + (dt / A_nodes) * S

        # Calculate the maximum change between iterations
        if n > warmup_iterations:
            delta_Q = np.max(np.abs(Q[1, 1:-1, n+1] - Q[1, 1:-1, n]) / np.abs(Q[1, 1:-1, n+1]))

            if n % 25 == 0:
                print(f"Iteration {n}: delta_Q = {delta_Q:0.4e}")
                print(Ma_outlet)

            # Check for steady state after warm-up
            if delta_Q < steady_state_threshold:
                print(f"Steady-state reached after {t[n]} s")
                break

        # Adjust the time step based on the CFL condition for interior cells
        lambda_max = np.max(np.abs(parameters['eigenvalues_function'](Q[:, 2:Nj-2, n])), axis=0)
        dt = parameters['CFL'] / np.max(lambda_max / dx)
        parameters['dt'] = dt

        # Update the time
        t[n+1] = t[n] + dt
        n += 1

    # Discard the solution in the ghost cells
    Q = Q[:, :, :n]
    t = t[:n]

    return Q, t


def R(Q, _, x, parameters):
    N, Nj = Q.shape

    # Compute the right-hand side using a constant cell reconstruction
    if parameters['reconstruction'] == 'constant':
        Q_L = Q[:, :-1]  # Left face
        Q_R = Q[:, 1:]   # Right face
        F = F_numeric(Q_L, Q_R, parameters)

        x_face = x[:-1] + (x[1:] - x[:-1]) / 2
        dx = np.ones((N, 1)) * (x_face[1:] - x_face[:-1])

        RHS = np.zeros((N, Nj))
        RHS[:, 1:Nj-1] = -(F[:, 1:] - F[:, :-1]) / dx

    return RHS


def F_numeric(Q_L, Q_R, parameters):
    if parameters['space_method'] == 'roe':
        return F_Roe(Q_L, Q_R, parameters)
    elif parameters['space_method'] == 'rusanov':
        return F_Rusanov(Q_L, Q_R, parameters)
    elif parameters['space_method'] == 'HLLC':
        return F_HLLC(Q_L, Q_R, parameters)
    elif parameters['space_method'] == 'HLLD':
        return F_HLLC(Q_L, Q_R, parameters)
    elif parameters['space_method'] == 'HLL':
        return F_HLL(Q_L, Q_R, parameters)
    elif parameters['space_method'] == 'HLLE':
        return F_HLL(Q_L, Q_R, parameters)
    elif parameters['space_method'] == 'LF':
        return F_LaxFriedrichs(Q_L, Q_R, parameters)
    else:
        raise ValueError("Choose a valid space method")

# def minmod(a, b):
#     """Minmod limiter function for the MUSCL approach."""
#     return np.sign(a) * np.maximum(0, np.minimum(np.abs(a), np.sign(a) * b))



# def R(Q, _, x, parameters):
#     N, Nj = Q.shape

#     # Compute the right-hand side using a constant cell reconstruction
#     if parameters['reconstruction'] == 'constant':
#         # Calculate differences between neighboring cells
#         dq_left = Q[:, 1:] - Q[:, :-1]  # Slope between left and center
#         dq_right = Q[:, 2:] - Q[:, 1:-1]  # Slope between center and right
#         dq_limited = minmod(dq_left[:, :-1], dq_right)  # Apply minmod limiter

#         # Define Q_L and Q_R with limited slopes, and keep boundary conditions
#         Q_L = np.zeros((N, Nj - 1))  # Left state on each face
#         Q_R = np.zeros((N, Nj - 1))  # Right state on each face
#         Q_L[:, 1:] = Q[:, 1:-1] - 0.5 * dq_limited  # Internal left state
#         Q_R[:, :-1] = Q[:, 1:-1] + 0.5 * dq_limited  # Internal right state
#         Q_L[:, 0] = Q[:, 0]  # Boundary
#         Q_R[:, -1] = Q[:, -1]  # Boundary

#         # Compute fluxes
#         F = F_numeric(Q_L, Q_R, parameters)

#         # Compute cell face positions and dx values
#         x_face = x[:-1] + (x[1:] - x[:-1]) / 2
#         dx = np.ones((N, 1)) * (x_face[1:] - x_face[:-1])

#         # Compute right-hand side (RHS)
#         RHS = np.zeros((N, Nj))
#         RHS[:, 1:Nj-1] = -(F[:, 1:] - F[:, :-1]) / dx

#     return RHS