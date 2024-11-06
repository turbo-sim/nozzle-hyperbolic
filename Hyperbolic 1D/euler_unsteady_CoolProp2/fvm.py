import numpy as np
import fluid_properties
from flux import *

def euler_eigenvalues(Q, fluid, parameters):
    # Extract variables from the state vector Q
    rho = Q[0, :]        # Density
    rhou = Q[1, :]       # Momentum (rho * u)
    p = parameters['p']  # Pressure, pre-computed before calling this function

    # Compute velocity
    u = rhou / rho       # Velocity

    # Initialize array for speed of sound
    c = np.zeros_like(rho)  # Speed of sound array

    # Loop through each cell to calculate speed of sound using CoolProp
    for i in range(len(rho)):        
        # Set the fluid state using density and internal energy
        state = fluid.set_state(fluid_properties.DmassP_INPUTS, rho[i], p[i])
        
        # Retrieve speed of sound for this state
        c[i] = state.speed_sound

    # Calculate the eigenvalues
    a = np.array([u - c, u, u + c])

    return a


def euler_flux(Q, p):
    # Extract variables from the state vector Q
    rho = Q[0, :]        # Density
    rhou = Q[1, :]       # Momentum (rho * u)
    rhoE = Q[2, :]       # Total energy
    
    # Compute velocity and check consistency
    u = rhou / rho       # Velocity

    # Compute the flux vector
    f = np.array([
        rhou,                    # Flux for mass conservation
        rhou * u + p,            # Flux for momentum conservation
        (rhoE + p) * u           # Flux for energy conservation
    ])

    return f


def fvm_euler_1d(x, t, parameters):
    # Extend the grid to ghost cells
    dx_const = parameters['dx']
    A_nodes = parameters['A_nodes'] 
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

    fluid_name = parameters['fluid_name']
    fluid = fluid_properties.Fluid(fluid_name)
    stagnation_state = fluid.set_state(fluid_properties.PT_INPUTS, p_stagnation, T_stagnation)
    cp_stagnation = stagnation_state.cpmass
    s0 = stagnation_state.smass
    h0 = stagnation_state.hmass

    # Define threshold for steady state
    steady_state_threshold = 1e-6
    max_iterations = 100000
    n = 0
    warmup_iterations = 50

    while t[n] <= t_end and n < max_iterations:
        if n + 1 >= Q.shape[2]:  # Expand Q and t if the current allocation is exhausted
            Q = np.concatenate((Q, np.zeros((N, Nj, 100))), axis=2)
            t = np.concatenate((t, np.zeros(100)))
            
        # Extrapolate linearly the inlet velocity
        u_in = 2 * Q[1, 1, n] / Q[0, 1, n] - Q[1, 2, n] / Q[0, 2, n]

        #Compute the inlet Mach number and static inlet conditions
        h = h0 - 0.5*u_in**2
        static_state = fluid.set_state(fluid_properties.HmassSmass_INPUTS,h, s0)
        rho_static = static_state.rhomass
        a_static = static_state.speed_sound
        M_in = u_in / a_static

        # Inlet boundary condition
        Q[0, 0, n] = rho_static
        Q[1, 0, n] = rho_static * u_in
        Q[2, 0, n] = static_state.umass * rho_static + 0.5 * rho_static * u_in**2  # Total energy (internal + kinetic)

        # Outlet boundary condition
        Q[0, -1, n] = 2 * Q[0, -2, n] * A_nodes[-2] / A_nodes[-1] - Q[0, -3, n] * A_nodes[-3] / A_nodes[-1]
        Q[1, -1, n] = 2 * Q[1, -2, n] * A_nodes[-2] / A_nodes[-1] - Q[1, -3, n] * A_nodes[-3] / A_nodes[-1]      

       # Calculate outlet Mach number
        u_outlet = Q[1, -1, n] / Q[0, -1, n]  # Velocity at the outlet
        rho_outlet = Q[0, -1, n]              # Density at the outlet
        e_outlet = (Q[2, -1, n] - 0.5 * rho_outlet * u_outlet**2)/rho_outlet

        outlet_state = fluid.set_state(fluid_properties.DmassUmass_INPUTS, rho_outlet, e_outlet)
        a_outlet = outlet_state.speed_sound       # Speed of sound at the outlet
        Ma_out = u_outlet / a_outlet              # Mach number at the outlet

        p = np.zeros(Q.shape[1])  # Initialize pressure array
        c = np.zeros(Q.shape[1])  # Initialize pressure array 
        for j in range(Q.shape[1]-1):
            rho = Q[0, j, n]  # Density at cell j
            u = Q[1, j, n] / rho  # Velocity at cell j
            try:
                # Calculate specific internal energy (from total energy minus kinetic energy)
                e_internal = Q[2, j, n]/ rho - 0.5 * u**2
                # Set the fluid state using density and internal energy
                internal_state = fluid.set_state(fluid_properties.DmassUmass_INPUTS, rho, e_internal)
                # Retrieve the pressure from the state
            except Exception as e:
                 print(f"Warning: {e}, density is ", rho)

            p[j] = internal_state.p
            c[j] = internal_state.speed_sound

        # Subsonic Outlet Boundary Condition
        if Ma_out < 1:
            # Set outlet pressure (subsonic flow assumes we control the pressure at the outlet)
            outlet_state = fluid.set_state(fluid_properties.DmassP_INPUTS, rho_outlet, p_out)
            e_outlet = outlet_state.umass  # Internal energy per unit mass
            Q[2, -1, n] = e_outlet * rho_outlet + 0.5 * rho_outlet * u_outlet**2  # Total energy
            p[-1] = p_out
        else:
            # For supersonic flow, simply extrapolate values from the interior
            Q[2, -1, n] = 2 * Q[2, -2, n] * A_nodes[-2] / A_nodes[-1] - Q[2, -3, n] * A_nodes[-3] / A_nodes[-1]  
            p[-1] = p[-2]

        print(Ma_out)
        
        parameters['p'] = p

        # Compute source term
        S = np.zeros((3, Nj))
        S[1, 1:Nj-1] = p[1:Nj-1] * (A_nodes[2:] - A_nodes[:-2]) / (2 * dx_const)
        S[1, 0] = p[0] * (A_nodes[1] - A_nodes[0]) / dx_const
        S[1, -1] = p[-1] * (A_nodes[-1] - A_nodes[-2]) / dx_const

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
            # Check for steady state after warm-up
            if delta_Q < steady_state_threshold:
                print(f"Steady-state reached after {t[n]} s")
                break

        # Adjust the time step based on the CFL condition for interior cells
        lambda_max = np.max(np.abs(parameters['eigenvalues_function'](Q[:, 2:Nj-2, n])), axis=0)
        x_face = x[1:Nj-2] + (x[2:Nj-1] - x[1:Nj-2]) / 2
        dx = x_face[1:] - x_face[:-1]
        dt = parameters['CFL'] / np.max(lambda_max / dx)
        parameters['dt'] = dt

        # Update the time
        t[n+1] = t[n] + dt
        n += 1

    # Discard the solution in the ghost cells
    Q = Q[:, :, :n]
    t = t[:n]

    return Q, t

def F_numeric(Q_L, Q_R, parameters):
    if parameters['space_method'] == 'roe':
        return F_Roe(Q_L, Q_R, parameters)
    elif parameters['space_method'] == 'rusanov':
        return F_Rusanov(Q_L, Q_R, parameters)
    elif parameters['space_method'] == 'HLLC':
        return F_HLLC(Q_L, Q_R, parameters)
    elif parameters['space_method'] == 'HLL':
        return F_HLL(Q_L, Q_R, parameters)
    elif parameters['space_method'] == 'LF':
        return F_LaxFriedrichs(Q_L, Q_R, parameters)
    else:
        raise ValueError("Choose a valid space method")
    
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