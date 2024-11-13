import numpy as np
from .. import fluid_properties

def F_Rusanov(Q_L, Q_R, parameters):
    p = parameters['p']  # Pressure, pre-computed before calling this function
    p_L = p[:-1]
    p_R = p[1:]

    # Compute the left and right fluxes
    F_L = parameters['flux_function'](Q_L, p_L)
    F_R = parameters['flux_function'](Q_R, p_R)
    A_nodes = parameters['A_nodes']
    A_nodes = np.ones((3, 1)) * A_nodes  # Broadcast A_nodes to match dimensions

    # Compute the characteristic speed
    a1 = np.max(np.abs(parameters['eigenvalues_function'](Q_L)), axis=0)
    a2 = np.max(np.abs(parameters['eigenvalues_function'](Q_R)), axis=0)
    a = np.maximum(a1, a2)
    a = np.vstack([a, a, a])  # Stack to create the 3 rows for each variable

    # Compute the Rusanov numerical flux
    F = 0.5 * (F_L * A_nodes[:, :-1] + F_R * A_nodes[:, 1:] - a * (Q_R * A_nodes[:, 1:] - Q_L * A_nodes[:, :-1]))

    return F

def F_Roe(Q_L, Q_R, parameters):
    p = parameters['p']  # Pressure, pre-computed before calling this function
    p_L = p[:-1]
    p_R = p[1:]

    F_L = parameters['flux_function'](Q_L, p_L)
    F_R = parameters['flux_function'](Q_R, p_L)
    F = np.zeros_like(F_L)
    A_nodes = parameters['A_nodes']
    p = parameters['p']  # Pressure, pre-computed before calling this function

    for i in range(Q_L.shape[1]):
        A = parameters['roe_function'](Q_L[:, i], Q_R[:, i], p_L[i], p_R[i])
        F[:, i] = 0.5 * (F_L[:, i] * A_nodes[i] + F_R[:, i] * A_nodes[i + 1] - A @ (Q_R[:, i] * A_nodes[i + 1] - Q_L[:, i] * A_nodes[i]))
    return F

def euler_roe_matrix(Q_L, Q_R, p_L, p_R, fluid):
    # Compute the left state
    rho_L = Q_L[0]
    rhou_L = Q_L[1]
    rhoE_L = Q_L[2]
    u_L = rhou_L / rho_L

    # Compute the right state
    rho_R = Q_R[0]
    rhou_R = Q_R[1]
    rhoE_R = Q_R[2]
    u_R = rhou_R / rho_R
    
    H_L = (rhoE_L + p_L) / rho_L
    H_R = (rhoE_R + p_R) / rho_R

    # Compute the left state
    state_L = fluid.set_state(fluid_properties.DmassP_INPUTS, rho_L, p_L)
    c_L = state_L.speed_sound

    # Compute the right state
    state_R = fluid.set_state(fluid_properties.DmassP_INPUTS, rho_R, p_R)
    c_R = state_R.speed_sound


    # Compute the Roe averages (keep these lines unchanged)
    sqrt_rho_L = np.sqrt(rho_L)
    sqrt_rho_R = np.sqrt(rho_R)
    u = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) / (sqrt_rho_L + sqrt_rho_R)
    H = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) / (sqrt_rho_L + sqrt_rho_R)

    # Compute averaged state for speed of sound
    avg_rho = (rho_L + rho_R) / 2  # Simple average for density
    avg_p = (p_L + p_R) / 2        # Simple average for pressure
    state = fluid.set_state(fluid_properties.DmassP_INPUTS, avg_rho, avg_p)
    c = state.speed_sound     # Real-gas speed of sound at averaged state

    # Compute the eigenvector matrix
    R = np.array([[1, 1, 1],
                  [u - c, u, u + c],
                  [H - c * u, 0.5 * u**2, H + c * u]])

    # Compute the eigenvalue matrix with entropy fix
    eig1_L = u_L - c_L
    eig1_R = u_R - c_R
    eig3_L = u_L + c_L
    eig3_R = u_R + c_R
    L = np.diag([H_fix(eig1_L, eig1_R, u - c, c),
                 u,
                 H_fix(eig3_L, eig3_R, u + c, c)])

    # Compute Roe's matrix
    A = R @ L @ np.linalg.inv(R)

    return A

def H_fix(a_L, a_R, a, c):
    a_abs = abs(a)
    delta = max(abs(a_R - a_L), 0.01 * c)  # threshold for entropy fix
    
    if a_abs >= delta:
        return a_abs
    else:
        return (a_abs**2 + delta**2) / (2 * delta)
    

def F_HLLC(Q_L, Q_R, parameters):

    p = parameters['p']  # Pressure, pre-computed before calling this function
    p_L = p[:-1]
    p_R = p[1:]

    # Compute the left and right fluxes
    F_L = parameters['flux_function'](Q_L, p_L)
    F_R = parameters['flux_function'](Q_R, p_R)
    A_nodes = parameters['A_nodes']  # A_nodes should be a 1xNj vector for the area at each node

    # Compute eigenvalues (wave speeds) for the left and right states
    eig_L = parameters['eigenvalues_function'](Q_L)
    eig_R = parameters['eigenvalues_function'](Q_R)
    
    S_L = np.minimum(eig_L[0, :], eig_L[2, :])  # Minimum left wave speed (element-wise)
    S_R = np.maximum(eig_R[0, :], eig_R[2, :])  # Maximum right wave speed (element-wise)
    
    # Compute density, velocity, and pressure for left and right states
    rho_L = Q_L[0, :]
    rho_R = Q_R[0, :]
    u_L = Q_L[1, :] / rho_L
    u_R = Q_R[1, :] / rho_R
    
    
    # Compute the contact wave speed (S_M)
    S_M = (p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)) / \
          (rho_L * (S_L - u_L) - rho_R * (S_R - u_R))
    
    # Compute intermediate states, ensuring sharp transitions across contact discontinuities
    rho_L_star = rho_L * (S_L - u_L) / (S_L - S_M)
    u_L_star = S_M
    p_L_star = p_L + rho_L * (S_L - u_L) * (S_M - u_L)
    rho_R_star = rho_R * (S_R - u_R) / (S_R - S_M)
    u_R_star = S_M
    p_R_star = p_R + rho_R * (S_R - u_R) * (S_M - u_R)

    fluid_name = parameters['fluid_name'] 
    fluid = fluid_properties.Fluid(fluid_name)
    
    energy_L_star = np.zeros(Q_L.shape[1])
    energy_R_star = np.zeros(Q_R.shape[1])

    for j in range(Q_L.shape[1]):

        state_L = fluid.set_state(fluid_properties.DmassP_INPUTS, rho_L_star[j], p_L_star[j])
        energy_L_star[j] = state_L.umass 
        state_R = fluid.set_state(fluid_properties.DmassP_INPUTS, rho_R_star[j], p_R_star[j])
        energy_R_star[j] = state_R.umass 

    Q_L_star = np.array([rho_L_star, rho_L_star * u_L_star, energy_L_star * rho_L_star])
    Q_R_star = np.array([rho_R_star, rho_R_star * u_R_star, energy_R_star * rho_R_star])
    
    # Preallocate flux matrix
    F = np.zeros_like(F_L)
    
    # Enhanced HLLC Flux cases
    for i in range(len(S_L)):
        if S_L[i] > 0:
            # Left-going wave dominates
            F[:, i] = F_L[:, i] * A_nodes[i]
        elif S_L[i] <= 0 <= S_M[i]:
            # Intermediate state on the left side
            F[:, i] = F_L[:, i] * A_nodes[i] + S_L[i] * (Q_L_star[:, i] - Q_L[:, i]) * A_nodes[i]
        elif S_M[i] <= 0 <= S_R[i]:
            # Intermediate state on the right side
            F[:, i] = F_R[:, i] * A_nodes[i] + S_R[i] * (Q_R_star[:, i] - Q_R[:, i]) * A_nodes[i]
        elif S_R[i] < 0:
            # Right-going wave dominates
            F[:, i] = F_R[:, i] * A_nodes[i]

    return F

def F_LaxFriedrichs(Q_L, Q_R, parameters):
    p = parameters['p']  # Pressure, pre-computed before calling this function
    p_L = p[:-1]
    p_R = p[1:]

    # Load parameters
    dx = parameters['dx']
    dt = parameters['dt']
    A_nodes = parameters['A_nodes']  # A_nodes should be a 1xNj vector for the area at each node
    
    # Compute the left and right fluxes
    F_L = parameters['flux_function'](Q_L, p_L)
    F_R = parameters['flux_function'](Q_R, p_R)
    
    # Compute the Lax-Friedrichs numerical flux
    F = 0.5 * (F_L * A_nodes[:-1] + F_R * A_nodes[1:] - dx / dt * (Q_R * A_nodes[1:] - Q_L * A_nodes[:-1]))
    
    return F

def F_HLL(Q_L, Q_R, parameters):

    # Extract pressure values for left and right states
    p = parameters['p']  # Pre-computed pressure before calling this function
    p_L = p[:-1]
    p_R = p[1:]

    # Compute the left and right fluxes
    F_L = parameters['flux_function'](Q_L, p_L)
    F_R = parameters['flux_function'](Q_R, p_R)
    A_nodes = parameters['A_nodes']  # A_nodes should be a 1xNj vector for the area at each node

    # Compute eigenvalues (wave speeds) for the left and right states
    eig_L = parameters['eigenvalues_function'](Q_L)
    eig_R = parameters['eigenvalues_function'](Q_R)

    # Determine the left and right wave speeds
    S_L = np.minimum(eig_L[0, :], eig_L[2, :])  # Minimum left wave speed (element-wise)
    S_R = np.maximum(eig_R[0, :], eig_R[2, :])  # Maximum right wave speed (element-wise)

    # Compute density and velocity for left and right states
    rho_L = Q_L[0, :]
    rho_R = Q_R[0, :]
    u_L = Q_L[1, :] / rho_L
    u_R = Q_R[1, :] / rho_R

    # Preallocate flux matrix
    F = np.zeros_like(F_L)

    # HLL Flux cases (applied element-wise)
    for i in range(len(S_L)):
        if S_L[i] >= 0:
            # If S_L is positive, use the left state flux
            F[:, i] = F_L[:, i] * A_nodes[i]
        elif S_R[i] <= 0:
            # If S_R is negative, use the right state flux
            F[:, i] = F_R[:, i] * A_nodes[i + 1]
        else:
            # Intermediate case: use HLL formula for flux averaging
            F[:, i] = (S_R[i] * F_L[:, i] * A_nodes[i] - S_L[i] * F_R[:, i] * A_nodes[i + 1] +
                       S_L[i] * S_R[i] * (Q_R[:, i] * A_nodes[i + 1] - Q_L[:, i] * A_nodes[i])) / (S_R[i] - S_L[i])

    return F
