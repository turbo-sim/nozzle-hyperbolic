import numpy as np
import barotropy as bpy

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
    state_L = fluid.get_state(bpy.DmassP_INPUTS, rho_L, p_L)
    c_L = state_L.speed_sound

    # Compute the right state
    state_R = fluid.get_state(bpy.DmassP_INPUTS, rho_R, p_R)
    c_R = state_R.speed_sound


    # Compute the Roe averages (keep these lines unchanged)
    sqrt_rho_L = np.sqrt(rho_L)
    sqrt_rho_R = np.sqrt(rho_R)
    u = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) / (sqrt_rho_L + sqrt_rho_R)
    H = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) / (sqrt_rho_L + sqrt_rho_R)

    # Compute averaged state for speed of sound
    avg_rho = (rho_L + rho_R) / 2  # Simple average for density
    avg_p = (p_L + p_R) / 2        # Simple average for pressure
    state = fluid.get_state(bpy.DmassP_INPUTS, avg_rho, avg_p)
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
    fluid = bpy.Fluid(name=fluid_name, backend="HEOS")

    energy_L_star = np.zeros(Q_L.shape[1])
    energy_R_star = np.zeros(Q_R.shape[1])

    for j in range(Q_L.shape[1]):

        state_L = fluid.get_state(bpy.DmassP_INPUTS, rho_L_star[j], p_L_star[j])
        energy_L_star[j] = state_L.umass 
        state_R = fluid.get_state(bpy.DmassP_INPUTS, rho_R_star[j], p_R_star[j])
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


## Flux schemes for perfect gas

def F_Rusanov_perfect(Q_L, Q_R, parameters):
    # Compute the left and right fluxes
    F_L = parameters['flux_function'](Q_L)
    F_R = parameters['flux_function'](Q_R)
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

def F_Roe_perfect(Q_L, Q_R, parameters):
    F_L = parameters['flux_function'](Q_L)
    F_R = parameters['flux_function'](Q_R)
    F = np.zeros_like(F_L)
    A_nodes = parameters['A_nodes']

    for i in range(Q_L.shape[1]):
        A = parameters['roe_function'](Q_L[:, i], Q_R[:, i])
        F[:, i] = 0.5 * (F_L[:, i] * A_nodes[i] + F_R[:, i] * A_nodes[i + 1] - A @ (Q_R[:, i] * A_nodes[i + 1] - Q_L[:, i] * A_nodes[i]))
    return F


def euler_roe_matrix_perfect(Q_L, Q_R, gamma):
    # Compute left state variables
    rho_L, rhou_L, rhoE_L = Q_L
    u_L = rhou_L / rho_L
    p_L = (gamma - 1) * (rhoE_L - 0.5 * rhou_L**2 / rho_L)
    H_L = (rhoE_L + p_L) / rho_L
    c_L = np.sqrt(gamma * p_L / rho_L)

    # Compute right state variables
    rho_R, rhou_R, rhoE_R = Q_R
    u_R = rhou_R / rho_R
    p_R = (gamma - 1) * (rhoE_R - 0.5 * rhou_R**2 / rho_R)
    H_R = (rhoE_R + p_R) / rho_R
    c_R = np.sqrt(gamma * p_R / rho_R)

    # Compute the Roe averages
    sqrt_rho_L = np.sqrt(rho_L)
    sqrt_rho_R = np.sqrt(rho_R)
    sqrt_rho_sum = sqrt_rho_L + sqrt_rho_R

    u = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) / sqrt_rho_sum
    H = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) / sqrt_rho_sum
    c = np.sqrt((gamma - 1) * (H - 0.5 * u**2))

    # Eigenvector matrix R
    R = np.array([
        [1, 1, 1],
        [u - c, u, u + c],
        [H - c * u, 0.5 * u**2, H + c * u]
    ])

    # Eigenvalues with entropy fix
    eig1_L = u_L - c_L
    eig1_R = u_R - c_R
    eig3_L = u_L + c_L
    eig3_R = u_R + c_R
    L_diag = [H_fix_perfect(eig1_L, eig1_R, u - c, c),
              u,
              H_fix_perfect(eig3_L, eig3_R, u + c, c)]
    L = np.diag(L_diag)

    # Compute Roe's matrix A by solving R * A = L * R
    A = R @ L @ np.linalg.inv(R)

    return A

def H_fix_perfect(a_L, a_R, a, c):
    a_abs = abs(a)
    delta = max(abs(a_R - a_L), 0.1 * c)  # threshold for entropy fix
    
    if a_abs >= delta:
        return a_abs
    else:
        return (a_abs**2 + delta**2) / (2 * delta)

def F_HLLC_perfect(Q_L, Q_R, parameters):
    gamma = parameters['gamma']
    
    # Compute the left and right fluxes
    F_L = parameters['flux_function'](Q_L)
    F_R = parameters['flux_function'](Q_R)
    A_nodes = parameters['A_nodes']  # A_nodes should be a 1xNj vector for the area at each node
    
    # Compute eigenvalues (wave speeds) for the left and right states
    eig_L = parameters['eigenvalues_function'](Q_L)
    eig_R = parameters['eigenvalues_function'](Q_R)
    
    S_L = np.minimum(eig_L[0, :], eig_L[2, :])  # Minimum left wave speed (element-wise)
    S_R = np.maximum(eig_R[0, :], eig_R[2, :])  # Maximum right wave speed (element-wise)
    
    rho_L = Q_L[0, :]
    rho_R = Q_R[0, :]
    u_L = Q_L[1, :] / Q_L[0, :]  # Element-wise division
    u_R = Q_R[1, :] / Q_R[0, :]  # Element-wise division
    p_L = (gamma - 1) * (Q_L[2, :] - 0.5 * Q_L[1, :]**2 / Q_L[0, :])  # Element-wise operations
    p_R = (gamma - 1) * (Q_R[2, :] - 0.5 * Q_R[1, :]**2 / Q_R[0, :])  # Element-wise operations
    
    # Compute the contact wave speed (S_M) (element-wise)
    S_M = (p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)) / \
          (rho_L * (S_L - u_L) - rho_R * (S_R - u_R))
    
    # Compute intermediate states
    rho_L_star = rho_L * (S_L - u_L) / (S_L - S_M)  # Element-wise division
    u_L_star = S_M
    p_L_star = p_L + rho_L * (S_L - u_L) * (S_M - u_L)  # Element-wise multiplication
    Q_L_star = np.array([rho_L_star, rho_L_star * u_L_star, p_L_star / (gamma - 1) + 0.5 * rho_L_star * u_L_star**2])  # Element-wise
    
    rho_R_star = rho_R * (S_R - u_R) / (S_R - S_M)  # Element-wise division
    u_R_star = S_M
    p_R_star = p_R + rho_R * (S_R - u_R) * (S_M - u_R)  # Element-wise multiplication
    Q_R_star = np.array([rho_R_star, rho_R_star * u_R_star, p_R_star / (gamma - 1) + 0.5 * rho_R_star * u_R_star**2])  # Element-wise
    
    # Ensure A_nodes matches the dimensions of the fluxes
    A_left = A_nodes[:-1]  # Area at the left of the interfaces
    A_right = A_nodes[1:]  # Area at the right of the interfaces
    
    # Preallocate flux matrix
    F = np.zeros_like(F_L)  # This ensures F is assigned a default value
    
    # HLLC Flux cases (applied element-wise)
    for i in range(len(S_L)):
        if S_L[i] > 0:
            F[:, i] = F_L[:, i] * A_left[i]  # Left-going wave dominates (use left flux)
        elif S_L[i] <= 0 <= S_M[i]:
            F[:, i] = F_L[:, i] * A_left[i] + S_L[i] * (Q_L_star[:, i] - Q_L[:, i]) * A_left[i]  # Corrected left flux
        elif S_M[i] <= 0 <= S_R[i]:
            F[:, i] = F_R[:, i] * A_right[i] + S_R[i] * (Q_R_star[:, i] - Q_R[:, i]) * A_right[i]  # Corrected right flux
        elif S_R[i] < 0:
            F[:, i] = F_R[:, i] * A_right[i]  # Right-going wave dominates (use right flux)
    
    return F


def F_LaxFriedrichs_perfect(Q_L, Q_R, parameters):
    # Load parameters
    dx = parameters['dx']
    dt = parameters['dt']
    A_nodes = parameters['A_nodes']  # A_nodes should be a 1xNj vector for the area at each node
    
    # Compute the left and right fluxes
    F_L = parameters['flux_function'](Q_L)
    F_R = parameters['flux_function'](Q_R)
    
    # Compute the Lax-Friedrichs numerical flux
    F = 0.5 * (F_L * A_nodes[:-1] + F_R * A_nodes[1:] - dx / dt * (Q_R * A_nodes[1:] - Q_L * A_nodes[:-1]))
    
    return F

def F_HLL_perfect(Q_L, Q_R, parameters):
    gamma = parameters['gamma']
    A_nodes = parameters['A_nodes']  # A_nodes should be a 1xNj vector for the area at each node
    
    # Compute the left and right fluxes
    F_L = parameters['flux_function'](Q_L)
    F_R = parameters['flux_function'](Q_R)
    
    # Extract variables for left and right states
    rho_L, rho_R = Q_L[0, :], Q_R[0, :]
    u_L, u_R = Q_L[1, :] / rho_L, Q_R[1, :] / rho_R
    p_L = (gamma - 1) * (Q_L[2, :] - 0.5 * rho_L * u_L**2)
    p_R = (gamma - 1) * (Q_R[2, :] - 0.5 * rho_R * u_R**2)
    c_L = np.sqrt(gamma * p_L / rho_L)
    c_R = np.sqrt(gamma * p_R / rho_R)
    
    # Calculate left and right wave speeds (S_L and S_R)
    S_L = np.minimum(u_L - c_L, u_R - c_R)
    S_R = np.maximum(u_L + c_L, u_R + c_R)
    
    # Preallocate flux matrix
    F = np.zeros_like(F_L)
    
    # Compute HLL flux
    for i in range(len(S_L)):
        if S_L[i] >= 0:
            # If the flow is moving to the right, take the left state flux
            F[:, i] = F_L[:, i] * A_nodes[i]
        elif S_R[i] <= 0:
            # If the flow is moving to the left, take the right state flux
            F[:, i] = F_R[:, i] * A_nodes[i + 1]
        else:
            # In between, use the HLL formula to average the flux
            F[:, i] = (S_R[i] * F_L[:, i] * A_nodes[i] - S_L[i] * F_R[:, i] * A_nodes[i + 1] +
                       S_L[i] * S_R[i] * (Q_R[:, i] * A_nodes[i + 1] - Q_L[:, i] * A_nodes[i])) / (S_R[i] - S_L[i])
    
    return F

def F_HLLE_perfect(Q_L, Q_R, parameters):
    gamma = parameters['gamma']
    A_nodes = parameters['A_nodes']  # Area at each node interface
    
    # Compute left and right fluxes
    F_L = parameters['flux_function'](Q_L)
    F_R = parameters['flux_function'](Q_R)
    
    # Extract variables for left and right states
    rho_L, rho_R = Q_L[0, :], Q_R[0, :]
    u_L, u_R = Q_L[1, :] / rho_L, Q_R[1, :] / rho_R
    p_L = (gamma - 1) * (Q_L[2, :] - 0.5 * rho_L * u_L**2)
    p_R = (gamma - 1) * (Q_R[2, :] - 0.5 * rho_R * u_R**2)
    c_L = np.sqrt(gamma * p_L / rho_L)
    c_R = np.sqrt(gamma * p_R / rho_R)
    
    # Calculate the wave speeds
    S_L = np.minimum(u_L - c_L, u_R - c_R)  # Minimum left wave speed
    S_R = np.maximum(u_L + c_L, u_R + c_R)  # Maximum right wave speed

    # Preallocate flux matrix
    F = np.zeros_like(F_L)

    # Compute HLLE flux
    for i in range(len(S_L)):
        if S_L[i] >= 0:
            # Flow moves right, use left state flux
            F[:, i] = F_L[:, i] * A_nodes[i]
        elif S_R[i] <= 0:
            # Flow moves left, use right state flux
            F[:, i] = F_R[:, i] * A_nodes[i + 1]
        else:
            # In between, apply HLLE average flux formula
            F[:, i] = (S_R[i] * F_L[:, i] * A_nodes[i] - S_L[i] * F_R[:, i] * A_nodes[i + 1] +
                       S_L[i] * S_R[i] * (Q_R[:, i] * A_nodes[i + 1] - Q_L[:, i] * A_nodes[i])) / (S_R[i] - S_L[i])

    return F


def F_HLLD_perfect(Q_L, Q_R, parameters):
    gamma = parameters['gamma']
    
    # Compute the left and right fluxes
    F_L = parameters['flux_function'](Q_L)
    F_R = parameters['flux_function'](Q_R)
    A_nodes = parameters['A_nodes']  # A_nodes should be a 1xNj vector for the area at each node
    
    # Compute eigenvalues (wave speeds) for the left and right states
    eig_L = parameters['eigenvalues_function'](Q_L)
    eig_R = parameters['eigenvalues_function'](Q_R)
    
    S_L = np.minimum(eig_L[0, :], eig_L[2, :])  # Minimum left wave speed
    S_R = np.maximum(eig_R[0, :], eig_R[2, :])  # Maximum right wave speed
    
    # Compute primary variables
    rho_L, rho_R = Q_L[0, :], Q_R[0, :]
    u_L, u_R = Q_L[1, :] / rho_L, Q_R[1, :] / rho_R  # Velocity
    p_L = (gamma - 1) * (Q_L[2, :] - 0.5 * rho_L * u_L**2)
    p_R = (gamma - 1) * (Q_R[2, :] - 0.5 * rho_R * u_R**2)
    
    # Compute intermediate (contact) wave speed S_M
    S_M = (p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)) / \
          (rho_L * (S_L - u_L) - rho_R * (S_R - u_R))
    
    # Compute additional slow wave speeds S_L_star and S_R_star (for HLLD-specific states)
    a_L = np.sqrt(gamma * p_L / rho_L)  # Speed of sound for left state
    a_R = np.sqrt(gamma * p_R / rho_R)  # Speed of sound for right state
    
    S_L_star = S_M - a_L  # Left-going slow wave speed
    S_R_star = S_M + a_R  # Right-going slow wave speed
    
    # Define intermediate states Q_L_star and Q_R_star (left and right star states)
    rho_L_star = rho_L * (S_L - u_L) / (S_L - S_M)
    rho_R_star = rho_R * (S_R - u_R) / (S_R - S_M)
    u_star = S_M
    p_L_star = p_L + rho_L * (S_L - u_L) * (S_M - u_L)
    p_R_star = p_R + rho_R * (S_R - u_R) * (S_M - u_R)
    
    Q_L_star = np.array([
        rho_L_star,
        rho_L_star * u_star,
        p_L_star / (gamma - 1) + 0.5 * rho_L_star * u_star**2
    ])
    Q_R_star = np.array([
        rho_R_star,
        rho_R_star * u_star,
        p_R_star / (gamma - 1) + 0.5 * rho_R_star * u_star**2
    ])
    
    # Define second-level star states Q_L_star_star and Q_R_star_star
    rho_L_star_star = rho_L_star * (S_L_star - u_star) / (S_L_star - S_M)
    rho_R_star_star = rho_R_star * (S_R_star - u_star) / (S_R_star - S_M)
    u_L_star_star = S_M
    u_R_star_star = S_M
    
    Q_L_star_star = np.array([
        rho_L_star_star,
        rho_L_star_star * u_L_star_star,
        p_L_star / (gamma - 1) + 0.5 * rho_L_star_star * u_L_star_star**2
    ])
    Q_R_star_star = np.array([
        rho_R_star_star,
        rho_R_star_star * u_R_star_star,
        p_R_star / (gamma - 1) + 0.5 * rho_R_star_star * u_R_star_star**2
    ])
    
    # Ensure areas are dimensionally consistent
    A_left = A_nodes[:-1]
    A_right = A_nodes[1:]
    
    # Initialize flux matrix
    F = np.zeros_like(F_L)
    
    # HLLD Flux cases based on wave speeds
    for i in range(len(S_L)):
        if S_L[i] > 0:
            # Left-going wave dominates
            F[:, i] = F_L[:, i] * A_left[i]
        elif S_L[i] <= 0 <= S_L_star[i]:
            # Left intermediate wave (first star state)
            F[:, i] = F_L[:, i] * A_left[i] + S_L[i] * (Q_L_star[:, i] - Q_L[:, i]) * A_left[i]
        elif S_L_star[i] <= 0 <= S_M[i]:
            # Second star state on the left
            F[:, i] = F_L[:, i] * A_left[i] + S_L_star[i] * (Q_L_star_star[:, i] - Q_L_star[:, i]) * A_left[i]
        elif S_M[i] <= 0 <= S_R_star[i]:
            # Second star state on the right
            F[:, i] = F_R[:, i] * A_right[i] + S_R_star[i] * (Q_R_star_star[:, i] - Q_R_star[:, i]) * A_right[i]
        elif S_R_star[i] <= 0 <= S_R[i]:
            # Right intermediate wave (first star state on the right)
            F[:, i] = F_R[:, i] * A_right[i] + S_R[i] * (Q_R_star[:, i] - Q_R[:, i]) * A_right[i]
        else:
            # Right-going wave dominates
            F[:, i] = F_R[:, i] * A_right[i]

    return F
