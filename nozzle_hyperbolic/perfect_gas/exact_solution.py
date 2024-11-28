import numpy as np
from scipy.optimize import fsolve, brentq, newton

def areaMachRelation(M, area_ratio, gamma):
    return (1/M) * ((2/(gamma + 1)) * (1 + (gamma - 1) / 2 * M**2))**((gamma + 1) / (2 * (gamma - 1))) - area_ratio

def areaMachDerivative(M, gamma):
    term1 = -1/M**2
    term2 = (1 + (gamma - 1) / 2 * M**2)
    term3 = (gamma + 1) / (gamma - 1)
    return term1 * ((2 / (gamma + 1)) * term2)**(term3 / 2)

def nozzle_exact2(x, parameters, total_bc):

    tol = 1e-5
    max_iter1 = 10000

    # Preallocate arrays for properties
    T = np.zeros((len(x), 3))
    p = np.zeros((len(x), 3))
    u = np.zeros((len(x), 3))
    rho = np.zeros((len(x), 3))
    
    T0 = total_bc[0]
    p0 = total_bc[1]
    M = np.zeros((len(x), 3))  # First column subsonic, second supersonic

    gamma = parameters['gamma']
    R = 287
    p_out = parameters['p_out']
    rx = parameters['r_function']
    A = lambda x: rx(x)**2 * np.pi
    A_nodes = A(x)
    Ac = np.min(A_nodes)
    area_ratio = A_nodes / Ac
    A_cr_index = np.argmin(A_nodes)

    Kc = 0.5 * (2 / (gamma + 1))**((gamma + 1) / (gamma - 1)) * (gamma - 1)

    cp = gamma * R / (gamma - 1)
    cv = cp - R
    dx = 0.001

    # Find subsonic and supersonic Mach numbers
    for i in range(len(x)):
        # Function handle for the area-Mach relationship
        f = lambda M: areaMachRelation(M, area_ratio[i], gamma)
        # Derivative of the area-Mach relationship
        f_prime = lambda M: areaMachDerivative(M, gamma)

        # Solve for Mach number with both subsonic and supersonic initial guesses
        M[i, 0] = fsolve(f, 0.1, xtol=tol, maxfev=max_iter1)[0]  # Subsonic guess
        M[i, 1] = fsolve(f, 8, xtol=tol, maxfev=max_iter1)[0]
  
        # Subsonic solution
        T[i, 0] = T0 / (1 + (gamma - 1) / 2 * M[i, 0]**2)
        p[i, 0] = p0 / (1 + (gamma - 1) / 2 * M[i, 0]**2)**(gamma / (gamma - 1))
        a_sub = np.sqrt(gamma * R * T[i, 0])
        u[i, 0] = M[i, 0] * a_sub
        rho[i, 0] = p[i, 0] / (R * T[i, 0])

        # Supersonic solution
        T[i, 1] = T0 / (1 + (gamma - 1) / 2 * M[i, 1]**2)
        p[i, 1] = p0 / (1 + (gamma - 1) / 2 * M[i, 1]**2)**(gamma / (gamma - 1))
        a_sup = np.sqrt(gamma * R * T[i, 1])
        u[i, 1] = M[i, 1] * a_sup
        rho[i, 1] = p[i, 1] / (R * T[i, 1])

    x_tent = x[A_cr_index+10]
    i_shock = None

    if (p[-1, 1] < p_out) and (p[-1, 0] > p_out):
        tolp = 1e-5
        res = 1
        max_iter = 1000
        iter = 0

        while res > tolp and iter < max_iter:
            f1 = lambda T1: T1/T0 - (Kc * Ac**2 / A(x_tent)**2 * (1 - T1/T0)**(-1))**((gamma - 1) / 2)
            T1_sol = fsolve(f1, T[A_cr_index, 1],xtol=tol, maxfev=max_iter1)[0]

            v1 = np.sqrt(2 * cp * (T0 - T1_sol))
            p1 = p0 * (T1_sol / T0)**(gamma / (gamma - 1))
            Ma1 = v1 / np.sqrt(gamma * R * T1_sol)

            p2 = p1 * (2 * gamma / (gamma + 1) * Ma1**2 - (gamma - 1) / (gamma + 1))
            Ma2 = np.sqrt((Ma1**2 + 2 / (gamma - 1)) / (2 * gamma * Ma1**2 / (gamma - 1) - 1))
            p20 = p2 * (1 + 0.5 * (gamma - 1) * Ma2**2)**(gamma / (gamma - 1))

            f2 = lambda Tu: 1 - (Tu / T0) - (p0 / p20)**2 * (Ac / A_nodes[-1])**2 * Kc * (Tu / T0)**(-2 / (gamma - 1))
            Tu_sol = fsolve(f2, T[A_cr_index, 1],xtol=tol, maxfev=max_iter1)[0]

            pu = p20 * (Tu_sol / T0)**(gamma / (gamma - 1))
            res = abs(pu - p_out) / p_out

            if pu > p_out:
                x_tent += dx
            elif pu < p_out:
                x_tent -= dx

            iter += 1

        # Find the index of the shock location
        i_shock = np.argmin(abs(x - x_tent))
        A_shock = A(x[i_shock])
        Ac_effective = A_shock / ((1 / Ma2) * (2 * (1 + 0.5 * (gamma - 1) * Ma2**2) / (gamma + 1))**((gamma + 1) / (2 * (gamma - 1))))
        area_ratio_effective = A_nodes / Ac_effective

        T[:, 2] = T[:, 1]
        rho[:, 2] = rho[:, 1]
        p[:, 2] = p[:, 1]
        u[:, 2] = u[:, 1]

        for i in range(i_shock, len(x)):
            f = lambda M: areaMachRelation(M, area_ratio_effective[i], gamma)
            f_prime = lambda M: areaMachDerivative(M, gamma)
            M[i, 2] = fsolve(f, 0.1, xtol=tol, maxfev=max_iter1)[0]
            # M[i, 2] = newton(f, 0.2, fprime=f_prime, tol=tol, maxiter=max_iter1)  # Subsonic guess
            p[i, 2] = p20 * (1 + 0.5 * (gamma - 1) * M[i, 2]**2)**(-gamma / (gamma - 1))
            T[i, 2] = T0 * (1 + 0.5 * (gamma - 1) * M[i, 2]**2)**(-1)
            rho[i, 2] = p[i, 2] / (R * T[i, 2])
            u[i, 2] = M[i, 2] * np.sqrt(gamma * R * T[i, 2])

    return M, p, u, rho, i_shock

