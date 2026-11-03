"""Independent classical perfect-gas verification on the archived wall contour.

These host calculations are comparison data only. They never initialize FVM.
The area-Mach and normal-shock formulas reuse the production reference module,
which has no dependency on the finite-volume residual or numerical flux.
"""

import numpy as np
from scipy.optimize import brentq

from nozzle_hyperbolic.models.euler_nozzle.analytical.perfect_gas import (
    area_mach,
    mach_from_area_ratio,
    normal_shock,
    pressure_ratio,
)


def profile(case, geometry, x):
    gamma, gas_constant = 1.4, 287.0
    p0, t0 = case["p0_Pa"], case["T0_K"]
    cp = gamma * gas_constant / (gamma - 1)
    # Polynomial extrema determine the physical throat independently of cells.
    roots = geometry.profile.deriv().roots()
    candidates = [geometry.x_min, geometry.x_max] + [
        float(z.real)
        for z in roots
        if abs(z.imag) < 1e-10 and geometry.x_min < z.real < geometry.x_max
    ]
    throat = min(candidates, key=lambda z: float(geometry.sample(np.array(z))["area"]))
    area = lambda z: np.asarray(geometry.sample(z)["area"])
    at = float(area(throat))
    ae = float(area(geometry.x_max))
    me_sub = mach_from_area_ratio(ae / at, gamma, "sub")
    me_sup = mach_from_area_ratio(ae / at, gamma, "sup")
    critical = float(pressure_ratio(me_sub, gamma))
    adapted = float(pressure_ratio(me_sup, gamma))
    shock_exit = adapted * float(normal_shock(me_sup, gamma)[0])
    target = case.get("p_out_Pa")
    ratio = adapted if target is None else target / p0

    def shock_data(xs):
        ass = float(area(xs))
        m1 = mach_from_area_ratio(ass / at, gamma, "sup")
        _, m2, total_ratio = normal_shock(m1, gamma)
        downstream_astar = ass / area_mach(m2, gamma)
        exit_mach = mach_from_area_ratio(ae / downstream_astar, gamma, "sub")
        return (
            float(total_ratio * pressure_ratio(exit_mach, gamma)),
            float(downstream_astar),
            float(total_ratio),
        )

    x = np.asarray(x)
    areas = area(x)
    shock_x = None
    total_p = np.full_like(x, p0)
    if ratio >= critical:
        regime = "subsonic"
        exit_mach = np.sqrt(2 / (gamma - 1) * (ratio ** (-(gamma - 1) / gamma) - 1))
        astar = ae / area_mach(exit_mach, gamma)
        mach = np.array(
            [mach_from_area_ratio(max(1.0, a / astar), gamma, "sub") for a in areas]
        )
    else:
        regime = "adapted_supersonic"
        mach = np.array(
            [
                mach_from_area_ratio(
                    max(1.0, a / at), gamma, "sub" if xx <= throat else "sup"
                )
                for a, xx in zip(areas, x)
            ]
        )
        if ratio > shock_exit:
            shock_x = brentq(
                lambda z: shock_data(z)[0] - ratio, throat + 1e-10, geometry.x_max
            )
            _, astar2, total_ratio = shock_data(shock_x)
            for i in np.flatnonzero(x > shock_x):
                mach[i] = mach_from_area_ratio(
                    max(1.0, areas[i] / astar2), gamma, "sub"
                )
            total_p[x > shock_x] *= total_ratio
            regime = "internal_shock"
    temperature = t0 / (1 + 0.5 * (gamma - 1) * mach**2)
    pressure = total_p * pressure_ratio(mach, gamma)
    density = pressure / (gas_constant * temperature)
    velocity = mach * np.sqrt(gamma * gas_constant * temperature)
    choked = (
        at
        * p0
        * np.sqrt(gamma / (gas_constant * t0))
        * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
    )
    return dict(
        label="Independent area–Mach / normal-shock reference",
        kind="analytical",
        x_m=x,
        p_Pa=pressure,
        M=mach,
        u=velocity,
        T=temperature,
        rho=density,
        h=cp * temperature,
        mdot=density * velocity * areas,
        regime=regime,
        shock_x_m=shock_x,
        throat_x_m=throat,
        throat_area_m2=at,
        area_ratio=ae / at,
        choked_mass_flow_kg_s=choked,
        assumptions="Inviscid adiabatic calorically perfect gas; exact polynomial contour; pointwise samples, not shock-containing cell averages",
    )
