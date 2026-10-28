"""Classical perfect-gas operating map; never used to initialize the solver.

The shared baseline supplies the nozzle geometry and reservoir conditions.
These algebraic area-Mach and normal-shock relations extend its references
to arbitrary imposed backpressures, including external adjustment regimes.
"""

import numpy as np
from scipy.optimize import brentq

from development.roberto.euler_nozzle import verify_euler_nozzle_steady as baseline
from nozzle_hyperbolic.geometry.duct import get_symmetric_nozzle


class PerfectGasReference:
    def __init__(self, fluid):
        self.gamma = float(fluid.constants.gamma)
        self.R = float(fluid.constants.R)
        self.cp = self.gamma * self.R / (self.gamma - 1.0)
        self.p0, self.T0 = baseline.P_STAGNATION, baseline.T_STAGNATION
        self.At = baseline.NOZZLE['A_throat']
        self.Ae = baseline.NOZZLE['A_inlet']
        self.length = baseline.NOZZLE['L']
        self.M_sub = self.mach(self.Ae / self.At, 'sub')
        self.M_sup = self.mach(self.Ae / self.At, 'sup')
        self.critical = self.pressure_ratio(self.M_sub)
        self.adapted = self.pressure_ratio(self.M_sup)
        self.shock_exit = self.adapted * self.shock(self.M_sup)[0]
        self.mdot_star = (self.At * self.p0 * np.sqrt(self.gamma / (self.R * self.T0))
                          * (2 / (self.gamma + 1)) ** ((self.gamma + 1) / (2 * (self.gamma - 1))))

    def area_mach(self, mach):
        g = self.gamma
        return (2 / (g + 1) * (1 + (g - 1) / 2 * mach**2)) ** ((g + 1) / (2 * (g - 1))) / mach

    def mach(self, ratio, branch):
        if abs(ratio - 1.0) < 2e-12:
            return 1.0
        bounds = (1e-10, 1.0) if branch == 'sub' else (1.0, 30.0)
        return brentq(lambda m: self.area_mach(m) - ratio, *bounds, xtol=1e-13)

    def pressure_ratio(self, mach):
        return (1 + (self.gamma - 1) / 2 * mach**2) ** (-self.gamma / (self.gamma - 1))

    def shock(self, m1):
        """Return p2/p1, M2, and p02/p01 from normal-shock conservation."""
        g = self.gamma
        p2p1 = 1 + 2 * g / (g + 1) * (m1**2 - 1)
        m2 = np.sqrt((1 + (g - 1) / 2 * m1**2) / (g * m1**2 - (g - 1) / 2))
        return p2p1, m2, self.pressure_ratio(m1) * p2p1 / self.pressure_ratio(m2)

    def shock_data(self, position):
        area = float(get_symmetric_nozzle(np.array([position]), **baseline.NOZZLE).A[0])
        m1 = self.mach(area / self.At, 'sup')
        _, m2, total_ratio = self.shock(m1)
        astar2 = area / self.area_mach(m2)
        me = self.mach(self.Ae / astar2, 'sub')
        return total_ratio * self.pressure_ratio(me), astar2, total_ratio

    def profile(self, x, back_ratio):
        """Return the nozzle-interior reference, excluding the exterior jet.

        At a shock exactly on x=L, values on the plane are upstream limits.
        Its downstream jump is returned separately. This convention matters
        when interpreting entropy generation in a finite interior domain.
        """
        x = np.asarray(x)
        area = np.asarray(get_symmetric_nozzle(x, **baseline.NOZZLE).A)
        r = float(back_ratio)
        if abs(r - self.critical) < 1e-12:
            r = self.critical
        shock_x, total_ratio, astar2 = None, 1.0, self.At
        if np.isclose(r, 1.0, atol=1e-14, rtol=0):
            regime, mach = 'stagnant', np.zeros_like(x)
        elif r >= self.critical:
            regime = 'critical_sonic' if abs(r - self.critical) < 1e-12 else 'subsonic'
            me = np.sqrt(2 / (self.gamma - 1) * (r ** (-(self.gamma - 1) / self.gamma) - 1))
            astar = self.Ae / self.area_mach(me)
            mach = np.array([self.mach(a / astar, 'sub') for a in area])
        else:
            if r > self.shock_exit + 1e-12:
                regime = 'internal_shock'
                shock_x = brentq(lambda xs: self.shock_data(xs)[0] - r,
                                  self.length / 2, self.length, xtol=1e-13)
                _, astar2, total_ratio = self.shock_data(shock_x)
            elif abs(r - self.shock_exit) <= 1e-12:
                regime, shock_x = 'shock_at_exit', self.length
                _, astar2, total_ratio = self.shock_data(shock_x)
            elif r > self.adapted + 1e-12:
                regime = 'overexpanded_external'
            elif abs(r - self.adapted) <= 1e-12:
                regime = 'adapted'
            else:
                regime = 'underexpanded_external'
            mach = np.array([self.mach(a / self.At, 'sub' if xx <= self.length / 2 else 'sup')
                             for xx, a in zip(x, area)])
            if shock_x is not None:
                for i in np.flatnonzero(x > shock_x):
                    mach[i] = self.mach(area[i] / astar2, 'sub')
        total_p = np.full_like(x, self.p0)
        if shock_x is not None:
            total_p[x > shock_x] *= total_ratio
        temp = self.T0 / (1 + (self.gamma - 1) / 2 * mach**2)
        pressure = total_p * self.pressure_ratio(mach)
        density = pressure / (self.R * temp)
        speed = mach * np.sqrt(self.gamma * self.R * temp)
        ds = -self.R * np.log(total_p / self.p0)
        return dict(x=x, p=pressure, T=temp, rho=density, u=speed, M=mach,
                    ds=ds, p_total=total_p, mdot=density * speed * area,
                    regime=regime, shock_x=shock_x,
                    shock_ds=-self.R * np.log(total_ratio),
                    downstream_exit_pressure=self.p0 * r if regime == 'shock_at_exit' else None)

    def landmarks(self):
        return dict(critical_ratio=self.critical, shock_exit_ratio=self.shock_exit,
                    adapted_ratio=self.adapted, choked_mdot_kg_s=self.mdot_star,
                    subsonic_exit_M=self.M_sub, supersonic_exit_M=self.M_sup,
                    gamma=self.gamma, gas_constant_J_kg_K=self.R)
