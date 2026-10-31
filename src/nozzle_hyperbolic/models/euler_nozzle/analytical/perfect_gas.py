"""Steady, inviscid, adiabatic nozzle references for a calorically perfect gas.

These host-side verification utilities are independent of the FVM operator.
Their perfect-gas assumptions do not constrain the general-EOS numerical model.
All quantities use SI units; ``gamma`` and ``R`` must be constant.

Sources
-------
The area-Mach, isentropic and normal-shock relations are the classical formulas
collected in NACA Report 1135, *Equations, Tables, and Charts for Compressible
Flow* (Ames Research Staff, 1953):
https://www.nasa.gov/wp-content/uploads/2023/03/equations-tables-charts-compressibleflow-report-1135.pdf
NASA Glenn also gives the area-Mach and normal-shock equations directly:
https://www.grc.nasa.gov/www/k-12/BGP/astar.html
https://www.grc.nasa.gov/WWW/k-12/airplane/normal.html

The backpressure operating-map assembly and choice of one-sided values at a
shock are project implementations of these relations, not a separate published
formula. The exterior jet is never represented.
"""

import numpy as np
from scipy.optimize import brentq

from nozzle_hyperbolic.geometry.duct import get_symmetric_nozzle


def _perfect_gas_constants(fluid, p0, T0):
    """Validate explicit gas constants and, when available, two EOS samples.

    Density, enthalpy differences (independent of datum), and sound speed are
    checked at (p0,T0) and (0.9*p0,0.95*T0). Two finite samples can reject an
    incompatible EOS; they do not prove ideal-gas behavior throughout a nozzle.
    Constants-only objects are an explicit ideal-gas specification and remain
    usable without an EOS implementation.
    """
    gamma, gas_constant = float(fluid.constants.gamma), float(fluid.constants.R)
    if (
        not np.all(np.isfinite([gamma, gas_constant, p0, T0]))
        or gamma <= 1
        or gas_constant <= 0
        or p0 <= 0
        or T0 <= 0
    ):
        raise ValueError(
            "Need finite gamma>1, R>0, p0>0 and T0>0 for a perfect-gas reference"
        )
    if callable(getattr(fluid, "get_state", None)):
        from jaxprop import PT_INPUTS

        cp = gamma * gas_constant / (gamma - 1)
        enthalpies = []
        for pressure, temperature in ((p0, T0), (0.9 * p0, 0.95 * T0)):
            state = fluid.get_state(PT_INPUTS, pressure, temperature)
            density, enthalpy = float(state["rho"]), float(state["h"])
            if not np.isfinite(enthalpy) or not np.isclose(
                density, pressure / (gas_constant * temperature), rtol=2e-6, atol=0
            ):
                raise ValueError(
                    "The supplied EOS is inconsistent with the perfect-gas reference"
                )
            try:
                sound_speed = float(state["a"])
            except KeyError:
                sound_speed = None
            if sound_speed is not None and (
                sound_speed <= 0
                or not np.isclose(
                    sound_speed**2,
                    gamma * gas_constant * temperature,
                    rtol=2e-6,
                    atol=0,
                )
            ):
                raise ValueError(
                    "The supplied EOS sound speed is inconsistent with the perfect-gas reference"
                )
            enthalpies.append(enthalpy)
        if not np.isclose(
            enthalpies[1] - enthalpies[0], -0.05 * cp * T0, rtol=2e-6, atol=1e-7
        ):
            raise ValueError(
                "The supplied EOS enthalpy is inconsistent with the perfect-gas reference"
            )
    return gamma, gas_constant


def area_mach(mach, gamma):
    """Return ``A/A* = [2/(gamma+1)*(1+(gamma-1)*M**2/2)]**k/M``.

    Here ``k=(gamma+1)/(2*(gamma-1))`` and ``A*`` is the sonic area on
    the same isentrope. ``mach`` may be a positive scalar or an array.
    Source: NACA 1135; NASA Glenn *Compressible Area Ratio*.
    """
    return (2 / (gamma + 1) * (1 + (gamma - 1) / 2 * np.asarray(mach) ** 2)) ** (
        (gamma + 1) / (2 * (gamma - 1))
    ) / np.asarray(mach)


def mach_from_area_ratio(ratio, gamma, branch="sub"):
    """Invert the area-Mach relation on the selected ``sub`` or ``sup`` branch.

    ``ratio`` is a scalar ``A/A* >= 1``. The numerical bracket caps the
    supersonic root at Mach 30; this is a reference-domain limit, not physics.
    """
    if branch not in ("sub", "sup"):
        raise ValueError("branch must be 'sub' or 'sup'")
    if not np.all(np.isfinite([ratio, gamma])) or ratio < 1 - 2e-12 or gamma <= 1:
        raise ValueError("Need finite A/A* >= 1 and gamma > 1")
    if abs(ratio - 1.0) < 2e-12:
        return 1.0
    bounds = (1e-10, 1.0) if branch == "sub" else (1.0, 30.0)
    return brentq(lambda m: area_mach(m, gamma) - ratio, *bounds, xtol=1e-13)


def pressure_ratio(mach, gamma):
    """Return isentropic static-to-stagnation pressure ``p/p0`` (NACA 1135)."""
    return (1 + (gamma - 1) / 2 * np.asarray(mach) ** 2) ** (-gamma / (gamma - 1))


def normal_shock(mach, gamma):
    """Return ``(p2/p1, M2, p02/p01)`` for an upstream Mach number >= 1.

    Mass, momentum and total enthalpy are conserved. ``T02=T01`` and the
    stagnation-pressure ratio follows by applying the isentropic static-to-total
    formula on each side. Source: NACA 1135 / NASA Glenn normal-shock equations.
    """
    if (
        not np.isfinite(gamma)
        or gamma <= 1
        or np.any(~np.isfinite(np.asarray(mach)))
        or np.any(np.asarray(mach) < 1)
    ):
        raise ValueError(
            "A compressive normal shock requires finite upstream Mach >= 1 and gamma > 1"
        )
    p2p1 = 1 + 2 * gamma / (gamma + 1) * (np.asarray(mach) ** 2 - 1)
    m2 = np.sqrt(
        (1 + (gamma - 1) / 2 * np.asarray(mach) ** 2)
        / (gamma * np.asarray(mach) ** 2 - (gamma - 1) / 2)
    )
    return p2p1, m2, pressure_ratio(mach, gamma) * p2p1 / pressure_ratio(m2, gamma)


def subsonic_reference(area, fluid, throat_mach, *, throat_area, p0, T0):
    """Sample a subsonic branch specified by its geometric-throat Mach number.

    ``0 < throat_mach <= 1`` includes the sonic limiting branch. The virtual
    sonic area is ``A*=throat_area/area_mach(throat_mach,gamma)``; using the
    geometric throat as ``A*`` in unchoked flow would give the wrong mass flow.
    This rearrangement is derived here from the classical area-Mach relation.
    Return pressure [Pa], density [kg/m3], and Mach arrays with keys p/rho/M.
    """
    if not 0 < throat_mach <= 1:
        raise ValueError("The subsonic reference needs 0 < throat_mach <= 1")
    gamma, gas_constant = _perfect_gas_constants(fluid, p0, T0)
    area = np.asarray(area, dtype=float)
    if (
        not np.isfinite(throat_area)
        or throat_area <= 0
        or area.ndim != 1
        or np.any(~np.isfinite(area))
        or np.any(area < throat_area * (1 - 1e-12))
    ):
        raise ValueError(
            "Need finite positive throat_area and a one-dimensional area array >= throat_area"
        )
    critical_area = throat_area / area_mach(throat_mach, gamma)
    ratios = np.asarray(area, dtype=float) / critical_area
    mach = np.array([mach_from_area_ratio(r, gamma) for r in ratios])
    temperature = T0 / (1 + 0.5 * (gamma - 1) * mach**2)
    pressure = p0 * pressure_ratio(mach, gamma)
    return dict(p=pressure, rho=pressure / (gas_constant * temperature), M=mach)


class PerfectGasReference:
    """Backpressure operating map for a symmetric converging-diverging nozzle.

    Parameters
    ----------
    fluid : object
        Supplies constant ``fluid.constants.gamma`` and ``fluid.constants.R``.
        If ``get_state`` is available, two reservoir-nearby PT samples check
        ideal-gas density, sound speed and constant-cp enthalpy differences.
        These checks reject incompatible EOS data but do not prove validity at
        every nozzle state. Constants-only objects explicitly specify an ideal
        gas without a property backend.
    p0, T0 : float
        Reservoir stagnation pressure [Pa] and temperature [K].
    length, inlet_area, throat_area : float
        Parabolic nozzle length [m], equal inlet/exit areas and throat area [m2].
        Geometry is defined by :func:`get_symmetric_nozzle`; the throat is at L/2.

    ``critical``, ``shock_exit`` and ``adapted`` are backpressure/p0 thresholds.
    An imposed pressure below the shock-at-exit threshold does not alter the
    supersonic nozzle-interior branch: jet adjustment occurs outside the model.
    """

    def __init__(
        self,
        fluid,
        *,
        p0=2.0e5,
        T0=300.0,
        length=0.2,
        inlet_area=3.0e-4,
        throat_area=1.5e-4
    ):
        self.gamma, self.R = _perfect_gas_constants(fluid, p0, T0)
        if (
            not np.all(np.isfinite([length, inlet_area, throat_area]))
            or length <= 0
            or not 0 < throat_area < inlet_area
        ):
            raise ValueError(
                "Need gamma>1, positive reservoir/geometry data and throat_area<inlet_area"
            )
        self.cp = self.gamma * self.R / (self.gamma - 1.0)
        self.p0, self.T0 = float(p0), float(T0)
        self.At, self.Ae, self.length = (
            float(throat_area),
            float(inlet_area),
            float(length),
        )
        self.M_sub = self.mach(self.Ae / self.At, "sub")
        self.M_sup = self.mach(self.Ae / self.At, "sup")
        self.critical = self.pressure_ratio(self.M_sub)
        self.adapted = self.pressure_ratio(self.M_sup)
        self.shock_exit = self.adapted * self.shock(self.M_sup)[0]
        self.mdot_star = (
            self.At
            * self.p0
            * np.sqrt(self.gamma / (self.R * self.T0))
            * (2 / (self.gamma + 1)) ** ((self.gamma + 1) / (2 * (self.gamma - 1)))
        )

    def area_mach(self, mach):
        """Return A/A* at the reference gas's constant gamma."""
        return area_mach(mach, self.gamma)

    def mach(self, ratio, branch):
        """Invert A/A* on the subsonic or supersonic branch."""
        return mach_from_area_ratio(ratio, self.gamma, branch)

    def pressure_ratio(self, mach):
        """Return p/p0 on a smooth isentropic segment."""
        return pressure_ratio(mach, self.gamma)

    def shock(self, m1):
        """Return p2/p1, M2 and p02/p01 across a normal shock."""
        return normal_shock(m1, self.gamma)

    def shock_data(self, position):
        """Return exit p/p0, downstream A*, and p02/p01 for a prescribed shock.

        The downstream sonic area is derived as ``As/area_mach(M2)``; the exit
        Mach is its subsonic area-Mach root. This is the project's assembly of
        the published local isentropic and normal-shock relations.
        """
        area = float(
            get_symmetric_nozzle(
                np.array([position]), L=self.length, A_inlet=self.Ae, A_throat=self.At
            ).A[0]
        )
        m1 = self.mach(area / self.At, "sup")
        _, m2, total_ratio = self.shock(m1)
        astar2 = area / self.area_mach(m2)
        me = self.mach(self.Ae / astar2, "sub")
        return total_ratio * self.pressure_ratio(me), astar2, total_ratio

    def profile(self, x, back_ratio):
        """Return nozzle-interior point values for ``back_ratio=back_pressure/p0``.

        ``x`` is a one-dimensional coordinate array in [0,L]. Results include
        p/T/rho/u/M, entropy rise ds [J/(kg K)], total pressure, mass flow and
        regime metadata. A shock plane uses its upstream limit (``x<=shock_x``).
        A shock exactly on x=L therefore has its downstream pressure/jump only
        in the metadata. Smooth entropy is constant; across a shock
        ``ds=-R*log(p02/p01)`` because stagnation temperature is unchanged.
        """
        x = np.asarray(x, dtype=float)
        r = float(back_ratio)
        if (
            x.ndim != 1
            or np.any(~np.isfinite(x))
            or np.any((x < 0) | (x > self.length))
        ):
            raise ValueError("x must be a finite one-dimensional array within [0,L]")
        if not 0 < r <= 1:
            raise ValueError(
                "Forward-flow reference requires 0 < back_pressure/p0 <= 1"
            )
        area = np.asarray(
            get_symmetric_nozzle(x, L=self.length, A_inlet=self.Ae, A_throat=self.At).A
        )
        if abs(r - self.critical) < 1e-12:
            r = self.critical
        shock_x, total_ratio, astar2 = None, 1.0, self.At
        if np.isclose(r, 1.0, atol=1e-14, rtol=0):
            regime, mach = "stagnant", np.zeros_like(x)
        elif r >= self.critical:
            regime = "critical_sonic" if abs(r - self.critical) < 1e-12 else "subsonic"
            me = np.sqrt(
                2 / (self.gamma - 1) * (r ** (-(self.gamma - 1) / self.gamma) - 1)
            )
            astar = self.Ae / self.area_mach(me)
            mach = np.array([self.mach(a / astar, "sub") for a in area])
        else:
            if r > self.shock_exit + 1e-12:
                regime = "internal_shock"
                shock_x = brentq(
                    lambda xs: self.shock_data(xs)[0] - r,
                    self.length / 2,
                    self.length,
                    xtol=1e-13,
                )
                _, astar2, total_ratio = self.shock_data(shock_x)
            elif abs(r - self.shock_exit) <= 1e-12:
                regime, shock_x = "shock_at_exit", self.length
                _, astar2, total_ratio = self.shock_data(shock_x)
            elif r > self.adapted + 1e-12:
                regime = "overexpanded_external"
            elif abs(r - self.adapted) <= 1e-12:
                regime = "adapted"
            else:
                regime = "underexpanded_external"
            mach = np.array(
                [
                    self.mach(a / self.At, "sub" if xx <= self.length / 2 else "sup")
                    for xx, a in zip(x, area)
                ]
            )
            if shock_x is not None:
                for i in np.flatnonzero(x > shock_x):
                    mach[i] = self.mach(area[i] / astar2, "sub")
        total_p = np.full_like(x, self.p0)
        if shock_x is not None:
            total_p[x > shock_x] *= total_ratio
        temp = self.T0 / (1 + (self.gamma - 1) / 2 * mach**2)
        pressure = total_p * self.pressure_ratio(mach)
        density = pressure / (self.R * temp)
        speed = mach * np.sqrt(self.gamma * self.R * temp)
        ds = -self.R * np.log(total_p / self.p0)
        return dict(
            x=x,
            p=pressure,
            T=temp,
            rho=density,
            u=speed,
            M=mach,
            ds=ds,
            p_total=total_p,
            mdot=density * speed * area,
            regime=regime,
            shock_x=shock_x,
            shock_ds=-self.R * np.log(total_ratio),
            downstream_exit_pressure=self.p0 * r if regime == "shock_at_exit" else None,
        )

    def landmarks(self):
        """Return operating pressure thresholds and choked mass flow [kg/s]."""
        return dict(
            critical_ratio=self.critical,
            shock_exit_ratio=self.shock_exit,
            adapted_ratio=self.adapted,
            choked_mdot_kg_s=self.mdot_star,
            subsonic_exit_M=self.M_sub,
            supersonic_exit_M=self.M_sup,
            gamma=self.gamma,
            gas_constant_J_kg_K=self.R,
        )
