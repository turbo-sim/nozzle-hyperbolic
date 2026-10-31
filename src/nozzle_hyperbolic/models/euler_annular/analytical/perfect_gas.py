"""Steady inviscid annular reference solutions, including swirl and a shock.

These *verification* functions assume a calorically perfect gas and fixed
geometry. They are independent of the production general-EOS residual,
boundary laws, conservative state maps, and numerical initialization.
The reference is semi-analytic: a bracketed scalar density root is solved at
each station; conserved cell averages additionally require quadrature.

The underlying perfect-gas isentropic and normal-shock relations are given in
NACA Report 1135, *Equations, Tables, and Charts for Compressible Flow* (1953),
sections "One-dimensional isentropic flow" and "Normal shock waves":
https://www.nasa.gov/wp-content/uploads/2023/03/equations-tables-charts-compressibleflow-report-1135.pdf

The swirl-dependent reduction below is derived for this package, rather than
copied from that report. With meridional arclength ``m``, cross-sectional area
``A(m)``, radius ``r(m)``, and no wall shear or heat transfer, the invariants are::

    mdot = rho * v_m * A
    ell = r * v_theta
    h0 = h + (v_m**2 + v_theta**2) / 2
    K = p / rho**gamma

Writing ``h = h_shift + gamma*K*rho**(gamma-1)/(gamma-1)`` gives the scalar
equation implemented in :meth:`PerfectGasReference._density`. Its unique
minimum separates meridionally subsonic and supersonic roots. Total Mach one
is not the sonic branch condition when swirl is nonzero.

The quadrature splitting and squared-distance coordinate at sonic endpoints
are package-derived numerical evaluation devices, not additional flow laws.
"""

from dataclasses import dataclass, replace

import jaxprop as jxp
import numpy as np
from scipy.optimize import brentq


class ReferenceBranchError(ValueError):
    """The requested steady isentropic branch does not exist at a station."""


@dataclass(frozen=True)
class PerfectGasReference:
    """Positive-meridional-flow invariants and calorically perfect-gas data.

    ``h_shift`` retains the fluid backend's enthalpy datum. Thus analytical
    ``W=(p,v_m,ell,h)`` can be compared directly with JAXProp numerical states,
    including backends whose enthalpy is not exactly ``cp*T``.

    All quantities use SI units: ``R`` is J/(kg K), ``mdot`` is kg/s,
    ``h0`` and ``h_shift`` are J/kg, and ``ell`` is m²/s. ``K`` is
    ``p/rho**gamma``; ``entropy_shift`` makes
    ``s = cp*log(T) - R*log(p) + entropy_shift`` consistent with the backend.
    ``default_branch`` is ``"sub"`` or ``"sup"`` according to *meridional*
    Mach. The radius and area must remain positive along the supplied geometry.

    This class does not solve for a shock location from back pressure. A shock
    is prescribed by ``shock_m``; the resulting outlet pressure can be used
    as a verification boundary datum.
    """

    gamma: float
    R: float
    h_shift: float
    entropy_shift: float
    mdot: float
    h0: float
    K: float
    ell: float
    geometry: object = None
    default_branch: str = "sub"

    def __post_init__(self):
        scalars = (
            self.gamma,
            self.R,
            self.h_shift,
            self.entropy_shift,
            self.mdot,
            self.h0,
            self.K,
            self.ell,
        )
        if not np.isfinite(scalars).all() or self.gamma <= 1 or self.R <= 0:
            raise ValueError(
                "Reference EOS/invariants must be finite with gamma > 1 and R > 0"
            )
        if self.mdot <= 0 or self.K <= 0 or self.h0 <= self.h_shift:
            raise ValueError(
                "Reference needs positive mass flow, entropy constant and thermal energy"
            )
        if self.default_branch not in ("sub", "sup"):
            raise ValueError("Reference branch must be 'sub' or 'sup'")

    @property
    def cp(self):
        """Constant isobaric heat capacity, J/(kg K)."""
        return self.gamma * self.R / (self.gamma - 1)

    @property
    def T0(self):
        """Stagnation temperature from the datum-consistent total enthalpy."""
        return (self.h0 - self.h_shift) / self.cp

    @classmethod
    def from_inlet(cls, fluid, geometry, *, p0, T0, M_m, M_theta=None, alpha=None):
        """Select invariants using reservoir data and a reference inlet Mach.

        ``M_theta`` and ``alpha`` are alternative ways to specify swirl; with
        angle data ``M_theta=M_m*tan(alpha)``. These inputs construct analytical
        fixtures; an FVM subsonic inlet must not prescribe Mach or mass flow in
        addition to its three incoming physical conditions.

        ``alpha`` is radians, measured from the positive meridional direction.
        The initial state follows ``T/T0 = 1/(1+(gamma-1)*M**2/2)`` using
        total ``M**2=M_m**2+M_theta**2``. See the module reference for the
        underlying isentropic relations. ``fluid`` must supply constant
        ``gamma`` and ``R`` and compatible ``get_state(PT_INPUTS, p, T)``
        values. An off-reservoir density/enthalpy check rejects an inconsistent
        backend, but does not establish that an arbitrary EOS is perfect over
        its entire domain.
        """
        if M_theta is not None and alpha is not None:
            raise ValueError("Specify M_theta or alpha, not both")
        if alpha is not None:
            if not np.isfinite(alpha) or abs(alpha) >= np.pi / 2:
                raise ValueError("Inlet angle must lie strictly between -pi/2 and pi/2")
            M_theta = M_m * np.tan(alpha)
        if M_theta is None:
            M_theta = 0.0
        if not np.isfinite((p0, T0, M_m, M_theta)).all() or min(p0, T0, M_m) <= 0:
            raise ValueError(
                "Positive reservoir pressure/temperature and meridional Mach are required"
            )
        try:
            gamma, gas_constant = float(fluid.constants.gamma), float(fluid.constants.R)
        except AttributeError as error:
            raise ValueError(
                "This analytical reference requires perfect-gas constants"
            ) from error
        if (
            not np.isfinite((gamma, gas_constant)).all()
            or gamma <= 1
            or gas_constant <= 0
        ):
            raise ValueError("Perfect-gas constants require finite gamma > 1 and R > 0")
        cp = gamma * gas_constant / (gamma - 1)
        reservoir = fluid.get_state(jxp.PT_INPUTS, p0, T0)
        h0, s0 = float(reservoir["h"]), float(reservoir["s"])
        h_shift = h0 - cp * T0
        entropy_shift = s0 - cp * np.log(T0) + gas_constant * np.log(p0)
        temperature = T0 / (1 + (gamma - 1) / 2 * (M_m**2 + M_theta**2))
        pressure = p0 * (temperature / T0) ** (gamma / (gamma - 1))
        density = pressure / (gas_constant * temperature)
        # Verify the backend agrees with the intended constant-cp gas at this
        # second state; extracting a gamma from a real fluid is insufficient.
        backend = fluid.get_state(jxp.PT_INPUTS, pressure, temperature)
        if not np.isclose(
            float(backend["rho"]), density, rtol=2e-9, atol=1e-12
        ) or not np.isclose(
            float(backend["h"]), cp * temperature + h_shift, rtol=2e-9, atol=1e-7
        ):
            raise ValueError(
                "The fluid backend is not the reference's calorically perfect gas"
            )
        metrics = geometry.evaluate(np.array(0.0))
        speed_of_sound = np.sqrt(gamma * gas_constant * temperature)
        v_m, v_theta = M_m * speed_of_sound, M_theta * speed_of_sound
        return cls(
            gamma,
            gas_constant,
            h_shift,
            entropy_shift,
            density * v_m * float(metrics.A),
            h0,
            pressure / density**gamma,
            float(metrics.r) * v_theta,
            geometry,
            "sub" if M_m <= 1 else "sup",
        )

    def _density(self, area, radius, branch):
        """Solve the package-derived invariant energy equation at one station.

        Substitute ``v_m=mdot/(A*rho)``, ``v_theta=ell/r``, and
        ``p=K*rho**gamma`` in the stagnation-energy invariant::

            gamma*K/(gamma-1) * rho**(gamma-1)
            + mdot**2/(2*A**2*rho**2)
            = h0-h_shift-ell**2/(2*r**2)

        Differentiating the left side yields the stationary density
        ``rho_star=(mdot**2/(gamma*K*A**2))**(1/(gamma+1))``, where
        ``v_m**2=gamma*K*rho_star**(gamma-1)=a**2``. The denser root is
        subsonic; the lighter root is supersonic. These reduction and bracket
        formulas were derived for this implementation from the listed invariants.
        """
        if branch not in ("sub", "sup"):
            raise ValueError("Reference branch must be 'sub' or 'sup'")
        available = self.h0 - self.h_shift - 0.5 * (self.ell / radius) ** 2
        if available <= 0:
            raise ReferenceBranchError(
                "Swirl kinetic energy exceeds available stagnation enthalpy"
            )
        critical_density = (self.mdot**2 / (self.gamma * self.K * area**2)) ** (
            1 / (self.gamma + 1)
        )

        def residual(density):
            return (
                self.gamma * self.K / (self.gamma - 1) * density ** (self.gamma - 1)
                + 0.5 * (self.mdot / (area * density)) ** 2
                - available
            )

        minimum = residual(critical_density)
        roundoff = 100 * np.finfo(float).eps * max(abs(available), 1.0)
        if minimum > roundoff:
            raise ReferenceBranchError(
                "Mass flow exceeds the meridional sonic capacity at this station"
            )
        if abs(minimum) <= roundoff:
            return critical_density
        if branch == "sub":
            lower = critical_density
            upper = (available * (self.gamma - 1) / (self.gamma * self.K)) ** (
                1 / (self.gamma - 1)
            )
        else:
            lower = self.mdot / (area * np.sqrt(2 * available))
            upper = critical_density
        return brentq(residual, lower, upper, xtol=5e-15, rtol=4 * np.finfo(float).eps)

    def _fields(self, m, geometry, branch):
        """Recover primitive and conserved densities from the density roots.

        ``q=(rho,rho*v_m,rho*ell,rho*E)``, with
        ``E=h-p/rho+(v_m**2+v_theta**2)/2``. The names ``Mm`` and ``Ma``
        are retained aliases for meridional and total Mach, respectively.
        """
        positions = np.asarray(m, dtype=float)
        metrics = geometry.evaluate(positions)
        area, radius = np.asarray(metrics.A), np.asarray(metrics.r)
        if callable(branch):
            branches = np.asarray(
                [branch(float(x)) for x in positions.reshape(-1)]
            ).reshape(positions.shape)
        else:
            branches = np.broadcast_to(branch, positions.shape)
        density = np.array(
            [
                self._density(float(a), float(r), str(b))
                for a, r, b in zip(
                    area.reshape(-1), radius.reshape(-1), branches.reshape(-1)
                )
            ]
        )
        density = density.reshape(positions.shape)
        pressure = self.K * density**self.gamma
        temperature = pressure / (density * self.R)
        enthalpy = self.cp * temperature + self.h_shift
        v_m = self.mdot / (area * density)
        v_theta = self.ell / radius
        sound = np.sqrt(self.gamma * self.R * temperature)
        ell = np.full_like(density, self.ell)
        total_energy = enthalpy - pressure / density + 0.5 * (v_m**2 + v_theta**2)
        entropy = (
            self.cp * np.log(temperature)
            - self.R * np.log(pressure)
            + self.entropy_shift
        )
        p_stagnation = pressure * (self.T0 / temperature) ** (
            self.gamma / (self.gamma - 1)
        )
        return dict(
            m=positions,
            r=radius,
            A=area,
            rho=density,
            p=pressure,
            T=temperature,
            v_m=v_m,
            v_theta=v_theta,
            ell=ell,
            h=enthalpy,
            E=total_energy,
            h0=np.full_like(density, self.h0),
            s=entropy,
            p0=p_stagnation,
            a=sound,
            M_m=v_m / sound,
            M_theta=v_theta / sound,
            M=np.hypot(v_m, v_theta) / sound,
            Mm=v_m / sound,
            Ma=np.hypot(v_m, v_theta) / sound,
            mdot=density * v_m * area,
            W=np.stack((pressure, v_m, ell, enthalpy), axis=-1),
            q=np.stack(
                (density, density * v_m, density * ell, density * total_energy), axis=-1
            ),
        )

    def fields_from_state(self, W, radius):
        """Evaluate analytic-EOS diagnostics of equivalent cell primitive states.

        ``W[..., :]`` contains ``(p,v_m,ell,h)`` and ``radius`` broadcasts
        over the leading dimensions. This helper applies the same gas and
        enthalpy datum as the reference to *arbitrary* supplied states; it
        does not enforce the steady invariants or assume upstream entropy.
        In particular, states obtained by conservative cell projection near a
        shock do not equal the pointwise analytic profile at the cell center.

        The reconstruction uses ``T=(h-h_shift)/cp``, ``rho=p/(R*T)``,
        ``a=sqrt(gamma*R*T)``, ``v_theta=ell/r``, and
        ``h0=h+(v_m**2+v_theta**2)/2``. These follow from the calorically
        perfect-gas EOS and definitions stated in the module docstring.
        ``alpha_deg`` is measured from the positive meridional direction.
        """
        W = np.asarray(W, dtype=float)
        radius = np.asarray(radius, dtype=float)
        if W.ndim < 1 or W.shape[-1] != 4 or not np.isfinite(W).all():
            raise ValueError(
                "Reference states must be finite with final dimension four"
            )
        if np.any(~np.isfinite(radius)) or np.any(radius <= 0):
            raise ValueError("Reference-state radius must be finite and positive")
        pressure, meridional, ell, enthalpy = np.moveaxis(W, -1, 0)
        temperature = (enthalpy - self.h_shift) / self.cp
        if np.any(temperature <= 0) or np.any(pressure <= 0):
            raise ValueError(
                "Reference states require positive pressure and temperature"
            )
        density = pressure / (self.R * temperature)
        sound = np.sqrt(self.gamma * self.R * temperature)
        tangential = ell / radius
        return dict(
            p=pressure,
            v_m=meridional,
            v_theta=tangential,
            rho=density,
            T=temperature,
            M_m=meridional / sound,
            M_theta=tangential / sound,
            M=np.hypot(meridional, tangential) / sound,
            alpha_deg=np.rad2deg(np.arctan2(tangential, meridional)),
            h0=enthalpy + 0.5 * (meridional**2 + tangential**2),
            ell=ell,
        )

    def normal_shock(self, shock_m, geometry=None, *, upstream_branch="sup"):
        """Normal jump at a prescribed reference station, using meridional Mach.

        Tangential velocity is continuous across the same-radius jump. The
        downstream reference retains mass flow, total enthalpy and angular
        momentum and adopts the increased entropy constant.

        The normal-shock density and pressure ratios use ``M_m`` in the
        NACA 1135 relations. Continuity of tangential velocity follows from
        tangential momentum balance across a shock with nonzero mass flux;
        its kinetic energy cancels in the normal energy jump. The use of
        ``ell`` at fixed shock radius is this package's annular adaptation.
        """
        geometry = self.geometry if geometry is None else geometry
        if geometry is None:
            raise ValueError("A geometry must be provided")
        upstream = self._fields(np.array(shock_m), geometry, upstream_branch)
        mach = float(upstream["M_m"])
        if mach <= 1:
            raise ReferenceBranchError(
                "A compressive normal shock needs meridional Mach > 1"
            )
        density_ratio = (self.gamma + 1) * mach**2 / ((self.gamma - 1) * mach**2 + 2)
        pressure_ratio = 1 + 2 * self.gamma / (self.gamma + 1) * (mach**2 - 1)
        density2 = float(upstream["rho"]) * density_ratio
        pressure2 = float(upstream["p"]) * pressure_ratio
        downstream_reference = replace(
            self,
            K=pressure2 / density2**self.gamma,
            geometry=geometry,
            default_branch="sub",
        )
        downstream = downstream_reference._fields(np.array(shock_m), geometry, "sub")
        return dict(
            m=float(shock_m),
            upstream=upstream,
            downstream=downstream,
            downstream_reference=downstream_reference,
            density_ratio=density_ratio,
            pressure_ratio=pressure_ratio,
            entropy_rise=float(downstream["s"] - upstream["s"]),
        )

    def profile(self, m, geometry=None, *, branch=None, shock_m=None):
        """Sample a smooth or shocked reference; the shock plane uses upstream limits.

        A callable ``branch(m)`` can explicitly select subsonic/supersonic
        isentropic roots around a compatible sonic point. It does not create a
        smooth sonic crossing where the geometry/invariants prohibit one.
        """
        geometry = self.geometry if geometry is None else geometry
        if geometry is None:
            raise ValueError("A geometry must be provided")
        branch = self.default_branch if branch is None else branch
        positions = np.asarray(m, dtype=float)
        if shock_m is None:
            return self._fields(positions, geometry, branch)
        if not np.isfinite(shock_m) or not 0 <= shock_m <= geometry.length:
            raise ValueError("shock_m must lie in the physical channel")
        jump = self.normal_shock(shock_m, geometry, upstream_branch=branch)
        flat = positions.reshape(-1)
        before = flat <= shock_m
        upstream = self._fields(flat[before], geometry, branch)
        downstream = jump["downstream_reference"]._fields(
            flat[~before], geometry, "sub"
        )
        result = {}
        for key in upstream:
            extra = upstream[key].shape[1:]
            values = np.empty((flat.size,) + extra)
            values[before], values[~before] = upstream[key], downstream[key]
            result[key] = values.reshape(positions.shape + extra)
        return result

    def cell_average(self, mesh, *, branch=None, shock_m=None, return_conserved=False):
        """Project exact conserved densities with the physical volume measure.

        Integrate ``q*A dm`` over each physical cell, splitting at the shock
        and geometry knots. Return the equivalent primitive state at the cell
        radius; this differs from averaging primitive values pointwise.
        At a sonic panel endpoint, a squared-distance coordinate regularizes
        the square-root branch behavior; its Jacobian is included explicitly.
        ``return_conserved=True`` also returns the integrated/average states.

        Specifically, ``integrated[i]=integral(q*A dm)`` and
        ``q[i]=integrated[i]/volume[i]``. The equivalent primitive ``W`` is
        recovered algebraically at the cell-center radius. This definition
        matches a finite-volume cell's stored quantities; it is not an exact
        volume average of primitive pressure, velocity or enthalpy.
        """
        nodes16, weights16 = np.polynomial.legendre.leggauss(16)
        nodes32, weights32 = np.polynomial.legendre.leggauss(32)

        def integrate(lower, upper, mapping, depth=0):
            estimates = []
            for nodes, weights in ((nodes16, weights16), (nodes32, weights32)):
                coordinate = lower + (upper - lower) * (nodes + 1) / 2
                m, jacobian = mapping(coordinate)
                fields = self.profile(m, mesh.geometry, branch=branch, shock_m=shock_m)
                integrand = fields["q"] * (fields["A"] * jacobian)[:, None]
                estimates.append((upper - lower) / 2 * (weights @ integrand))
            low, high = estimates
            if np.all(np.abs(high - low) <= 2e-11 * np.maximum(np.abs(high), 1e-14)):
                return high
            if depth >= 10:
                raise ReferenceBranchError(
                    "Analytical cell projection did not meet quadrature tolerance"
                )
            middle = (lower + upper) / 2
            return integrate(lower, middle, mapping, depth + 1) + integrate(
                middle, upper, mapping, depth + 1
            )

        def panel(lower, upper):
            endpoint_mach = self.profile(
                np.array([lower, upper]), mesh.geometry, branch=branch, shock_m=shock_m
            )["M_m"]
            sonic = np.abs(endpoint_mach - 1.0) <= 1e-7
            if np.all(sonic):
                # Separate endpoint transformations when both ends are sonic.
                # A constant sonic solution is also covered by these two panels.
                middle = (lower + upper) / 2
                mappings = (
                    lambda t: (
                        lower + (middle - lower) * t * t,
                        2 * (middle - lower) * t,
                    ),
                    lambda t: (
                        upper - (upper - middle) * t * t,
                        2 * (upper - middle) * t,
                    ),
                )
                return sum(
                    (integrate(0.0, 1.0, mapping) for mapping in mappings), np.zeros(4)
                )
            if sonic[0]:
                return integrate(
                    0.0,
                    1.0,
                    lambda t: (
                        lower + (upper - lower) * t * t,
                        2 * (upper - lower) * t,
                    ),
                )
            if sonic[1]:
                return integrate(
                    0.0,
                    1.0,
                    lambda t: (
                        upper - (upper - lower) * t * t,
                        2 * (upper - lower) * t,
                    ),
                )
            return integrate(lower, upper, lambda m: (m, np.ones_like(m)))

        integrated = []
        for lower, upper in zip(mesh.m_faces[:-1], mesh.m_faces[1:]):
            breaks = [
                lower,
                *(k for k in mesh.geometry.knot_distances if lower < k < upper),
                upper,
            ]
            if shock_m is not None and lower < shock_m < upper:
                breaks.append(shock_m)
            breaks = sorted(breaks)
            integrated.append(
                sum((panel(a, b) for a, b in zip(breaks[:-1], breaks[1:])), np.zeros(4))
            )
        integrated = np.asarray(integrated)
        q = integrated / np.asarray(mesh.volumes)[:, None]
        density = q[:, 0]
        v_m, ell = q[:, 1] / density, q[:, 2] / density
        v_theta = ell / np.asarray(mesh.r_center)
        thermal_energy = q[:, 3] / density - 0.5 * (v_m**2 + v_theta**2) - self.h_shift
        temperature = thermal_energy / (self.cp - self.R)
        if np.any(temperature <= 0):
            raise ReferenceBranchError(
                "Equivalent cell-average state has nonpositive temperature"
            )
        pressure = density * self.R * temperature
        W = np.column_stack((pressure, v_m, ell, self.cp * temperature + self.h_shift))
        if return_conserved:
            return dict(W=W, q=q, integrated=integrated)
        return W
