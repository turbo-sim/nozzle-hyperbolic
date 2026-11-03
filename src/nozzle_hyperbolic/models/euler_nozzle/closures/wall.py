"""Nozzle-owned stationary-wall policies reusable by annular Euler.

Constitutive correlations use hydraulic diameter and physical fluid data.
The calling model supplies wall measures and velocity directions for sources.

Haaland and the laminar 64/Re law return *Darcy* factors. Wall stress is
``tau=f_D*rho*speed**2/8``. A stationary adiabatic wall removes momentum but
does not remove total energy; signed heat flux alone supplies wall energy.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np


def Haaland(Re, epsilon, D):
    """Historical turbulent Darcy correlation, retaining its -1.81 constant.

    This function alone does not include laminar flow. The historical value
    zero for Re<=0 is retained for compatibility. New shear evaluation should
    use :func:`haaland_wall_shear` or :class:`HaalandWall`, which include the
    analytical laminar limit. No perfect-gas thermodynamics is used here.
    """
    Re = jnp.asarray(Re)
    safe_re = jnp.where(Re > 0, Re, 1.0)
    term = (epsilon/(3.7*D))**1.11+6.9/safe_re
    safe_term = jnp.where(term > 0, term, 1.0)
    factor = (-1.81*jnp.log10(safe_term))**-2
    return jnp.where((Re > 0)&(term > 0), factor, 0.0)


def darcy_friction_factor(Re, roughness, hydraulic_diameter):
    """Laminar 64/Re below 2300, turbulent Haaland above it.

    No empirical transition smoothing is introduced. The factor is infinity
    at Re=0 and NaN for negative Re. There is no Reynolds floor: the wall-shear
    functions evaluate the finite laminar limit directly instead.
    """
    Re = jnp.asarray(Re)
    safe_re = jnp.where(Re > 0, Re, 1.0)
    factor = jnp.where(Re < 2300.0, 64.0/safe_re,
                       Haaland(safe_re, roughness, hydraulic_diameter))
    return jnp.where(Re > 0, factor, jnp.where(Re == 0, jnp.inf, jnp.nan))


def _drag_coefficient(speed_squared, density, viscosity, diameter, roughness):
    """Finite tau/speed, with finite derivatives at the origin."""
    # Do not differentiate sqrt at zero, even in an unselected JAX branch.
    nonzero = speed_squared > 0
    speed = jnp.sqrt(jnp.where(nonzero, speed_squared, 1.0))
    reynolds = density*speed*diameter/viscosity
    turbulent = Haaland(reynolds, roughness, diameter)*density*speed/8.0
    laminar = 8.0*viscosity/diameter
    laminar_region = (density*diameter/viscosity)**2*speed_squared < 2300.0**2
    return jnp.where(laminar_region, laminar, turbulent)


def haaland_wall_shear(velocity, density, viscosity, hydraulic_diameter, roughness=0.0):
    """Signed wall shear, continuous through zero and opposing either flow.

    A model subtracts this stress times wall area from axial momentum. The
    laminar expression is exactly ``8*viscosity*velocity/hydraulic_diameter``;
    its zero-speed derivative is finite and does not depend on a Re floor.
    """
    velocity = jnp.asarray(velocity)
    coefficient = _drag_coefficient(velocity**2, density, viscosity,
                                    hydraulic_diameter, roughness)
    return coefficient*velocity


def shear_stress_wall(W, r, fluid, *, prescribed=0.0):
    """Prescribed nonnegative stress magnitude (Pa); model applies direction."""
    W = jnp.asarray(W)
    return jnp.broadcast_to(jnp.asarray(prescribed, dtype=W.dtype), W.shape[:-1])


def heat_flux_wall(W, r, fluid, *, prescribed=0.0):
    """Prescribed heat flux (W/m²), positive into the fluid; default is zero."""
    W = jnp.asarray(W)
    return jnp.broadcast_to(jnp.asarray(prescribed, dtype=W.dtype), W.shape[:-1])


def _validate_profile(name, values, *, positive=False, nonnegative=False):
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    if positive and not np.all(values > 0):
        raise ValueError(f"{name} must be positive")
    if nonnegative and not np.all(values >= 0):
        raise ValueError(f"{name} must be nonnegative")


def _property(state, names):
    for name in names:
        try:
            return jnp.asarray(state[name])
        except (KeyError, TypeError, AttributeError):
            if hasattr(state, name):
                return jnp.asarray(getattr(state, name))
    raise KeyError(f"The supplied EOS state must provide one of {names}")


def _wall_state(physical, radius, fluid):
    physical = jnp.asarray(physical)
    state = fluid.get_state(jxp.HmassP_INPUTS, physical[..., -1], physical[..., 0])
    density = _property(state, ("rho", "d", "density"))
    viscosity = _property(state, ("mu", "viscosity", "dynamic_viscosity"))
    speed_squared = physical[..., 1]**2
    if physical.shape[-1] == 4:
        speed_squared = speed_squared+(physical[..., 2]/jnp.asarray(radius))**2
    elif physical.shape[-1] != 3:
        raise ValueError("Wall policies require nozzle (p,u,h) or annular (p,vm,ell,h) primitives")
    return density, viscosity, speed_squared


@dataclass(frozen=True)
class PrescribedWall:
    """Zero-default/prescribed annular or nozzle stress and heat profiles.

    Each profile is scalar or has the physical cells' shape. State-dependent
    user laws can instead implement the same callable signature. Nonzero
    prescribed shear has no unique direction at exactly zero speed; the Euler
    models use zero force there. Use a viscous law for smooth zero-speed drag.
    No universal explicit heat/stress bound can be inferred from these data.
    """

    tau_w: object = 0.0
    q_w: object = 0.0

    def __post_init__(self):
        _validate_profile("tau_w", self.tau_w, nonnegative=True)
        _validate_profile("q_w", self.q_w)

    def __call__(self, physical, radius, fluid):
        return (shear_stress_wall(physical, radius, fluid, prescribed=self.tau_w),
                heat_flux_wall(physical, radius, fluid, prescribed=self.q_w))


@dataclass(frozen=True)
class HaalandWall:
    """Haaland/laminar wall law using density and viscosity from get_state().

    ``hydraulic_diameter`` and ``roughness`` are fixed scalar/cell profiles.
    ``heat_flux`` is prescribed, positive into the fluid. Models should use
    ``drag_coefficient`` times each velocity component, rather than dividing
    the returned stress magnitude by a norm at zero speed. The model owns
    wall area/volume geometry and momentum/torque/energy source assembly.
    """

    hydraulic_diameter: object
    roughness: object = 0.0
    heat_flux: object = 0.0

    def __post_init__(self):
        _validate_profile("hydraulic_diameter", self.hydraulic_diameter, positive=True)
        _validate_profile("roughness", self.roughness, nonnegative=True)
        _validate_profile("heat_flux", self.heat_flux)

    def drag_coefficient(self, physical, radius, fluid):
        density, viscosity, speed_squared = _wall_state(physical, radius, fluid)
        return _drag_coefficient(speed_squared, density, viscosity,
                                 jnp.asarray(self.hydraulic_diameter), jnp.asarray(self.roughness))

    def __call__(self, physical, radius, fluid):
        _, _, speed_squared = _wall_state(physical, radius, fluid)
        speed = jnp.where(speed_squared > 0,
                          jnp.sqrt(jnp.where(speed_squared > 0, speed_squared, 1.0)), 0.0)
        return (self.drag_coefficient(physical, radius, fluid)*speed,
                heat_flux_wall(physical, radius, fluid, prescribed=self.heat_flux))

    def explicit_timestep(self, physical, radius, fluid, wall_measure_per_volume):
        """Local frozen-property drag relaxation bound, not an EOS heat bound.

        Returns the nonoscillatory Euler bound rho/(P/A*d(tau)/d(speed)); it
        also bounds the transverse decay eigenvalue tau/speed. Properties are
        evaluated at the current state and held fixed for this estimate.
        The solver must combine it with wave CFL and validate each trial.
        The sharp laminar/turbulent switch has no bound for crossing the
        empirical discontinuity, and prescribed heating needs model-specific
        admissibility checks. No restriction is returned for zero wall area.
        """
        density, viscosity, speed_squared = _wall_state(physical, radius, fluid)
        diameter, roughness = jnp.asarray(self.hydraulic_diameter), jnp.asarray(self.roughness)
        coefficient = _drag_coefficient(speed_squared, density, viscosity, diameter, roughness)
        safe_speed = jnp.sqrt(jnp.where(speed_squared > 0, speed_squared, 1.0))
        speed = jnp.where(speed_squared > 0, safe_speed, 0.0)

        def radial_stress(s):
            return _drag_coefficient(s**2, density, viscosity, diameter, roughness)*s

        _, derivative = jax.jvp(radial_stress, (speed,), (jnp.ones_like(speed),))
        drag_rate = jnp.asarray(wall_measure_per_volume)*jnp.maximum(coefficient, derivative)/density
        safe_rate = jnp.where(drag_rate > 0, drag_rate, 1.0)
        return jnp.where(drag_rate > 0, 1.0/safe_rate, jnp.inf)
