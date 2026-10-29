"""General-EOS annular state maps; primitive order is (p, v_m, ell, h)."""

import jax
import jax.numpy as jnp
import jaxprop as jxp


_PROPERTY_NAMES = {
    "rho": ("rho", "d", "density"),
    "a": ("a", "speed_sound", "speed_of_sound"),
    "G": ("G", "gruneisen"),
    "T": ("T", "temperature"),
    "s": ("s", "entropy"),
    "p": ("p", "pressure"),
    "h": ("h", "enthalpy"),
}


def state_property(state, name):
    """Read an existing JAXProp property or its established aliases."""
    for key in _PROPERTY_NAMES.get(name, (name,)):
        try:
            return jnp.asarray(state[key])
        except (KeyError, AttributeError, TypeError):
            pass
        if hasattr(state, key):
            return jnp.asarray(getattr(state, key))
    raise KeyError(f"The supplied fluid state does not provide {name!r}")


def thermodynamic_state(W, fluid):
    """Evaluate the supplied EOS directly at trial enthalpy and pressure."""
    W = jnp.asarray(W)
    return fluid.get_state(jxp.HmassP_INPUTS, W[..., 3], W[..., 0])


def primitive_to_conservative(W, r, fluid, rho=None):
    """Return unweighted (rho, rho*v_m, rho*ell, rho*E) at radius r."""
    W, r = jnp.asarray(W), jnp.asarray(r)
    p, v_m, ell, h = (W[..., k] for k in range(4))
    if rho is None:
        rho = state_property(thermodynamic_state(W, fluid), "rho")
    rho = jnp.broadcast_to(jnp.asarray(rho), p.shape)
    h0 = h + 0.5 * (v_m**2 + (ell / r)**2)
    return jnp.stack((rho, rho*v_m, rho*ell, rho*h0-p), axis=-1)


def transverse_conservative(W, r, fluid, rho=None):
    """Return ordinary transverse-Euler storage at one fixed face radius."""
    q = primitive_to_conservative(W, r, fluid, rho=rho)
    return q.at[..., 2].set(q[..., 2] / jnp.asarray(r))


def conservative_to_primitive(q, r, fluid, pressure_guess=None, pressure_bounds=None):
    """Recover primitives through a supported EOS pair or a strict HP inversion.

    This setup/postprocessing operation is not called by the primitive-variable
    implicit residual or its algebraic boundary block. If the EOS does not
    implement density/internal energy, supply physical ``pressure_bounds`` or
    an explicit ``pressure_guess`` used to build an initial factor-ten bracket.
    Declared table bounds are also respected. Bracketing or recovery failure
    raises; no guessed midpoint or idealized EOS is substituted. At setup,
    nonfinite or inconsistent direct-pair results also trigger strict HP
    recovery: a host callback can return NaNs for an unsupported input pair.
    Native direct-pair JIT/AD remains supported; a traced call must provide a
    working density/internal-energy pair because HP bracketing is host-only.
    """
    q, r = jnp.asarray(q), jnp.asarray(r)
    rho = q[..., 0]
    v_m, ell = q[..., 1]/rho, q[..., 2]/rho
    e = q[..., 3]/rho - 0.5*(v_m**2+(ell/r)**2)
    try:
        state = fluid.get_state(jxp.DmassUmass_INPUTS, rho, e)
        candidate = jnp.stack((state_property(state, "p"), v_m, ell,
                               state_property(state, "h")), axis=-1)
    except (KeyError, NotImplementedError):
        return _recover_conservative_hp(q, r, fluid, pressure_guess, pressure_bounds)
    if isinstance(candidate, jax.core.Tracer):
        return candidate
    if bool(jnp.all(admissible(candidate, r, fluid))):
        hp_state = thermodynamic_state(candidate, fluid)
        a = state_property(hp_state, "a")
        recovered = primitive_to_conservative(candidate, r, fluid,
                                               rho=state_property(hp_state, "rho"))
        physical_scale = jnp.stack((rho, rho*a, rho*r*a, rho*a**2), axis=-1)
        scale = jnp.maximum(jnp.maximum(jnp.abs(q), jnp.abs(recovered)), physical_scale)
        tolerance = getattr(fluid, "state_consistency_rtol", 1e-8)
        if bool(jnp.all(jnp.isfinite(recovered)
                        & (jnp.abs(recovered-q) <= tolerance*scale))):
            return candidate
    return _recover_conservative_hp(q, r, fluid, pressure_guess, pressure_bounds)


def _recover_conservative_hp(q, r, fluid, pressure_guess, pressure_bounds):
    """Host-only, independently checked setup recovery along h=e+p/rho."""
    import numpy as np
    from scipy.optimize import brentq

    q_array = np.asarray(q, dtype=float)
    shape = q_array.shape[:-1]
    radii = np.broadcast_to(np.asarray(r, dtype=float), shape).reshape(-1)
    data = q_array.reshape(-1, 4)
    if pressure_bounds is None:
        lower = getattr(fluid, "p_min", None)
        upper = getattr(fluid, "p_max", None)
        if lower is not None and upper is not None:
            pressure_bounds = (lower, upper)
        elif pressure_guess is not None:
            guesses = np.broadcast_to(np.asarray(pressure_guess, dtype=float), shape)
            if not np.all(np.isfinite(guesses)) or not np.all(guesses > 0):
                raise ValueError("Automatic HP brackets require finite positive pressure guesses")
            pressure_bounds = (guesses/10.0, guesses*10.0)
        else:
            raise ValueError("This EOS needs explicit pressure_bounds or pressure_guess for conservative HP recovery")
    lower, upper = [np.broadcast_to(np.asarray(value, dtype=float), shape).reshape(-1)
                    for value in pressure_bounds]
    recovered = []
    for cell, radius, p_lo, p_hi in zip(data, radii, lower, upper):
        density, momentum, angular, energy = cell
        if not np.all(np.isfinite(cell)) or density <= 0 or not np.isfinite(radius) or radius <= 0:
            raise ValueError("Conservative recovery requires finite state, positive density and radius")
        v_m, ell = momentum/density, angular/density
        e = energy/density-0.5*(v_m**2+(ell/radius)**2)
        if hasattr(fluid, "p_min"):
            p_lo = max(p_lo, float(fluid.p_min))
        if hasattr(fluid, "p_max"):
            p_hi = min(p_hi, float(fluid.p_max))
        if hasattr(fluid, "h_min"):
            p_lo = max(p_lo, density*(float(fluid.h_min)-e))
        if hasattr(fluid, "h_max"):
            p_hi = min(p_hi, density*(float(fluid.h_max)-e))
        if not np.isfinite(p_lo) or not np.isfinite(p_hi) or p_lo >= p_hi:
            raise ValueError("No admissible pressure bracket for conservative HP recovery")

        def density_defect(pressure):
            candidate = jnp.asarray([pressure, v_m, ell, e+pressure/density])
            state = thermodynamic_state(candidate, fluid)
            rho_eos = float(state_property(state, "rho"))
            if not np.isfinite(rho_eos):
                raise ValueError("Nonfinite EOS density in conservative HP recovery")
            return rho_eos-density

        pressure = brentq(density_defect, p_lo, p_hi, xtol=1e-9, rtol=1e-12)
        candidate = jnp.asarray([pressure, v_m, ell, e+pressure/density])
        if not bool(admissible(candidate, radius, fluid)):
            raise ValueError("Conservative HP recovery returned an inadmissible state")
        if abs(density_defect(pressure))/density > 1e-9:
            raise ValueError("Conservative HP recovery failed its independent density audit")
        recovered.append(candidate)
    return jnp.stack(recovered).reshape(shape+(4,))


def primitive_fields(W, r, fluid):
    """Return physical diagnostic fields without a stagnation-state inversion."""
    W, r = jnp.asarray(W), jnp.asarray(r)
    state = thermodynamic_state(W, fluid)
    rho, a = state_property(state, "rho"), state_property(state, "a")
    p, v_m, ell, h = (W[..., k] for k in range(4))
    v_theta = ell/r
    kinetic = 0.5*(v_m**2+v_theta**2)
    fields = dict(p=p, v_m=v_m, ell=ell, h=h, v_theta=v_theta,
                  rho=rho, a=a, G=state_property(state, "G"), e=h-p/rho,
                  E=h-p/rho+kinetic, h0=h+kinetic,
                  M_m=v_m/a, M_theta=v_theta/a,
                  M=jnp.sqrt(v_m**2+v_theta**2)/a)
    for name in ("T", "s"):
        try:
            fields[name] = state_property(state, name)
        except KeyError:
            pass
    return fields


def admissible(W, r, fluid):
    """Check physical states, declared bounds and the HP thermodynamic response.

    When temperature/entropy are supplied, T*(ds/dh)_p must agree with one.
    This reference-invariant Gibbs identity detects EOS calls that silently
    floor temperature while passing the original enthalpy through unchanged.
    ``thermodynamic_consistency_rtol`` (default 1e-3) may declare a backend's
    derivative accuracy; ``state_consistency_rtol`` defaults to 1e-8 for
    returned input-coordinate checks. The Gibbs audit costs one entropy JVP
    per checked batch. Declared domain bounds are checked before any EOS call;
    a batch with an out-of-domain member is conservatively rejected as a whole.
    No result is clipped or repaired.
    """
    W, r = jnp.asarray(W), jnp.asarray(r)
    domain_valid = (jnp.all(jnp.isfinite(W), axis=-1)
                    & jnp.isfinite(r) & (r > 0))
    for field, index in (("p", 0), ("h", 3)):
        for suffix, comparison in (("min", jnp.greater_equal),
                                   ("max", jnp.less_equal)):
            bound = getattr(fluid, f"{field}_{suffix}", None)
            if bound is not None:
                domain_valid = domain_valid & comparison(W[..., index], bound)
    return jax.lax.cond(jnp.all(domain_valid),
                        lambda: _admissible_eos(W, r, fluid),
                        lambda: jnp.zeros_like(domain_valid, dtype=bool))


def _admissible_eos(W, r, fluid):
    """EOS checks evaluated only after finite inputs and declared-domain checks."""
    state = thermodynamic_state(W, fluid)
    rho, a, G = (state_property(state, k) for k in ("rho", "a", "G"))
    valid = (jnp.all(jnp.isfinite(W), axis=-1) & jnp.isfinite(r) & (r > 0)
             & jnp.isfinite(rho) & (rho > 0) & jnp.isfinite(a) & (a > 0)
             & jnp.isfinite(G))
    coordinate_rtol = getattr(fluid, "state_consistency_rtol", 1e-8)
    for name, coordinate, scale in (
        ("h", W[..., 3], jnp.maximum(jnp.maximum(jnp.abs(W[..., 3]), a**2), 1.0)),
        ("p", W[..., 0], jnp.maximum(jnp.maximum(jnp.abs(W[..., 0]), rho*a**2), 1.0)),
    ):
        try:
            returned = state_property(state, name)
            valid = (valid & jnp.isfinite(returned)
                     & (jnp.abs(returned-coordinate) <= coordinate_rtol*scale))
        except KeyError:
            pass
    try:
        temperature = state_property(state, "T")
        valid = valid & jnp.isfinite(temperature) & (temperature > 0)
        try:
            entropy = state_property(state, "s")

            def entropy_at_h(enthalpy):
                candidate = W.at[..., 3].set(enthalpy)
                return state_property(thermodynamic_state(candidate, fluid), "s")

            _, entropy_h = jax.jvp(entropy_at_h, (W[..., 3],),
                                   (jnp.ones_like(W[..., 3]),))
            response = temperature*entropy_h
            tolerance = getattr(fluid, "thermodynamic_consistency_rtol", 1e-3)
            valid = (valid & jnp.isfinite(entropy) & jnp.isfinite(response)
                     & (jnp.abs(response-1.0) <= tolerance))
        except KeyError:
            pass
    except KeyError:
        pass
    return valid


def wave_speed(W, r, fluid):
    """Meridional acoustic spectral radius; swirl is not an advection speed."""
    return jnp.abs(jnp.asarray(W)[..., 1]) + state_property(
        thermodynamic_state(W, fluid), "a")
