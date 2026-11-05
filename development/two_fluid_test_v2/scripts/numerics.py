"""Four explicit ablations; physical laws and the baseline remain in src/."""

from dataclasses import asdict

import jax
import jax.numpy as jnp

from nozzle_hyperbolic.models.two_fluid import TwoFluidModel, primitives
from nozzle_hyperbolic.models.two_fluid.flux import compatible_flux_solve, smoothed_sign
from nozzle_hyperbolic.models.two_fluid.matrices import (
    J_matrix, K_matrix, X_matrix, common_quantities, eigenstructure, flux_vector, phase_props,
)
from nozzle_hyperbolic.numerics.reconstruction import reconstruct_faces


def smooth_absolute(speed, width):
    return jnp.where(abs(speed) >= width, abs(speed), .5 * (speed**2 / width + width))


def data(u, gas, liquid, virtual_mass):
    eg, el = phase_props(gas, u[0], u[4]), phase_props(liquid, u[0], u[5])
    cq = common_quantities(u, eg, el, virtual_mass)
    return eg, el, cq, eigenstructure(u, eg, el, cq)


def weights(left, right, mean_data, gas, liquid, virtual_mass, cfg, new_sonic):
    eg, el, cq, (speeds, _, _, _) = mean_data
    sound = .5 * (speeds[2] - speeds[3])
    if new_sonic:
        width = .02 * sound
        denominator = smooth_absolute(speeds, width)
    else:
        ev_left = data(left, gas, liquid, virtual_mass)[3][0]
        ev_right = data(right, gas, liquid, virtual_mass)[3][0]
        floor = cfg.delta_factor * abs(sound)
        crossing = (ev_left < 0.) & (ev_right > 0.)
        width = jnp.where(crossing, ev_right - ev_left, floor)
        denominator = jnp.where(crossing,
            (speeds**2 + width**2) / (2. * jnp.maximum(width, 1e-300)),
            jnp.maximum(abs(speeds), floor))
        width = floor
    return speeds / denominator, smoothed_sign(cq.u1, width), width


def face(left, right, gas, liquid, virtual_mass, cfg, acoustic_floor, inverse_free, new_sonic):
    """Native signed contraction or inverse-free primitive-jump viscosity.

    A=0,B=0 here is an audit reference only; the actual baseline calls src/.
    A=1,B=0 uses lambda*s_old. A=1,B=1 uses Harten absolute magnitudes,
    matching v1 with native low-Mach floors. Their grouped viscosity remains
    a bounded approximation, not exact |G| at a material Jordan block.
    """
    mean = .5 * (left + right)
    md = data(mean, gas, liquid, virtual_mass)
    eg, el, cq, (speeds, r, l, d) = md
    egl, ell = phase_props(gas, left[0], left[4]), phase_props(liquid, left[0], left[5])
    egr, elr = phase_props(gas, right[0], right[4]), phase_props(liquid, right[0], right[5])
    fl, fr = flux_vector(left, egl, ell), flux_vector(right, egr, elr)
    s, s12, width = weights(left, right, md, gas, liquid, virtual_mass, cfg, new_sonic)
    sound = .5 * (speeds[2] - speeds[3])
    mach = abs(.5 * (speeds[2] + speeds[3])) / sound
    z = jnp.ones(6).at[2:4].set(jnp.clip(mach, acoustic_floor, 1.))

    if inverse_free:
        jump = right - left
        magnitude = smooth_absolute(speeds, width) if new_sonic else speeds * s
        material = smooth_absolute(cq.u1, width) if new_sonic else cq.u1 * s12
        magnitude = magnitude * z

        def grouped(_):
            out = material * jump
            for index in (2, 3, 4, 5):
                out = out + (magnitude[index] - material) * r[:, index] * (l[index] @ jump) / d[index]
            return out

        def full(_):
            return r @ (magnitude * ((l @ jump) / d))

        def transition(_):
            t = jnp.clip((abs(cq.du) - cfg.du_blend_lo) / (cfg.du_blend_hi - cfg.du_blend_lo), 0., 1.)
            w = .5 * (1. + jnp.cos(jnp.pi * t))
            return w * grouped(None) + (1. - w) * full(None)

        action = jax.lax.cond(abs(cq.du) < cfg.du_blend_lo, grouped,
            lambda _: jax.lax.cond(abs(cq.du) >= cfg.du_blend_hi, full, transition, None), None)
        return .5 * (fl + fr) - .5 * (J_matrix(mean, eg, el) @ action)

    k = K_matrix(mean, eg, el)

    def grouped(_):
        out = .5 * (1. + s12) * fl + .5 * (1. - s12) * fr
        y = compatible_flux_solve(k, fl - fr, mean)
        for index in (2, 3, 4, 5):
            rho = k @ r[:, index]
            amplitude = (l[index] @ y) / d[index]
            out = out + z[index] * .5 * (s[index] - s12) * rho * amplitude
        return out

    def full(_):
        y = compatible_flux_solve(k, fl - fr, mean)
        return .5 * (fl + fr) + .5 * (k @ r) @ (s * z * ((l @ y) / d))

    def transition(_):
        t = jnp.clip((abs(cq.du) - cfg.du_blend_lo) / (cfg.du_blend_hi - cfg.du_blend_lo), 0., 1.)
        w = .5 * (1. + jnp.cos(jnp.pi * t))
        return w * grouped(None) + (1. - w) * full(None)

    return jax.lax.cond(abs(cq.du) < cfg.du_blend_lo, grouped, transition, None)


def path_nc(left, right, gas, liquid, virtual_mass):
    jump = right - left

    def one(position):
        u = left + position * jump
        eg, el = phase_props(gas, u[0], u[4]), phase_props(liquid, u[0], u[5])
        cq = common_quantities(u, eg, el, virtual_mass)
        return X_matrix(u, eg, el, cq) @ jump

    offset = .5 / jnp.sqrt(3.)
    return .5 * jnp.sum(jax.lax.map(one, jnp.array((.5 - offset, .5 + offset))), axis=0)


class AblationModel(TwoFluidModel):
    name = "two_fluid_ablation_v2"

    def __init__(self, *args, variant, scheme, **kwargs):
        self.variant, self.scheme = variant, scheme
        self.inverse_free = variant in ("inverse_old_sonic", "both")
        self.new_sonic = variant in ("sonic_only", "both")
        super().__init__(*args, **kwargs)

    def numerical_face_flux(self, left, right, scheme):
        cfg = self.compatibility
        floors = jnp.full((left.shape[0],), cfg.lowmach_diss)
        if cfg.lowmach_diss_boundary is not None:
            floors = floors.at[0].set(cfg.lowmach_diss_boundary).at[-1].set(cfg.lowmach_diss_boundary)
        return jax.lax.map(lambda pair: face(pair[0], pair[1], self.fluid_g, self.fluid_l,
            self.virtual_mass, cfg, pair[2], self.inverse_free, self.new_sonic),
            (primitives(left), primitives(right), floors))

    def reconstructed_primitives(self, full):
        return reconstruct_faces(primitives(full), jnp.asarray(self.mesh.points), jnp.asarray(self.mesh.faces),
            self.scheme.reconstruction, variable_scales=self.reconstruction_scales,
            smooth_coefficient=self.scheme.smooth_limiter_coefficient)

    def integrated_nonconservative(self, full, raw_flux):
        if not self.inverse_free:
            return super().integrated_nonconservative(full, raw_flux)
        left, right = self.reconstructed_primitives(full)
        evaluate = lambda pair: path_nc(pair[0], pair[1], self.fluid_g, self.fluid_l, self.virtual_mass)
        jumps = jax.lax.map(evaluate, (left, right))
        inside = jnp.zeros((self.mesh.n_cells, 6), full.dtype)
        if self.scheme.reconstruction != "constant":
            inside = jax.lax.map(evaluate, (right[:-1], left[1:]))
        return .5 * self._face_areas[1:, None] * jumps[1:] + .5 * self._face_areas[:-1, None] * jumps[:-1] + self._areas[:, None] * inside


def adapt(base, scheme, variant):
    if variant == "baseline":
        return base
    if base.stabilization.mass_upwind_cutoff is not None:
        raise ValueError("This scoped study requires the existing unmodified mass-upwind policy to be disabled")
    return AblationModel(base.mesh, base.fluid_g, base.fluid_l,
        initial_state=base.initial_state(), boundaries=base.boundaries, closures=base.closures,
        compatibility=base.compatibility, stabilization=base.stabilization, pins=base.pins,
        virtual_mass=base.virtual_mass, source_scale=base.source_scale, area_source=base.area_source,
        variant=variant, scheme=scheme)
