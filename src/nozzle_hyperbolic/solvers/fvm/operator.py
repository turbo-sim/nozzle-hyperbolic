"""Shared reconstruction, numerical flux and coupled implicit spatial operator.

This module knows neither Euler variables nor an EOS. Model callbacks supply
physical fluxes, characteristic frames, sources, boundary equations and domain.

The supported contract is a square conservative balance system on fixed cells,
with one independent state trace at each endpoint. Arrays are cell-major:
``full=(n+2,v)``, physical cells ``(n,v)``, face states/fluxes ``(n+1,v)``, and
scaled algebraic boundary rows ``(2,v)``. The model's iteration variables are
also its reconstruction variables. Callbacks must be JAX traceable.

``cell_conserved`` returns densities per physical volume. Face conserved states,
physical fluxes and optional characteristic projectors use one model-owned face
frame; ``flux_to_conserved_frame`` converts the numerical flux into transported
balances. The core multiplies that result by face area exactly once. In contrast,
``integrated_source(full)`` already includes its entire cell geometry measure.
Boundary rows are dimensionless, have zero storage, and are never dt-weighted.

An optional ``face_data(W)`` callback may share expensive physical evaluations.
It returns a dict with ``conserved``, ``physical_flux``, ``speed_bounds=(lo,hi)``
and opaque JAX-pytree ``auxiliary``. The optional characteristic callback
``face_characteristics_from_data(left,right,delta,aux_left,aux_right)`` reuses
that auxiliary data. Models without fused callbacks keep the simple independent
callbacks; scalar laws require neither auxiliary data nor characteristic data.

Models declare source and boundary dependency radii in full-state block indices.
The core unions those with its reconstruction stencil. Nonlocal boundary laws
declare ``boundary_coupling='global'`` and require dense AD and dense solution;
ordinary colored/banded assembly must not silently discard their corner blocks.
The shared FVM engine requires JAX 64-bit mode before binding:
its configured Newton and conservation tolerances are below float32 precision.
"""

from .config import Scheme, TimestepInfo
from .elimination import make_eliminated_implicit_step
import operator as index_operator

import jax
import jax.numpy as jnp
import numpy as np

from nozzle_hyperbolic.numerics.characteristic import (
    characteristic_flux_with_entropy_viscosity,
)
from nozzle_hyperbolic.numerics.jacobians import make_block_operators
from nozzle_hyperbolic.numerics.reconstruction import reconstruct_faces
from nozzle_hyperbolic.numerics.fluxes import rusanov, hll
from nozzle_hyperbolic.numerics.roe import roe, roe_absolute
from nozzle_hyperbolic.numerics.filters import fourth_difference_flux
from nozzle_hyperbolic.solvers.time_marching.implicit import make_newton_solver
from nozzle_hyperbolic.solvers.time_marching.explicit import make_explicit_step
from nozzle_hyperbolic.solvers.boundaries import make_boundary_solver


class Operator:
    def __init__(self, model, scheme=Scheme()):
        if not jax.config.x64_enabled:
            raise ValueError(
                "Enable jax_enable_x64 before binding the implicit operator"
            )
        self.model, self.mesh, self.scheme = model, model.mesh, scheme
        self.effective_boundary_treatment = scheme.effective_boundary_treatment
        n, nvar = model.mesh.n_cells, model.nvar
        if (
            isinstance(nvar, bool)
            or not isinstance(nvar, (int, np.integer))
            or nvar < 1
            or len(model.equation_names) != nvar
        ):
            raise ValueError(
                "A model must declare its positive equation count and names"
            )
        if getattr(model, "n_equations", nvar) != nvar:
            raise ValueError(
                "The current conservative interface requires a square system"
            )
        radius = (
            1
            if scheme.reconstruction == "constant" and scheme.filter_strength == 0
            else 2
        )
        for name in ("source_dependency_radius", "boundary_dependency_radius"):
            value = getattr(model, name, None)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 0
            ):
                raise ValueError(f"Model must declare a nonnegative integer {name}")
            radius = max(radius, int(value))
        coupling = getattr(model, "boundary_coupling", None)
        if coupling not in ("local", "global"):
            raise ValueError("Model must declare boundary_coupling='local' or 'global'")
        if coupling == "global" and (
            scheme.jacobian_mode != "dense" or scheme.linear_solver != "dense"
        ):
            raise ValueError(
                "Global boundary coupling requires dense Jacobian and dense linear solver"
            )
        self.radius = min(radius, n + 1)
        for name in ("conservative_scales", "variable_scales", "global_flux_scales"):
            values = np.asarray(getattr(model, name))
            if values.shape != (nvar,) or not np.all(
                np.isfinite(values) & (values > 0)
            ):
                raise ValueError(
                    f"{name} must be one positive finite scale per equation"
                )
        if not np.isfinite(model.reference_time) or model.reference_time <= 0:
            raise ValueError("Positive fixed reference time required")
        fused_data = callable(getattr(model, "face_data", None))
        fused_characteristics = fused_data and callable(
            getattr(model, "face_characteristics_from_data", None)
        )
        fused_roe = fused_data and callable(
            getattr(model, "face_roe_data_from_data", None)
        )
        if (
            scheme.flux == "characteristic"
            and not fused_characteristics
            and not callable(getattr(model, "face_characteristics", None))
        ):
            raise ValueError("This model does not supply characteristic flux data")
        if (
            scheme.flux == "roe"
            and not fused_roe
            and not callable(getattr(model, "face_roe_data", None))
        ):
            raise ValueError("This model does not supply Roe flux data")
        self.shape = (n + 2, nvar)
        # Trace shapes once at binding, not during Newton iterations. This
        # catches accidental broadcasting and nonsquare callback contracts.
        dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
        full_abstract = jax.ShapeDtypeStruct(self.shape, dtype)
        cell_abstract = jax.ShapeDtypeStruct((n, nvar), dtype)
        face_abstract = jax.ShapeDtypeStruct((n + 1, nvar), dtype)

        def shape_of(name, *args):
            callback = getattr(model, name, None)
            if not callable(callback):
                raise ValueError(f"Model must supply callable {name}")
            return jax.eval_shape(callback, *args)

        def require_shape(value, expected, name):
            if not hasattr(value, "shape") or value.shape != expected:
                raise ValueError(f"{name} must return shape {expected}")

        for name, arg, expected in (
            ("cell_conserved", cell_abstract, (n, nvar)),
            ("flux_to_conserved_frame", face_abstract, (n + 1, nvar)),
            ("integrated_source", full_abstract, (n, nvar)),
            ("wave_speed", cell_abstract, (n,)),
        ):
            require_shape(shape_of(name, arg), expected, name)

        def require_bounds(bounds):
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                raise ValueError(
                    "Face speed bounds must return lower and upper speed vectors"
                )
            for bound in bounds:
                require_shape(bound, (n + 1,), "face speed bounds")

        if fused_data:
            data = shape_of("face_data", face_abstract)
            required = {"conserved", "physical_flux", "speed_bounds", "auxiliary"}
            if not isinstance(data, dict) or not required.issubset(data):
                raise ValueError(
                    f"face_data must return a dict containing {sorted(required)}"
                )
            require_shape(data["conserved"], (n + 1, nvar), "face_data conserved")
            require_shape(
                data["physical_flux"], (n + 1, nvar), "face_data physical_flux"
            )
            require_bounds(data["speed_bounds"])
        else:
            for name in ("face_conserved", "face_physical_flux"):
                require_shape(shape_of(name, face_abstract), (n + 1, nvar), name)
            if scheme.flux in ("rusanov", "hll"):
                require_bounds(shape_of("face_speed_bounds", face_abstract))
        require_shape(
            shape_of(
                "boundary_residual", full_abstract, jax.ShapeDtypeStruct((), dtype)
            ),
            (2, nvar),
            "boundary_residual",
        )
        boundary_flux = getattr(model, "boundary_face_flux", None)
        if boundary_flux is not None:
            require_shape(
                shape_of("boundary_face_flux", full_abstract, face_abstract),
                (n + 1, nvar),
                "boundary_face_flux",
            )
        if scheme.filter_strength > 0:
            require_shape(
                shape_of("filter_conserved", full_abstract),
                self.shape,
                "filter_conserved",
            )
            require_shape(
                shape_of("filter_signal_speed", full_abstract),
                (n + 1,),
                "filter_signal_speed",
            )
        source_timestep = getattr(model, "explicit_timestep", None)
        if source_timestep is not None:
            require_shape(
                shape_of("explicit_timestep", full_abstract), (), "explicit_timestep"
            )
        boundary_speed = getattr(model, "boundary_wave_speed", None)
        if boundary_speed is not None:
            require_shape(
                shape_of("boundary_wave_speed", full_abstract),
                (2,),
                "boundary_wave_speed",
            )
        admissible = shape_of("admissible", full_abstract, face_abstract, face_abstract)
        require_shape(admissible, (), "admissible")
        if admissible.dtype != jnp.bool_:
            raise ValueError("admissible must return a scalar boolean")
        if scheme.flux == "characteristic":
            if fused_characteristics:
                projectors, eigenvalues, width = shape_of(
                    "face_characteristics_from_data",
                    face_abstract,
                    face_abstract,
                    scheme.delta_factor,
                    data["auxiliary"],
                    data["auxiliary"],
                )
            else:
                projectors, eigenvalues, width = shape_of(
                    "face_characteristics",
                    face_abstract,
                    face_abstract,
                    scheme.delta_factor,
                )
            if (
                len(eigenvalues.shape) != 2
                or eigenvalues.shape[0] != n + 1
                or eigenvalues.shape[1] < 1
            ):
                raise ValueError(
                    "face_characteristics must return (faces,waves) eigenvalues"
                )
            require_shape(width, eigenvalues.shape, "characteristic width")
            require_shape(
                projectors,
                (*eigenvalues.shape, nvar, nvar),
                "characteristic projectors",
            )
        if scheme.flux == "roe":
            if fused_roe:
                roe_shapes = shape_of(
                    "face_roe_data_from_data",
                    face_abstract,
                    face_abstract,
                    scheme.delta_factor,
                    data["auxiliary"],
                    data["auxiliary"],
                )
            else:
                roe_shapes = shape_of(
                    "face_roe_data",
                    face_abstract,
                    face_abstract,
                    scheme.delta_factor,
                )
            required = {"projectors", "eigenvalues", "width", "valid"}
            if not isinstance(roe_shapes, dict) or not required.issubset(roe_shapes):
                raise ValueError(
                    f"Roe data must be a dict containing {sorted(required)}"
                )
            eigenvalues = roe_shapes["eigenvalues"]
            if (
                len(eigenvalues.shape) != 2
                or eigenvalues.shape[0] != n + 1
                or eigenvalues.shape[1] < 1
            ):
                raise ValueError("Roe eigenvalues must return (faces,waves)")
            require_shape(
                roe_shapes["projectors"],
                (*eigenvalues.shape, nvar, nvar),
                "Roe projectors",
            )
            require_shape(roe_shapes["width"], eigenvalues.shape, "Roe widths")
            require_shape(roe_shapes["valid"], (n + 1,), "Roe validity")
            if roe_shapes["valid"].dtype != jnp.bool_:
                raise ValueError("Roe validity must be a boolean face vector")
        points = jnp.asarray(self.mesh.points)
        faces = jnp.asarray(self.mesh.faces)
        area = jnp.asarray(self.mesh.area_faces)
        volumes = jnp.asarray(self.mesh.volumes)
        qscale = jnp.asarray(model.conservative_scales)
        variable_scale = jnp.broadcast_to(
            jnp.asarray(model.variable_scales), self.shape
        ).ravel()
        content_scale = volumes[:, None] * qscale
        reference_time = float(model.reference_time)

        def reconstruct(full):
            return reconstruct_faces(
                full,
                points,
                faces,
                scheme.reconstruction,
                variable_scales=model.variable_scales,
                smooth_coefficient=scheme.smooth_limiter_coefficient,
            )

        def roe_data(left, right, left_data=None, right_data=None):
            if fused_roe:
                if left_data is None:
                    left_data, right_data = model.face_data(left), model.face_data(
                        right
                    )
                return model.face_roe_data_from_data(
                    left,
                    right,
                    scheme.delta_factor,
                    left_data["auxiliary"],
                    right_data["auxiliary"],
                )
            return model.face_roe_data(left, right, scheme.delta_factor)

        def roe_data_valid(data):
            return (
                jnp.all(data["valid"])
                & jnp.all(jnp.isfinite(data["eigenvalues"]))
                & jnp.all(jnp.isfinite(data["projectors"]))
                & jnp.all(jnp.isfinite(data["width"]) & (data["width"] >= 0))
            )

        def face_flux(full):
            left, right = reconstruct(full)
            if fused_data:
                left_data, right_data = model.face_data(left), model.face_data(right)
                ql, qr = left_data["conserved"], right_data["conserved"]
                fl, fr = left_data["physical_flux"], right_data["physical_flux"]
            else:
                ql, qr = model.face_conserved(left), model.face_conserved(right)
                fl, fr = model.face_physical_flux(left), model.face_physical_flux(right)
            if scheme.flux == "characteristic":
                if fused_characteristics:
                    projectors, eigenvalues, width = (
                        model.face_characteristics_from_data(
                            left,
                            right,
                            scheme.delta_factor,
                            left_data["auxiliary"],
                            right_data["auxiliary"],
                        )
                    )
                else:
                    projectors, eigenvalues, width = model.face_characteristics(
                        left, right, scheme.delta_factor
                    )
                flux = characteristic_flux_with_entropy_viscosity(
                    fl, fr, qr - ql, projectors, eigenvalues, width
                )
            elif scheme.flux == "roe":
                data = (
                    roe_data(left, right, left_data, right_data)
                    if fused_data
                    else roe_data(left, right)
                )
                flux = roe(
                    fl,
                    fr,
                    qr - ql,
                    data["projectors"],
                    data["eigenvalues"],
                    data["width"],
                )
            else:
                if fused_data:
                    low_l, high_l = left_data["speed_bounds"]
                    low_r, high_r = right_data["speed_bounds"]
                else:
                    low_l, high_l = model.face_speed_bounds(left)
                    low_r, high_r = model.face_speed_bounds(right)
                numerical = rusanov if scheme.flux == "rusanov" else hll
                flux = numerical(fl, fr, qr - ql, (low_l, high_l), (low_r, high_r))
            flux = model.flux_to_conserved_frame(flux)
            if scheme.filter_strength > 0:
                flux = fourth_difference_flux(
                    flux,
                    model.filter_conserved(full),
                    model.filter_signal_speed(full),
                    scheme.filter_strength,
                )
            if boundary_flux is not None:
                flux = boundary_flux(full, flux)
            return area[:, None] * flux

        def source(full):
            return model.integrated_source(full)

        def storage(physical):
            return volumes[:, None] * model.cell_conserved(physical)

        def spatial(full):
            flux = face_flux(full)
            return flux[1:] - flux[:-1] - source(full)

        def pack(cells, boundary):
            return jnp.concatenate([boundary[:1], cells, boundary[1:]]).ravel()

        def physical_residual(new, old, dt):
            # Uniform scaling prevents huge pseudo-time steps from amplifying
            # roundoff in the implicit residual; the zero set is unchanged.
            row_scale = jnp.maximum(1.0, dt / reference_time)
            return (
                (storage(new[1:-1]) - storage(old[1:-1])) / row_scale
                + (dt / row_scale) * spatial(new)
            ) / content_scale

        def residual(new_flat, old_flat, dt, time):
            new, old = new_flat.reshape(self.shape), old_flat.reshape(self.shape)
            return pack(
                physical_residual(new, old, dt), model.boundary_residual(new, time)
            )

        def steady(full_flat, time=0.0):
            full = full_flat.reshape(self.shape)
            return pack(
                reference_time * spatial(full) / content_scale,
                model.boundary_residual(full, time),
            )

        def valid(flat):
            full = flat.reshape(self.shape)
            left, right = reconstruct(full)
            finite = (
                jnp.all(jnp.isfinite(full))
                & jnp.all(jnp.isfinite(left))
                & jnp.all(jnp.isfinite(right))
            )
            physical = jax.lax.cond(
                finite,
                lambda: model.admissible(full, left, right),
                lambda: jnp.asarray(False),
            )
            if scheme.flux == "roe":
                return jax.lax.cond(
                    physical,
                    lambda: roe_data_valid(roe_data(left, right)),
                    lambda: jnp.asarray(False),
                )
            return physical

        block = make_block_operators(
            residual,
            n + 2,
            nvar,
            self.radius,
            jacobian_mode=scheme.jacobian_mode,
            linear_solver=scheme.linear_solver,
        )
        newton = make_newton_solver(
            residual,
            block["direction"],
            lambda _: variable_scale,
            max_iterations=scheme.newton_iterations,
            tolerance=scheme.newton_update_tolerance,
            residual_tolerance=scheme.newton_residual_tolerance,
            admissible=valid,
            backtracking=scheme.newton_backtracking,
            max_backtracks=scheme.max_backtracks,
        )

        def implicit_step(full, dt, time=0.0):
            result, info = newton(full.ravel(), full.ravel(), dt, time)
            return result.reshape(self.shape), info

        boundary_valid = getattr(
            model, "boundary_admissible", lambda full: valid(full.ravel())
        )
        complete = make_boundary_solver(
            model.boundary_residual,
            self.shape,
            model.variable_scales,
            admissible=boundary_valid,
            max_iterations=scheme.newton_iterations,
            update_tolerance=scheme.newton_update_tolerance,
            residual_tolerance=scheme.newton_residual_tolerance,
            max_backtracks=scheme.max_backtracks,
        )
        if (
            scheme.time_integrator == "euler-implicit"
            and self.effective_boundary_treatment == "segregated"
        ):
            reduced = make_eliminated_implicit_step(
                boundary_residual=model.boundary_residual,
                complete_boundaries=complete,
                physical_residual=physical_residual,
                coupled_residual=residual,
                admissible=valid,
                shape=self.shape,
                variable_scales=model.variable_scales,
                radius=self.radius,
                scheme=scheme,
            )
            implicit_step = reduced["step"]
            self.reduced_residual = jax.jit(reduced["residual"])
            self.reduced_jacobian = jax.jit(reduced["jacobian"])
            self.reduced_dense_jacobian = jax.jit(reduced["dense_jacobian"])
            self.reduced_direction = jax.jit(reduced["direction"])

        def recover_residual(flat, target):
            return ((storage(flat.reshape(n, nvar)) - target) / content_scale).ravel()

        recover_blocks = make_block_operators(
            recover_residual,
            n,
            nvar,
            0,
            jacobian_mode="colored",
            linear_solver="banded",
        )

        def recover_valid(flat):
            cells = flat.reshape(n, nvar)
            predicate = getattr(model, "cell_admissible", None)
            if predicate is not None:
                return predicate(cells)
            full = jnp.concatenate((cells[:1], cells, cells[-1:]), axis=0)
            return valid(full.ravel())

        recover_newton = make_newton_solver(
            recover_residual,
            recover_blocks["direction"],
            lambda _: jnp.broadcast_to(
                jnp.asarray(model.variable_scales), (n, nvar)
            ).ravel(),
            max_iterations=scheme.newton_iterations,
            tolerance=scheme.newton_update_tolerance,
            residual_tolerance=scheme.newton_residual_tolerance,
            admissible=recover_valid,
            backtracking=True,
            max_backtracks=scheme.max_backtracks,
        )

        def recover(target, guess):
            flat, info = recover_newton(guess.ravel(), target)
            return flat.reshape(n, nvar), info

        def balance(full):
            flux = face_flux(full)
            return flux[0] - flux[-1] + jnp.sum(source(full), axis=0)

        def stage_rates(full):
            flux = face_flux(full)
            integrated = source(full)
            return flux[1:] - flux[:-1] - integrated, flux[0] - flux[-1] + jnp.sum(
                integrated, axis=0
            )

        if scheme.time_integrator == "euler-implicit":
            step = implicit_step

            def with_balance(full, dt, time=0.0):
                new, info = step(full, dt, time)
                return new, info, dt * balance(new)

        else:
            with_balance = make_explicit_step(
                storage,
                spatial,
                recover,
                complete,
                balance,
                lambda full: valid(full.ravel()),
                method=scheme.time_integrator,
                evaluate_stage=stage_rates,
            )

            def step(full, dt, time=0.0):
                new, info, _ = with_balance(full, dt, time)
                return new, info

        self.reconstruct = jax.jit(reconstruct)
        self.face_flux = jax.jit(face_flux)
        self.source = jax.jit(source)
        self.storage = jax.jit(storage)
        self.spatial_residual = jax.jit(spatial)
        self.residual = jax.jit(residual)
        self.steady_residual = jax.jit(steady)
        self.boundary_residual = jax.jit(model.boundary_residual)
        self.is_admissible = jax.jit(lambda full: valid(full.ravel()))
        self.step = jax.jit(step)
        self.step_with_balance = jax.jit(with_balance)
        self.complete_boundaries = complete
        self.recover_storage = jax.jit(recover)
        if scheme.flux == "roe":
            self.roe_data = jax.jit(lambda full: roe_data(*reconstruct(full)))

        def timestep(full, cfl, time=0.0):
            completion_ok = jnp.asarray(True)
            explicit = scheme.time_integrator != "euler-implicit"
            if explicit:
                full, info = complete(full, time)
                completion_ok = info.converged & info.admissible

            def evaluate(_):
                speed = model.wave_speed(full[1:-1])
                base = jnp.min(
                    jnp.asarray(self.mesh.widths) / jnp.maximum(speed, 1e-300)
                )
                okay = (
                    jnp.all(jnp.isfinite(speed) & (speed >= 0))
                    & jnp.isfinite(cfl)
                    & (cfl > 0)
                )
                if explicit and scheme.flux == "roe":
                    # Interface Roe speeds can exceed cell physical speeds,
                    # e.g. a large transverse-velocity jump in an annular face.
                    # Bound the actual selected numerical dissipation using
                    # callback data, without knowing any physical variables.
                    data = roe_data(*reconstruct(full))
                    face_speed = jnp.max(
                        roe_absolute(data["eigenvalues"], data["width"]), axis=-1
                    )
                    local_speed = jnp.maximum(face_speed[:-1], face_speed[1:])
                    base = jnp.minimum(
                        base,
                        jnp.min(
                            jnp.asarray(self.mesh.widths)
                            / jnp.maximum(local_speed, 1e-300)
                        ),
                    )
                    okay = okay & roe_data_valid(data)
                if boundary_speed is not None:
                    bounds = boundary_speed(full)
                    base = jnp.minimum(
                        base,
                        jnp.min(
                            jnp.asarray(self.mesh.widths)[jnp.array([0, -1])]
                            / jnp.maximum(bounds, 1e-300)
                        ),
                    )
                    okay = okay & jnp.all(jnp.isfinite(bounds) & (bounds >= 0))
                source_limit = jnp.asarray(jnp.inf, full.dtype)
                filter_limit = jnp.asarray(jnp.inf, full.dtype)
                if explicit and source_timestep is not None:
                    source_limit = source_timestep(full)
                    okay = okay & (~jnp.isnan(source_limit)) & (source_limit > 0)
                if explicit and scheme.filter_strength > 0:
                    weights = model.filter_signal_speed(full)
                    local = jnp.maximum(weights[:-1], weights[1:])
                    filter_limit = jnp.min(
                        jnp.asarray(self.mesh.widths)
                        / jnp.maximum(8.0 * scheme.filter_strength * local, 1e-300)
                    )
                    okay = okay & jnp.all(jnp.isfinite(weights) & (weights >= 0))
                wave_limit = cfl * base
                value = wave_limit
                if explicit:
                    # Separate stable bounds cannot simply be minimized when
                    # their decay rates add in one Euler/RK stage. Combine the
                    # requested wave-CFL bound, absolute source bound and
                    # explicit filter bound as inverse times. Scale by the
                    # smallest bound to avoid overflow for very short times;
                    # infinity contributes zero rate. Invalid inputs retain
                    # their diagnostic values but cannot enter unsafe ratios.
                    bounds = jnp.stack((wave_limit, source_limit, filter_limit))
                    safe_bounds = jnp.where(
                        (bounds > 0) & ~jnp.isnan(bounds), bounds, jnp.inf
                    )
                    minimum = jnp.min(safe_bounds)
                    scale = jnp.where(jnp.isfinite(minimum), minimum, 1.0)
                    rate_sum = jnp.sum(scale / safe_bounds)
                    value = jnp.where(
                        rate_sum > 0,
                        scale / jnp.where(rate_sum > 0, rate_sum, 1.0),
                        jnp.inf,
                    )
                okay = okay & ~jnp.isnan(value) & (value > 0)
                return TimestepInfo(
                    jnp.where(okay, value, jnp.nan),
                    base,
                    source_limit,
                    filter_limit,
                    okay,
                )

            def failed(_):
                nan = jnp.asarray(jnp.nan, full.dtype)
                return TimestepInfo(nan, nan, nan, nan, jnp.asarray(False))

            return jax.lax.cond(completion_ok, evaluate, failed, operand=None)

        self.timestep = jax.jit(timestep)
        self.compute_dt = jax.jit(
            lambda full, cfl, time=0.0: timestep(full, cfl, time).dt
        )
        self.direction = jax.jit(block["direction"])
        self.jacobian = jax.jit(block["jacobian"])
        self.dense_jacobian = jax.jit(block["dense_jacobian"])
        self.jacobian_difference = jax.jit(block["jacobian_difference"])
