"""Model-independent Osher-type viscosity along a declared state path.

The caller supplies physical endpoint fluxes in one common face frame. Models
own conservative storage, characteristic projectors/speeds, thermodynamics and
an optional ``face_state_path(left, right, parameter)``. The default is affine
in reconstruction variables, not silently affine in conservative variables.

Fixed Gauss rules are approximations, not scalar entropy or system monotonicity
theorems. Sonic partitioning captures sign-changing brackets of lambda and
lambda +/- Harten width on eight intervals. It assumes captured simple roots,
consistent characteristic-family labels, an admissible hyperbolic path and
supported storage JVPs. Sampling cannot certify unseen crossings or every
point of an arbitrary path. No invalid state is clipped or replaced by HLL.
"""

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
import numpy as np


_GAUSS = {
    count: tuple(
        0.5 * (value + 1) if index == 0 else 0.5 * value
        for index, value in enumerate(np.polynomial.legendre.leggauss(count))
    )
    for count in (3, 5, 9)
}


def _face_data(model, state):
    fused = getattr(model, "face_data", None)
    if callable(fused):
        return fused(state)
    return dict(
        conserved=model.face_conserved(state),
        physical_flux=model.face_physical_flux(state),
        auxiliary=None,
    )


def _characteristics(model, state, delta_factor, data):
    fused = getattr(model, "face_characteristics_from_data", None)
    if callable(fused) and callable(getattr(model, "face_data", None)):
        return fused(state, state, delta_factor, data["auxiliary"], data["auxiliary"])
    return model.face_characteristics(state, state, delta_factor)


def _conserved(model, state):
    # An explicit callback may own a supported analytic storage differential.
    callback = getattr(model, "face_conserved", None)
    return (
        callback(state) if callable(callback) else _face_data(model, state)["conserved"]
    )


def _physical_flux(model, state):
    callback = getattr(model, "face_physical_flux", None)
    return (
        callback(state)
        if callable(callback)
        else _face_data(model, state)["physical_flux"]
    )


def _path_state(model, left, right, parameter):
    parameter = jnp.broadcast_to(parameter, left.shape[:-1])[..., None]
    callback = getattr(model, "face_state_path", None)
    if callable(callback):
        return callback(left, right, parameter)
    return left + parameter * (right - left), right - left


def _absolute_speed(eigenvalues, width):
    # Keep inactive division finite, including the exact zero-width limit.
    safe = jnp.where(width > 0, width, 1.0)
    inner = 0.5 * (eigenvalues**2 / safe + safe)
    return jnp.where(
        (width > 0) & (jnp.abs(eigenvalues) < width), inner, jnp.abs(eigenvalues)
    )


def _matrix_action(projectors, weights, vector):
    return jnp.einsum("...k,...kij,...j->...i", weights, projectors, vector)


@dataclass(frozen=True)
class OsherFlux:
    """Path viscosity with fixed Gauss quadrature or compact sonic panels.

    ``__call__(model, left, right, delta_factor, flux_left, flux_right)`` returns
    a single conservative numerical flux per face before frame conversion and
    area multiplication. States have shape ``(faces, variables)``. A declared
    path receives a parameter of shape ``(faces, 1)`` and returns state/tangent
    arrays of the same shape as the endpoint states. Nested differentiation of
    conservative storage must be supported by the physical model/backend.

    ``quadrature_points`` is 3, 5 (default) or 9 for either policy.
    ``integration='gauss'`` uses one unsplit rule; ``'sonic_partitioned'``
    applies the same rule on the captured sonic panels. Integration policy
    concerns the state-path integral, independently of time integration.

    Sonic roots use safeguarded Illinois/secant iterations, then bisection for
    unresolved brackets. A total capacity of four times the number of wave
    families is packed per face; this is not a per-family quota. Any overflow uses
    the complete candidate set and never drops a captured root. Root derivatives
    use the implicit-function rule; singular active interior roots produce a
    nonfinite derivative. Domain endpoints remain fixed integration cuts.
    """

    quadrature_points: int = 5
    integration: str = "gauss"
    intervals: int = 8
    bisections: int = 24
    fast_iterations: int = 8
    root_residual_relative_tolerance: float = 2.0**-24

    def __post_init__(self):
        if (
            type(self.quadrature_points) is not int
            or self.quadrature_points not in _GAUSS
        ):
            raise ValueError("Osher quadrature must use 3, 5 or 9 Gauss points")
        if not isinstance(self.integration, str) or self.integration not in (
            "gauss",
            "sonic_partitioned",
        ):
            raise ValueError("integration must be gauss or sonic_partitioned")
        if any(
            type(value) is not int or value < 1
            for value in (self.intervals, self.bisections)
        ):
            raise ValueError(
                "Sonic interval and bisection counts must be positive integers"
            )
        if type(self.fast_iterations) is not int or self.fast_iterations < 0:
            raise ValueError("Illinois iteration count must be a nonnegative integer")
        if (
            not math.isfinite(self.root_residual_relative_tolerance)
            or self.root_residual_relative_tolerance <= 0
        ):
            raise ValueError("Root residual tolerance must be positive and finite")

    @property
    def sonic_partition(self):
        """Internal branch selector derived from the explicit integration policy."""
        return self.integration == "sonic_partitioned"

    def crossing_data(self, model, left, right, delta_factor, *, compact=True):
        """Return traceable sampled brackets, roots and actual integration cuts.

        ``compact=False`` retains every sampled candidate and is the complete
        capacity-overflow fallback. This does not add sampling resolution.
        """
        grid = jnp.linspace(0.0, 1.0, self.intervals + 1, dtype=left.dtype)

        def eigen_data(l, r, parameter):
            state, _ = _path_state(model, l, r, parameter)
            data = _face_data(model, state)
            _, eigenvalues, width = _characteristics(model, state, delta_factor, data)
            return eigenvalues, width

        eigenvalues, widths = jax.vmap(lambda s: eigen_data(left, right, s))(grid)
        waves = eigenvalues.shape[-1]
        values = jnp.stack(
            (eigenvalues + widths, eigenvalues, eigenvalues - widths), axis=-1
        )
        shape = (self.intervals * waves * 3, left.shape[0])
        lower_values = jnp.transpose(values[:-1], (0, 2, 3, 1)).reshape(shape)
        upper_values = jnp.transpose(values[1:], (0, 2, 3, 1)).reshape(shape)
        zero_left, zero_right = lower_values == 0, upper_values == 0
        active = (
            (jnp.signbit(lower_values) != jnp.signbit(upper_values))
            & ~zero_left
            & ~zero_right
        ) | (zero_left ^ zero_right)
        families = jnp.tile(jnp.repeat(jnp.arange(waves), 3), self.intervals)
        levels = jnp.tile(
            jnp.asarray([-1.0, 0.0, 1.0], dtype=left.dtype), self.intervals * waves
        )
        lowers = jnp.repeat(grid[:-1], waves * 3)
        uppers = jnp.repeat(grid[1:], waves * 3)
        raw_counts = jnp.sum(active, axis=0)
        capacity = min(shape[0], 4 * waves) if compact else shape[0]
        if compact:
            indices = jax.vmap(
                lambda flags: jnp.nonzero(flags, size=capacity, fill_value=0)[0]
            )(active.T).T
            gather = lambda array: jnp.take_along_axis(array.T, indices.T, axis=1).T
            lower_values, upper_values = gather(lower_values), gather(upper_values)
            families, levels = jnp.take(families, indices), jnp.take(levels, indices)
            lowers, uppers = jnp.take(lowers, indices), jnp.take(uppers, indices)
            active = jnp.arange(capacity)[:, None] < raw_counts[None, :]
        else:
            indices = jnp.broadcast_to(jnp.arange(capacity)[:, None], shape)
            families, levels = (
                jnp.broadcast_to(array[:, None], shape) for array in (families, levels)
            )
            lowers, uppers = (
                jnp.broadcast_to(array[:, None], shape) for array in (lowers, uppers)
            )

        def g(l, r, parameter, family, level):
            eigenvalues, width = eigen_data(l, r, parameter)
            return (
                jnp.take_along_axis(eigenvalues, family[:, None], axis=-1)[:, 0]
                - level * jnp.take_along_axis(width, family[:, None], axis=-1)[:, 0]
            )

        @jax.custom_jvp
        def solve(l, r, lower, upper, family, level, mask, fa0, fb0):
            def solve_active(_):
                a, b = lower, upper
                tolerance = self.root_residual_relative_tolerance * jnp.maximum(
                    abs(fa0), abs(fb0)
                )
                done = ~mask | (fa0 == 0) | (fb0 == 0)
                root = jnp.where(fa0 == 0, a, jnp.where(fb0 == 0, b, 0.5 * (a + b)))
                count = jnp.zeros_like(root)
                # Bounds, true values, Illinois weights, last moved side,
                # convergence flags, accepted root, per-face evaluation count.
                initial = (
                    a,
                    b,
                    fa0,
                    fb0,
                    fa0,
                    fb0,
                    jnp.zeros_like(root),
                    done,
                    root,
                    count,
                )

                def iteration(_, carry):
                    def advance(current):
                        aa, bb, fa, fb, wa, wb, last, finished, best, evaluations = (
                            current
                        )
                        denominator = wb - wa
                        safe = jnp.where(denominator != 0, denominator, 1.0)
                        secant = (aa * wb - bb * wa) / safe
                        valid = (
                            (denominator != 0)
                            & jnp.isfinite(secant)
                            & (secant > aa)
                            & (secant < bb)
                        )
                        point = jnp.where(valid, secant, 0.5 * (aa + bb))
                        value = g(l, r, point, family, level)
                        running = mask & ~finished
                        converged = (abs(value) <= tolerance) | (
                            (bb - aa) <= (upper - lower) * 2.0**-self.bisections
                        )
                        same = (
                            (fa != 0)
                            & (value != 0)
                            & (jnp.signbit(fa) == jnp.signbit(value))
                        )
                        move_a, move_b = running & same, running & ~same
                        next_wa = jnp.where(
                            move_a,
                            value,
                            jnp.where(move_b, jnp.where(last == -1, 0.5 * wa, fa), wa),
                        )
                        next_wb = jnp.where(
                            move_b,
                            value,
                            jnp.where(move_a, jnp.where(last == 1, 0.5 * wb, fb), wb),
                        )
                        return (
                            jnp.where(move_a, point, aa),
                            jnp.where(move_b, point, bb),
                            jnp.where(move_a, value, fa),
                            jnp.where(move_b, value, fb),
                            next_wa,
                            next_wb,
                            jnp.where(running, jnp.where(same, 1.0, -1.0), last),
                            finished | (running & converged),
                            jnp.where(running, point, best),
                            evaluations + running.astype(root.dtype),
                        )

                    return jax.lax.cond(
                        jnp.any(mask & ~carry[7]),
                        advance,
                        lambda current: current,
                        carry,
                    )

                a, b, fa, _, _, _, _, done, root, count = jax.lax.fori_loop(
                    0, self.fast_iterations, iteration, initial
                )
                fallback = mask & ~done

                def fallback_solve(current):
                    aa, bb, faa, best, evaluations = current

                    def bisect(_, carry):
                        aaa, bbb, faaa, accepted, number = carry
                        middle = 0.5 * (aaa + bbb)
                        value = g(l, r, middle, family, level)
                        same = (
                            (faaa != 0)
                            & (value != 0)
                            & (jnp.signbit(faaa) == jnp.signbit(value))
                        )
                        return (
                            jnp.where(fallback & same, middle, aaa),
                            jnp.where(fallback & ~same, middle, bbb),
                            jnp.where(fallback & same, value, faaa),
                            jnp.where(fallback, middle, accepted),
                            number + fallback.astype(root.dtype),
                        )

                    aa, bb, _, best, evaluations = jax.lax.fori_loop(
                        0, self.bisections, bisect, (aa, bb, faa, best, evaluations)
                    )
                    return jnp.where(fallback, 0.5 * (aa + bb), best), evaluations

                root, count = jax.lax.cond(
                    jnp.any(fallback),
                    fallback_solve,
                    lambda current: (current[3], current[4]),
                    (a, b, fa, root, count),
                )
                return jnp.where(mask, root, 1.0), count, fallback.astype(root.dtype)

            empty = lambda _: (
                jnp.ones(left.shape[0], dtype=left.dtype),
                jnp.zeros(left.shape[0], dtype=left.dtype),
                jnp.zeros(left.shape[0], dtype=left.dtype),
            )
            return jax.lax.cond(jnp.any(mask), solve_active, empty, operand=None)

        @solve.defjvp
        def solve_jvp(primals, tangents):
            l, r, lower, upper, family, level, mask, fa0, fb0 = primals
            dl, dr, *_ = tangents
            root, count, fallback = solve(*primals)

            def implicit_derivative(_):
                _, state_derivative = jax.jvp(
                    lambda a, b: g(a, b, root, family, level), (l, r), (dl, dr)
                )
                _, parameter_derivative = jax.jvp(
                    lambda s: g(l, r, s, family, level), (root,), (jnp.ones_like(root),)
                )
                safe = jnp.where(parameter_derivative != 0, parameter_derivative, 1.0)
                derivative = jnp.where(mask, -state_derivative / safe, 0.0)
                derivative = jnp.where(
                    mask & (parameter_derivative == 0), jnp.nan, derivative
                )
                return jnp.where((root <= 0) | (root >= 1), 0.0, derivative)

            derivative = jax.lax.cond(
                jnp.any(mask),
                implicit_derivative,
                lambda _: jnp.zeros_like(root),
                operand=None,
            )
            return (root, count, fallback), (
                derivative,
                jnp.zeros_like(count),
                jnp.zeros_like(fallback),
            )

        roots, count, fallback = jax.lax.map(
            lambda args: solve(left, right, *args),
            (lowers, uppers, families, levels, active, lower_values, upper_values),
        )
        cuts = jnp.sort(
            jnp.concatenate(
                (
                    jnp.zeros((1, left.shape[0]), dtype=left.dtype),
                    roots,
                    jnp.ones((1, left.shape[0]), dtype=left.dtype),
                ),
                axis=0,
            ),
            axis=0,
        )
        return dict(
            cuts=cuts,
            roots=roots,
            active_brackets=active,
            grid=grid,
            grid_eigenvalues=eigenvalues,
            grid_widths=widths,
            root_families=families,
            root_levels=levels,
            root_evaluations_per_candidate=count,
            root_fallback_per_candidate=fallback,
            packed_candidate_indices=indices,
            raw_active_bracket_count_per_face=raw_counts,
            capacity=capacity,
            crossing_overflow=raw_counts > capacity,
        )

    def _integral(
        self, model, left, right, delta_factor, crossings=None, *, diagnostic=False
    ):
        nodes, weights = (
            jnp.asarray(value, dtype=left.dtype)
            for value in _GAUSS[self.quadrature_points]
        )

        def evaluate(parameter):
            state, tangent = _path_state(model, left, right, parameter)
            data = _face_data(model, state)
            projectors, eigenvalues, width = _characteristics(
                model, state, delta_factor, data
            )
            _, derivative = jax.jvp(
                lambda point: _conserved(model, point), (state,), (tangent,)
            )
            absolute = _matrix_action(
                projectors, _absolute_speed(eigenvalues, width), derivative
            )
            if diagnostic:
                signed = _matrix_action(projectors, eigenvalues, derivative)
                _, flux_derivative = jax.jvp(
                    lambda point: _physical_flux(model, point), (state,), (tangent,)
                )
                return jnp.stack((absolute, signed, flux_derivative))
            return absolute

        if crossings is None:
            values = jax.vmap(evaluate)(nodes)
            return jnp.einsum("q,q...i->...i", weights, values)
        # Viscosity acts on transported contents, which can have fewer
        # components than constrained iteration states.
        zeros = jnp.zeros_like(_conserved(model, left))
        empty = jnp.stack((zeros, zeros, zeros)) if diagnostic else zeros

        def panel(total, bounds):
            lower, upper = bounds
            span = upper - lower

            def integrate(_):
                values = jax.vmap(lambda node: evaluate(lower + node * span))(nodes)
                if diagnostic:
                    return (
                        jnp.einsum("q,qbfi->bfi", weights, values) * span[None, :, None]
                    )
                return jnp.einsum("q,qfi->fi", weights, values) * span[:, None]

            value = jax.lax.cond(
                jnp.any(span > 0), integrate, lambda _: empty, operand=None
            )
            return total + value, None

        return jax.lax.scan(
            panel, empty, (crossings["cuts"][:-1], crossings["cuts"][1:])
        )[0]

    def __call__(self, model, left, right, delta_factor, flux_left, flux_right):
        if not self.sonic_partition:
            viscosity = self._integral(model, left, right, delta_factor)
        else:
            compact = self.crossing_data(model, left, right, delta_factor)

            def full(_):
                crossings = self.crossing_data(
                    model, left, right, delta_factor, compact=False
                )
                return self._integral(model, left, right, delta_factor, crossings)

            viscosity = jax.lax.cond(
                jnp.any(compact["crossing_overflow"]),
                full,
                lambda _: self._integral(model, left, right, delta_factor, compact),
                operand=None,
            )
        return 0.5 * (flux_left + flux_right - viscosity)

    def path_admissible(self, model, full, left, right, delta_factor):
        """Check actual Gauss nodes; this is not proof of whole-path validity.

        This separate check repeats sonic root construction. Model-invalid nodes
        are rejected before characteristic evaluation. Both flux and validity
        preserve each physical callback's original ``(faces, variables)`` shape.
        """
        nodes = jnp.asarray(_GAUSS[self.quadrature_points][0], dtype=left.dtype)

        def valid_node(parameter):
            state, tangent = _path_state(model, left, right, parameter)
            finite = jnp.all(jnp.isfinite(state)) & jnp.all(jnp.isfinite(tangent))
            physical = jax.lax.cond(
                finite,
                lambda: jnp.all(model.admissible(full, state, state)),
                lambda: jnp.asarray(False),
            )

            def characteristic_valid():
                data = _face_data(model, state)
                projectors, eigenvalues, width = _characteristics(
                    model, state, delta_factor, data
                )
                return (
                    jnp.all(jnp.isfinite(data["conserved"]))
                    & jnp.all(jnp.isfinite(data["physical_flux"]))
                    & jnp.all(jnp.isfinite(projectors))
                    & jnp.all(jnp.isfinite(eigenvalues))
                    & jnp.all(jnp.isfinite(width) & (width >= 0))
                )

            return jax.lax.cond(
                physical, characteristic_valid, lambda: jnp.asarray(False)
            )

        def panel_valid(crossings):
            def panel(passed, bounds):
                lower, upper = bounds
                span = upper - lower
                valid = jax.lax.cond(
                    jnp.any(span > 0),
                    lambda: jnp.all(
                        jax.lax.map(lambda node: valid_node(lower + node * span), nodes)
                    ),
                    lambda: jnp.asarray(True),
                )
                return passed & valid, None

            return jax.lax.cond(
                jnp.all(jnp.isfinite(crossings["cuts"])),
                lambda: jax.lax.scan(
                    panel,
                    jnp.asarray(True),
                    (crossings["cuts"][:-1], crossings["cuts"][1:]),
                )[0],
                lambda: jnp.asarray(False),
            )

        if not self.sonic_partition:
            return jnp.all(jax.lax.map(valid_node, nodes))
        compact = self.crossing_data(model, left, right, delta_factor)
        return jax.lax.cond(
            jnp.any(compact["crossing_overflow"]),
            lambda: panel_valid(
                self.crossing_data(model, left, right, delta_factor, compact=False)
            ),
            lambda: panel_valid(compact),
        )

    def max_sampled_speed(self, model, left, right, delta_factor):
        """Return the largest sampled regularized speed for each face.

        Include both endpoints, actual viscosity quadrature nodes and, for the
        sonic method, the root-search grid. Harten viscosity can exceed the
        physical absolute wave speed, so the latter alone is insufficient for
        an explicit timestep guard. This is a sampled spectral-viscosity bound,
        not a certified supremum between nodes or a system positivity theorem.
        Nonfinite/negative-width characteristic data remain nonfinite failures.
        A separate call repeats sonic root construction and sampled physics.
        """
        nodes = jnp.asarray(_GAUSS[self.quadrature_points][0], dtype=left.dtype)

        def regularized_bound(eigenvalues, widths):
            valid = jnp.all(
                jnp.isfinite(eigenvalues) & jnp.isfinite(widths) & (widths >= 0),
                axis=-1,
            )
            maximum = jnp.max(_absolute_speed(eigenvalues, widths), axis=-1)
            return jnp.where(valid, maximum, jnp.nan)

        def at_parameter(parameter):
            state, _ = _path_state(model, left, right, parameter)
            data = _face_data(model, state)
            _, eigenvalues, widths = _characteristics(model, state, delta_factor, data)
            return regularized_bound(eigenvalues, widths)

        endpoints = jnp.maximum(at_parameter(0.0), at_parameter(1.0))
        if not self.sonic_partition:
            return jnp.maximum(
                endpoints, jnp.max(jax.vmap(at_parameter)(nodes), axis=0)
            )

        def with_crossings(crossings):
            grid_bound = jnp.max(
                regularized_bound(
                    crossings["grid_eigenvalues"], crossings["grid_widths"]
                ),
                axis=0,
            )

            def panel(maximum, bounds):
                lower, upper = bounds
                span = upper - lower
                value = jax.lax.cond(
                    jnp.any(span > 0),
                    lambda: jnp.max(
                        jax.vmap(lambda node: at_parameter(lower + node * span))(nodes),
                        axis=0,
                    ),
                    lambda: jnp.zeros(left.shape[0], dtype=left.dtype),
                )
                return jnp.maximum(maximum, value), None

            maximum = jax.lax.scan(
                panel,
                jnp.maximum(endpoints, grid_bound),
                (crossings["cuts"][:-1], crossings["cuts"][1:]),
            )[0]
            return jnp.where(
                jnp.all(jnp.isfinite(crossings["cuts"]), axis=0), maximum, jnp.nan
            )

        compact = self.crossing_data(model, left, right, delta_factor)
        return jax.lax.cond(
            jnp.any(compact["crossing_overflow"]),
            lambda: with_crossings(
                self.crossing_data(model, left, right, delta_factor, compact=False)
            ),
            lambda: with_crossings(compact),
        )

    def diagnostics(self, model, left, right, delta_factor):
        """Host-side audit with actual fallback cuts and finite-jump defects."""
        crossings = None
        metadata = {}
        if self.sonic_partition:
            packed = self.crossing_data(model, left, right, delta_factor)
            overflow = bool(jnp.any(packed["crossing_overflow"]))
            crossings = (
                self.crossing_data(model, left, right, delta_factor, compact=False)
                if overflow
                else packed
            )
            metadata = dict(
                **crossings,
                compact_capacity=packed["capacity"],
                compact_overflow=overflow,
                intervals=self.intervals,
                bisections=self.bisections,
                fast_iterations=self.fast_iterations,
                root_residual_relative_tolerance=self.root_residual_relative_tolerance,
                panel_count_per_face=jnp.sum(
                    crossings["cuts"][1:] > crossings["cuts"][:-1], axis=0
                )
            )
        absolute, signed, flux_derivative = self._integral(
            model, left, right, delta_factor, crossings, diagnostic=True
        )
        flux_jump = _physical_flux(model, right) - _physical_flux(model, left)
        return dict(
            **metadata,
            quadrature_points=self.quadrature_points,
            integration=self.integration,
            path=(
                "model callback"
                if callable(getattr(model, "face_state_path", None))
                else "straight reconstruction-variable segment"
            ),
            dissipation=absolute,
            path_A_dU=signed,
            path_dF=flux_derivative,
            physical_flux_jump=flux_jump,
            finite_jump_quadrature_defect=signed - flux_jump,
            differential_characteristic_integral_defect=signed - flux_derivative,
            path_values_finite=jnp.all(jnp.isfinite(absolute))
            & jnp.all(jnp.isfinite(signed)),
            assumptions="Captured simple crossings; admissible hyperbolic path; consistent families and supported storage JVPs. Sampled-node checks do not certify the whole path."
        )
