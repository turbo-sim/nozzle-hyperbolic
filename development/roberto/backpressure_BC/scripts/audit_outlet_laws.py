"""Executable value/AD audit of the experimental local outlet laws.

This is study verification, not a production pytest test. It invokes no FVM
time integration. The original-map stand-in copies the nearest interior state,
letting the audit isolate new local wave physics from source extrapolation.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from development.roberto.backpressure_BC.scripts.outlet_laws import make_outlet_law

jax.config.update("jax_enable_x64", True)


def _flux(state, gamma):
    pressure, velocity, enthalpy = np.asarray(state)
    density = gamma * pressure / ((gamma - 1.0) * enthalpy)
    return np.array(
        [
            density * velocity,
            density * velocity**2 + pressure,
            density * velocity * (enthalpy + 0.5 * velocity**2),
        ]
    )


def _conserved(state, gamma):
    pressure, velocity, enthalpy = np.asarray(state)
    density = gamma * pressure / ((gamma - 1.0) * enthalpy)
    return np.array(
        [
            density,
            density * velocity,
            density * (enthalpy + 0.5 * velocity**2) - pressure,
        ]
    )


def run_boundary_audit(gamma=1.4):
    """Check local Rankine-Hugoniot balances, branch choices and AD finiteness."""
    fluid = SimpleNamespace(constants=SimpleNamespace(gamma=float(gamma)))
    pressure = 1e5
    density = 1.2
    enthalpy = gamma * pressure / ((gamma - 1.0) * density)
    sound = np.sqrt(gamma * pressure / density)
    scales = jnp.array([pressure, sound, enthalpy])
    original = lambda full: full.at[-1].set(full[-2])
    records = []

    def check(mach, ratio, variant, expected_mode):
        full = jnp.tile(jnp.array([pressure, mach * sound, enthalpy]), (5, 1))
        outlet, diagnostics = make_outlet_law(
            original, fluid, pressure * ratio, variant
        )
        completed = outlet(full)
        info = diagnostics(full)
        assert int(info["mode"]) == expected_mode, (variant, mach, ratio, info)
        assert np.isfinite(np.asarray(completed)).all()

        def normalized(point):
            return outlet(full.at[-2].set(point * scales))[-1] / scales

        initial = full[-2] / scales
        forward = np.asarray(jax.jacfwd(normalized)(initial))
        reverse = np.asarray(jax.jacrev(normalized)(initial))
        assert np.isfinite(forward).all() and np.isfinite(reverse).all()
        derivative_difference = float(np.max(np.abs(forward - reverse)))
        assert derivative_difference < 2e-12
        direction = jnp.array([0.2, -0.07, 0.11])
        _, tangent = jax.jvp(normalized, (initial,), (direction,))
        epsilon = 1e-6
        finite_difference = (
            normalized(initial + epsilon * direction)
            - normalized(initial - epsilon * direction)
        ) / (2 * epsilon)
        fd_difference = float(np.max(np.abs(tangent - finite_difference)))
        assert fd_difference < 2e-8
        record = dict(
            variant=variant,
            mach=mach,
            pressure_ratio=ratio,
            mode=int(info["mode"]),
            forward_reverse_difference=derivative_difference,
            directional_finite_difference_error=fd_difference,
            backflow=bool(info["backflow"]),
        )
        records.append(record)
        return np.asarray(full[-2]), np.asarray(completed[-1]), info

    for mach, ratio in ((0.4, 0.8), (1.001, 0.5), (2.2, 0.5)):
        upstream, downstream, _ = check(mach, ratio, "source", 0)
        np.testing.assert_array_equal(upstream, downstream)

    for mach in (1.05, 1.4, 2.2, 4.0):
        shock_ratio = 1 + 2 * gamma / (gamma + 1) * (mach**2 - 1)
        upstream, downstream, info = check(mach, 1.1 * shock_ratio, "pressure_wave", 3)
        speed = float(info["wave_speed"])
        flux_jump = _flux(downstream, gamma) - _flux(upstream, gamma)
        storage_jump = speed * (
            _conserved(downstream, gamma) - _conserved(upstream, gamma)
        )
        defect = flux_jump - storage_jump
        relative = float(
            np.max(np.abs(defect) / np.maximum(np.abs(_flux(upstream, gamma)), 1.0))
        )
        assert relative < 3e-15
        records[-1]["moving_shock_flux_relative_defect"] = relative
        assert speed < 0

    check(2.2, 1.2, "pressure_wave", 1)
    check(2.2, 0.5, "pressure_wave", 1)
    check(1.001, 1.0001, "pressure_wave", 1)
    check(1.001, 0.5, "pressure_wave", 1)
    check(0.4, 0.8, "pressure_wave", 4)
    check(0.4, 0.01, "pressure_wave", 5)
    check(0.4, 1.0, "pressure_wave", 4)
    check(0.4, 1.1, "pressure_wave", 3)
    # Wall guarding blocks attempted inflow without fabricating any
    # exterior temperature/entropy. Its compression wave must still satisfy
    # moving Rankine-Hugoniot balances and its trace has exact zero mass/energy flux.
    coefficient = gamma * (gamma + 1.0) / 2.0
    beta = (gamma - 1.0) / (gamma + 1.0)
    for mach in (0.05, 0.4, 1.2, 2.2, 4.0):
        wall_ratio = (
            1
            + 0.5 * coefficient * mach**2
            + 0.5
            * mach
            * np.sqrt(coefficient**2 * mach**2 + 4 * coefficient * (1 + beta))
        )
        upstream, wall, info = check(mach, 1.5 * wall_ratio, "pressure_wave", 7)
        assert wall[1] == 0 and bool(info["wall_guard_active"])
        assert bool(info["attempted_backflow"]) and not bool(info["backflow"])
        assert not bool(info["unsupported_backflow"]) and not bool(
            info["pressure_sensitive"]
        )
        speed = float(info["wave_speed"])
        defect = (
            _flux(wall, gamma)
            - _flux(upstream, gamma)
            - speed * (_conserved(wall, gamma) - _conserved(upstream, gamma))
        )
        relative = float(
            np.max(np.abs(defect) / np.maximum(np.abs(_flux(upstream, gamma)), 1.0))
        )
        assert relative < 3e-15
        assert _flux(wall, gamma)[0] == 0 and _flux(wall, gamma)[2] == 0
        records[-1]["wall_shock_flux_relative_defect"] = relative
    check(-0.1, 4.0, "pressure_wave", 6)

    # At rest the admissible velocity half-space ends at zero. Check AD on
    # the selected wall branch and a one-sided (positive velocity) difference,
    # not a central difference through an intentionally unsupported inflow.
    rest_full = jnp.tile(jnp.array([pressure, 0.0, enthalpy]), (5, 1))
    wall_outlet, wall_info = make_outlet_law(
        original, fluid, 4 * pressure, "pressure_wave"
    )
    rest_point = rest_full[-2] / scales

    def normalized_wall(point):
        return wall_outlet(rest_full.at[-2].set(point * scales))[-1] / scales

    forward = np.asarray(jax.jacfwd(normalized_wall)(rest_point))
    reverse = np.asarray(jax.jacrev(normalized_wall)(rest_point))
    expected = np.array([[1.0, gamma, 0.0], [0.0, 0.0, 0.0], [0.0, gamma - 1.0, 1.0]])
    np.testing.assert_allclose(forward, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(reverse, expected, rtol=2e-12, atol=2e-12)
    epsilon = 1e-7
    direction = jnp.array([0.0, 1.0, 0.0])
    _, tangent = jax.jvp(normalized_wall, (rest_point,), (direction,))
    finite_difference = (
        normalized_wall(rest_point + epsilon * direction) - normalized_wall(rest_point)
    ) / epsilon
    rest_fd_error = float(np.max(np.abs(tangent - finite_difference)))
    assert rest_fd_error < 1e-6 and int(wall_info(rest_full)["mode"]) == 7
    records.append(
        dict(
            variant="pressure_wave",
            mach=0.0,
            pressure_ratio=4.0,
            mode=7,
            forward_reverse_difference=float(np.max(np.abs(forward - reverse))),
            directional_finite_difference_error=rest_fd_error,
            finite_difference_policy="one-sided nonnegative normal velocity",
            backflow=False,
        )
    )

    # A genuine supersonic-outflow range must remain exactly pressure-insensitive.
    full = jnp.tile(jnp.array([pressure, 2.2 * sound, enthalpy]), (5, 1))
    first = make_outlet_law(original, fluid, 0.5 * pressure, "pressure_wave")[0](full)
    second = make_outlet_law(original, fluid, 1.2 * pressure, "pressure_wave")[0](full)
    np.testing.assert_array_equal(first, second)

    # A genuinely outgoing wave near sonic speed must not inherit the source
    # tanh blend's pressure sensitivity. Use a pressure-dependent source stand-
    # in here, so this audit would fail for the old original-map-only branch.
    full = jnp.tile(jnp.array([pressure, 1.001 * sound, enthalpy]), (5, 1))
    samples = []
    source_samples = []
    for ratio in (0.5, 0.8):
        prescribed = ratio * pressure

        def blended_source(state):
            mach = state[-2, 1] / jnp.sqrt((gamma - 1.0) * state[-2, 2])
            weight = 0.5 * (1 + jnp.tanh((1 - mach) / 0.004))
            ghost = (
                state[-2].at[0].set(weight * prescribed + (1 - weight) * state[-2, 0])
            )
            return state.at[-1].set(ghost)

        outlet, diagnostics = make_outlet_law(
            blended_source, fluid, prescribed, "pressure_wave"
        )
        samples.append(np.asarray(outlet(full)))
        source_samples.append(np.asarray(blended_source(full)))
        assert not bool(diagnostics(full)["pressure_sensitive"])
    assert not np.array_equal(source_samples[0], source_samples[1])
    np.testing.assert_array_equal(samples[0], samples[1])
    return dict(
        passed=True,
        gamma=float(gamma),
        local_cases=records,
        checks=[
            "unchanged source outlet",
            "moving shock conservation",
            "incoming/outgoing compression selection",
            "incoming/outgoing expansion and sonic fan selection",
            "JIT finite values",
            "finite forward/reverse AD and matching JVP/finite differences",
            "unsupported backflow reported",
            "legitimate outgoing pressure independence",
            "near-sonic outgoing pressure independence without inherited Mach blending",
            "wall guard moving-shock conservation and zero wall flux",
            "finite rest-state wall AD with correct acoustic slope",
            "negative interior velocity remains unsupported",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "boundary_audit.json",
    )
    args = parser.parse_args()
    result = run_boundary_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
