"""The host bridge uses perfect-gas data only; no real-fluid flow campaign."""

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time

import jax
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np
from numpy.testing import assert_allclose
import pytest

from nozzle_hyperbolic.thermodynamics.adapters import as_jax_fluid


class HostPerfectGas(jxp.Fluid):
    """Configured host-only Fluid interface backed by actual JAXProp perfect gas.

    Deliberately do not initialize a CoolProp fluid. This fixture exercises the
    raw-Fluid adapter dispatch while preserving the perfect-gas test scope.
    """

    def __init__(self):
        self.native = jxp.FluidPerfectGas(name="Air")
        self.name = "configured perfect gas"
        self.backend = "host-only test backend"
        self.identifier = "original identifier"
        self.phase = "original phase policy"
        self.exceptions = True
        self.p_min, self.p_max = 100.0, 1e7
        self.T_min, self.T_max = 100.0, 1000.0
        self.configuration = {"selected_option": "preserved"}
        self.active = self.peak_active = 0
        self.calls = []
        self.guard = Lock()
        self.delay = 0.0
        self.error = None

    def get_state(self, pair, x, y):
        if isinstance(x, jax.core.Tracer) or isinstance(y, jax.core.Tracer):
            raise TypeError("This fixture only accepts host inputs")
        if self.error is not None:
            raise self.error
        x, y = float(x), float(y)
        if pair == jxp.PT_INPUTS and not (
            self.p_min <= x <= self.p_max and self.T_min <= y <= self.T_max
        ):
            raise ValueError("Outside the host perfect-gas domain")
        with self.guard:
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
        try:
            time.sleep(self.delay)
            self.calls.append((pair, x, y))
            state = self.native.get_state(pair, x, y)
            return jxp.FluidState(
                fluid_name=self.name,
                identifier=self.identifier,
                **{key: np.asarray(value) for key, value in state.to_dict().items()}
            )
        finally:
            with self.guard:
                self.active -= 1


def test_native_identity_and_repeated_adapter_identity():
    perfect = jxp.FluidPerfectGas(name="Air")
    assert as_jax_fluid(perfect) is perfect
    # Identity checks do not evaluate real-fluid or bicubic properties.
    for native_class in (jxp.FluidJAX, jxp.FluidBicubic):
        native = object.__new__(native_class)
        assert as_jax_fluid(native) is native
    host = HostPerfectGas()
    adapted = as_jax_fluid(host)
    assert adapted.source is host
    assert as_jax_fluid(adapted) is adapted


def test_host_mixture_dispatch_preserves_composition_with_perfect_gas_fixture():
    class HostPerfectGasMixture(jxp.FluidMix):
        # Exercise the separate host-mixture type without a real-fluid campaign.
        __init__ = HostPerfectGas.__init__
        get_state = HostPerfectGas.get_state

    source = HostPerfectGasMixture()
    source.mixture_ratio = 4.0
    source.fluid_1, source.fluid_2 = object(), object()
    adapted = as_jax_fluid(source)
    assert adapted.source is source
    assert adapted.mixture_ratio == 4.0
    assert adapted.fluid_1 is source.fluid_1 and adapted.fluid_2 is source.fluid_2
    expected = source.native.get_state(jxp.PT_INPUTS, 2e5, 300.0)
    density = jax.jit(lambda h, p: adapted.get_state(jxp.HmassP_INPUTS, h, p)["rho"])
    assert_allclose(density(expected["h"], expected["p"]), expected["rho"], rtol=2e-13)


def test_scalar_and_batched_jit_keep_configured_instance_and_metadata():
    host = HostPerfectGas()
    adapted = as_jax_fluid(host)
    evaluate = jax.jit(lambda p, T: adapted.get_state(jxp.PT_INPUTS, p, T))
    scalar = evaluate(jnp.array(2e5), jnp.array(300.0))
    expected = host.native.get_state(jxp.PT_INPUTS, 2e5, 300.0)
    assert scalar.fluid_name == host.name and scalar.identifier == host.identifier
    for key in ("rho", "h", "s", "a", "G", "T"):
        assert_allclose(scalar[key], expected[key], rtol=2e-13)
    pressure = jnp.array([[1e5], [2e5]])
    temperature = jnp.array([[250.0, 300.0, 350.0]])
    batched = evaluate(pressure, temperature)
    expected = host.native.get_state(jxp.PT_INPUTS, pressure, temperature)
    assert batched["rho"].shape == (2, 3)
    assert_allclose(batched["rho"], expected["rho"], rtol=2e-13)
    assert adapted.p_min == host.p_min and adapted.T_max == host.T_max
    assert adapted.configuration is host.configuration
    assert host.phase == "original phase policy" and host.exceptions is True
    assert host.backend == "host-only test backend"


def test_existing_callback_autodiff_and_vmap_match_perfect_gas_data():
    host = HostPerfectGas()
    adapted = as_jax_fluid(host)
    density = lambda p, T: adapted.get_state(jxp.PT_INPUTS, p, T)["rho"]
    gradient = jax.jit(jax.grad(density, argnums=(0, 1)))(
        jnp.array(2e5), jnp.array(300.0)
    )
    expected = jax.grad(
        lambda p, T: host.native.get_state(jxp.PT_INPUTS, p, T)["rho"], argnums=(0, 1)
    )(jnp.array(2e5), jnp.array(300.0))
    assert_allclose(np.asarray(gradient), np.asarray(expected), rtol=3e-6, atol=1e-11)
    pressures = jnp.array([1e5, 1.5e5, 2e5])
    actual = jax.jit(jax.vmap(lambda p: density(p, jnp.array(300.0))))(pressures)
    assert_allclose(
        actual,
        host.native.get_state(jxp.PT_INPUTS, pressures, 300.0)["rho"],
        rtol=2e-13,
    )


def test_domain_failure_becomes_nan_without_mutating_source_policy():
    host = HostPerfectGas()
    adapted = as_jax_fluid(host)
    state = jax.jit(lambda T: adapted.get_state(jxp.PT_INPUTS, 2e5, T))(jnp.array(-1.0))
    assert np.isnan(state["rho"]) and np.isnan(state["h"]) and np.isnan(state["a"])
    assert host.exceptions is True and host.identifier == "original identifier"
    with pytest.raises(ValueError, match="Outside"):
        host.get_state(jxp.PT_INPUTS, 2e5, -1.0)


@pytest.mark.parametrize(
    "error", [KeyError("programming error"), TypeError("programming error")]
)
def test_programming_errors_are_not_reclassified_as_invalid_states(error):
    host = HostPerfectGas()
    adapted = as_jax_fluid(host)
    host.error = error
    with pytest.raises(Exception, match="programming error"):
        np.asarray(adapted.get_state(jxp.PT_INPUTS, 2e5, 300.0)["rho"])


def test_host_decorator_wrapped_programming_error_is_not_suppressed():
    host = HostPerfectGas()
    host.error = RuntimeError("wrapped programming error")
    host.error.__cause__ = TypeError("underlying programming error")
    adapted = as_jax_fluid(host)
    with pytest.raises(Exception, match="programming error"):
        np.asarray(adapted.get_state(jxp.PT_INPUTS, 2e5, 300.0)["rho"])


def test_host_callbacks_for_shared_source_are_serialized():
    host = HostPerfectGas()
    # Precompile the actual perfect-gas fixture before the parallel callbacks.
    host.get_state(jxp.PT_INPUTS, 2e5, 300.0)
    host.peak_active = 0
    host.delay = 0.005
    first, second = as_jax_fluid(host), as_jax_fluid(host)

    def evaluate(index):
        adapter = first if index % 2 else second
        return float(adapter.get_state(jxp.PT_INPUTS, 2e5 + index, 300.0)["rho"])

    with ThreadPoolExecutor(max_workers=4) as executor:
        values = list(executor.map(evaluate, range(8)))
    assert np.isfinite(values).all() and host.peak_active == 1
