"""Reuse JAXProp's callback bridge for configured host-only fluid objects.

Native JAXProp fluids keep their existing implementation and identity. A raw
``jaxprop.Fluid`` or host ``FluidMix`` is evaluated through JAXProp's own JIT/batching/custom-JVP
wrapper, preserving its original instance, backend, phase and exception policy.
The adapter adds no equation of state or differentiation approximation.
"""

from threading import RLock

import jaxprop as jxp
from jaxprop.coolprop.jax_wrapper import get_state as _callback_state
from jaxprop.helpers_props import PROPERTIES_CANONICAL


# JAX can schedule callbacks concurrently. CoolProp AbstractState is mutable;
# serialize callback evaluation even when two adapters share the same source.
# This lock does not change or attempt to guard callers bypassing the adapter.
_HOST_EVALUATION_LOCK = RLock()


class _HostSource:
    def __init__(self, source):
        self.source = source

    def get_state(self, input_pair, x, y):
        with _HOST_EVALUATION_LOCK:
            try:
                return self.source.get_state(input_pair, x, y)
            except (ValueError, RuntimeError) as error:
                # JAXProp's host decorator can wrap any underlying error in
                # RuntimeError. Preserve programming errors such as TypeError,
                # KeyError and AttributeError rather than turning them into an
                # apparently ordinary inadmissible thermodynamic state.
                cause = error
                while cause.__cause__ is not None:
                    cause = cause.__cause__
                if not isinstance(cause, (ValueError, RuntimeError)):
                    raise
                # The existing JAXProp callback maps None to NaN properties.
                return None


class HostFluidJAXAdapter:
    """JAX-callable view of one original configured host fluid.

    Metadata/domain attributes are forwarded to ``source``. The source is
    neither reconstructed nor modified by the adapter. Its normal internal
    evaluation state may still change when its own ``get_state`` runs. Treat
    source configuration as fixed while reusing compiled numerical kernels.
    Derivatives have the accuracy and limitations of JAXProp's existing bridge.
    """

    def __init__(self, source):
        self.source = source
        self._host_source = _HostSource(source)

    def __getattr__(self, name):
        return getattr(self.source, name)

    def get_state(self, input_pair, x, y):
        properties = _callback_state(input_pair, x, y, self._host_source)
        return jxp.FluidState(
            fluid_name=getattr(self.source, "name", None),
            identifier=getattr(self.source, "identifier", None),
            **{name: properties[name] for name in PROPERTIES_CANONICAL})


def as_jax_fluid(fluid):
    """Adapt raw JAXProp Fluid/FluidMix; pass native/custom JAX fluids through.

    In particular FluidPerfectGas, FluidBicubic and FluidJAX are returned by
    identity. Reapplying this function to an existing adapter is also a no-op.
    The host proxy converts only thermodynamic ValueError/RuntimeError failures
    to NaNs, without changing the original object's ``exceptions`` setting.
    """
    if isinstance(fluid, HostFluidJAXAdapter):
        return fluid
    if isinstance(fluid, (jxp.Fluid, jxp.FluidMix)):
        return HostFluidJAXAdapter(fluid)
    return fluid
