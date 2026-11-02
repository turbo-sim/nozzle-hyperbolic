"""Optional adapter to NozzleX's shared annular friction and heat laws.

No correlation is reimplemented here. Install NozzleX explicitly to use this
adapter; the base hyperbolic package does not require or import it at startup.
"""

import jax.numpy as jnp
import jaxprop as jxp

from ..state import thermodynamic_state


class NozzleXWall:
    """Bind NozzleX laws to a fixed native annular mesh and inlet context.

The imported empirical laws retain their Reynolds floors, entrance correction
and forward-flow assumption. Curvature losses use the geometric magnitude.
Configuration angles are degrees, matching NozzleX. Other annular model
angles retain their normal radians convention.
    """

    dependency_radius = 0

    def __init__(self, native_mesh, model_options, *, alpha_in_deg):
        from nozzlex.vaneless_channel.wall_models import evaluate_wall_sources
        from nozzlex.vaneless_channel.vaneless_channel_model import ModelOptions
        from nozzlex.vaneless_channel.friction_models import make_friction_model
        from nozzlex.vaneless_channel.heat_models import make_heat_model

        self._evaluate = evaluate_wall_sources
        self.options = ModelOptions(
            friction=make_friction_model(model_options["friction_model"]),
            heat_transfer=make_heat_model(model_options["heat_model"]),
        )
        metrics = native_mesh.geometry.evaluate(native_mesh.m_centers)
        self.geometry = {key: jnp.asarray(value) for key, value in dict(
            m=native_mesh.m_centers, r=metrics.r, b=metrics.b, A=metrics.A,
            dAds=metrics.dA_dm, curvature=abs(metrics.curvature),
        ).items()}
        self.params = dict(m_total=float(native_mesh.geometry.length),
                           b_in=float(native_mesh.geometry.evaluate(0.0).b),
                           alpha_in=float(alpha_in_deg))

    def fields(self, physical, radius, fluid):
        state = thermodynamic_state(physical, fluid)
        vm, vt = physical[..., 1], physical[..., 2] / radius
        h0 = physical[..., 3] + 0.5 * (vm**2 + vt**2)
        state0 = fluid.get_state(jxp.HmassSmass_INPUTS, h0, state["s"])
        return self._evaluate(vm, vt, state, state0, self.geometry,
                              self.params, self.options)

    def directional_stress(self, physical, radius, fluid):
        loads = self.fields(physical, radius, fluid)
        return loads["tau_m"], loads["tau_t"], loads["q_w"]
