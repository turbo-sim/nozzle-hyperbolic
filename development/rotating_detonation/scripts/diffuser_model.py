"""Fixed reservoir, angle-driven inlet and constant-pressure outlet.

The inlet prescribes stagnation enthalpy, entropy and flow angle. Mach number
and mass flow respond to the duct solution. Both endpoint laws require forward
meridionally subsonic flow; no regime switch or state repair is applied.
"""

import jax.numpy as jnp
import jaxprop as jp
import numpy as np

from nozzle_hyperbolic.models.euler_annular import AnnularModel
from nozzle_hyperbolic.models.euler_annular.equations import physical_flux
from nozzle_hyperbolic.models.euler_annular.state import (
    primitive_fields,
    state_property,
    thermodynamic_state,
)
from nozzle_hyperbolic.solvers.fvm import Mesh


class DiffuserAnnular(AnnularModel):
    """Angle-driven characteristic inlet and fixed-pressure physical trace."""

    def __init__(self, *args, config, driven=False, **kwargs):
        if config["outlet"].get("kind", "pressure") != "pressure":
            raise ValueError("This study uses a constant-pressure outlet")
        super().__init__(*args, **kwargs)
        self.config, self.driven = config, driven
        self._left_factor = float((self.mesh.faces[0] - self.mesh.centers[0])
                                  / (self.mesh.centers[0] - self.mesh.centers[1]))
        self._right_factor = float((self.mesh.faces[-1] - self.mesh.centers[-1])
                                   / (self.mesh.centers[-1] - self.mesh.centers[-2]))

    def inlet_angle(self, time=0.0):
        inlet, forcing = self.config["inlet"], self.config["forcing"]
        wave = jnp.sin(2 * jnp.pi * forcing["frequency_Hz"] * time)
        scale = 1.0 if self.driven else 0.0
        return jnp.deg2rad(inlet["angle_deg"]
                          + scale * forcing["angle_amplitude_deg"] * wave)

    def target_outlet_pressure(self, time=0.0):
        return jnp.asarray(self.config["outlet"]["pressure_Pa"])

    def target_outlet(self, time=0.0):
        """The prescribed constant outlet static pressure in Pa."""
        return self.target_outlet_pressure(time)

    def boundary_residual(self, full, time=0.0):
        cells, inlet, outlet = full[1:-1], full[0], full[-1]
        inside = thermodynamic_state(cells[jnp.array([0, 1, -2, -1])], self.fluid)
        density = state_property(inside, "rho")
        sound = state_property(inside, "a")
        entropy = state_property(inside, "s")
        left = cells[0] + self._left_factor * (cells[0] - cells[1])
        right = cells[-1] + self._right_factor * (cells[-1] - cells[-2])
        trace = thermodynamic_state(full[jnp.array([0, -1])], self.fluid)
        trace_entropy = state_property(trace, "s")
        scales = self.boundary_scales
        angle = self.inlet_angle(time)
        inlet_rows = jnp.stack((
            (inlet[3] + 0.5 * (inlet[1]**2 + (inlet[2] / self._r_face[0])**2)
             - self.inlet_data["h0"]) / scales["h"],
            (trace_entropy[0] - self.inlet_data["s0"]) / scales["s"],
            (inlet[2] / self._r_face[0] * jnp.cos(angle)
             - inlet[1] * jnp.sin(angle)) / scales["a"],
            (inlet[0] - left[0] - density[0] * sound[0] * (inlet[1] - left[1])) / scales["p"],
        ))
        incoming = (outlet[0] - self.target_outlet_pressure(time)) / scales["p"]
        entropy_out = entropy[-1] + self._right_factor * (entropy[-1] - entropy[-2])
        # Established annular row order: imposed datum, outgoing acoustic,
        # outgoing angular momentum, outgoing entropy.
        outlet_rows = jnp.stack((
            incoming,
            (outlet[0] - right[0] + density[-1] * sound[-1]
             * (outlet[1] - right[1])) / scales["p"],
            (outlet[2] - right[2]) / scales["ell"],
            (trace_entropy[-1] - entropy_out) / scales["s"],
        ))
        return jnp.stack((inlet_rows, outlet_rows))

    def boundary_face_flux(self, full, flux):
        # This is the solved physical outlet trace, not an exterior ghost.
        # A second HLL split would change the imposed physical trace law.
        # The shared engine supplies the face-area factor exactly once.
        return flux.at[-1].set(physical_flux(full[-1], self._r_face[-1], self.fluid))

    def constraint_metrics(self, full):
        """JAX diagnostics including both face traces and all physical cells."""
        fields = primitive_fields(full, self._r_full, self.fluid)
        vm, speed = fields["v_m"], fields["M"] * fields["a"]
        return dict(
            normal_mach_min=jnp.min(fields["M_m"]),
            normal_mach_max=jnp.max(fields["M_m"]),
            inlet_total_mach=fields["M"][0],
            inlet_cell_total_mach=fields["M"][1],
            inlet_normal_mach=fields["M_m"][0],
            outlet_normal_mach=fields["M_m"][-1],
            inlet_vm=vm[0], outlet_vm=vm[-1],
            meridional_velocity_ratio=vm[-1] / vm[0],
            total_speed_ratio=speed[-1] / speed[0],
            outlet_mass_flow=fields["rho"][-1] * vm[-1] * self.mesh.area_faces[-1],
        )

    def boundary_diagnostics(self, full):
        result = self.constraint_metrics(full)
        result.update(mode=jnp.int32(2), direct_flux=jnp.asarray(True))
        return result

    def solution_audit(self, full, sonic_tolerance=2e-3):
        """Reject a branch that fails the requested physical operating range.

        Sonic tolerance is intentionally not used to permit a supersonic
        normal state. These endpoint laws require subsonic normal flow.
        """
        values = {key: float(value) for key, value in self.constraint_metrics(full).items()}
        return bool(self.boundary_admissible(full)
                    and values["normal_mach_min"] > 0
                    and values["normal_mach_max"] < 1
                    and values["meridional_velocity_ratio"] < 1)

    transient_solution_audit = solution_audit


def make_model(config, native_mesh, *, driven=False, initial_state=None):
    """Bind the angle-driven pressure case to an existing native annular mesh."""
    mesh = Mesh(native_mesh.m_faces, native_mesh.m_centers, native_mesh.volumes,
                native_mesh.A_face, native_mesh.A_center, native_mesh.dA_dm,
                np.diff(native_mesh.A_face), native_mesh.geometry)
    inlet = config["inlet"]
    alpha = inlet["angle_deg"]
    amplitude = config["forcing"]["angle_amplitude_deg"] if driven else 0
    if not -90 < alpha - amplitude <= alpha + amplitude < 90:
        raise ValueError("The entire inlet angle waveform must remain in (-90,90) degrees")
    return DiffuserAnnular(
        mesh, native_mesh, jp.FluidPerfectGas(config["fluid"]),
        inlet_data={"p0": inlet["stagnation_pressure_Pa"],
                    "T0": inlet["stagnation_temperature_K"],
                    "alpha_in": np.deg2rad(alpha)},
        inlet_regime="subsonic", outlet_regime="subsonic",
        outlet_pressure=config["outlet"]["pressure_Pa"],
        initial_state=initial_state, config=config, driven=driven,
    )
