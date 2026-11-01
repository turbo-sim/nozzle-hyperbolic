"""Production-model builders and independent steady verification projections.

Analytical data select physical boundary conditions and assess results. They
never initialize a numerical state. No Scheme or time controller is chosen here.
"""

from dataclasses import dataclass, field
from functools import cached_property

import jax
import jaxprop as jxp
import numpy as np

from nozzle_hyperbolic.geometry.annular import radial_channel, conical_channel, circular_arc_channel
from nozzle_hyperbolic.geometry.fvm import build_annular_mesh
from nozzle_hyperbolic.geometry.nozzle import SymmetricNozzleGeometry
from nozzle_hyperbolic.models.euler_annular.model import AnnularModel
from nozzle_hyperbolic.models.euler_annular.analytical import PerfectGasReference as AnnularReference
from nozzle_hyperbolic.models.euler_nozzle.model import NozzleModel
from nozzle_hyperbolic.models.euler_nozzle.analytical.perfect_gas import PerfectGasReference as NozzleReference, mach_from_area_ratio
from .catalog import CaseDefinition, load_case


def _numpy(fields):
    return {key: np.asarray(value) if not isinstance(value, (str, type(None))) else value for key, value in fields.items()}


@dataclass
class VerificationProblem:
    """A physical case on one mesh, ready for a user-selected numerical method."""

    definition: CaseDefinition
    model: object
    geometry: object
    fluid: object
    reference: object
    back_pressure: float
    shock_position: float | None
    inlet_data: dict = field(default_factory=dict)
    native_mesh: object = None

    @property
    def mesh(self):
        return self.model.mesh

    @cached_property
    def initial_state(self):
        """Cold model-owned start; no analytical profile or shock seed."""
        return np.asarray(self.model.initial_state()).copy()

    @property
    def case_id(self):
        return self.definition.case_id

    @property
    def shock_m(self):
        return self.shock_position

    def numerical_fields(self, full):
        """Physical cell diagnostics through the production EOS/model callbacks."""
        full = np.asarray(full)
        if full.shape != (self.mesh.n_cells + 2, self.model.nvar):
            raise ValueError("Expected the full cell-and-trace state array")
        result = _numpy(self.model.fields(full))
        return self._augment_fields(result, self.mesh.centers)

    def _augment_fields(self, fields, coordinates):
        coordinates = np.asarray(coordinates)
        fields["coordinate"] = coordinates
        if self.definition.model_name == "euler_annular":
            inlet = self.reference.profile(np.array([0.0]))
            p_in = float(inlet["p"][0])
            p0 = float(self.definition.data["inlet"]["p0"])
            fields["Cp"] = (fields["p"] - p_in) / (p0 - p_in)
            fields["velocity"] = fields["v_m"]
        else:
            fields["velocity"] = fields["u"]
        return fields

    def reference_fields(self, coordinates=None):
        """Independent analytical point values; shock planes use upstream limits."""
        positions = np.atleast_1d(np.asarray(self.mesh.centers if coordinates is None else coordinates, dtype=float))
        if self.definition.model_name == "euler_annular":
            result = self.reference.profile(positions, shock_m=self.shock_position)
        else:
            reference = self.reference
            result = reference.profile(positions, self.back_pressure / reference.p0)
            shift = float(self.model.h0) - reference.cp * reference.T0
            h = reference.cp * result["T"] + shift
            h0 = h + 0.5 * result["u"]**2
            result.update(
                h=h, h0=h0, a=np.sqrt(reference.gamma * reference.R * result["T"]),
                s=float(self.model.s0) + result["ds"],
                A=self.geometry.sample(positions)["area"],
                W=np.column_stack((result["p"], result["u"], h)),
                q=np.column_stack((result["rho"], result["rho"] * result["u"], result["rho"] * h0 - result["p"])),
            )
        return self._augment_fields(_numpy(result), positions)

    @cached_property
    def _reference_cells(self):
        if self.definition.model_name == "euler_annular":
            projected = self.reference.cell_average(self.native_mesh, shock_m=self.shock_position, return_conserved=True)
        else:
            # Independent quadrature of analytical conserved density and exact
            # physical area. Split cells at the throat and prescribed shock.
            nodes, weights = np.polynomial.legendre.leggauss(32)
            integrated = []
            for lower, upper in zip(self.mesh.faces[:-1], self.mesh.faces[1:]):
                special = [self.geometry.length / 2]
                if self.shock_position is not None:
                    special.append(self.shock_position)
                splits = sorted([lower, *(point for point in special if lower < point < upper), upper])
                total = np.zeros(3)
                for left, right in zip(splits[:-1], splits[1:]):
                    x = 0.5 * (left + right) + 0.5 * (right - left) * nodes
                    values = self.reference_fields(x)
                    total += 0.5 * (right - left) * (weights @ (values["q"] * values["A"][:, None]))
                integrated.append(total)
            integrated = np.asarray(integrated)
            q = integrated / np.asarray(self.mesh.volumes)[:, None]
            rho = q[:, 0]
            velocity = q[:, 1] / rho
            shift = float(self.model.h0) - self.reference.cp * self.reference.T0
            thermal = q[:, 2] / rho - 0.5 * velocity**2 - shift
            temperature = thermal / (self.reference.cp - self.reference.R)
            pressure = rho * self.reference.R * temperature
            W = np.column_stack((pressure, velocity, self.reference.cp * temperature + shift))
            projected = dict(W=W, q=q, integrated=integrated)
        traces = self.reference_fields(np.array([self.mesh.faces[0], self.mesh.faces[-1]]))["W"]
        full = np.vstack((traces[0], projected["W"], traces[1]))
        return dict(projected, fields=self.numerical_fields(full))

    def cell_reference(self):
        """Equivalent primitive states, conserved averages and integrated contents.

        These are analytical finite-volume projections, not point samples or
        averages of primitive variables. Returned arrays must be treated as
        read-only assessment data and must not be used as a hidden warm start.
        """
        return self._reference_cells

    def reference_cell_state(self):
        return self._reference_cells["W"].copy()

    def reference_cell_contents(self):
        return self._reference_cells["integrated"].copy()

    def reference_cell_fields(self):
        """Fields of the equivalent analytically projected cell states."""
        return {key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in self._reference_cells["fields"].items()}

    @property
    def field_scales(self):
        """Fixed physical scales, including safe nonzero scales for zero swirl."""
        data = self.definition.data
        pressure = float(data["inlet"]["p0"])
        sound = float(self.model.variable_scales[1])
        density = float(self.model.conservative_scales[0])
        radius = (float(self.native_mesh.r_face[0]) if self.native_mesh is not None else 1.0)
        return dict(p=pressure, rho=density, u=sound, v_m=sound, velocity=sound,
                    v_theta=sound, a=sound, M=1.0, M_m=1.0, M_theta=1.0,
                    ell=radius * sound, h=sound**2, h0=sound**2,
                    T=float(data["inlet"]["T0"]), Cp=1.0,
                    mdot=float(self.model.global_flux_scales[0]),
                    s=sound**2 / float(data["inlet"]["T0"]), alpha_deg=180.0)

    @property
    def legacy_payload(self):
        """Resolved plain data usable by an isolated frozen-package worker."""
        data = self.definition.data
        inlet = data["inlet"]
        inlet_data = {key: np.asarray(value).tolist() if isinstance(value, (np.ndarray, jax.Array)) else value for key, value in self.inlet_data.items()}
        return dict(
            case_id=self.case_id, model=self.definition.model_name,
            fluid=data["fluid"], geometry=data["geometry"],
            reservoir=dict(p0=inlet["p0"], T0=inlet["T0"]), inlet=inlet,
            p0=inlet["p0"], T0=inlet["T0"],
            back_pressure=self.back_pressure, outlet_pressure=self.back_pressure,
            inlet_data=inlet_data, inlet_regime=data["boundaries"]["inlet"],
            outlet_regime=data["boundaries"]["outlet"],
            outlet_formulation=data["boundaries"]["outlet_formulation"],
            walls=data["walls"], nvar=self.model.nvar,
            initial_state=self.initial_state.tolist(),
            points=np.asarray(self.mesh.points).tolist(), faces=np.asarray(self.mesh.faces).tolist(), centers=np.asarray(self.mesh.centers).tolist(),
            conservative_scales=np.asarray(self.model.conservative_scales).tolist(),
            global_flux_scales=np.asarray(self.model.global_flux_scales).tolist(),
            variable_scales=np.asarray(self.model.variable_scales).tolist(),
            reference_time=float(self.model.reference_time),
            shock_position=self.shock_position,
        )

    @property
    def resolved(self):
        return self.legacy_payload


def _nozzle(definition, cells, fluid):
    data = definition.data
    parameters = {key: value for key, value in data["geometry"].items() if key != "kind"}
    geometry = SymmetricNozzleGeometry(**parameters)
    inlet, selection = data["inlet"], data["reference"]
    reference = NozzleReference(fluid, p0=inlet["p0"], T0=inlet["T0"], **parameters)
    condition, value = selection["condition"], selection.get("value")
    shock = None
    if condition == "throat_mach":
        astar = reference.At / reference.area_mach(value)
        ratio = reference.pressure_ratio(reference.mach(reference.Ae / astar, "sub"))
    elif condition == "shock_fraction":
        shock = reference.length * (1 + value) / 2
        ratio = reference.shock_data(shock)[0]
    elif condition == "external_midpoint":
        ratio = (reference.shock_exit + reference.adapted) / 2
    elif condition == "adapted_factor":
        ratio = reference.adapted * value
    else:
        ratio = getattr(reference, condition)
        if condition == "shock_exit":
            shock = reference.length
    pressure = float(inlet["p0"] * ratio)
    boundaries = data["boundaries"]
    model = NozzleModel(geometry.build_mesh(cells), fluid, p0=inlet["p0"], T0=inlet["T0"], back_pressure=pressure,
                        inlet_regime=boundaries["inlet"], outlet_regime=boundaries["outlet"], outlet_formulation=boundaries["outlet_formulation"])
    return VerificationProblem(definition, model, geometry, fluid, reference, pressure, shock)


def _annular(definition, cells, fluid):
    data = definition.data
    parameters = data["geometry"]
    kind = parameters["kind"]
    if kind == "radial":
        geometry = radial_channel(parameters["r_in"], parameters["r_out"], parameters["height"])
    elif kind == "conical":
        geometry = conical_channel(parameters["length"], parameters["r_in"], parameters["height"], np.deg2rad(parameters["inclination_degrees"]), np.deg2rad(parameters["semi_divergence_degrees"]))
    else:
        geometry = circular_arc_channel(tuple(parameters["center"]), parameters["radius"], np.deg2rad(parameters["theta_start_degrees"]), np.deg2rad(parameters["theta_end_degrees"]), parameters["height"])
    inlet, boundaries = data["inlet"], data["boundaries"]
    alpha = np.deg2rad(inlet["alpha_degrees"]) if "alpha_degrees" in inlet else np.arctan(inlet.get("alpha_tangent", 0.0))
    mach = inlet["M_m"]
    if mach == "sonic_exit":
        if alpha != 0 or boundaries["inlet"] != "subsonic":
            raise ValueError("A sonic_exit reference requires zero swirl and subsonic inlet")
        area_ratio = float(geometry.evaluate(0.0).A / geometry.evaluate(geometry.length).A)
        mach = mach_from_area_ratio(area_ratio, float(fluid.constants.gamma), branch="sub")
    reference = AnnularReference.from_inlet(fluid, geometry, p0=inlet["p0"], T0=inlet["T0"], M_m=mach, alpha=alpha)
    fraction = data["reference"]["shock_fraction"]
    shock = None if fraction is None else fraction * geometry.length
    upstream = reference.profile(np.array([0.0]))
    downstream = reference.profile(np.array([geometry.length]), shock_m=shock)
    pressure = float(downstream["p"][0])
    inlet_data = (dict(state=upstream["W"][0], p0=inlet["p0"])
                  if boundaries["inlet"] == "supersonic" else
                  dict(p0=inlet["p0"], T0=inlet["T0"], alpha_in=float(alpha), h0=reference.h0,
                       s0=float(fluid.get_state(jxp.PT_INPUTS, inlet["p0"], inlet["T0"])["s"])))
    mesh, native = build_annular_mesh(geometry, cells)
    model = AnnularModel(mesh, native, fluid, inlet_data=inlet_data, outlet_pressure=pressure,
                         inlet_regime=boundaries["inlet"], outlet_regime=boundaries["outlet"], outlet_formulation=boundaries["outlet_formulation"],
                         tau_w=data["walls"]["shear_stress"], q_w=data["walls"]["heat_flux"])
    return VerificationProblem(definition, model, geometry, fluid, reference, pressure, shock, inlet_data, native)


def build_case(case, cells, *, fluid=None):
    """Build a production problem and analytical reference, without solving it.

    ``cells`` is a numerical mesh choice supplied by the caller. Physical YAML
    definitions deliberately contain no reconstruction, time integrator, CFL,
    nonlinear tolerance, or initialization-acceleration settings.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable jax_enable_x64 before building verification cases")
    if isinstance(cells, bool) or not isinstance(cells, (int, np.integer)) or cells < 2:
        raise ValueError("cells must be an integer >= 2")
    definition = load_case(case) if isinstance(case, str) else case
    if not isinstance(definition, CaseDefinition):
        raise TypeError("case must be a qualified ID or CaseDefinition")
    if fluid is None:
        fluid = jxp.FluidPerfectGas(name=definition.data["fluid"]["name"])
    builder = _nozzle if definition.model_name == "euler_nozzle" else _annular
    return builder(definition, int(cells), fluid)
