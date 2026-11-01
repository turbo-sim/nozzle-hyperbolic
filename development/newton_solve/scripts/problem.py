"""Canonical perfect-gas nozzle targets and disclosed numerical initial fields."""
from dataclasses import dataclass
from functools import lru_cache

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from nozzle_hyperbolic.models.euler_nozzle.model import NozzleModel
from nozzle_hyperbolic.verification import build_case
from .common import identity

CASES = {"nozzle_shock35": "euler_nozzle/shock_35", "nozzle_shock70": "euler_nozzle/shock_70"}


def canonical_fields(values):
    result = {key: np.asarray(value) for key, value in values.items()
              if value is not None and not isinstance(value, (str, dict))}
    for target, candidates in {"x": ("x", "coordinate"), "u": ("u", "velocity"),
                               "M": ("M",), "mass_flow": ("mass_flow", "mdot")}.items():
        for key in candidates:
            if key in result:
                result[target] = result[key]
                break
    return result


@dataclass
class CaseContext:
    case_id: str
    physical: object
    adapted: bool = False

    @property
    def back_pressure(self):
        return float(self.physical.back_pressure)

    @property
    def shock_position(self):
        return self.physical.shock_position

    def base_model(self, cells):
        return build_case(self.physical.definition, cells=cells, fluid=self.physical.fluid).model

    def build_model(self, cells, outlet="acoustic", initial=None):
        original = self.base_model(cells)
        if self.adapted:
            return original
        return NozzleModel(original.mesh, original.fluid, p0=original.p0, T0=original.T0,
                           back_pressure=original.back_pressure, inlet_regime=original.inlet_regime,
                           outlet_regime="subsonic", outlet_formulation=outlet, initial_state=initial)

    def initial_field(self, model, start, seed=None):
        if start == "stagnant":
            full = np.zeros((model.mesh.n_cells+2, model.nvar))
            full[:, 0], full[:, -1] = model.p0, model.h0
            return full
        if start != "adapted" or seed is None or not seed["verified"]:
            raise ValueError("Adapted start requires a verified numerical adapted seed")
        points, values = np.asarray(seed["points"]), np.asarray(seed["state"])
        if not np.array_equal(points, np.asarray(model.mesh.points)):
            raise ValueError("Target adapted state must be the exact 200-cell numerical seed; no target interpolation")
        return values.copy()

    def reference_fields(self, coordinates):
        return canonical_fields(self.physical.reference_fields(coordinates))

    def numerical_fields(self, model, full):
        physical = build_case(self.physical.definition, cells=model.mesh.n_cells, fluid=model.fluid)
        fields = canonical_fields(physical.numerical_fields(np.asarray(full)))
        fields.update(x=np.asarray(model.mesh.centers), A=np.asarray(model.mesh.area_centers))
        fields.setdefault("mass_flow", fields["rho"]*fields["u"]*fields["A"])
        return fields

    def scales(self, model):
        p, a = np.asarray(model.variable_scales)[:2]
        temp = self.physical.definition.data["inlet"]["T0"]
        return dict(p=p, u=a, a=a, rho=model.conservative_scales[0], h=a*a, h0=a*a,
                    s=a*a/temp, T=temp, M=1.0, mass_flow=model.global_flux_scales[0])

    def metadata(self):
        model = self.base_model(50)
        dense = np.linspace(model.mesh.faces[0], model.mesh.faces[-1], 601)
        if self.shock_position is not None:
            dense = np.unique(np.r_[dense, self.shock_position, np.nextafter(self.shock_position, np.inf)])
        result = dict(id=self.case_id, label=self.physical.definition.data["label"], model=model.name,
                      fluid="Air perfect gas", back_pressure=self.back_pressure,
                      shock_position=self.shock_position, family="nozzle",
                      domain=[float(model.mesh.faces[0]), float(model.mesh.faces[-1])],
                      accuracy_targets=self.physical.definition.data["accuracy"],
                      reference=self.reference_fields(dense))
        result["physical_sha256"] = identity({key: value for key, value in result.items() if key != "reference"})
        return result


@lru_cache(maxsize=3)
def target_context(case_id):
    return CaseContext(case_id, build_case(CASES[case_id], cells=50))


@lru_cache(maxsize=1)
def adapted_context():
    return CaseContext("nozzle_adapted", build_case("euler_nozzle/adapted", cells=50), True)


@dataclass
class BoundContext:
    context: CaseContext
    model: object
    operator: object
    config: dict
    settings: dict

    @property
    def mesh(self):
        return self.model.mesh
