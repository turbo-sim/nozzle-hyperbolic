"""Canonical physical cases and explicit stagnant/numerical-adapted starts."""
from dataclasses import dataclass
from functools import lru_cache

import jax
jax.config.update("jax_enable_x64", True)
import jaxprop as jxp
import numpy as np

from nozzle_hyperbolic.models.euler_nozzle.model import NozzleModel
from nozzle_hyperbolic.models.euler_annular.model import AnnularModel
from nozzle_hyperbolic.verification import build_case
from development.real_gas.scripts.problem import load_context
from development.real_gas.scripts.common import identity

PG_CASES = {
    "nozzle_shock35": "euler_nozzle/shock_35", "nozzle_shock70": "euler_nozzle/shock_70",
    "annular_shock35": "euler_annular/shock35", "annular_shock70": "euler_annular/shock70",
}


def family_of(case_id):
    if case_id in PG_CASES:
        return "annular" if case_id.startswith("annular") else "nozzle"
    if case_id.endswith("_shock"):
        return case_id[:-6]
    raise ValueError(f"Unknown target case {case_id}")


def canonical_fields(values):
    """Consistent report names without changing physical field definitions."""
    result = {key: np.asarray(value) for key, value in values.items()
              if value is not None and not isinstance(value, (str, dict))}
    if "M_m" in result and "M" in result:
        result["M_total"] = result["M"]
    aliases = {"x": ("x", "coordinate", "m"), "u": ("u", "v_m", "velocity"),
               "a": ("a", "a_table"), "M": ("M_m", "M", "M_table"),
               "mass_flow": ("mass_flow", "mdot")}
    for target, candidates in aliases.items():
        for key in candidates:
            if key in result:
                result[target] = result[key]
                break
    return result


def clone_model(original, outlet, *, initial=None, adapted=False):
    """Construct the same production physics with the requested endpoint law."""
    if original.name == "euler_nozzle":
        model = NozzleModel(
            original.mesh, original.fluid, p0=original.p0, T0=original.T0,
            back_pressure=original.back_pressure,
            inlet_regime=original.inlet_regime,
            outlet_regime="supersonic" if adapted else "subsonic",
            outlet_formulation=outlet, initial_state=initial,
        )
    else:
        model = AnnularModel(
            original.mesh, original.native_mesh, original.fluid,
            inlet_data=original.inlet_data, outlet_pressure=original.outlet_pressure,
            inlet_regime=original.inlet_regime,
            outlet_regime="supersonic" if adapted else "subsonic",
            outlet_formulation=outlet, initial_state=initial,
            tau_w=original.tau_wall, q_w=original.heat_wall,
        )
    return model


@dataclass
class Context:
    case_id: str
    physical: object
    real_gas: bool
    adapted: bool = False

    @property
    def family(self):
        return self.case_id if self.adapted else family_of(self.case_id)

    @property
    def back_pressure(self):
        return float(self.physical.back_pressure)

    @property
    def shock_position(self):
        return self.physical.shock_position

    @property
    def table_sha256(self):
        return self.physical.eos.table_sha256 if self.real_gas else None

    def base_model(self, cells):
        if self.real_gas:
            from development.real_gas.scripts.common import STUDY as REAL_STUDY
            from .common import load
            return self.physical.model(cells, cold_start=load(REAL_STUDY / "settings.json")["cold_start"])
        return build_case(self.physical.definition, cells, fluid=self.physical.fluid).model

    def model(self, cells, outlet="acoustic", *, start="stagnant", seed=None):
        model = clone_model(self.base_model(cells), outlet, adapted=self.adapted)
        if self.adapted:
            # Adapted preparation is an independent numerical solve from the
            # canonical model's physical guess, never an analytical profile.
            return self.base_model(cells)
        if start == "stagnant":
            if self.real_gas or model.name == "euler_nozzle":
                p0, h0 = model.p0, model.h0
            else:
                inlet = self.physical.definition.data["inlet"]
                p0 = inlet["p0"]
                h0 = float(model.fluid.get_state(jxp.PT_INPUTS, p0, inlet["T0"])["h"])
            full = np.zeros((cells + 2, model.nvar))
            full[:, 0], full[:, -1] = p0, h0
        elif start == "adapted":
            if seed is None or not seed.get("verified", False):
                raise ValueError("Adapted initialization requires a verified numerical seed")
            old, values = np.asarray(seed["points"]), np.asarray(seed["state"])
            if not np.array_equal(old[[0, -1]], np.asarray(model.mesh.points)[[0, -1]]):
                raise ValueError("Seed and target domain endpoints differ")
            full = np.column_stack([np.interp(model.mesh.points, old, values[:, j])
                                    for j in range(model.nvar)])
        else:
            raise ValueError("Unknown start")
        model._initial_state = full
        return model

    def reference_fields(self, coordinates):
        return canonical_fields(self.physical.reference_fields(coordinates))

    def numerical_fields(self, model, full):
        full = np.asarray(full)
        if not self.real_gas:
            problem = build_case(self.physical.definition, model.mesh.n_cells, fluid=model.fluid)
            result = canonical_fields(problem.numerical_fields(full))
        else:
            values = full[1:-1]
            thermo = self.physical.eos.state(values[:, -1], values[:, 0])
            result = canonical_fields(thermo)
            result.update(u=values[:, 1], h0=values[:, -1] + values[:, 1]**2/2,
                          a=thermo["a_table"], M=values[:, 1]/thermo["a_table"],
                          mass_flow=thermo["rho"]*values[:, 1]*model.mesh.area_centers)
        result.update(x=np.asarray(model.mesh.centers), A=np.asarray(model.mesh.area_centers))
        if "mass_flow" not in result:
            result["mass_flow"] = result["rho"]*result["u"]*result["A"]
        return result

    def scales(self, model):
        p = float(model.variable_scales[0])
        a = float(model.variable_scales[1])
        rho = float(model.conservative_scales[0])
        temp = float(self.physical.definition["T0"] if self.real_gas else self.physical.definition.data["inlet"]["T0"])
        return dict(p=p, u=a, v_m=a, v_theta=a, a=a, rho=rho, h=a*a, h0=a*a,
                    s=a*a/temp, T=temp, M=1.0, M_m=1.0, Q=1.0,
                    ell=float(model.variable_scales[2]) if model.nvar == 4 else a,
                    mass_flow=float(model.global_flux_scales[0]))

    def metadata(self):
        model = self.base_model(50)
        label = (self.physical.definition["label"] + " — internal shock" if self.real_gas
                 else self.physical.definition.data["label"])
        dense = np.linspace(model.mesh.faces[0], model.mesh.faces[-1], 601)
        if self.shock_position is not None:
            shock = self.shock_position
            dense = np.unique(np.r_[dense, shock, np.nextafter(shock, np.inf)])
        fields = self.reference_fields(dense)
        metadata = dict(id=self.case_id, label=label, model=model.name,
                        fluid="CO2 bicubic 100×100" if self.real_gas else "Air perfect gas",
                        back_pressure=self.back_pressure, shock_position=self.shock_position,
                        family=self.family, table_sha256=self.table_sha256,
                        domain=[float(model.mesh.faces[0]), float(model.mesh.faces[-1])],
                        accuracy_targets=None if self.real_gas else self.physical.definition.data["accuracy"],
                        reference=fields)
        metadata["physical_sha256"] = identity({key: value for key, value in metadata.items() if key != "reference"})
        return metadata


@lru_cache(maxsize=10)
def target_context(case_id):
    if case_id in PG_CASES:
        return Context(case_id, build_case(PG_CASES[case_id], 50), False)
    family = family_of(case_id)
    return Context(case_id, load_context(family, "shock"), True)


@lru_cache(maxsize=4)
def adapted_context(family):
    pg = {"nozzle": "euler_nozzle/adapted", "annular": "euler_annular/supersonic_expansion"}
    if family in pg:
        return Context(family, build_case(pg[family], 50), False, True)
    return Context(family, load_context(family, "adapted"), True, True)
