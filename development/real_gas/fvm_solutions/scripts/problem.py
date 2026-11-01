"""The four established CO2 cases on exact-area production FVM meshes.

The numerical cold state is model-owned. Analytical data are assessment data
only and are never passed to the solver as initialization.
"""

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import shutil

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np
from scipy.optimize import brentq

from nozzle_hyperbolic.models.euler_nozzle.model import NozzleModel
from nozzle_hyperbolic.numerics.mesh import Mesh
from development.real_gas.co2_nozzle_reference.scripts.thermodynamics import BicubicCO2
from development.real_gas.co2_nozzle_reference.scripts.reference import (
    Nozzle, SmoothPath, downstream_exit,
)


STUDY = Path(__file__).resolve().parents[1]
BASELINE = STUDY.parent / "co2_nozzle_reference"


def _area_primitive(nozzle, x):
    x = np.asarray(x, float)
    coefficient = np.where(
        x < 0, (nozzle.inlet_ratio - 1) / nozzle.x_in**2,
        (nozzle.exit_ratio - 1) / nozzle.x_out**2,
    )
    return nozzle.At * (x + coefficient * x**3 / 3)


def build_mesh(nozzle, cells):
    """Uniform physical cells; exact area, volume, and integrated area source."""
    if isinstance(cells, bool) or not isinstance(cells, (int, np.integer)) or cells < 2:
        raise ValueError("cells must be an integer >= 2")
    faces = np.linspace(nozzle.x_in, nozzle.x_out, int(cells) + 1)
    centers = (faces[:-1] + faces[1:]) / 2
    area_faces = nozzle.area(faces)
    return Mesh(
        faces=faces, centers=centers,
        volumes=np.diff(_area_primitive(nozzle, faces)),
        area_faces=area_faces, area_centers=nozzle.area(centers),
        darea_centers=nozzle.derivative(centers),
        area_derivative_integrals=np.diff(area_faces), geometry=nozzle,
    )


@dataclass
class CaseContext:
    case_id: str
    regime: str
    definition: dict
    baseline: dict
    eos: object
    nozzle: object
    upstream: object
    downstream: object
    mdot: float
    back_pressure: float
    shock_position: float | None

    @property
    def fluid(self):
        return self.eos.fluid

    @property
    def label(self):
        return self.definition["label"]

    @property
    def key(self):
        return f"{self.case_id}_{self.regime}"

    def model(self, cells, cold_start=None):
        """Production nozzle model with no analytic initial field or shock seed."""
        model = NozzleModel(
            build_mesh(self.nozzle, cells), self.fluid,
            p0=self.definition["p0"], T0=self.definition["T0"],
            back_pressure=self.back_pressure,
            inlet_regime="subsonic",
            outlet_regime="supersonic" if self.regime == "adapted" else "subsonic",
            outlet_formulation="acoustic",
        )
        if self.regime == "adapted" and cold_start and cold_start["adapted"] == "table_mach_ramp":
            # A prescribed Mach ramp is only a rough cold guess. It uses no
            # analytic area/Mach solution, choked mass flow, or shock position.
            target = float(cold_start["outlet_mach_table"])
            def residual(u):
                h = model.h0 - 0.5 * u**2
                a = float(self.fluid.get_state(jxp.HmassP_INPUTS, h, self.back_pressure)["a"])
                return u - target * a
            maximum_u = np.sqrt(2 * (model.h0 - self.eos.settings["h_min"])) * (1 - 1e-12)
            velocities = np.linspace(0, maximum_u, 201)
            values = [residual(u) for u in velocities]
            brackets = [(a, b) for a, b, fa, fb in zip(velocities[:-1], velocities[1:], values[:-1], values[1:])
                        if fa * fb < 0]
            if not brackets:
                raise ValueError("No in-table cold outlet Mach bracket")
            outlet_u = brentq(residual, *brackets[0], xtol=1e-9)
            fraction = (model.mesh.points - self.nozzle.x_in) / (self.nozzle.x_out - self.nozzle.x_in)
            inlet_u = float(cold_start["inlet_velocity_over_a0"]) * model.a0
            u = inlet_u + fraction * (outlet_u - inlet_u)
            p = model.p0 + fraction * (self.back_pressure - model.p0)
            model._initial_state = np.column_stack((p, u, model.h0 - 0.5 * u**2))
        return model

    def _mass_flux(self, path, p):
        h = path.enthalpy(p)
        kinetic = 2 * (path.h0 - h)
        if np.any(kinetic < -1e-6):
            raise ValueError("Invalid reference kinetic energy")
        rho = np.asarray(self._rho_vector(jnp.asarray(h), jnp.asarray(p)))
        return rho * np.sqrt(np.maximum(kinetic, 0))

    def _pressures(self, target, path, limb):
        """Vectorized bracketed bisection of the exact dense-path mass flux."""
        target = np.asarray(target, float)
        if limb == "sup":
            low, high, increasing = path.p_low, path.p_star, True
        elif limb == "sub":
            low, high, increasing = path.p_star, path.p_high, False
        elif limb == "postshock":
            low, high, increasing = path.p_low, path.p_high, False
        else:
            raise ValueError("Unknown reference limb")
        lower_flux, upper_flux = self._mass_flux(path, np.array([low, high]))
        if np.any(target < min(lower_flux, upper_flux) - 1e-10) or np.any(
            target > max(lower_flux, upper_flux) * (1 + 2e-13)
        ):
            raise ValueError("Reference target outside selected mass-flux branch")
        lo, hi = np.full_like(target, low), np.full_like(target, high)
        for _ in range(55):
            mid = (lo + hi) / 2
            above = self._mass_flux(path, mid) > target
            move_high = above if increasing else ~above
            hi = np.where(move_high, mid, hi)
            lo = np.where(move_high, lo, mid)
        return (lo + hi) / 2

    def reference_fields(self, coordinates):
        """Continuous pointwise reference; x at a shock means its upstream trace."""
        x = np.atleast_1d(np.asarray(coordinates, float))
        if np.any(x < self.nozzle.x_in) or np.any(x > self.nozzle.x_out):
            raise ValueError("Reference position outside nozzle")
        p, h = np.empty_like(x), np.empty_like(x)
        target = self.mdot / self.nozzle.area(x)
        after = np.zeros(x.shape, bool) if self.shock_position is None else x > self.shock_position
        before = ~after
        for mask, path, limb in (
            (before & (x < 0), self.upstream, "sub"),
            (before & (x >= 0), self.upstream, "sup"),
            (after, self.downstream, "postshock"),
        ):
            if np.any(mask):
                p[mask] = self._pressures(target[mask], path, limb)
                h[mask] = path.enthalpy(p[mask])
        u = np.sqrt(np.maximum(2 * (self.upstream.h0 - h), 0))
        result = self.eos.state(h, p)
        area = self.nozzle.area(x)
        result.update(
            x=x, A=area, u=u, h0=h + u**2 / 2,
            M_table=u / result["a_table"], M_euler=u / result["a_euler"],
            mass_flow=result["rho"] * u * area,
            W=np.column_stack((p, u, h)),
            q=np.column_stack((result["rho"], result["rho"] * u,
                               result["rho"] * (h + u**2 / 2) - p)),
        )
        return result

    def reference_cell_contents(self, mesh, order=8):
        """Conserved A-weighted integrals, split at throat and shock.

        These are supplementary conservation comparisons, not initial states
        or an inversion of averaged conservative quantities into primitives.
        """
        nodes, weights = np.polynomial.legendre.leggauss(order)
        intervals = []
        owners = []
        for i, (lower, upper) in enumerate(zip(mesh.faces[:-1], mesh.faces[1:])):
            special = [0.0] + ([] if self.shock_position is None else [self.shock_position])
            edges = sorted([lower, *(v for v in special if lower < v < upper), upper])
            for left, right in zip(edges[:-1], edges[1:]):
                intervals.append((left, right))
                owners.append(i)
        intervals = np.asarray(intervals)
        half = np.diff(intervals, axis=1)[:, 0] / 2
        locations = np.mean(intervals, axis=1)[:, None] + half[:, None] * nodes
        values = self.reference_fields(locations.ravel())
        integrated = (values["q"] * values["A"][:, None]).reshape(-1, order, 3)
        contributions = half[:, None] * np.einsum("j,ijk->ik", weights, integrated)
        result = np.zeros((mesh.n_cells, 3))
        np.add.at(result, owners, contributions)
        return result


@lru_cache(maxsize=4)
def load_context(case_id, regime):
    """Load a prepared matching reference when the FVM table domain is changed."""
    if regime not in ("adapted", "shock"):
        raise ValueError("regime must be adapted or shock")
    definitions = json.loads((BASELINE / "cases.json").read_text(encoding="utf-8"))
    profile = json.loads((STUDY / "settings.json").read_text(encoding="utf-8"))
    prepared = STUDY / "outputs/reference_summary.json"
    if profile.get("table_overrides"):
        from .prepare_reference import reference_fingerprint
        if not prepared.exists():
            raise RuntimeError("Run scripts.prepare_reference before using an extended FVM table")
        records = json.loads(prepared.read_text(encoding="utf-8"))
        if records.get("metadata", {}).get("reference_fingerprint") != reference_fingerprint(profile)[0]:
            raise RuntimeError("Prepared FVM references are stale; run scripts.prepare_reference")
    else:
        records = json.loads((BASELINE / "outputs/summary.json").read_text(encoding="utf-8"))
    saved = next((c for c in records["cases"] if c["id"] == case_id), None)
    case = None if saved is None else saved["settings"]
    if case is None:
        raise ValueError(f"Unknown baseline case {case_id!r}")
    output = STUDY / "outputs"
    cache = output / "fluid_tables"
    cache.mkdir(parents=True, exist_ok=True)
    table_name = case["table"]["name"]
    original_table = BASELINE / "outputs/fluid_tables" / f"{table_name}.pkl"
    cached_table = cache / original_table.name
    if not cached_table.exists():
        if not original_table.exists():
            raise FileNotFoundError("Prepare the matching FVM references and tables first")
        shutil.copyfile(original_table, cached_table)
    original_log = BASELINE / "outputs" / f"{table_name}_table.log"
    if original_log.exists() and not (output / original_log.name).exists():
        shutil.copyfile(original_log, output / original_log.name)
    eos = BicubicCO2(case["table"], output)
    if eos.table_sha256 != saved["table_sha256"]:
        raise ValueError("FVM table bytes differ from its matching analytical reference")
    nozzle = Nozzle(**(definitions["geometry"] | case.get("geometry_override", {})))
    reservoir = eos.reservoir(case["p0"], case["T0"])
    upstream = SmoothPath(eos, reservoir["h"], reservoir["p"], reservoir["h"],
                          eos.settings["p_min"] * 1.001)
    throat = upstream.choke()
    mdot = nozzle.At * throat["G"]
    if abs(mdot / saved["mdot"] - 1) > 1e-8:
        raise ValueError("Rebuilt reference mass flow differs from retained baseline")
    downstream = None
    shock_position = None
    if regime == "shock":
        downstream, _ = downstream_exit(eos, saved["shock"]["post"], reservoir["h"],
                                         mdot, nozzle.area(nozzle.x_out))
        shock_position = float(saved["shock"]["x"])
        back_pressure = float(case["backpressure"])
    else:
        back_pressure = float(saved["adapted_exit"]["p"])
    context = CaseContext(case_id, regime, case, saved, eos, nozzle, upstream, downstream,
                          mdot, back_pressure, shock_position)
    context._rho_vector = jax.jit(jax.vmap(eos._rho))
    return context
