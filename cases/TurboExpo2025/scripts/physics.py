"""Published geometries and equilibrium fluids for the current production FVM.

Only host-side preparation lives here. The physical equations, spatial fluxes,
admissibility tests and implicit integration remain production implementations.
"""

from contextlib import redirect_stderr, redirect_stdout
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import zipfile

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import CubicSpline

from nozzle_hyperbolic.models.euler_nozzle import NozzleModel
from nozzle_hyperbolic.models.euler_nozzle.closures.wall import HaalandWall
from nozzle_hyperbolic.models.euler_nozzle.closures.mixture import viscosity_Beattie
from nozzle_hyperbolic.numerics.mesh import Mesh

ROOT = Path(__file__).resolve().parents[1]
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _definitions():
    for name in ("case_definitions.json", "settings.json"):
        path = ROOT / name
        if path.exists():
            document = json.loads(path.read_text(encoding="utf-8"))
            if "geometries" in document:
                return document
    raise FileNotFoundError("Published geometry definitions have not been prepared")


def geometry_metadata(case):
    """Return the explicit geometry descriptor, without mutating case settings."""
    value = case["geometry"]
    return dict(
        value if isinstance(value, dict) else _definitions()["geometries"][value]
    )


@lru_cache(maxsize=16)
def _workbook_coordinates(filename, sheet_name, x_column="A", y_column="B"):
    """Read original XLSX bytes with the standard library, preserving units."""
    path = ROOT / filename
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            tree = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            strings = ["".join(node.itertext()) for node in tree.findall("s:si", NS)]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets = workbook.findall("s:sheets/s:sheet", NS)
        selected = next((s for s in sheets if s.attrib["name"] == sheet_name), None)
        if selected is None:
            raise ValueError(f"Missing worksheet {sheet_name!r} in {filename}")
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = next(
            r.attrib["Target"]
            for r in relationships
            if r.attrib["Id"] == selected.attrib[f"{{{REL}}}id"]
        )
        sheet_path = target.lstrip("/") if target.startswith("/") else "xl/" + target
        sheet = ET.fromstring(archive.read(sheet_path))
        pairs = []
        for row in sheet.findall("s:sheetData/s:row", NS):
            values = {}
            for cell in row.findall("s:c", NS):
                column = "".join(c for c in cell.attrib["r"] if c.isalpha())
                value = cell.find("s:v", NS)
                if value is not None:
                    raw = value.text
                    values[column] = (
                        strings[int(raw)] if cell.attrib.get("t") == "s" else float(raw)
                    )
            if all(
                isinstance(values.get(c), (float, int)) for c in (x_column, y_column)
            ):
                pairs.append((values[x_column], values[y_column]))
    data = np.asarray(pairs, dtype=float)
    if data.ndim != 2 or len(data) < 2 or not np.all(np.isfinite(data)):
        raise ValueError(f"Missing finite coordinate pairs in {filename}:{sheet_name}")
    if np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError(
            f"Unordered/duplicate geometry coordinates in {filename}:{sheet_name}"
        )
    return data


class WallGeometry:
    """A physical wall profile, including exact spline/polynomial volumes."""

    def __init__(self, descriptor):
        self.descriptor = descriptor
        self.kind = descriptor["kind"]
        self.width = float(descriptor.get("width_m", 1.0))
        if self.kind == "piecewise_half_height":
            self.x_min = float(descriptor.get("x_min", -0.015))
            self.x_max = float(descriptor.get("x_max", 0.00838))
            self.joins = np.asarray([0.0])
            self.profile = None
        else:
            xy = _workbook_coordinates(
                descriptor["workbook"],
                descriptor.get("sheet", "Sheet1"),
                descriptor.get("x_column", "A"),
                descriptor.get("y_column", "B"),
            )
            x = xy[:, 0] * float(descriptor.get("x_scale", 1.0))
            y = xy[:, 1] * float(descriptor.get("y_scale", 1.0))
            if "inlet_half_height_m" in descriptor:
                y = y.copy()
                y[0] = descriptor["inlet_half_height_m"]
            self.x_min = float(descriptor.get("x_min", x[0]))
            self.x_max = float(descriptor.get("x_max", x[-1]))
            if self.kind == "polynomial_radius":
                self.profile = Polynomial.fit(x, y, int(descriptor["degree"])).convert()
                self.joins = np.asarray([])
            else:
                self.profile = CubicSpline(x, y, extrapolate=False)
                self.joins = x[1:-1]
                if self.kind == "spline_half_height":
                    self.area_primitive = self.profile.antiderivative()
                    self.primitive_factor = 2 * self.width

    def half_height(self, x):
        x = np.asarray(x, dtype=float)
        if self.profile is not None:
            return self.profile(x)
        d = self.descriptor
        h = float(d.get("throat_half_height_m", 0.0002))
        left = float(d.get("convergent_slope", -0.32))
        right = float(d.get("divergent_slope", 0.0702 / 8.38))
        return h + np.where(x <= 0, left * x, right * x)

    def sample(self, x):
        x = np.asarray(x, dtype=float)
        height = self.half_height(x)
        if self.profile is None:
            derivative = np.where(
                x <= 0,
                self.descriptor.get("convergent_slope", -0.32),
                self.descriptor.get("divergent_slope", 0.0702 / 8.38),
            )
        else:
            derivative = (
                self.profile.deriv()(x)
                if isinstance(self.profile, Polynomial)
                else self.profile(x, 1)
            )
        if self.kind.endswith("radius"):
            area = np.pi * height**2
            da = 2 * np.pi * height * derivative
            perimeter = 2 * np.pi * height
        else:
            area = 2 * self.width * height
            da = 2 * self.width * derivative
            perimeter = 2 * (self.width + 2 * height)
        if not np.all(np.isfinite(area)) or np.any(area <= 0) or np.any(height <= 0):
            raise ValueError(
                "Geometry has nonpositive/nonfinite physical cross section"
            )
        return dict(
            area=area,
            area_derivative=da,
            perimeter=perimeter,
            hydraulic_diameter=4 * area / perimeter,
        )

    def volumes(self, faces):
        if hasattr(self, "area_primitive"):
            return getattr(self, "primitive_factor", 1.0) * np.diff(
                self.area_primitive(faces)
            )
        # Sixteen Gauss nodes integrate the squared degree-12 radius exactly,
        # avoiding catastrophic cancellation in its monomial antiderivative.
        # Four nodes suffice for a squared piecewise cubic radius.
        nodes, weights = leggauss(16 if self.kind == "polynomial_radius" else 4)
        values = []
        for left, right in zip(faces[:-1], faces[1:]):
            breaks = np.r_[
                left, self.joins[(self.joins > left) & (self.joins < right)], right
            ]
            value = 0.0
            for a, b in zip(breaks[:-1], breaks[1:]):
                x = 0.5 * (a + b) + 0.5 * (b - a) * nodes
                value += 0.5 * (b - a) * np.dot(weights, self.sample(x)["area"])
            values.append(value)
        return np.asarray(values)


def make_mesh(case, cells):
    """Uniform physical FVM cells; endpoint traces are added by the model."""
    if isinstance(cells, bool) or not isinstance(cells, (int, np.integer)) or cells < 2:
        raise ValueError("Use at least two integer physical cells")
    geometry = WallGeometry(geometry_metadata(case))
    faces = np.linspace(geometry.x_min, geometry.x_max, cells + 1)
    centers = 0.5 * (faces[:-1] + faces[1:])
    area_faces = geometry.sample(faces)["area"]
    sampled = geometry.sample(centers)
    return Mesh(
        faces=faces,
        centers=centers,
        volumes=geometry.volumes(faces),
        area_faces=area_faces,
        area_centers=sampled["area"],
        darea_centers=sampled["area_derivative"],
        area_derivative_integrals=np.diff(area_faces),
        geometry=geometry,
    )


def resolve_case(case):
    """Resolve an ideal shock-position specification to its outlet pressure.

    The analytical calculation selects boundary data only; it supplies no flow
    field or initialization to the finite-volume solver.
    """
    result = dict(case)
    if case.get("outlet_regime") == "supersonic" and case.get("p_out_Pa") is None:
        result.update(
            initial_back_pressure_Pa=0.25 * float(case["p0_Pa"]),
            initial_back_pressure_source="Cold initialization datum 0.25*p0, as existing successful equilibrium nozzle clients; no physical supersonic pressure condition",
        )
    fraction = case.get("shock_fraction")
    subsonic_setup = case.get("id") == "G_subsonic"
    if not subsonic_setup and (fraction is None or case.get("p_out_Pa") is not None):
        return result
    if fraction is not None and not 0 < fraction < 1:
        raise ValueError(
            "The shock fraction must lie strictly inside the divergent section"
        )
    from nozzle_hyperbolic.models.euler_nozzle.analytical.perfect_gas import (
        area_mach,
        mach_from_area_ratio,
        normal_shock,
        pressure_ratio,
    )

    geometry = WallGeometry(geometry_metadata(case))
    roots = geometry.profile.deriv().roots()
    candidates = [geometry.x_min, geometry.x_max] + [
        float(z.real)
        for z in roots
        if abs(z.imag) < 1e-10 and geometry.x_min < z.real < geometry.x_max
    ]
    area = lambda x: float(geometry.sample(np.asarray(x))["area"])
    throat = min(candidates, key=area)
    gamma = 1.4
    if subsonic_setup:
        exit_mach = mach_from_area_ratio(
            area(geometry.x_max) / area(throat), gamma, "sub"
        )
        critical = float(case["p0_Pa"] * pressure_ratio(exit_mach, gamma))
        if float(case["p_out_Pa"]) <= critical:
            result.update(
                archive_p_out_Pa=case["p_out_Pa"],
                p_out_Pa=1.002 * critical,
                sonic_limit_p_out_Pa=critical,
                p_out_source="Within 0.2 percent above sonic limiting exit pressure on selected contour; archived 646999 Pa selects internal shock",
            )
        return result
    position = throat + float(fraction) * (geometry.x_max - throat)
    shock_area = area(position)
    m1 = mach_from_area_ratio(shock_area / area(throat), gamma, "sup")
    _, m2, total_ratio = normal_shock(m1, gamma)
    astar2 = shock_area / area_mach(m2, gamma)
    exit_mach = mach_from_area_ratio(area(geometry.x_max) / astar2, gamma, "sub")
    result.update(
        p_out_Pa=float(case["p0_Pa"] * total_ratio * pressure_ratio(exit_mach, gamma)),
        target_shock_x_m=position,
        p_out_source="Exact polynomial contour and classical normal-shock/area-Mach law; shock fraction of divergent length",
    )
    return result


class BoundedBicubic:
    """Native interpolants on an explicitly audited complete-stencil rectangle."""

    def __init__(self, source, domain):
        self.source = source
        for name, value in domain.items():
            setattr(self, name, float(value))

    def __getattr__(self, name):
        return getattr(self.source, name)

    def get_state(self, input_pair, x, y):
        return self.source.get_state(input_pair, x, y)


class _WallFluid:
    """Change only the wall correlation's viscosity to the paper's Beattie law."""

    def __init__(self, source):
        self.source = source

    def get_state(self, pair, x, y):
        state = self.source.get_state(pair, x, y)
        quality, pressure = state["Q"], state["p"]
        saturated = self.source.saturation_transport
        phase = (quality >= 0) & (quality <= 1) & (pressure <= saturated["p"][-1])
        safe_quality = jnp.where(phase, quality, 0.5)
        props = {
            k: jnp.interp(jnp.log(pressure), jnp.log(saturated["p"]), saturated[k])
            for k in ("rho_l", "rho_v", "mu_l", "mu_v")
        }
        mixture = viscosity_Beattie(
            props["mu_l"], props["mu_v"], props["rho_l"], props["rho_v"], safe_quality
        )
        result = (
            state.to_dict(include_aliases=True)
            if hasattr(state, "to_dict")
            else dict(state)
        )
        viscosity = jnp.where(phase, mixture, state["mu"])
        result.update(mu=viscosity, viscosity=viscosity, dynamic_viscosity=viscosity)
        return result


class BeattieHaalandWall:
    """Production Haaland/laminar stress with explicit Beattie two-phase μ."""

    dependency_radius = 0

    def __init__(self, hydraulic_diameter, roughness):
        self.wall = HaalandWall(hydraulic_diameter, roughness=roughness, heat_flux=0.0)

    def drag_coefficient(self, physical, radius, fluid):
        return self.wall.drag_coefficient(physical, radius, _WallFluid(fluid))

    def __call__(self, physical, radius, fluid):
        return self.wall(physical, radius, _WallFluid(fluid))

    def explicit_timestep(self, physical, radius, fluid, wall_measure_per_volume):
        return self.wall.explicit_timestep(
            physical, radius, _WallFluid(fluid), wall_measure_per_volume
        )


def _saturation_transport(source):
    """Unmodified HEOS phase transport sampled on a documented pressure grid."""
    host = jxp.Fluid(source.fluid_name, backend="HEOS")
    critical = float(host.critical_point["pressure"])
    pressures = np.geomspace(source.p_min, 0.98 * critical, 240)
    values = {key: [] for key in ("rho_l", "rho_v", "mu_l", "mu_v")}
    for pressure in pressures:
        for label, quality in (("l", 0.0), ("v", 1.0)):
            state = host.get_state(jxp.PQ_INPUTS, float(pressure), quality)
            values["rho_" + label].append(float(state["rho"]))
            values["mu_" + label].append(float(state["mu"]))
    if not all(
        np.all(np.isfinite(v)) and np.all(np.asarray(v) > 0) for v in values.values()
    ):
        raise ValueError(
            "Invalid saturated density/viscosity in Beattie wall preparation"
        )
    return {
        "p": jnp.asarray(pressures),
        **{k: jnp.asarray(v) for k, v in values.items()},
    }


def _stencil_audit(fluid, domain, *, allow_gruneisen_holes=False):
    """Audit native interpolation cells without replacing any EOS value.

    A CO2 table can contain isolated undefined Gruneisen stencils. When
    explicitly requested, retain those holes, enumerate them, and verify that
    the unmodified interpolator returns nonfinite G there. NozzleModel's
    existing finite-G admissibility check then excludes every such state.
    All other required properties must retain complete finite stencils.
    """
    h = np.asarray(fluid.h_vals)
    p = np.exp(np.asarray(fluid.logP_vals))
    # Include both neighboring cells at an exact grid-node endpoint. Floating
    # arithmetic in the interpolator's floor can select either neighbor.
    i0 = max(0, int(np.searchsorted(h, domain["h_min"], side="left")) - 1)
    i1 = min(len(h) - 1, int(np.searchsorted(h, domain["h_max"], side="right")))
    j0 = max(0, int(np.searchsorted(p, domain["p_min"], side="left")) - 1)
    j1 = min(len(p) - 1, int(np.searchsorted(p, domain["p_max"], side="right")))
    checked = {}
    rejected_regions = []
    for name in (
        "density",
        "temperature",
        "entropy",
        "speed_of_sound",
        "gruneisen",
        "quality_mass",
    ):
        values = np.asarray(fluid.table[name]["value"])[i0 : i1 + 1, j0 : j1 + 1]
        coefficients = np.asarray(fluid.table[name]["coeffs"])[i0:i1, j0:j1]
        bad_nodes = ~np.isfinite(values)
        bad_cells = ~np.isfinite(coefficients).all(axis=-1)
        if bad_nodes.any() or bad_cells.any():
            if name != "gruneisen" or not allow_gruneisen_holes:
                raise ValueError(f"Invalid interpolation stencil for {name}")
            for local_i, local_j in np.argwhere(bad_cells):
                i, j = int(local_i + i0), int(local_j + j0)
                rejected_regions.append(
                    dict(
                        i=i,
                        j=j,
                        h_interval_J_kg=[float(h[i]), float(h[i + 1])],
                        p_interval_Pa=[float(p[j]), float(p[j + 1])],
                    )
                )
            if not rejected_regions:
                raise ValueError(
                    "Undefined G nodes have no explicitly rejected interpolation cells"
                )
            centers_h = jnp.asarray(
                [np.mean(r["h_interval_J_kg"]) for r in rejected_regions]
            )
            centers_p = jnp.asarray(
                [np.sqrt(np.prod(r["p_interval_Pa"])) for r in rejected_regions]
            )
            sampled_g = np.asarray(
                fluid.get_state(jxp.HmassP_INPUTS, centers_h, centers_p)["G"]
            )
            if not np.all(~np.isfinite(sampled_g)):
                raise ValueError(
                    "Native interpolator did not expose every inadmissible G stencil"
                )
        if name in ("density", "temperature", "speed_of_sound") and np.any(values <= 0):
            raise ValueError(f"Unpopulated interpolation stencil for {name}")
        finite_values = values[np.isfinite(values)]
        checked[name] = dict(
            node_min=float(finite_values.min()),
            node_max=float(finite_values.max()),
            nonfinite_nodes=int(bad_nodes.sum()),
            nonfinite_coefficient_cells=int(bad_cells.sum()),
        )
    return dict(
        domain=domain,
        node_count=int((i1 - i0 + 1) * (j1 - j0 + 1)),
        checked=checked,
        complete_rectangle=not rejected_regions,
        admissible_domain_kind=(
            "rectangle with native finite-G exclusions"
            if rejected_regions
            else "complete rectangle"
        ),
        rejected_gruneisen_stencils=rejected_regions,
        native_finite_G_rejection_verified=bool(rejected_regions),
        stencil_bounds=dict(
            h_min=float(h[i0]),
            h_max=float(h[i1]),
            p_min=float(p[j0]),
            p_max=float(p[j1]),
        ),
    )


def prepare_fluid(case, output):
    """Prepare one shared fluid table; no flow equations are solved here."""
    started = time.perf_counter()
    name = str(case["fluid"])
    if name.lower() in ("air", "perfect_gas", "ideal_gas", "ideal air"):
        fluid = jxp.FluidPerfectGas("Air")
        fluid = eqx.tree_at(
            lambda f: (f.constants.gamma, f.constants.R),
            fluid,
            (jnp.asarray(1.4), jnp.asarray(287.0)),
        )
        return fluid, dict(
            backend="JAXProp FluidPerfectGas",
            gamma=1.4,
            R_J_kgK=287.0,
            preparation_s=time.perf_counter() - started,
            equilibrium=True,
        )
    output = Path(output)
    tables = output / "fluid_tables"
    tables.mkdir(parents=True, exist_ok=True)
    if name.lower() in ("co2", "carbondioxide"):
        configuration = dict(
            fluid_name="CO2",
            backend="HEOS",
            h_min=200e3,
            h_max=650e3,
            p_min=6e5,
            p_max=150e5,
            N_h=300,
            N_p=300,
            table_name="turboexpo_CO2_equilibrium_300x300",
        )
    elif name.lower() == "mdm":
        host = jxp.Fluid("MDM", backend="HEOS")
        reservoir = [
            host.get_state(jxp.PT_INPUTS, p, t)
            for p, t in ((9.20e5, 541.0), (4.59e5, 512.0), (0.84e5, 537.0))
        ]
        enthalpies = [float(s["h"]) for s in reservoir]
        drops = [float(s["a"]) ** 2 for s in reservoir]
        configuration = dict(
            fluid_name="MDM",
            backend="HEOS",
            h_min=min(enthalpies) - 2 * max(drops),
            h_max=max(enthalpies) + 0.25 * max(drops),
            p_min=1e3,
            p_max=10.5e5,
            N_h=300,
            N_p=300,
            table_name="turboexpo_MDM_equilibrium_300x300",
        )
    else:
        raise ValueError(f"Unspecified published fluid {name!r}")
    configuration.update(case.get("properties", {}))
    log = io.StringIO()
    table_path = tables / (configuration["table_name"] + ".pkl")
    cached = table_path.exists()
    try:
        with redirect_stdout(log), redirect_stderr(log):
            source = jxp.FluidBicubic(**configuration, table_dir=str(tables))
    finally:
        with (tables / (configuration["table_name"] + ".log")).open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(log.getvalue())
    # Keep the admissible rectangle a full cell inside the table boundary.
    h = np.asarray(source.h_vals)
    p = np.exp(np.asarray(source.logP_vals))
    domain = dict(
        h_min=float(h[1]), h_max=float(h[-2]), p_min=float(p[1]), p_max=float(p[-2])
    )
    audit = _stencil_audit(
        source, domain, allow_gruneisen_holes=name.lower() in ("co2", "carbondioxide")
    )
    fluid = BoundedBicubic(source, domain)
    if name.lower() in ("co2", "carbondioxide"):
        fluid.saturation_transport = _saturation_transport(source)
    metadata = dict(
        backend="JAXProp equilibrium HEOS bicubic",
        configuration=configuration,
        domain_audit=audit,
        loaded_existing_table=cached,
        table_sha256=hashlib.sha256(table_path.read_bytes()).hexdigest(),
        table_file=str(table_path),
        preparation_s=time.perf_counter() - started,
        equilibrium=True,
        viscosity="JAXProp thermodynamics; Beattie two-phase mixture viscosity for the wall law",
        wall_saturation_transport=(
            dict(
                points=240,
                pressure_grid="logarithmic",
                interpolation="linear in log(p)",
                upper_pressure="0.98 times critical pressure",
                properties="HEOS saturated rho_l,rho_v,mu_l,mu_v",
            )
            if hasattr(fluid, "saturation_transport")
            else None
        ),
    )
    return fluid, metadata


def build_model(case, cells, fluid, initial=None):
    """Bind the current conservative HEM model and physical wall policy."""
    case = resolve_case(case)
    mesh = make_mesh(case, cells)
    sampled = mesh.geometry.sample(mesh.centers)
    wall = None
    if case.get("friction", False):
        if hasattr(fluid, "saturation_transport"):
            wall = BeattieHaalandWall(
                sampled["hydraulic_diameter"], float(case.get("roughness_m", 0.0))
            )
        else:
            wall = HaalandWall(
                sampled["hydraulic_diameter"],
                roughness=float(case.get("roughness_m", 0.0)),
                heat_flux=0.0,
            )
    outlet = case.get(
        "outlet_regime", "subsonic" if case.get("p_out_Pa") else "supersonic"
    )
    back_pressure = case.get("p_out_Pa")
    if back_pressure is None and outlet == "supersonic":
        back_pressure = case.get("initial_back_pressure_Pa")
    return NozzleModel(
        mesh,
        fluid,
        p0=float(case["p0_Pa"]),
        T0=float(case["T0_K"]),
        back_pressure=back_pressure,
        outlet_regime=outlet,
        initial_state=initial,
        wall_closure=wall,
        wall_perimeter=sampled["perimeter"],
    )


# Historical clients used both names; preserve one unambiguous implementation.
build_mesh = make_mesh
