"""Map the frozen NICFD inputs to the shared constrained two-fluid model.

The supplementary CSV is a comparison reference. Only the explicitly selected
``reference`` initialization uses its primitive profile as a numerical seed.
"""

from dataclasses import asdict
from functools import lru_cache
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np
from numpy.polynomial.legendre import leggauss
import yaml

from nozzle_hyperbolic.geometry.duct import get_nozzle_elliot
from nozzle_hyperbolic.numerics.mesh import Mesh
from nozzle_hyperbolic.models.two_fluid import (
    TwoFluidModel, BoundaryConfig, ClosureConfig, CompatibilityConfig,
    StabilizationConfig,
)
from nozzle_hyperbolic.solvers.fvm import Scheme, SteadySettings

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream"
REGIMES = ("subsonic", "supersonic_shock", "supersonic_adapted")
REFERENCE_SHA = "2dc04ad9d474afb727857139df89537fcadb5769"


def numbers(value):
    """PyYAML treats some scientific-notation scalars as strings."""
    if isinstance(value, dict):
        return {key: numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [numbers(item) for item in value]
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def specification(regime):
    if regime not in REGIMES:
        raise ValueError(f"Unknown regime {regime!r}")
    path = UPSTREAM / "data" / "two_component" / regime
    return numbers(yaml.safe_load((path / "settings.yaml").read_text(encoding="utf-8")))


def reference(regime):
    return np.genfromtxt(UPSTREAM / "data" / "two_component" / regime / "solution.csv",
                         delimiter=",", names=True)


@lru_cache(maxsize=32)
def geometry_arrays(coordinates):
    data = get_nozzle_elliot(np.asarray(coordinates))
    return np.asarray(data.A), np.asarray(data.dAdx)


def mesh_for(cells, length=.31, *, stretch=1.):
    """Shared FVM layout with integrated physical geometry.

    Each quadrature interval is split at the contour joins. Geometry is the
    existing physical Elliot implementation, not a fit to reference flow data.
    """
    if cells < 2 or stretch <= 0:
        raise ValueError("Use >=2 physical cells and positive mesh stretch")
    faces = length * np.linspace(0., 1., cells + 1)**stretch
    centers = .5 * (faces[:-1] + faces[1:])
    area, derivative = geometry_arrays(tuple(centers))
    face_area, _ = geometry_arrays(tuple(faces))
    joins = (0.09940, 0.14163, 0.153142)
    z, weight = leggauss(24)
    quadrature, factors, indices = [], [], []
    for i, (left, right) in enumerate(zip(faces[:-1], faces[1:])):
        breaks = [left] + [x for x in joins if left < x < right] + [right]
        for a, b in zip(breaks[:-1], breaks[1:]):
            quadrature.extend(.5 * (a + b) + .5 * (b - a) * z)
            factors.extend(.5 * (b - a) * weight)
            indices.extend([i] * len(z))
    qa, _ = geometry_arrays(tuple(quadrature))
    volumes = np.bincount(indices, weights=qa * factors, minlength=cells)
    return Mesh(faces=faces, centers=centers, volumes=volumes,
                area_faces=face_area, area_centers=area, darea_centers=derivative,
                area_derivative_integrals=np.diff(face_area))


def prepare_fluids(config):
    fluids = []
    metadata = []
    for phase in ("gas", "liquid"):
        spec = dict(config["eos"][phase])
        spec["table_dir"] = str(ROOT / "outputs" / "fluid_tables")
        fluid = jxp.FluidBicubic(**spec)
        fluids.append(fluid)
        metadata.append(dict(phase=phase, configuration=spec,
                             metadata=getattr(fluid, "metadata", None),
                             table_name=getattr(fluid, "table_name", None)))
    return tuple(fluids), metadata


def build_case(config, fluids, cells, *, initialization="cold", stretch=1.,
               newton_iterations=None, cfl=None, max_steps=None, area_law="original"):
    fg, fl = fluids
    bc, numeric, source = config["bc"], config["numerics"], config["sources"]
    mesh = mesh_for(cells, config["grid"]["x2"], stretch=stretch)
    reservoirs = [fluid.get_state(jxp.PT_INPUTS, bc["p0"], bc[f"T_{k}0"])
                  for fluid, k in ((fg, "g"), (fl, "l"))]
    hg, hl = (float(item["h"]) for item in reservoirs)
    rg, rl = (float(item["rho"]) for item in reservoirs)
    quality, slip = bc["quality"], bc["slip"]
    alpha = quality * rl / ((1 - quality) * rg * slip + quality * rl)
    boundary = BoundaryConfig(inlet=bc["inlet"], outlet=bc["outlet"],
        quality=quality, slip=slip, H_g0=hg, H_l0=hl,
        s0=float(reservoirs[0 if quality >= .5 else 1]["s"]),
        dominant_phase="gas" if quality >= .5 else "liquid",
        back_pressure=bc["p_out"], pressure_relaxation=bc["p_out_relax"])
    compatibility = CompatibilityConfig(nc_form=numeric["nc_form"],
        lowmach_diss=numeric["lowmach_diss"], lowmach_diss_boundary=numeric["lowmach_diss_bc"],
        du_blend_lo=numeric["du_blend_lo"], du_blend_hi=numeric["du_blend_hi"],
        delta_factor=numeric["delta_factor"])
    # The September NICFD code called the number-density law directly. The
    # later October default polynomial law is not the publication's closure.
    closures = ClosureConfig(Nb=source["bubble_number"], Nd=source["droplet_number"],
                             mass_transfer=source["mass_transfer"], area_law=area_law)
    stabilization = StabilizationConfig(void_strength=numeric["eps_void"],
        speed_scale=numeric["eps_void_scale"], void_velocity_cutoff=numeric["eps_void_ucut"],
        void_velocity_width=numeric["eps_void_uwidth"])
    if initialization == "reference":
        data = reference(config["case"]["regime"])
        initial = np.column_stack([np.interp(mesh.points, data["x"], data[name])
                                   for name in ("p", "u_g", "u_l", "alpha_g", "h_g", "h_l")])
    elif initialization == "cold":
        # Explicitly documented independent seed: positive flow avoids imposing
        # an arbitrary branch through an absent upstream archived restart.
        speed = 10. * mesh.area_faces[0] / geometry_arrays(tuple(mesh.points))[0]
        initial = np.column_stack((np.full_like(speed, bc["p0"]), slip * speed, speed,
                                   np.full_like(speed, alpha), hg - .5 * (slip * speed)**2,
                                   hl - .5 * speed**2))
    else:
        raise ValueError("initialization must be cold or reference")
    model = TwoFluidModel(mesh, fg, fl, initial_primitives=initial, boundaries=boundary,
                         closures=closures, compatibility=compatibility,
                         stabilization=stabilization, virtual_mass=numeric["k_vm"])
    scheme = Scheme(flux="model_compatibility", reconstruction=numeric["recon"],
        boundary_treatment="coupled", newton_backtracking=True,
        newton_iterations=int(numeric["newton_iter"] if newton_iterations is None else newton_iterations))
    settings = SteadySettings(cfl=float(cfl if cfl is not None else numeric["CFL"]),
        cfl_growth=config["time_control"]["dt_grow"],
        max_steps=int(max_steps if max_steps is not None else config["run"]["n_steps"]))
    effective = dict(boundaries=asdict(boundary), closures=asdict(closures),
        compatibility=asdict(compatibility), stabilization=asdict(stabilization),
        scheme=asdict(scheme), steady=asdict(settings),
        reference_total_rows=int(config["grid"]["Nx"]), physical_cells=cells,
        closure_reference_revision="e185f571f8ea0512a7a16ce53a4b2458f72c33c2",
        initialization=initialization, geometry="existing get_nozzle_elliot; integrated volumes",
        differences=["square banded Newton instead of normal equations",
                     "fixed physical scales instead of adaptive column scales",
                     "generic local Van Albada instead of upstream regularized global formula",
                     "generic CFL/retry controller and strict original-residual gates",
                     "coupled algebraic traces instead of fixed-point/isentrope lookup completion",
                     "physical cells and endpoint traces use the shared FVM mesh convention",
                     "JAXProp installed version recorded; reference used 0.5.6",
                     "upstream ic=restart supplies no restart identity; initialization is explicit"])
    return model, scheme, settings, effective


def environment():
    return {"python": platform.python_version(), **{name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "jaxprop", "numpy", "scipy")}}


def source_hashes():
    repo = ROOT.parents[1]
    folders = (repo / "src/nozzle_hyperbolic/models/two_fluid", repo / "src/nozzle_hyperbolic/solvers/fvm")
    paths = [path for folder in folders for path in folder.glob("*.py")]
    paths += list((ROOT / "scripts").glob("*.py"))
    return {str(p.relative_to(repo)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}
