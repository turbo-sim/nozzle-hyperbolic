"""Euler-nozzle study inputs, separate from the model-independent FAS core."""

from functools import lru_cache

import numpy as np

from nozzle_hyperbolic.verification.catalog import case_from_dict, load_case
from nozzle_hyperbolic.verification.problems import build_case


CASE_IDS = (
    "euler_nozzle/subsonic",
    "euler_nozzle/shock_30",
    "euler_nozzle/shock_70",
    "euler_nozzle/adapted",
)


@lru_cache(maxsize=None)
def definition(case_id):
    """Load physical data; derive the requested 30% shock without editing src."""
    if "/" not in case_id:
        case_id = "euler_nozzle/" + case_id
    if case_id not in CASE_IDS:
        raise ValueError(f"Unsupported nozzle study case {case_id!r}")
    if case_id != "euler_nozzle/shock_30":
        return load_case(case_id)
    data = load_case("euler_nozzle/shock_35").to_dict()
    data.update(
        id="shock_30",
        label="Shock at 30% of divergence",
        description=(
            "Study-local derivative of packaged shock_35; only shock_fraction "
            "is changed to 0.30. Independent reference selects backpressure; "
            "the numerical initial state remains model-owned and cold."
        ),
    )
    data["reference"]["value"] = 0.30
    return case_from_dict(
        data,
        source_name="development/multigrid:shock_30 derived from euler_nozzle/shock_35",
    )


def build_problem(case_id, cells, fluid=None):
    return build_case(definition(case_id), int(cells), fluid=fluid)


def build_hierarchy(case_id, grids):
    """Reuse one phase-fluid object; audit the uniform nested physical meshes."""
    problems = []
    fluid = None
    for count in grids:
        problem = build_problem(case_id, count, fluid=fluid)
        fluid = problem.fluid
        problems.append(problem)
    for coarse, fine in zip(problems, problems[1:]):
        if fine.mesh.n_cells != 2 * coarse.mesh.n_cells:
            raise ValueError("The prototype requires adjacent 2:1 nested levels")
        np.testing.assert_allclose(
            coarse.mesh.faces, fine.mesh.faces[::2], rtol=0, atol=1e-15
        )
        np.testing.assert_allclose(
            coarse.mesh.volumes,
            np.asarray(fine.mesh.volumes).reshape(-1, 2).sum(axis=1),
            rtol=2e-12,
            atol=1e-20,
        )
        np.testing.assert_allclose(
            coarse.mesh.area_faces, fine.mesh.area_faces[::2], rtol=2e-14
        )
    return problems


def numerical_fields(problem, state):
    fields = problem.numerical_fields(state)
    return {
        "coordinate": np.asarray(problem.mesh.centers),
        **{
            key: np.asarray(value)
            for key, value in fields.items()
            if np.ndim(value) == 1
        },
    }


def reference_fields(problem):
    x = np.linspace(problem.mesh.faces[0], problem.mesh.faces[-1], 1001)
    if problem.shock_position is not None:
        eps = (x[-1] - x[0]) * 1e-10
        x = np.unique(
            np.r_[x, problem.shock_position - eps, problem.shock_position + eps]
        )
    fields = problem.reference_fields(x)
    return {
        "coordinate": x,
        **{
            key: np.asarray(value)
            for key, value in fields.items()
            if np.shape(value) == x.shape
        },
    }
