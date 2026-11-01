"""Installed physical-case resources, model boundaries and reference projection.

These are source API tests. They do not import development benchmark code or
advance a numerical solution.
"""

from hashlib import sha256
from importlib.resources import files
import json

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from nozzle_hyperbolic.verification import (
    available_cases, build_case, case_from_dict, load_case,
)


NOZZLE = {
    "subsonic", "sonic_limit", "shock_35", "shock_70", "shock_exit",
    "overexpanded", "adapted", "underexpanded",
}
ANNULAR = {
    "radial_outflow", "radial_inflow", "sonic_exit", "swirl_diffusion",
    "conical_subsonic", "curved_subsonic", "supersonic_expansion", "shock35",
    "shock70", "radial_outflow_m005", "radial_outflow_m020", "radial_outflow_swirl",
}


def test_installed_catalogue_has_the_retained_physical_families_and_byte_identity():
    expected = {f"euler_nozzle/{name}" for name in NOZZLE} | {f"euler_annular/{name}" for name in ANNULAR}
    assert set(available_cases()) == expected
    assert set(available_cases("euler_nozzle")) == {f"euler_nozzle/{name}" for name in NOZZLE}
    for name in expected:
        definition = load_case(name)
        raw = files("nozzle_hyperbolic").joinpath(definition.source_name).read_bytes()
        assert sha256(raw).hexdigest() == definition.source_sha256
        assert definition.case_id == name
        assert "numerical" not in definition.data


def test_definition_copies_and_strict_schema_prevent_undeclared_physics_changes():
    original = load_case("euler_nozzle/subsonic")
    altered = original.to_dict()
    altered["inlet"]["p0"] = 3e5
    assert original.data["inlet"]["p0"] == 2e5
    independent = case_from_dict(altered)
    altered["inlet"]["p0"] = 4e5
    assert independent.data["inlet"]["p0"] == 3e5
    assert independent.source_sha256 != original.source_sha256
    invalid = original.to_dict()
    invalid["numerical"] = {"cfl": 1.0}
    with pytest.raises(ValueError, match="unknown keys"):
        case_from_dict(invalid)
    invalid = original.to_dict()
    invalid["geometry"]["throat_area"] = 0
    with pytest.raises(ValueError, match="positive"):
        case_from_dict(invalid)
    invalid = load_case("euler_annular/radial_outflow").to_dict()
    invalid["inlet"]["alpha_degrees"] = 30.0
    invalid["inlet"]["alpha_tangent"] = 0.1
    with pytest.raises(ValueError, match="not both"):
        case_from_dict(invalid)
    with pytest.raises(ValueError, match="case ID"):
        load_case("euler_nozzle/../../unrelated")


@pytest.mark.parametrize("case_id", available_cases())
def test_every_physical_case_builds_finite_cold_states_and_usable_reference(case_id):
    problem = build_case(case_id, cells=4)
    model, mesh = problem.model, problem.mesh
    initial = problem.initial_state
    assert initial.shape == (6, model.nvar)
    assert np.isfinite(initial).all()
    assert bool(np.all(model.cell_admissible(initial[1:-1])))
    analytical = problem.reference_fields()
    assert analytical["W"].shape == (4, model.nvar)
    assert analytical["q"].shape == (4, model.nvar)
    assert np.isfinite(analytical["W"]).all()
    assert np.all(analytical["p"] > 0)
    assert np.all(analytical["rho"] > 0)
    assert not np.allclose(initial[1:-1], analytical["W"])
    payload = problem.legacy_payload
    json.dumps(payload, allow_nan=False)
    np.testing.assert_array_equal(payload["initial_state"], initial)
    np.testing.assert_array_equal(payload["variable_scales"], model.variable_scales)
    np.testing.assert_array_equal(payload["faces"], mesh.faces)
    assert payload["model"] == problem.definition.model_name
    assert payload["back_pressure"] > 0
    assert all(np.isfinite(value) and value > 0 for value in problem.field_scales.values())


def test_nozzle_conditions_select_the_declared_interior_regimes():
    unchoked = build_case("euler_nozzle/subsonic", 6)
    throat = unchoked.reference_fields([unchoked.geometry.length / 2])
    np.testing.assert_allclose(throat["M"], 0.6, rtol=2e-12)
    for name, fraction in (("shock_35", 0.35), ("shock_70", 0.70)):
        problem = build_case("euler_nozzle/" + name, 6)
        expected = problem.geometry.length * (1 + fraction) / 2
        assert problem.shock_position == expected
        points = np.array([expected - 1e-8, expected + 1e-8])
        reference = problem.reference_fields(points)
        assert reference["M"][0] > 1 > reference["M"][1]
        assert reference["p"][1] > reference["p"][0]
        np.testing.assert_allclose(reference["mdot"], reference["mdot"][0], rtol=1e-11)
        np.testing.assert_allclose(reference["h0"], reference["h0"][0], rtol=1e-13)
    exit_shock = build_case("euler_nozzle/shock_exit", 6)
    endpoint = exit_shock.reference_fields([exit_shock.geometry.length])
    assert endpoint["M"][0] > 1
    assert endpoint["downstream_exit_pressure"] == exit_shock.back_pressure
    for name in ("overexpanded", "adapted", "underexpanded"):
        problem = build_case("euler_nozzle/" + name, 6)
        assert problem.model.outlet_regime == "supersonic"
        assert problem.reference_fields([problem.geometry.length])["M"][0] > 1


@pytest.mark.parametrize("case_id", [
    "euler_nozzle/subsonic", "euler_nozzle/shock_35", "euler_nozzle/sonic_limit",
    "euler_annular/radial_outflow_swirl", "euler_annular/shock35",
    "euler_annular/sonic_exit",
])
def test_independent_finite_volume_projection_matches_storage_and_is_not_point_sampling(case_id):
    problem = build_case(case_id, cells=6)
    reference = problem.cell_reference()
    recovered = np.asarray(problem.model.cell_conserved(reference["W"]))
    scale = np.asarray(problem.model.conservative_scales)
    np.testing.assert_allclose(recovered / scale, reference["q"] / scale,
                               rtol=5e-12, atol=5e-13)
    np.testing.assert_allclose(reference["integrated"], reference["q"] * problem.mesh.volumes[:, None], rtol=2e-14)
    assert not np.allclose(reference["W"], problem.reference_fields()["W"], rtol=1e-9, atol=1e-10)
    np.testing.assert_array_equal(problem.reference_cell_contents(), reference["integrated"])
    np.testing.assert_array_equal(problem.reference_cell_state(), reference["W"])
    assert all(np.isfinite(value).all() for value in problem.reference_cell_fields().values() if isinstance(value, np.ndarray))


def test_sonic_limit_notes_do_not_disable_accuracy_targets():
    for name in ("euler_nozzle/sonic_limit", "euler_annular/sonic_exit"):
        case = load_case(name).data
        assert case["known_limit"]
        assert case["accuracy"]["recommended_finest_cells"] == 400
        assert all(value > 0 for value in case["accuracy"].values())
    smooth = load_case("euler_annular/radial_outflow").data["accuracy"]
    assert smooth["conservative_normalized_l1"] == 1e-3
    assert smooth["Cp_length_mean_absolute"] == 0.01
    for name in ("shock35", "shock70"):
        accuracy = load_case("euler_annular/" + name).data["accuracy"]
        assert accuracy == {"recommended_finest_cells": 400, "shock_position_cells": 2.0}


def test_mesh_is_a_caller_choice_not_a_silent_yaml_override():
    with pytest.raises(ValueError, match="integer >= 2"):
        build_case("euler_nozzle/subsonic", True)
    with pytest.raises(ValueError, match="integer >= 2"):
        build_case("euler_nozzle/subsonic", 1)
