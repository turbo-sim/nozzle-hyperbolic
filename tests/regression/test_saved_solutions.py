"""Frozen numerical results supplement analytical verification, not replace it."""

from dataclasses import asdict
from hashlib import sha256
import json
import zipfile

import numpy as np
import pytest

from nozzle_hyperbolic.solvers.fvm import Scheme, solve_transient

from tests._support.baselines import (
    BASELINES,
    read_snapshot,
    steady_arrays,
    steady_settings,
)
from tests._support.regression import assert_saved_state, assert_transient_budget
from tests._support.transient_cases import perturbed_two_fluid

pytestmark = pytest.mark.slow


def test_steady_profiles_match_the_pinned_numerical_version(steady_run):
    expected, configuration, _ = read_snapshot(steady_run.problem.case_id)
    current_configuration = steady_settings(steady_run)
    # Exact source bytes remain in the provenance manifest. Physical comparison
    # uses canonical parsed data so Git's LF/CRLF conversion is harmless.
    configuration.pop("case_sha256")
    current_configuration.pop("case_sha256")
    assert configuration == current_configuration, "Physical/numerical settings changed"
    actual = steady_arrays(steady_run)
    for name in ("faces", "centers", "volumes"):
        assert actual[name].shape == expected[name].shape
        np.testing.assert_allclose(actual[name], expected[name], rtol=2e-12, atol=1e-15)
    assert_saved_state(actual["state"], expected["state"], actual["variable_scales"])
    assert_saved_state(
        actual["conserved"], expected["conserved"], actual["conservative_scales"]
    )


def test_two_fluid_complete_solution_matches_the_pinned_numerical_version():
    model, initial, time = perturbed_two_fluid()
    scheme = Scheme(
        flux="model_compatibility",
        reconstruction="muscl_van_leer",
        newton_backtracking=True,
    )
    expected, configuration, _ = read_snapshot("two_fluid/pressure_perturbation")
    assert configuration["time"] == time
    assert configuration["scheme"] == asdict(scheme)
    np.testing.assert_allclose(
        np.asarray(initial) / model.variable_scales,
        np.asarray(configuration["initial_state"]) / model.variable_scales,
        rtol=2e-13,
        atol=1e-15,
    )
    solution = solve_transient(model, (time,), scheme, initial_state=initial)[-1]
    assert_transient_budget(model, initial, solution)
    assert_saved_state(solution.state, expected["state"], model.variable_scales)
    assert_saved_state(
        model.cell_conserved(solution.state[1:-1]),
        expected["conserved"],
        model.conservative_scales,
    )


def test_baseline_source_archive_matches_its_recorded_identity():
    manifest = json.loads((BASELINES / "manifest.json").read_text(encoding="utf-8"))
    archive = BASELINES / "source.zip"
    assert sha256(archive.read_bytes()).hexdigest() == manifest["source_archive_sha256"]
    with zipfile.ZipFile(archive) as stream:
        assert {
            name: sha256(stream.read(name)).hexdigest() for name in stream.namelist()
        } == manifest["source_files"]
    assert manifest["backend"] == "cpu" and manifest["precision"] == "float64"
