"""Generate reviewable physical snapshots into a NEW explicitly named folder.

Example: poetry run python -m tests.regression.generate_baselines --output review_data
Normal pytest runs only read committed data and never call this generator.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import jax

jax.config.update("jax_enable_x64", True)  # This CLI is independent of pytest.
import numpy as np

from nozzle_hyperbolic.solvers.fvm import Scheme, solve_transient

from tests._support.baselines import source_files, steady_arrays, steady_settings
from tests._support.regression import (
    STEADY_CASES,
    assert_transient_budget,
    solve_steady_case,
)
from tests._support.transient_cases import perturbed_two_fluid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    destination = arguments.output.resolve()
    if destination.exists():
        parser.error(
            "Output already exists: generate into a new folder and review the diff"
        )
    root = Path(__file__).resolve().parents[2]
    identity = source_files(root)
    configurations = {}
    snapshots = {}
    for name in STEADY_CASES:
        print("Generating", name, flush=True)
        run = solve_steady_case(name)
        snapshots[name] = steady_arrays(run)
        configurations[name] = steady_settings(run)

    name = "two_fluid/pressure_perturbation"
    model, initial, time = perturbed_two_fluid()
    scheme = Scheme(
        flux="model_compatibility",
        reconstruction="muscl_van_leer",
        newton_backtracking=True,
    )
    solution = solve_transient(model, (time,), scheme, initial_state=initial)[-1]
    assert_transient_budget(model, initial, solution)
    snapshots[name] = dict(
        state=np.asarray(solution.state),
        conserved=np.asarray(model.cell_conserved(solution.state[1:-1])),
        variable_scales=model.variable_scales,
        conservative_scales=model.conservative_scales,
        faces=model.mesh.faces,
    )
    configurations[name] = dict(
        kind="transient",
        time=time,
        scheme=asdict(scheme),
        cells=model.mesh.n_cells,
        physics="closure-free perfect-gas phase fixtures; finite pressure perturbation",
        initial_state=np.asarray(initial).tolist(),
    )
    assert (
        source_files(root) == identity
    ), "Source changed while baselines were generated"
    destination.mkdir(parents=True)
    manifest = dict(
        schema_version=1,
        created_utc=datetime.now(timezone.utc).isoformat(),
        git_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        dirty_source=bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=all", "--", "src"],
                cwd=root,
                text=True,
            ).strip()
        ),
        source_files=identity,
        python=sys.version,
        versions={
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "jaxprop", "numpy", "scipy")
        },
        precision="float64",
        backend=jax.default_backend(),
        comparison=dict(normalized_atol=5e-8, rtol=5e-7),
        cases={},
    )
    for name, arrays in snapshots.items():
        filename = name.replace("/", "_") + ".npz"
        path = destination / filename
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        manifest["cases"][name] = dict(
            file=filename,
            sha256=sha256(path.read_bytes()).hexdigest(),
            configuration=configurations[name],
        )
    with zipfile.ZipFile(
        destination / "source.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for name in identity:
            archive.writestr(name, (root / name).read_bytes())
    manifest["source_archive_sha256"] = sha256(
        (destination / "source.zip").read_bytes()
    ).hexdigest()
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print("Generated", len(snapshots), "verified snapshots in", destination)


if __name__ == "__main__":
    main()
