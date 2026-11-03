"""Portable numerical snapshots with explicit settings and source provenance."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from tests._support.regression import GRIDS, STEADY_SCHEME, STEADY_SETTINGS


BASELINES = Path(__file__).resolve().parents[1] / "regression/data/physical_v1"


def steady_arrays(run):
    """Extract physical arrays, excluding iteration-count and timing details."""
    problem = run.problem
    return dict(
        state=np.asarray(run.solution.state),
        conserved=np.asarray(
            problem.model.cell_conserved(jnp.asarray(run.solution.state[1:-1]))
        ),
        faces=np.asarray(problem.mesh.faces),
        centers=np.asarray(problem.mesh.centers),
        volumes=np.asarray(problem.mesh.volumes),
        variable_scales=np.asarray(problem.model.variable_scales),
        conservative_scales=np.asarray(problem.model.conservative_scales),
    )


def steady_settings(run):
    """Describe the exact physical case and numerical initialization policy."""
    canonical_case = json.dumps(
        run.problem.definition.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return dict(
        kind="steady",
        case_id=run.problem.case_id,
        case_sha256=run.problem.definition.source_sha256,
        case_data_sha256=sha256(canonical_case).hexdigest(),
        grids=list(GRIDS),
        scheme=asdict(STEADY_SCHEME),
        settings=asdict(STEADY_SETTINGS),
        initialization="model-owned cold guess; only successful numerical grid transfer",
        analytical_initialization=False,
    )


def source_files(root):
    """Hash every physical source module and packaged case at generation time."""
    return {
        path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "src").rglob("*"))
        if path.is_file() and path.suffix in (".py", ".yaml")
    }


def read_snapshot(case_id, directory=BASELINES):
    """Reject missing/corrupted baselines rather than regenerating expected data."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    record = manifest["cases"][case_id]
    path = directory / record["file"]
    assert (
        path.parent == directory
    ), "A snapshot filename must stay in its data directory"
    assert sha256(path.read_bytes()).hexdigest() == record["sha256"], path
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name]) for name in archive.files}
    for name, array in arrays.items():
        assert np.all(np.isfinite(array)), (case_id, name)
    return arrays, record["configuration"], manifest
