"""Small data/geometry checks before any validation campaign."""

import hashlib
import json
import subprocess

import numpy as np

from .data import ROOT, audit, settings
from .run import make_mesh


def main():
    evidence = audit()
    revision = settings()["andrea_revision"]
    for path in (ROOT / "data").glob("*.xlsx"):
        upstream = "dev_scripts/Lettieri_nozzle.xlsx" if path.name == "Lettieri_nozzle.xlsx" else "validation_cases/Lettieri2018/" + path.name
        original = subprocess.check_output(["git", "show", revision + ":" + upstream])
        assert hashlib.sha256(original).hexdigest() == evidence["files"][path.name]["sha256"]
    meshes = [make_mesh(24, stretch)[0] for stretch in (1.0, 1.6)]
    np.testing.assert_allclose(sum(meshes[0].volumes), sum(meshes[1].volumes), rtol=1e-14)
    for mesh in meshes:
        np.testing.assert_allclose(np.sum(mesh.area_derivative_integrals), mesh.area_faces[-1] - mesh.area_faces[0], rtol=1e-14)
        # The integrated pressure/geometry identity at uniform pressure.
        pressure = 5.0e6
        balance = pressure * np.diff(mesh.area_faces) - pressure * mesh.area_derivative_integrals
        np.testing.assert_array_equal(balance, np.zeros(mesh.n_cells))
    double, _ = make_mesh(24, width=2 * settings()["geometry"]["width_m"])
    np.testing.assert_allclose(double.volumes, 2 * meshes[0].volumes, rtol=1e-14)
    np.testing.assert_allclose(double.area_faces, 2 * meshes[0].area_faces, rtol=1e-14)
    cfg = settings()["geometry"]
    # The paper gives only two significant digits for this nominal area.
    np.testing.assert_allclose(2 * cfg["width_m"] * 0.0015431, cfg["publication_throat_area_m2"], rtol=1e-4)
    evidence["checks"] = ["five 13-point pressure profiles", "L1/L5 exact agreement with publication pressures",
                          "byte-exact upstream Excel copies", "positive uniform/nonuniform mesh",
                          "mesh-independent exact spline volume", "uniform-pressure geometry identity",
                          "full-channel width and half-channel mass-flow conversion"]
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs/data_audit.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print("All data and geometry audits passed.")


if __name__ == "__main__":
    main()
