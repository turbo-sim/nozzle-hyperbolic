"""Audit frozen files, geometry and reference-column identities without a solve."""

import hashlib
import ast
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

from .case import ROOT, UPSTREAM, REGIMES, specification, reference, geometry_arrays, mesh_for
from nozzle_hyperbolic.models.two_fluid._alberto_closures import interfacial_area_original
from .run import regime_audit


def main():
    manifest = json.loads((UPSTREAM / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((UPSTREAM / name).read_bytes()).hexdigest() == expected["sha256"], name
    context = ROOT / "reference_context"
    historical = json.loads((context / "manifest.json").read_text(encoding="utf-8"))
    for name, item in historical["files"].items():
        assert hashlib.sha256((context / name).read_bytes()).hexdigest() == item["sha256"]
    tree = ast.parse((context / "closures_nicfd.py").read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "interfacial_area")
    namespace = {"jax": jax, "jnp": jnp}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "frozen_NICFD_area_law", "exec"), namespace)
    for alpha in (1e-8, 1e-5, .001, .05, .3, .5, .7, .95, .9775, .999, 1-1e-8):
        kwargs = dict(Nb=1e11, Nd=1e11, d_min_b=5e-6)
        np.testing.assert_allclose(interfacial_area_original(jnp.asarray(alpha), **kwargs),
                                   namespace["interfacial_area"](jnp.asarray(alpha), **kwargs), rtol=1e-14)
    print("Configured original area law agrees with the frozen September NICFD function.")
    for regime in REGIMES:
        config, data = specification(regime), reference(regime)
        assert len(data) == config["grid"]["Nx"]
        assert np.all(np.diff(data["x"]) > 0)
        assert data["x"][0] == 0 and data["x"][-1] == config["grid"]["x2"]
        for name in data.dtype.names:
            assert np.isfinite(data[name]).all(), (regime, name)
        area, _ = geometry_arrays(tuple(data["x"]))
        np.testing.assert_allclose(area, data["A"], rtol=2e-10)
        mg, ml = data["alpha_g"] * data["rho_g"], (1 - data["alpha_g"]) * data["rho_l"]
        np.testing.assert_allclose(data["quality_static"], mg / (mg + ml), atol=2e-12)
        np.testing.assert_allclose(data["quality_flow"], mg * data["u_g"] / (mg * data["u_g"] + ml * data["u_l"]), atol=2e-12)
        # Each CSV operand is rounded independently to 13 significant digits;
        # subtracting alpha from one amplifies its absolute rounding error.
        np.testing.assert_allclose(data["mdot"], data["A"] * (mg * data["u_g"] + ml * data["u_l"]), rtol=1e-10)
        np.testing.assert_allclose(data["Ma_mix"], abs(data["u_mix"]) / data["a_mix"], rtol=2e-12)
        assert regime_audit(data, regime)["passed"]
        print(regime, "reference rows", len(data), "physical", len(data)-2,
              "outlet M", data["Ma_mix"][-2], flush=True)
    # Integrated volumes must not depend on how the same domain is partitioned.
    uniform, stretched = mesh_for(12), mesh_for(17, stretch=1.4)
    np.testing.assert_allclose(sum(uniform.volumes), sum(stretched.volumes), rtol=2e-10)
    print("Frozen byte hashes, geometry, reference identities and integrated meshes passed.")
    wrong_branch = {name: reference("subsonic")[name].copy() for name in ("Ma_mix", "u_g", "u_l")}
    wrong_branch["Ma_mix"][20] = 1.05
    assert not regime_audit(wrong_branch, "subsonic")["passed"]
    print("Intended-regime audit rejects a supersonic pocket in a subsonic case.")


if __name__ == "__main__":
    main()
