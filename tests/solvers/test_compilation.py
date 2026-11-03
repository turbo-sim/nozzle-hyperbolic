"""Cache reuse is measured in fresh processes and with the same JIT callable."""

import io
import json
import os
import subprocess
import sys
import textwrap

import pytest

from nozzle_hyperbolic.solvers.compilation import CompilationReport


@pytest.mark.parametrize("verbose", [False, True])
def test_disk_cache_and_in_process_reuse_are_observed_separately(tmp_path, verbose):
    code = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        import jax
        from nozzle_hyperbolic.solvers.compilation import (
            CompilationReport, configure_compilation_cache,
        )
        root, label = Path(sys.argv[1]), sys.argv[2]
        verbose = sys.argv[3] == 'true'
        configure_compilation_cache(root / 'cache')
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
        import numpy as np

        @jax.jit
        def wave(x):
            return jnp.sin(x) + x * x

        x = np.linspace(0., 1., 16)
        with CompilationReport(root / (label + '.json'), verbose=verbose):
            actual = wave(x).block_until_ready()
        np.testing.assert_allclose(actual, np.sin(x) + x*x, rtol=1e-14)
        # Same function/shapes/dtypes, different dynamic input values.
        with CompilationReport(root / (label + '_replay.json'), verbose=verbose):
            wave(x + 1.).block_until_ready()
        """
    )
    outputs = {}
    for label in ("cold", "warm"):
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                code,
                str(tmp_path),
                label,
                str(verbose).lower(),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=dict(
                os.environ,
                JAX_COMPILATION_CACHE_DIR=str(tmp_path / "cache"),
                JAX_ENABLE_COMPILATION_CACHE="true",
            ),
        )
        outputs[label] = result.stdout
    cold = json.loads((tmp_path / "cold.json").read_text(encoding="utf-8"))
    warm = json.loads((tmp_path / "warm.json").read_text(encoding="utf-8"))
    assert cold["non_cached_requests"] > 0
    assert cold["persistent_cache_hits"] == 0
    assert warm["persistent_cache_hits"] > 0
    assert warm["non_cached_requests"] == 0
    assert warm["cached_request_s"] > 0
    if verbose:
        assert "Compile/load START" in outputs["cold"]
        assert "Compile/load END" in outputs["cold"]
        assert "REUSED executable from disk" in outputs["warm"]
    else:
        # Each report produces one line, including the in-process replay.
        assert len(outputs["cold"].strip().splitlines()) == 2
        assert len(outputs["warm"].strip().splitlines()) == 2
        assert "compiled" in outputs["cold"]
        assert "loaded" in outputs["warm"]
        assert "compile/load time" in outputs["warm"]
    for label in ("cold", "warm"):
        replay = json.loads(
            (tmp_path / f"{label}_replay.json").read_text(encoding="utf-8")
        )
        assert replay["backend_requests"] == 0
        assert replay["persistent_cache_hits"] == 0
        assert replay["completed"]


def test_failed_scope_records_failure_and_releases_the_reporter(tmp_path):
    path = tmp_path / "failure.json"
    stream = io.StringIO()
    with pytest.raises(ValueError, match="original failure"):
        with CompilationReport(path, stream=stream):
            raise ValueError("original failure")
    assert len(stream.getvalue().strip().splitlines()) == 1
    failed = json.loads(path.read_text(encoding="utf-8"))
    assert not failed["completed"]
    assert failed["backend_requests"] == 0
    # Cleanup must allow a later independent report without clearing JAX or
    # removing monitoring listeners owned by other application components.
    with CompilationReport(stream=io.StringIO()) as later:
        pass
    assert later.stats["completed"]
