"""Fresh-process imports and precision errors leave global suite state intact."""

import json
import subprocess
import sys


def test_plain_package_and_numerical_engine_import_no_models_eos_or_plotting():
    code = """
import json
import sys
import nozzle_hyperbolic
plain = [name for name in sys.modules if name.startswith(('nozzle_hyperbolic.models', 'jaxprop', 'matplotlib'))]
import nozzle_hyperbolic.solvers.fvm
core = [name for name in sys.modules if name.startswith(('nozzle_hyperbolic.models', 'jaxprop', 'matplotlib'))]
print(json.dumps({'plain': plain, 'core': core}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout) == {"plain": [], "core": []}


def test_implicit_operator_rejects_disabled_double_precision_in_a_fresh_process():
    code = """
import jax
from nozzle_hyperbolic.models.scalar import AdvectionModel
from nozzle_hyperbolic.solvers.fvm import Operator, Scheme
jax.config.update('jax_enable_x64', False)
try:
    Operator(AdvectionModel(4), Scheme())
except ValueError as error:
    assert 'jax_enable_x64' in str(error), error
else:
    raise AssertionError('Implicit operator silently accepted single precision')
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
