"""Common data/domain helpers for retained custom adapted-start experiments."""

import hashlib
from pathlib import Path
import numpy as np
import pysolver_view as psv
from development.roberto.historical.euler_nozzle import verify_euler_nozzle_steady as baseline
from .investigate_ptc import CASE_IDS, point_hash
from .problem import SteadyNozzleProblem

STUDY = Path(__file__).resolve().parents[1]
CASES = dict(zip(CASE_IDS, baseline.CASES))

def source_hashes():
    """Fingerprint Euler/shared production code, excluding independent annular work."""
    import nozzle_hyperbolic

    package = Path(nozzle_hyperbolic.__file__).resolve().parent
    return {
        "src/nozzle_hyperbolic/"
        + p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in package.rglob("*.py")
        if "euler_annular" not in p.relative_to(package).parts
    }

def clean(value):
    """Encode actual records, preserving unavailable/nonfinite values as null."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)

class RestAdmissibleNozzleProblem(SteadyNozzleProblem):
    """Use positive thermodynamic variables and finite signed velocity."""

    def _validate(self, z):
        """Validate dimensionless interior coordinates without modifying them."""
        point = np.asarray(z, dtype=float)
        if point.shape != self.x0.shape:
            raise ValueError("Incorrect interior-state shape")
        physical = (point * self.variable_scale).reshape(-1, 3)
        if not np.isfinite(physical).all() or np.any(physical[:, [0, 2]] <= 0):
            raise psv.DomainError(
                "Trial pressure/enthalpy must be positive and all variables finite"
            )
        return point

    @staticmethod
    def state_is_admissible(full):
        """Accept rest and signed velocity in implicit trial fields as well."""
        full = np.asarray(full)
        return np.isfinite(full).all() and np.all(full[1:-1, [0, 2]] > 0)
