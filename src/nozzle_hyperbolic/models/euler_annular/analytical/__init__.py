"""Independent analytic/semi-analytic annular Euler verification references.

Nothing in the production discretization or solver imports this package.
Perfect-gas restrictions belong to the references, not to ``AnnularModel``.
"""

from .perfect_gas import PerfectGasReference, ReferenceBranchError
from .transient import cell_reference, density_pulse, exact_homogeneous

__all__ = [
    "PerfectGasReference",
    "ReferenceBranchError",
    "cell_reference",
    "density_pulse",
    "exact_homogeneous",
]
