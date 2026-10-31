"""Independent analytical references; no numerical FVM kernels depend on these."""

from .perfect_gas import PerfectGasReference, ReferenceBranchError

__all__ = ["PerfectGasReference", "ReferenceBranchError"]
