"""Versioned physical verification cases, independent of numerical settings.

Physical YAML resources live alongside the corresponding models. Builders use
production models and independently scoped analytical references; benchmark
controllers, timing, reconstruction and nonlinear policy belong to clients.
"""

from .catalog import CaseDefinition, available_cases, case_from_dict, load_case
from .problems import VerificationProblem, build_case

__all__ = [
    "CaseDefinition", "VerificationProblem", "available_cases", "case_from_dict",
    "load_case", "build_case",
]
