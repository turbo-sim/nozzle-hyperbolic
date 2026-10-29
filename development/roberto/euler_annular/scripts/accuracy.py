"""Analytical accuracy reassessment of saved fields; no numerical solver imports."""

import math


CP_MEAN_ABSOLUTE_LIMIT = 0.01
ACCURACY_DEFINITION = "annular_cp_arclength_mean_absolute_v1"


def assessed_errors(saved, run):
    """Add a physical-length mean absolute Cp error to an immutable run record.

    Midpoint quadrature uses the actual physical-cell widths, including on
    graded meshes. The original maximum error remains supplementary evidence.
    """
    faces = saved["m_faces"]
    numerical, analytical = run["fields"]["Cp"], saved["reference"]["Cp"]
    if len(faces) != len(numerical)+1 or len(numerical) != len(analytical):
        raise ValueError("Cp error requires one reference/numerical value per physical cell")
    widths = [right-left for left, right in zip(faces[:-1], faces[1:])]
    if not widths or any(not math.isfinite(width) or width <= 0 for width in widths):
        raise ValueError("Cp error requires finite, strictly increasing face coordinates")
    mean = math.fsum(width*abs(actual-exact)
                     for width, actual, exact in zip(widths, numerical, analytical))/math.fsum(widths)
    return dict(run["errors"], Cp_mean_absolute=mean)


def accuracy_target_met(saved, run):
    """Current accuracy assessment, separate from the original solve's status."""
    errors = assessed_errors(saved, run)
    if saved.get("shock_m") is not None:
        return math.isfinite(errors["shock_error_cells"]) and errors["shock_error_cells"] <= 2
    values = (errors["Cp_mean_absolute"], *errors["conservative_L1"])
    return (all(math.isfinite(value) for value in values)
            and errors["Cp_mean_absolute"] <= CP_MEAN_ABSOLUTE_LIMIT
            and max(errors["conservative_L1"]) < 1e-3)


def target_definition(shocked=False):
    return (dict(shock_error_cells=2.) if shocked else
            dict(definition=ACCURACY_DEFINITION, Cp_mean_absolute=CP_MEAN_ABSOLUTE_LIMIT,
                 Cp_weighting="physical meridional cell width / domain length",
                 Cp_comparison="<=", conservative_L1_max=1e-3,
                 conservative_L1_comparison="<"))
