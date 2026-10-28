"""Present retained custom adapted-start outcomes in the unified study report."""

from pathlib import Path


def render(report, output):
    """Use the saved summary; report construction never evaluates the model."""
    from .report import render as render_report

    study = Path(__file__).resolve().parents[1]
    return render_report(study/"outputs", study/"REPORT.html", adapted_output=Path(output))
