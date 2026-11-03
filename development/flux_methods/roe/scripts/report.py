"""Rebuild the final Roe HTML and Markdown from saved evidence; no simulations.

After cloning, first run development.flux_methods.roe.scripts.restore_evidence.
The shared benchmark renderer owns the data/evidence model; this study opts
into the common flux-report presentation shell and saved-array Sod plots.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from development.historical.benchmarking.scripts.report import (
    _flow_inputs, audit_html, read_campaign, render,
)


from .presentation import unify_report


STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=STUDY / "outputs/roe_steady_verification/summary.json")
    parser.add_argument("--output", type=Path, default=STUDY / "REPORT.html")
    parser.add_argument("--presentation-only", action="store_true",
                        help="Rebuild HTML and packaging audit only; preserve existing numerical/analysis evidence and Markdown")
    args = parser.parse_args()
    outputs = STUDY / "outputs"
    required = [args.campaign, outputs / "roe_audit.json", outputs / "roe_transient.json",
                outputs / "real_gas_summary.json", outputs / "figures/sod_final_profiles.png"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("Saved evidence is missing. Run development.flux_methods.roe.scripts.restore_evidence first. Missing: " + ", ".join(missing))
    campaign, originals, source_hash = read_campaign(args.campaign)
    audit_raw = (outputs / "roe_audit.json").read_bytes()
    flow, flow_files = _flow_inputs(outputs / "roe_transient.json", outputs / "real_gas_summary.json",
                                   outputs / "figures/sod_final_profiles.png")
    html = render(campaign, original_files=originals, input_name=args.campaign.name,
                  input_sha256=source_hash, roe_audit=json.loads(audit_raw), roe_audit_raw=audit_raw,
                  flow_validation=flow, flow_original_files=flow_files)
    html = unify_report(html, sod_raw=flow_files["validation/roe_sod.json"])
    audit = audit_html(html)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8", newline="\n")
    (args.campaign.parent / "report_structure_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if not args.presentation_only and campaign.get("verification_only") is True:
        subprocess.run([sys.executable, "-B", "-m", "development.flux_methods.roe.scripts.analyze_steady",
                        "--input", str(args.campaign)], cwd=ROOT, check=True)
        shock_id = "euler_annular__shock35__muscl_van_leer__n400__cap10__current__flux_roe.json"
        shock = args.campaign.parent / "scenarios" / shock_id
        if shock.is_file():
            subprocess.run([sys.executable, "-B", "-m", "development.flux_methods.roe.scripts.analyze_shock_history",
                            "--input", str(shock), "--campaign", str(args.campaign)], cwd=ROOT, check=True)
    if not args.presentation_only:
        subprocess.run([sys.executable, "-B", "-m", "development.flux_methods.roe.scripts.summarize",
                        "--campaign", str(args.campaign), "--output", str(args.output.with_name("VERIFICATION_REPORT.md")),
                        "--skip-sod-plot"], cwd=ROOT, check=True)
    print(f"Wrote {args.output} ({audit['html_bytes']:,} bytes); offline evidence audit passed.")


if __name__ == "__main__":
    main()
