"""Build the portable Turbo Expo reproduction report from saved evidence only.

This module deliberately imports no thermodynamics or numerical solver. Updating
the presentation never advances a flow solution or changes saved measurements.
"""

from __future__ import annotations

import base64
from datetime import datetime
import gzip
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
ASSETS = Path(__file__).parent / "report_assets"
VENDOR = REPO / "development/flux_methods/shared/assets/vendor"


def clean(value):
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
    return value


def packed(value):
    return (
        json.dumps(
            clean(value), ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        .replace("<", "\\u003c")
        .replace("&", "\\u0026")
    )


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_evidence():
    definition = read_json(ROOT / "case_definitions.json")
    runs = []
    for path in sorted((ROOT / "outputs").rglob("result.json")):
        row = read_json(path)
        if not isinstance(row, dict) or not ("case" in row or "case_id" in row):
            continue
        row["result_file"] = str(path.relative_to(ROOT)).replace("\\", "/")
        case = row.get("case", {})
        row["case_id"] = row.get(
            "case_id", case.get("id") if isinstance(case, dict) else case
        )
        row["flux"] = row.get("flux", row.get("scheme", {}).get("flux", "unknown"))
        row["cells"] = int(row.get("cells", 0))
        row["id"] = str(path.parent.relative_to(ROOT)).replace("\\", "/")
        fields_file = path.with_name("solution.npz")
        if fields_file.exists():
            with np.load(fields_file, allow_pickle=False) as saved:
                count = len(saved["x_m"]) if "x_m" in saved else row["cells"]
                # Constant perfect-gas properties can be saved as zero-dimensional
                # arrays by the EOS. Broadcast those diagnostic values for plotting.
                row["fields"] = {
                    key: (
                        np.full(count, saved[key].item())
                        if saved[key].ndim == 0
                        else saved[key]
                    )
                    for key in saved.files
                    if key != "state"
                }
        verification = path.with_name("verification.json")
        if verification.exists() and not row.get("verification"):
            row["verification"] = read_json(verification)
        runs.append(row)
    references = {}
    for path in sorted((ROOT / "data").glob("*.json")):
        if "reference" in path.stem or "publication" in path.stem:
            references[path.stem] = read_json(path)
    digitized_path = ROOT / "outputs/paper_digitization.json"
    if digitized_path.exists():
        digitized = read_json(digitized_path)
        if digitized.get("validated") is True:
            references["paper_digitization"] = digitized
    qa_images = [
        dict(
            name=path.name,
            data_url="data:image/png;base64,"
            + base64.b64encode(path.read_bytes()).decode("ascii"),
        )
        for path in sorted(
            (ROOT / "outputs").glob("paper_figure9*_digitization_QA.png")
        )
    ]
    provenance_path = ROOT / "provenance.json"
    summary_path = ROOT / "outputs/summary.json"
    phases_path = ROOT / "outputs/phase_diagrams.json"
    return dict(
        definition=definition,
        runs=runs,
        references=references,
        digitization_qa=qa_images,
        provenance=read_json(provenance_path) if provenance_path.exists() else {},
        summary=read_json(summary_path) if summary_path.exists() else {},
        phase_diagrams=read_json(phases_path) if phases_path.exists() else {},
        generated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )


def evidence_archive():
    """Include raw inputs, states, histories and renderer source in one download."""
    paths = []
    for name in (
        "case_definitions.json",
        "settings.json",
        "protocol.json",
        "provenance.json",
        "README.md",
        "RESULTS.md",
        "VALIDATION_COVERAGE.md",
    ):
        if (ROOT / name).exists():
            paths.append(ROOT / name)
    for folder in (ROOT / "data", ROOT / "scripts"):
        paths.extend(
            p
            for p in folder.rglob("*")
            if p.is_file()
            and p.suffix.lower() != ".pdf"
            and "__pycache__" not in p.parts
        )
    paths.extend(
        p
        for p in (ROOT / "outputs").rglob("*")
        if p.is_file()
        and p.suffix in (".json", ".npz", ".csv", ".log", ".zip")
        and "browser_audit" not in p.parts
    )
    paths.extend((ROOT / "outputs").glob("paper_figure9*_digitization_QA.png"))
    paths = sorted(set(paths))
    fingerprints = {
        str(p.relative_to(ROOT))
        .replace("\\", "/"): hashlib.sha256(p.read_bytes())
        .hexdigest()
        for p in paths
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, str(path.relative_to(ROOT)).replace("\\", "/"))
        archive.writestr("SHA256.json", json.dumps(fingerprints, indent=2))
        for name in ("PLOTLY_LICENSE.txt", "MATHJAX_LICENSE.txt", "THIRD_PARTY.md"):
            archive.write(VENDOR / name, "licenses/" + name)
    return base64.b64encode(buffer.getvalue()).decode("ascii"), fingerprints


def main():
    evidence = load_evidence()
    archive, fingerprints = evidence_archive()
    template = (ASSETS / "template.html").read_text(encoding="utf-8")
    replacements = {
        "STYLE": (ASSETS / "style.css").read_text(encoding="utf-8"),
        "PLOTLY": gzip.decompress(
            (VENDOR / "plotly-2.35.2.min.js.gz").read_bytes()
        ).decode("utf-8"),
        "MATHJAX": gzip.decompress(
            (VENDOR / "mathjax-3.2.2-tex-svg.js.gz").read_bytes()
        ).decode("utf-8"),
        "DATA": packed(evidence),
        "ARCHIVE": archive,
        "APP": (ASSETS / "app.js").read_text(encoding="utf-8"),
    }
    for name, value in replacements.items():
        template = template.replace("@@" + name + "@@", value)
    output = ROOT / "REPORT.html"
    output.write_text(template, encoding="utf-8")
    manifest = dict(
        report="REPORT.html",
        sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        runs=len(evidence["runs"]),
        cases=len(evidence["definition"]["cases"]),
        inputs=fingerprints,
        generated_at=evidence["generated_at"],
    )
    (ROOT / "report_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"Wrote {output}: {len(evidence['runs'])} retained attempts; {output.stat().st_size:,} bytes"
    )


if __name__ == "__main__":
    main()
