"""Build the standalone Plotly benchmark workbench from saved campaign data.

This module imports no numerical solver and never advances a flow solution.
Run from the repository root with:
    python -B -m development.roberto.benchmarking.scripts.report
"""

from __future__ import annotations

import argparse
import base64
import gzip
from hashlib import sha256
from html import escape
from html.parser import HTMLParser
from io import BytesIO
import json
import math
from pathlib import Path
import re
import zipfile


STUDY = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / "report_assets"
PLOTLY_SHA256 = "6d21266ce1bd7d9e5ab4e115989c70c20de0382fd973a8f26ab58619eba4d603"


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json(value):
    """JSON that is valid in a script element, including untrusted descriptions."""
    return (
        json.dumps(_clean(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _script_safe(value):
    return re.sub(r"</script", r"<\/script", value, flags=re.IGNORECASE)


def _download_uri(mime, raw):
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


def _share_references(campaign):
    """Losslessly share identical plotting references; original ZIP stays exact."""
    if campaign is None or not isinstance(campaign.get("results"),list):return campaign
    previous=campaign.get("presentation_references",{});pool={};rows=[]
    for source in campaign["results"]:
        row=dict(source);reference=row.get("reference")
        if isinstance(reference,dict) and "shared_reference_sha256" in reference:
            reference=previous[reference["shared_reference_sha256"]]
        if reference:
            digest=sha256(_json(reference).encode("utf-8")).hexdigest()
            pool[digest]=reference;row["reference"]={"shared_reference_sha256":digest}
        rows.append(row)
    return dict(campaign,results=rows,presentation_references=pool)


def read_campaign(path: Path):
    """Read result JSONs and verified declared artifacts for exact-byte export.

    Only explicitly referenced files inside the campaign directory are read.
    Artifact verification failures remain visible in presentation warnings and
    the artifact inventory; unverified artifact bytes are never embedded.
    """
    path = path.resolve()
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"Campaign must be a JSON object: {path}")
    originals = {path.name: raw}
    warnings = list(data.get("presentation_warnings", []))
    numerical_artifacts = []
    for collection in ("results", "records", "scenarios"):
        rows = data.get(collection)
        if not isinstance(rows, list):
            continue
        materialized = []
        for row in rows:
            row = {"file": row} if isinstance(row, str) else row
            if not isinstance(row, dict):
                raise ValueError(f"Invalid result entry in {path}")
            filename = row.get("file")
            if not isinstance(filename, str):
                materialized.append(row)
                continue
            linked = (path.parent / filename).resolve()
            if not linked.is_relative_to(path.parent) or linked.suffix.lower() != ".json":
                warnings.append(f"Result reference was not read outside the campaign directory: {filename}")
                materialized.append({**row, "presentation_missing_data": True})
                continue
            if not linked.is_file():
                warnings.append(f"Referenced result data are missing: {filename}")
                materialized.append({**row, "presentation_missing_data": True})
                continue
            blob = linked.read_bytes()
            detail = json.loads(blob)
            if not isinstance(detail, dict):
                raise ValueError(f"Result must be a JSON object: {linked}")
            # Summary labels/status retain precedence; detailed fields/histories are added.
            materialized.append({**detail, **row})
            originals[linked.relative_to(path.parent).as_posix()] = blob
            if isinstance(detail.get("raw_result_file"), str):
                numerical_artifacts.append(dict(file=detail["raw_result_file"],sha256=detail.get("raw_result_sha256"),
                                                role="Original untimed numerical solve; elapsed values are diagnostics, not performance samples"))
        data[collection] = materialized
    artifacts = data.get("artifacts", [])
    if not isinstance(artifacts, list):
        warnings.append("Campaign artifact declarations must be a list; no artifacts were read.")
        artifacts = []
    artifacts = list(artifacts) + numerical_artifacts
    inventory = []
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("file"), str):
            warnings.append("Invalid campaign artifact declaration; expected file and SHA256 fields.")
            inventory.append({"declaration": artifact, "status": "invalid declaration"})
            continue
        filename = artifact["file"]
        item = {**artifact, "status": "unverified"}
        inventory.append(item)
        linked = (path.parent / filename).resolve()
        if not filename or not linked.is_relative_to(path.parent):
            item["status"] = "unsafe path"
            warnings.append(f"Campaign artifact was not read outside the campaign directory: {filename}")
            continue
        expected_hash = artifact.get("sha256")
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
            item["status"] = "invalid checksum declaration"
            warnings.append(f"Campaign artifact was not embedded without a valid declared SHA256: {filename}")
            continue
        if not linked.is_file():
            item["status"] = "missing"
            warnings.append(f"Declared campaign artifact is missing: {filename}")
            continue
        blob = linked.read_bytes()
        actual_hash = sha256(blob).hexdigest()
        item.update(actual_sha256=actual_hash, bytes=len(blob))
        if actual_hash != expected_hash.lower():
            item["status"] = "checksum mismatch"
            warnings.append(f"Campaign artifact SHA256 does not match its declaration; bytes were not embedded: {filename}")
            continue
        archive_name = linked.relative_to(path.parent).as_posix()
        originals[archive_name] = blob
        item.update(status="embedded; SHA256 verified", archive_entry=archive_name)
    if inventory:
        data["presentation_artifacts"] = inventory
    if warnings:
        data["presentation_warnings"] = warnings
    return data, originals, sha256(raw).hexdigest()


def _evidence_zip(campaign_files, baseline_files):
    memory = BytesIO()
    with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for prefix, files in (("campaign", campaign_files), ("baseline", baseline_files)):
            for name, raw in sorted(files.items()):
                entry = zipfile.ZipInfo(f"{prefix}/{name}", date_time=(1980, 1, 1, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(entry, raw)
    return memory.getvalue()


def _roe_validation(audit, campaign, raw):
    """Small presentation payload; the exact full audit remains in the ZIP."""
    if not isinstance(audit, dict) or not isinstance(audit.get("summary"), dict):
        raise ValueError("Roe audit must contain its recorded summary")
    summary = audit["summary"]
    source_files = campaign.get("provenance", {}).get("source", {}).get("files", {})
    checked, mismatches, unavailable = [], [], []
    for filename, digest in audit.get("source", {}).get("fingerprints_sha256", {}).items():
        filename = "/" + filename.replace("\\", "/").lstrip("/")
        marker = "/src/nozzle_hyperbolic/"
        if marker not in filename:
            continue
        relative = "src/nozzle_hyperbolic/" + filename.split(marker, 1)[-1]
        expected = source_files.get(relative)
        if expected is None:
            unavailable.append(relative)
        elif expected != digest:
            mismatches.append(relative)
        else:
            checked.append(relative)
    compact = {key: value for key, value in audit.items()
               if key not in ("endpoints", "pair_draws", "pair_audits")}
    compact["pair_audits"] = [{key: value for key, value in item.items()
                               if key not in ("records", "failures")}
                              for item in audit.get("pair_audits", [])]
    compact["evidence"] = dict(archive_entry="campaign/validation/roe_audit.json",
                              sha256=sha256(raw).hexdigest(), bytes=len(raw))
    compact["campaign_source_comparison"] = dict(matched=checked, mismatches=mismatches,
                                                  unavailable=unavailable,
                                                  scope="Recorded production source files; backend/audit-script fingerprints remain available in the audit provenance",
                                                  verified=bool(checked) and not mismatches and not unavailable)
    compact["algebra_checks_passed"] = bool(
        summary.get("pair_checks_passed", 0) > 0
        and summary.get("pair_checks_failed") == 0
        and summary.get("invalid_roe_pairs") == 0
        and summary.get("property_failure_count") == 0
        and summary.get("tiny_jump_valid", 0) > 0
        and summary.get("tiny_jump_valid") == summary.get("tiny_jump_total")
        and not audit.get("source", {}).get("files_changed_during_audit"))
    return compact


def _flow_inputs(sod_path=None, real_gas_path=None, figure_path=None):
    """Read only explicitly selected saved flow evidence and hashed artifacts."""
    files,flow={},{}
    if sod_path is not None:
        raw=Path(sod_path).read_bytes();data=json.loads(raw)
        files["validation/roe_sod.json"]=raw
        flow["sod"]={key:value for key,value in data.items() if key!="records"}
        excluded={"final_state","exact_final_conservative_cell_averages","physical_cell_centers","history","output_times"}
        flow["sod"]["records"]=[{key:value for key,value in row.items() if key not in excluded} for row in data.get("records",[])]
    if real_gas_path is not None:
        real_gas_path=Path(real_gas_path).resolve();raw=real_gas_path.read_bytes();data=json.loads(raw)
        if not isinstance(data,list):raise ValueError("Real-gas flow summary must contain its saved attempt array")
        files["validation/roe_real_gas_summary.json"]=raw;flow["real_gas"]=data
        base=real_gas_path.parent
        for attempt in data:
            for collection in ("raw_json_sha256","saved_state_sha256"):
                for filename,digest in attempt.get(collection,{}).items():
                    relative=Path(filename)
                    artifact=(base/Path(*relative.parts[1:]) if relative.parts and relative.parts[0]=="outputs" else base/relative).resolve()
                    if not artifact.is_relative_to(base) or artifact.suffix not in (".json",".npz"):
                        raise ValueError("Unsafe real-gas evidence path: "+filename)
                    artifact_raw=artifact.read_bytes()
                    if sha256(artifact_raw).hexdigest()!=digest:raise ValueError("Real-gas evidence hash differs: "+filename)
                    files["validation/real_gas/"+artifact.relative_to(base).as_posix()]=artifact_raw
    if figure_path is not None:
        if sod_path is None:raise ValueError("--roe-sod-figure requires --roe-sod")
        raw=Path(figure_path).read_bytes()
        if not raw.startswith(b"\x89PNG\r\n\x1a\n"):raise ValueError("Saved Sod figure must be a PNG")
        files["validation/sod_final_profiles.png"]=raw
        flow["sod_figure"]=_download_uri("image/png",raw)
    if files:
        flow["evidence"]=[dict(archive_entry="campaign/"+name,sha256=sha256(raw).hexdigest(),bytes=len(raw)) for name,raw in sorted(files.items())]
    return (flow or None),files


def render(campaign, *, baseline=None, original_files=None, baseline_files=None,
           input_name="summary.json", input_sha256=None, baseline_sha256=None,
           regression_threshold=1.15, title="Steady-flow benchmark suite",
           roe_audit=None, roe_audit_raw=None, flow_validation=None, flow_original_files=None):
    """Render all report assets inline; callers can use in-memory campaign data."""
    if campaign.get("verification_only") is True and title=="Steady-flow benchmark suite":
        title="Roe steady verification" if campaign.get("protocol",{}).get("fluxes")==["roe"] else "Untimed steady verification"
    plotly_raw = gzip.decompress((ASSETS / "vendor/plotly-2.35.2.min.js.gz").read_bytes())
    if sha256(plotly_raw).hexdigest() != PLOTLY_SHA256:
        raise ValueError("Vendored Plotly runtime failed its integrity check")
    license_raw = (ASSETS / "vendor/PLOTLY_LICENSE.txt").read_bytes()
    source_bytes = json.dumps(_clean(campaign), indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
    original_files = dict(original_files or {input_name: source_bytes})
    for filename,raw in (flow_original_files or {}).items():
        if filename in original_files and original_files[filename]!=raw:raise ValueError("Conflicting flow evidence: "+filename)
        original_files[filename]=raw
    if flow_validation is not None:
        flow_validation=dict(flow_validation)
        expected=campaign.get("provenance",{}).get("source",{}).get("files",{})
        comparisons=[]
        def compare_flow_source(label,files):
            relevant={}
            for name,digest in files.items():
                normalized="/"+name.replace("\\","/").lstrip("/");marker="/src/nozzle_hyperbolic/"
                if marker in normalized:relevant["src/nozzle_hyperbolic/"+normalized.split(marker,1)[1]]=digest
            unavailable=[name for name in relevant if name not in expected]
            mismatches=[name for name,digest in relevant.items() if name in expected and expected[name]!=digest]
            return dict(evidence=label,production_files_checked=len(relevant),matched_files=len(relevant)-len(unavailable)-len(mismatches),
                        unavailable=unavailable,unavailable_count=len(unavailable),mismatches=mismatches,mismatch_count=len(mismatches),
                        matched=bool(relevant) and not mismatches and not unavailable,
                        scope="Recorded production file fingerprints; aggregate hashes from different manifest formats are not compared")
        for filename,raw in (flow_original_files or {}).items():
            if filename.endswith("/source_manifest.json"):
                files=json.loads(raw).get("files",{})
                comparisons.append(compare_flow_source("campaign/"+filename,files))
        if flow_validation.get("sod") is not None:
            comparisons.append(compare_flow_source("campaign/validation/roe_sod.json",flow_validation["sod"].get("source",{}).get("fingerprints_sha256",{})))
        flow_validation["campaign_source_comparisons"]=comparisons
    validation = None
    if roe_audit is not None:
        audit_raw = roe_audit_raw or json.dumps(_clean(roe_audit), indent=2, allow_nan=False).encode("utf-8")
        if json.loads(audit_raw) != roe_audit:
            raise ValueError("Roe audit payload differs from its original JSON bytes")
        filename = "validation/roe_audit.json"
        if filename in original_files and original_files[filename] != audit_raw:
            raise ValueError("Roe validation evidence conflicts with an existing campaign artifact")
        original_files[filename] = audit_raw
        validation = _roe_validation(roe_audit, campaign, audit_raw)
    evidence = _evidence_zip(original_files, baseline_files or {})
    payload = {
        "campaign": _share_references(campaign),
        "baseline": _share_references(baseline),
        "roe_validation": validation,
        "roe_flow_validation": flow_validation,
        "report": {
            "schema_version": 1,
            "input_name": input_name,
            "input_sha256": input_sha256 or sha256(source_bytes).hexdigest(),
            "baseline_sha256": baseline_sha256,
            "regression_threshold": regression_threshold,
            "plotly_version": "2.35.2",
            "plotly_sha256": PLOTLY_SHA256,
            "evidence_sha256": sha256(evidence).hexdigest(),
            "original_files": sorted(original_files),
            "baseline_files": sorted(baseline_files or {}),
            "original_file_sha256": {name: sha256(raw).hexdigest() for name, raw in original_files.items()},
            "baseline_file_sha256": {name: sha256(raw).hexdigest() for name, raw in (baseline_files or {}).items()},
        },
    }
    template = (ASSETS / "template.html").read_text(encoding="utf-8")
    values = {
        "__TITLE__": escape(title),
        "__ROE_THEORY__": (ASSETS / "roe_theory.html").read_text(encoding="utf-8"),
        "/* __STYLE__ */": (ASSETS / "style.css").read_text(encoding="utf-8"),
        "/* __PLOTLY__ */": _script_safe(plotly_raw.decode("utf-8")),
        "/* __APP__ */": _script_safe((ASSETS / "app.js").read_text(encoding="utf-8")),
        "__DATA__": _json(payload),
        "__ORIGINAL_DOWNLOAD__": _download_uri("application/zip", evidence),
        "__ORIGINAL_FILENAME__": "benchmark_evidence.zip",
        "__LICENSE_DOWNLOAD__": _download_uri("text/plain", license_raw),
    }
    for marker, value in values.items():
        template = template.replace(marker, value)
    # The archive contains original bytes, not a reserialized approximation.
    template = template.replace("Original campaign · JSON.gz", "Original campaign evidence · ZIP")
    return template


class _ReportParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.external = []
        self.downloads = []
        self.in_data = False
        self.data_text = ""

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "script" and "src" in attrs:
            self.external.append(attrs["src"])
        if tag == "link" and attrs.get("rel") in ("stylesheet", "preload"):
            self.external.append(attrs.get("href"))
        if tag == "a" and "download" in attrs:
            self.downloads.append(attrs)
        if tag == "script" and attrs.get("id") == "benchmark-data":
            self.in_data = True

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_data = False

    def handle_data(self, data):
        if self.in_data:
            self.data_text += data


def audit_html(html):
    """Check packaging and original evidence without executing numerical code."""
    parser = _ReportParser()
    parser.feed(html)
    payload = json.loads(parser.data_text)
    for campaign in (payload.get("campaign"),payload.get("baseline")):
        if campaign is None:continue
        pool=campaign.get("presentation_references",{})
        for digest,reference in pool.items():
            assert sha256(_json(reference).encode("utf-8")).hexdigest()==digest
        for row in campaign.get("results",[]):
            reference=row.get("reference",{})
            if isinstance(reference,dict) and "shared_reference_sha256" in reference:assert reference["shared_reference_sha256"] in pool
    evidence_link = next(a for a in parser.downloads if a["download"] == "benchmark_evidence.zip")
    evidence = base64.b64decode(evidence_link["href"].split(",", 1)[1])
    with zipfile.ZipFile(BytesIO(evidence)) as archive:
        assert archive.testzip() is None
        expected = ["campaign/" + p for p in payload["report"]["original_files"]]
        expected += ["baseline/" + p for p in payload["report"]["baseline_files"]]
        assert sorted(archive.namelist()) == sorted(expected)
        for prefix, key in (("campaign", "original_file_sha256"), ("baseline", "baseline_file_sha256")):
            for name, digest in payload["report"].get(key, {}).items():
                assert sha256(archive.read(f"{prefix}/{name}")).hexdigest() == digest
        validation = payload.get("roe_validation")
        if validation is not None:
            audit_raw = archive.read(validation["evidence"]["archive_entry"])
            assert sha256(audit_raw).hexdigest() == validation["evidence"]["sha256"]
            assert json.loads(audit_raw)["summary"] == validation["summary"]
        flow=payload.get("roe_flow_validation")
        if flow is not None:
            for artifact in flow.get("evidence",[]):
                assert sha256(archive.read(artifact["archive_entry"])).hexdigest()==artifact["sha256"]
            if flow.get("sod") is not None:
                raw_sod=json.loads(archive.read("campaign/validation/roe_sod.json"))
                assert raw_sod.get("assessment")==flow["sod"].get("assessment")
            if flow.get("real_gas") is not None:
                assert json.loads(archive.read("campaign/validation/roe_real_gas_summary.json"))==flow["real_gas"]
    ids_unique = len(parser.ids) == len(set(parser.ids))
    required = {"page-summary", "page-case", "page-protocol", "page-theory", "roe-theory-secant", "roe-theory-waves", "field-chart", "residual-chart", "cfl-chart", "refinement-chart", "samples-chart", "regression-chart"}
    if payload.get("roe_validation") is not None:
        required.update(("roe-validation", "roe-validation-table", "roe-validation-json", "roe-audit-download"))
    if payload.get("roe_flow_validation") is not None:
        required.update(("roe-flow-validation","roe-flow-json","roe-flow-download"))
    result = {
        "passed": not parser.external and ids_unique and required.issubset(parser.ids),
        "external_dependencies": parser.external,
        "unique_html_ids": ids_unique,
        "report_data_valid": True,
        "evidence_archive_sha256": sha256(evidence).hexdigest(),
        "evidence_archive_entries": sorted(expected),
        "html_bytes": len(html.encode("utf-8")),
    }
    assert result["passed"], result
    assert result["evidence_archive_sha256"] == payload["report"]["evidence_sha256"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=STUDY / "outputs/initial_baseline/summary.json")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--roe-audit", type=Path, help="Saved Roe property/matrix audit JSON; embedded verbatim in the evidence ZIP")
    parser.add_argument("--roe-sod",type=Path,help="Saved Roe Sod JSON; no transient solve is run")
    parser.add_argument("--roe-real-gas",type=Path,help="Saved actual-EOS attempt summary JSON; declared raw artifacts are verified and embedded")
    parser.add_argument("--roe-sod-figure",type=Path,help="Optional saved Sod PNG, embedded inline and in the evidence ZIP")
    parser.add_argument("--output", type=Path, default=STUDY / "REPORT.html")
    parser.add_argument("--regression-threshold", type=float, default=1.15)
    args = parser.parse_args()
    if not math.isfinite(args.regression_threshold) or args.regression_threshold <= 1:
        parser.error("--regression-threshold must be finite and greater than one")
    campaign, original, source_hash = read_campaign(args.input)
    baseline, baseline_files, baseline_hash = (None, {}, None)
    if args.baseline:
        baseline, baseline_files, baseline_hash = read_campaign(args.baseline)
    roe_raw = args.roe_audit.read_bytes() if args.roe_audit else None
    roe_audit = json.loads(roe_raw) if roe_raw is not None else None
    flow,flow_files=_flow_inputs(args.roe_sod,args.roe_real_gas,args.roe_sod_figure)
    html = render(campaign, baseline=baseline, original_files=original,
                  baseline_files=baseline_files, input_name=args.input.name,
                  input_sha256=source_hash, baseline_sha256=baseline_hash,
                  regression_threshold=args.regression_threshold,
                  roe_audit=roe_audit, roe_audit_raw=roe_raw,
                  flow_validation=flow,flow_original_files=flow_files)
    audit = audit_html(html)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8", newline="\n")
    audit_path = args.input.parent / "report_structure_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({audit['html_bytes']:,} bytes); offline structure/evidence audit passed.")


if __name__ == "__main__":
    main()
