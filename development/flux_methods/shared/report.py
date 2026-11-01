"""Render independent flux studies from saved JSON only; never import a solver."""
from __future__ import annotations

import argparse
import base64
import gzip
from importlib import import_module
from hashlib import sha256
from html import escape
from io import BytesIO
import json
import math
from pathlib import Path
import re
import zipfile

from .presentation import theme_css, theme_script, render_document, math_runtime

SHARED = Path(__file__).resolve().parent
STUDIES = SHARED.parent
ASSETS = SHARED / "assets"
CASES = ("euler_annular/shock35", "euler_annular/shock70",
         "euler_nozzle/shock_35", "euler_nozzle/shock_70")
FIELDS = {"p": ("Pressure", "Pa"), "velocity": ("Normal velocity", "m/s"),
          "T": ("Temperature", "K"), "h0": ("Total enthalpy", "J/kg")}


def report_profile(study):
    path=STUDIES/study/"report_config.json"
    if not path.is_file():
        raise ValueError(f"No report configuration for {study}")
    result=json.loads(path.read_text(encoding="utf-8"))
    if result.get("study")!=study or not result.get("configurations"):
        raise ValueError("Invalid study report configuration")
    return result


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dumps(value):
    return json.dumps(clean(value), ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")


def compact(value):
    """Preserve checks and short diagnostic records; large tensors stay on disk."""
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) <= 16:
            return [compact(v) for v in value]
        shape, probe = [], value
        while isinstance(probe, list):
            shape.append(len(probe))
            probe = probe[0] if probe else None
        return {"full_array_preserved_in_outputs": True, "shape": shape}
    return value


def flatten(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            result.update(flatten(item, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result[prefix] = value
    return result


def fields(value):
    value = value.get("fields", value) if isinstance(value, dict) else {}
    result = {k: v for k, v in value.items() if isinstance(v, list)
              and all(x is None or isinstance(x, (int, float)) for x in v)}
    if "velocity" not in result:
        for key in ("v_m", "u"):
            if key in result:
                result["velocity"] = result[key]
                break
    return result


def profile(value, fallback_x=()):
    value = fields(value)
    x = value.get("coordinate", value.get("x", list(fallback_x)))
    return {"x": x, "fields": {key: value[key] for key in FIELDS if key in value}}


def relative_file(path, base):
    path = path.resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError(f"Evidence path leaves its campaign: {path}")
    return path.relative_to(base).as_posix()


def read_campaign(study, summary_path):
    """Read the manifest population, including failed/incomplete saved attempts."""
    profile = report_profile(study)
    specifications = {item["id"]:item for item in profile["configurations"]}
    summary_path = summary_path.resolve()
    campaign = summary_path.parent
    originals = {"summary.json": summary_path.read_bytes()}
    summary = json.loads(originals["summary.json"])
    audits = {}
    for name in ("settings.json", "fresh_audit.json", "audit.json", "source_audit.json"):
        path = campaign / name
        if path.is_file():
            originals[name] = path.read_bytes()
            audits[name] = json.loads(originals[name])
    audit_rows = {}
    for name, audit in audits.items():
        if name not in ("fresh_audit.json", "audit.json"):
            continue
        for row in audit.get("records", []):
            file = str(row.get("file", ""))
            if file:
                audit_rows[file.replace("\\", "/")] = row
                audit_rows.setdefault(Path(file).name, row)
    records = []
    for entry in summary.get("records", []):
        entry = {"file": entry} if isinstance(entry, str) else entry
        path = (campaign / entry["file"]).resolve()
        name = relative_file(path, campaign)
        if not path.is_file():
            records.append({"id": entry.get("id", name), "source_file": name,
                            "case": entry.get("case"), "cells": entry.get("cells"),
                            "reconstruction": entry.get("reconstruction"), "variant": entry.get("variant"),
                            "success": False, "reason": "missing_saved_row", "missing": True})
            continue
        blob = path.read_bytes()
        digest = sha256(blob).hexdigest()
        if entry.get("sha256") and entry["sha256"] != digest:
            raise ValueError(f"Summary SHA-256 differs from saved row: {name}")
        originals[name] = blob
        row = json.loads(blob)
        variant = row.get("variant", row.get("method", row.get("flux")))
        case = row.get("case", row.get("case_id"))
        if case not in CASES or variant not in specifications:
            raise ValueError(f"Unexpected case/variant in {name}: {case}, {variant}")
        expected_flux=profile["public_flux"]
        if row.get("flux")!=expected_flux or row.get("scheme",{}).get("flux",expected_flux)!=expected_flux:
            raise ValueError(f"Unexpected production flux ID in {name}")
        if profile.get("require_method_is_flux") and row.get("method")!=expected_flux:
            raise ValueError(f"Method metadata is not the public flux in {name}")
        for key,expected in specifications[variant]["parameters"].items():
            actual=row.get(key,row.get("scheme",{}).get(key))
            if type(actual) is not type(expected) or actual!=expected:
                raise ValueError(f"Configuration parameter {key} disagrees in {name}")
        audit = audit_rows.get(name, audit_rows.get(path.name))
        stale = bool(audit and audit.get("file_sha256", audit.get("sha256")) != digest)
        records.append(dict(row, case=case, variant=variant, source_file=name, file_sha256=digest,
                            fresh_audit=audit if audit and not stale else None, stale_audit=stale))
    documents = {}
    for name in ("README.md", "THEORY.md"):
        path = STUDIES / study / name
        if path.is_file():
            originals[f"documents/{name}"] = path.read_bytes()
            documents[name] = path.read_text(encoding="utf-8-sig")
    return summary, records, audits, originals, documents


def normalize(row, report_config):
    field = profile(row.get("fields", {}), row.get("x", ()))
    point = profile(row.get("reference_point", {}), field["x"])
    cell = profile(row.get("reference_cell", {}), field["x"])
    history = [{key: item.get(key) for key in ("step", "phase", "cfl", "overall_relative", "inner_iterations")}
               for item in row.get("history", [])
               if item.get("accepted") or item.get("phase") in ("initial", "accepted")]
    work = dict(row.get("work", {}))
    work.setdefault("accepted_steps", sum(h.get("phase") == "accepted" for h in row.get("history", [])))
    work.setdefault("rejected_attempts", sum(h.get("phase") == "rejected" for h in row.get("history", [])))
    work.setdefault("newton_corrections", sum(h.get("inner_iterations", 0) for h in row.get("history", [])))
    errors = {**flatten(row.get("errors", {}), "errors"),
              **flatten(row.get("conservative_errors", {}), "conservative_errors")}
    conserved = row.get("conservative_errors", {})
    names = conserved.get("equation_names", [])
    for key in ("l1", "normalized_l1", "linf"):
        if isinstance(conserved.get(key), list) and len(conserved[key]) == len(names):
            errors.update({f"conservative_errors.{name}.{key}": value
                           for name, value in zip(names, conserved[key])})
    shape = flatten(row.get("profile_metrics", {}), "profile_metrics")
    controls = {k: v for k, v in row.get("scheme", {}).items()
                if k not in {"flux", *(key for item in report_config["configurations"] for key in item["parameters"])}}
    common = {"source": row.get("source_sha256"), "case": row.get("case_sha256"),
              "scheme": controls, "settings": row.get("settings"), "initial": row.get("initial_state")}
    return dict(id=row.get("id", row["source_file"]), case=row.get("case"), cells=row.get("cells"),
                reconstruction=row.get("reconstruction"), variant=row.get("variant"), flux=row.get("flux"),
                parameters={key:row.get(key,row.get("scheme",{}).get(key)) for item in report_config["configurations"] for key in item["parameters"]},
                initialization=row.get("initialization", "cold"), alignment=row.get("alignment", 0),
                success=bool(row.get("success")), reason=row.get("reason", "unknown"),
                error=compact(row.get("error") or {"exception": row.get("exception"), "assessment_error": row.get("assessment_error")}),
                profile=field, reference_point=point, reference_cell=cell, history=history, work=work,
                shock_position=row.get("shock_position"), errors=errors, shape=shape,
                metrics=compact(row.get("metrics", {})), profile_metrics=compact(row.get("profile_metrics", {})),
                fresh_audit=compact(row.get("fresh_audit")), stale_audit=bool(row.get("stale_audit")),
                independent_check=compact(row.get("independent_check", {})), scheme=row.get("scheme", {}),
                settings=row.get("settings", {}), source_sha256=row.get("source_sha256"),
                implementation_sha256=row.get("implementation_sha256"), case_sha256=row.get("case_sha256"),
                record_type=row.get("record_type", "native_solve"),
                audit_required_true=required_audit_flags(row, report_config),
                origin=compact(row.get("origin", row.get("api_origin", row.get("migration", {})))),
                recorded_source_sha256=row.get("recorded_source_sha256",row.get("source_sha256")),
                evaluation_source_sha256=row.get("evaluation_source_sha256"),
                comparison_identity=sha256(dumps(common).encode()).hexdigest(),
                source_file=row["source_file"], file_sha256=row.get("file_sha256"),
                elapsed_with_compile_s=row.get("elapsed_with_compile_s"), missing=bool(row.get("missing")))


def archive(files):
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for name, blob in sorted(files.items()):
            output.writestr(name, blob)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def required_audit_flags(row, profile):
    kind = row.get("record_type", "native_solve")
    return list(dict.fromkeys([*profile["audit_required_true"],
                              *profile.get("audit_required_by_record_type", {}).get(kind, [])]))


def valid_audit(row, audits, profile):
    audit = row.get("fresh_audit") or {}
    top = audits.get("fresh_audit.json", audits.get("audit.json", {}))
    return bool(audit) and not row.get("stale_audit") and not audit.get("error") and audit.get("all_checks_passed") is True and all(
        audit.get(key) is True for key in required_audit_flags(row, profile)
    ) and all(top.get(key) is not False for key in (
        "snapshot_source_match", "snapshot_client_match", "snapshot_case_match",
        "evaluation_epoch_stable", "audit_valid"))


def build(study, summary_path):
    summary, rows, audits, originals, documents = read_campaign(study, summary_path)
    profile=report_profile(study)
    if summary.get("record_type") != "api_migration_view":
        profile.update(profile.get("native_overrides", {}))
    ids=[item["id"] for item in profile["configurations"]]
    records = [normalize(row,profile) for row in rows]
    supplemental = {}
    entries=summary.get("public_supplemental_evidence",profile.get("supplemental_evidence",[]))
    for entry in entries:
        entry={"file":entry} if isinstance(entry,str) else entry
        path=(summary_path.resolve().parent/entry["file"]).resolve()
        if not path.is_relative_to(STUDIES.resolve()):
            raise ValueError("Supplemental evidence leaves the study family")
        blob=path.read_bytes()
        digest=sha256(blob).hexdigest()
        if entry.get("sha256") and entry["sha256"]!=digest:
            raise ValueError("Supplemental evidence SHA mismatch")
        name="supplemental/"+path.name
        originals[name]=blob
        supplemental[name]=dict(source=entry["file"],sha256=digest,bytes=len(blob),scope=entry.get("scope","Supplemental derived-data or evaluation evidence; separate from row checks"),summary=compact(json.loads(blob)))
    payload = dict(study=study, title=profile["title"],
                   cases=list(CASES), variants=ids, labels={item["id"]:item["label"] for item in profile["configurations"]}, colors={item["id"]:item["color"] for item in profile["configurations"]}, fields=FIELDS, public_flux=profile["public_flux"], report_profile=profile, audit_required_true=profile["audit_required_true"],
                   records=records, summary=compact(summary), audits=compact(audits), documents=documents,
                   complete=bool(summary.get("completed", summary.get("final", False))),
                   supplemental_evidence=supplemental,
                   evidence_manifest=[{"path": name, "bytes": len(blob), "sha256": sha256(blob).hexdigest()}
                                      for name, blob in sorted(originals.items())])
    notes=import_module(f"development.flux_methods.{study}.{profile['notes_module']}")
    top_audit=audits.get("fresh_audit.json",audits.get("audit.json",{}))
    payload.update(notes.make_notes(records,audits,originals,profile,{
        "valid_audit":lambda row,checks:valid_audit(row,checks,profile),
        "audit_complete":top_audit.get("complete") is True}))
    if profile.get("public_evidence_only"):
        for name,blob in originals.items():
            text=blob.decode("utf-8-sig")
            if re.search(r"osher(?:[359]|_sonic)",text,re.I):
                raise ValueError(f"Retired public API identifiers are archive-only: {name}")
        if re.search(r"osher(?:[359]|_sonic)",dumps(payload),re.I):
            raise ValueError("Retired API identifiers reached the public report payload")
    download = {"plot_data.json": dumps(payload).encode("utf-8"),
                "original_evidence_manifest.json": dumps(payload["evidence_manifest"]).encode("utf-8"),
                "audit_summaries.json": dumps(payload["audits"]).encode("utf-8")}
    download.update({name: blob for name, blob in originals.items() if name.startswith(("documents/", "supplemental/"))})
    plotly_license=(ASSETS/"vendor/PLOTLY_LICENSE.txt").read_bytes()
    download["licenses/PLOTLY_LICENSE.txt"]=plotly_license
    download["licenses/MATHJAX_LICENSE.txt"]=(ASSETS/"vendor/MATHJAX_LICENSE.txt").read_bytes()
    theory=render_document(documents.get("THEORY.md",""),prefix=study+"-theory")
    readme=render_document(documents.get("README.md",""),prefix=study+"-protocol")
    index="".join(f'<a class="level-{item["level"]}" href="#{escape(item["id"],quote=True)}">{escape(item["text"])}</a>'
                  for item in theory["headings"] if item["level"] in (2,3))
    values = {"TITLE":escape(profile["title"]),"THEME_CSS":theme_css(),"THEME_SCRIPT":theme_script(),
              "MATH_RUNTIME":math_runtime(),"THEORY_HTML":theory["html"],"THEORY_TOC":index,"README_HTML":readme["html"],
              "LICENSE_DOWNLOAD":"data:text/plain;charset=utf-8;base64,"+base64.b64encode(plotly_license).decode("ascii"),
              "CSS": (ASSETS / "report.css").read_text(encoding="utf-8"), "PAYLOAD": dumps(payload),
              "PLOTLY": gzip.decompress((ASSETS / "vendor/plotly-2.35.2.min.js.gz").read_bytes()).decode("utf-8"),
              "APPLICATION": (ASSETS / "report_app.js").read_text(encoding="utf-8"), "ARCHIVE": archive(download)}
    template = (ASSETS / "report_template.html").read_text(encoding="utf-8")
    html = re.sub(r"\{\{([A-Z_]+)\}\}", lambda m: values[m[1]], template)
    checks = {"one_case_dropdown": html.count('<select id="case">') == 1,
              "no_static_figures": not re.search(r"<img\b|<figure\b", template, re.I),
              "offline": not re.search(r"<(?:script|link)[^>]+(?:src|href)\s*=", template, re.I),
              "four_cases": set(payload["cases"]) == set(CASES),
              "valid_row_scope": all(r["case"] in CASES and r["variant"] in ids for r in records),
              "no_stale_audits": not any(r["stale_audit"] for r in records),
              "common_theme":'id="report-theme-css"' in html and 'id="report-theme-runtime"' in html,
              "rendered_theory":bool(theory["headings"]) and 'id="mathjax-runtime"' in html and 'class="math-display"' in theory["html"],
              "accessible_navigation":template.count('role="tablist"')==2 and all('>'+label+'</button>' in template for label in ("Overview","Case explorer","Theory","Protocol &amp; evidence")),
              "supplemental_originals_sha_bound": all(sha256(download[name]).hexdigest() == item["sha256"]
                                                       for name, item in supplemental.items()),
              "archive_exact": True}
    encoded = re.search(r"data:application/zip;base64,([^\"]+)", html)[1]
    with zipfile.ZipFile(BytesIO(base64.b64decode(encoded))) as saved:
        checks["archive_exact"] = set(saved.namelist()) == set(download) and all(saved.read(n) == b for n, b in download.items())
    result = {"passed": all(checks.values()), "checks": checks, "record_count": len(records),
              "case_count": len(CASES), "browser_rendering_verified": False,
              "scope": "Offline structure, exact archive and source-row SHA checks; no solver imported."}
    return html, payload, result


def markdown(payload):
    records = payload["records"]
    lines = [f"# {payload['title']}: saved verification", "", "Open [REPORT.html](REPORT.html) for the offline Plotly case explorer.", "",
             f"Retained: {len(records)} solution records; {sum(r['success'] for r in records)} recorded successes; "
             f"{sum(bool(r['fresh_audit']) for r in records)} hash-joined evaluation audits; "
             f"{sum(valid_audit(r,payload['audits'],payload['report_profile']) for r in records)} with passing physical/identity checks. "
             f"Saved population completed flag: {payload['complete']}; "
             f"evaluation population complete: {payload['audits'].get('fresh_audit.json',payload['audits'].get('audit.json',{})).get('complete') is True}.", "",
             payload["report_profile"].get("evidence_scope","Recorded execution identities and evaluation identities remain distinct; presentation performs no numerical audit."), "",
             "The four requested nozzle/annular shock cases, both reconstructions and every retained configuration remain explicit selections. Missing or unsuccessful runs remain visible. Coverage, original-equation convergence and analytical accuracy are separate checks.", "",
             "Refinement uses saved errors against analytical conservative cell projections; point-reference comparisons remain separate. See [THEORY.md](THEORY.md) for the common flux and its explicit parameters.", "",
             ("Listed supplemental production-test/API checks and matrix postprocessing are separate from fresh row audits. Their compact summaries and exact original JSON bytes are embedded with filename and SHA-256 bindings."
              if payload.get("supplemental_evidence") else
              "This report embeds current saved-row Operator evaluations. No separate production-test or matrix-postprocessing supplement is included."), "",
             "This is numerical verification, not a performance benchmark. Compilation-inclusive elapsed diagnostics do not establish a cost ranking. Residual/CFL plots show actual accepted histories; accepted steps, rejected attempts and all Newton corrections are separate work measures.", "",
             "The HTML embeds all plotted values, rendered mathematical notes, clean audit summaries, manifests, Plotly and MathJax. The download contains current-facing derived data and documents; immutable historical raw records stay archive-only. Structural and executable application audits are separate from browser rendering.", ""]
    lines += [payload["report_profile"].get("rendering_note","Browser rendering is unverified."), ""]
    if payload.get("epoch_recovery", {}).get("statement"):
        lines += [payload["epoch_recovery"]["statement"], ""]
    if payload.get("current_focused_status"):
        lines += [payload["current_focused_status"], ""]
    if payload.get("findings", {}).get("verified"):
        lines[4:4] = ["## Matrix findings", "", *["- " + text for text in payload["findings"]["statements"]], "",
                      *payload["findings"].get("accuracy_details", []), ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=tuple(p.name for p in STUDIES.iterdir() if (p/"report_config.json").is_file()), required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    study_root = STUDIES / args.study
    source = args.input or study_root / "outputs/verification/summary.json"
    output = args.output or study_root / "REPORT.html"
    html, payload, audit = build(args.study, source)
    output.write_text(html, encoding="utf-8", newline="\n")
    (study_root / "REPORT.md").write_text(markdown(payload), encoding="utf-8")
    audit.update(report_sha256=sha256(output.read_bytes()).hexdigest(), bytes=output.stat().st_size)
    (source.parent / "report_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
