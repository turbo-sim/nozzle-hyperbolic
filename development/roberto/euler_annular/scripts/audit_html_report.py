"""Audit the single-file offline HTML artifact without a browser or numerical run.

Run from the repository root:
  poetry run python -m development.roberto.euler_annular.scripts.audit_html_report

Only artifact_audit.json and artifact_audit.log are written under the selected
audit output directory. Saved numerical inputs, source, HTML and ZIP are read-only.
"""

import argparse
import base64
import io
import ast
from bisect import bisect_right
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
from zipfile import BadZipFile, ZipFile

from .html_report_data import FIELD_METADATA, HISTORY_COLUMNS


STUDY = Path(__file__).resolve().parents[1]
PAGES = ("REPORT.html",)


def digest(raw):
    return sha256(raw).hexdigest()


def text_newlines(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")


def reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def strict_json(text):
    return json.loads(text, parse_constant=reject_constant)


def literal(path, name):
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name
                                                for target in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise ValueError(f"Missing authoritative configuration literal {name} in {path}")


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids, self.hrefs, self.resources, self.buttons, self.inputs = [], [], [], [], []
        self.scripts, self.styles, self.math = [], [], []
        self.h2_ids = []
        self.csp = []
        self.active_script, self.active_style, self.active_math = None, None, None

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if attrs.get("id"):
            self.ids.append(attrs["id"])
        if "href" in attrs:
            self.hrefs.append(dict(tag=tag, href=attrs["href"], rel=attrs.get("rel")))
        if "xlink:href" in attrs:
            self.hrefs.append(dict(tag=tag, href=attrs["xlink:href"], rel=attrs.get("rel")))
        if tag == "h2" and attrs.get("id"):
            self.h2_ids.append(attrs["id"])
        for name in ("src", "srcset", "poster", "data"):
            if name in attrs:
                self.resources.append(dict(tag=tag, attribute=name, value=attrs[name]))
        if tag == "link" and attrs.get("rel", "").lower() in (
                "stylesheet", "preload", "modulepreload", "prefetch", "icon"):
            self.resources.append(dict(tag=tag, attribute="href", value=attrs.get("href", "")))
        if tag == "button" and "data-stride" in attrs:
            self.buttons.append(attrs)
        if tag == "input":
            self.inputs.append(attrs)
        if tag == "meta" and attrs.get("http-equiv", "").lower() == "content-security-policy":
            self.csp.append(attrs.get("content", ""))
        if self.active_math is not None:
            self.active_math["depth"] += 1
        if tag == "script":
            self.active_script = dict(attrs=attrs, text="")
        elif tag == "style":
            self.active_style = ""
        elif "math-block" in attrs.get("class", "").split():
            self.active_math = dict(kind="block", depth=1, text="")
        elif "math-inline" in attrs.get("class", "").split():
            self.active_math = dict(kind="inline", depth=1, text="")

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag == "script" and self.active_script is not None:
            self.scripts.append(self.active_script)
            self.active_script = None
        elif tag == "style" and self.active_style is not None:
            self.styles.append(self.active_style)
            self.active_style = None
        if self.active_math is not None:
            self.active_math["depth"] -= 1
            if self.active_math["depth"] == 0:
                self.math.append(self.active_math)
                self.active_math = None

    def handle_data(self, text):
        if self.active_script is not None:
            self.active_script["text"] += text
        elif self.active_style is not None:
            self.active_style += text
        elif self.active_math is not None:
            self.active_math["text"] += text


class Audit:
    def __init__(self, study):
        self.study = study.resolve()
        self.checks, self.details, self.observed = [], {}, {}
        self.comparisons, self.differences = 0, []

    def check(self, name, passed, details=None):
        self.checks.append(dict(name=name, passed=bool(passed)))
        if details is not None:
            self.details[name] = details

    def equal(self, name, actual, expected):
        self.comparisons += 1
        if actual != expected:
            self.differences.append(name)

    def read(self, path):
        raw = path.read_bytes()
        absolute = str(path.resolve())
        if absolute in self.observed and self.observed[absolute] != digest(raw):
            raise ValueError(f"Artifact/input changed during audit: {path}")
        self.observed[absolute] = digest(raw)
        return raw

    def json(self, path):
        return strict_json(self.read(path).decode("utf-8"))

    def finish(self):
        changed = [path for path, original in self.observed.items()
                   if not Path(path).is_file() or digest(Path(path).read_bytes()) != original]
        self.check("all_observed_artifacts_and_inputs_unchanged_during_audit", not changed, changed)
        return dict(
            schema_version=1, generated_at_utc=datetime.now(timezone.utc).isoformat(),
            success=all(check["passed"] for check in self.checks), checks=self.checks,
            failed_checks=[check["name"] for check in self.checks if not check["passed"]],
            details=self.details, observed_files=len(self.observed),
            numerical_sources_or_saved_json_modified=False,
            browser_visual_qa=dict(performed=False, claim=False,
                reason="Chrome app was not approved in this session; this is a static artifact audit."),
            browser_runtime_qa=dict(performed_by_this_auditor=False,
                note="Root-owned JavaScript runtime/figure checks are separate evidence. Static checks do not establish visual layout quality."),
        )


def embedded_build(page):
    candidates = []
    decoder = json.JSONDecoder(parse_constant=reject_constant)
    for script in page.scripts:
        match = re.search(r"window\.REPORT_BUILD\s*=\s*", script["text"])
        if match:
            candidates.append(decoder.raw_decode(script["text"][match.end():])[0])
    if len(candidates) != 1:
        raise ValueError(f"Expected one embedded REPORT_BUILD, found {len(candidates)}")
    return candidates[0]


def source_root(saved, run):
    return (run["success"] and run["metrics"].get("admissible", False)
            and all(math.isfinite(run["metrics"][key])
                    and run["metrics"][key] <= saved["settings"]["tolerance"]
                    for key in ("spatial_rms", "boundary_max", "conservation_max")))


def source_accuracy(saved, run):
    errors = run["errors"]
    if saved.get("shock_m") is not None:
        return math.isfinite(errors["shock_error_cells"]) and errors["shock_error_cells"] <= 2
    values = (errors["Cp_max"], *errors["conservative_L1"])
    return (all(math.isfinite(value) for value in values) and errors["Cp_max"] < 1e-3
            and max(errors["conservative_L1"]) < 1e-3)


def compare_history(audit, name, actual, records):
    columns = [key for key in HISTORY_COLUMNS if any(key in row for row in records)]
    accepted = [[row.get(key) for key in columns] for row in records if row.get("accepted", True)]
    rejected = [[row.get(key) for key in columns] for row in records if not row.get("accepted", True)]
    expected = dict(columns=columns, accepted=accepted, rejected=rejected,
                    counts=dict(total=len(records), accepted=len(accepted), rejected=len(rejected)),
                    sampled=False)
    audit.equal(name, actual, expected)


def compare_mesh(audit, mesh, methods, captured_inputs):
    path = audit.study/mesh["path"]
    saved = audit.json(path)
    name = mesh["path"]
    audit.equal(name+" captured", mesh["path"] in captured_inputs, True)
    for key in ("cells", "m", "m_faces", "r", "volumes", "settings"):
        audit.equal(name+" "+key, mesh[key], saved[key])
    fields = [field[0] for field in FIELD_METADATA]
    source_audit = (audit.json(audit.study/mesh["audit"]["path"])
                    if mesh["audit"]["available"] else None)
    if source_audit is not None:
        audit.equal(name+" audit flag", mesh["audit"]["audit_pass"], source_audit["audit_pass"])
    audit.equal(name+" reference", mesh["reference"], {key: saved["reference"][key] for key in fields})
    audit.equal(name+" method IDs", set(mesh["methods"]), set(methods))
    rows = []
    for method in methods:
        actual, run = mesh["methods"][method], saved["methods"][method]
        prefix = name+" "+method
        for key in ("success", "message", "metrics", "errors"):
            audit.equal(prefix+" "+key, actual[key], run[key])
        audit.equal(prefix+" fields", actual["fields"], {key: run["fields"][key] for key in fields})
        audit.equal(prefix+" root flag", actual["root_pass"], source_root(saved, run))
        audit.equal(prefix+" accuracy flag", actual["accuracy_pass"], source_accuracy(saved, run))
        audit.equal(prefix+" recorded accuracy", actual["recorded_analytical_target_met"], run["analytical_target_met"])
        audit.equal(prefix+" flags", actual["flags"], dict(
            admissible=run["metrics"].get("admissible"),
            boundary_mach=run["metrics"].get("boundary_mach"),
            boundary_constraints=run.get("boundary_constraints")))
        if source_audit is not None:
            for key, value in mesh["audit"]["methods"][method].items():
                audit.equal(prefix+" current audit "+key, value, source_audit["methods"][method].get(key))
        compare_history(audit, prefix+" unsampled accepted/rejected history", actual["history"], run.get("history", []))
        rows.append(dict(case=saved["case"], cells=saved["cells"], method=method,
                         root=source_root(saved, run), accuracy=source_accuracy(saved, run),
                         saved=saved, run=run,
                         current_audit=source_audit["methods"][method] if source_audit is not None else None))
    return rows


def audit_payload(audit, payload, build):
    study = audit.study
    cases = literal(study/"scripts"/"verify_annular.py", "DEFAULT_CASES")
    methods = literal(study/"scripts"/"verify_annular.py", "RECONSTRUCTIONS")
    meshes = literal(study/"scripts"/"verify_annular.py", "DEFAULT_MESHES")
    inputs = {entry["path"]: entry for entry in build["input_files"]}
    audit.check("captured_input_paths_unique", len(inputs) == len(build["input_files"]))
    audit.equal("build and payload captured inputs", build["input_files"], payload["provenance"]["input_files"])
    changed_inputs = []
    for name, item in inputs.items():
        path = Path(name) if Path(name).is_absolute() else study/name
        if not path.is_file():
            changed_inputs.append(dict(path=name, reason="missing"))
            continue
        raw = audit.read(path)
        if digest(raw) != item["sha256"] or len(raw) != item["bytes"]:
            changed_inputs.append(dict(path=name, reason="bytes or SHA-256 differ"))
        if path.suffix == ".json":
            strict_json(raw.decode("utf-8"))
    audit.check("captured_inputs_match_build_sha256", not changed_inputs,
                dict(files=len(inputs), changed=changed_inputs,
                     note="Only captured inputs are checked; later generated README/manifests are not newly added to this immutable list."))
    source_changes = []
    for name, original in payload["provenance"]["current_source_sha256"].items():
        path = study.parents[2]/name
        if not path.is_file() or digest(audit.read(path)) != original:
            source_changes.append(name)
    audit.check("captured_current_numerical_sources_match", not source_changes, source_changes)
    audit.equal("case IDs/order", [case["id"] for case in payload["cases"]], list(cases))
    audit.equal("method IDs/order", [method["id"] for method in payload["methods"]], list(methods))
    audit.equal("mesh IDs/order", payload["meshes"], list(meshes))
    primary = []
    for case in payload["cases"]:
        audit.equal(case["id"]+" mesh completeness", [mesh["cells"] for mesh in case["meshes"]], list(meshes))
        expected_grid = []
        for mesh in case["meshes"]:
            audit.equal(mesh["path"]+" canonical path", mesh["path"], f"outputs/verification/{case['id']}_{mesh['cells']}.json")
            rows = compare_mesh(audit, mesh, methods, inputs)
            primary.extend(rows)
            expected_grid.append(dict(cells=mesh["cells"], **{
                row["method"]: dict(**row["run"]["errors"], root_pass=row["root"], accuracy_pass=row["accuracy"])
                for row in rows}))
        audit.equal(case["id"]+" grid-error data", case["grid_errors"], expected_grid)
    finest = [row for row in primary if row["cells"] == max(meshes)]
    summary = payload["summary"]
    derived = dict(cases=len(cases), primary_files=len(cases)*len(meshes), method_entries=len(primary),
                   root_passes=sum(row["root"] for row in primary), finest_method_entries=len(finest),
                   finest_accuracy_passes=sum(row["accuracy"] for row in finest),
                   finest_accuracy_failures=[dict(case=row["case"], method=row["method"])
                                             for row in finest if not row["accuracy"]])
    audited = [row["current_audit"] for row in primary if row["current_audit"] is not None]
    derived.update(audited_method_entries=len(audited),
                   audited_root_passes=sum(bool(row["root_target_met"]) for row in audited),
                   audited_array_reproduction_passes=sum(bool(row["saved_arrays_reproduced"]) for row in audited))
    for key, value in derived.items():
        audit.equal("derived summary "+key, summary[key], value)
    audit.equal("build summary", build["summary"], summary)
    if payload.get("audits") is not None:
        audit.equal("canonical aggregate independent audit", payload["audits"],
                    audit.json(study/"outputs"/"verification"/"audits.json"))
    audit.check("primary_matrix_complete", len(primary) == len(cases)*len(meshes)*len(methods),
                dict(**derived, numerical_comparisons_have_zero_tolerance=True))
    for mesh in payload.get("nonuniform", []):
        compare_mesh(audit, mesh, methods, inputs)
    for mesh in payload.get("transient", {}).get("runs", []):
        saved = audit.json(study/mesh["path"])
        name = mesh["path"]
        audit.equal(name+" reference", mesh["reference"], saved["reference"]["fields"])
        audit.equal(name+" initial reference", mesh["initial_reference"], saved["initial_reference"]["fields"])
        for method in methods:
            actual, run = mesh["methods"][method], saved["methods"][method]
            for key in ("fields", "checks", "physical_time", "effective_options", "field_L1", "conservative_L1"):
                audit.equal(name+" "+method+" "+key, actual[key], run[key])
            compare_history(audit, name+" "+method+" history", actual["history"], run["history"])
            budget = actual["budget"]
            for key, value in run["physical_budget"].items():
                if key != "accepted_steps":
                    audit.equal(name+" "+method+" budget "+key, budget[key], value)
            columns = budget["steps"]["columns"]
            audit.equal(name+" "+method+" budget rows", budget["steps"]["rows"],
                        [[row.get(key) for key in columns] for row in run["physical_budget"]["accepted_steps"]])
            audit.equal(name+" "+method+" budget unsampled", budget["steps"]["sampled"], False)
    audit.check("payload_exact_saved_fields_histories_flags_errors_and_budgets", not audit.differences,
                dict(comparisons=audit.comparisons, differences=audit.differences))
    audit.check("strict_payload_reports_no_current_nonfinite_values",
        payload["provenance"]["nonfinite_source_values"]["count"] == 0
        and payload["provenance"]["nonfinite_payload_replacements"]["count"] == 0)


def audit_pages(audit, pages, raw, build):
    broken, external_navigation, resources, duplicate_ids = [], [], [], []
    for name, page in pages.items():
        duplicate_ids.extend(dict(page=name, id=key, count=count)
                             for key, count in Counter(page.ids).items() if count > 1)
        resources.extend(dict(page=name, **item) for item in page.resources
                         if not item["value"].startswith(("data:", "blob:", "#")))
        for text in page.styles:
            for target in re.findall(r"url\(\s*['\"]?([^)'\"\s]+)", text, re.IGNORECASE):
                if not target.startswith(("data:", "blob:", "#")):
                    resources.append(dict(page=name, tag="style", attribute="url", value=target))
        for item in page.hrefs:
            url = urlsplit(item["href"])
            if url.scheme in ("https", "http", "mailto") and item["tag"] == "a":
                external_navigation.append(dict(page=name, href=item["href"]))
                continue
            if url.scheme in ("data", "blob"):
                continue
            target = unquote(url.path) or name
            fragment = unquote(url.fragment)
            if (url.scheme or url.netloc or target not in pages
                    or (fragment and fragment not in pages[target].ids)):
                broken.append(dict(page=name, href=item["href"]))
        audit.check(name+" restrictive_offline_CSP", any(
            "default-src 'none'" in value and "connect-src 'none'" in value for value in page.csp))
        audit.check(name+" no_unresolved_template_or_math_tokens", not re.search(
            r"@@[A-Z_]+@@|MATHEQUATIONTOKEN[0-9]+END|data-equation-token=", raw[name]))
    audit.check("no_duplicate_static_DOM_ids", not duplicate_ids, duplicate_ids)
    audit.check("all_internal_page_and_anchor_links_resolve", not broken,
                dict(broken=broken, external_citation_navigation=external_navigation))
    audit.check("no_external_or_relative_runtime_assets", not resources, resources)
    source_path = audit.study/"scripts/report_assets/theory.html"
    source = audit.read(source_path)
    audit.check("theory_HTML_source_matches_build", digest(source) == build["chapter_sources"]["theory.html"])
    chapter=PageParser();chapter.feed(source.decode("utf-8"));chapter.close()
    actual=pages["REPORT.html"]
    audit.check("theory_all_display_and_inline_math_preserved_exactly", chapter.math == actual.math
                and build["equations"]==len(chapter.math), dict(equations=len(chapter.math)))
    required={"spatial-discretization","boundary-equations","time-integration"}
    audit.check("theory_required_section_anchors",required.issubset(actual.ids))
    audit.check("theory_section_count_matches_build",len(chapter.h2_ids)==build["theory_sections"])
    names=re.findall(r'data-report-page="([^"]+)"',raw["REPORT.html"])
    choices=re.findall(r'data-page-choice="([^"]+)"',raw["REPORT.html"])
    audit.check("one_file_four_internal_pages",names==build["internal_pages"]==choices==["results","theory","implementation","notes"])
    audit.check("retired_directory_and_duplicate_root_notes_removed",
        not (audit.study/"cfl_investigation").exists()
        and sorted(p.name for p in audit.study.glob("*.md"))==["README.md"]
        and all((audit.study/"old_notes"/name).is_file() for name in
                ("questionaire.md","IMPLEMENTATION_PLAN.md","review_manifest.json")))
    audit.check("no_neighboring_HTML_or_removed_notes_required",not any(
        item["href"].split("#")[0] for item in actual.hrefs if not item["href"].startswith(("https:","http:","mailto:","data:","blob:"))))
    records=[r for r in actual.scripts if r["attrs"].get("id")=="embedded-downloads"]
    downloads=strict_json(records[0]["text"])
    for key,item in downloads.items():
        binary=base64.b64decode(item["base64"],validate=True)
        audit.check("embedded_download_"+key,digest(binary)==item["sha256"]==build["embedded_downloads"][key]["sha256"])
        if key=="controller_evidence":
            audit.check("embedded_controller_ZIP_matches_saved_evidence",binary==audit.read(audit.study/"outputs/controller_sweep/evidence.zip"))
            with ZipFile(io.BytesIO(binary)) as bundle:
                audit.check("embedded_controller_ZIP_integrity",bundle.testzip() is None)
                summary=audit.json(audit.study/"outputs/controller_sweep/summary.json")
                audit.check("all_42_controller_runs_preserved",summary["count"]==42 and all(
                    digest(bundle.read(row["file"]))==row["sha256"] for row in summary["runs"]))


def audit_markers(audit, page, scripts, build):
    controls = [item for item in page.inputs if item.get("id") == "marker-skip"]
    app_path = audit.study/"scripts"/"report_assets"/"report.js"
    app_raw = audit.read(app_path)
    app = app_raw.decode("utf-8")
    escaped = re.sub(r"</script", r"<\\/script", app, flags=re.IGNORECASE)
    compact = re.sub(r"\s+", "", app)
    semantics = dict(
        embedded_app_matches_source=any(text_newlines(script["text"]) == text_newlines(escaped) for script in scripts),
        app_matches_build_asset_hash=digest(app_raw) == build["assets"]["report.js"],
        default_skip_ten='stride:11,markerSkip:10' in compact,
        full_lines='type:"scatter",x,y,' in compact,
        white_circle='marker:{color:"white",symbol:"circle",size:9' in compact,
        marker_layer='mode:"markers",zorder:3' in compact,
        stagger='Math.floor(state.stride/2)' in compact,
        reference_on_top='[1,2,0].forEach' in compact,
        valid_integer='Number.isSafeInteger(skipped)&&skipped>=0' in compact,
        safe_heading_lookup='getElementById(decodeURIComponent(a.hash.slice(1)))' in compact,
    )
    metadata = build.get("marker_skip_control", {})
    audit.check("default_skip_ten_and_arbitrary_nonnegative_skip_input",
                build["default_marker_stride"] == 11 and len(controls) == 1
                and all(controls[0].get(key) == value for key, value in
                        {"type":"number", "min":"0", "step":"1", "value":"10"}.items())
                and all(metadata.get(key) == value for key, value in {"min":0, "step":1, "default":10}.items()),
                dict(inputs=controls, build_control=metadata))
    audit.check("static_plot_and_navigation_semantics", all(semantics.values()), semantics)


def _deboor(points, knots, degree, u):
    span = min(max(bisect_right(knots, u)-1, degree), len(points)-1)
    work = [list(points[span-degree+j]) for j in range(degree+1)]
    for order in range(1, degree+1):
        for j in range(degree, order-1, -1):
            index = span-degree+j
            denominator = knots[index+degree-order+1]-knots[index]
            alpha = (u-knots[index])/denominator if denominator else 0.0
            work[j] = [(1-alpha)*left+alpha*right for left, right in zip(work[j-1], work[j])]
    return work[-1]


def _curve(control, names):
    """Independent homogeneous de Boor evaluation, used only by this audit."""
    weights, knots, degree = control["weights"], control["knots"], control["degree"]
    count = len(weights)
    if (not 1 <= degree < count or len(knots) != count+degree+1
            or any(weight <= 0 for weight in weights)
            or any(right < left for left, right in zip(knots, knots[1:]))):
        raise ValueError("Invalid recorded NURBS control data")
    homogeneous = [[control[name][index]*weights[index] for name in names]+[weights[index]]
                   for index in range(count)]
    derivative = [[degree*(right-left)/(knots[index+degree+1]-knots[index+1])
                   for left, right in zip(homogeneous[index], homogeneous[index+1])]
                  for index in range(count-1)]
    def evaluate(u):
        value = _deboor(homogeneous, knots, degree, u)
        change = _deboor(derivative, knots[1:-1], degree-1, u)
        position = [entry/value[-1] for entry in value[:-1]]
        tangent = [(entry-coordinate*change[-1])/value[-1]
                   for entry, coordinate in zip(change[:-1], position)]
        return position, tangent
    return evaluate


def _gauss_rule(count=16):
    nodes = []
    for index in range(1, count+1):
        value = math.cos(math.pi*(index-.25)/(count+.5))
        for _ in range(12):
            previous, polynomial = 1.0, value
            for degree in range(2, count+1):
                previous, polynomial = polynomial, ((2*degree-1)*value*polynomial-(degree-1)*previous)/degree
            derivative = count*(value*polynomial-previous)/(value*value-1)
            change = polynomial/derivative
            value -= change
            if abs(change) < 2e-16:
                break
        nodes.append((value, 2/((1-value*value)*derivative*derivative)))
    return nodes


def _arclength_geometry(control):
    evaluate, nodes = _curve(control, ("x", "r")), _gauss_rule()
    knots = sorted(set(control["knots"]))
    def length_to(u):
        total = 0.0
        for lower, bound in zip(knots, knots[1:]):
            upper = min(bound, u)
            if upper <= lower:
                continue
            half, middle = (upper-lower)/2, (upper+lower)/2
            total += half*sum(weight*math.hypot(*evaluate(middle+half*point)[1]) for point, weight in nodes)
        return total
    length = length_to(1.0)
    affine = (control["degree"] == 1 and len(control["weights"]) == 2
              and control["weights"][0] == control["weights"][1])
    def at_m(distance):
        if distance <= 0:
            u = 0.0
        elif distance >= length:
            u = 1.0
        elif affine:
            u = distance/length
        else:
            lower, upper, u = 0.0, 1.0, distance/length
            for _ in range(15):
                error = length_to(u)-distance
                if abs(error) < 3e-15:
                    break
                if error > 0:
                    upper = u
                else:
                    lower = u
                candidate = u-error/math.hypot(*evaluate(u)[1])
                u = candidate if lower < candidate < upper else (lower+upper)/2
        position, derivative = evaluate(u)
        norm = math.hypot(*derivative)
        return position, [entry/norm for entry in derivative]
    return length, at_m


def _interpolate(x, coordinates, values):
    index = min(max(bisect_right(coordinates, x)-1, 0), len(coordinates)-2)
    factor = (x-coordinates[index])/(coordinates[index+1]-coordinates[index])
    return values[index]+factor*(values[index+1]-values[index])


def audit_channels(audit, payload, build):
    channels = payload["channels"]
    failures, cases, arrows, samples = [], [], 0, 0
    worst = dict(position_m=0.0, tangent=0.0, width_m=0.0, phase_rad=0.0)
    def close(actual, expected, label, tolerance=3e-11):
        if not math.isfinite(actual) or abs(actual-expected) > tolerance*max(1.0, abs(expected)):
            failures.append(label)
    expected_ids = [case["id"] for case in payload["cases"]]
    if payload.get("nonuniform"):
        expected_ids.append("nonuniform")
    if payload.get("transient", {}).get("runs"):
        expected_ids.append("transient")
    audit.equal("channel case completeness", list(channels["cases"]), expected_ids)
    for source in channels["provenance"]["input_files"]:
        raw = audit.read(audit.study/source["path"])
        if digest(raw) != source["sha256"] or len(raw) != source["bytes"]:
            failures.append("changed channel input "+source["path"])
    for name, original in channels["provenance"]["source_hashes"].items():
        if digest(audit.read(audit.study.parents[2]/name)) != original:
            failures.append("changed channel preparation source "+name)
    cache_path = audit.study/"outputs"/"html_report"/"channel_geometry.json"
    if cache_path.is_file():
        cache = audit.json(cache_path)
        audit.equal("independent cached channel records match embedded records", channels["cases"], cache["cases"])
    for identifier, record in channels["cases"].items():
        cases.append(identifier)
        g, cp = record["geometry"], record["control_points"]
        m, x, radius, height = (g[key] for key in ("m", "x", "r", "b"))
        count = len(m)
        if (count < 3 or any(len(g[key]) != count for key in
                            ("x", "r", "b", "x_prime", "r_prime", "A", "normal_x", "normal_r"))
                or not all(right > left for left, right in zip(m, m[1:]))):
            raise ValueError("Incomplete/non-increasing channel geometry: "+identifier)
        length, evaluate = _arclength_geometry(cp["midline"])
        scalar = _curve(cp["height"], ("b",))
        close(g["length"], length, identifier+" length")
        close(m[0], 0, identifier+" inlet station")
        close(m[-1], length, identifier+" outlet station")
        degree, knots = cp["height"]["degree"], cp["height"]["knots"]
        greville = [sum(knots[i+1:i+degree+1])/degree for i in range(len(cp["height"]["b"]))]
        audit.equal(identifier+" actual Greville anchors", cp["height"]["sigma"], greville)
        audit.equal(identifier+" height controls are not geometric sample points", cp["height"]["physical_midline_points"], False)
        for i, distance in enumerate(m):
            position, tangent = evaluate(distance)
            width = scalar(distance/length)[0][0]
            worst["position_m"] = max(worst["position_m"], abs(x[i]-position[0]), abs(radius[i]-position[1]))
            worst["tangent"] = max(worst["tangent"], abs(g["x_prime"][i]-tangent[0]), abs(g["r_prime"][i]-tangent[1]))
            worst["width_m"] = max(worst["width_m"], abs(height[i]-width))
            for actual, expected in ((x[i], position[0]), (radius[i], position[1]),
                    (g["x_prime"][i], tangent[0]), (g["r_prime"][i], tangent[1]),
                    (height[i], width), (g["A"][i], 2*math.pi*radius[i]*height[i]),
                    (g["normal_x"][i], -tangent[1]), (g["normal_r"][i], tangent[0])):
                close(actual, expected, identifier+" NURBS/metric station "+str(i))
            if radius[i] <= 0 or height[i] <= 0:
                failures.append(identifier+" nonpositive radius/height")
            for side, eta in (("minus", -.5), ("plus", .5)):
                close(record["walls"][side]["x"][i], x[i]-eta*height[i]*g["r_prime"][i], identifier+" wall x")
                close(record["walls"][side]["r"][i], radius[i]+eta*height[i]*g["x_prime"][i], identifier+" wall r")
        source = audit.json(audit.study/record["source"]["path"])
        run = source["methods"][record["method"]]
        source_m = [source["m_faces"][0], *source["m"], source["m_faces"][-1]]
        source_vm, source_ell = [row[1] for row in run["state"]], [row[2] for row in run["state"]]
        flow = record["flow"]
        audit.equal(identifier+" source primitives include actual face traces", flow["source_points"],
                    dict(m=source_m, v_m=source_vm, ell=source_ell))
        if flow["valid"]:
            union = sorted(set(m+source_m))
            derivative = [_interpolate(s, source_m, source_ell)/(evaluate(s)[0][1]**2
                          *_interpolate(s, source_m, source_vm)) for s in union]
            theta = [0.0]
            for j in range(1, len(union)):
                theta.append(theta[-1]+.5*(derivative[j-1]+derivative[j])*(union[j]-union[j-1]))
            for i, distance in enumerate(m):
                vm = _interpolate(distance, source_m, source_vm)
                ell = _interpolate(distance, source_m, source_ell)
                phase = _interpolate(distance, union, theta)
                worst["phase_rad"] = max(worst["phase_rad"], abs(flow["theta"][i]-phase))
                for actual, expected in ((flow["v_m"][i], vm), (flow["ell"][i], ell),
                        (flow["v_theta"][i], ell/radius[i]),
                        (flow["dtheta_dm"][i], ell/(radius[i]**2*vm)), (flow["theta"][i], phase)):
                    close(actual, expected, identifier+" velocity/swirl phase "+str(i))
            close(flow["total_angle_rad"], theta[-1], identifier+" total swirl phase")
        elif record["paths_3d"] or any(path["arrows"] for path in record["cross_gap_paths"]):
            failures.append(identifier+" unavailable flow replaced with invented arrows")
        for path in record["cross_gap_paths"]+record["paths_3d"]:
            three_d = "z" in path
            coordinates = list(zip(path["x"], path["y"], path["z"])) if three_d else list(zip(path["x"], path["r"]))
            for i in range(count):
                eta = path["eta"]
                close(path["x"][i], x[i]-eta*height[i]*g["r_prime"][i], identifier+" path x")
                close(path["r"][i], radius[i]+eta*height[i]*g["x_prime"][i], identifier+" path r")
                if three_d:
                    close(path["theta"][i], flow["theta"][i]+path["theta0"], identifier+" seed phase")
                    close(path["y"][i], path["r"][i]*math.cos(path["theta"][i]), identifier+" path revolution y")
                    close(path["z"][i], path["r"][i]*math.sin(path["theta"][i]), identifier+" path revolution z")
            for arrow in path["arrows"]:
                index = bisect_right(m, arrow["m"])-1
                tail, head = list(coordinates[index-1]), list(coordinates[index+1])
                audit.equal(identifier+" actual forward arrow tail", arrow["tail"], tail)
                audit.equal(identifier+" actual forward arrow head", arrow["head"], head)
                chord = [end-start for start, end in zip(tail, head)]
                norm = math.hypot(*chord)
                for actual, expected in zip(arrow["direction"], [value/norm for value in chord]):
                    close(actual, expected, identifier+" forward unit arrow")
                close(arrow["chord_length"], norm, identifier+" arrow chord length")
                if not arrow["m_head"] > arrow["m_tail"] or sum(a*b for a,b in zip(arrow["direction"], chord)) <= 0:
                    failures.append(identifier+" backward flow arrow")
                arrows += 1
            samples += count
    audit.check("independent_NURBS_geometry_controls_swirl_and_forward_arrows", not failures,
                dict(cases=cases, rendering_stations=sum(len(r["geometry"]["m"]) for r in channels["cases"].values()),
                     path_samples=samples, arrows=arrows, maximum_independent_errors=worst, failures=failures,
                     method="Independent stdlib homogeneous de Boor curves/derivatives, 16-point Gauss arclength with safeguarded inversion, saved primitive interpolation and union-grid trapezoidal swirl phase; no production imports or numerical solve."))
    audit.check("channel_payload_and_control_records_exact", not audit.differences,
                dict(comparisons=audit.comparisons, differences=audit.differences))
    audit.check("channel_views_and_control_default_match_build", build.get("channel_visualizations") == dict(
        cases=len(channels["cases"]), views=["meridional", "frontal", "annular"], layout="side_by_side", control_points_default=False))


def audit_report(study):
    audit = Audit(Path(study))
    try:
        build = audit.json(audit.study/"outputs"/"html_report"/"build.json")
        generator = audit.read(audit.study/"scripts"/"write_html_report.py")
        audit.check("captured_generator_matches_build_sha256", digest(generator) == build["generator_sha256"])
        asset_changes = [name for name, original in build["assets"].items()
                         if digest(audit.read(audit.study/"scripts"/"report_assets"/name)) != original]
        audit.check("captured_local_assets_match_build_sha256", not asset_changes, asset_changes)
        raw = {name: audit.read(audit.study/name).decode("utf-8") for name in PAGES}
        pages = {}
        for name, source in raw.items():
            page = PageParser()
            page.feed(source)
            page.close()
            pages[name] = page
        embedded = {name: embedded_build(page) for name, page in pages.items()}
        audit.check("one_embedded_build_record", len(embedded)==1)
        audit.check("embedded_build_matches_build_json", all(
            all(build.get(key) == value for key, value in record.items()) for record in embedded.values()))
        json_scripts = [script for script in pages["REPORT.html"].scripts
                        if script["attrs"].get("id") == "report-data"
                        and script["attrs"].get("type") == "application/json"]
        if len(json_scripts) != 1:
            raise ValueError(f"Expected one embedded JSON payload, found {len(json_scripts)}")
        payload_text = json_scripts[0]["text"]
        payload = strict_json(payload_text)
        audit.check("strict_embedded_json_and_payload_sha256", digest(payload_text.encode("utf-8")) == build["payload_sha256"])
        audit_payload(audit, payload, build)
        audit_channels(audit, payload, build)
        audit_pages(audit, pages, raw, build)
        audit_markers(audit, pages["REPORT.html"], pages["REPORT.html"].scripts, build)
        for name, expected in build["artifacts"].items():
            artifact = audit.read(audit.study/name)
            audit.check(name+" matches_build_hash_and_bytes", digest(artifact) == expected["sha256"]
                        and len(artifact) == expected["bytes"])

    except (ValueError, KeyError, TypeError, OSError, UnicodeError, BadZipFile) as error:
        audit.check("audit_completed_without_structural_error", False,
                    dict(exception=type(error).__name__, message=str(error)))
    return audit.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_report(args.study)
    output = args.output or args.study/"outputs"/"html_report"
    output.mkdir(parents=True, exist_ok=True)
    (output/"artifact_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    lines = [f"Static artifact audit: {'PASS' if report['success'] else 'FAIL'}"]
    lines.extend(f"{'PASS' if item['passed'] else 'FAIL'} {item['name']}" for item in report["checks"])
    lines.extend([f"Observed read-only files: {report['observed_files']}",
                  "Browser visual QA: not performed; Chrome app was not approved in this session.",
                  "Browser runtime/figure checks are separate root-owned evidence."])
    text = "\n".join(lines)+"\n"
    (output/"artifact_audit.log").write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
