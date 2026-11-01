"""Structural audit of the standalone angle-forcing report; no browser/solver."""

from __future__ import annotations

import argparse
import base64
import gzip
import html
from hashlib import sha256
from html.parser import HTMLParser
from io import BytesIO
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


STUDY = Path(__file__).resolve().parents[1]
OUT = STUDY/"outputs/diffuser"
VENDOR = STUDY/"scripts/vendor"


class Markup(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.tabs, self.sections, self.external, self.images, self.downloads = [], [], [], [], {}, []
        self.controls = {}

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if attrs.get("id"):
            self.ids.append(attrs["id"])
        if "data-tab" in attrs:
            self.tabs.append(attrs["data-tab"])
        if tag == "section":
            self.sections.append(attrs.get("id"))
        if tag == "input" and attrs.get("id"):
            self.controls[attrs["id"]] = attrs
        if tag in ("script", "link", "iframe", "object", "embed", "source", "video", "audio"):
            for name in ("src", "href", "data"):
                if attrs.get(name) and not attrs[name].startswith(("data:", "#")):
                    self.external.append((tag, name, attrs[name]))
        if tag == "img":
            self.images[attrs.get("id")] = attrs.get("src", "")
        if "data-save-image" in attrs:
            self.downloads.append((attrs["data-save-image"], attrs.get("data-filename")))


def literal(source, name):
    match = re.search(r"\bconst\s+"+re.escape(name)+r"\s*=", source)
    if not match:
        raise AssertionError(f"Missing embedded {name}")
    return json.JSONDecoder().raw_decode(source[match.end():].lstrip())[0]


def raw_digest(path):
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_artifact_path(path):
    try:
        return path.resolve().relative_to(STUDY).as_posix()
    except ValueError:
        return str(path.resolve())


def check_compact_evidence(payload, results, results_path):
    """Independently enforce bulk omission and the exact long-record projection."""
    omitted = ("states", "history", "metrics", "rejections")
    expected = dict(results)
    expected["transient"] = {}
    projections = {}
    for name, record in results.get("transient", {}).items():
        value = {key: item for key, item in record.items() if key not in omitted}
        count = len(record.get("times", []))
        frames = int(record.get("config", results.get("config", {})).get("forcing", {}).get("frames_per_cycle", 24))
        assert frames > 0
        if count > 28*frames+1:
            indices = np.unique(np.concatenate((np.arange(3*frames+1), np.arange(count-25*frames-1, count)))).tolist()
            value["times"] = [record["times"][i] for i in indices]
            for key in ("fields", "boundary_fields"):
                if key in record:
                    assert all(len(series) == count for series in record[key].values())
                    value[key] = {field: [series[i] for i in indices] for field, series in record[key].items()}
            if "contents" in record:
                assert len(record["contents"]) == count
                value["contents"] = [record["contents"][i] for i in indices]
            value["boundary_history"] = dict(times=record["times"], fields=record.get("boundary_fields", {}))
            value["snapshot_selection"] = dict(raw_indices=indices, original_count=count,
                retained_count=len(indices), frames_per_cycle=frames, startup_cycles=3, final_cycles=25,
                reason="Long settling record: retain the first three and final twenty-five forcing cycles of spatial snapshots. Full sampled boundary history is embedded separately; all raw spatial snapshots remain in the fingerprinted numerical artifacts.")
            projections[name] = value["snapshot_selection"]
        expected["transient"][name] = value
    assert payload["results"] == expected, "Embedded data differs from the exact allowed archive projection"
    for name, record in results.get("transient", {}).items():
        embedded = payload["results"]["transient"][name]
        assert all(key not in embedded for key in omitted)
        for key in ("summary", "periodicity", "config"):
            if key in record:
                assert embedded[key] == record[key], f"A retained {name}.{key} series was altered"
        if name in projections:
            assert embedded["boundary_history"] == {"times": record["times"], "fields": record.get("boundary_fields", {})}
    manifest = payload["raw_artifact_manifest"]
    assert manifest["snapshot_projections"] == projections
    paths = [f"transient.{name}.{key}" for name, record in results.get("transient", {}).items()
             for key in omitted if key in record]

    def check(entry, path, keys):
        assert entry["path"] == expected_artifact_path(path), "Raw artifact path mismatch"
        assert entry["available"] and path.is_file(), "Full raw numerical evidence is missing"
        assert entry["omitted_keys"] == keys, "Raw artifact omitted-key manifest mismatch"
        assert entry["bytes"] == path.stat().st_size
        assert entry["sha256"] == raw_digest(path), "Raw artifact SHA256 mismatch"

    check(manifest["raw_results"], results_path, paths)
    assert set(manifest["raw_runs"]) == set(results.get("transient", {}))
    for name, record in results.get("transient", {}).items():
        check(manifest["raw_runs"][name], results_path.parent/"runs"/f"{name}.json",
              [key for key in omitted if key in record])
    return len(manifest["raw_runs"])+1


def check_equations(text, payload):
    manifest = payload["math_rendering"]
    assert manifest["engine"] == "MathJax" and manifest["version"] == "3.2.2"
    assert "fontCache:none" in manifest["format"]
    assert raw_digest(VENDOR/manifest["package_file"]) == manifest["package_sha256"]
    assert raw_digest(VENDOR/manifest["latex_source_file"]) == manifest["latex_source_sha256"]
    assert raw_digest(VENDOR/manifest["renderer_file"]) == manifest["renderer_sha256"]
    sources = json.loads((VENDOR/manifest["latex_source_file"]).read_text(encoding="utf-8"))
    equations = re.findall(r'<div class="equation" data-math-key="([^"]+)" data-latex="([^"]*)">(<svg[\s\S]*?</svg>)</div>', text)
    assert {key for key, _, _ in equations} == set(manifest["used_formulas"])
    assert len(equations) >= 8, "Expected the geometry, boundary, inventory and averaging equations"
    for key, latex, svg in equations:
        expected = manifest["formulas"][key]
        assert html.unescape(latex) == sources[key] == expected["tex"]
        assert sha256(svg.encode()).hexdigest() == expected["sha256"]
        assert svg == (VENDOR/expected["svg_file"]).read_text(encoding="utf-8")
        root = ET.fromstring(svg)
        assert root.tag.endswith("svg") and root.attrib.get("role") == "img"
        assert root.attrib.get("aria-label") == expected["tex"]
        assert any(node.tag.endswith("path") for node in root.iter()), "Missing SVG glyph paths"
        assert not re.search(r'<(?:script|image|use)\b|data-mjx-error|merror|https?://(?!www.w3.org)', svg)
    return len(equations)


def audit(report_path=STUDY/"REPORT.html", results_path=OUT/"results.json"):
    report_path, results_path = Path(report_path), Path(results_path)
    text = report_path.read_text(encoding="utf-8")
    markup = Markup(); markup.feed(text)
    assert len(markup.ids) == len(set(markup.ids)), "Duplicate document IDs"
    assert len(markup.tabs) == 6 and set(markup.tabs) == set(markup.sections), "Invalid chapter navigation"
    assert set(markup.tabs) <= set(markup.ids)
    assert not markup.external, f"External rendering dependency: {markup.external}"
    styles = "\n".join(re.findall(r"<style[^>]*>([\s\S]*?)</style>", text))
    assert not re.search(r"@import|url\(\s*['\"]?(?:https?:|//)", styles, re.I), "External CSS/font dependency"
    scripts = re.findall(r"<script[^>]*>([\s\S]*?)</script>", text)
    assert len(scripts) == 2, "Expected embedded Plotly and application scripts"
    app = scripts[-1]
    plots, frames, encoded = [literal(app, name) for name in ("plots", "frames", "archive")]
    payload = json.loads(gzip.decompress(base64.b64decode(encoded, validate=True)))
    equation_count = check_equations(text, payload)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    digest = sha256(results_path.read_bytes()).hexdigest()
    raw_artifacts_verified = check_compact_evidence(payload, results, results_path)
    assert payload["results_sha256"] == digest
    selected_id = results["selected_id"]
    selected = results["transient"][selected_id]
    slider_source = payload["results"]["transient"][selected_id]
    indices = slider_source.get("snapshot_selection", {}).get("raw_indices", list(range(len(selected["times"]))))
    assert frames["times"] == [selected["times"][i] for i in indices] and frames["x"] == selected["x"]
    assert frames["raw_indices"] == indices and frames["original_count"] == len(selected["times"])
    assert frames["snapshot_selection"] == slider_source.get("snapshot_selection")
    gaps = [dict(after_frame=i, raw_after=left, raw_before=right,
        from_time_s=selected["times"][left], to_time_s=selected["times"][right])
        for i, (left, right) in enumerate(zip(indices[:-1], indices[1:])) if right-left > 1]
    assert frames["gaps"] == gaps
    assert int(markup.controls["frame"]["max"]) == len(indices)-1
    if gaps:
        assert "Its gap is explicit" in text and "frame-raw-index" in markup.ids
    assert payload["visuals"]["selected_id"] == selected_id
    assert payload["visuals"]["results_sha256"] == digest
    if payload.get("audit"):
        assert payload["audit"]["results_sha256"] == digest
    for plot in plots:
        assert plot["id"] in markup.ids
        for trace in plot["data"]:
            x, y = np.asarray(trace["x"]), np.asarray(trace["y"])
            assert x.ndim == y.ndim == 1 and len(x) == len(y), f"Trace shape mismatch in {plot['id']}"
            assert np.isfinite(x).all() and np.isfinite(y).all(), "Nonfinite plotted evidence"
    for key, values in frames["fields"].items():
        if values:
            assert values["encoding"] == "float64-le/base64", "Unexpected slider data encoding"
            values = np.frombuffer(base64.b64decode(values["data"], validate=True), dtype="<f8").reshape(values["shape"])
            assert values.shape == (len(frames["times"]), len(frames["x"]))
            assert np.isfinite(values).all()
            original = np.asarray([selected["fields"][key][i] for i in indices], dtype="<f8")
            assert np.array_equal(values.view("<u8"), original.view("<u8")), "Slider float64 values changed"
    assert {key for key, _ in markup.downloads} == set(markup.images), "Image download targets differ"
    assert len(markup.images) == 3, "Expected steady contours and two GIFs"
    image_records = {}
    manifest = payload["visuals"]
    expected_counts = {"startup-gif": manifest["startup_frames"], "periodic-gif": manifest["late_frames"]}
    filenames = dict(markup.downloads)
    for key, source in markup.images.items():
        match = re.fullmatch(r"data:(image/(?:png|gif));base64,([A-Za-z0-9+/=\s]+)", source)
        assert match, f"Image {key} is not embedded as a data URI"
        raw = base64.b64decode(match.group(2), validate=True)
        assert raw == (results_path.parent/filenames[key]).read_bytes(), "Embedded image differs from exported media"
        with Image.open(BytesIO(raw)) as image:
            count = getattr(image, "n_frames", 1)
            record = dict(format=image.format, width=image.width, height=image.height, frames=count)
            if key in expected_counts:
                assert image.format == "GIF" and count == expected_counts[key]
                durations = []
                for frame in range(count):
                    image.seek(frame); durations.append(image.info.get("duration"))
                assert set(durations) == {50}, "GIF playback differs from declared 20 fps"
                record["frame_duration_ms"] = 50
            image_records[key] = record
    comparison = payload["results"].get("comparison", {})
    if comparison.get("mean"):
        times = np.asarray(comparison["averaging_samples"]["times"])
        for key, values in comparison["averaging_samples"]["fields"].items():
            mean = np.trapezoid(np.asarray(values), times, axis=0)/(times[-1]-times[0])
            np.testing.assert_allclose(mean, comparison["mean"][key], rtol=1e-13, atol=1e-12)
    periodic = results["periodicity"][selected_id]["verified"]
    assert manifest["periodic_verified"] == periodic
    forcing = results["config"]["forcing"]
    if comparison.get("mean"):
        expected_cycles = forcing["collection_cycles"]
        assert comparison["complete_cycles"] == expected_cycles
        averaged_times = np.asarray(comparison["averaging_samples"]["times"])
        assert len(averaged_times) == expected_cycles*forcing["frames_per_cycle"]+1
        np.testing.assert_allclose(averaged_times[-1]-averaged_times[0], expected_cycles/forcing["frequency_Hz"], rtol=0, atol=1e-10/forcing["frequency_Hz"])
        recorded_window = manifest.get("analysis_window_s", manifest.get("collection_window_s"))
        np.testing.assert_allclose(recorded_window, averaged_times[[0,-1]], rtol=0, atol=1e-10/forcing["frequency_Hz"])
        assert manifest["late_window_s"][0] >= averaged_times[0]-1e-12
        assert manifest["late_window_s"][1] <= averaged_times[-1]+1e-12
        assert "representative cycles out of" in text
    if periodic:
        assert comparison.get("verified_periodic") is True
        assert selected["summary"]["collection_cycles_completed"] == expected_cycles
    else:
        assert not comparison.get("verified_periodic"), "Finite mean is mislabeled periodic"
        assert "Late valid transient, not periodic" in text
        assert manifest["late_file"] == "late_transient.gif"
        primary = results["transient"][results.get("startup_id", selected_id)]
        if primary.get("success") and primary.get("reason") == "cycle_limit_reached":
            duration = (primary["times"][-1]-primary["times"][0])*forcing["frequency_Hz"]
            np.testing.assert_allclose(duration, forcing["maximum_cycles"], rtol=0, atol=1e-7)
            if comparison.get("mean"):
                np.testing.assert_allclose(averaged_times[-1], primary["times"][-1], rtol=0, atol=1e-10/forcing["frequency_Hz"])
        for value in results.get("temporal_comparison", {}).get("comparisons", {}).values():
            window = value.get("windows", {}).get("matched_transient")
            if window:
                assert window["periodic_state_claimed"] is False
                np.testing.assert_allclose([window["window_start_s"], window["window_end_s"]], averaged_times[[0,-1]], rtol=0, atol=1e-10/forcing["frequency_Hz"])
                assert window["samples"] == len(averaged_times)
    summary = dict(passed=True, mode="Offline structural/content audit; browser behavior is checked separately",
        results_sha256=digest, selected_id=selected_id, startup_id=results.get("startup_id", selected_id),
        chapters=markup.tabs, plots=len(plots), external_rendering_dependencies=0,
        embedded_archive_matches_exact_pruning=True,
        omitted_transient_keys=["states", "history", "metrics", "rejections"],
        raw_artifact_hashes_verified=raw_artifacts_verified, images=image_records,
        projected_spatial_records={name: dict(original_count=value["original_count"], retained_count=value["retained_count"])
            for name, value in payload["raw_artifact_manifest"]["snapshot_projections"].items()},
        slider_float64_bitwise_verified=True,
        slider_retained_snapshots=len(indices), slider_original_snapshots=len(selected["times"]),
        prerendered_mathjax_svg_equations=equation_count,
        report_bytes=report_path.stat().st_size,
        periodic_verified=periodic, average_kind=comparison.get("averaging_kind"))
    output = results_path.parent/"html_audit.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=STUDY/"REPORT.html")
    parser.add_argument("--results", type=Path, default=OUT/"results.json")
    args = parser.parse_args(); audit(args.report, args.results)


if __name__ == "__main__":
    main()
