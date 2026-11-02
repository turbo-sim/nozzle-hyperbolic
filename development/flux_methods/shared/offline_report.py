"""Portable saved-data study reports; no physical model or solver imports."""

import base64
import gzip
from hashlib import sha256
from html import escape
from io import BytesIO
import json
from pathlib import Path
import re
from zipfile import ZipFile, ZIP_DEFLATED

ASSETS = Path(__file__).resolve().parent / "assets"
REPO = Path(__file__).resolve().parents[3]


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if isinstance(value, float):
        import math
        return value if math.isfinite(value) else None
    return str(value) if isinstance(value, Path) else value


def json_text(value):
    return (json.dumps(clean(value), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            .replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def check_inputs(study):
    qa = study / "outputs/report_presentation"
    manifest = json.loads((qa / "input_manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest.items():
        assert sha256((study / name).read_bytes()).hexdigest() == digest, name
    return manifest


def prepare_inputs(study, files):
    """Freeze a new saved-data population; never replace an existing digest."""
    study = Path(study).resolve()
    qa = study / "outputs/report_presentation"
    qa.mkdir(parents=True, exist_ok=True)
    target = qa / "input_manifest.json"
    manifest = check_inputs(study) if target.exists() else {}
    updated = dict(manifest)
    for file in files:
        file = Path(file).resolve()
        if file.is_file():
            name = file.relative_to(study).as_posix()
            digest = sha256(file.read_bytes()).hexdigest()
            assert name not in updated or updated[name] == digest, name
            updated[name] = digest
    if updated != manifest or not target.exists():
        target.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    return updated


def render(study, data, sections, application, files=()):
    study = Path(study).resolve()
    manifest = check_inputs(study)
    shared = [ASSETS / name for name in ("report_theme.css", "report_theme.js", "study_explorer.css", "study_explorer.js", "study_explorer.html", "vendor/plotly-2.35.2.min.js.gz", "vendor/PLOTLY_LICENSE.txt")]
    utilities = [Path(__file__), Path(__file__).with_name("audit_offline_report.py"), Path(__file__).with_name("audit_saved_explorers.mjs"), Path(__file__).with_name("report_dom.mjs")]
    data["presentation"] = dict(numerical_cases_rerun=False, inputs_sha256=manifest, plotly_version="2.35.2",
                                shared_assets_sha256={p.name: sha256(p.read_bytes()).hexdigest() for p in shared})
    payload = json_text(data)
    archive = BytesIO()
    with ZipFile(archive, "w", ZIP_DEFLATED) as z:
        z.writestr("plotted_data.json", payload.encode("utf-8"))
        z.writestr("input_manifest.json", json.dumps(manifest, indent=2))
        for name in manifest:
            z.write(study / name, "original/" + name)
        for p in (*files, *utilities):
            p = Path(p)
            if p.is_file():
                z.write(p, "renderer/" + p.resolve().relative_to(REPO).as_posix())
        for p in shared:
            z.write(p, "renderer/" + p.relative_to(REPO).as_posix())
        for p in (study / "outputs/report_presentation").glob("*_audit.json"):
            z.write(p, "checks/" + p.name)
    def download(name, raw, mime):
        return f'<a download="{escape(name)}" data-sha256="{sha256(raw).hexdigest()}" href="data:{mime};base64,{base64.b64encode(raw).decode()}">{escape(name)}</a>'
    links = download("report_data_and_evidence.zip", archive.getvalue(), "application/zip") + download("plotted_data.json", payload.encode(), "application/json")
    links += download("PLOTLY_LICENSE.txt", (ASSETS / "vendor/PLOTLY_LICENSE.txt").read_bytes(), "text/plain")
    values = dict(sections, THEME_CSS=(ASSETS / "report_theme.css").read_text(encoding="utf-8"),
        THEME_JS=(ASSETS / "report_theme.js").read_text(encoding="utf-8"), STYLE=(ASSETS / "study_explorer.css").read_text(encoding="utf-8"),
        COMMON_JS=(ASSETS / "study_explorer.js").read_text(encoding="utf-8"), APP=Path(application).read_text(encoding="utf-8"),
        PAYLOAD=payload, DOWNLOADS=links,
        PLOTLY=gzip.decompress((ASSETS / "vendor/plotly-2.35.2.min.js.gz").read_bytes()).decode().replace("</script", "<\\/script"))
    template = (ASSETS / "study_explorer.html").read_text(encoding="utf-8")
    html = re.sub(r"\{\{([A-Z_]+)\}\}", lambda m: values[m[1]], template)
    (study / "REPORT.html").write_text(html, encoding="utf-8")
    check_inputs(study)
    qa = study / "outputs/report_presentation"
    (qa / "render.json").write_text(json.dumps(dict(inputs_unchanged=len(manifest), numerical_cases_rerun=False,
        html_sha256=sha256(html.encode()).hexdigest(), payload_sha256=sha256(payload.encode()).hexdigest()), indent=2) + "\n")
    print(f"Rendered {study / 'REPORT.html'}; {len(manifest)} input files unchanged; no solves.")
