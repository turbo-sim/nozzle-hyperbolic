"""Restore exact saved numerical evidence embedded in the final HTML report.

Uses only the standard library. Verifies the archive and every recorded SHA-256
before writing beneath this study's outputs/. Existing different files cause an
error; this command never overwrites them or runs a numerical solver.
"""

import argparse
import base64
from hashlib import sha256
from html.parser import HTMLParser
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import zipfile


STUDY = Path(__file__).resolve().parents[1]


class EvidenceParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.archive_uri = None
        self.in_data = False
        self.data = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and attrs.get("download") == "benchmark_evidence.zip":
            if self.archive_uri is not None:
                raise ValueError("Duplicate numerical evidence archive")
            self.archive_uri = attrs.get("href")
        if tag == "script" and attrs.get("id") == "benchmark-data":
            self.in_data = True

    def handle_data(self, value):
        if self.in_data:
            self.data.append(value)

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_data = False


def destination(name):
    """Map the final report's archive layout to the study's current layout."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name or path.as_posix() != name:
        raise ValueError("Unsafe archive entry: " + name)
    aliases = {
        "campaign/validation/roe_audit.json": "roe_audit.json",
        "campaign/validation/roe_sod.json": "roe_transient.json",
        "campaign/validation/roe_real_gas_summary.json": "real_gas_summary.json",
        "campaign/validation/sod_final_profiles.png": "figures/sod_final_profiles.png",
    }
    if name in aliases:
        return Path(aliases[name])
    if name.startswith("campaign/validation/real_gas/"):
        return Path(name.removeprefix("campaign/validation/real_gas/"))
    if name == "campaign/summary.json" or name.startswith(("campaign/raw/", "campaign/scenarios/")):
        return Path("roe_steady_verification") / name.removeprefix("campaign/")
    raise ValueError("Unrecognized final-study evidence entry: " + name)


def restore(report, output):
    output = Path(output).resolve()
    if not output.is_relative_to((STUDY / "outputs").resolve()):
        raise ValueError("Restored evidence must stay beneath development/roe_method/outputs")
    parser = EvidenceParser()
    parser.feed(Path(report).read_text(encoding="utf-8"))
    prefix = "data:application/zip;base64,"
    if not parser.archive_uri or not parser.archive_uri.startswith(prefix):
        raise ValueError("The HTML has no embedded numerical evidence ZIP")
    raw = base64.b64decode(parser.archive_uri[len(prefix):], validate=True)
    payload = json.loads("".join(parser.data))
    metadata = payload["report"]
    if sha256(raw).hexdigest() != metadata["evidence_sha256"]:
        raise ValueError("Embedded archive SHA-256 differs from the report declaration")
    expected = {"campaign/" + name: digest for name, digest in metadata["original_file_sha256"].items()}
    if metadata.get("baseline_files"):
        raise ValueError("Use the final Roe report, without a separate baseline campaign")
    files = []
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        if len(archive.namelist()) != len(expected) or set(archive.namelist()) != set(expected):
            raise ValueError("Archive entries differ from the complete declared evidence set")
        for name, digest in expected.items():
            blob = archive.read(name)
            if sha256(blob).hexdigest() != digest:
                raise ValueError("Evidence SHA-256 mismatch: " + name)
            target = (output / destination(name)).resolve()
            if not target.is_relative_to(output):
                raise ValueError("Unsafe restore target: " + str(target))
            if target.exists() and target.read_bytes() != blob:
                raise FileExistsError("Different existing evidence; use a fresh output subdirectory: " + str(target))
            files.append((target, blob))
    if len({path for path, _ in files}) != len(files):
        raise ValueError("Multiple archive entries resolve to the same destination")
    # All checks finish before any filesystem mutation.
    for target, blob in files:
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
    return dict(files_verified=len(files), output=str(output), archive_sha256=sha256(raw).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=STUDY / "REPORT.html")
    parser.add_argument("--output", type=Path, default=STUDY / "outputs")
    args = parser.parse_args()
    print(json.dumps(restore(args.report, args.output), indent=2))


if __name__ == "__main__":
    main()
