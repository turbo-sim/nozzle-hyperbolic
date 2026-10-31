"""Export a committed Euler package for isolated historical comparisons."""

import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[5]
STUDY = Path(__file__).resolve().parents[1]
BASELINE_REF = "dee35ea"

def freeze_source(output, revision):
    """Export/verify the committed package under this study's ignored outputs."""
    commit = (
        subprocess.check_output(["git", "rev-parse", revision], cwd=ROOT)
        .decode()
        .strip()
    )
    archive = zipfile.ZipFile(
        io.BytesIO(
            subprocess.check_output(
                ["git", "archive", "--format=zip", commit, "src/nozzle_hyperbolic"],
                cwd=ROOT,
            )
        )
    )
    directory = (Path(output).resolve() / "source_snapshot").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    for member in archive.infolist():
        if member.is_dir():
            continue
        target = (directory / member.filename).resolve()
        if not target.is_relative_to(directory):
            raise ValueError("Unexpected archive path")
        blob = archive.read(member.filename)
        if target.exists():
            # A prior checkout can use different line endings. Retain its bytes
            # after verifying identical source text, so saved fingerprints stay valid.
            actual = target.read_bytes().replace(b"\r\n", b"\n")
            if actual != blob.replace(b"\r\n", b"\n"):
                raise ValueError(
                    f"Existing snapshot differs from baseline: {member.filename}"
                )
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
    metadata = dict(
        commit=commit,
        package_path="src/nozzle_hyperbolic",
        files={
            p.relative_to(directory)
            .as_posix(): hashlib.sha256(p.read_bytes())
            .hexdigest()
            for p in (directory / "src/nozzle_hyperbolic").rglob("*.py")
        },
    )
    (directory / "snapshot_manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return directory / "src", commit
