"""Preserve the measured source, physical cases and native baseline after a run.

This command uses only the standard library and never imports a flow solver.
Relative input/output paths are resolved from the benchmarking study directory.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import zipfile


STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[2]
FROZEN = ROOT / "legacy/pre_generic_fvm"
SOURCE = "src/nozzle_hyperbolic"
HARNESS = (
    "scripts/common.py",
    "scripts/environment.py",
    "scripts/run.py",
    "scripts/worker.py",
    "scripts/legacy_adapter.py",
    "settings.yaml",
)
MANIFEST_NAME = "snapshot_manifest.json"
PURPOSE = "Measured production source and physical cases, benchmark harness, frozen native baseline and dependency manifests"


def _digest(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _identity(value) -> str:
    return _digest(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )


def _relative_name(value: str) -> str:
    """Accept a canonical, portable relative file name, never a traversal."""
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or ":" in value
        or "\0" in value
    ):
        raise ValueError(f"Unsafe archive/source path: {value!r}")
    name = PurePosixPath(value)
    if (
        name.is_absolute()
        or value in (".", "..")
        or name.as_posix() != value
        or any(part in ("", ".", "..") for part in name.parts)
    ):
        raise ValueError(f"Unsafe archive/source path: {value!r}")
    return value


def _inside(base: Path, name: str) -> Path:
    path = (base / _relative_name(name)).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError(f"File escapes its permitted directory: {name}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _hash_map(value, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Missing {label} file hashes")
    result = {}
    for name, digest in value.items():
        name = _relative_name(name)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid {label} SHA256 for {name}")
        result[name] = digest
    return result


def _saved_snapshot(value, label: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"Missing measured {label} provenance")
    files = _hash_map(value.get("files"), label)
    if _identity(files) != value.get("sha256"):
        raise ValueError(
            f"Saved {label} aggregate fingerprint does not match its file hashes"
        )
    return files


def _source_inventory() -> set[str]:
    return {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / SOURCE).rglob("*")
        if path.is_file()
        and path.suffix in (".py", ".yaml")
        and "__pycache__" not in path.parts
    }


def collect_files(campaign: dict) -> tuple[dict[str, bytes], dict]:
    """Capture exact measured bytes or reject changed/missing provenance."""
    if campaign.get("completed") is not True:
        raise ValueError("Only a completed campaign can be archived")
    if (
        campaign.get("source_unchanged") is not True
        or campaign.get("harness_unchanged") is not True
    ):
        raise ValueError(
            "Campaign must attest unchanged source and harness at completion"
        )
    provenance = campaign.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Missing campaign provenance")
    measured_source = _saved_snapshot(provenance.get("source"), "source")
    for name in measured_source:
        if (
            not name.startswith(SOURCE + "/")
            or PurePosixPath(name).suffix not in (".py", ".yaml")
            or "__pycache__" in PurePosixPath(name).parts
        ):
            raise ValueError(f"Unexpected measured source path: {name}")
    actual_names = _source_inventory()
    if actual_names != set(measured_source):
        raise ValueError(
            "Live source inventory differs from the measured campaign; "
            f"added={sorted(actual_names-set(measured_source))}, "
            f"missing={sorted(set(measured_source)-actual_names)}"
        )
    measured_harness = _saved_snapshot(provenance.get("harness"), "harness")
    missing_harness = set(HARNESS) - set(measured_harness)
    if missing_harness:
        raise ValueError(
            "Measured harness omits required campaign runtime files: "
            f"{sorted(missing_harness)}"
        )

    blobs: dict[str, bytes] = {}
    inventory: dict[str, dict] = {}

    def capture(base, relative, *, expected=None, role):
        path = _inside(base, relative)
        name = (base.relative_to(ROOT) / relative).as_posix()
        _relative_name(name)
        raw = path.read_bytes()
        digest = _digest(raw)
        if expected is not None and digest != expected:
            raise ValueError(f"File changed since its recorded measurement: {name}")
        if name in blobs:
            raise ValueError(f"Duplicate archive entry: {name}")
        blobs[name] = raw
        inventory[name] = {"sha256": digest, "bytes": len(raw), "role": role}

    for name, digest in sorted(measured_source.items()):
        capture(
            ROOT,
            name,
            expected=digest,
            role="measured production source or physical case",
        )
    for name, digest in sorted(measured_harness.items()):
        capture(STUDY, name, expected=digest, role="measured benchmark harness")

    manifest_relative = "source_manifest.json"
    manifest_path = _inside(FROZEN, manifest_relative)
    manifest_raw = manifest_path.read_bytes()
    manifest_digest = _digest(manifest_raw)
    if manifest_digest != provenance.get("legacy_manifest_sha256"):
        raise ValueError("Frozen source manifest differs from the measured campaign")
    frozen_files = _hash_map(json.loads(manifest_raw), "frozen native")
    if len(frozen_files) != 110:
        raise ValueError(
            f"Expected the preserved 110-file native manifest, found {len(frozen_files)}"
        )
    for name, digest in sorted(frozen_files.items()):
        capture(
            FROZEN, name, expected=digest, role="manifest-verified frozen native file"
        )
    capture(
        FROZEN,
        manifest_relative,
        expected=manifest_digest,
        role="unchanged frozen native checksum manifest",
    )
    for name in ("pyproject.toml", "poetry.lock"):
        capture(ROOT, name, role="dependency manifest captured at archive creation")
    test_name = "tests/test_verification_cases.py"
    if (ROOT / test_name).is_file():
        capture(ROOT, test_name, role="verification test captured at archive creation")

    metadata = {
        "schema_version": 1,
        "purpose": PURPOSE,
        "campaign": {
            "id": campaign.get("campaign_id"),
            "created_at": campaign.get("created_at"),
            "protocol_sha256": campaign.get("protocol_sha256"),
            "git_head": provenance.get("git_head"),
            "source_sha256": provenance["source"]["sha256"],
            "harness_sha256": provenance["harness"]["sha256"],
            "legacy_manifest_sha256": manifest_digest,
        },
        "verification": {
            "production_source_inventory_and_bytes_match_campaign": True,
            "measured_harness_matches_campaign": True,
            "measured_harness_files_verified": len(measured_harness),
            "frozen_native_manifest_files_verified": len(frozen_files),
        },
        "notes": [
            "Archive paths are relative to the repository root.",
            "Dependency manifests and the optional verification test were captured at archive creation; their campaign-time hashes were not recorded.",
            "This manifest lists every other archive member; its own SHA256 is covered by the campaign artifact descriptor's archive SHA256.",
            "Saved numerical results and environment provenance remain in summary.json and the report evidence archive.",
        ],
        "files": inventory,
    }
    blobs[MANIFEST_NAME] = (
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    return blobs, metadata


def archive_bytes(files: dict[str, bytes]) -> bytes:
    """Generate a reproducible ZIP with fixed dates, modes and entry ordering."""
    memory = BytesIO()
    with zipfile.ZipFile(
        memory, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name, raw in sorted(files.items()):
            _relative_name(name)
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, raw, compresslevel=6)
    return memory.getvalue()


def _write_immutable(path: Path, raw: bytes) -> None:
    """Never replace an archive that already exists, even with matching bytes."""
    if path.exists():
        if not path.is_file() or path.read_bytes() != raw:
            raise FileExistsError(
                f"Existing archive differs; it was left unchanged: {path}"
            )
        return
    try:
        stream = path.open("xb")
    except FileExistsError:
        if not path.is_file() or path.read_bytes() != raw:
            raise FileExistsError(
                f"Existing archive differs; it was left unchanged: {path}"
            )
        return
    try:
        with stream:
            stream.write(raw)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _attach(path: Path, original: bytes, campaign: dict, descriptor: dict) -> None:
    declarations = campaign.get("artifacts", [])
    if not isinstance(declarations, list):
        raise ValueError("Campaign artifacts must be a list")
    matches = [
        item
        for item in declarations
        if isinstance(item, dict) and item.get("file") == descriptor["file"]
    ]
    if matches:
        if len(matches) != 1 or matches[0] != descriptor:
            raise ValueError(
                "Existing campaign artifact declaration conflicts with this immutable archive"
            )
        return
    updated = dict(campaign, artifacts=[*declarations, descriptor])
    raw = (json.dumps(updated, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if path.read_bytes() != original:
        raise RuntimeError(
            "Campaign changed while preparing its artifact; summary was left unchanged"
        )
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def create_snapshot(
    input_path: Path, output_path: Path | None = None, *, attach: bool = False
) -> dict:
    input_path = (
        input_path if input_path.is_absolute() else STUDY / input_path
    ).resolve()
    if (
        not input_path.is_relative_to(STUDY.resolve())
        or input_path.suffix.lower() != ".json"
    ):
        raise ValueError("Input must be a campaign JSON inside benchmarking/")
    original = input_path.read_bytes()
    campaign = json.loads(original)
    if not isinstance(campaign, dict):
        raise ValueError("Campaign summary must be a JSON object")
    if output_path is None:
        output_path = input_path.parent / "source_and_cases.zip"
    else:
        output_path = output_path if output_path.is_absolute() else STUDY / output_path
    output_path = output_path.resolve()
    if (
        not output_path.is_relative_to(input_path.parent)
        or output_path.suffix.lower() != ".zip"
    ):
        raise ValueError("Snapshot ZIP must remain inside its campaign directory")
    blobs, _ = collect_files(campaign)
    raw = archive_bytes(blobs)
    descriptor = {
        "file": output_path.relative_to(input_path.parent).as_posix(),
        "sha256": _digest(raw),
        "bytes": len(raw),
        "purpose": PURPOSE,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable(output_path, raw)
    if attach:
        _attach(input_path, original, campaign, descriptor)
    return descriptor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("outputs/initial_baseline/summary.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Default: source_and_cases.zip beside the input summary",
    )
    parser.add_argument(
        "--attach",
        action="store_true",
        help="Declare the verified ZIP in summary.artifacts for report embedding",
    )
    args = parser.parse_args()
    descriptor = create_snapshot(args.input, args.output, attach=args.attach)
    print(json.dumps(descriptor, indent=2))


if __name__ == "__main__":
    main()
