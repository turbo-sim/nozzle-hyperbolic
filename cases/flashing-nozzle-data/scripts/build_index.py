"""Concatenate every ``sources/*/cases.csv`` into one master table.

Reads each source folder, validates it against ``SCHEMA.md`` as far as a machine can, and
writes ``index/cases_index.csv``: all case rows, prefixed with the provenance fields from the
source's ``meta.yaml`` and a globally unique ``uid``.

Standard library only, on purpose. This script must keep working when the surrounding project's
environment does not.

Usage
-----
    python scripts/build_index.py
    python scripts/build_index.py --strict          # warnings become a non-zero exit code
    python scripts/build_index.py --out somewhere.csv
"""

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES_DIR = ROOT / "sources"
DEFAULT_OUT = ROOT / "index" / "cases_index.csv"

META_REQUIRED = ["bibkey", "fluid", "reference", "x_origin", "digitizer", "notes"]
META_CARRIED = ["bibkey", "fluid", "doi", "report_id", "reference", "x_origin"]
CASE_REQUIRED = ["case_id", "nozzle_id", "fluid", "p0_Pa", "T0_K", "status"]
CASE_NUMERIC = [
    "p0_Pa",
    "T0_K",
    "subcooling_K",
    "p_back_Pa",
    "p_exit_Pa",
    "mdot_kg_s",
    "G_throat_kg_m2_s",
]
STATUS_VALUES = {"verified", "provisional", "placeholder"}
ID_RE = re.compile(r"^[a-z0-9_]+$")


# ----------------------------------------------------------------------------------------------
# readers
# ----------------------------------------------------------------------------------------------


def read_csv_with_comments(path):
    """Read a CSV, skipping the ``#`` comment lines that may precede the header.

    See SCHEMA.md: comments are allowed before the header only. A ``#`` line after the header is
    reported as a problem rather than silently dropped, because a naive reader would lose a row.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.strip() and not line.lstrip().startswith("#"):
            header_idx = i
            break
    if header_idx is None:
        return [], [], ["file has no header row"]

    body = lines[header_idx:]
    problems = []
    for i, line in enumerate(body):
        if line.lstrip().startswith("#"):
            problems.append(
                f"comment line after the header (line {header_idx + i + 1}); "
                "comments are only allowed before the header"
            )
    reader = csv.DictReader(body)
    return reader.fieldnames or [], list(reader), problems


def read_flat_yaml(path):
    """Parse the deliberately small subset of YAML that ``meta.yaml`` is allowed to use.

    Top-level ``key: value`` scalars, plus ``key: |`` block scalars for ``notes``. Anything
    else is ignored. This exists so that PyYAML is not a dependency of the archive.
    """
    data = {}
    key = None
    block = None

    for raw in path.read_text(encoding="utf-8").splitlines():
        if block is not None:
            # inside a block scalar: blank or indented lines belong to it
            if not raw.strip() or raw[:1] in (" ", "\t"):
                block.append(raw.strip())
                continue
            data[key] = "\n".join(block).strip()
            block = None

        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if raw[:1] in (" ", "\t"):  # nested content we do not read
            continue

        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$", raw)
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip()
        if value in ("|", ">", "|-", ">-"):
            block = []
            continue
        data[key] = value.strip('"').strip("'")

    if block is not None:
        data[key] = "\n".join(block).strip()
    return data


# ----------------------------------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------------------------------


def check_source(source_dir, seen_uids):
    """Validate one ``sources/<bibkey>/`` folder. Returns ``(rows, errors, warnings)``."""
    errors, warnings, rows = [], [], []
    rel = source_dir.name

    meta_path = source_dir / "meta.yaml"
    cases_path = source_dir / "cases.csv"
    if not meta_path.exists():
        return rows, [f"{rel}: missing meta.yaml"], warnings
    if not cases_path.exists():
        return rows, [f"{rel}: missing cases.csv"], warnings

    meta = read_flat_yaml(meta_path)
    for field in META_REQUIRED:
        if not meta.get(field):
            errors.append(f"{rel}/meta.yaml: required field '{field}' is missing or empty")
    if meta.get("bibkey") and meta["bibkey"] != source_dir.name:
        errors.append(f"{rel}/meta.yaml: bibkey '{meta['bibkey']}' does not match the folder name")
    if not meta.get("doi") and not meta.get("report_id"):
        warnings.append(f"{rel}/meta.yaml: neither doi nor report_id is set; source is hard to find")

    fieldnames, case_rows, problems = read_csv_with_comments(cases_path)
    errors += [f"{rel}/cases.csv: {p}" for p in problems]
    for column in CASE_REQUIRED:
        if column not in fieldnames:
            errors.append(f"{rel}/cases.csv: required column '{column}' is missing")
    if not case_rows:
        warnings.append(f"{rel}/cases.csv: no data rows")

    for line_no, row in enumerate(case_rows, start=2):
        case_id = (row.get("case_id") or "").strip()
        where = f"{rel}/cases.csv row {line_no} ({case_id or '<no case_id>'})"

        if not case_id:
            errors.append(f"{where}: empty case_id")
            continue
        if not ID_RE.match(case_id):
            errors.append(f"{where}: case_id must match [a-z0-9_]+")

        uid = f"{source_dir.name}__{case_id}"
        if uid in seen_uids:
            errors.append(f"{where}: duplicate case_id within the source")
        seen_uids.add(uid)

        status = (row.get("status") or "").strip()
        if status not in STATUS_VALUES:
            errors.append(f"{where}: status '{status}' not in {sorted(STATUS_VALUES)}")
        if status == "verified" and not meta.get("verified_by"):
            errors.append(f"{where}: status=verified but meta.yaml has no verified_by")
        if status == "placeholder":
            warnings.append(f"{where}: placeholder row, not usable for validation")

        for column in CASE_NUMERIC:
            value = (row.get(column) or "").strip()
            if not value:
                continue
            try:
                float(value)
            except ValueError:
                errors.append(f"{where}: column '{column}' value '{value}' is not a number")

        nozzle_id = (row.get("nozzle_id") or "").strip()
        if nozzle_id and not (source_dir / "geometry" / f"{nozzle_id}.csv").exists():
            errors.append(f"{where}: no geometry file geometry/{nozzle_id}.csv")

        for column in ("profile_p", "profile_T"):
            stem = (row.get(column) or "").strip()
            if stem and not (source_dir / "profiles" / f"{stem}.csv").exists():
                errors.append(f"{where}: {column} points at a missing file profiles/{stem}.csv")

        if row.get("fluid") and meta.get("fluid") and row["fluid"] != meta["fluid"]:
            warnings.append(
                f"{where}: fluid '{row['fluid']}' differs from meta.fluid '{meta['fluid']}'"
            )

        out = {"uid": uid}
        out.update({k: meta.get(k, "") for k in META_CARRIED})
        out.update({k: (v or "").strip() for k, v in row.items() if k})
        rows.append(out)

    for geometry in sorted((source_dir / "geometry").glob("*.csv")):
        _, geo_rows, geo_problems = read_csv_with_comments(geometry)
        errors += [f"{rel}/geometry/{geometry.name}: {p}" for p in geo_problems]
        xs = [float(r["x_m"]) for r in geo_rows if (r.get("x_m") or "").strip()]
        if xs != sorted(xs):
            errors.append(f"{rel}/geometry/{geometry.name}: x_m is not monotonically increasing")

    return rows, errors, warnings


# ----------------------------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the master case index.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output CSV path")
    parser.add_argument("--strict", action="store_true", help="exit non-zero on warnings too")
    args = parser.parse_args(argv)

    source_dirs = sorted(d for d in SOURCES_DIR.iterdir() if d.is_dir())
    if not source_dirs:
        print(f"no source folders under {SOURCES_DIR}", file=sys.stderr)
        return 1

    all_rows, all_errors, all_warnings = [], [], []
    seen_uids = set()
    for source_dir in source_dirs:
        rows, errors, warnings = check_source(source_dir, seen_uids)
        all_rows += rows
        all_errors += errors
        all_warnings += warnings
        print(
            f"{source_dir.name:<20} {len(rows):>4} cases, "
            f"{len(errors)} errors, {len(warnings)} warnings"
        )

    for warning in all_warnings:
        print(f"  WARNING  {warning}")
    for error in all_errors:
        print(f"  ERROR    {error}", file=sys.stderr)

    if all_errors:
        print(f"\n{len(all_errors)} error(s); index not written.", file=sys.stderr)
        return 1

    # union of columns, with the identity and provenance fields first and in a fixed order
    lead = ["uid"] + META_CARRIED
    rest = sorted({k for row in all_rows for k in row} - set(lead))
    columns = lead + rest

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    by_status = {}
    for row in all_rows:
        status = row.get("status", "")
        by_status[status] = by_status.get(status, 0) + 1
    summary = ", ".join(f"{n} {s}" for s, n in sorted(by_status.items()))
    print(f"\nwrote {args.out} -- {len(all_rows)} cases ({summary})")

    if all_warnings and args.strict:
        print(f"{len(all_warnings)} warning(s) with --strict.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
