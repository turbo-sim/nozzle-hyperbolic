"""Run the full source test suite and retain source, command and JUnit evidence."""

from hashlib import sha256
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import xml.etree.ElementTree as ET

from .common import ROOT, STUDY, provenance, finish, save


def main():
    output = STUDY / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    log, xml = output / "production_tests.log", output / "production_tests.xml"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests",
        f"--junitxml={xml}",
    ]
    data = dict(
        provenance=provenance(),
        command=command,
        audit_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        passed=False,
    )
    start = perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
    data.update(returncode=result.returncode, elapsed_s=perf_counter() - start)
    if xml.exists():
        suites = list(ET.parse(xml).getroot().iter("testsuite"))
        counts = {
            name: sum(int(suite.get(name, 0)) for suite in suites)
            for name in ("tests", "failures", "errors", "skipped")
        }
        counts["passed"] = (
            counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
        )
        data["counts"] = counts
    finish(data)
    data["passed"] = bool(
        result.returncode == 0 and data["provenance"]["source_unchanged"]
    )
    save(output / "production_tests_provenance.json", data)
    print(log.read_text(encoding="utf-8"))
    assert data["passed"], "Production tests failed or source changed during the tests"


if __name__ == "__main__":
    main()
