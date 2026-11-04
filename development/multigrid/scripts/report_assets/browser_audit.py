"""Exercise an instrumented copy of the saved HTML in isolated headless Chrome."""
from __future__ import annotations

import argparse
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


class ResultParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.collect = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "pre" and dict(attrs).get("id") == "browser-qa-result":
            self.collect = True

    def handle_endtag(self, tag):
        if tag == "pre":
            self.collect = False

    def handle_data(self, text):
        if self.collect:
            self.parts.append(text)


def chrome_path():
    candidates = [
        Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
        Path("C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"),
    ]
    found = shutil.which("chrome") or shutil.which("google-chrome")
    if found:
        candidates.insert(0, Path(found))
    return next((candidate for candidate in candidates if candidate.is_file()), None)


from browser_visual import render_screenshots


def audit(report, output_dir, chrome=None, screenshots=False):
    report = Path(report).resolve()
    folder = Path(output_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    browser = Path(chrome) if chrome else chrome_path()
    if browser is None or not browser.is_file():
        raise FileNotFoundError("Chrome executable was not found; pass --chrome.")
    original = report.read_bytes()
    report_hash = sha256(original).hexdigest()
    script = Path(__file__).with_name("browser_audit.js").read_text(encoding="utf-8")
    injection = (
        "<script>window.BROWSER_QA_REPORT_SHA256=" + json.dumps(report_hash)
        + ";\n" + script + "\n</script>\n"
    )
    html = original.decode("utf-8")
    if "</body>" not in html:
        raise ValueError("Report body was not found.")
    instrumented = folder / "browser_qa.html"
    instrumented.write_text(html.replace("</body>", injection + "</body>"), encoding="utf-8")
    profile = tempfile.mkdtemp(prefix="chrome-controls-profile-", dir=folder)
    command = [
        str(browser), "--headless=new", "--disable-gpu", "--no-first-run",
        "--no-default-browser-check", "--disable-default-apps", "--disable-extensions",
        "--disable-background-networking", "--disable-component-update", "--disable-sync",
        "--allow-file-access-from-files", "--window-size=1440,1100",
        "--virtual-time-budget=180000", "--timeout=240000", "--dump-dom",
        "--user-data-dir=" + profile, instrumented.as_uri() + "#explorer",
    ]
    dom_file = folder / "browser_qa_dom.html"
    stderr_file = folder / "browser_qa_chrome.stderr.log"
    with dom_file.open("wb") as stdout, stderr_file.open("wb") as stderr:
        completed = subprocess.run(
            command, stdout=stdout, stderr=stderr, timeout=300,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False,
        )
    parser = ResultParser()
    parser.feed(dom_file.read_text(encoding="utf-8", errors="replace"))
    try:
        result = json.loads("".join(parser.parts))
    except json.JSONDecodeError as error:
        raise RuntimeError("Chrome did not emit a readable audit result; inspect " + str(dom_file)) from error
    result.update({
        "chrome_exit_code": completed.returncode,
        "instrumented_copy_sha256": sha256(instrumented.read_bytes()).hexdigest(),
        "report_unchanged": report.read_bytes() == original,
        "isolated_profile": profile,
    })
    result["passed"] = (
        result.get("passed") is True and result["report_unchanged"]
        and result.get("report_sha256") == report_hash and completed.returncode == 0
    )
    result_file = folder / "browser_controls_audit.json"
    result_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if screenshots and result["passed"]:
        render_screenshots(report, folder, browser)
    print(json.dumps({key: result.get(key) for key in (
        "passed", "status", "report_sha256", "actual_browser", "real_plotly",
        "real_mathjax", "groups", "checks", "downloads", "network_requests",
        "report_unchanged", "chrome_exit_code", "errors",
    )}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chrome", type=Path)
    parser.add_argument("--screenshots", action="store_true")
    args = parser.parse_args()
    raise SystemExit(0 if audit(args.report, args.output_dir, args.chrome, args.screenshots)["passed"] else 1)
