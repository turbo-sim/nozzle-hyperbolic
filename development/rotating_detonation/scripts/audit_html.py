"""Check offline HTML packaging, navigation targets and compressed download."""

import base64
import gzip
from html.parser import HTMLParser
import json
from pathlib import Path
import re

STUDY = Path(__file__).resolve().parents[1]


class AuditParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.tabs = []
        self.external = []
        self.text = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.append(a["id"])
        if "data-tab" in a:
            self.tabs.append(a["data-tab"])
        if (
            tag in ("script", "img", "link")
            and ("src" in a or "href" in a)
            and not a.get("src", a.get("href", "")).startswith("data:")
        ):
            self.external.append(a)

    def handle_data(self, data):
        self.text.append(data)


def main():
    text = (STUDY / "REPORT.html").read_text(encoding="utf-8")
    p = AuditParser()
    p.feed(text)
    assert len(p.ids) == len(set(p.ids)), "Duplicate IDs"
    assert set(p.tabs) == {
        "geometry",
        "steady",
        "transient",
        "mean",
        "timing",
        "theory",
    }
    assert set(p.tabs).issubset(p.ids)
    assert not p.external, "Report has external rendering dependencies"
    own = text.split("<script>const campaign=")[1].split("</script>")[0]
    archive = re.search(r'const archive="([A-Za-z0-9+/=]+)";', own).group(1)
    data = json.loads(gzip.decompress(base64.b64decode(archive)))
    saved = json.loads((STUDY / "outputs/results.json").read_text())
    analysis = data.pop("report_analysis", {})
    assert data == saved, "Embedded download differs from evidence"
    if "mean_comparison" in analysis:
        import numpy as np

        mean = analysis["mean_comparison"]
        samples = mean["averaging_samples"]
        times = np.array(samples["times"])
        for field, values in samples["fields"].items():
            calculated = np.trapezoid(np.array(values), times, axis=0) / (
                times[-1] - times[0]
            )
            assert np.allclose(
                calculated, mean["mean"][field], rtol=1e-12, atol=1e-12
            ), "Embedded periodic mean mismatch"
    specs = json.loads(own.split("const plots=")[1].split(";\nconst config=")[0])
    assert all(spec["id"] in p.ids for spec in specs)
    assert all(
        len(trace["x"]) == len(trace["y"]) for spec in specs for trace in spec["data"]
    )
    assert "frame" in p.ids and "download" in p.ids and "theme" in p.ids
    result = dict(
        passed=True,
        plot_count=len(specs),
        bytes=len(text.encode()),
        embedded_download_matches=True,
        external_rendering_dependencies=0,
        browser_visual_check="Unavailable: computer-use URL verification stopped UI automation",
    )
    (STUDY / "outputs/html_audit.json").write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    main()
