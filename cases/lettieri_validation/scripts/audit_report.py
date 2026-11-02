"""Offline structural audit of report navigation, images and embedded data."""

import ast
import base64
from html.parser import HTMLParser
import json
import re

from .data import ROOT


class Document(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.links, self.images, self.external_assets = set(), [], [], []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if "id" in values:
            assert values["id"] not in self.ids, "Duplicate navigation target"
            self.ids.add(values["id"])
        if tag == "a":
            self.links.append(values)
        if tag == "img":
            self.images.append(values)
        if tag in ("script", "link"):
            self.external_assets.append(values)


def main():
    page = Document()
    page.feed((ROOT / "REPORT.html").read_text(encoding="utf-8"))
    assert not page.external_assets, "Standalone report must not load external assets"
    for link in page.links:
        target = link["href"]
        if target.startswith("#"):
            assert target[1:] in page.ids
        if "download" in link:
            prefix = "data:application/json;base64,"
            assert target.startswith(prefix)
            payload = json.loads(base64.b64decode(target[len(prefix):]))
            assert len(payload) == len(list((ROOT / "outputs").glob("*/result.json")))
            assert {row["case_id"] for row in payload} == {"L1", "L2", "L3", "L4", "L5"}
            assert all("history" in row or row["reason"] == "exception" for row in payload)
    assert len(page.images) == 3
    for picture in page.images:
        prefix = "data:image/png;base64,"
        assert picture["src"].startswith(prefix) and picture.get("alt")
        assert base64.b64decode(picture["src"][len(prefix):]).startswith(b"\x89PNG\r\n\x1a\n")
    for document in ROOT.glob("*.md"):
        text = document.read_text(encoding="utf-8")
        assert "\\(" not in text and "\\[" not in text
        assert "$\nho" not in text, "Escaped Python carriage return corrupted a math symbol"
        for path in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):
            assert (ROOT / path).exists()
    for script in (ROOT / "scripts").glob("*.py"):
        ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    print("Report assets, navigation targets, JSON download and Markdown math passed structural checks.")


if __name__ == "__main__":
    main()
