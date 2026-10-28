"""Render retained study notes with protected dollar-delimited LaTeX."""

import html
from pathlib import Path
import re
import markdown

STUDY = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / "report_assets"


def document(name, prefix):
    """Typeset preserved prose/math, embedding images and resolving local links."""
    source = (ASSETS / "notes" / name).read_text(encoding="utf-8")
    math = []

    def equation(match, block=False):
        key = f"LATEXPLACEHOLDER{len(math)}END"
        tag, cls, delimiters = (
            ("div", "math-display", "$$") if block else ("span", "math-inline", "$")
        )
        math.append(
            (
                key,
                f'<{tag} class="{cls}">{delimiters}{html.escape(match[1])}{delimiters}</{tag}>',
            )
        )
        return f"\n\n{key}\n\n" if block else key

    protected = re.sub(r"\$\$([\s\S]*?)\$\$", lambda m: equation(m, True), source)
    protected = re.sub(r"(?<![\\$])\$(?!\$)([^\n$]+?)\$(?!\$)", equation, protected)
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists"])
    body = re.sub(
        r"^<h1[^>]*>.*?</h1>\s*", "", md.convert(protected), count=1, flags=re.S
    )
    for key, value in math:
        body = body.replace(f"<p>{key}</p>", value).replace(key, value)
    ids = set(re.findall(r'\bid="([^"]+)"', body))
    body = re.sub(r'\bid="([^"]+)"', lambda m: f'id="{prefix}-{m[1]}"', body)

    def link(match):
        href, label = html.unescape(match[1]), match[2]
        if href.startswith(("https://", "http://")):
            return match[0]
        if href.startswith("#") and href[1:] in ids:
            return f'<a href="#{prefix}-{href[1:]}">{label}</a>'
        tab = {
            "PHYSICS_AND_BOUNDARIES.md": "theory",
            "report_backpressure_sensitivity.html": "results",
            "REPORT.html": "results",
            "BOUNDARY_DESIGN_RECOMMENDATIONS.md": "recommendations",
            "REPORT.md": "results",
            "README.md": "results",
        }.get(Path(href).name)
        if tab:
            return f'<a href="#{tab}">{label}</a>'
        return f'<span title="Repository provenance: {html.escape(href, quote=True)}">{label} <code>(repository artifact)</code></span>'

    body = re.sub(r'<a href="([^"]+)"[^>]*>(.*?)</a>', link, body, flags=re.S)

    def image(match):
        path = (STUDY / html.unescape(match[1])).resolve()
        if path.is_file() and path.suffix.lower() == ".png":
            import base64

            return (
                '<img class="archive-image" alt="Retained historical figure" src="data:image/png;base64,'
                + base64.b64encode(path.read_bytes()).decode()
                + '">'
            )
        return (
            '<p class="small">Historical figure path: <code>'
            + html.escape(match[1])
            + "</code>. See the interactive results above.</p>"
        )

    body = re.sub(r'<img[^>]*src="([^"]+)"[^>]*>', image, body)
    headings = [
        (target, re.sub("<.*?>", "", label))
        for target, label in re.findall(r'<h2 id="([^"]+)">(.*?)</h2>', body)
    ]
    return body, headings
