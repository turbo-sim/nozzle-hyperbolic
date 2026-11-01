"""Offline presentation utilities shared by the three flux-method reports.

Only rendering lives here. Source notes and numerical evidence are inputs,
never rewritten by the HTML presentation layer.
"""
from __future__ import annotations

import gzip
from hashlib import sha256
from html import escape, unescape
from pathlib import Path
import re
import unicodedata

from markdown_it import MarkdownIt

ASSETS = Path(__file__).resolve().parent / "assets"
MATHJAX_SHA256 = "d4295dc33744836935c1399feece5159577b34c5c8ffb9f1c6324cd82e03a882"
MATHJAX_LICENSE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"


def theme_css():
    return (ASSETS / "report_theme.css").read_text(encoding="utf-8")


def theme_script():
    return (ASSETS / "report_theme.js").read_text(encoding="utf-8")


def math_runtime():
    """Inline the byte-verified existing SVG math runtime and its license.

    The selected packages are inside this bundle. No autoload/require package
    is enabled, and menus are disabled, so typesetting needs no network/fonts.
    """
    runtime = gzip.decompress((ASSETS / "vendor/mathjax-3.2.2-tex-svg.js.gz").read_bytes())
    license_raw = (ASSETS / "vendor/MATHJAX_LICENSE.txt").read_bytes()
    if sha256(runtime).hexdigest() != MATHJAX_SHA256:
        raise ValueError("Vendored MathJax runtime failed its integrity check")
    if sha256(license_raw).hexdigest() != MATHJAX_LICENSE_SHA256:
        raise ValueError("Vendored MathJax license failed its integrity check")
    config = r'''window.MathJax={
      tex:{inlineMath:[['$','$']],displayMath:[['$$','$$']],
        packages:['base','ams','newcommand','noundefined','configmacros'],
        macros:{texttt:['\\mathtt{#1}',1]}},
      svg:{fontCache:'local'},options:{enableMenu:false},
      startup:{typeset:true}
    };'''
    safe_runtime = runtime.decode("utf-8").replace("</script", "<\\/script")
    return ('<script id="mathjax-config">' + config + '</script>'
            '<script id="mathjax-runtime">' + safe_runtime + '</script>'
            '<script id="mathjax-license" type="text/plain">'
            + license_raw.decode("utf-8").replace("</script", "<\\/script") + '</script>')


def render_document(source, prefix="theory"):
    """Render exact Markdown prose with protected dollar-delimited equations.

    Markdown is parsed by the same installed parser as the architecture guide.
    Equations stay exact LaTeX for the embedded MathJax renderer. Relative
    repository links are annotated references, so a copied HTML has no broken
    dependencies on neighboring Markdown files.
    """
    protected = []

    def token(html):
        marker = f"FLUXPRESENTATIONTOKEN{len(protected)}END"
        protected.append((marker, html))
        return marker

    # Inline/code-fence contents are excluded from dollar parsing. Preserve
    # their Markdown syntax for the regular renderer rather than raw HTML.
    code = []
    def code_token(match):
        marker=f"FLUXCODETOKEN{len(code)}END"
        code.append((marker,match[0]))
        return marker
    text = re.sub(r"^```[^\n]*\n[\s\S]*?^```\s*$", code_token, source, flags=re.M)
    text = re.sub(r"`[^`\n]+`", code_token, text)

    def math(match, block=False):
        tag, cls, delimiter = ("div", "math-display", "$$") if block else ("span", "math-inline", "$")
        value=f'<{tag} class="{cls}">{delimiter}{escape(match[1])}{delimiter}</{tag}>'
        marker=token(value)
        return f"\n\n{marker}\n\n" if block else marker

    text = re.sub(r"\$\$([\s\S]*?)\$\$", lambda m: math(m,True), text)
    text = re.sub(r"(?<![\\$])\$(?!\$)([^\n$]+?)\$(?!\$)", math, text)
    for marker, original in code:
        text=text.replace(marker,original)
    body = MarkdownIt("commonmark", {"html":False}).enable("table").render(text)
    for marker, value in protected:
        body=body.replace(f"<p>{marker}</p>",value).replace(marker,value)

    headings=[]
    used=set()
    def heading(match):
        level=int(match[1]);label=re.sub(r"<[^>]*>","",unescape(match[2]))
        slug=unicodedata.normalize("NFKD",label).encode("ascii","ignore").decode().lower()
        slug=re.sub(r"[^a-z0-9]+","-",slug).strip("-") or "section"
        target=f"{prefix}-{slug}";counter=2
        while target in used:
            target=f"{prefix}-{slug}-{counter}";counter+=1
        used.add(target)
        headings.append(dict(id=target,text=label,title=label,level=level))
        return f'<h{level} id="{target}">{match[2]}</h{level}>'
    body=re.sub(r"<h([1-6])>([\s\S]*?)</h\1>",heading,body)

    def link(match):
        href=unescape(match[1])
        if href.startswith(("https://","http://","mailto:","#")):
            return match[0]
        return f'<span class="repository-reference" title="Repository source: {escape(href,quote=True)}">{match[2]}</span>'
    body=re.sub(r'<a href="([^"]+)"[^>]*>([\s\S]*?)</a>',link,body)
    return dict(html='<div class="rendered-document math-content">'+body+'</div>',headings=headings)
