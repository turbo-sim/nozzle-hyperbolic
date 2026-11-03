"""Build the offline, accessible browser companion to the architecture map.

The SVG and Markdown remain the canonical, independently useful documents.
This renderer embeds both without a web server or external browser resources.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
from html import escape
import json
from pathlib import Path
import re
import unicodedata

from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin


def _slug(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", normalized.lower()).strip("-") or "section"


def _render_markdown(markdown: str) -> str:
    renderer = MarkdownIt("commonmark", {"html": False}).enable("table")
    # Parse math before Markdown can consume TeX escapes, subscripts or emphasis.
    # The plugin also keeps dollar signs in fenced/inline code untouched.
    def preserve_math(content: str, options: dict[str, bool]) -> str:
        delimiter = "$$" if options["display_mode"] else "$"
        return f"{delimiter}{escape(content, quote=False)}{delimiter}"

    renderer.use(dollarmath_plugin, allow_labels=False, renderer=preserve_math)
    tokens = renderer.parse(markdown)
    counts: dict[str, int] = {}
    for index, token in enumerate(tokens):
        # Keep the guide's introductory links useful in a copied standalone HTML.
        # Source-code links deliberately remain relative repository navigation.
        for child in token.children or ():
            if child.type != "link_open":
                continue
            if child.attrGet("href") == "ARCHITECTURE.svg":
                child.attrSet("href", "__SVG_DOWNLOAD__")
                child.attrSet("download", "ARCHITECTURE.svg")
            elif child.attrGet("href") == "ARCHITECTURE.html":
                child.attrSet("href", "#architecture-map-view")
                child.attrSet("data-architecture-map", "")
        if token.type == "heading_open" and index + 1 < len(tokens):
            base = _slug(tokens[index + 1].content)
            count = counts.get(base, 0)
            counts[base] = count + 1
            token.attrSet("id", base if count == 0 else f"{base}-{count + 1}")
    return renderer.renderer.render(tokens, renderer.options, {})


def _download_uri(mime: str, value: str) -> str:
    encoded = base64.b64encode(value.encode("utf-8")).decode("ascii")
    return f"data:{mime};charset=utf-8;base64,{encoded}"


def _mathjax_assets() -> tuple[str, str]:
    """Read the locally vendored, integrity-checked runtime and Apache license."""
    assets = Path(__file__).resolve().parents[1] / "_assets"
    provenance = json.loads((assets / "mathjax-source.json").read_text(encoding="utf-8"))
    values: list[str] = []
    for name in ("mathjax-3.2.2-tex-svg.js", "MATHJAX_LICENSE.txt"):
        raw = (
            gzip.decompress((assets / f"{name}.gz").read_bytes())
            if name.endswith(".js")
            else (assets / name).read_bytes()
        )
        expected = provenance[name]
        if len(raw) != expected["bytes"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
            raise ValueError(f"Vendored documentation asset failed its integrity check: {name}")
        values.append(raw.decode("utf-8"))
    # Escape a script end tag even if a future verified bundle contains one.
    return re.sub(r"</script", r"<\/script", values[0], flags=re.IGNORECASE), values[1]


def render_viewer(svg: str, markdown: str) -> str:
    """Return a standalone HTML viewer containing the supplied map and guide."""
    # An XML declaration is useful in the standalone SVG, but invalid inside HTML.
    inline_svg = re.sub(r"^\s*<\?xml[^>]*\?>\s*", "", svg)
    inline_svg = re.sub(r"<!DOCTYPE[^>]*>\s*", "", inline_svg, flags=re.IGNORECASE)
    mathjax, license_text = _mathjax_assets()
    return (
        _TEMPLATE.replace("<!-- ARCHITECTURE_SVG -->", inline_svg)
        .replace("<!-- ARCHITECTURE_GUIDE -->", _render_markdown(markdown))
        .replace("__SVG_DOWNLOAD__", _download_uri("image/svg+xml", svg))
        .replace("__MD_DOWNLOAD__", _download_uri("text/markdown", markdown))
        .replace("__MATHJAX_LICENSE_DOWNLOAD__", _download_uri("text/plain", license_text))
        .replace("<!-- ARCHITECTURE_MATHJAX -->", mathjax)
    )


_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="color-scheme" content="light">
  <meta name="description" content="Interactive architecture map and source guide for nozzle-hyperbolic. Fully offline, with downloadable vector map and Markdown guide.">
  <title>nozzle-hyperbolic · Package architecture</title>
  <style>
    :root{--ink:#132b42;--muted:#5a6c7c;--line:#dbe4eb;--paper:#fff;--wash:#f3f6f9;--blue:#1769a6;--blue-wash:#e9f3fb;--radius:14px;--shadow:0 5px 24px #15364c0a;font-family:Inter,"Segoe UI",Arial,sans-serif;color:var(--ink);background:var(--wash);font-synthesis:none}
    *{box-sizing:border-box}body{margin:0}button,a{-webkit-tap-highlight-color:transparent}button{font:inherit}a{color:var(--blue);text-underline-offset:3px}button:focus-visible,a:focus-visible,[tabindex]:focus-visible{outline:3px solid #d69a2a;outline-offset:4px}button{cursor:pointer}button:disabled{cursor:default;opacity:.45}[hidden]{display:none!important}
    .skip-link{position:absolute;left:16px;top:-60px;z-index:50;padding:12px 18px;background:var(--ink);color:white;border-radius:8px}.skip-link:focus{top:12px}
    .page-header{display:flex;align-items:center;justify-content:space-between;gap:25px;max-width:1920px;margin:auto;padding:25px 32px 20px}.identity{display:flex;align-items:center;gap:14px}.brand{width:43px;height:43px;display:grid;place-items:center;background:var(--ink);border-radius:12px;color:white;font-size:14px;font-weight:750;letter-spacing:-.7px;flex-shrink:0}.eyebrow{text-transform:uppercase;letter-spacing:1.8px;color:var(--muted);font-size:10px;font-weight:750;margin:0 0 5px}.page-header h1{font-size:24px;line-height:1.15;letter-spacing:-.6px;margin:0;font-weight:720}.subtitle{margin:7px 0 0;font-size:12px;color:var(--muted);line-height:1.5}.header-actions{display:flex;align-items:center;gap:14px;flex-shrink:0}.tabs{display:flex;padding:4px;border:1px solid var(--line);background:#e9eff4;border-radius:10px;gap:4px}.tab{padding:9px 20px;border:0;background:transparent;border-radius:7px;color:var(--muted);font-size:12px;font-weight:700}.tab[aria-selected=true]{background:white;box-shadow:0 1px 5px #132b4212;color:var(--ink)}.download-wrap{position:relative}.download-button{border:1px solid var(--line);border-radius:9px;background:white;color:var(--ink);font-size:12px;padding:12px 14px;display:flex;gap:9px;align-items:center}.download-menu{position:absolute;right:0;top:calc(100% + 8px);width:205px;background:white;border:1px solid var(--line);border-radius:10px;box-shadow:0 10px 35px #132b4220;padding:6px;z-index:20}.download-menu a{display:block;text-decoration:none;padding:11px;border-radius:6px;font-size:12px;font-weight:600}.download-menu a:hover{background:var(--blue-wash)}
    .workspace{display:grid;grid-template-columns:206px minmax(0,1fr) 260px;gap:16px;max-width:1920px;margin:0 auto;padding:0 32px 26px}.panel{background:var(--paper);border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow);min-width:0}.section-nav{padding:19px 12px;align-self:start;max-height:calc(100vh - 155px);overflow:auto;scrollbar-width:thin}.panel-kicker{font-size:10px;text-transform:uppercase;letter-spacing:1.4px;color:var(--muted);font-weight:750;margin:0 10px 14px}.nav-group{margin:18px 10px 7px;font-size:10px;letter-spacing:1px;text-transform:uppercase;color:#8393a0}.topic-link{display:block;text-align:left;border:0;background:transparent;border-radius:7px;width:100%;padding:9px 10px;color:#40566a;font-size:12px;line-height:1.4}.topic-link:hover{background:var(--wash);color:var(--ink)}.topic-link[aria-current=true]{background:var(--blue-wash);color:var(--blue);font-weight:700}.nav-note{margin:20px 10px 2px;border-top:1px solid var(--line);padding-top:13px;color:var(--muted);font-size:11px;line-height:1.65}
    .map-panel{display:flex;flex-direction:column;overflow:hidden}.map-toolbar{display:flex;align-items:center;justify-content:space-between;gap:12px;border-bottom:1px solid var(--line);padding:12px 16px;min-height:59px}.toolbar-title{font-size:12px;font-weight:700}.toolbar-status{display:block;font-size:10px;color:var(--muted);margin-top:4px;font-weight:400}.zoom-controls{display:flex;align-items:center;gap:4px}.icon-button,.text-button{height:31px;min-width:31px;border:1px solid transparent;background:transparent;border-radius:6px;color:#41596d;font-size:18px;display:inline-flex;align-items:center;justify-content:center}.icon-button:hover,.text-button:hover{background:var(--wash);border-color:var(--line)}.text-button{font-size:11px;padding:0 8px;font-weight:650}.zoom-label{min-width:43px;text-align:center;font-size:10px;color:var(--muted);font-variant-numeric:tabular-nums}.control-divider{height:18px;width:1px;background:var(--line);margin:0 5px}.map-viewport{height:calc(100vh - 215px);min-height:570px;overflow:hidden;position:relative;background:#fbfcfe;background-image:radial-gradient(#d8e3ed 0.7px,transparent .7px);background-size:16px 16px;touch-action:none}.map-viewport>svg{display:block;width:100%;height:100%;cursor:grab;user-select:none;-webkit-user-select:none}.map-viewport.is-panning>svg{cursor:grabbing}.map-viewport [data-topic]{cursor:pointer}.map-viewport [data-topic]:focus-visible{outline:3px solid #d69a2a;outline-offset:4px}.map-viewport [data-topic].is-selected>rect:first-of-type{stroke:var(--blue);stroke-width:3px;filter:drop-shadow(0 0 6px #1769a625)}.map-footer{padding:10px 16px;border-top:1px solid var(--line);display:flex;justify-content:space-between;gap:10px;font-size:10px;line-height:1.5;color:var(--muted)}.key{padding:1px 4px;border:1px solid var(--line);border-radius:3px;background:var(--wash);font-family:inherit;font-size:9px;white-space:nowrap}
    .inspector{align-self:start;overflow:hidden}.inspector-header{padding:20px 19px 16px;background:#f9fbfd;border-bottom:1px solid var(--line)}.inspector .panel-kicker{margin:0 0 10px}.inspector h2{font-size:18px;line-height:1.35;letter-spacing:-.25px;margin:0}.inspector-body{padding:18px 19px}.detail-description{font-size:12px;line-height:1.8;color:#4c6376;white-space:pre-line;margin:0 0 18px}.guide-link{display:inline-flex;align-items:center;gap:10px;background:var(--blue-wash);border:1px solid #d2e5f3;border-radius:7px;color:var(--blue);padding:10px 12px;font-size:11px;font-weight:700}.inspector-hint{padding-top:18px;margin:18px 0 0;border-top:1px solid var(--line);font-size:11px;line-height:1.7;color:var(--muted)}.offline-label{display:flex;align-items:center;gap:7px;font-size:10px;color:var(--muted);padding:15px 19px;border-top:1px solid var(--line)}.offline-dot{height:6px;width:6px;border-radius:50%;background:#4c9675}
    .guide-layout{max-width:1180px;margin:0 auto;padding:0 32px 45px;display:grid;grid-template-columns:220px minmax(0,1fr);gap:25px}.guide-toc{position:sticky;top:20px;align-self:start;padding:20px 16px;max-height:calc(100vh - 40px);overflow:auto}.guide-toc .panel-kicker{margin-left:5px}.guide-toc a{display:block;padding:8px 5px;font-size:12px;line-height:1.45;color:var(--muted);text-decoration:none;border-radius:5px}.guide-toc a:hover{color:var(--blue);background:var(--blue-wash)}.guide-article{padding:33px 40px;font-size:14px;line-height:1.85;min-width:0}.guide-article h1{font-size:28px;line-height:1.25;letter-spacing:-.65px;margin:0 0 18px}.guide-article h2{font-size:20px;line-height:1.4;margin:35px 0 13px;padding-top:20px;border-top:1px solid var(--line);scroll-margin-top:25px;letter-spacing:-.3px}.guide-article h3{font-size:16px;line-height:1.5;margin:26px 0 10px;scroll-margin-top:25px}.guide-article p{margin:12px 0}.guide-article li{padding-left:3px;margin:5px 0}.guide-article ul,.guide-article ol{padding-left:22px}.guide-article code{font-family:"Cascadia Code",Consolas,monospace;font-size:.86em;background:#eef3f7;border-radius:4px;padding:2px 4px;overflow-wrap:anywhere}.guide-article pre{background:#102c41;color:#e4eff6;padding:18px 20px;border-radius:10px;overflow:auto;font-size:12px;line-height:1.7}.guide-article pre code{padding:0;background:none;color:inherit;overflow-wrap:normal}.guide-article table{border-collapse:collapse;width:100%;font-size:12px;line-height:1.7;margin:20px 0;display:block;overflow:auto}.guide-article th{text-align:left;background:#eef4f8;color:#35536b;font-weight:700}.guide-article td,.guide-article th{border:1px solid var(--line);padding:10px 12px;vertical-align:top}.guide-article tr:nth-child(even) td{background:#fafcfd}.guide-article blockquote{border-left:3px solid #7aa8c9;margin:20px 0;padding:4px 18px;background:#f3f8fc;color:#405e75}.guide-article img{max-width:100%;height:auto}.guide-article a{overflow-wrap:anywhere}.guide-article hr{border:0;border-top:1px solid var(--line);margin:30px 0}.guide-article :target{animation:highlight 2s ease-out}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}@keyframes highlight{from{background:#fff1c7}to{background:transparent}}
    .guide-article .math.block{margin:20px 0}.guide-article mjx-container[display="true"]{max-width:100%;overflow-x:auto;overflow-y:hidden;padding:4px 0}.guide-article mjx-container{color:var(--ink)}
    @media(min-width:1600px){.workspace{grid-template-columns:220px minmax(0,1fr) 290px}.inspector-body{padding:20px 22px}.inspector-header{padding:22px}.detail-description{font-size:13px}.map-viewport{min-height:660px}}
    @media(max-width:1150px){.workspace{grid-template-columns:175px minmax(0,1fr);gap:12px;padding:0 20px 22px}.inspector{grid-column:2;display:grid;grid-template-columns:205px minmax(0,1fr)}.inspector-header{border-right:1px solid var(--line);border-bottom:0}.inspector-body{padding:16px 20px}.detail-description{margin-bottom:12px}.inspector-hint{display:none}.offline-label{grid-column:1/-1;padding:10px 19px}.page-header{padding:22px 20px 18px}.map-viewport{min-height:570px;height:calc(100vh - 220px)}}
    @media(max-width:760px){.page-header{align-items:flex-start;flex-direction:column;gap:15px;padding:20px 16px 16px}.page-header h1{font-size:22px}.header-actions{width:100%;justify-content:space-between}.subtitle{max-width:380px}.workspace{display:block;padding:0 12px 20px}.section-nav{display:none}.map-toolbar{padding:10px 12px;gap:5px}.toolbar-title{font-size:11px}.toolbar-status{font-size:9px}.text-button{padding:0 6px}.map-viewport{height:70vh;min-height:470px}.inspector{display:block;margin-top:13px}.inspector-header{border-right:0;border-bottom:1px solid var(--line);padding:16px 18px}.inspector h2{font-size:17px}.map-footer{font-size:9px;padding:9px 12px}.keyboard-hint{display:none}.guide-layout{display:block;padding:0 12px 25px}.guide-toc{position:static;max-height:220px;margin-bottom:13px}.guide-toc a{display:inline-block;font-size:11px;margin-right:12px}.guide-article{padding:25px 21px;font-size:13px}.guide-article h1{font-size:25px}.guide-article h2{font-size:19px}.brand{height:39px;width:39px}}
    @media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important;animation:none!important}}
    @media print{.page-header,.section-nav,.inspector,.map-toolbar,.map-footer,.guide-toc,.skip-link{display:none!important}.workspace{display:block;padding:0}.map-panel{border:0;box-shadow:none}.map-viewport{height:auto;min-height:0;background:white;overflow:visible}.map-viewport>svg{height:auto;aspect-ratio:4/5}.guide-layout{display:block;padding:0}.guide-article{border:0;box-shadow:none;padding:0;font-size:10pt}.guide-article h2{break-after:avoid}.guide-article pre{white-space:pre-wrap}.guide-article table{display:table}}
  </style>
</head>
<body>
  <a class="skip-link" href="#architecture-content">Skip to architecture</a>
  <header class="page-header">
    <div class="identity">
      <span class="brand" aria-hidden="true">NH</span>
      <div><p class="eyebrow">nozzle-hyperbolic / Technical reference</p><h1>Package architecture</h1><p class="subtitle">A visual map of the physical models, numerical engine and solution workflow.</p></div>
    </div>
    <div class="header-actions">
      <div class="tabs" role="tablist" aria-label="Architecture views">
        <button class="tab" id="architecture-map-tab" role="tab" aria-selected="true" aria-controls="architecture-map-view" tabindex="0">Map</button>
        <button class="tab" id="architecture-guide-tab" role="tab" aria-selected="false" aria-controls="architecture-guide-view" tabindex="-1">Guide</button>
      </div>
      <div class="download-wrap">
        <button class="download-button" id="architecture-download-button" aria-expanded="false" aria-controls="architecture-download-menu"><span aria-hidden="true">↓</span> Download</button>
        <div class="download-menu" id="architecture-download-menu" hidden><a href="__SVG_DOWNLOAD__" download="ARCHITECTURE.svg">Vector diagram · SVG</a><a href="__MD_DOWNLOAD__" download="ARCHITECTURE.md">Source guide · Markdown</a><a href="__MATHJAX_LICENSE_DOWNLOAD__" download="MATHJAX_LICENSE.txt">MathJax 3.2.2 · License</a></div>
      </div>
    </div>
  </header>
  <main id="architecture-content" tabindex="-1">
    <section id="architecture-map-view" role="tabpanel" aria-labelledby="architecture-map-tab">
      <div class="workspace">
        <nav class="section-nav panel" aria-label="Map sections"><p class="panel-kicker">Explore the architecture</p><div id="architecture-topic-links"></div><p class="nav-note">Select a section to zoom in.<br>Read the Guide for definitions, options and source locations.</p></nav>
        <div class="map-panel panel">
          <div class="map-toolbar"><div class="toolbar-title">Architecture map<span class="toolbar-status" id="architecture-map-status">Select a card to explore its role</span></div><div class="zoom-controls" aria-label="Diagram view controls"><button class="icon-button" id="architecture-zoom-out" aria-label="Zoom out" title="Zoom out (−)">−</button><span class="zoom-label" id="architecture-zoom-label" aria-hidden="true">100%</span><button class="icon-button" id="architecture-zoom-in" aria-label="Zoom in" title="Zoom in (+)">+</button><span class="control-divider" aria-hidden="true"></span><button class="text-button" id="architecture-fit" title="Fit the full diagram (F)">Fit</button><button class="text-button" id="architecture-reset" title="Reset the view and selection (0)">Reset</button></div></div>
          <div class="map-viewport" id="architecture-viewport" aria-label="Interactive vector architecture map; use section links or zoom controls to explore"><!-- ARCHITECTURE_SVG --></div>
          <div class="map-footer"><span>Drag to pan · Scroll to zoom · Select a card for details</span><span class="keyboard-hint">Keys: <kbd class="key">+</kbd> <kbd class="key">−</kbd> zoom · <kbd class="key">F</kbd> fit · <kbd class="key">0</kbd> reset</span></div>
        </div>
        <aside class="inspector panel" aria-labelledby="architecture-detail-title"><div class="inspector-header"><p class="panel-kicker">Section details</p><h2 id="architecture-detail-title">From physics to solution</h2></div><div class="inspector-body"><p class="detail-description" id="architecture-detail-description">The map follows the package from user configuration and physical models through the shared finite-volume machinery to solution controls and verification. Select any card to inspect it, then open the corresponding guide section for details.</p><button class="guide-link" id="architecture-open-guide">Read the architecture guide <span aria-hidden="true">↗</span></button><p class="inspector-hint">The standalone SVG stays sharp at any scale. The Markdown guide contains the detailed explanation and links to source files.</p></div><div class="offline-label"><span class="offline-dot" aria-hidden="true"></span>Self-contained · Works offline</div></aside>
      </div>
    </section>
    <section id="architecture-guide-view" role="tabpanel" aria-labelledby="architecture-guide-tab" hidden><div class="guide-layout"><nav class="guide-toc panel" aria-label="Guide contents"><p class="panel-kicker">In this guide</p><div id="architecture-guide-links"></div></nav><article class="guide-article panel" id="architecture-guide-article"><!-- ARCHITECTURE_GUIDE --></article></div></section>
    <p class="sr-only" id="architecture-announcement" aria-live="polite" aria-atomic="true"></p>
  </main>
  <script>
  // The combined TeX/SVG runtime and fonts are embedded below. Restrict TeX to
  // bundled packages: no autoload, external extensions, menu or network fonts.
  window.MathJax = {
    loader: {load: []},
    tex: {
      inlineMath: [['$', '$']], displayMath: [['$$', '$$']], processEscapes: true,
      packages: ['base', 'ams', 'newcommand', 'noundefined', 'configmacros']
    },
    svg: {fontCache: 'local'},
    options: {enableMenu: false},
    startup: {typeset: false}
  };
  </script>
  <script id="architecture-mathjax-runtime">
  /* MathJax 3.2.2, Apache-2.0. Full license is embedded in Download > MathJax License.
     Upstream: https://github.com/mathjax/MathJax/tree/3.2.2
     Bundle SHA-256: d4295dc33744836935c1399feece5159577b34c5c8ffb9f1c6324cd82e03a882 */
  <!-- ARCHITECTURE_MATHJAX -->
  </script>
  <script>
  (() => {
    'use strict';
    const byId = id => document.getElementById('architecture-' + id);
    const viewport = byId('viewport'), svg = viewport.querySelector('svg');
    const mapTab = byId('map-tab'), guideTab = byId('guide-tab');
    const guideArticle = byId('guide-article');
    let activeTopic = null;
    let guideMathPromise = null;
    let guideNavigationToken = 0;
    function typesetGuide() {
      if (!guideMathPromise) {
        guideMathPromise = window.MathJax.startup.promise
          .then(() => window.MathJax.typesetPromise([guideArticle]))
          .then(() => { document.body.dataset.mathReady = 'true'; })
          .catch(error => { console.error('Architecture mathematics rendering failed:', error); announce('Math rendering unavailable; the guide retains its readable TeX source.'); });
      }
      return guideMathPromise;
    }
    const sectionFor = {
      api:'configuration-example', models:'models', eos:'models', geometry:'geometry-and-closures', closures:'geometry-and-closures', contract:'model-contract', spatial:'spatial-discretization', reconstruction:'spatial-discretization', fluxes:'spatial-discretization', 'boundary-laws':'boundary-conditions', 'boundary-treatment':'boundary-conditions', 'time-explicit':'time-integration', 'time-implicit':'time-integration', jacobian:'jacobian-and-newton', newton:'jacobian-and-newton', control:'initialization-and-convergence', initialization:'initialization-and-convergence', outputs:'initialization-and-convergence', verification:'verification-and-limits'
    };
    const labels = {api:'User configuration', models:'Physical models', eos:'Thermodynamics', geometry:'Geometry', closures:'Physical closures', contract:'Model interface', spatial:'Spatial discretization', reconstruction:'Reconstruction', fluxes:'Numerical fluxes', 'boundary-laws':'Boundary conditions', 'boundary-treatment':'Boundary solution', 'time-explicit':'Explicit integration', 'time-implicit':'Implicit integration', jacobian:'Jacobian & linear solve', newton:'Newton iteration', control:'Convergence & controls', initialization:'Initialization', outputs:'Results & diagnostics', verification:'Verification'};
    function announce(message) { byId('announcement').textContent = message; }
    function showView(name, focus = false) {
      const mapActive = name === 'map';
      const changed = byId('map-view').hidden === mapActive;
      guideNavigationToken += 1;
      byId('map-view').hidden = !mapActive;
      byId('guide-view').hidden = mapActive;
      for (const tab of [mapTab, guideTab]) {
        const selected = tab === (mapActive ? mapTab : guideTab);
        tab.setAttribute('aria-selected', String(selected)); tab.tabIndex = selected ? 0 : -1;
      }
      if (!mapActive) typesetGuide();
      if (changed) window.scrollTo({top:0,behavior:'instant'});
      if (focus) (mapActive ? mapTab : guideTab).focus();
    }
    mapTab.addEventListener('click', () => showView('map'));
    guideTab.addEventListener('click', () => showView('guide'));
    guideArticle.addEventListener('click', event => {
      if (event.target.closest('a[data-architecture-map]')) {
        event.preventDefault(); showView('map', true);
      }
    });
    [mapTab, guideTab].forEach(tab => tab.addEventListener('keydown', event => {
      if (['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) {
        event.preventDefault(); showView(event.key === 'Home' ? 'map' : event.key === 'End' ? 'guide' : tab === mapTab ? 'guide' : 'map', true);
      }
    }));
    const headings = [...guideArticle.querySelectorAll('h2')];
    headings.forEach(heading => {
      const link = document.createElement('a'); link.href = '#' + heading.id; link.textContent = heading.textContent;
      byId('guide-links').append(link);
    });
    function openGuide(topic) {
      showView('guide');
      const id = sectionFor[topic] || 'overview';
      const target = guideArticle.querySelector('#' + id) || guideArticle.querySelector('h1');
      const navigation = guideNavigationToken;
      if (target) {
        history.replaceState(null,'','#' + target.id);
        target.scrollIntoView({behavior:'instant',block:'start'});
        target.tabIndex = -1; target.focus({preventScroll:true});
        typesetGuide().then(() => {
          if (!byId('guide-view').hidden && navigation === guideNavigationToken) target.scrollIntoView({behavior:'instant',block:'start'});
        });
      }
    }
    byId('open-guide').addEventListener('click', () => openGuide(activeTopic));
    const downloadButton = byId('download-button'), downloadMenu = byId('download-menu');
    downloadButton.addEventListener('click', () => {
      downloadMenu.hidden = !downloadMenu.hidden;
      downloadButton.setAttribute('aria-expanded', String(!downloadMenu.hidden));
    });
    document.addEventListener('click', event => {
      if (!event.target.closest('.download-wrap')) { downloadMenu.hidden = true; downloadButton.setAttribute('aria-expanded','false'); }
    });
    document.addEventListener('keydown', event => {
      if (event.key === 'Escape' && !downloadMenu.hidden) { downloadMenu.hidden = true; downloadButton.setAttribute('aria-expanded','false'); downloadButton.focus(); }
    });
    if (!svg) { byId('map-status').textContent = 'Map unavailable. Open the Guide for the architecture.'; return; }
    svg.setAttribute('id','architecture-inline-svg');
    svg.setAttribute('role','group');
    const initial = svg.viewBox.baseVal;
    const original = {x:initial.x, y:initial.y, width:initial.width || 1760, height:initial.height || 2200};
    let view = {...original};
    svg.removeAttribute('width'); svg.removeAttribute('height');
    svg.setAttribute('preserveAspectRatio','xMidYMid meet');
    function setView(next, speak = false) {
      if (!Number.isFinite(next.width) || !Number.isFinite(next.height) || next.width <= 0 || next.height <= 0) return;
      view = next;
      svg.setAttribute('viewBox', `${view.x} ${view.y} ${view.width} ${view.height}`);
      const percent = Math.round(original.width / view.width * 100);
      byId('zoom-label').textContent = percent + '%';
      if (speak) announce('Diagram zoom ' + percent + ' percent.');
    }
    function zoom(factor, anchor) {
      const ratio = Math.min(12, Math.max(.4, original.width / view.width * factor));
      const width = original.width / ratio, height = view.height * width / view.width;
      const point = anchor || {x:view.x + view.width / 2, y:view.y + view.height / 2};
      const fx = (point.x - view.x) / view.width, fy = (point.y - view.y) / view.height;
      setView({x:point.x - width * fx, y:point.y - height * fy, width, height});
    }
    function pointAt(event) {
      const point = svg.createSVGPoint(); point.x = event.clientX; point.y = event.clientY;
      return point.matrixTransform(svg.getScreenCTM().inverse());
    }
    function fitCard(card) {
      const local = card.getBBox(), matrix = svg.getCTM().inverse().multiply(card.getCTM());
      const corners = [[local.x,local.y],[local.x + local.width,local.y],[local.x,local.y + local.height],[local.x + local.width,local.y + local.height]].map(([x,y]) => {
        const point = svg.createSVGPoint(); point.x = x; point.y = y; return point.matrixTransform(matrix);
      });
      const x = Math.min(...corners.map(point => point.x)), y = Math.min(...corners.map(point => point.y));
      const box = {x,y,width:Math.max(...corners.map(point => point.x)) - x,height:Math.max(...corners.map(point => point.y)) - y};
      const margin = 55;
      // Match the visible frame's aspect ratio so a section uses the available area.
      const bounds = viewport.getBoundingClientRect(), aspect = bounds.width / bounds.height;
      let width = box.width + margin * 2, height = box.height + margin * 2;
      if (width / height < aspect) width = height * aspect; else height = width / aspect;
      setView({x:box.x + box.width / 2 - width / 2,y:box.y + box.height / 2 - height / 2,width,height});
    }
    const cards = [...svg.querySelectorAll('[data-topic]')];
    const navButtons = new Map();
    function selectCard(card, focusDiagram = true) {
      activeTopic = card.dataset.topic;
      const title = card.querySelector('title')?.textContent.trim() || labels[activeTopic] || activeTopic;
      const description = card.querySelector('desc')?.textContent.trim() || [...card.querySelectorAll('text')].map(text => text.textContent.trim()).filter(Boolean).join('\n');
      byId('detail-title').textContent = title;
      byId('detail-description').textContent = description;
      byId('open-guide').firstChild.textContent = 'Read this guide section ';
      byId('map-status').textContent = labels[activeTopic] || title;
      cards.forEach(item => item.classList.toggle('is-selected', item === card));
      navButtons.forEach((button, topic) => button.setAttribute('aria-current', String(topic === activeTopic)));
      if (focusDiagram) fitCard(card);
      announce(title + ' selected. Details are available beside the map.');
    }
    cards.forEach(card => {
      const topic = card.dataset.topic;
      card.setAttribute('tabindex','0'); card.setAttribute('role','button');
      card.setAttribute('aria-label', (labels[topic] || topic) + ': show section details');
      card.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); selectCard(card); }
      });
      const button = document.createElement('button'); button.className = 'topic-link';
      button.textContent = labels[topic] || card.querySelector('title')?.textContent || topic;
      button.addEventListener('click', () => selectCard(card));
      byId('topic-links').append(button); navButtons.set(topic,button);
    });
    byId('zoom-in').addEventListener('click', () => { zoom(1.35); announce(byId('zoom-label').textContent + ' zoom.'); });
    byId('zoom-out').addEventListener('click', () => { zoom(1 / 1.35); announce(byId('zoom-label').textContent + ' zoom.'); });
    byId('fit').addEventListener('click', () => setView({...original}, true));
    function resetView() {
      setView({...original}); activeTopic = null;
      cards.forEach(card => card.classList.remove('is-selected'));
      navButtons.forEach(button => button.removeAttribute('aria-current'));
      byId('detail-title').textContent = 'From physics to solution';
      byId('detail-description').textContent = 'The map follows the package from user configuration and physical models through the shared finite-volume machinery to solution controls and verification. Select any card to inspect it, then open the corresponding guide section for details.';
      byId('open-guide').firstChild.textContent = 'Read the architecture guide ';
      byId('map-status').textContent = 'Select a card to explore its role';
      announce('Full architecture map restored.');
    }
    byId('reset').addEventListener('click', resetView);
    viewport.addEventListener('wheel', event => {
      event.preventDefault(); zoom(Math.exp(-event.deltaY * .0015), pointAt(event));
    }, {passive:false});
    let drag = null;
    viewport.addEventListener('pointerdown', event => {
      if (event.button !== 0) return;
      const matrix = svg.getScreenCTM();
      drag = {id:event.pointerId, x:event.clientX, y:event.clientY, view:{...view}, scale:matrix.a, moved:false, card:event.target.closest('[data-topic]')};
      viewport.setPointerCapture(event.pointerId);
    });
    viewport.addEventListener('pointermove', event => {
      if (!drag || event.pointerId !== drag.id) return;
      const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 5) { drag.moved = true; viewport.classList.add('is-panning'); }
      if (drag.moved) setView({...drag.view,x:drag.view.x - dx / drag.scale,y:drag.view.y - dy / drag.scale});
    });
    function finishDrag(event) {
      if (!drag || event.pointerId !== drag.id) return;
      const completed = drag; drag = null; viewport.classList.remove('is-panning');
      if (viewport.hasPointerCapture(event.pointerId)) viewport.releasePointerCapture(event.pointerId);
      if (event.type === 'pointerup' && !completed.moved && completed.card) selectCard(completed.card);
    }
    viewport.addEventListener('pointerup', finishDrag);
    viewport.addEventListener('pointercancel', finishDrag);
    document.addEventListener('keydown', event => {
      if (byId('map-view').hidden || event.ctrlKey || event.metaKey || event.altKey || event.target.closest('input,textarea,select,[contenteditable]')) return;
      if (event.key === '+' || event.key === '=') { event.preventDefault(); zoom(1.35); }
      else if (event.key === '-') { event.preventDefault(); zoom(1 / 1.35); }
      else if (event.key.toLowerCase() === 'f') { event.preventDefault(); setView({...original}, true); }
      else if (event.key === '0') { event.preventDefault(); resetView(); }
    });
    // Guide deep links also work when the file is opened directly from disk.
    function followHash() {
      if (!location.hash) return;
      const target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
      if (target && guideArticle.contains(target)) {
        showView('guide');
        const navigation = guideNavigationToken;
        target.scrollIntoView({behavior:'instant',block:'start'});
        typesetGuide().then(() => {
          if (!byId('guide-view').hidden && navigation === guideNavigationToken) target.scrollIntoView({behavior:'instant',block:'start'});
        });
      }
    }
    window.addEventListener('hashchange', followHash);
    setView({...original}); followHash();
  })();
  </script>
</body>
</html>
'''
