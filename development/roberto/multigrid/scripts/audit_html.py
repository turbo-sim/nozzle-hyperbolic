"""Offline browser audit for the original and uniformly tight grid-sequencing report pages.

Uses optional Playwright with local Edge. Only study-local audit files and a
browser profile are written. This does not rerun numerical solves.
"""

import base64
import hashlib
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

STUDY = Path(__file__).resolve().parents[1]


def main():
    output = STUDY / "outputs/html_audit"
    output.mkdir(exist_ok=True)
    isolated = output / "standalone_copy"
    isolated.mkdir(exist_ok=True)
    target = isolated / "report.html"
    target.write_bytes((STUDY / "REPORT.html").read_bytes())
    errors, requests, records = [], [], []
    with sync_playwright() as pw:
        context = pw.chromium.launch_persistent_context(
            str(output / "browser_profile"),
            channel="msedge",
            headless=True,
            viewport={"width": 1600, "height": 1000},
            offline=True,
            accept_downloads=True,
            downloads_path=str(output),
        )
        page = context.pages[0]
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("request", lambda r: requests.append(r.url))
        page.goto(target.as_uri(), wait_until="load")
        page.evaluate("async()=>await window.multigridPageReady")
        for tab in ["four_grid", "two_grid", "results", "theory"]:
            page.click(f'[data-page-link="{tab}"]')
            page.evaluate("async()=>await window.multigridPageReady")
            for theme in ["light", "dark"]:
                page.click(f'[data-theme-choice="{theme}"]')
                if tab != "theory":
                    page.evaluate(
                        "async tab=>await window.gridStudies[tab].ready()", tab
                    )
                state = page.evaluate(
                    """() => {
                  const root=[...document.querySelectorAll('[data-report-page]')].find(e=>!e.hidden),tab=root.id;
                  const ids=[...document.querySelectorAll('[id]')].map(e=>e.id);
                  return {tab,theme:document.documentElement.dataset.theme,plots:root.querySelectorAll('.js-plotly-plot').length,
                    pages:document.querySelectorAll('[data-report-page]').length,active_links:document.querySelectorAll('[data-page-link][aria-current="page"]').length,
                    duplicate_ids:ids.length-new Set(ids).size,overflow:document.documentElement.scrollWidth>innerWidth,
                    math_errors:document.querySelectorAll('[data-mml-node="merror"]').length,
                    math:root.querySelectorAll('mjx-container').length,
                    scrollable_equations:[...root.querySelectorAll('.math-block')].filter(e=>e.scrollWidth>e.clientWidth+2||['auto','scroll'].includes(getComputedStyle(e).overflowX)).length};
                }"""
                )
                assert state["tab"] == tab and state["theme"] == theme
                assert (
                    state["pages"] == 4
                    and state["active_links"] == 1
                    and state["duplicate_ids"] == 0
                )
                assert (
                    not state["overflow"]
                    and state["math_errors"] == state["scrollable_equations"] == 0
                )
                assert state["plots"] == (0 if tab == "theory" else 7)
                if tab == "theory":
                    assert state["math"] > 15
                page.screenshot(path=str(output / f"{tab}_{theme}.png"))
                records.append(state)
            if tab != "theory":
                prefix = {"four_grid": "four-", "two_grid": "two-", "results": ""}[tab]
                page.select_option("#" + prefix + "case", "1")
                page.select_option("#" + prefix + "recon", "muscl_van_leer")
                page.fill("#" + prefix + "skip", "2")
                page.evaluate("async tab=>await window.gridStudies[tab].ready()", tab)
                checks = page.evaluate(
                    """tab=> {
                  const pre={four_grid:'four-',two_grid:'two-',results:''}[tab],study=window.gridStudies[tab],c=study.payload[1];
                  const r=c.runs.find(r=>r.method==='grid_sequence'&&r.reconstruction==='muscl_van_leer');
                  const traces=document.getElementById(pre+'work-history').data;
                  const seq=traces.filter(t=>t.name.startsWith('Coarse to fine'));
                  return {levels:r.levels.length,matching_levels:seq.every((t,i)=>JSON.stringify(t.y)===JSON.stringify(r.levels[i].residual)),
                    sample_count:document.getElementById(pre+'timing').data.filter(t=>t.type==='scatter').reduce((n,t)=>n+t.y.length,0)};
                }""",
                    tab,
                )
                assert checks["levels"] == (2 if tab == "two_grid" else 4)
                assert checks["matching_levels"] and checks["sample_count"] == 120
                page.locator("#" + prefix + "histories").screenshot(
                    path=str(output / f"{tab}_histories.png")
                )
        # All offered downloads are embedded exact bytes and distinguish campaigns.
        downloads = page.locator("a[download]").evaluate_all(
            "els=>els.map(e=>({name:e.download,href:e.getAttribute('href')}))"
        )
        assert len(downloads) == 12
        for item in downloads:
            raw = base64.b64decode(item["href"].split(",", 1)[1])
            if item["name"].startswith("four_grid_"):
                folder = STUDY / "outputs/four_grid_tight"
                filename = item["name"].removeprefix("four_grid_")
            elif item["name"].startswith("two_grid_"):
                folder = STUDY / "outputs/two_grid_tight"
                filename = item["name"].removeprefix("two_grid_")
            else:
                folder = STUDY / "outputs"
                filename = item["name"]
            original = (
                folder / "figures" / filename
                if filename.endswith(".pdf")
                else folder / filename
            )
            assert (
                hashlib.sha256(raw).digest()
                == hashlib.sha256(original.read_bytes()).digest()
            )
        page.set_viewport_size({"width": 1280, "height": 900})
        page.click('[data-page-link="four_grid"]')
        page.evaluate("async()=>await window.multigridPageReady")
        assert not page.evaluate("document.documentElement.scrollWidth>innerWidth")
        page.goto(target.as_uri() + "#four-tight-stages")
        page.evaluate("async()=>await window.multigridPageReady")
        assert page.locator("#four_grid").is_visible()
        assert not errors, errors
        assert not any(x.startswith(("http:", "https:")) for x in requests), requests
        context.close()
    result = dict(
        passed=True,
        offline=True,
        isolated_html=True,
        pages=records,
        embedded_downloads=12,
        plot_arrays_match_saved_data=True,
        browser_errors=errors,
    )
    (output / "browser_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        "PASS: four offline report pages, both themes, 21 interactive plots, independent controls, exact histories and twelve embedded downloads."
    )


if __name__ == "__main__":
    main()
