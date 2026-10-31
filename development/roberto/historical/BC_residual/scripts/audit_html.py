"""Check a single-file report offline using optional Playwright and local Edge."""

import csv
import io
import json
from pathlib import Path
import re
from playwright.sync_api import sync_playwright

STUDY = Path(__file__).resolve().parents[1]


def main():
    output = STUDY / "outputs/html_audit"
    output.mkdir(parents=True, exist_ok=True)
    isolated = output / "standalone_copy"
    isolated.mkdir(exist_ok=True)
    target = isolated / "report.html"
    target.write_bytes((STUDY / "REPORT.html").read_bytes())
    text = target.read_text(encoding="utf-8")
    payload = json.loads(
        re.search(
            r'<script id="report-data" type="application/json">(.*?)</script>',
            text,
            re.S,
        )[1]
    )
    original = json.loads((STUDY / "outputs/steady/summary.json").read_text())
    assert payload["benchmark"] == original
    errors, requests, records = [], [], []
    with sync_playwright() as pw:
        context = pw.chromium.launch_persistent_context(
            str(output / "browser_profile"),
            channel="msedge",
            headless=True,
            viewport={"width": 1600, "height": 1000},
            accept_downloads=True,
            downloads_path=str(output),
        )
        context.set_offline(True)
        page = context.pages[0]
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("request", lambda r: requests.append(r.url))
        page.goto(target.as_uri() + "?theme=light#results")
        page.wait_for_function(
            "document.documentElement.dataset.reportReady === 'true'"
        )
        for tab in ("results", "theory", "recommendations"):
            page.click(f'[data-report-tab="{tab}"]')
            page.wait_for_function(
                "document.documentElement.dataset.reportReady === 'true'"
            )
            for theme in ("light", "dark"):
                page.click(f'[data-theme-choice="{theme}"]')
                page.wait_for_function(
                    "document.documentElement.dataset.reportReady === 'true'"
                )
                state = page.evaluate(
                    """() => {
                    const api=window.bcResidualReport, tab=api.active, root=document.getElementById(tab);
                    const ids=[...document.querySelectorAll('[id]')].map(e=>e.id);
                    return {tab,theme:document.documentElement.dataset.theme,
                      plots:root.querySelectorAll('.js-plotly-plot').length,
                      math:root.querySelectorAll('mjx-container').length,
                      math_errors:document.querySelectorAll('[data-mml-node="merror"]').length,
                      duplicate_ids:ids.length-new Set(ids).size,
                      page_overflow:document.documentElement.scrollWidth>innerWidth,
                      scrollable_equations:[...root.querySelectorAll('.math-display')].filter(e=>e.scrollWidth>e.clientWidth+2||['auto','scroll'].includes(getComputedStyle(e).overflowX)).length,
                      active_tabs:document.querySelectorAll('[data-report-tab][aria-current="page"]').length,
                      data_match:tab!=='results'||api.bundle.benchmark.cases.every((c,i)=>{
                        const rt=document.getElementById(`residual-${i}`).data,pt=document.getElementById(`pressure-${i}`).data;
                        return c.runs.every((r,j)=>JSON.stringify(rt[j].x)===JSON.stringify(r.history.map(h=>h.update))&&JSON.stringify(rt[j].y)===JSON.stringify(r.history.map(h=>h.residual_rms))&&JSON.stringify(pt[j].y)===JSON.stringify(r.final_state.slice(1,-1).map(v=>v[0]/1000)));
                      }),
                      timing_samples:tab==='results'?document.getElementById('timings').data.filter(t=>t.name.endsWith(' samples')).reduce((n,t)=>n+t.y.length,0):null};
                }"""
                )
                assert state["tab"] == tab and state["theme"] == theme
                assert state["active_tabs"] == 1 and state["duplicate_ids"] == 0
                assert not state["page_overflow"] and state["math_errors"] == 0
                assert state["scrollable_equations"] == 0 and state["data_match"]
                if tab == "results":
                    assert state["plots"] == 11 and state["timing_samples"] == 100
                else:
                    assert state["math"] >= 8
                page.screenshot(path=str(output / f"{tab}_{theme}.png"))
                records.append(state)
        page.click('[data-report-tab="results"]')
        page.wait_for_function(
            "document.documentElement.dataset.reportReady === 'true'"
        )
        for button, name in [
            ("download-csv", "timings.csv"),
            ("download-json", "evidence.json"),
        ]:
            with page.expect_download() as event:
                page.locator("#" + button).click()
            event.value.save_as(str(output / name))
        assert (
            len(list(csv.DictReader(io.StringIO((output / "timings.csv").read_text()))))
            == 100
        )
        assert json.loads((output / "evidence.json").read_text()) == original
        page.click("#reset-plots")
        page.wait_for_function(
            "document.documentElement.dataset.reportReady === 'true'"
        )
        page.goto(target.as_uri() + "#recommendations-recommendation")
        page.wait_for_function(
            "document.documentElement.dataset.reportReady === 'true'"
        )
        assert page.locator("body").get_attribute("data-page") == "recommendations"
        assert not errors, errors
        assert not any(u.startswith(("http:", "https:")) for u in requests), requests
        context.close()
    result = dict(
        passed=True,
        offline=True,
        isolated_html_copy=True,
        all_110_records_unchanged=True,
        tabs=records,
        browser_errors=errors,
        requests=requests,
    )
    (output / "browser_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        "PASS: isolated offline HTML; three tabs; both themes; 11 plots; equations; all 100 timing samples and full-data download"
    )


if __name__ == "__main__":
    main()
