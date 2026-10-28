"""Verify the standalone report in headless Edge (optional Playwright tool).

Run with a Python environment already containing playwright. All browser state,
audit evidence and screenshots remain under this study's ignored outputs/.
The numerical summary and generated reports are read-only.
"""

import json
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def main():
    from playwright.sync_api import sync_playwright

    output = STUDY / "outputs" / "html_audit"
    output.mkdir(parents=True, exist_ok=True)
    records, errors = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch_persistent_context(
            str(output / "browser_profile"),
            channel="msedge",
            headless=True,
            viewport={"width": 1600, "height": 1000},
            accept_downloads=True,
            downloads_path=str(output),
        )
        page = browser.pages[0]
        page.on("pageerror", lambda error: errors.append(str(error)))
        # Copy just one HTML into a separate folder: no supporting files.
        standalone = output / "standalone_copy"
        standalone.mkdir(exist_ok=True)
        isolated = standalone / "report.html"
        isolated.write_bytes((STUDY / "REPORT.html").read_bytes())
        browser.set_offline(True)
        requests = []
        page.on("request", lambda request: requests.append(request.url))
        page.goto(isolated.as_uri() + "?theme=light#uniform")
        for initialization in ("uniform", "adapted", "theory"):
            page.click(f'[data-report-tab="{initialization}"]')
            page.wait_for_function(
                "document.documentElement.dataset.reportReady === 'true'"
            )
            for theme in ("light", "dark"):
                page.click(f'[data-theme-choice="{theme}"]')
                page.wait_for_function(
                    "document.documentElement.dataset.reportReady === 'true'"
                )
                checks = page.evaluate(
                    """() => {
                    const root = document.documentElement;
                    const tab = document.body.dataset.page;
                    const bundle = window.backpressureBCReport.bundles[tab];
                    const panel = document.getElementById(tab);
                    const links = [...document.querySelectorAll('[data-report-tab]')];
                    return {
                        theme: root.dataset.theme,
                        initialization: bundle?.report.metadata.selected_initialization || 'theory',
                        plots: panel.querySelectorAll('.js-plotly-plot').length,
                        page_links: links.map(a => a.getAttribute('href')),
                        active_page_count: links.filter(a=>a.getAttribute('aria-current') === 'page').length,
                        page_overflow: root.scrollWidth > innerWidth,
                        math_count: panel.querySelectorAll('mjx-container').length,
                        math_errors: document.querySelectorAll('[data-mml-node="merror"]').length,
                        scrollable_equations: [...panel.querySelectorAll('.math-display')].filter(e=>e.scrollWidth>e.clientWidth+2 || ['auto','scroll'].includes(getComputedStyle(e).overflowX)).length,
                        plot_data_match: !bundle || bundle.report.cases.every((c,i)=> {
                            const residual = document.getElementById(`${tab}-residual-${i}`).data;
                            const cfl = document.getElementById(`${tab}-cfl-${i}`).data;
                            return c.runs.every(r=> {
                                const name = r.variant === 'source' ? 'Current/source BC' : 'New pressure-wave BC';
                                const rt = residual.find(t=>t.name===name);
                                const ct = cfl.find(t=>t.name===name);
                                const accepted = r.history.filter(h=>h.accepted_index>0);
                                return JSON.stringify(rt.y)===JSON.stringify(r.history.map(h=>Math.max(h.residual_rms,1e-18)))
                                    && (accepted.length ? JSON.stringify(ct.y)===JSON.stringify(accepted.map(h=>h.effective_cfl)) : !ct);
                            });
                        }),
                    };
                }"""
                )
                assert checks["theme"] == theme
                assert checks["initialization"] == initialization
                assert checks["plots"] == (0 if initialization == "theory" else 8)
                assert checks["active_page_count"] == 1
                assert len(checks["page_links"]) == 3
                assert checks["page_links"] == ["#uniform", "#adapted", "#theory"]
                assert not checks["page_overflow"]
                assert checks["math_errors"] == checks["scrollable_equations"] == 0
                assert checks["plot_data_match"]
                if initialization == "theory":
                    assert checks["math_count"] > 20
                page.screenshot(
                    path=str(output / f"standalone_{initialization}_{theme}.png")
                )
                records.append(dict(tab=initialization, **checks))
            if initialization != "theory":
                page.check(f"#{initialization}-show-initial")
                page.wait_for_function(
                    "document.documentElement.dataset.reportReady === 'true'"
                )
                page.evaluate(
                    "async tab => await window.backpressureBCReport.controllers[tab].ready()",
                    initialization,
                )
                assert (
                    page.locator(f"#{initialization}-pressure-0").evaluate(
                        "e => e.data.length"
                    )
                    == 4
                )
                page.click(f"#{initialization}-reset-plots")
                page.wait_for_function(
                    "document.documentElement.dataset.reportReady === 'true'"
                )
                with page.expect_download() as event:
                    page.click(f"#{initialization}-download-csv")
                event.value.save_as(str(output / f"outcomes_{initialization}.csv"))
                assert (
                    len(
                        (output / f"outcomes_{initialization}.csv")
                        .read_text()
                        .splitlines()
                    )
                    == 5
                )
                page.locator(f"#{initialization}-cfl").screenshot(
                    path=str(output / f"cfl_{initialization}.png")
                )
        assert not any(
            url.startswith(("http:", "https:")) for url in requests
        ), requests
        # Deep links survive reload and address a section inside this same file.
        page.goto(isolated.as_uri() + "#adapted-convergence")
        page.wait_for_function(
            "document.documentElement.dataset.reportReady === 'true'"
        )
        assert page.locator("body").get_attribute("data-page") == "adapted"
        assert not errors, errors
        browser.close()
    result = dict(
        passed=True,
        standalone_copy=True,
        offline=True,
        pages=records,
        browser_errors=errors,
        requests=requests,
    )
    (output / "browser_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        "PASS: one standalone offline HTML, three tabs, both themes, numerical plot arrays, controls and continuous equations"
    )


if __name__ == "__main__":
    main()
