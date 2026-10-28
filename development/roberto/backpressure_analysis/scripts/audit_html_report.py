"""Optional Playwright audit of the rendered report; no numerical solves.

Requires Playwright in the auditing environment and a local Chromium browser.
Screenshots and browser checks are saved under outputs/html_review/.
"""

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

STUDY = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser', type=Path, default=Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'))
    args = parser.parse_args()
    output = STUDY / 'outputs' / 'html_review'
    output.mkdir(parents=True, exist_ok=True)
    errors, requests = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=str(args.browser), headless=True)
        context = browser.new_context(viewport={'width': 1440, 'height': 1000}, offline=True, accept_downloads=True)
        page = context.new_page()
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda req: requests.append(req.url))
        page.goto((STUDY / 'report_backpressure_sensitivity.html').as_uri(), wait_until='load', timeout=60000)
        page.emulate_media(reduced_motion='reduce')
        page.wait_for_function("document.documentElement.dataset.reportReady === 'true' || document.documentElement.dataset.reportError", timeout=60000)
        assert page.locator('html').get_attribute('data-report-ready') == 'true', page.locator('#report-status').inner_text()
        assert page.locator('#run-table-body tr').count() == 104
        figures = page.locator('.js-plotly-plot').count()
        print('Rendered charts:', figures, flush=True)
        page.screenshot(path=str(output / 'desktop_overview.png'))
        page.locator('#operating-map').scroll_into_view_if_needed()
        page.screenshot(path=str(output / 'desktop_operating_map.png'))
        page.locator('#show-restarts').check()
        page.wait_for_function("document.getElementById('mass-chart').data.length === 4")
        page.locator('#show-restarts').uncheck()
        page.wait_for_function("document.getElementById('mass-chart').data.length === 3")
        page.evaluate("Plotly.relayout('mass-chart', {'xaxis.range': [.55,.4]})")
        page.locator('#reset-charts').click()
        page.wait_for_function("Math.abs(document.getElementById('mass-chart').layout.xaxis.range[0]-1.005)<1e-9")
        page.locator('#profiles').scroll_into_view_if_needed()
        page.locator('#profile-mode').select_option('cold')
        page.wait_for_function("window.backpressureReport.profile().mode === 'cold'")
        options = page.locator('#profile-select option').evaluate_all('els=>els.map(e=>({value:e.value,label:e.textContent}))')
        selected = next(v['value'] for v in options if v['label'].startswith('0.42000000'))
        page.locator('#profile-select').select_option(selected)
        page.wait_for_function('Math.abs(window.backpressureReport.profile().p_back_ratio-.42)<1e-10')
        page.locator('#profile-outlet').check()
        page.locator('#profile-ghosts').check()
        page.wait_for_function("document.getElementById('profile-p').data.some(t=>t.name === 'Algebraic ghost states')")
        page.screenshot(path=str(output / 'desktop_profile_controls.png'))
        page.locator('#profile-p').scroll_into_view_if_needed()
        page.screenshot(path=str(output / 'desktop_profiles.png'))
        page.locator('#table-mode').select_option('cold')
        assert page.locator('#run-table-body tr').count() == 14
        page.locator('#table-search').fill('0.42000000')
        assert page.locator('#run-table-body tr').count() == 1
        with page.expect_download() as downloaded:
            page.locator('#download-csv').click()
        downloaded.value.save_as(str(output / 'download_check.csv'))
        assert (output / 'download_check.csv').read_text() == (STUDY / 'outputs/baseline/operating_map.csv').read_text()
        with page.expect_download() as downloaded_json:
            page.locator('#download-json').click()
        downloaded_json.value.save_as(str(output / 'download_check.json'))
        assert len(json.loads((output / 'download_check.json').read_text())['report']['runs']) == 104
        page.locator('#table-search').fill('')
        page.locator('#table-mode').select_option('all')
        page.locator('#history-region').select_option('all')
        page.wait_for_function("document.getElementById('history-chart').data.length === 87")
        # Verify the animation and its native range control after restyling.
        page.locator('#profile-select').select_option('0')
        page.locator('#profile-index').focus()
        page.keyboard.press('ArrowRight')
        page.wait_for_function("document.getElementById('profile-select').value === '1'")
        start_ratio = page.evaluate('window.backpressureReport.profile().p_back_ratio')
        page.locator('#profile-play').click()
        page.wait_for_function(f'window.backpressureReport.profile().p_back_ratio !== {start_ratio}')
        page.locator('#profile-stop').click()
        assert page.locator('#profile-play').is_enabled()
        assert page.evaluate("window.backpressureReport.bundle.provenance.palette === 'Blues'")
        assert page.evaluate("!getComputedStyle(document.querySelector('h1')).fontFamily.includes('Georgia')")
        page.set_viewport_size({'width': 1280, 'height': 900})
        page.evaluate('window.scrollTo(0,0)')
        page.wait_for_timeout(300)
        page.screenshot(path=str(output / 'desktop_laptop_overview.png'))
        overflow = page.evaluate('document.documentElement.scrollWidth > window.innerWidth')
        page.locator('#profiles').scroll_into_view_if_needed()
        page.screenshot(path=str(output / 'desktop_laptop_profiles.png'))
        page.locator('#history-chart').scroll_into_view_if_needed()
        page.screenshot(path=str(output / 'desktop_laptop_histories.png'))
        assert not overflow, 'Laptop-width document has horizontal overflow'
        page.set_viewport_size({'width': 1920, 'height': 1080})
        page.evaluate('window.scrollTo(0,0)')
        page.wait_for_timeout(300)
        page.screenshot(path=str(output / 'desktop_wide_overview.png'))
        assert not page.evaluate('document.documentElement.scrollWidth > window.innerWidth')
        assert not errors, errors
        external = [u for u in requests if u.startswith(('http://', 'https://'))]
        assert not external, external
        checks = dict(rendered_charts=figures, embedded_states=104, table_filter=True, profile_selector=True,
                      ghost_toggle=True, branch_overlay_toggle=True, zoom_reset=True, csv_download_exact=True,
                      json_download=True,
                      all_history_traces=87, desktop_widths_checked=[1280, 1440, 1920],
                      horizontal_overflow=overflow, animation_play_pause=True, slider_keyboard=True, blue_palette=True,
                      browser_errors=errors, external_network_requests=external, offline=True)
        (output / 'qa.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
        print(json.dumps(checks, indent=2), flush=True)
        browser.close()


if __name__ == '__main__':
    main()
