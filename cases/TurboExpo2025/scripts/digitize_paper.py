"""Recover visible Figure 9 samples without changing the numerical campaign.

The supplied PDF embeds raster figures, not vector curves. This script extracts
the original 3600-pixel images, calibrates against labelled major ticks, and
selects visible color-stroke centers. It never estimates obscured curves or
interpolates through dashed gaps. Optional PyMuPDF is needed only for extraction;
it can be installed in isolated analysis tooling rather than the solver venv.

  python -m cases.TurboExpo2025.scripts.digitize_paper --pdf <attached.pdf> --extract-only
  poetry run python -m cases.TurboExpo2025.scripts.digitize_paper --from-extracted

Inspect both QA overlays before repeating the last command with
``--confirm-visual-review``. The report accepts only visually reviewed data.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import find_objects, label


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
EXPECTED_SHA256 = "6c0348936149bf73a800780d03b865f7700700c9fc13e54e3852e7ddcf5ab2a6"
PANELS = {
    "a": dict(xref=390, purple="P1", orange="P3", purple_max_mm=43.0, max_mm=50.0),
    "b": dict(xref=388, purple="P2", orange="P4", purple_max_mm=25.0, max_mm=40.0),
}
COLORS = {"purple": (147, 38, 103), "orange": (240, 111, 32)}
P0 = {"P1": 13e6, "P2": 13e6, "P3": 10e6, "P4": 10e6}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def groups(indices):
    if not len(indices):
        return []
    return np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)


def extract(pdf):
    import pymupdf

    if digest(pdf) != EXPECTED_SHA256:
        raise ValueError(
            "This calibration applies only to the inspected attachment SHA-256"
        )
    document = pymupdf.open(pdf)
    page = document[10]
    images = page.get_images(full=True)
    if len(page.get_drawings()) != 0:
        raise ValueError(
            "Source page vector-path identity differs from the inspected attachment"
        )
    metadata = dict(
        pdf_name=pdf.name,
        pdf_sha256=digest(pdf),
        pdf_page_1_based=11,
        printed_page=10,
        vector_drawing_count=0,
        embedded_images=[],
    )
    for panel, config in PANELS.items():
        record = next(image for image in images if image[0] == config["xref"])
        if tuple(record[2:4]) != (3600, 2700):
            raise ValueError(
                "Figure raster dimensions differ from the inspected calibration"
            )
        path = OUT / f"paper_figure9{panel}_source.png"
        pymupdf.Pixmap(document, config["xref"]).save(path)
        metadata["embedded_images"].append(
            dict(
                panel=panel,
                xref=config["xref"],
                size_px=[3600, 2700],
                image_sha256=digest(path),
                path=path.name,
                pdf_rectangle_pt=list(page.get_image_rects(config["xref"])[0]),
            )
        )
    (OUT / "paper_digitization_source.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def calibration(image):
    dark = np.max(image, axis=2) < 70
    # Major tick centers are obtained from the black tick strokes, not curve
    # locations. The same labelled tick layout is present in both panels.
    xs = [
        float(np.mean(g))
        for g in groups(np.flatnonzero(dark[2160:2199].sum(axis=0) > 30))
    ]
    expected_x_pixels = [868.5, 1433.5, 1997.5, 2561.5, 3126.5]
    x_pixels = [
        min(xs, key=lambda value: abs(value - target)) for target in expected_x_pixels
    ]
    ys = [
        float(np.mean(g))
        for g in groups(np.flatnonzero(dark[:, 536:580].sum(axis=1) > 35))
    ]
    expected_y_pixels = [464.5, 696.5, 928.5, 1160.5, 1392.5, 1624.5, 1856.5, 2088.5]
    y_pixels = [
        min(ys, key=lambda value: abs(value - target)) for target in expected_y_pixels
    ]
    if (
        max(abs(np.array(x_pixels) - expected_x_pixels)) > 1
        or max(abs(np.array(y_pixels) - expected_y_pixels)) > 1
    ):
        raise ValueError("Could not recover the inspected major tick calibration")
    x_ticks = np.array([-20.0, 0.0, 20.0, 40.0, 60.0])
    y_ticks = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    x_fit = np.polyfit(x_ticks, x_pixels, 1)
    y_fit = np.polyfit(y_ticks, y_pixels, 1)
    x_error = float(np.max(abs(np.polyval(x_fit, x_ticks) - x_pixels)))
    y_error = float(np.max(abs(np.polyval(y_fit, y_ticks) - y_pixels)))
    return dict(
        x_tick_mm=x_ticks.tolist(),
        x_tick_pixels=x_pixels,
        p_over_p0_ticks=y_ticks.tolist(),
        p_over_p0_tick_pixels=y_pixels,
        x_pixel_per_mm=float(x_fit[0]),
        x_zero_pixel=float(x_fit[1]),
        y_pixel_per_pressure_ratio=float(y_fit[0]),
        pressure_zero_pixel=float(y_fit[1]),
        max_x_calibration_residual_px=x_error,
        max_y_calibration_residual_px=y_error,
        excluded_legend_box_pixels=[1700, 123, 3122, 900],
        plot_inner_box_pixels=[536, 123, 3122, 2200],
    )


def samples(mask, calibration_data, stride, min_column_pixels=4):
    """Keep only single, unbranched visible stroke intersections."""
    output = []
    for x in range(537, 3122, stride):
        pieces = groups(np.flatnonzero(mask[:, x]))
        if len(pieces) != 1:
            continue
        pixels = pieces[0]
        if len(pixels) < min_column_pixels or len(pixels) > 100:
            continue
        y = float(np.median(pixels))
        output.append(
            dict(
                x_px=float(x), y_px=y, half_band_px=float((pixels[-1] - pixels[0]) / 2)
            )
        )
    return output


def make_series(case_id, color_name, kind, raw, cal, panel):
    x = [
        (point["x_px"] - cal["x_zero_pixel"]) / cal["x_pixel_per_mm"] / 1000
        for point in raw
    ]
    pressure_ratio = [
        (point["y_px"] - cal["pressure_zero_pixel"]) / cal["y_pixel_per_pressure_ratio"]
        for point in raw
    ]
    y_uncertainty = [
        (point["half_band_px"] + 1 + cal["max_y_calibration_residual_px"])
        / abs(cal["y_pixel_per_pressure_ratio"])
        for point in raw
    ]
    x_uncertainty = (
        (9 + cal["max_x_calibration_residual_px"]) / cal["x_pixel_per_mm"] / 1000
    )
    name = "HEM Rusanov" if kind == "digitized_paper_hem" else "reference CFD"
    return dict(
        case_id=case_id,
        label=f"Paper {case_id} {'Rusanov HEM' if kind == 'digitized_paper_hem' else 'CFD'} (digitized)",
        kind=kind,
        source_flux="rusanov" if kind == "digitized_paper_hem" else None,
        x_m=x,
        p_Pa=[value * P0[case_id] for value in pressure_ratio],
        p_over_p0=pressure_ratio,
        pressure_uncertainty_Pa=[value * P0[case_id] for value in y_uncertainty],
        x_uncertainty_m=x_uncertainty,
        sample_pixels=raw,
        source=f"GT2025-153250 Figure 9{panel}, PDF page 11 (printed page 10)",
        plotted_color_rgb=list(COLORS[color_name]),
        samples=len(raw),
        uncertainty_note="Conservative visible-stroke half-band plus one-pixel/tick-fit allowance, not a statistical confidence interval. No original-solver-data accuracy is inferred.",
        scope="Visible unambiguous color-stroke samples only. Occluded overlaps, branched intersections and steep shock pixels are omitted. Samples are markers; no curve is reconstructed across gaps.",
    )


def digitize(panel, config, metadata):
    path = OUT / f"paper_figure9{panel}_source.png"
    image = np.asarray(Image.open(path).convert("RGB"))
    if image.shape != (2700, 3600, 3):
        raise ValueError("Unexpected embedded raster dimensions")
    cal = calibration(image)
    result, components = [], {}
    overlay = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(overlay)
    for color_name, rgb in COLORS.items():
        delta = image.astype(float) - np.array(rgb, dtype=float)
        mask = np.sum(delta * delta, axis=2) < 30**2
        mask[:123] = False
        mask[2200:] = False
        mask[:, :536] = False
        mask[:, 3122:] = False
        mask[:900, 1700:] = False
        labels, count = label(mask, np.ones((3, 3)))
        sizes = np.bincount(labels.ravel())
        bounds = find_objects(labels)
        # A dashed stroke contains about 1000 pixels. The continuous HEM
        # components contain >5000 pixels and are separated only by the other
        # curve color overwriting them. Small attached branches are excluded by
        # the per-column single-intersection check below.
        principal = [int(index) for index in np.flatnonzero(sizes >= 5000) if index]
        continuous = np.isin(labels, principal)
        case_id = config[color_name]
        cutoff_left = int(np.ceil(cal["x_zero_pixel"] + 3 * cal["x_pixel_per_mm"]))
        continuous[:, :cutoff_left] = False
        # Upstream the four curves overlap, and downstream HEM/CFD strokes
        # merge again. Those regions cannot identify individual histories.
        max_mm = config["purple_max_mm"] if color_name == "purple" else config["max_mm"]
        cutoff_right = int(
            np.floor(cal["x_zero_pixel"] + max_mm * cal["x_pixel_per_mm"])
        )
        continuous[:, cutoff_right:] = False
        raw = samples(continuous, cal, stride=8)
        result.append(
            make_series(case_id, color_name, "digitized_paper_hem", raw, cal, panel)
        )
        for point in raw[::5]:
            x, y = point["x_px"], point["y_px"]
            draw.ellipse(
                (x - 10, y - 10, x + 10, y + 10), outline=(0, 190, 180), width=4
            )
        dashed = []
        dashed_right = int(
            np.floor(cal["x_zero_pixel"] + config["max_mm"] * cal["x_pixel_per_mm"])
        )
        for component in range(1, count + 1):
            if component in principal or not 100 <= sizes[component] <= 2500:
                continue
            yy, xx = bounds[component - 1]
            if xx.stop <= cutoff_left or xx.stop - xx.start < 5:
                continue
            component_mask = labels == component
            component_mask[:, :cutoff_left] = False
            component_mask[:, dashed_right:] = False
            points = samples(component_mask, cal, stride=8)
            # Retain at most the center of each visibly separate dash. Its
            # position is observed; no missing dash is inferred.
            if points:
                dashed.append(points[len(points) // 2])
        dashed.sort(key=lambda point: point["x_px"])
        result.append(
            make_series(case_id, color_name, "digitized_paper_cfd", dashed, cal, panel)
        )
        for point in dashed:
            x, y = point["x_px"], point["y_px"]
            draw.line((x - 11, y - 11, x + 11, y + 11), fill=(0, 90, 230), width=4)
            draw.line((x - 11, y + 11, x + 11, y - 11), fill=(0, 90, 230), width=4)
        components[color_name] = dict(
            continuous_component_ids=principal,
            continuous_component_pixel_counts=[
                int(sizes[index]) for index in principal
            ],
            HEM_samples=len(raw),
            CFD_dash_centers=len(dashed),
            retained_HEM_window_mm=[3, max_mm],
            retained_CFD_window_mm=[3, config["max_mm"]],
        )
    font = ImageFont.truetype("arial.ttf", 34)
    draw.rectangle((0, 0, 3600, 74), fill="white")
    draw.text(
        (35, 16),
        f"Figure 9{panel} QA: cyan circles = retained HEM samples; blue crosses = retained CFD dash centers",
        fill=(0, 0, 0),
        font=font,
    )
    qa = OUT / f"paper_figure9{panel}_digitization_QA.png"
    overlay.save(qa)
    return result, dict(
        calibration=cal,
        components=components,
        source_image=path.name,
        source_sha256=digest(path),
        QA_image=qa.name,
        QA_sha256=digest(qa),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--from-extracted", action="store_true")
    parser.add_argument("--confirm-visual-review", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if not args.from_extracted:
        if args.pdf is None:
            parser.error("--pdf is required unless --from-extracted is used")
        extract(args.pdf)
    if args.extract_only:
        return
    source = json.loads(
        (OUT / "paper_digitization_source.json").read_text(encoding="utf-8")
    )
    output = dict(
        schema_version=1,
        source=source,
        validated=bool(args.confirm_visual_review),
        method="Calibrated color-stroke sampling of original embedded raster images; exact vector extraction unavailable",
        original_numeric_arrays_available=False,
        cases={},
        panels={},
        generated_at=datetime.now(timezone.utc).isoformat(),
        visual_review=dict(
            passed=bool(args.confirm_visual_review),
            statement=(
                "Both original-raster overlays inspected; accepted samples lie on the indicated visible solid/dashed curves and labelled axes/legend agree."
                if args.confirm_visual_review
                else "Pending inspection of both QA overlays"
            ),
        ),
    )
    for panel, config in PANELS.items():
        series, detail = digitize(panel, config, source)
        output["panels"][panel] = detail
        for item in series:
            output["cases"].setdefault(item["case_id"], []).append(item)
    path = OUT / "paper_digitization.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            dict(
                path=str(path),
                validated=output["validated"],
                samples={
                    case: [(item["kind"], item["samples"]) for item in series]
                    for case, series in output["cases"].items()
                },
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
