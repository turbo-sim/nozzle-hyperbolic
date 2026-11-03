"""Rebuild immutable workbook snapshots, references and paper case definitions.

Run from the repository root with ``python -B cases/TurboExpo2025/data/import_sources.py``.
Only the standard library is required. No solver is imported or advanced.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
REVISION = "6961d73a1aa6ce3ee409b2042fb5b0874ac8a15d"
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
SOURCE_FILES = {
    "perfect_gas_nozzle.xlsx": "dev_scripts/main/nozzles points.xlsx",
    "Spinelli_nozzles.xlsx": "dev_scripts/main/Spinelli_nozzles.xlsx",
    "Lettieri_nozzle.xlsx": "dev_scripts/main/Lettieri_nozzle.xlsx",
    "Spinelli_S1.xlsx": "validation_cases/Spinelli2019/profiles/case_m15_z63.xlsx",
    "Spinelli_S2.xlsx": "validation_cases/Spinelli2019/profiles/case_m15_z80.xlsx",
    "Spinelli_S3.xlsx": "validation_cases/Spinelli2019/profiles/case_m15_z98.xlsx",
    "Spinelli_compressibility.xlsx": "validation_cases/Spinelli2019/profiles/Z_case_m15_z638098.xlsx",
    "Lettieri_L1.xlsx": "postprocess_TurboExpo2025/Lettieri/Case1.xlsx",
    "Lettieri_L1_CFD.xlsx": "postprocess_TurboExpo2025/Lettieri/Case1_CFD.xlsx",
    "Lettieri_L5.xlsx": "postprocess_TurboExpo2025/Lettieri/Case5.xlsx",
    "Lettieri_L5_CFD.xlsx": "postprocess_TurboExpo2025/Lettieri/Case5_CFD.xlsx",
    "Berana_experiments.xlsx": "postprocess_TurboExpo2025/Berana/pin90_T45.xlsx",
    "Berana_Petruccelli_2D.xlsx": "dev_scripts/main/petruccelli_berana_2D.xlsx",
    "Berana_Petruccelli_3D.xlsx": "dev_scripts/main/petruccelli_berana_3D.xlsx",
    "LICENSE_UPSTREAM.txt": "LICENSE.md",
}


def git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=REPO)


def rows(filename, sheet_index=1):
    """Return cells keyed by column, retaining the upstream sheet row order."""
    with zipfile.ZipFile(ROOT / "data" / filename) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = [
                "".join(n.itertext())
                for n in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall(
                    "s:si", NS
                )
            ]
        xml = ET.fromstring(archive.read(f"xl/worksheets/sheet{sheet_index}.xml"))
        result = []
        for row in xml.findall("s:sheetData/s:row", NS):
            values = {}
            for cell in row.findall("s:c", NS):
                column = "".join(c for c in cell.attrib["r"] if c.isalpha())
                value = cell.find("s:v", NS)
                if value is not None:
                    values[column] = (
                        strings[int(value.text)]
                        if cell.attrib.get("t") == "s"
                        else float(value.text)
                    )
                elif cell.attrib.get("t") == "inlineStr":
                    values[column] = "".join(cell.find("s:is", NS).itertext())
            if values:
                result.append(values)
        return result


def write_json(path, document):
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def reference(
    label,
    kind,
    values,
    source,
    x_column="A",
    y_column="B",
    x_scale=1.0,
    y_scale=1.0,
    y_name="p_Pa",
    x_offset=0.0,
    **extra,
):
    points = sorted(
        (r[x_column] * x_scale + x_offset, r[y_column] * y_scale)
        for r in values
        if isinstance(r.get(x_column), (int, float))
        and isinstance(r.get(y_column), (int, float))
    )
    return {
        "label": label,
        "kind": kind,
        "x_m": [p[0] for p in points],
        y_name: [p[1] for p in points],
        "source": source,
        **extra,
    }


def build_references():
    result = {}
    throat = 0.6690685012574956
    old_throat = 10.177035175879398
    length_over_H = 14.57
    pressure_errors = {
        "S1": [0.006666676, 0.006666676, 0, 0],
        "S2": [0.006666676, 0.006666676, 0, 0],
        "S3": [0.041111172, 0.041111172, 0.015556757, 0.012221062],
    }
    z_errors = {
        "S1": [0.015590551, 0.014173228, 0.011811024, 0.010866142],
        "S2": [0.012283465, 0.010866142, 0.008031496, 0.008031496],
        "S3": [0.052440945, 0.059055118, 0.017480315, 0.030708661],
    }
    zrows = rows("Spinelli_compressibility.xlsx")[1:]
    for name, p0, column in [
        ("S1", 9.20e5, "B"),
        ("S2", 4.59e5, "C"),
        ("S3", 0.84e5, "D"),
    ]:
        note = "Normalized similarity coordinate, represented by x_m for a common field schema. Archived x/H is divided by archived L/H=14.57, without tuning or translating coordinates to the current prediction. The archived reported throat10.177035175879398H differs from the recovered spline throat0.669069L. Absolute physical scale and exact original coordinate mapping are unrecovered; coordinate comparison is approximate."
        result[name] = [
            reference(
                "Spinelli experiment: pressure",
                "experiment",
                rows(f"Spinelli_{name}.xlsx")[1:],
                f"Spinelli_{name}.xlsx",
                x_scale=1 / length_over_H,
                y_scale=p0,
                p_over_p0_uncertainty=pressure_errors[name],
                coordinate_note=note,
            ),
            reference(
                "Spinelli experiment: compressibility",
                "experiment",
                zrows,
                "Spinelli_compressibility.xlsx",
                y_column=column,
                y_name="compressibility_factor",
                x_scale=1 / length_over_H,
                uncertainty=z_errors[name],
                coordinate_note=note,
            ),
            reference(
                "Reconstructed throat-aligned coordinate diagnostic",
                "coordinate_diagnostic",
                rows(f"Spinelli_{name}.xlsx")[1:],
                f"Spinelli_{name}.xlsx",
                x_scale=1 / length_over_H,
                x_offset=throat - old_throat / length_over_H,
                y_scale=p0,
                coordinate_note="Alternative diagnostic only, excluded from experimental error scoring. Reconstructed shift aligns archived reported throat with recovered geometry; not an additional experiment or primary coordinate reference.",
            ),
        ]
    for name in ["L1", "L5"]:
        experiment = reference(
            "Lettieri experiment",
            "experiment",
            rows(f"Lettieri_{name}.xlsx")[1:],
            f"Lettieri_{name}.xlsx",
            x_scale=1e-3,
            y_scale=1e5,
            coordinate_note="Digitized x in mm, pressure in bar. Source positions are approximate; values are not exact point coordinates from a primary table.",
        )
        cfd_rows = rows(f"Lettieri_{name}_CFD.xlsx")[1:]
        cfd_pressure = reference(
            "Romei and Persico HEM CFD: pressure",
            "reference_cfd",
            cfd_rows,
            f"Lettieri_{name}_CFD.xlsx",
            x_scale=1e-3,
            y_scale=1e5,
        )
        cfd_liquid = reference(
            "Romei and Persico HEM CFD: liquid fraction",
            "reference_cfd",
            cfd_rows,
            f"Lettieri_{name}_CFD.xlsx",
            x_column="D",
            y_column="E",
            y_name="liquid_mass_fraction",
            x_scale=1e-3,
        )
        result[name] = [experiment, cfd_pressure, cfd_liquid]
        result[f"{name}_inviscid"] = result[name]
    for name, sheet in [("B1", 3), ("B2", 2), ("B3", 1)]:
        result[name] = [
            reference(
                "Berana experiment",
                "experiment",
                rows("Berana_experiments.xlsx", sheet)[1:],
                f"Berana_experiments.xlsx:sheet{sheet}",
                x_scale=1e-3,
                y_scale=1e6,
                coordinate_note="Upstream positions in mm; pressure values near 9 correspond to MPa, not bar. Values sorted by x only, with no smoothing or pressure edits.",
            )
        ]
    for dim in ["2D", "3D"]:
        result["B3"].append(
            reference(
                f"Petruccelli {dim} CFD",
                "reference_cfd",
                rows(f"Berana_Petruccelli_{dim}.xlsx")[1:],
                f"Berana_Petruccelli_{dim}.xlsx",
                y_scale=90e5,
                coordinate_note="Upstream x already in m; pressure ratio multiplied by B3 p0=90bar. The 3D workbook sheet title incorrectly says 2D; filename and archived publication attribution identify the series.",
            )
        )
    for suffix in ["k0", "k1", "k3"]:
        result[f"B3_{suffix}"] = result["B3"]
    for name in ["P1", "P2", "P3", "P4"]:
        result[name] = []
    return result


def build_definitions():
    geometries = {
        "perfect_gas": {
            "kind": "polynomial_radius",
            "workbook": "data/perfect_gas_nozzle.xlsx",
            "sheet": "nozzle Ma1.5",
            "x_column": "A",
            "y_column": "B",
            "x_scale": 1.0,
            "y_scale": 1.0,
            "x_min": 0.0,
            "x_max": 1.0,
            "degree": 12,
            "notes": [
                "Degree 12 is independently identified by the paper's Aout/A*=1.86 and critical mass flow22.48kg/s: recovered polynomial gives1.860880112 and22.47995917. Archived old driver degree6 and old exact script degree8 do not reproduce these scalars.",
                "The normalized wall shape is treated as radius in a1m computational duct, following the perfect-gas paper plotting sources.",
            ],
        },
        "spinelli": {
            "kind": "spline_half_height",
            "workbook": "data/Spinelli_nozzles.xlsx",
            "sheet": "nozzle Ma1.5",
            "x_column": "A",
            "y_column": "B",
            "x_scale": 1.0,
            "y_scale": 1.0,
            "x_min": 0.0,
            "x_max": 1.0,
            "width_m": 1.0,
            "degree": 12,
            "notes": [
                "Frictionless similarity geometry: both x and wall ordinate are normalized by source length, computational length1. No absolute dimensional mass-flow validation is claimed.",
                "CubicSpline(not-a-knot), as archived fit_nozzle_shape.py. Archived driver had alternative polynomial and spline branches, and historical field NPZ is missing. Exact source geometry interpolation cannot be proven.",
                "Primary experimental coordinates use archived x/H divided by14.57. A separately marked reconstructed coordinate diagnostic is supplied because archived throat10.177H disagrees with normalized shape's throat at0.669069L; it is excluded from experimental scoring.",
            ],
        },
        "lettieri": {
            "kind": "spline_half_height",
            "workbook": "data/Lettieri_nozzle.xlsx",
            "sheet": "Sheet1",
            "x_column": "A",
            "y_column": "B",
            "x_scale": 1.0,
            "y_scale": 1.0,
            "x_min": -0.032013,
            "x_max": 0.066355,
            "width_m": 0.00648,
            "degree": 10,
            "notes": [
                "Original workbook half-height retained byte-for-byte: inlet0.00638m, no publication-specific correction silently substituted.",
                "Rectangular section, A=2*h*width, hydraulic diameter4A/[2(width+2h)], source width6.48mm.",
            ],
        },
        "petruccelli": {
            "kind": "spline_radius",
            "workbook": "data/Lettieri_nozzle.xlsx",
            "sheet": "Sheet1",
            "x_column": "A",
            "y_column": "B",
            "x_scale": 1.0,
            "y_scale": 1.0,
            "x_min": -0.032013,
            "x_max": 0.066355,
            "degree": 10,
            "notes": [
                "Paper Section3.4 reuses Lettieri contour with circular rather than rectangular section: A=pi*r^2, Dh=2r."
            ],
        },
        "berana": {
            "kind": "piecewise_half_height",
            "x_min": -0.015,
            "x_max": 0.00838,
            "throat_x_m": 0.0,
            "inlet_half_height_m": 0.005,
            "throat_half_height_m": 0.0002,
            "outlet_half_height_m": 0.0002702,
            "width_m": 0.001,
            "convergent_slope": -0.32,
            "divergent_slope": 0.0702 / 8.38,
            "notes": [
                "Exact archived paper-version main.py piecewise formula: h=.0002-.32*x upstream; h=.0002+(.0702/8.38)*x downstream.",
                "Source geometry gives throat0.4mm2 and outlet0.5404mm2; paper text rounds/quotes outlet0.52mm2. Preserve formula and disclose discrepancy rather than tuning area to results.",
            ],
        },
    }
    cases = []

    def add(
        id,
        family,
        fluid,
        p0,
        T0,
        p_out,
        cells,
        geometry,
        fluxes,
        roughness=0.0,
        friction=False,
        figures=(),
        **extra,
    ):
        case = {
            "id": id,
            "family": family,
            "fluid": fluid,
            "p0_Pa": p0,
            "T0_K": T0,
            "p_out_Pa": p_out,
            "cells": cells,
            "geometry": geometry,
            "fluxes": list(fluxes),
            "roughness_m": roughness,
            "friction": friction,
            "outlet_regime": "supersonic" if p_out is None else "subsonic",
            "paper_figures": list(figures),
            **extra,
        }
        cases.append(case)

    add(
        "G_subsonic",
        "perfect_gas",
        "Air",
        7e5,
        500.0,
        646999.0,
        500,
        "perfect_gas",
        ["rusanov", "roe", "hll"],
        figures=[2],
        gamma=1.4,
        R_J_kgK=287.0,
        reference_kind="analytical",
        notes=[
            "Subsonic backpressure646999Pa from archived independent perfect-gas expansion script; paper only describes slightly below choking pressure ratio."
        ],
    )
    add(
        "G_adapted",
        "perfect_gas",
        "Air",
        7e5,
        500.0,
        None,
        500,
        "perfect_gas",
        ["rusanov", "roe", "hll"],
        figures=[2, 3, 4],
        gamma=1.4,
        R_J_kgK=287.0,
        reference_kind="analytical",
    )
    add(
        "G_shock_pr156",
        "perfect_gas",
        "Air",
        7e5,
        500.0,
        7e5 / 1.56,
        500,
        "perfect_gas",
        ["rusanov", "roe", "hll"],
        figures=[3],
        gamma=1.4,
        R_J_kgK=287.0,
        reference_kind="analytical",
        notes=[
            "Paper PR1.56 is rounded; exact backpressure of historical NPZ unavailable."
        ],
    )
    for fraction in [0.25, 0.5, 0.75]:
        add(
            f"G_shock_{int(100*fraction):02d}",
            "perfect_gas",
            "Air",
            7e5,
            500.0,
            None,
            500,
            "perfect_gas",
            ["rusanov", "roe", "hll"],
            figures=[4],
            gamma=1.4,
            R_J_kgK=287.0,
            reference_kind="analytical",
            shock_fraction=fraction,
            outlet_regime="subsonic",
            notes=[
                "Backpressure is derived independently from the analytic normal-shock solution at the requested fraction of the divergent length; this is reference setup, not a numerical field warm start."
            ],
        )
    for name, p0, T0, Z0 in [
        ("S1", 9.20e5, 541.0, 0.63),
        ("S2", 4.59e5, 512.0, 0.81),
        ("S3", 0.84e5, 537.0, 0.98),
    ]:
        add(
            name,
            "spinelli",
            "MDM",
            p0,
            T0,
            None,
            250,
            "spinelli",
            ["rusanov", "hll"],
            figures=[5, 6],
            paper_Z0=Z0,
            notes=[
                "Table2 rounded reservoir temperature retained; primary experimental coordinates use archived x/H divided by14.57. Coordinate ambiguity remains; a reconstructed throat-aligned alternative is separately marked diagnostic."
            ],
        )
    for name, p0, T0 in [("L1", 58.96e5, 314.7), ("L5", 84.74e5, 313.9)]:
        add(
            name,
            "lettieri",
            "CO2",
            p0,
            T0,
            None,
            200,
            "lettieri",
            ["rusanov", "hll"],
            friction=True,
            figures=[7, 8],
            notes=[
                "Paper Table2 rounded T0 retained; smooth wall Haaland friction with Beattie viscosity intended. Main text200cells differs from Appendix finest250."
            ],
        )
        add(
            name + "_inviscid",
            "lettieri",
            "CO2",
            p0,
            T0,
            None,
            200,
            "lettieri",
            ["rusanov", "hll"],
            figures=[8],
            reference_case=name,
        )
    for name, p0, p_out in [
        ("P1", 130e5, 78e5),
        ("P2", 130e5, 91e5),
        ("P3", 100e5, 60e5),
        ("P4", 100e5, 70e5),
    ]:
        add(
            name,
            "petruccelli",
            "CO2",
            p0,
            348.0,
            p_out,
            600,
            "petruccelli",
            ["rusanov", "hll"],
            roughness=2e-6,
            friction=True,
            figures=(
                [7, 9, 10]
                if name in ["P1", "P3"]
                else [7, 9, 11] if name == "P4" else [7, 9]
            ),
            notes=[
                "Paper compared finite-rate Hertz-Knudsen CFD. Original P1-P4 CFD arrays are absent from the archived repository; no fabricated reference curve is supplied."
            ],
        )
    for name, p_out in [("B1", 50.1e5), ("B2", 56.9e5), ("B3", 59.6e5)]:
        add(
            name,
            "berana",
            "CO2",
            90e5,
            318.0,
            p_out,
            1600,
            "berana",
            ["rusanov", "hll"],
            roughness=2e-6,
            friction=True,
            figures=[7, 12],
            notes=[
                "1600cells is AppendixA finest mesh; main text does not specify Berana production cell count."
            ],
        )
    for k in [0, 1, 3]:
        add(
            f"B3_k{k}",
            "berana_sensitivity",
            "CO2",
            90e5,
            318.0,
            59.6e5,
            1600,
            "berana",
            ["rusanov", "hll"],
            roughness=k * 1e-6,
            friction=True,
            figures=[12],
            reference_case="B3",
        )
    return {
        "schema_version": 1,
        "paper": {
            "title": "A Hyperbolic One-Dimensional Model for Two-Phase Flows in Converging-Diverging Nozzles",
            "authors": [
                "Andrea Cioffi",
                "Alberto Bergamin",
                "Roberto Agromayor",
                "Fredrik Haglind",
            ],
            "year": 2025,
            "doi": "10.1115/GT2025-153250",
            "url": "https://doi.org/10.1115/GT2025-153250",
            "source_pdf_name": "A_Hyperbolic_One_Dimensional_Model_for_Two_Phase_Flows_in_Converging_Diverging_Nozzles.pdf",
            "source_pdf_sha256": "",
            "archived_code_revision": REVISION,
        },
        "numerical_policy": {
            "spatial_reconstruction": "constant",
            "spatial_order": 1,
            "fluxes_perfect_gas": ["rusanov", "roe", "hll"],
            "fluxes_equilibrium": ["rusanov", "hll"],
            "time_integrator": "euler-implicit",
            "paper_time_integrator": "explicit Euler",
            "paper_cfl": 0.5,
            "paper_stopping": "two-norm of relative difference of consecutive solutions below1e-6",
            "current_stopping": "original spatial residual plus boundary and global monitors; actual thresholds recorded with each run",
            "friction_intent": "Haaland Darcy friction and Beattie two-phase dynamic viscosity, using current HEM implementation",
            "initialization": "current model cold initialization unless explicitly recorded otherwise; no historical paper or analytic solution field used as a hidden seed",
            "notes": [
                "Two-phase Roe is deliberately excluded, as in the paper.",
                "Current formulation changes are permitted by the user and must be disclosed. This is a replication campaign with current physics and implicit integration, not bitwise regression against missing old NPZ.",
            ],
        },
        "geometries": geometries,
        "cases": cases,
        "mesh_studies": [
            {
                "case_id": "G_adapted",
                "flux": "rusanov",
                "cells": [50, 100, 200, 300, 400],
                "observables": ["p_throat", "p_out", "mass_flow"],
                "notes": "Paper does not identify which perfect-gas boundary regime was used for Table4; adapted selected explicitly for this replication.",
            },
            {
                "case_id": "S2",
                "flux": "rusanov",
                "cells": [50, 100, 150, 200, 250],
                "observables": ["p_throat", "p_out", "mass_flow"],
            },
            {
                "case_id": "L5",
                "flux": "rusanov",
                "cells": [50, 100, 150, 200, 250],
                "observables": ["p_throat", "p_out", "mass_flow"],
            },
            {
                "case_id": "P1",
                "flux": "rusanov",
                "cells": [200, 300, 400, 500, 600],
                "observables": ["p_throat", "p_min", "mass_flow"],
            },
            {
                "case_id": "B3",
                "flux": "rusanov",
                "cells": [100, 200, 400, 800, 1600],
                "observables": ["p_throat", "p_min", "mass_flow"],
            },
        ],
        "paper_tables": {
            "table1": {
                "description": "Historical paper values; entropy generation at25/50/75percent divergent length and choked mass flow",
                "entropy_exact_J_kgK": [18.66, 60.66, 105.79],
                "mass_flow_exact_kg_s": 22.48,
                "relative_error_percent": {
                    "rusanov": {"entropy": [-5.22, -3.62, -2.51], "mass_flow": -0.48},
                    "roe": {"entropy": [4.14, 1.37, 1.70], "mass_flow": -0.37},
                    "hll": {"entropy": [-5.16, -2.72, -4.51], "mass_flow": -0.44},
                },
            },
            "table3": {
                "description": "Historical mass-flow comparison: paper labels kg/s, but these values use half-channel per-depth mass flux rho*u*h, i.e.kg/(s*m). CFD columns are independent references; HEM columns are historical reported results.",
                "actual_comparison_units": "kg/(s*m)",
                "paper_printed_units": "kg/s",
                "current_conversion": "mass_flow_full_channel_kg_s/(2*width_m)",
                "width_m": 0.00648,
                "units_note": "See existing cases/lettieri_validation/README.md geometry/mass-flow convention and archived TurboExpo plotting source m3=u*rho*0.002006. Report actual full-channel kg/s separately.",
                "L1": {
                    "CFD_mixture": 29.43,
                    "CFD_barotropic": 29.46,
                    "paper_HEM_rusanov": 29.37,
                    "paper_HEM_hll": 29.41,
                },
                "L5": {
                    "CFD_mixture": 48.94,
                    "CFD_barotropic": 48.95,
                    "paper_HEM_rusanov": 49.06,
                    "paper_HEM_hll": 48.50,
                },
            },
            "table4": {
                "description": "Historical percent differences from finest grid, not experimental error; Rusanov",
                "columns": [
                    "cells",
                    "p_throat_percent",
                    "p_out_or_min_percent",
                    "mass_flow_percent",
                ],
                "perfect_gas": [
                    [50, 9.52, 12.15, 4.38],
                    [100, 3.22, 5.44, 1.48],
                    [200, 3.04, 1.84, 0.22],
                    [300, 1.01, 0.62, 0.02],
                    [400, 0, 0, 0],
                ],
                "S2": [
                    [50, 11.36, 11.36, 2.49],
                    [100, 4.63, 4.63, 1.14],
                    [150, 2.13, 2.13, 0.57],
                    [200, 0.85, 0.85, 0.25],
                    [250, 0, 0, 0],
                ],
                "L5": [
                    [50, 9.66, 5.44, 5.44],
                    [100, 2.60, 2.26, 2.26],
                    [150, 0.54, 0.74, 0.74],
                    [200, 0.12, 0.35, 0.35],
                    [250, 0, 0, 0],
                ],
                "P1": [
                    [200, 4.99, 5.62, 1.01],
                    [300, 3.62, 2.82, 0.55],
                    [400, 1.45, 1.42, 0.28],
                    [500, 1.15, 0.57, 0.12],
                    [600, 0, 0, 0],
                ],
                "B3": [
                    [100, 22.03, 57.25, 6.88],
                    [200, 15.25, 31.65, 3.47],
                    [400, 10.88, 11.24, 2.06],
                    [800, 2.45, 4.47, 1.10],
                    [1600, 0, 0, 0],
                ],
            },
        },
        "figure_map": {
            "1": {
                "description": "Characteristic boundary diagrams",
                "cases": [],
                "notes": ["Methods explanation; no numerical field comparison"],
            },
            "2": {
                "description": "Perfect-gas subsonic/adapted pressure, density and Mach for three fluxes",
                "cases": ["G_subsonic", "G_adapted"],
                "notes": [],
            },
            "3": {
                "description": "Perfect-gas shockPR1.56 and analytical scheme comparison",
                "cases": ["G_shock_pr156"],
                "notes": ["Pressure ratio is rounded in paper"],
            },
            "4": {
                "description": "Perfect-gas shocks at25/50/75percent divergent length",
                "cases": ["G_shock_25", "G_shock_50", "G_shock_75"],
                "notes": [
                    "Paper plotted Roe; all three paper fluxes are calculated here"
                ],
            },
            "5": {
                "description": "Spinelli states and expansions in temperature-entropy plane",
                "cases": ["S1", "S2", "S3"],
                "notes": [
                    "Phase-diagram context, not an experimental thermodynamic trajectory"
                ],
            },
            "6": {
                "description": "Spinelli pressure/compressibility against digitized experiment",
                "cases": ["S1", "S2", "S3"],
                "notes": [
                    "Coordinate origin/interpolation ambiguity disclosed; primary raw x/H divided by14.57"
                ],
            },
            "7": {
                "description": "CO2 validation states and expansions in temperature-entropy plane",
                "cases": ["L1", "L5", "P1", "P2", "P3", "P4", "B1", "B2", "B3"],
                "notes": ["Phase-diagram context"],
            },
            "8": {
                "description": "L1/L5 pressure/liquid fraction with/without friction, experiment and HEM CFD",
                "cases": ["L1", "L1_inviscid", "L5", "L5_inviscid"],
                "notes": ["Original rounded paper reservoir temperatures retained"],
            },
            "9": {
                "description": "P1-P4 pressure",
                "cases": ["P1", "P2", "P3", "P4"],
                "notes": [
                    "Independent CFD curve arrays unavailable; no quantitative reference claim"
                ],
            },
            "10": {
                "description": "P1/P3 Mach and liquid fraction",
                "cases": ["P1", "P3"],
                "notes": ["Independent CFD liquid fraction arrays unavailable"],
            },
            "11": {
                "description": "P4 Rusanov/HLL pressure",
                "cases": ["P4"],
                "notes": ["Independent CFD curve arrays unavailable"],
            },
            "12": {
                "description": "B3 roughness0/1/2/3micron, B1-B3 experiments, B3 2D/3D CFD and geometry",
                "cases": ["B1", "B2", "B3", "B3_k0", "B3_k1", "B3_k3"],
                "notes": [
                    "0micron means hydraulically smooth friction, not frictionless"
                ],
            },
        },
        "known_limits": [
            "Historical final field NPZ files were not committed: archived notes explicitly say they existed only on Andrea's local computer.",
            "No original Petruccelli P1-P4 CFD arrays are present. Their current field predictions can be shown, but paper-curve quantitative agreement remains unverified until reference data are supplied.",
            "Spinelli geometry dimensional scale and exact original interpolation/coordinate transform are incomplete in archived source. Primary reference x/H is divided by14.57; a reconstructed throat-aligned alternative is diagnostic only and excluded from experimental scoring.",
            "Paper describes adapted perfect-gas pressure ratio1.89; with Mach2.1 the stagnation-to-static ratio is about9.1. Preserve paper statement as a known inconsistency, and compute references from the equations.",
            "Paper main text/Appendix mesh counts differ for perfect gas500versus400 and Lettieri200versus250; both production and listed mesh grids retained.",
            "Rounded Table2 temperatures intentionally differ from higher precision values in the separate Lettieri validation suite.",
        ],
    }


def main():
    (ROOT / "data").mkdir(parents=True, exist_ok=True)
    manifest = {
        "repository": "https://github.com/turbo-sim/nozzle-hyperbolic",
        "revision": REVISION,
        "revision_subject": git("show", "-s", "--format=%s", REVISION).decode().strip(),
        "license": "MIT repository license retained in LICENSE_UPSTREAM.txt; experimental/CFD sources retain their original scholarly attribution",
        "files": {},
    }
    for name, source in SOURCE_FILES.items():
        raw = git("show", f"{REVISION}:{source}")
        (ROOT / "data" / name).write_bytes(raw)
        manifest["files"][name] = {
            "source_path": source,
            "git_blob": git("rev-parse", f"{REVISION}:{source}").decode().strip(),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
    definitions = build_definitions()
    pdf = Path.home() / "Downloads" / definitions["paper"]["source_pdf_name"]
    if pdf.exists():
        definitions["paper"]["source_pdf_sha256"] = hashlib.sha256(
            pdf.read_bytes()
        ).hexdigest()
    write_json(ROOT / "case_definitions.json", definitions)
    write_json(ROOT / "data" / "references.json", build_references())
    write_json(ROOT / "data" / "source_manifest.json", manifest)
    print(
        json.dumps(
            {
                "cases": len(definitions["cases"]),
                "solver_configurations": sum(
                    len(c["fluxes"]) for c in definitions["cases"]
                ),
                "source_files": len(SOURCE_FILES),
                "revision": REVISION,
            }
        )
    )


if __name__ == "__main__":
    main()
