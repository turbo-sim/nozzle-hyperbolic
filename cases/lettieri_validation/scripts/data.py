"""Read immutable upstream workbooks and explicit publication transcriptions.

The small XLSX reader uses the standard library: the case runner does not need
an extra Excel dependency. Original workbook bytes are never rewritten.
"""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def workbook_rows(path):
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            tree = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            strings = ["".join(n.itertext()) for n in tree.findall("s:si", NS)]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in sheet.findall("s:sheetData/s:row", NS):
            values = {}
            for cell in row.findall("s:c", NS):
                column = "".join(c for c in cell.attrib["r"] if c.isalpha())
                value = cell.find("s:v", NS)
                if value is not None:
                    raw = value.text
                    values[column] = strings[int(raw)] if cell.attrib.get("t") == "s" else float(raw)
                elif cell.attrib.get("t") == "inlineStr":
                    values[column] = "".join(cell.find("s:is", NS).itertext())
            if values:
                rows.append(values)
        return rows


def definitions():
    return json.loads((ROOT / "data/publication.json").read_text(encoding="utf-8"))


def settings():
    return json.loads((ROOT / "settings.json").read_text(encoding="utf-8"))


def experiment(case):
    document = definitions()
    definition = document["cases"][case]
    if definition["upstream_excel"]:
        rows = workbook_rows(ROOT / "data" / definition["upstream_excel"])[1:]
        x = np.array([r["A"] for r in rows]) * 1e-3
        pressure = np.array([r["B"] for r in rows]) * 1e5
        origin = "upstream digitized positions; publication Table 2 pressures"
    else:
        x = np.asarray(document["nominal_tap_x_mm"]) * 1e-3
        pressure = np.asarray(definition["pressure_bar"]) * 1e5
        origin = "publication Table 2 pressures; approximate nominal tap positions"
    return x, pressure, origin


def cfd(case):
    filename = definitions()["cases"][case]["upstream_cfd"]
    if filename is None:
        return None
    rows = workbook_rows(ROOT / "data" / filename)[1:]
    p = np.array([[r["A"] * 1e-3, r["B"] * 1e5] for r in rows if "A" in r and "B" in r])
    q = np.array([[r["D"] * 1e-3, r["E"]] for r in rows if "D" in r and "E" in r])
    # The CFD workbook contains one out-of-order liquid-fraction point.
    # Sorting is only for plotting and interpolation, not value modification.
    return dict(pressure=p[p[:, 0].argsort()], liquid_fraction=q[q[:, 0].argsort()])


def audit():
    doc, cfg = definitions(), settings()
    evidence = {"andrea_revision": cfg["andrea_revision"], "files": {}, "cases": {}}
    for path in sorted((ROOT / "data").glob("*.xlsx")):
        evidence["files"][path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(workbook_rows(path)) - 1}
    nominal = np.asarray(doc["nominal_tap_x_mm"])
    for name, case in doc["cases"].items():
        x, p, origin = experiment(name)
        assert len(p) == 13 and np.all(np.isfinite(p)) and np.all(p > 0)
        assert np.all(np.diff(x) > 0)
        np.testing.assert_allclose(p / 1e5, case["pressure_bar"], atol=1e-12, rtol=0)
        if case["upstream_excel"]:
            np.testing.assert_array_equal(np.round(x * 1e3 / 7) * 7, nominal)
        evidence["cases"][name] = {"pressure_count": len(p), "position_origin": origin, "cfd_available": cfd(name) is not None}
    geometry = workbook_rows(ROOT / "data/Lettieri_nozzle.xlsx")[1:]
    assert len(geometry) == 33
    assert geometry[0]["A"] == -0.032013 and geometry[-1]["A"] == 0.066355
    assert geometry[0]["B"] == 0.00638
    evidence["geometry"] = {"point_count": len(geometry), "raw_inlet_half_height_m": geometry[0]["B"], "publication_corrected_inlet_half_height_m": cfg["geometry"]["inlet_half_height_m"]}
    print(json.dumps(evidence, indent=2))
    return evidence


if __name__ == "__main__":
    audit()
