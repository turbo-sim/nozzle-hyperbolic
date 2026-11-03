"""Generate the documentation's editable, accessible block-coloring SVG.

Run from the repository root with Poetry. Uses only the standard library;
the diagram illustrates the actual j % (V * (2*r + 1)) seed rule.
"""

from html import escape
from pathlib import Path


def build_svg():
    """Return a radius-two, three-variable Jacobian illustration."""
    width, height = 1160, 930
    ink, muted, line = "#17324d", "#536b80", "#dce5ed"
    # Three shades per block family; labels also identify every scalar color.
    palette = [
        "#175ea8", "#4484c4", "#85afd7",
        "#047f76", "#32a89a", "#82c9ba",
        "#ae6310", "#d58b32", "#edba70",
        "#7951a8", "#a07ac4", "#c5acdc",
        "#ac3b57", "#d16d83", "#e7a8b6",
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Block Jacobian coloring: fifteen seeds for a radius-two stencil</title>',
        '<desc id="desc">Ten point blocks with three variables each. The declared '
        'Jacobian band extends two blocks on either side. Scalar column colors repeat '
        'every fifteen columns. Columns zero and fifteen share a seed because '
        'their possible residual row supports do not overlap.</desc>',
        '<style>text{font-family:Inter,Segoe UI,Arial,sans-serif} '
        '.mono{font-family:Consolas,monospace}</style>',
        f'<rect width="{width}" height="{height}" rx="18" fill="#fff"/>',
    ]

    def rect(x, y, w, h, fill, stroke="none", **attrs):
        attributes = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                     f'fill="{fill}" stroke="{stroke}" {attributes}/>')

    def text(x, y, value, size=16, color=ink, weight=400, **attrs):
        attributes = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" '
                     f'fill="{color}" font-weight="{weight}" {attributes}>'
                     f'{escape(str(value))}</text>')

    text(38, 45, "A local stencil. A small set of directional seeds.", 28, weight=700)
    text(38, 76, "10 point blocks  ·  3 variables per block  ·  radius r = 2", 17, muted)
    rect(38, 100, 1084, 54, "#eef4f9", rx=10)
    text(60, 134, "color(j) = j mod 15", 20, weight=650, **{"class": "mono"})
    text(448, 133, "15 colors = 5 block families × 3 variable colors", 19)

    x0, y0, step, block = 92, 230, 16, 48
    text(x0, 187, "Full residual Jacobian", 21, weight=650)
    text(x0, 210, "Column colors identify the differentiation seeds", 14, muted)
    labels = ["L", "1", "2", "3", "4", "5", "6", "7", "8", "R"]
    for j, label in enumerate(labels):
        text(x0+j*block+24, y0-9, label, 14, text_anchor="middle", weight=600)
        text(x0-17, y0+j*block+29, label, 14, text_anchor="middle", weight=600)
        for v in range(3):
            rect(x0+j*block+v*step, y0-3, step-1, 3, palette[(3*j+v) % 15])
    for i in range(10):
        for j in range(10):
            active = abs(i-j) <= 2
            for a in range(3):
                for b in range(3):
                    rect(x0+j*block+b*step+1, y0+i*block+a*step+1,
                         step-2, step-2, palette[(3*j+b) % 15] if active else "#f1f4f7",
                         fill_opacity="0.80" if active else "1")
            if active:
                rect(x0+j*block, y0+i*block, block, block, "none", "#7890a5", stroke_width="0.7")
    rect(x0, y0, 480, 480, "none", "#6d8499", stroke_width="1.2")
    text(x0+240, 740, "Unknown point block (columns)", 15, muted, text_anchor="middle")
    text(29, 470, "Residual point block (rows)", 15, muted,
         transform="rotate(-90 29 470)", text_anchor="middle")
    text(x0, 764, "L / R: algebraic endpoint traces; 1–8: physical cells", 14, muted)

    px = 638
    text(px, 187, "One color → one JVP", 21, weight=650)
    text(px, 215, "Seed c = 0: columns 0 and 15", 16, muted)
    # Pair of column supports: first variable in block L and in physical cell 5.
    for j, left in ((0, px+8), (5, px+196)):
        text(left+64, 250, f"column {3*j}", 17, weight=600, text_anchor="middle")
        for i in range(10):
            active = abs(i-j) <= 2
            rect(left+37, 268+i*27, 54, 24,
                 palette[0] if active else "#f1f4f7", rx=3)
            text(left+25, 285+i*27, labels[i], 13, muted, text_anchor="end")
        text(left+64, 568, "block rows L–2" if j == 0 else "block rows 3–7", 15,
             text_anchor="middle", weight=600)
    rect(px, 593, 460, 96, "#edf6f4", rx=10)
    text(px+18, 620, "The row supports do not overlap.", 18, weight=650)
    text(px+18, 647, "J(v₀ + v₁₅) contains both columns", 17)
    text(px+18, 672, "without mixing their residual entries.", 17)
    text(px, 725, "Each outlined matrix block is 3 × 3.", 15, muted)
    text(px, 750, "Shaded band = allowed dependencies.", 15, muted)
    text(px, 775, "Some allowed entries can still be zero.", 15, muted)

    rect(38, 803, 1084, 94, "#f6f8fb", line, rx=10)
    text(58, 830, "Scalar colors: three variables in each of five repeating block families", 16, weight=650)
    for c, color in enumerate(palette):
        x = 59+c*69
        rect(x, 845, 58, 34, color, rx=5)
        text(x+29, 868, c, 16, "#fff" if c % 3 != 2 else ink,
             weight=650, text_anchor="middle")
    parts.append("</svg>")
    return "\n".join(parts)+"\n"


if __name__ == "__main__":
    output = Path(__file__).resolve().parent / "source/_static/jacobian_coloring.svg"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_svg(), encoding="utf-8", newline="\n")
    print(f"Wrote {output}")
