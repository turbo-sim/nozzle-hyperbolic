"""
plot_two_component.py — the LEFT column of the paper's comparison figure.

Four panels sharing one x-axis, one colour per back-pressure regime:

    1. pressure p
    2. phase velocities u_l (dashed) and u_g (solid)
    3. mixture Mach number, with the sonic line M = 1
    4. void fraction alpha_g (solid, left) and flow quality x (dashed, right)

Reads only `data/two_component/*/solution.csv`. Writes into `figures/`.

    python scripts/plot_two_component.py
    python scripts/plot_two_component.py --layout combined
"""
import argparse

from figure_common import Column, add_layout_arg, render

COLUMN = Column(
    prefix="two_component",
    family="two_component",
    cases=[
        ("subsonic",           "subsonic"),
        ("supersonic_shock",   "supersonic-shock"),
        ("supersonic_adapted", "supersonic-adapted"),
    ],
    liquid_label=r"$u_\ell$ (water)",
    gas_label=r"$u_g$ (nitrogen)",
    alpha_ylim=(0.9, 1.0),
    quality_ylim=(0.0, 0.6),
    m_left=0.85,            # room for the y titles and their numbers
    m_right=0.55,           # room for the quality numbers on the right twin
    left_ylabels=True,
    quality_right_label=False,
    void_legend_loc="best",
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_layout_arg(ap)
    args = ap.parse_args()
    print("two-component column:")
    render(COLUMN, layout=args.layout)


if __name__ == "__main__":
    main()
