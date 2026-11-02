"""
plot_phase_change.py — the rigth column of the paper's comparison figure.

Four panels sharing one x-axis, , one colour per back-pressure regime:

    1. pressure p
    2. phase velocities u_l (dashed) and u_g (solid)
    3. mixture Mach number, with the sonic line M = 1
    4. void fraction alpha_g (solid, left) and flow quality x (dashed, right)

Reads only `data/phase_change/*/solution.csv`. Writes into `figures/`.

    python scripts/plot_phase_change.py
    python scripts/plot_phase_change.py --layout combined
"""
import argparse

from figure_common import Column, add_layout_arg, render

COLUMN = Column(
    prefix="phase_change",
    family="phase_change",
    cases=[
        ("subsonic",           "subsonic"),
        ("supersonic_shock",   "supersonic-shock"),
        ("supersonic_adapted", "supersonic-adapted"),
    ],
    liquid_label=r"$u_\ell$ (liquid)",
    gas_label=r"$u_g$ (vapour)",
    # Flashing takes the void fraction from a 1e-3 ghost to nearly 1, so this
    # column needs the full range — unlike the two-component column.
    alpha_ylim=(0.0, 1.0),
    quality_ylim=(0.0, 0.6),
    m_left=0.58,            # y ticks only: no titles on this side
    m_right=0.95,           # quality numbers and the "quality" title
    left_ylabels=False,
    quality_right_label=True,
    void_legend_loc="upper left",
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_layout_arg(ap)
    args = ap.parse_args()
    print("phase-change column:")
    render(COLUMN, layout=args.layout)


if __name__ == "__main__":
    main()
