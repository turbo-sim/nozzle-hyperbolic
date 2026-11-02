"""
make_all_figures.py — regenerate every paper figure from `data/`.

    python scripts/make_all_figures.py
    python scripts/make_all_figures.py --layout combined 
    (this latter is to generate a single column with all the figures)
"""
import argparse

import figure_common
import plot_phase_change
import plot_two_component


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    figure_common.add_layout_arg(ap)
    args = ap.parse_args()

    written = []
    for name, column in (("two-component", plot_two_component.COLUMN),
                         ("phase-change", plot_phase_change.COLUMN)):
        print(f"{name} column:")
        written += figure_common.render(column, layout=args.layout)

    print(f"\n{len(written)} figure(s) written to "
          f"{figure_common.FIGURES.relative_to(figure_common.REPO)}/")


if __name__ == "__main__":
    main()
