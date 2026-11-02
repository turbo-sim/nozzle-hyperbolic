"""Run the study, or regenerate its report from saved numerical evidence."""

import argparse
import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(name,"1")
os.environ.setdefault("MPLBACKEND","Agg")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only",action="store_true")
    parser.add_argument("--variants",nargs="+")
    parser.add_argument("--grids",nargs="+",type=int)
    args=parser.parse_args()
    # Use `python -m cases.verification_annular.run_case` from repo root.
    if not args.report_only:
        from .scripts.run import run
        run(args.variants,args.grids)
    from .scripts.report import build_report
    build_report()


if __name__=="__main__":
    main()
