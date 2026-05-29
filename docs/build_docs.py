"""
build_docs.py
=============
Build or live-preview the Sphinx documentation.

Usage
-----
    python build_docs.py                   # live preview (sphinx-autobuild)
    python build_docs.py --no-autobuild    # one-shot HTML build
    python build_docs.py --clean-only      # delete build dir and exit
    python build_docs.py --build-bib       # refresh bibliography before building
"""

import os
import shutil
import subprocess
import argparse


DOCS_DIR      = "."
BUILD_DIR     = "_build"
API_OUTPUT_DIR = "source/api"
SRC_DIR       = "../nozzle_hyperbolic"

EXCLUDE_MODULES = [
    "../nozzle_hyperbolic/old",
]


def delete_build(build_dir=BUILD_DIR):
    """Delete a previously generated build directory."""
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
        print(f"Deleted build directory: {build_dir}")


def run_sphinx_apidoc(output_dir=API_OUTPUT_DIR, src_dir=SRC_DIR, exclude=None, force=False):
    """
    Run sphinx-apidoc to generate .rst API stubs from docstrings.

    Parameters
    ----------
    output_dir
        Directory that receives the generated .rst stubs.
    src_dir
        Root directory of the Python package.
    exclude
        Sub-paths to exclude from documentation.
    force
        Overwrite existing stubs when True.
    """
    cmd = ["sphinx-apidoc", "-o", output_dir, src_dir]

    if isinstance(exclude, list):
        cmd.extend(exclude)

    cmd += [
        "-e",        # one page per module
        "--no-toc",  # suppress modules.rst
        "-M",        # module first
    ]

    if force:
        cmd.append("-f")

    subprocess.check_call(cmd)
    print("sphinx-apidoc completed successfully.")


def run_sphinx_build(docs_dir=DOCS_DIR, build_dir=BUILD_DIR, builder="html"):
    """
    Run sphinx-build for a one-shot documentation build.

    Parameters
    ----------
    docs_dir
        Directory containing conf.py.
    build_dir
        Output directory.
    builder
        Sphinx builder name (html, latex, dirhtml ...).
    """
    cmd = ["sphinx-build", "-b", builder, docs_dir, build_dir]
    subprocess.check_call(cmd)
    print(f"sphinx-build ({builder}) completed successfully.")


def run_sphinx_autobuild(docs_dir=DOCS_DIR, build_dir=BUILD_DIR, port=8000):
    """
    Run sphinx-autobuild for live preview with hot-reload.

    Parameters
    ----------
    docs_dir
        Directory containing conf.py.
    build_dir
        Output directory.
    port
        Local port to serve the HTML site.
    """
    cmd = [
        "sphinx-autobuild",
        docs_dir,
        build_dir,
        "--open-browser",
        "--port", str(port),
        "--watch", docs_dir,
    ]
    subprocess.check_call(cmd)


def run_script(script_path):
    """Run an auxiliary Python script."""
    subprocess.check_call(["python", script_path])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build or live-preview the Sphinx documentation.")
    parser.add_argument("--no-autobuild", action="store_true",
                        help="Use sphinx-build instead of sphinx-autobuild.")
    parser.add_argument("--clean-only", action="store_true",
                        help="Delete the build directory and exit.")
    parser.add_argument("--force-apidoc", action="store_true",
                        help="Force sphinx-apidoc to overwrite existing stubs.")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port for sphinx-autobuild (default: 8000).")
    parser.add_argument("--build-bib", action="store_true",
                        help="Regenerate bibliography from Zotero before building.")
    args = parser.parse_args()

    delete_build()

    if args.clean_only:
        raise SystemExit(0)

    if args.build_bib:
        run_script("build_bibliography.py")

    run_sphinx_apidoc(
        output_dir=API_OUTPUT_DIR,
        src_dir=SRC_DIR,
        exclude=EXCLUDE_MODULES,
        force=args.force_apidoc,
    )

    if args.no_autobuild:
        run_sphinx_build(docs_dir=DOCS_DIR, build_dir=BUILD_DIR)
    else:
        run_sphinx_autobuild(docs_dir=DOCS_DIR, build_dir=BUILD_DIR, port=args.port)
