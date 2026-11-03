"""Sphinx settings; build_docs.py is copied unchanged from Turbodash."""

import jax
from nozzle_hyperbolic import __version__

# Match the precision required by the documented model/solver interfaces.
jax.config.update("jax_enable_x64", True)

project = "Nozzle-hyperbolic"
author = "Andrea Cioffi, Alberto Bergamin and Roberto Agromayor"
copyright = "2026, Andrea Cioffi, Alberto Bergamin and Roberto Agromayor"
release = __version__

docs_build_config = {
    "source_dir": "source",
    "build_dir": "_build/html",
    "api_output_dir": "source/api",
    "src_dir": "../src/nozzle_hyperbolic",
    "exclude_modules": [],
    "sphinx_build_options": ["--fail-on-warning", "--keep-going"],
    "zotero_api_key_env": "ZOTERO_API_KEY",
    "zotero_group_id": None,
    "bibliography_file": "source/references/bibliography.bib",
}

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx_design",
]
source_suffix = {".rst": "restructuredtext", ".md": "myst"}
root_doc = "index"
myst_enable_extensions = ["amsmath", "colon_fence", "dollarmath"]
myst_heading_anchors = 3
autosummary_generate = True
autodoc_default_options = {"members": True, "show-inheritance": True}
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_ivar = True
autodoc_typehints = "description"


def _skip_reexported_members(app, what, name, obj, skip, options):
    """Document each object at its owning module, with one canonical target."""
    if what == "module":
        current = app.env.temp_data.get("autodoc:module")
        owner = getattr(obj, "__module__", None)
        if current and owner and current != owner:
            return True
    return None


def setup(app):
    app.connect("autodoc-skip-member", _skip_reexported_members)


# Only docs/source is read by Sphinx; docs/notes stays outside the site.
exclude_patterns = ["Thumbs.db", ".DS_Store"]
html_theme = "sphinx_book_theme"
html_title = project
html_static_path = ["source/_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "repository_url": "https://github.com/turbo-sim/nozzle-hyperbolic",
    "repository_branch": "roberto-branch",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "path_to_docs": "docs/source",
    "show_toc_level": 2,
}
