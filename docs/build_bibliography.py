"""
build_bibliography.py
=====================
Export the shared Zotero group library to a BibTeX file used by
sphinxcontrib-bibtex.

Usage
-----
    python build_bibliography.py

    # Or via build_docs.py:
    python build_docs.py --build-bib
"""

from pyzotero import zotero
import bibtexparser
from bibtexparser.bwriter import BibTexWriter

API_KEY     = None          # leave None for public groups
GROUP_ID    = "5252389"
OUTPUT_FILE = "source/bibliography.bib"


def ensure_unique_keys(entries):
    """Guarantee that every BibTeX entry has a unique citation key."""
    seen = set()
    for entry in entries:
        base = entry.get("ID", "no_id")
        key  = base
        suffix = 1
        while key in seen:
            key = f"{base}_{suffix}"
            suffix += 1
        entry["ID"] = key
        seen.add(key)
    return entries


def prioritize_doi_over_url(entries):
    """Remove url field when a doi is present to keep references clean."""
    for entry in entries:
        if "doi" in entry and "url" in entry:
            del entry["url"]
    return entries


def replace_latex_commands(bibtex_str):
    """Replace LaTeX macros with Unicode equivalents."""
    replacements = {
        r"\textbar": "|",
    }
    for latex_cmd, unicode_char in replacements.items():
        bibtex_str = bibtex_str.replace(latex_cmd, unicode_char)
    return bibtex_str


def export_zotero_to_bibtex(group_id, api_key=None, output_file=OUTPUT_FILE):
    """
    Fetch the Zotero group library and write it to a BibTeX file.

    Parameters
    ----------
    group_id
        Zotero Group ID (visible in the group URL on zotero.org).
    api_key
        Zotero API key. None works for public groups.
    output_file
        Destination path for the generated .bib file.
    """
    zot = zotero.Zotero(group_id, "group", api_key)

    bibtex_db = zot.everything(zot.items(format="bibtex"))
    ensure_unique_keys(bibtex_db.entries)
    prioritize_doi_over_url(bibtex_db.entries)

    db = bibtexparser.bibdatabase.BibDatabase()
    db.entries = bibtex_db.entries

    writer = BibTexWriter()
    bibtex_str = bibtexparser.dumps(db, writer)
    bibtex_str = replace_latex_commands(bibtex_str)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(bibtex_str)


if __name__ == "__main__":
    export_zotero_to_bibtex(GROUP_ID, API_KEY, OUTPUT_FILE)
    print(f"Bibliography written to: {OUTPUT_FILE}")
