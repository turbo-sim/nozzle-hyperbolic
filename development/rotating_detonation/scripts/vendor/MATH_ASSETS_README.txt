The report embeds equations prerendered from LaTeX with MathJax 3.2.2.
Each SVG contains its own glyph paths and uses the document text color.
No MathJax runtime, CDN, external font, or Node installation is required to
regenerate or view the Python-generated report from these retained SVGs.

To change the formulas, edit equations.json and run from this directory:

  node --input-type=module -e "import('./render_math.mjs').then(m=>m.render()).then(console.log)"

The renderer loads the retained official npm archive directly in memory;
there is no npm installation, network access, or unpacked dependency tree.
The source URL, archive/source/renderer SHA256 hashes, exact LaTeX, and each
SVG hash are recorded in mathjax-render-manifest.json. The report and its
structural audit verify the retained SVGs. MATHJAX_LICENSE.txt contains the
MathJax Apache-2.0 license. The separate Plotly files keep their own license.
