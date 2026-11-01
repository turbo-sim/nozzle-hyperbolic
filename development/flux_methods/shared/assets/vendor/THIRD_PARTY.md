# Offline report runtime

This report vendors the unmodified Plotly.js 2.35.2 browser bundle from the
repository's retained verified assets. It is distributed under the MIT license,
preserved in [PLOTLY_LICENSE.txt](PLOTLY_LICENSE.txt).

The decompressed bundle SHA-256 is
`6d21266ce1bd7d9e5ab4e115989c70c20de0382fd973a8f26ab58619eba4d603`.
The report generator checks that digest and embeds both the runtime and license.
Reading a generated report requires no server, CDN, fonts, or network requests.
Plotly numerical arrays and the campaign evidence are embedded in the HTML.

## Offline mathematics

The theory pages also vendor the existing MathJax 3.2.2 TeX-to-SVG bundle
from the repository architecture viewer. The verified unmodified runtime is
`mathjax-3.2.2-tex-svg.js.gz`; its provenance and decompressed SHA-256 are in
[mathjax-source.json](mathjax-source.json). The Apache 2.0 license is preserved
in [MATHJAX_LICENSE.txt](MATHJAX_LICENSE.txt) and embedded in generated HTML.
The SVG font cache is local and no external font, autoload extension or menu
is required to typeset the supplied equations.
