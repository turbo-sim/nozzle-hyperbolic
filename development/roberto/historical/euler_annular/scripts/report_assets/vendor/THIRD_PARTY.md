# Offline report libraries

The compressed JavaScript files are build inputs. The HTML generator embeds
their decompressed contents, together with the complete applicable license.
No library is fetched while reading either generated page.

| Library | Version | Source | License |
| --- | --- | --- | --- |
| Plotly.js | 2.35.2 | Unmodified `plotly.min.js` from the existing local Plotly Python 5.24.1 distribution | MIT; see `PLOTLY_LICENSE.txt` |
| MathJax | 3.2.2 | Official pinned [TeX/SVG component](https://raw.githubusercontent.com/mathjax/MathJax/3.2.2/es5/tex-svg.js) | Apache-2.0; see `MATHJAX_LICENSE.txt` |

Plotly's uncompressed SHA-256 is
`6d21266ce1bd7d9e5ab4e115989c70c20de0382fd973a8f26ab58619eba4d603`.
MathJax's source URL, size and uncompressed SHA-256 are recorded in
`mathjax-source.json`. MathJax uses its combined TeX/SVG component and embedded
glyph paths, as described by the [official component documentation](https://docs.mathjax.org/en/v3.2/web/components/combined.html).
Automatic TeX extension loading and the contextual menu are disabled. The
document's content policy forbids external scripts, fonts and connections.

The report's CSS tokens and theme behavior are inspired by the local TurboDash
optimization workbench. Its branding and application code are not embedded.
