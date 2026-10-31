# Offline documentation runtime

The architecture viewer uses the unmodified **MathJax 3.2.2 TeX/SVG** browser
bundle, distributed under the Apache License 2.0. The full license is preserved
in [MATHJAX_LICENSE.txt](MATHJAX_LICENSE.txt); the original upstream URL, byte
count, and SHA-256 digest are recorded in [mathjax-source.json](mathjax-source.json).

The compressed bundle and metadata were copied from the repository's existing
verified report assets. They are maintained here so the documentation generator
does not depend on a development study directory. The generator validates the
decompressed bundle and license against the recorded digests, then embeds the
runtime and a downloadable copy of the license into `ARCHITECTURE.html`.

The combined bundle includes its TeX parser, SVG renderer, and SVG font data.
The viewer enables only bundled TeX packages and disables the browser menu and
autoload extensions. Opening the generated HTML requires no server, CDN, fonts,
or network requests. Only following an external source link can leave the file.
