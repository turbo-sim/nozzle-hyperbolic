# Project instructions

- For every new Newton, LM, HYBR, or SLSQP investigation, use the custom
  PySolverView backends with their initial-step/radius controls. Verify and
  record `library="pysolver"` and the actual backend method. Do not silently
  substitute vanilla SciPy/MINPACK solvers.
- The retained `development_new/newton_solve/scripts/investigate_ptc.py` comparison
  uses custom Newton, physical-storage PTC, and implicit integration. Historical
  SciPy wrappers and abandoned continuation drivers have been removed; their
  findings are summarized in that directory's README.md.
- Audit convergence with the original spatial residual independently of a
  backend's native success flag. Keep failed runs and accepted/trial history
  distinctions explicit.
- Use `$...$` for inline Markdown math and `$$...$$` for display math.

- The pytest suite covers production code under `src/` only. Do not add tests
  for `development_new/` experiments or import development scripts from tests.
  Keep experiment verification in their executable audits and reports.
