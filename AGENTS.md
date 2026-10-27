# Project instructions

- For every new Newton, LM, HYBR, or SLSQP investigation, use the custom
  PySolverView backends with their initial-step/radius controls. Verify and
  record `library="pysolver"` and the actual backend method. Do not silently
  substitute vanilla SciPy/MINPACK solvers.
- `development_new/newton_solve/minpack_solvers.py` and the old
  `compare_root_methods.py` preserve historical SciPy experiments; they are
  not the default for new investigations.
- Audit convergence with the original spatial residual independently of a
  backend's native success flag. Keep failed runs and accepted/trial history
  distinctions explicit.
- Use `$...$` for inline Markdown math and `$$...$$` for display math.
