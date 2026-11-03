"""Opt-in persistent JAX caching and observed compile/load reporting."""

import json
import os
from pathlib import Path
import sys
from threading import Lock, local
from time import perf_counter

import jax
from jax import monitoring


_BACKEND_EVENT = "/jax/core/compile/backend_compile_duration"
_CACHE_HIT = "/jax/compilation_cache/cache_hits"
_active_report = None
_listeners_registered = False


def configure_compilation_cache(directory):
    """Enable disk storage of eligible executables before creating JAX arrays.

    Parameters
    ----------
    directory : str or pathlib.Path
        Local cache directory. ``JAX_COMPILATION_CACHE_DIR``, when set, takes
        precedence. Keep this directory between Python runs to permit reuse.

    Returns
    -------
    pathlib.Path
        Resolved cache location. JAX creates it when writing eligible entries.

    Notes
    -----
    Sets the minimum compilation time to zero and disables the entry-size
    threshold, so even small eligible kernels can be cached. Call before the
    first compilation; this does not clear existing in-memory or disk caches.
    JAX's explicit ``JAX_ENABLE_COMPILATION_CACHE=false`` setting is respected.
    Executables involving host callbacks may be ineligible for persistent reuse.
    """
    path = Path(os.environ.get("JAX_COMPILATION_CACHE_DIR") or directory)
    path = path.expanduser().resolve()
    jax.config.update("jax_compilation_cache_dir", str(path))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    return path


def _start(event, value, **metadata):
    if event == _BACKEND_EVENT and _active_report is not None:
        _active_report._start(metadata.get("fun_name", "unknown"))


def _event(event, **metadata):
    if event == _CACHE_HIT and _active_report is not None:
        stack = getattr(_active_report._threads, "stack", [])
        if stack:
            stack[-1]["disk_hit"] = True


def _finish(event, duration, **metadata):
    if event == _BACKEND_EVENT and _active_report is not None:
        _active_report._finish(float(duration))


class CompilationReport:
    """Observe JAX compile/cache requests without changing numerical kernels.

    Parameters
    ----------
    report_path : str or pathlib.Path, optional
        Write a JSON summary on context exit, including exceptional exits.
    verbose : bool, optional
        Print individual backend compile/load events when True. The default,
        False, prints one final status line with the observed compile/load time.
    stream : file-like, optional
        Console destination, defaulting to current standard output.

    Attributes
    ----------
    stats : dict
        Observed request counts, persistent hits, compile/load elapsed time and
        whole-context wall time. Available after leaving the context.

    Notes
    -----
    Uses public ``jax.monitoring`` listeners and the event names emitted by
    JAX 0.6.2. A backend request can compile or load a cached executable; its
    duration includes lookup/serialization and is not pure compiler CPU time.
    In-process reuse usually emits neither event. Zero observed requests is
    reported as such, rather than assumed to prove that a kernel was executed.

    Only one report can be active per process. Listeners are registered once
    and become inactive on exit; listeners owned by other callers are retained.
    Kernels compile lazily, so events can occur during setup, solving or output.
    Synchronize device results before leaving the context when interpreting
    whole-context wall time. Durations are sums of observed backend events.
    """

    def __init__(self, report_path=None, *, verbose=False, stream=None):
        self.report_path = None if report_path is None else Path(report_path)
        self.verbose = verbose
        self.stream = sys.stdout if stream is None else stream
        self._threads = local()
        self._lock = Lock()
        self.stats = {}

    def __enter__(self):
        global _active_report, _listeners_registered
        if _active_report is not None:
            raise RuntimeError("Only one CompilationReport can be active")
        if not _listeners_registered:
            monitoring.register_scalar_listener(_start)
            monitoring.register_event_listener(_event)
            monitoring.register_event_duration_secs_listener(_finish)
            _listeners_registered = True
        self.stats = dict(
            jax_version=jax.__version__,
            cache_directory=jax.config.jax_compilation_cache_dir,
            persistent_cache_enabled=bool(
                jax.config.jax_enable_compilation_cache
                and jax.config.jax_compilation_cache_dir
            ),
            backend_requests=0,
            persistent_cache_hits=0,
            non_cached_requests=0,
            compile_or_load_s=0.0,
            non_cached_request_s=0.0,
            cached_request_s=0.0,
        )
        self._threads.stack = []
        self._started = perf_counter()
        _active_report = self
        if self.verbose:
            print(
                f"[JAX] Persistent cache: {self.stats['cache_directory']} "
                f"(enabled={self.stats['persistent_cache_enabled']})",
                file=self.stream,
                flush=True,
            )
        return self

    def _start(self, name):
        if not hasattr(self._threads, "stack"):
            self._threads.stack = []
        self._threads.stack.append(dict(name=name, disk_hit=False))
        if self.verbose:
            print(f"[JAX] Compile/load START: {name}", file=self.stream, flush=True)

    def _finish(self, elapsed):
        stack = getattr(self._threads, "stack", [])
        if not stack:
            return
        request = stack.pop()
        hit = request["disk_hit"]
        with self._lock:
            self.stats["backend_requests"] += 1
            self.stats["persistent_cache_hits" if hit else "non_cached_requests"] += 1
            self.stats["compile_or_load_s"] += elapsed
            self.stats["cached_request_s" if hit else "non_cached_request_s"] += elapsed
        if self.verbose:
            outcome = (
                "REUSED executable from disk" if hit else "no disk hit; build requested"
            )
            print(
                f"[JAX] Compile/load END: {request['name']} | {elapsed:.6f} s | {outcome}",
                file=self.stream,
                flush=True,
            )

    def __exit__(self, exc_type, exc_value, traceback):
        global _active_report
        _active_report = None
        self.stats["wall_s"] = perf_counter() - self._started
        self.stats["completed"] = exc_type is None
        builds = self.stats["non_cached_requests"]
        hits = self.stats["persistent_cache_hits"]
        if builds and hits:
            status = f"compiled {builds} and loaded {hits} cached executables"
        elif builds:
            status = f"compiled {builds} executables"
        elif hits:
            status = f"loaded {hits} cached executables"
        else:
            status = "no new compile/load events observed"
        suffix = " (run failed)" if exc_type is not None else ""
        print(
            f"[JAX] {status}; compile/load time {self.stats['compile_or_load_s']:.2f} s{suffix}.",
            file=self.stream,
            flush=True,
        )
        if self.report_path is not None:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            self.report_path.write_text(
                json.dumps(self.stats, indent=2) + "\n", encoding="utf-8"
            )
        return False
