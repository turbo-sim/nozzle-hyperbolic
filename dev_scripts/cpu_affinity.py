"""Pin the current process to a set of logical CPUs (single-phase drivers).

On the i7-13850HX laptop logical CPUs 0-15 are P-cores and 16-27 E-cores
(~1.9x slower single-thread); unpinned JAX runs land on either and their
timings flip by ~2x.  Call pin_cpus("0-15") before any JAX computation.
(main_two_fluid.py keeps its own copy, _pin_cpus.)
"""
import os
import sys


def pin_cpus(spec):
    """Restrict this process to the logical CPUs in `spec` ("0-15", "0,2,4-7")."""
    cpus = set()
    for part in str(spec).split(","):
        a, _, b = part.strip().partition("-")
        cpus.update(range(int(a), int(b or a) + 1))
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        k32.SetProcessAffinityMask.restype = wintypes.BOOL
        ok = k32.SetProcessAffinityMask(k32.GetCurrentProcess(),
                                        sum(1 << c for c in cpus))
        if not ok:
            raise OSError(f"SetProcessAffinityMask failed (error {ctypes.get_last_error()})")
    else:
        os.sched_setaffinity(0, cpus)
    print(f"CPU affinity: pinned to logical CPUs {spec}")
