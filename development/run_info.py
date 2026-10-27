"""
Reproducibility record of a run: <run folder>/run_info.yaml (+ git_diff.patch).

Records what the resolved settings file cannot: the code version (git commit,
uncommitted changes as a patch, jaxprop version), the software versions, the
machine (CPU, affinity, power source — battery runs are ~1.8× slower), the
environment variables that change a run, and checksums of the input files
(restart solution, EOS tables).
"""

import datetime
import hashlib
import os
import pathlib
import platform
import subprocess
import sys

import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent
ENV_KEYS = ("SETTINGS_YAML", "SRC_SCALE", "XLA_FLAGS", "JAX_ENABLE_X64",
            "JAX_PLATFORMS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "MPLBACKEND")
CODE_SUFFIXES = (".py", ".yaml", ".yml", ".toml", ".cfg", ".ini")
CODE_GLOBS = [f"*{s}" for s in CODE_SUFFIXES]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _git(args, cwd):
    try:
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.decode("utf-8", "replace") if r.returncode in (0, 1) else None


def git_state(repo, patch_path=None):
    """Commit, branch and local changes of `repo`.  With patch_path, the
    uncommitted changes to code and config files (tracked diff + new files,
    results folders excluded) are written there, so the code can be rebuilt."""
    commit = (_git(["rev-parse", "HEAD"], repo) or "").strip()
    if not commit:
        return {"repo": str(repo), "commit": None}
    status = _git(["status", "--porcelain", "--untracked-files=all"], repo) or ""
    changed = [l for l in status.splitlines() if l.strip()]
    info = {
        "repo": str(repo),
        "commit": commit,
        "branch": (_git(["rev-parse", "--abbrev-ref", "HEAD"], repo) or "").strip(),
        "dirty": bool(changed),
        "n_changed_files": len(changed),
    }
    if patch_path is not None and changed:
        # text code/config changes only (no figures, no deleted files)
        parts = [_git(["diff", "HEAD", "--diff-filter=d", "--", *CODE_GLOBS], repo) or ""]
        for line in changed:
            if not line.startswith("??"):
                continue
            rel = line[3:].strip().strip('"')
            p = repo / rel
            if (p.suffix in CODE_SUFFIXES and "results" not in rel
                    and p.is_file() and p.stat().st_size < 2_000_000):
                parts.append(_git(["diff", "--no-index", "--", "/dev/null", rel], repo) or "")
        pathlib.Path(patch_path).write_bytes("".join(parts).encode("utf-8"))
        info["patch"] = pathlib.Path(patch_path).name
    return info


def _package(name):
    try:
        from importlib import metadata
        return metadata.version(name)
    except Exception:
        return None


def _power():
    """AC / battery (Windows only)."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class _SPS(ctypes.Structure):
        _fields_ = [("ACLineStatus", wintypes.BYTE), ("BatteryFlag", wintypes.BYTE),
                    ("BatteryLifePercent", wintypes.BYTE), ("SystemStatusFlag", wintypes.BYTE),
                    ("BatteryLifeTime", wintypes.DWORD), ("BatteryFullLifeTime", wintypes.DWORD)]
    s = _SPS()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
        return None
    ac = {0: "battery", 1: "AC"}.get(s.ACLineStatus & 0xFF, "unknown")
    pct = s.BatteryLifePercent & 0xFF
    return {"source": ac, "battery_percent": None if pct == 255 else pct}


def _affinity():
    """Logical CPUs this process may run on, as a compact range string."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            proc, sysm = ctypes.c_size_t(), ctypes.c_size_t()
            k32.GetProcessAffinityMask.argtypes = [wintypes.HANDLE,
                                                   ctypes.POINTER(ctypes.c_size_t),
                                                   ctypes.POINTER(ctypes.c_size_t)]
            if not k32.GetProcessAffinityMask(k32.GetCurrentProcess(),
                                              ctypes.byref(proc), ctypes.byref(sysm)):
                return None
            cpus = [i for i in range(64) if proc.value >> i & 1]
        else:
            cpus = sorted(os.sched_getaffinity(0))
    except Exception:
        return None
    out, start = [], None
    for i, c in enumerate(cpus):
        if start is None:
            start = c
        if i + 1 == len(cpus) or cpus[i + 1] != c + 1:
            out.append(f"{start}" if start == c else f"{start}-{c}")
            start = None
    return ",".join(out)


def _cpu_name():
    if sys.platform == "win32":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or None


def write_run_info(out_dir, cfg, *, resolved_file=None, input_files=None, extra=None):
    """Write <out_dir>/run_info.yaml (and git_diff.patch when the code has
    uncommitted changes).  input_files: {label: path} hashed with sha256."""
    out_dir = pathlib.Path(out_dir)
    s = cfg._settings
    info = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "command": " ".join([sys.executable] + sys.argv),
        "cwd": os.getcwd(),
        "settings": {
            "case_file": str(s.path),
            "chain": [str(p) for p in s.chain],
            "overrides": dict(s.overrides),
            "resolved_file": None if resolved_file is None else pathlib.Path(resolved_file).name,
        },
        "code": {"nozzle_hyperbolic": git_state(REPO, out_dir / "git_diff.patch")},
        "versions": {
            "python": platform.python_version(),
            **{k: _package(k) for k in ("jax", "jaxlib", "numpy", "scipy", "jaxprop",
                                        "CoolProp", "equinox")},
        },
        "machine": {
            "host": platform.node(),
            "os": platform.platform(),
            "cpu": _cpu_name(),
            "logical_cpus": os.cpu_count(),
            "affinity": _affinity(),
            "power": _power(),
        },
        "env": {k: os.environ[k] for k in ENV_KEYS if k in os.environ},
        "inputs": {},
    }
    try:
        import jaxprop
        jp = pathlib.Path(jaxprop.__file__).resolve().parent.parent
        st = git_state(jp)
        if st.get("commit"):
            info["code"]["jaxprop"] = st
    except Exception:
        pass
    for label, p in (input_files or {}).items():
        p = pathlib.Path(p)
        info["inputs"][label] = ({"path": str(p.resolve()), "sha256": sha256(p)}
                                 if p.is_file() else {"path": str(p), "sha256": None})
    if extra:
        info.update(extra)
    path = out_dir / "run_info.yaml"
    path.write_bytes(yaml.safe_dump(info, sort_keys=False, allow_unicode=True,
                                    width=100).encode("utf-8"))
    return path
