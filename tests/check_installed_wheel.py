"""Check wheel contents from an isolated install, outside the source checkout.

Run ``poetry build -f wheel`` first, then pass the wheel to this script.
The clean environment intentionally has no numerical dependencies: this checks
the lazy package import and shipped source/case resources. Numerical behavior
and model imports are covered by pytest in the locked Poetry environment.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import tempfile
import venv


def main():
    """Install the specified wheel without resolving or modifying dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve(strict=True)
    source_root = Path(__file__).resolve().parents[1] / "src/nozzle_hyperbolic"
    expected = {
        path.relative_to(source_root).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix in (".py", ".yaml")
    }
    with tempfile.TemporaryDirectory(prefix="nozzle-wheel-") as temporary:
        root = Path(temporary)
        expected_path = root / "expected.json"
        expected_path.write_text(json.dumps(expected), encoding="utf-8")
        venv.EnvBuilder(with_pip=True, system_site_packages=False).create(root / "venv")
        windows_python = root / "venv/Scripts/python.exe"
        python = windows_python if windows_python.exists() else root / "venv/bin/python"
        subprocess.run(
            [str(python), "-m", "pip", "install", "--no-deps", str(wheel)], check=True
        )
        code = """
from pathlib import Path
from hashlib import sha256
from importlib.resources import files
import json
import sys
import nozzle_hyperbolic
installed = Path(nozzle_hyperbolic.__file__).resolve()
assert installed.is_relative_to(Path(sys.prefix)), installed
expected = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
assert sum(name.endswith('.yaml') for name in expected) == 20
resources = files('nozzle_hyperbolic')
for name, digest in expected.items():
    assert sha256(resources.joinpath(name).read_bytes()).hexdigest() == digest, name
assert not any(name.startswith(('jax', 'matplotlib', 'nozzle_hyperbolic.models')) for name in sys.modules)
print('Installed wheel: isolated import and all source/20 case resources passed')
"""
        subprocess.run(
            [str(python), "-I", "-c", code, str(expected_path)], cwd=root, check=True
        )


if __name__ == "__main__":
    main()
