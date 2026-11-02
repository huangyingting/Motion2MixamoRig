"""Pinned upstream checkouts using an existing Python; never replace its ML stack."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from . import paths
from .models import get_model_spec

def setup_model(name: str, *, install_deps: bool = False) -> Path:
    spec = get_model_spec(name)
    if spec["adapter"] != "external":
        raise ValueError("GVHMR is installed with the base project; use m2mr doctor")
    minimum = tuple(map(int, spec["python_min"].split(".")))
    if sys.version_info[:2] < minimum:
        raise ValueError(f"{name} needs Python {spec['python_min']} or newer")
    from .extractors import get_extractor

    extractor = get_extractor(name)
    root = extractor.root
    root.parent.mkdir(parents=True, exist_ok=True)
    if not root.exists():
        subprocess.run(["git", "clone", spec["upstream"], str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "checkout", "--detach", spec["revision"]], check=True)
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != spec["revision"]:
        raise ValueError(f"{root} is at {revision}; expected {spec['revision']}; existing checkout untouched")
    if spec["submodules"]:
        subprocess.run([
            "git", "-C", str(root), "submodule", "update", "--init", "--depth", "1",
            *spec["submodules"],
        ], check=True)
    configured = extractor.python
    if not configured.is_file():
        venv = root / ".venv"
        if sys.prefix == sys.base_prefix:
            raise ValueError("run setup-model from the project's venv, not system Python")
        if venv.exists() or venv.is_symlink():
            raise ValueError(f"existing runtime {venv} is incomplete; not replacing it")
        venv.symlink_to(Path(sys.prefix), target_is_directory=True)
        configured = extractor.python
    if install_deps:
        # Explicit opt-in; --no-deps prevents upstream torch/numpy pins replacing working packages.
        installed = json.loads(subprocess.check_output([
            str(configured), "-c",
            "import json,importlib.metadata as m; "
            "print(json.dumps([d.metadata['Name'].lower().replace('-', '_') for d in m.distributions()]))",
        ], text=True))
        missing = [
            requirement for requirement in spec.get("packages", ())
            if re.split(r"[\s<>=!~\[]", requirement, maxsplit=1)[0].lower().replace("-", "_") not in installed
        ]
        if missing:
            subprocess.run([
                str(configured), "-m", "pip", "install", "--no-deps", *missing,
            ], check=True)
    checks = extractor.check()
    receipt = {
        "model": name, "revision": revision, "python": str(configured),
        "checks": [{"label": c.label, "ok": c.ok, "hint": c.hint} for c in checks],
        "assets_policy": "Use upstream licensed downloads; no provider agreement or login is automated.",
        "inference_verified": False,
    }
    destination = root / "setup.json"
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
    return destination
