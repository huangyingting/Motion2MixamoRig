"""Optional upstream extractors with configurable Python runtimes."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .. import paths
from ..models import MODEL_SPECS, get_model_spec
from .base import CheckResult

_PROJECTS = {name: spec["upstream"] for name, spec in MODEL_SPECS.items() if spec["adapter"] == "external"}
_ENTRYPOINTS = {name: MODEL_SPECS[name]["entrypoint"] for name in _PROJECTS}
_ASSETS = {name: tuple(MODEL_SPECS[name]["assets"]) for name in _PROJECTS}


class ExternalExtractor:
    """Run an upstream API in its own configured runtime, then normalize its output."""

    supports_images = False

    def __init__(self, name: str):
        if name not in _PROJECTS:
            raise ValueError(f"unknown external extractor: {name}")
        self.name = name
        self.spec = get_model_spec(name)
        self.supports_images = self.spec["supports_images"]
        self.hand_model = self.spec["hand_model"]
        self.weights_note = (
            f"prepare upstream assets in the {name} runtime; see {_PROJECTS[name]} "
            "and docs/model-extractors.md (provider terms apply)"
        )
        self.hand_weights_note = "native fingers; local MANO template required for Mixamo retargeting"

    @property
    def root(self) -> Path:
        return Path(
            os.environ.get(f"M2MR_{self.name.upper()}_ROOT", paths.MODEL_RUNTIMES / self.name)
        ).expanduser().resolve()

    @property
    def python(self) -> Path:
        default = self.root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        return Path(
            os.environ.get(f"M2MR_{self.name.upper()}_PYTHON", default)
        ).expanduser().absolute()

    def check(self, detect_hands: bool = False) -> list[CheckResult]:
        root, python = self.root, self.python
        hint = f"install {_PROJECTS[self.name]} separately; configure M2MR_{self.name.upper()}_ROOT/PYTHON"
        results = [
            CheckResult(
                f"{self.name} source     {root}",
                (root / _ENTRYPOINTS[self.name]).is_file(), hint,
            ),
            CheckResult(
                f"{self.name} Python     {python}",
                python.is_file() and os.access(python, os.X_OK),
                hint,
            ),
        ]
        if all(item.ok for item in results):
            try:
                probe = subprocess.run(
                    [str(python), "-c", "import numpy, torch, smplx"],
                    cwd=root, capture_output=True, text=True, timeout=30,
                )
                ok = probe.returncode == 0
                problem = probe.stderr.strip()[-1500:] or hint
            except (OSError, subprocess.TimeoutExpired) as exc:
                ok, problem = False, str(exc)
            results.append(
                CheckResult(
                    f"{self.name} core dependencies", ok, problem if not ok else "",
                )
            )
        for relative in _ASSETS[self.name]:
            asset = root / relative
            results.append(
                CheckResult(
                    f"{self.name} asset      {relative}", asset.is_file(),
                    f"prepare {asset} using the upstream setup guide; gated assets require your agreement",
                )
            )
        if detect_hands:
            if self.hand_model is None:
                results.append(
                    CheckResult(
                        "hand detection", False, "PromptHMR does not estimate fingers", section="hand",
                    )
                )
            else:
                from .hamer import MANO_HINT, mano_present

                results.append(
                    CheckResult("MANO finger template", mano_present(), MANO_HINT, section="hand")
                )
        return results

    def extract(
        self, source: Path, output_npz: Path, *, work_dir: Path,
        device: str = "cpu", verbose: bool = True, detect_hands: bool = False,
        person_track: Path | None = None,
    ) -> Path:
        source = Path(source).resolve()
        if paths.is_image(source):
            raise ValueError(f"{self.name} integration currently accepts videos only")
        if device not in self.spec["devices"]:
            raise ValueError(f"the upstream {self.name} requires one of {self.spec['devices']}")
        selection = None
        if person_track is not None:
            from ..person_tracking import SUPPORTED_EXTRACTORS, read_person_track

            if self.name not in SUPPORTED_EXTRACTORS:
                raise ValueError(f"{self.name} does not yet support explicit person selection")
            selection = read_person_track(person_track, source)
        missing = [item for item in self.check(detect_hands) if not item.ok]
        if missing:
            raise FileNotFoundError("\n".join(f"{item.label}: {item.hint}" for item in missing))
        work = Path(work_dir).resolve() / self.name
        if selection is not None:
            work /= f"person_{selection['person_track_sha256']}"
        work.mkdir(parents=True, exist_ok=True)
        params_path = work / "motion_params.npz"
        # Never accept stale parameters if this worker fails to publish a new result.
        params_path.unlink(missing_ok=True)
        command = [
            str(self.python), str(Path(__file__).with_name("_external_worker.py")),
            self.name, str(source), str(work), str(params_path), device,
        ]
        if selection is not None:
            command.append(str(Path(person_track).resolve()))
        subprocess.run(
            command,
            cwd=self.root, check=True,
            stdout=None if verbose else subprocess.DEVNULL,
        )
        from .native import convert_native_motion

        return convert_native_motion(
            params_path, output_npz, source=source, extractor=self.name,
            body_model=self.root / self.spec["body_asset"],
            model_type=self.spec["model_type"],
            detect_hands=detect_hands, device=device,
        )
