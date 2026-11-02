"""Write and merge the source-prefixed run report for a pipeline run."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .paths import find_run_file, run_output_path


def completed_run(
    directory: Path, source: Path, rig: Path, extractor: str, *,
    hands: str | None = None, person_track_sha256: str | None = None,
    foot_lock: bool | None = None, hand_smooth_s: float | None = None,
) -> dict | None:
    """Reuse complete products only when their source, rig and subject match."""
    from .source_metadata import file_digest

    path = find_run_file(directory, "run.json")
    if not path.is_file():
        return None
    report = json.loads(path.read_text())
    if report.get("status") != "completed":
        return None
    expected = {
        "source_sha256": file_digest(source),
        "extractor": extractor,
        "person_track_sha256": person_track_sha256,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"{directory}: {key} differs; use --force to replace this output")
    if Path(report["rig"]).resolve() != Path(rig).resolve() or (
        report.get("rig_sha256") and report["rig_sha256"] != file_digest(rig)
    ):
        raise ValueError(f"{directory}: rig differs; use --force to replace this output")
    if hands is not None and report.get("hands") != hands:
        raise ValueError(f"{directory}: finger model differs; use --force to replace this output")
    requested = report.get("foot_lock_requested")
    if foot_lock is not None and requested is not None and requested != foot_lock:
        raise ValueError(f"{directory}: foot lock differs; use --force to replace this output")
    if hand_smooth_s is not None and report.get("hand_smooth_s") not in (None, hand_smooth_s):
        raise ValueError(f"{directory}: finger smoothing differs; use --force to replace this output")
    outputs = report.get("outputs", {})
    for role in ("skeleton_motion", "mixamo_rotations", "mixamo_character_glb",
                 "mixamo_motion_bvh", "diagnostics"):
        name = outputs.get(role)
        if not isinstance(name, str) or Path(name).name != name or not (directory / name).is_file():
            return None
    return report


def remove_video_previews(directory: Path) -> list[Path]:
    """Drop only the portable preview videos listed in a model's run report."""
    directory = Path(directory)
    report = json.loads(find_run_file(directory, "run.json").read_text())
    videos = report.get("outputs", {}).get("videos", {})
    files = []
    for name in videos.values():
        if not isinstance(name, str) or Path(name).name != name or Path(name).suffix.lower() != ".mp4":
            raise ValueError(f"{directory}: invalid preview filename {name!r}")
        files.append(directory / name)
    for file in files:
        file.unlink(missing_ok=True)
    if "videos" in report.get("outputs", {}):
        report["outputs"].pop("videos")
        if not report["outputs"].get("images"):
            report.pop("presentation", None)
        write_run_json(directory, report)
    return files


def copy_run_deliveries(
    old: Path, directory: Path, source: Path, metadata: Path, *,
    previews: bool = True, overwrite: bool = False,
) -> dict:
    """Copy portable products only; leave extraction caches and originals untouched."""
    old = Path(old).resolve()
    report = json.loads(find_run_file(old, "run.json").read_text())
    directory.mkdir(parents=True, exist_ok=overwrite)
    fixed = {
        "skeleton_motion": "skeleton_motion.npz", "mixamo_rotations": "mixamo_rotations.npz",
        "mixamo_character_glb": "mixamo_character.glb", "diagnostics": "diagnostics.json",
        "person_track": "person_track.npz",
    }
    outputs = {}
    for role, value in report["outputs"].items():
        panels = value if isinstance(value, dict) else {None: value}
        for panel, filename in panels.items():
            if role == "videos" and (not previews or panel not in ("compare", "before_after_360_compare")):
                continue
            original = (old / filename).resolve()
            if not original.is_relative_to(old) or not original.is_file():
                continue
            if role in fixed:
                destination = run_output_path(directory, source, fixed[role])
            elif role == "mixamo_motion_bvh":
                destination = directory / f"{source.stem}.bvh"
            else:
                name = original.name if original.name.startswith(f"{source.stem}_") else f"{source.stem}_{original.name}"
                destination = directory / name
            shutil.copy2(original, destination)
            if panel is None:
                outputs[role] = destination.name
            else:
                outputs.setdefault(role, {})[panel] = destination.name
    for role in ("skeleton_motion", "mixamo_rotations", "mixamo_character_glb"):
        if role not in outputs:
            raise FileNotFoundError(f"{old}: missing required {role}")
    bvh = directory / f"{source.stem}.bvh"
    from .mixamo.bvh import canonicalize_bvh, has_mixamo_joint_names, write_bvh_from_npz

    if not bvh.exists():
        write_bvh_from_npz(directory / outputs["mixamo_rotations"], bvh, rig_path=Path(report["rig"]))
    elif has_mixamo_joint_names(bvh):
        canonicalize_bvh(bvh, bvh)
    outputs["mixamo_motion_bvh"] = bvh.name
    if "diagnostics" in outputs:
        path = directory / outputs["diagnostics"]
        diagnostics = json.loads(path.read_text())
        diagnostics["source"] = outputs["mixamo_character_glb"]
        path.write_text(json.dumps(diagnostics, indent=2) + "\n")
    report.update(
        outputs=outputs, organized_from=str(old), original_cache=report.get("original_cache") or str(old / "cache"),
        source_metadata=str(metadata),
        source_sha256=json.loads(metadata.read_text())["file"]["sha256"],
    )
    if "person_track" in outputs:
        report["person_track"] = str(directory / outputs["person_track"])
    report_path = write_run_json(directory, report)
    from .source_metadata import record_source_run

    record_source_run(metadata, report_path, report.get("extractor") or "unknown", "completed", operation="delivery_reuse")
    return report


def repair_delivery_references(directory: Path) -> None:
    report = json.loads(find_run_file(directory, "run.json").read_text())
    diagnostics_path = directory / report["outputs"].get("diagnostics", "diagnostics.json")
    if diagnostics_path.is_file():
        diagnostics = json.loads(diagnostics_path.read_text())
        diagnostics["source"] = report["outputs"]["mixamo_character_glb"]
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n")


def write_run_json(run_dir: Path, report: dict[str, Any]) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    source = report.get("image") or report.get("video")
    path = (
        run_output_path(run_dir, Path(source), "run.json")
        if source else find_run_file(run_dir, "run.json")
    )
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def write_failed_report(
    run_dir: Path | None,
    *,
    stage: str,
    error_code: str,
    message: str,
    extra: dict[str, Any] | None = None,
) -> Path | None:
    """Write a failed `run.json`. Keeps any earlier fields already on disk."""
    if run_dir is None:
        return None
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}
    existing = find_run_file(run_dir, "run.json")
    if existing.is_file():
        try:
            loaded = json.loads(existing.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                report.update(loaded)
        except (OSError, json.JSONDecodeError):
            pass
    report.update(
        {
            "status": "failed",
            "stage": stage,
            "error_code": error_code,
            "message": message,
        }
    )
    if extra:
        for key, value in extra.items():
            if key not in {"status", "stage", "error_code", "message"}:
                report[key] = value
    if existing.is_file():
        existing.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return existing
    return write_run_json(run_dir, report)
