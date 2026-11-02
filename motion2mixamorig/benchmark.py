"""Run available extractors on identical inputs without inventing an accuracy winner."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import checks, paths
from .extractors import extractor_names, get_extractor
from .models import get_model_spec


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _run_signatures(video: Path, runs: dict[str, Path]) -> dict:
    signatures = {}
    for name, folder in runs.items():
        path = paths.run_output_path(folder, video, "run.json")
        report = json.loads(path.read_text()) if path.is_file() else {}
        glb = folder / report.get("outputs", {}).get("mixamo_character_glb", "_missing_")
        signatures[name] = _digest(glb) if glb.is_file() else None
    return signatures


def summarize_run(run_dir: Path, source: Path) -> dict:
    report = json.loads(paths.run_output_path(run_dir, source, "run.json").read_text())
    outputs = report["outputs"]
    diagnostics = json.loads((run_dir / outputs["diagnostics"]).read_text())
    intersections = diagnostics["summary"]["self_intersection"]
    with np.load(run_dir / outputs["skeleton_motion"], allow_pickle=False) as data:
        joints = data["joints_3d"]
        fps = float(data["fps"])
    frames = len(joints)
    jerk = None
    if frames >= 4:
        relative = joints - joints[:, :1]
        jerk = float(np.linalg.norm(np.diff(relative, n=3, axis=0) * fps**3, axis=-1).mean())
    return {
        "frame_count": frames,
        "fps": fps,
        "duration_s": frames / fps,
        "self_intersection": intersections,
        "intersection_frame_fraction": intersections["frame_count"] / frames,
        "raw_root_relative_joint_jerk_m_s3": jerk,
        "quality_report": str(run_dir / outputs["diagnostics"]),
        "outputs": outputs,
    }


def run_benchmark(
    video: Path, rig: Path, *, models: tuple[str, ...] | None = None,
    device: str = "cuda", output_dir: Path | None = None, include_hands: bool = False,
    no_preview: bool = False,
    reuse_runs: dict[str, Path] | None = None,
    person_track: Path | None = None,
    force: bool = False,
) -> Path:
    from .pipeline import run_pipeline

    video, rig = Path(video).resolve(), Path(rig).resolve()
    selection = None
    if person_track is not None:
        from .person_tracking import SUPPORTED_EXTRACTORS, read_person_track

        selection = read_person_track(person_track, video)
        if any(name not in SUPPORTED_EXTRACTORS for name in (models or ("gvhmr", "exmr", "flowhmr"))):
            raise ValueError("selected-person benchmarks support gvhmr, exmr and flowhmr")
    for source, checker in ((video, checks.check_video), (rig, checks.check_rig)):
        if not source.is_file():
            raise FileNotFoundError(source)
        problem = checker(source, **({"person_track": person_track} if source == video and selection else {}))
        if problem:
            raise ValueError(f"{source.name}: {problem}")
    from .comparison import COMPARISON_MODELS, load_comparison_run, render_model_comparison

    names = tuple(dict.fromkeys(models if models is not None else COMPARISON_MODELS))
    if not names:
        raise ValueError("select at least one extractor")
    for name in names:
        get_extractor(name)
    from .source_metadata import save_source_metadata

    directory = Path(output_dir).resolve() if output_dir is not None else paths.source_output_dir(video)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = save_source_metadata(video, directory)
    destination = directory / f"{video.stem}_benchmark.json"
    previous = json.loads(destination.read_text()) if destination.is_file() else {}
    if not force and previous:
        if (
            previous.get("video_sha256") != _digest(video)
            or previous.get("rig_sha256") != _digest(rig)
            or previous.get("person_track_sha256") != (selection["person_track_sha256"] if selection else None)
        ):
            raise ValueError(f"{directory}: comparison inputs differ; use --force to replace outputs")
    comparison = {
        "schema": "m2mr-benchmark-v1",
        "video": str(video), "video_sha256": _digest(video),
        "source_metadata": str(metadata),
        "rig": str(rig), "rig_sha256": _digest(rig), "device": device,
        "ground_truth": None, "winner": None,
        "assessment": (
            "Intersections and raw joint jerk are quality proxies, not accuracy. "
            "No model can be declared better without ground truth or visual review. "
            "Native model postprocessing differs; all runs use the same Mixamo foot lock and ground policy."
        ),
        "not_applicable": {
            "mamma": "requires synchronized multi-camera footage and calibration; not a single-video extractor",
            "pear": "RGB MoCap's PEAR connector is live preview, not temporal offline motion export",
        },
        "runs": [],
        **({
            "person_id": selection["person_id"],
            "person_track_sha256": selection["person_track_sha256"],
        } if selection else {}),
    }

    def publish() -> None:
        destination.write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n")

    variants = [
        (
            paths.model_variant(name, get_model_spec(name)["hand_model"])
            if models is None or include_hands else name,
            name, models is None or include_hands,
        )
        for name in names
    ]
    reuse_runs = dict(reuse_runs or {})
    if models is None and "gvhmr_hands" in reuse_runs:
        if "gvhmr_hamer" in reuse_runs:
            raise ValueError("use only one of gvhmr_hamer and legacy gvhmr_hands")
        reuse_runs["gvhmr_hamer"] = reuse_runs.pop("gvhmr_hands")
    unknown = set(reuse_runs) - {label for label, _, _ in variants}
    if unknown:
        raise ValueError(f"unknown reused variants: {', '.join(sorted(unknown))}")
    if force and reuse_runs:
        raise ValueError("--force cannot be combined with --reuse-run")
    publish()
    body_runs: dict[str, Path] = {}
    from .run_report import completed_run, remove_video_previews

    for label, name, hands in variants:
        entry = {"model": name, "variant": label, "detect_hands": hands, "status": "running"}
        comparison["runs"].append(entry)
        publish()
        try:
            target = directory / name
            existing = None if force else completed_run(
                target, video, rig, name,
                hands=get_model_spec(name)["hand_model"] if hands else None,
                person_track_sha256=selection["person_track_sha256"] if selection else None,
                foot_lock=True,
            )
            if existing is not None:
                remove_video_previews(target)
                entry.update(status="completed", reused=True, run_dir=str(target), **summarize_run(target, video))
                publish()
                print(f"{label}: already completed")
                continue
            if label in reuse_runs:
                run_dir = Path(reuse_runs[label]).resolve()
                reused = json.loads(paths.run_output_path(run_dir, video, "run.json").read_text())
                if reused.get("person_track_sha256") != (selection["person_track_sha256"] if selection else None):
                    raise ValueError(f"{label}: reused run has another person selection")
                if hands and name in COMPARISON_MODELS:
                    load_comparison_run(run_dir, video, rig, name)
                else:
                    report = json.loads(paths.run_output_path(run_dir, video, "run.json").read_text())
                    if report.get("status") != "completed" or report.get("extractor") != name:
                        raise ValueError(f"{label}: incompatible or incomplete reused run")
                    if Path(report["video"]).resolve() != video or Path(report["rig"]).resolve() != rig:
                        raise ValueError(f"{label}: reused run has different inputs")
                from .run_report import copy_run_deliveries

                original_run = run_dir
                run_dir = target
                if original_run.resolve() != run_dir.resolve():
                    copy_run_deliveries(original_run, run_dir, video, metadata, previews=False, overwrite=True)
                remove_video_previews(run_dir)
                entry.update(
                    status="completed", reused=True, reused_from=str(original_run),
                    run_dir=str(run_dir), **summarize_run(run_dir, video),
                )
                if not hands:
                    body_runs[name] = run_dir
                publish()
                print(f"{label}: reused")
                continue
            missing = [item for item in get_extractor(name).check(hands) if not item.ok]
            if missing:
                entry.update(
                    status="unavailable",
                    reasons=[f"{item.label}: {item.hint}" for item in missing],
                )
            else:
                start = time.perf_counter()
                # GVHMR's finger pass can reuse the identical body's cached wrists.
                skeleton = (
                    paths.run_output_path(body_runs[name], video, "skeleton_motion.npz")
                    if hands and name == "gvhmr" and name in body_runs else None
                )
                run_dir = run_pipeline(
                    video, rig, extractor=name, device=device, detect_hands=hands,
                    output_dir=target, skeleton_npz=skeleton, no_preview=True, force=force,
                    source_metadata_path=metadata,
                    **({"person_track": person_track} if selection else {}),
                )
                entry.update(
                    status="completed", elapsed_s=time.perf_counter() - start,
                    run_dir=str(run_dir), **summarize_run(run_dir, video),
                )
                if not hands:
                    body_runs[name] = run_dir
        except Exception as exc:
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        publish()
        print(f"{label}: {entry['status']}")
    if no_preview and previous.get("preview"):
        selected = {item["model"]: Path(item["run_dir"]) for item in comparison["runs"]
                    if item["status"] == "completed" and item["detect_hands"]}
        cached = previous["preview"]
        if set(selected) == set(COMPARISON_MODELS) and cached.get("run_signatures") == _run_signatures(video, selected):
            comparison["preview"] = cached
        else:
            (directory / f"{video.stem}_compare.mp4").unlink(missing_ok=True)
        publish()
    if not no_preview:
        selected = {}
        for name in COMPARISON_MODELS:
            candidates = [
                item for item in comparison["runs"]
                if item["model"] == name and item["detect_hands"] and item["status"] == "completed"
            ]
            if candidates:
                selected[name] = Path(candidates[0]["run_dir"])
        if len(selected) == len(COMPARISON_MODELS):
            try:
                signatures = _run_signatures(video, selected)
                from .presentation import VIEW_VERSION

                preview = directory / f"{video.stem}_compare.mp4"
                cached = previous.get("preview", {})
                if (
                    not force and preview.is_file() and cached.get("status") == "completed"
                    and cached.get("run_signatures") == signatures
                    and cached.get("presentation_version") == VIEW_VERSION
                ):
                    comparison["preview"] = cached
                    publish()
                    print("four-panel preview: already completed")
                    return destination
                temporary = directory / f".{video.stem}_compare.pending.mp4"
                temporary.unlink(missing_ok=True)
                try:
                    rendered = render_model_comparison(video, rig, selected, temporary)
                    if rendered.is_file() and rendered != preview:
                        rendered.replace(preview)
                finally:
                    temporary.unlink(missing_ok=True)
                comparison["preview"] = {
                    "status": "completed", "file": str(preview),
                    "panels": ["Original Video", "GVHMR + HaMeR", "ExMR", "FlowHMR"],
                    "runs": {name: str(folder) for name, folder in selected.items()},
                    "run_signatures": signatures,
                    "presentation_version": VIEW_VERSION,
                }
            except Exception as exc:
                comparison["preview"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        else:
            comparison["preview"] = {
                "status": "unavailable",
                "reason": "Requires completed GVHMR + HaMeR, ExMR native fingers and FlowHMR native fingers.",
            }
        publish()
        print(f"four-panel preview: {comparison['preview']['status']}")
    print(f"comparison: {destination}")
    return destination
