"""Reuse a verified full-video single-person detection track, not estimated motion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import paths
from ..source_metadata import file_digest
from .hamer import find_body_cache


def resample_boxes(boxes: np.ndarray, source_meta: tuple, target_meta: tuple) -> np.ndarray:
    source_width, source_height, source_frames, source_fps = source_meta
    width, height, frames, fps = target_meta
    boxes = np.asarray(boxes, dtype=np.float32)
    if min(source_width, source_height, source_frames, source_fps, width, height, frames, fps) <= 0:
        raise ValueError("invalid tracking video dimensions or timing")
    if boxes.shape != (source_frames, 4) or not np.isfinite(boxes).all():
        raise ValueError("body tracking cache does not cover its staged video")
    if abs(source_frames / source_fps - frames / fps) > max(3 / fps, 3 / source_fps):
        raise ValueError("body tracking cache does not cover the full source duration")
    if not np.isclose(source_width / source_height, width / height, rtol=0.01):
        raise ValueError("tracking video has a different crop/aspect ratio")
    source_times = np.arange(source_frames) / source_fps
    target_times = np.arange(frames) / fps
    result = np.stack([
        np.interp(target_times, source_times, boxes[:, channel]) for channel in range(4)
    ], axis=1).astype(np.float32)
    result *= np.asarray([width / source_width, height / source_height] * 2, dtype=np.float32)
    if np.any(result[:, 2:] <= result[:, :2]):
        raise ValueError("tracking cache contains degenerate person boxes")
    return result


def _video_meta(video: Path):
    import cv2

    cap = cv2.VideoCapture(str(video))
    try:
        return cap.get(3), cap.get(4), int(cap.get(7)), cap.get(5)
    finally:
        cap.release()


def _prepare_flow_boxes(source: Path, work: Path, processor, boxes, source_meta, provenance: dict) -> dict:
    directory = work / source.stem
    directory.mkdir(parents=True, exist_ok=True)
    working_video = Path(processor.transcode(str(source), backup_dir=str(directory), force_fps=30))
    full_boxes = resample_boxes(np.asarray(boxes), source_meta, _video_meta(working_video))
    destination = directory / f"{source.stem}_bbox.npz"
    frame_range = np.asarray([0, len(full_boxes) - 1], dtype=np.int64)
    unchanged = False
    if destination.exists():
        with np.load(destination, allow_pickle=False) as previous:
            unchanged = np.array_equal(previous["start_end"], frame_range) and np.array_equal(previous["bbox"], full_boxes)
    if not unchanged:
        for suffix in ("static_camera.npz", "vggt_camera.npz", "sam3d_feat_static.pt", "sam3d_feat.pt"):
            (directory / f"{source.stem}_{suffix}").unlink(missing_ok=True)
        np.savez(destination, bbox=full_boxes, start_end=frame_range)
    provenance["frame_count"] = len(full_boxes)
    (directory / f"{source.stem}_bbox_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"FlowHMR: using verified full-source person boxes ({len(full_boxes)} frames)", flush=True)
    return provenance


def prepare_full_frame_tracking(source: Path, work: Path, processor, *, person_track: Path | None = None) -> dict | None:
    if person_track is not None:
        from ..person_tracking import read_person_track

        selection = read_person_track(person_track, source)
        return _prepare_flow_boxes(source, work, processor, selection["bbox"], selection["source_meta"], {
            "schema": "m2mr-tracking-v1", "source_sha256": selection["source_sha256"],
            "tracking_source": "selected_person_track", "tracking_run": str(person_track),
            "person_id": selection["person_id"], "person_track_sha256": selection["person_track_sha256"],
            "purpose": "Explicit person selection; body and fingers remain native FlowHMR estimates.",
        })
    registry = paths.OUTPUTS / "organization.json"
    if not source.is_file():
        return None
    reports = []
    if work.name == "flowhmr" and work.parent.name == "cache":
        reports.extend(work.parents[2].glob(f"*/{source.stem}_run.json"))
    if registry.exists():
        reports.extend(
            paths.find_run_file(Path(entry["run_dir"]), "run.json")
            for entry in json.loads(registry.read_text()).get("runs", {}).values()
            if entry.get("status") == "completed"
        )
    source_sha = file_digest(source)
    for report_path in reports:
        run = report_path.parent
        report = json.loads(report_path.read_text())
        if report.get("status") != "completed" or report.get("extractor") != "gvhmr" or report.get("source_sha256") != source_sha:
            continue
        if report.get("person_track_sha256"):
            continue
        candidates = [run, Path(report.get("organized_from") or run)]
        if report.get("original_cache"):
            candidates.append(Path(report["original_cache"]).parent)
        if report.get("skeleton_reused_from"):
            candidates.append(Path(report["skeleton_reused_from"]).parent)
        cache = next((value for directory in candidates if (value := find_body_cache(directory)) is not None), None)
        if cache is None:
            continue
        import torch

        staged = cache / "0_input_video.mp4"
        boxes = torch.load(cache / "preprocess" / "bbx.pt", map_location="cpu", weights_only=True)["bbx_xyxy"]
        provenance = {
            "schema": "m2mr-tracking-v1", "source_sha256": source_sha,
            "tracking_source": "gvhmr_cached_boxes", "tracking_run": str(run),
            "staged_video": str(staged),
            "purpose": "Full-clip detection boxes only; body and finger motion remain native FlowHMR estimates.",
        }
        return _prepare_flow_boxes(source, work, processor, boxes, _video_meta(staged), provenance)
    return None
