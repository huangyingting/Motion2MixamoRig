"""Explicit, full-clip person tracks for group videos; no pose data or image cropping."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from . import paths
from .extractors.tracking import _video_meta, resample_boxes
from .source_metadata import file_digest, record_source_run, save_source_metadata

SCHEMA = "m2mr-person-track-v1"
SUPPORTED_EXTRACTORS = ("gvhmr", "exmr", "flowhmr")


def validate_boxes(boxes: np.ndarray, meta: tuple) -> np.ndarray:
    width, height, frames, fps = meta
    boxes = np.asarray(boxes, dtype=np.float32)
    if min(width, height, frames, fps) <= 0 or not np.isfinite(meta).all():
        raise ValueError("invalid person-track video dimensions or timing")
    if boxes.shape != (frames, 4) or not np.isfinite(boxes).all():
        raise ValueError("selected person must be detected on every frame; partial tracks are not padded")
    if (
        np.any(boxes[:, :2] < 0)
        or np.any(boxes[:, 2:] > [width, height])
        or np.any(boxes[:, 2:] <= boxes[:, :2])
    ):
        raise ValueError("person-track boxes must be nonempty and within the video")
    if frames > 1:
        intersection = np.maximum(
            0, np.minimum(boxes[:-1, 2:], boxes[1:, 2:]) - np.maximum(boxes[:-1, :2], boxes[1:, :2]),
        ).prod(axis=1)
        area = (boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
        overlap = intersection / (area[:-1] + area[1:] - intersection)
        if np.any(overlap < 0.05):
            raise ValueError("selected track has an abrupt jump; review possible identity switches or cuts")
    return boxes


def read_person_track(track: Path, source: Path) -> dict:
    track, source = Path(track).resolve(), Path(source).resolve()
    before = file_digest(track)
    with np.load(track, allow_pickle=False) as data:
        if str(data["schema"]) != SCHEMA:
            raise ValueError(f"person track must use schema {SCHEMA}")
        source_sha = str(data["source_sha256"])
        person_id = int(data["person_id"])
        meta = (float(data["width"]), float(data["height"]), int(data["frame_count"]), float(data["fps"]))
        boxes = validate_boxes(data["bbox"], meta)
        if "detected" in data.files:
            observed = np.asarray(data["detected"])
            if observed.shape != (meta[2],) or observed.dtype != np.bool_:
                raise ValueError("person-track detected mask must be boolean with one value per frame")
            restored, _ = complete_detection_boxes(
                {int(i): boxes[i] for i in np.flatnonzero(observed)},
                meta, int(data.get("max_gap_frames", 0)),
            )
            if not np.allclose(restored, boxes, atol=1e-4, rtol=0):
                raise ValueError("person-track gap boxes do not match recorded detection provenance")
    if before != file_digest(track):
        raise ValueError("person-track file changed while being read")
    if person_id < 1 or source_sha != file_digest(source):
        raise ValueError("person track does not match this source video")
    actual = _video_meta(source)
    if tuple(meta[:3]) != tuple(actual[:3]) or not np.isclose(meta[3], actual[3], rtol=1e-5):
        raise ValueError("person-track dimensions or timeline do not match this video")
    return {
        "bbox": boxes, "source_meta": meta, "person_id": person_id,
        "source_sha256": source_sha, "person_track_sha256": before, "path": track,
    }


def boxes_for_video(selection: dict, video: Path) -> np.ndarray:
    return resample_boxes(selection["bbox"], selection["source_meta"], _video_meta(video))


def complete_detection_boxes(detections: dict, meta: tuple, max_gap_frames: int = 0) -> tuple:
    frames = int(meta[2])
    observed = np.asarray([index in detections for index in range(frames)])
    if not observed[0] or not observed[-1]:
        raise ValueError("person must be detected on the first and last frames")
    indices = np.flatnonzero(observed)
    longest = int(np.max(np.diff(indices) - 1)) if len(indices) > 1 else 0
    if longest > max_gap_frames:
        raise ValueError(f"person has a {longest}-frame detection gap (limit {max_gap_frames})")
    original = np.stack([detections[int(index)] for index in indices])
    if longest:
        for left, right in zip(range(len(indices) - 1), range(1, len(indices))):
            if indices[right] - indices[left] > 1:
                intersection = np.maximum(
                    0, np.minimum(original[left, 2:], original[right, 2:])
                    - np.maximum(original[left, :2], original[right, :2]),
                ).prod()
                area = (original[[left, right], 2:] - original[[left, right], :2]).prod(axis=1)
                if intersection / (area.sum() - intersection) < 0.5:
                    raise ValueError("detection gap endpoints are inconsistent; refusing an identity jump")
    boxes = np.stack([
        np.interp(np.arange(frames), indices, original[:, channel]) for channel in range(4)
    ], axis=1)
    return validate_boxes(boxes, meta), observed


def track_people(
    source: Path, *, output_dir: Path | None = None, device: str = "cpu", max_gap_frames: int = 0,
) -> Path:
    import cv2
    from ultralytics import YOLO

    source = Path(source).resolve()
    if paths.is_image(source):
        raise ValueError("person tracking currently accepts videos only")
    if max_gap_frames < 0:
        raise ValueError("max_gap_frames must be nonnegative")
    source_sha = file_digest(source)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        cap.release()
        raise ValueError("cannot decode the input video")
    meta = _video_meta(source)
    width, height, frames, fps = meta
    if min(meta) <= 0 or not np.isfinite(meta).all():
        cap.release()
        raise ValueError("video must have readable dimensions, frame count and FPS")
    directory = Path(output_dir).resolve() if output_dir else (
        paths.source_output_dir(source) / "tracks" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    if directory.exists() and any(directory.iterdir()):
        cap.release()
        raise FileExistsError("person-tracking output directory must be empty")
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = paths.WEIGHTS / "yolo/yolov8x.pt"
    if not checkpoint.exists():
        checkpoint = paths.WEIGHTS / "yolo/yolov8n.pt"
    detector = YOLO(str(checkpoint))
    tracks, first_frame, first_boxes = {}, None, []
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[:2] != (int(height), int(width)):
                raise ValueError("decoded video dimensions changed during tracking")
            result = detector.track(
                frame, persist=True, tracker="bytetrack.yaml", classes=[0],
                conf=0.1, device=device, verbose=False,
            )[0]
            if result.boxes is not None and result.boxes.id is not None:
                ids = result.boxes.id.cpu().numpy().astype(int)
                boxes = result.boxes.xyxy.cpu().numpy().copy()
                boxes[:, :2] = np.maximum(boxes[:, :2], 0)
                boxes[:, 2:] = np.minimum(boxes[:, 2:], [width, height])
                for person_id, box in zip(ids, boxes):
                    tracks.setdefault(int(person_id), {})[index] = box
                    if index == 0:
                        first_boxes.append((int(person_id), box))
            if index == 0:
                first_frame = frame.copy()
            index += 1
            if index % 100 == 0:
                print(f"Tracking people: {index}/{frames}", flush=True)
    finally:
        cap.release()
    if index != frames or file_digest(source) != source_sha:
        raise ValueError("video was truncated or changed during tracking")
    summary = {
        "schema": SCHEMA, "video": str(source), "source_sha256": source_sha,
        "frame_count": frames, "fps": fps, "width": width, "height": height,
        "tracker": "YOLO + ByteTrack", "tracks": [],
        "max_gap_frames": max_gap_frames,
        "note": "One selected person per run. Only full-clip tracks are selectable. Review IDs visually; occlusion can still cause identity switches.",
    }
    for person_id, detections in sorted(tracks.items()):
        entry = {
            "person_id": person_id, "detected_frames": len(detections), "selectable": False,
            "first_frame": min(detections), "last_frame": max(detections),
        }
        try:
            boxes, observed = complete_detection_boxes(detections, meta, max_gap_frames)
            destination = directory / f"{source.stem}_person_{person_id}.npz"
            np.savez_compressed(
                destination, schema=SCHEMA, source_sha256=source_sha, person_id=person_id,
                bbox=boxes, width=width, height=height, frame_count=frames, fps=fps,
                detected=observed, max_gap_frames=max_gap_frames,
            )
            entry.update(selectable=True, file=destination.name, interpolated_box_frames=int((~observed).sum()))
        except ValueError as exc:
            entry["reason"] = str(exc)
        summary["tracks"].append(entry)
    for person_id, box in first_boxes:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(first_frame, (x1, y1), (x2, y2), (0, 220, 255), 3)
        cv2.putText(first_frame, str(person_id), (x1, max(30, y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 220, 255), 3)
    image = directory / f"{source.stem}_people.jpg"
    if first_frame is None or not cv2.imwrite(str(image), first_frame):
        raise RuntimeError("could not save person-selection preview")
    summary["preview"] = image.name
    destination = directory / f"{source.stem}_people.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    metadata = save_source_metadata(source, paths.source_output_dir(source))
    record_source_run(metadata, destination, "yolo_bytetrack", "completed", operation="person_tracking")
    return destination
