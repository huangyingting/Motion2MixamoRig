"""Source identity and media timing, without credentials or arbitrary container tags."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def file_digest(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_identity(source: Path) -> dict:
    source = Path(source).resolve()
    return {
        "filename": source.name, "path": str(source),
        "sha256": file_digest(source), "size_bytes": source.stat().st_size,
    }


def safe_source_url(url: str) -> str:
    parts = urlsplit(url)
    # Signed CDN URLs, userinfo and arbitrary query parameters may contain credentials.
    from urllib.parse import parse_qs, urlencode

    query = parse_qs(parts.query)
    public_query = {"v": query["v"][0]} if parts.hostname in ("youtube.com", "www.youtube.com", "m.youtube.com") and "v" in query else {}
    return urlunsplit((parts.scheme, parts.hostname or "", parts.path, urlencode(public_query), ""))


def probe_media(source: Path) -> dict:
    binary = shutil.which("ffprobe")
    if binary is None:
        return {"status": "unavailable", "reason": "ffprobe is not installed"}
    try:
        result = subprocess.run([
            binary, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(source),
        ], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return {"status": "unavailable", "reason": "ffprobe did not complete"}
    if result.returncode:
        return {"status": "unavailable", "reason": "ffprobe could not read this source"}
    try:
        raw = json.loads(result.stdout)
    except ValueError:
        return {"status": "unavailable", "reason": "ffprobe returned invalid metadata"}
    fields = (
        "index", "codec_type", "codec_name", "width", "height", "pix_fmt",
        "r_frame_rate", "avg_frame_rate", "time_base", "start_time",
        "duration", "nb_frames", "sample_rate", "channels", "channel_layout",
    )
    streams = []
    for stream in raw.get("streams", []):
        if stream.get("codec_type") not in ("video", "audio"):
            continue
        item = {key: stream[key] for key in fields if key in stream}
        rotations = [side["rotation"] for side in stream.get("side_data_list", []) if "rotation" in side]
        if rotations:
            item["rotation_degrees"] = rotations[0]
        elif "rotate" in stream.get("tags", {}):
            item["rotation_degrees"] = stream["tags"]["rotate"]
        if item.get("codec_type") == "video":
            item["variable_frame_rate"] = "unknown"
            item["frame_rate_note"] = "Reported rates are not proof of constant frame timing."
        streams.append(item)
    container = {key: raw.get("format", {})[key] for key in (
        "format_name", "duration", "start_time", "bit_rate",
    ) if key in raw.get("format", {})}
    return {"status": "available", "container": container, "streams": streams}


def save_source_metadata(source: Path, directory: Path, *, provenance: dict | None = None) -> Path:
    source, directory = Path(source).resolve(), Path(directory)
    identity = source_identity(source)
    destination = directory / f"{source.stem}_source.json"
    if destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        if previous["file"]["sha256"] != identity["sha256"]:
            destination = directory / f"{source.stem}__{identity['sha256']}_source.json"
            if destination.exists():
                return destination.resolve()
        else:
            return destination.resolve()
    recorded = datetime.now(timezone.utc).isoformat()
    origin = {}
    sidecar = source.with_name(f"{source.stem}_origin.json")
    if provenance is None and sidecar.is_file():
        provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    for key in ("platform", "video_id", "title", "downloaded_at"):
        if provenance and isinstance(provenance.get(key), str):
            origin[key] = provenance[key]
    if provenance and provenance.get("url"):
        origin["url"] = safe_source_url(provenance["url"])
    metadata = {
        "schema": "m2mr-source-v1", "recorded_at": recorded,
        "file": identity, "origin": origin, "media": probe_media(source),
        "processing": [],
    }
    directory.mkdir(parents=True, exist_ok=True)
    # Publish a complete JSON file exclusively, so another job never reads a partial write.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    try:
        try:
            os.link(temporary, destination)
        except FileExistsError:
            previous = json.loads(destination.read_text(encoding="utf-8"))
            if previous["file"]["sha256"] != identity["sha256"]:
                raise
    finally:
        temporary.unlink()
    return destination.resolve()


def record_source_run(metadata: Path, report: Path, model: str, status: str, *, operation: str = "motion_generation") -> None:
    """Append known processing history, serializing concurrent Blender/CLI jobs."""
    metadata = Path(metadata)
    with (metadata.parent / ".source-metadata.lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt

            lock.seek(0)
            if not lock.read(1):
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            data = json.loads(metadata.read_text())
            history = data.setdefault("processing", [])
            entry = {
                "operation": operation, "report": str(Path(report).resolve()),
                "model": model, "status": status, "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
            history[:] = [item for item in history if item.get("report") != entry["report"]]
            history.append(entry)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=metadata.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
            try:
                os.replace(temporary, metadata)
            finally:
                temporary.unlink(missing_ok=True)
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
