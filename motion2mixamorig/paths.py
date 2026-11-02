"""Single source of truth for project paths: assets, weights, outputs."""

from __future__ import annotations

import os
import json
import re
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ASSETS = PROJECT_ROOT / "assets"
BODY_MODELS = ASSETS / "body_models"
SMPLX_NEUTRAL = BODY_MODELS / "smplx" / "SMPLX_NEUTRAL.npz"
MANO_DIR = BODY_MODELS / "mano"
MANO_RIGHT_PKL = MANO_DIR / "MANO_RIGHT.pkl"
MANO_RIGHT_NPZ = MANO_DIR / "MANO_RIGHT.npz"
MIXAMO_DIR = ASSETS / "mixamo"
DEFAULT_RIG = MIXAMO_DIR / "X_Bot.fbx"
VIDEO_DIR = ASSETS / "video"
IMAGE_DIR = ASSETS / "image"

WEIGHTS = PROJECT_ROOT / "weights"
MODEL_RUNTIMES = WEIGHTS / "runtimes"
HAMER_WEIGHTS = WEIGHTS / "hamer"
HAMER_CKPT = HAMER_WEIGHTS / "hamer.ckpt"
HAMER_MEAN_PARAMS = HAMER_WEIGHTS / "mano_mean_params.npz"
OUTPUTS = PROJECT_ROOT / "outputs"

VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".webm")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

_OUTPUT_SUFFIXES = {
    "run.json": "run.json",
    "skeleton_motion.npz": "skeleton.npz",
    "mixamo_rotations.npz": "rotations.npz",
    "mixamo_character.glb": "character.glb",
    "diagnostics.json": "diagnostics.json",
    "diagnostics.html": "diagnostics.html",
    "compare.mp4": "compare.mp4",
    "person_track.npz": "person_track.npz",
}


def run_output_path(run_dir: Path, source: Path, legacy_name: str) -> Path:
    """Name an output after its input file, retaining the artifact's role."""
    suffix = _OUTPUT_SUFFIXES[legacy_name]
    return Path(run_dir) / f"{Path(source).stem}_{suffix}"


def find_run_file(run_dir: Path, legacy_name: str) -> Path:
    """Find a prefixed artifact, falling back to the fixed-name legacy layout."""
    directory = Path(run_dir)
    suffix = _OUTPUT_SUFFIXES[legacy_name]
    matches = sorted(p for p in directory.glob(f"*_{suffix}") if p.name != legacy_name)
    if len(matches) > 1:
        raise ValueError(f"Multiple {suffix} files in {directory}; pass one run folder")
    return matches[0] if matches else directory / legacy_name


def _list_media(folder: Path, suffixes: tuple[str, ...]) -> list[Path]:
    if not folder.is_dir():
        return []
    files = [
        p
        for p in folder.iterdir()
        if p.suffix.lower() in suffixes and not p.name.startswith(".")
    ]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def is_image(path: Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def list_videos() -> list[Path]:
    """User-supplied videos in assets/video/, most recently added first."""
    return _list_media(VIDEO_DIR, VIDEO_SUFFIXES)


def latest_video() -> Path | None:
    """The video most recently placed into assets/video/ (by file mtime)."""
    videos = list_videos()
    return videos[0] if videos else None


def list_images() -> list[Path]:
    """User-supplied stills in assets/image/, most recently added first."""
    return _list_media(IMAGE_DIR, IMAGE_SUFFIXES)


def latest_image() -> Path | None:
    """The still most recently placed into assets/image/ (by file mtime)."""
    images = list_images()
    return images[0] if images else None


def source_output_dir(source: Path) -> Path:
    """Group by source stem; same-name different contents get a hash suffix."""
    from .source_metadata import source_identity

    source = Path(source)
    registry = OUTPUTS / "organization.json"
    if registry.is_file():
        inventory = json.loads(registry.read_text())
        if inventory.get("schema") == "m2mr-organization-v2":
            registered = inventory.get("sources", {}).get(str(source.resolve()))
            if registered:
                known = Path(registered).resolve()
                metadata = known / f"{source.stem}_source.json"
                if known.is_relative_to(OUTPUTS.resolve()) and metadata.is_file():
                    if json.loads(metadata.read_text())["file"]["sha256"] == source_identity(source)["sha256"]:
                        return known
    stem = re.sub(r'[<>:"/\\|?*]', "_", source.stem).strip(". ") or "source"
    directory = OUTPUTS / stem
    metadata = directory / f"{source.stem}_source.json"
    if metadata.exists():
        identity = source_identity(source)
        if json.loads(metadata.read_text())["file"]["sha256"] != identity["sha256"]:
            directory = OUTPUTS / f"{stem}__{identity['sha256'][:12]}"
    return directory


def new_batch_dir(source: Path, now: datetime | None = None) -> Path:
    """Allocate an exclusive batch below the source directory."""
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    parent = source_output_dir(source)
    parent.mkdir(parents=True, exist_ok=True)
    for index in range(10000):
        directory = parent / (stamp if index == 0 else f"{stamp}_{index:02d}")
        try:
            directory.mkdir()
            return directory
        except FileExistsError:
            continue
    raise FileExistsError(f"cannot allocate batch under {parent}")


def model_variant(name: str, hands: str | None = None) -> str:
    return f"{name}_{hands}" if hands and hands != name else name


def new_run_dir(source: Path, now: datetime | None = None, *, model: str = "gvhmr") -> Path:
    """Use one stable outputs/<source>/<extractor>/ directory per body model."""
    from .source_metadata import save_source_metadata

    if not re.fullmatch(r"[a-z][a-z0-9_]*", model):
        raise ValueError(f"invalid model directory name: {model}")
    parent = source_output_dir(source)
    save_source_metadata(source, parent)
    run_dir = parent / model.split("_", 1)[0]
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def export_gvhmr_env() -> None:
    """Point GVHMR at this project's weights/ and assets/ before importing it.

    The gvhmr package resolves its asset roots from these variables at import
    time. Auto-downloaded checkpoints land in weights/; the gated SMPL-X body
    model is read from assets/body_models/ where the user placed it.
    """
    os.environ.setdefault("GVHMR_CHECKPOINTS", str(WEIGHTS))
    os.environ.setdefault("GVHMR_BODY_MODELS", str(BODY_MODELS))
