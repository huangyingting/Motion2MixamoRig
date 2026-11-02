"""Original video and three model animations, sharing a camera and source timeline."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from . import paths
from .models import get_model_spec

COMPARISON_MODELS = ("gvhmr", "exmr", "flowhmr")
COMPARISON_LABELS = ("GVHMR + HaMeR", "ExMR", "FlowHMR")


def load_comparison_run(run_dir: Path, video: Path, rig: Path, model: str):
    report = json.loads(paths.run_output_path(run_dir, video, "run.json").read_text())
    if report.get("status") != "completed":
        raise ValueError(f"{model}: run is not completed")
    if Path(report["video"]).resolve() != video.resolve() or Path(report["rig"]).resolve() != rig.resolve():
        raise ValueError(f"{model}: run uses another source video or rig")
    if report.get("extractor") != model:
        raise ValueError(f"{model}: run was produced by {report.get('extractor')}")
    if report.get("source_sha256"):
        from .source_metadata import file_digest

        if report["source_sha256"] != file_digest(video):
            raise ValueError(f"{model}: source video contents have changed")
        if report.get("rig_sha256") and report["rig_sha256"] != file_digest(rig):
            raise ValueError(f"{model}: rig contents have changed")
    required_hands = "hamer" if model == "gvhmr" else model
    if report.get("hands") != required_hands:
        raise ValueError(f"{model}: comparison requires {required_hands} fingers, not {report.get('hands')}")
    expected = get_model_spec(model).get("world_to_shared")
    if expected is not None:
        with np.load(run_dir / report["outputs"]["skeleton_motion"], allow_pickle=False) as data:
            if "world_to_shared" not in data.files or not np.array_equal(data["world_to_shared"], expected):
                raise ValueError(f"{model}: old world axes; regenerate from cached parameters before comparison")
    with np.load(run_dir / report["outputs"]["mixamo_rotations"], allow_pickle=False) as data:
        quats = data["quat_wxyz_rig"]
        hips = data["hips_translation_cm_rig"]
        names = data["bone_names"].tolist()
        fps = float(data["fps"])
    if not len(quats) or quats.shape != (len(hips), len(names), 4):
        raise ValueError(f"{model}: invalid rotation dimensions")
    if not np.isfinite(quats).all() or not np.isfinite(hips).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"{model}: non-finite animation")
    matrices = Rotation.from_quat(quats.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix()
    matrices = matrices.reshape(len(quats), len(names), 3, 3)
    return SimpleNamespace(
        r_anim_rig=[dict(zip(names, frame)) for frame in matrices],
        hips_cm_rig=hips, n=len(quats), fps=fps,
        person_track_sha256=report.get("person_track_sha256"),
    )


def comparison_indices(source_frames: int, animation_frames: int) -> np.ndarray:
    if source_frames < 1 or animation_frames < 1:
        raise ValueError("comparison needs nonempty source and animation")
    return np.rint(np.linspace(0, animation_frames - 1, source_frames)).astype(np.int64)


def render_model_comparison(video: Path, rig: Path, runs: dict[str, Path], destination: Path) -> Path:
    import cv2

    from .mixamo.animation import skin_frames
    from .mixamo.render import (
        GRID_THEME, MESH_COLORS_BGR,
        draw_mesh, draw_shadow, ground_backdrop, label_panel,
    )
    from .mixamo.skinned_mesh import load_character_asset
    from .mixamo.tpose_calibration import load_calibration
    from .videos import (
        CHARACTER_AMBIENT,
        _OriginalReader, _StreamWriter, _grid2x2, panel_size,
    )
    from .presentation import animation_view
    from .extract import video_native_fps

    video, rig, destination = Path(video).resolve(), Path(rig).resolve(), Path(destination)
    if destination.exists():
        raise FileExistsError(f"comparison already exists: {destination}")
    if set(runs) != set(COMPARISON_MODELS):
        raise ValueError("comparison requires GVHMR + HaMeR, ExMR and FlowHMR runs")
    animations = [load_comparison_run(Path(runs[name]), video, rig, name) for name in COMPARISON_MODELS]
    if len({getattr(animation, "person_track_sha256", None) for animation in animations}) != 1:
        raise ValueError("comparison runs must use the same person selection")
    cap = cv2.VideoCapture(str(video))
    try:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    fps = video_native_fps(video)
    if not frames or not fps:
        raise ValueError("cannot determine source video timeline")
    duration = frames / fps
    for name, anim in zip(COMPARISON_MODELS, animations):
        if abs(anim.n / anim.fps - duration) > 3 / fps:
            raise ValueError(f"{name}: animation does not cover the full source clip")
    indices = [comparison_indices(frames, anim.n) for anim in animations]
    calibration = load_calibration(rig, verbose=False)
    asset = load_character_asset(rig)
    width, height = panel_size(video)
    views = [animation_view(anim, asset, calibration, width, height) for anim in animations]
    faces = [mesh.faces for mesh in asset.meshes]
    colors = [
        mesh.face_colors if mesh.face_colors is not None
        else MESH_COLORS_BGR.get(mesh.name, np.array([150.0, 150.0, 150.0]))
        for mesh in asset.meshes
    ]
    silent = destination.with_name(destination.stem + "_silent.mp4")
    reader = _OriginalReader(video, width, height)
    writer = _StreamWriter(silent, fps, width * 2, height * 2)
    last_indices = [None] * len(animations)
    last_panels = [None] * len(animations)
    try:
        for frame in range(frames):
            original = reader.frame(frame)
            label_panel(original, "Original Video")
            panels = []
            for slot, (label, anim, selection, view) in enumerate(zip(COMPARISON_LABELS, animations, indices, views)):
                index = int(selection[frame])
                if index != last_indices[slot]:
                    camera = view.frame_camera(index)
                    ground = view.frame_ground(index)
                    verts = [view.points(vertices) for vertices in skin_frames(asset, anim, [index])[0]]
                    panel = ground_backdrop(camera, ground, width, height, GRID_THEME)
                    draw_shadow(panel, camera, np.concatenate(verts), ground, GRID_THEME)
                    draw_mesh(panel, verts, faces, colors, camera, CHARACTER_AMBIENT)
                    label_panel(panel, label)
                    last_indices[slot], last_panels[slot] = index, panel
                panels.append(last_panels[slot])
            writer.write(_grid2x2(original, *panels))
    finally:
        reader.close()
        writer.close()
    # Do not use -shortest: a shorter audio stream must not drop the final video frame.
    subprocess.run([
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
        "-i", str(silent), "-i", str(video), "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "copy", "-c:a", "aac", "-t", f"{duration:.9f}", "-movflags", "+faststart",
        str(destination),
    ], check=True)
    silent.unlink()
    return destination
