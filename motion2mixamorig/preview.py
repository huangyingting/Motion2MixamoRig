"""Render saved animation products without re-running extraction or retargeting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from . import paths
from .mixamo.animation import RetargetedAnimation
from .mixamo.kinematics import forward_kinematics, world_positions_m
from .run_report import write_run_json


def load_preview(directory):
    from .human_motion import load_human_motion
    from .mixamo.retarget import joint_mixamo
    from .mixamo.skinned_mesh import load_character_asset
    from .mixamo.tpose_calibration import load_calibration

    directory = Path(directory)
    report = json.loads(paths.find_run_file(directory, "run.json").read_text())
    source = Path(report.get("image") or report["video"])
    rig = Path(report["rig"])
    calibration = load_calibration(rig)
    asset = load_character_asset(rig, textures=True)
    motion = load_human_motion(directory / report["outputs"]["skeleton_motion"])
    with np.load(directory / report["outputs"]["mixamo_rotations"], allow_pickle=False) as data:
        names = data["bone_names"].tolist()

        def rotations(key):
            values = data[key]
            matrices = Rotation.from_quat(values.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix()
            return [dict(zip(names, frame)) for frame in matrices.reshape(len(values), len(names), 3, 3)]

        skeleton_rotations = rotations("quat_wxyz_skeleton")
        rig_rotations = rotations("quat_wxyz_rig")
        hips_skeleton = data["hips_translation_cm_skeleton"].copy()
        hips_rig = data["hips_translation_cm_rig"].copy()
        frames = data["frames"].tolist()
        fps = float(data["fps"])
    anim = RetargetedAnimation(
        frames=frames, fps=fps, r_anim_skeleton=skeleton_rotations, r_anim_rig=rig_rotations,
        hips_cm_skeleton=list(hips_skeleton), hips_cm_rig=list(hips_rig),
        mixamo_pos=[
            world_positions_m(calibration.skeleton_rig,
                              forward_kinematics(calibration.skeleton_rig, rotations, hips_translation_cm=hips))
            for rotations, hips in zip(skeleton_rotations, hips_skeleton)
        ],
        rig_pos=[
            world_positions_m(asset.rig, forward_kinematics(asset.rig, rotations, hips_translation_cm=hips))
            for rotations, hips in zip(rig_rotations, hips_rig)
        ],
        human_pos=[{name: joint_mixamo(motion, frame, name) for name in motion.joint_names} for frame in frames],
    )
    return anim, asset, calibration, source, report


def render_saved_preview(directory, *, verbose=True):
    from .videos import render_run_images, render_run_videos

    directory = Path(directory)
    anim, asset, calibration, source, report = load_preview(directory)
    render = render_run_images if report.get("image") else render_run_videos
    import tempfile

    with tempfile.TemporaryDirectory(prefix=".preview-", dir=directory) as staging:
        rendered = render(anim, asset, calibration, source, Path(staging), verbose=verbose)
        products = {}
        for key, path in rendered.items():
            destination = directory / path.name
            path.replace(destination)
            products[key] = destination
    report["outputs"]["images" if report.get("image") else "videos"] = {key: path.name for key, path in products.items()}
    report["presentation"] = anim.presentation_view.metadata()
    write_run_json(directory, report)
    return products
