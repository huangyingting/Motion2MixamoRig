"""Display-only heading alignment and fixed-scale, root-following previews."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from .mixamo.kinematics import forward_kinematics, normalize, world_positions_m
from .mixamo.render import Camera, Ground, build_camera


VIEW_VERSION = "m2mr-normalized-preview-v1"
HIPS = "mixamorig:Hips"
LEFT_HIP = "mixamorig:LeftUpLeg"
RIGHT_HIP = "mixamorig:RightUpLeg"


@dataclass
class PreviewView:
    camera: Camera
    rotation: np.ndarray
    offset: np.ndarray
    roots: np.ndarray
    up: np.ndarray
    yaw_deg: float

    def points(self, points):
        return np.asarray(points) @ self.rotation.T + self.offset

    def positions(self, positions):
        return {name: self.points(point) for name, point in positions.items()}

    def frame_camera(self, frame):
        root = self.points(self.roots[frame])
        horizontal = root - self.up * float(root @ self.up)
        return replace(self.camera, center=self.camera.center + horizontal)

    def metadata(self):
        return {
            "version": VIEW_VERSION,
            "camera": "orthographic",
            "scale": "fixed rig rest-pose scale",
            "heading": "initial pelvis aligned to rig front",
            "yaw_correction_deg": round(self.yaw_deg, 3),
            "root_motion": "camera follows horizontal root; floor retains world motion",
            "animation_modified": False,
        }

    def frame_ground(self, frame):
        root = self.points(self.roots[frame])
        return Ground(y=0, cx=float(np.floor(root[0] / 0.5) * 0.5),
                      cz=float(np.floor(root[2] / 0.5) * 0.5))


def normalized_view(rest_points, rest, poses, axes, width, height):
    up = np.array([0.0, 1.0, 0.0])
    target = normalize(np.asarray(axes["forward"]) - up * float(np.asarray(axes["forward"]) @ up))
    lateral = rest[RIGHT_HIP] - rest[LEFT_HIP]
    sign = 1 if np.cross(lateral, up) @ target >= 0 else -1
    forward = sign * np.cross(poses[0][RIGHT_HIP] - poses[0][LEFT_HIP], up)
    forward -= up * float(forward @ up)
    angle = 0.0
    if np.linalg.norm(forward) > 1e-8:
        forward = normalize(forward)
        angle = float(np.arctan2(up @ np.cross(forward, target), forward @ target))
    rotation = Rotation.from_rotvec(up * angle).as_matrix()
    roots = np.stack([pose[HIPS] for pose in poses])
    first = rotation @ roots[0]
    offset = -first + up * float(first @ up)
    points = np.asarray(rest_points).copy()
    rest_horizontal = rest[HIPS] - up * float(rest[HIPS] @ up)
    points -= rest_horizontal
    stature = float(np.ptp(points @ up))
    # Reserve fixed overhead space for raised arms, not each model's motion bounds.
    framing = np.concatenate([points, points + up * stature * 0.25, points - up * stature * 0.05])
    camera_axes = {**axes, "up": up, "forward": target}
    camera = build_camera(camera_axes, "front", framing, width, height, fill=0.9, pitch_deg=6)
    return PreviewView(camera, rotation, offset, roots, up, float(np.degrees(angle)))


def animation_view(anim, asset, calibration, width, height):
    from .mixamo.animation import skin_frames

    rest_world = forward_kinematics(asset.rig, {})
    rest = world_positions_m(asset.rig, rest_world)
    poses = [
        world_positions_m(asset.rig, forward_kinematics(asset.rig, rotations, hips_translation_cm=hips))
        for rotations, hips in zip(anim.r_anim_rig, anim.hips_cm_rig)
    ]
    hold = SimpleNamespace(r_anim_rig=[{}], hips_cm_rig=[None])
    points = np.concatenate(skin_frames(asset, hold, [0])[0])
    return normalized_view(points, rest, poses, calibration.rig_axes, width, height)
