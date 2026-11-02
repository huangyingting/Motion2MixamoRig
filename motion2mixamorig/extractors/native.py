"""Normalize flat-hand SMPL-X / SMPL-H parameters into the shared SMPL-24 skeleton."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import paths
from ..extract import playback_fps, video_duration_seconds
from ..models import get_model_spec
from .base import write_skeleton_npz


def validate_params(params: dict) -> int:
    n = len(params["transl"])
    if not n:
        raise ValueError("the model returned no motion frames")
    for key, width in (("transl", 3), ("global_orient", 3), ("body_pose", 63)):
        values = np.asarray(params[key])
        if values.shape != (n, width) or not np.isfinite(values).all():
            raise ValueError(f"{key} must be finite with shape ({n}, {width})")
    betas = np.asarray(params["betas"])
    if betas.ndim != 2 or len(betas) not in (1, n) or not np.isfinite(betas).all():
        raise ValueError("betas must be finite with shape (1 or frames, coefficients)")
    if not 1 <= betas.shape[1] <= 16:
        raise ValueError("expected 1 to 16 body shape coefficients")
    fps = float(params["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("model frame rate must be finite and positive")
    for key in ("left_hand_pose", "right_hand_pose"):
        if key in params:
            values = np.asarray(params[key])
            if values.shape != (n, 45) or not np.isfinite(values).all():
                raise ValueError(f"{key} must be finite with shape ({n}, 45)")
    if "foot_contact" in params:
        contact = np.asarray(params["foot_contact"])
        if contact.shape != (n, 4) or not np.isfinite(contact).all():
            raise ValueError("foot_contact must be finite with shape (frames, 4)")
        if np.any((contact < 0) | (contact > 1)):
            raise ValueError("foot_contact must contain probabilities in [0, 1]")
    return n


def smpl24_joints(params: dict, body_model: Path, model_type: str, device: str) -> np.ndarray:
    import smplx
    import torch

    paths.export_gvhmr_env()
    from gvhmr import PROJ_ROOT

    n = validate_params(params)
    if model_type not in ("smplx", "smplh"):
        raise ValueError(f"unsupported body model: {model_type}")
    device = "cpu" if device == "mps" else device
    batch_size = min(n, 64)
    if model_type == "smplx":
        model = smplx.SMPLX(
            str(body_model), batch_size=batch_size, num_betas=params["betas"].shape[1],
            use_pca=False, flat_hand_mean=True,
        ).to(device)
    else:
        from smplx.lbs import lbs

        # AMASS omits unused hand PCA data; SMPLH's wrapper also truncates its 16 betas.
        with np.load(body_model, allow_pickle=False) as data:
            geometry = {
                key: torch.as_tensor(data[key], dtype=torch.float32, device=device)
                for key in ("v_template", "shapedirs", "J_regressor", "weights")
            }
            geometry["shapedirs"] = geometry["shapedirs"][..., :params["betas"].shape[1]]
            geometry["posedirs"] = torch.as_tensor(
                data["posedirs"].reshape(-1, data["posedirs"].shape[-1]).T,
                dtype=torch.float32, device=device,
            )
            geometry["parents"] = torch.as_tensor(
                data["kintree_table"][0].astype(np.int64), device=device,
            )
            geometry["parents"][0] = -1
            geometry["lbs_weights"] = geometry.pop("weights")
    maps = PROJ_ROOT / "gvhmr/utils/body_model"
    regressor = torch.load(
        maps / "smpl_neutral_J_regressor.pt", map_location=device, weights_only=False,
    )
    vertex_map = (
        torch.load(maps / "smplx2smpl_sparse.pt", map_location=device, weights_only=False)
        if model_type == "smplx" else None
    )
    joints = []
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            inputs = {
                key: torch.as_tensor(params[key][start:end], dtype=torch.float32, device=device)
                for key in ("transl", "global_orient", "body_pose", "left_hand_pose", "right_hand_pose")
                if key in params
            }
            betas = params["betas"]
            inputs["betas"] = torch.as_tensor(
                np.repeat(betas, end - start, axis=0) if len(betas) == 1 else betas[start:end],
                dtype=torch.float32, device=device,
            )
            for key in ("left_hand_pose", "right_hand_pose"):
                inputs.setdefault(key, torch.zeros(end - start, 45, device=device))
            if model_type == "smplx":
                for key in ("jaw_pose", "leye_pose", "reye_pose"):
                    inputs[key] = torch.zeros(end - start, 3, device=device)
                inputs["expression"] = torch.zeros(end - start, 10, device=device)
            # SMPL-X landmarks assume the constructor's fixed batch size.
            if end - start < batch_size:
                inputs = {
                    key: torch.cat([value, value[-1:].expand(batch_size - len(value), -1)])
                    for key, value in inputs.items()
                }
            if model_type == "smplx":
                verts = model(**inputs).vertices
            else:
                pose = torch.cat([
                    inputs[key] for key in
                    ("global_orient", "body_pose", "left_hand_pose", "right_hand_pose")
                ], dim=1)
                verts, _ = lbs(inputs["betas"], pose, **geometry)
                verts = verts + inputs["transl"][:, None]
            verts = verts[:end - start]
            if vertex_map is not None:
                verts = torch.stack([torch.matmul(vertex_map, v) for v in verts])
            joints.append(torch.einsum("jv,fvc->fjc", regressor, verts).cpu().numpy())
    return np.concatenate(joints).astype(np.float32)


def native_hands(params: dict, extractor: str) -> dict:
    from scipy.spatial.transform import Rotation

    from .hamer import HAND_JOINT_NAMES, ensure_mano_npz

    poses = np.stack([params["left_hand_pose"], params["right_hand_pose"]], axis=1)
    rotations = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix()
    with np.load(ensure_mano_npz(), allow_pickle=False) as mano:
        template = np.asarray(mano["J"], dtype=np.float32)
    return {
        "hand_pose": rotations.reshape(len(poses), 2, 15, 3, 3).astype(np.float32),
        "hand_valid": np.ones((len(poses), 2), dtype=bool),
        "hand_template_joints": template,
        "hand_joint_names": np.asarray(HAND_JOINT_NAMES),
        "hands": np.asarray(extractor),
        "hand_pose_convention": np.asarray(
            "Parent-relative rotation matrices in flat SMPL-X/SMPL-H hand axes; "
            "index 0 = left, index 1 = right; MANO kintree finger order."
        ),
    }


def normalize_world_coordinates(params: dict, joints: np.ndarray, extractor: str):
    from scipy.spatial.transform import Rotation

    world_rotation = np.asarray(
        get_model_spec(extractor).get("world_to_shared", np.eye(3)), dtype=np.float32,
    )
    # SMPL rotates around its shaped rest pelvis J0, not the coordinate origin.
    rest_pelvis = joints[:, 0] - params["transl"]
    translation = (params["transl"] + rest_pelvis) @ world_rotation.T - rest_pelvis
    orientation = Rotation.from_matrix(
        world_rotation @ Rotation.from_rotvec(params["global_orient"]).as_matrix()
    ).as_rotvec()
    return joints @ world_rotation.T, translation, orientation, world_rotation


def convert_native_motion(
    params_path: Path, output_npz: Path, *, source: Path, extractor: str,
    body_model: Path, model_type: str, detect_hands: bool = False, device: str = "cpu",
) -> Path:
    with np.load(params_path, allow_pickle=False) as data:
        params = {key: data[key] for key in data.files}
    n = validate_params(params)
    duration = video_duration_seconds(source)
    native_fps = float(params["fps"])
    if duration is not None and abs(n / native_fps - duration) > 2 / native_fps:
        raise ValueError(
            f"{extractor} motion covers {n / native_fps:.3f}s, but source covers "
            f"{duration:.3f}s; refusing to stretch a partial track to the full video"
        )
    if detect_hands and not all(key in params for key in ("left_hand_pose", "right_hand_pose")):
        raise ValueError(f"{extractor} did not return finger rotations")
    joints = smpl24_joints(params, body_model, model_type, device)
    joints, translation, orientation, world_rotation = normalize_world_coordinates(params, joints, extractor)
    extras = {
        "video": np.asarray(str(Path(source).resolve())),
        "body_model": np.asarray(str(body_model)),
        "coordinate_convention": np.asarray("Y-up metres, shared Motion2MixamoRig world frame"),
        "world_to_shared": world_rotation,
        "foot_contact": params.get("foot_contact"),
    }
    for key in ("tracking_source", "tracking_run", "person_id", "person_track_sha256"):
        if key in params:
            extras[key] = params[key]
    if detect_hands:
        extras.update(native_hands(params, extractor))
    return write_skeleton_npz(
        output_npz, extractor=extractor, joints_3d=joints,
        fps=playback_fps(source, n, fallback=float(params["fps"])),
        root_translation=translation, root_orientation=orientation,
        **extras,
    )
