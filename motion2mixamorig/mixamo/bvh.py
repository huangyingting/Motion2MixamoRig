"""Write an animated Mixamo skeleton to a Biovision Hierarchy (.bvh) motion file.

Uses the exact same bind pose, per-frame rest-relative rotations (`r_anim_rig`),
and root translations (`hips_cm_rig`) as `write_glb` in `gltf.py`, so the
exported `.bvh` matches `mixamo_character.glb` frame-for-frame and imports
cleanly into Three.js BVHLoader (`pose-studio`) and Blender.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .fbx_skeleton import bone_suffix
from .kinematics import forward_kinematics
from .skinned_mesh import YBotAsset, load_character_asset


HUMANOID_NAMES = {
    "Hips": "hips", "Spine": "spine", "Spine1": "chest", "Spine2": "upperChest",
    "Neck": "neck", "Head": "head",
}
for _side in ("Left", "Right"):
    for _source, _target in (
        ("Shoulder", "Shoulder"), ("Arm", "UpperArm"), ("ForeArm", "LowerArm"),
        ("Hand", "Hand"), ("UpLeg", "UpperLeg"), ("Leg", "LowerLeg"),
        ("Foot", "Foot"), ("ToeBase", "Toes"),
    ):
        HUMANOID_NAMES[f"{_side}{_source}"] = f"{_side.lower()}{_target}"
    for _finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        _segments = (
            ("Metacarpal", "Proximal", "Distal") if _finger == "Thumb"
            else ("Proximal", "Intermediate", "Distal")
        )
        _target_finger = "Little" if _finger == "Pinky" else _finger
        for _index, _segment in enumerate(_segments, 1):
            HUMANOID_NAMES[f"{_side}Hand{_finger}{_index}"] = (
                f"{_side.lower()}{_target_finger}{_segment}"
            )


def bvh_joint_name(name: str) -> str:
    suffix = bone_suffix(name)
    return HUMANOID_NAMES.get(suffix, suffix)


def _terminal(name: str) -> bool:
    suffix = bone_suffix(name)
    return suffix in {"HeadTop_End", "LeftToe_End", "RightToe_End"} or any(
        suffix == f"{side}Hand{finger}4"
        for side in ("Left", "Right") for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky")
    )


def has_mixamo_joint_names(path: Path) -> bool:
    """Detect old joint labels without reading the animation's numeric rows."""
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            words = line.split()
            if words and words[0] == "MOTION":
                break
            if len(words) == 2 and words[0] in ("ROOT", "JOINT"):
                if bvh_joint_name(words[1]) != words[1] or _terminal(words[1]):
                    return True
    return False


def canonicalize_bvh(source: Path, destination: Path) -> Path:
    """Rename an existing export and turn terminal markers into End Sites, losslessly."""
    lines = Path(source).read_text(encoding="utf-8").splitlines()
    motion = lines.index("MOTION")
    nodes, stack = [], []
    channel_count = 0
    for line in lines[1:motion]:
        words = line.split()
        if not words:
            continue
        if words[0] in ("ROOT", "JOINT", "End"):
            pending = {
                "name": words[1] if words[0] != "End" else None,
                "kind": words[0], "children": [],
            }
            if stack:
                stack[-1]["children"].append(pending)
            if pending["name"] is not None:
                nodes.append(pending)
        elif words[0] == "{":
            stack.append(pending)
        elif words[0] == "}":
            stack.pop()
        elif words[0] == "OFFSET":
            stack[-1]["offset"] = " ".join(words)
        elif words[0] == "CHANNELS":
            count = int(words[1])
            stack[-1].update(
                channels=" ".join(words), indices=list(range(channel_count, channel_count + count)),
                terminal_channels=words[2:] == ["Zrotation", "Xrotation", "Yrotation"],
            )
            channel_count += count
    output, indices = ["HIERARCHY"], []
    names = set()

    def emit(node: dict, depth: int) -> None:
        terminal = node["kind"] == "JOINT" and _terminal(node["name"]) and (
            not any(child["name"] is not None for child in node["children"])
        ) and node.get("terminal_channels", False)
        indent, inner = "\t" * depth, "\t" * (depth + 1)
        if node["name"] is None or terminal:
            output.extend([
                f"{indent}End Site", f"{indent}{{",
                f"{inner}{node['offset']}", f"{indent}}}",
            ])
            return
        name = bvh_joint_name(node["name"])
        if name in names:
            raise ValueError(f"duplicate BVH humanoid name: {name}")
        names.add(name)
        indices.extend(node["indices"])
        output.extend([
            f"{indent}{node['kind']} {name}", f"{indent}{{",
            f"{inner}{node['offset']}", f"{inner}{node['channels']}",
        ])
        for child in node["children"]:
            emit(child, depth + 1)
        output.append(f"{indent}}}")

    emit(nodes[0], 0)
    output.extend(lines[motion:motion + 3])
    rows = [line.split() for line in lines[motion + 3:] if line.strip()]
    if len(rows) != int(lines[motion + 1].split()[1]):
        raise ValueError("BVH frame count does not match its numeric rows")
    for row in rows:
        if len(row) != channel_count:
            raise ValueError("BVH numeric row does not match its channel count")
        output.append(" ".join(row[index] for index in indices))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".bvh.tmp")
    try:
        temporary.write_text("\n".join(output) + "\n", encoding="utf-8")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _rotmat_to_euler_zxy_deg(r: np.ndarray) -> tuple[float, float, float]:
    """Decompose 3x3 rotation matrix R = Rz(z) @ Rx(x) @ Ry(y) into degrees (z, x, y).

    For R = Rz(z) * Rx(x) * Ry(y):
        R = [
            [cz*cy - sz*sx*sy, -sz*cx, cz*sy + sz*sx*cy],
            [sz*cy + cz*sx*sy,  cz*cx, sz*sy - cz*sx*cy],
            [          -cx*sy,     sx,            cx*cy],
        ]
    """
    sx = float(np.clip(r[2, 1], -1.0, 1.0))
    x = float(np.arcsin(sx))
    cx = float(np.cos(x))
    if abs(cx) > 1e-6:
        z = float(np.arctan2(-r[0, 1], r[1, 1]))
        y = float(np.arctan2(-r[2, 0], r[2, 2]))
    else:
        z = float(np.arctan2(r[1, 0], r[0, 0]))
        y = 0.0
    rad2deg = 180.0 / np.pi
    return z * rad2deg, x * rad2deg, y * rad2deg


def write_bvh(
    out_path: Path,
    asset: YBotAsset,
    r_anim_rig: list[dict[str, np.ndarray]],
    hips_cm_rig: list[np.ndarray],
    fps: float,
    *,
    scale: float = 0.01,
) -> Path:
    """Write a `.bvh` file with world-aligned zero-rotation rest pose in meters (`scale=0.01`).

    To ensure universal compatibility with BVH importers (Three.js `BVHLoader` in
    `pose-studio`, Blender, MotionBuilder, Unity):
    - Mixamo joints use VRM humanoid names (hips, upperChest, leftUpperArm, etc.).
      Terminal markers are End Sites, giving standard rigs 22 body + 30 finger joints.
      Unrecognized helper names still strip prefixes (colons break PropertyBinding).
    - `ROOT hips` uses `OFFSET 0 0 0` so `BVHLoader` does not double-add root offset.
    - All joint axes at rest are aligned with the global coordinate axes (identity rest
      orientation). Remove each bone's world bind rotation first:
          `D_t[i] = W_t[i] @ W_bind[i]^T`
      Then the BVH joint rotation is `D_t[p]^T @ D_t[i]` (root: `D_t[root]`).
    """
    rig = asset.rig
    root_name = rig.root
    fk_rest = forward_kinematics(rig)
    rest_pos_m = {name: mat[:3, 3] * scale for name, mat in fk_rest.items()}

    joint_order: list[str] = []
    joint_names: set[str] = set()
    lines: list[str] = ["HIERARCHY"]

    def emit_node(name: str, depth: int, is_root: bool = False) -> None:
        joint_order.append(name)
        bone = rig.bones[name]
        bvh_name = bvh_joint_name(name)
        if bvh_name in joint_names:
            raise ValueError(f"duplicate BVH humanoid name: {bvh_name}")
        joint_names.add(bvh_name)
        indent = "\t" * depth
        inner = "\t" * (depth + 1)
        if is_root:
            lines.append(f"{indent}ROOT {bvh_name}")
            offset = np.zeros(3, dtype=np.float64)
        else:
            lines.append(f"{indent}JOINT {bvh_name}")
            assert bone.parent is not None
            offset = rest_pos_m[name] - rest_pos_m[bone.parent]
        lines.append(f"{indent}{{")
        lines.append(f"{inner}OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}")
        if is_root:
            lines.append(
                f"{inner}CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation"
            )
        else:
            lines.append(f"{inner}CHANNELS 3 Zrotation Xrotation Yrotation")

        original_children = [c for c in bone.children if c in rig.bones]
        children = [
            child for child in original_children
            if not (_terminal(child) and not rig.bones[child].children)
        ]
        if children:
            for child in children:
                emit_node(child, depth + 1, is_root=False)
        else:
            if original_children:
                end_offset = rest_pos_m[original_children[0]] - rest_pos_m[name]
            elif bone.parent is not None:
                parent_vec = rest_pos_m[name] - rest_pos_m[bone.parent]
                norm = float(np.linalg.norm(parent_vec))
                end_offset = parent_vec * 0.25 if norm > 1e-6 else np.array([0.0, 0.05, 0.0])
            else:
                end_offset = np.array([0.0, 0.05, 0.0])
            lines.append(f"{inner}End Site")
            lines.append(f"{inner}{{")
            lines.append(
                f"{inner}\tOFFSET {end_offset[0]:.6f} {end_offset[1]:.6f} {end_offset[2]:.6f}"
            )
            lines.append(f"{inner}}}")
        lines.append(f"{indent}}}")

    emit_node(root_name, 0, is_root=True)

    n_frames = len(r_anim_rig)
    frame_time = 1.0 / float(fps) if fps > 0 else 1.0 / 30.0
    lines.append("MOTION")
    lines.append(f"Frames: {n_frames}")
    lines.append(f"Frame Time: {frame_time:.6f}")

    for f_idx in range(n_frames):
        fk = forward_kinematics(rig, r_anim_rig[f_idx], hips_cm_rig[f_idx])
        world_deltas = {
            name: mat[:3, :3] @ fk_rest[name][:3, :3].T for name, mat in fk.items()
        }
        root_pos = fk[root_name][:3, 3] * scale
        vals: list[str] = [
            f"{root_pos[0]:.6f}",
            f"{root_pos[1]:.6f}",
            f"{root_pos[2]:.6f}",
        ]
        for name in joint_order:
            bone = rig.bones[name]
            w_curr = world_deltas[name]
            if bone.parent is None:
                r_bvh = w_curr
            else:
                w_parent = world_deltas[bone.parent]
                r_bvh = w_parent.T @ w_curr
            z_deg, x_deg, y_deg = _rotmat_to_euler_zxy_deg(r_bvh)
            vals.append(f"{z_deg:.6f}")
            vals.append(f"{x_deg:.6f}")
            vals.append(f"{y_deg:.6f}")
        lines.append(" ".join(vals))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def write_bvh_from_npz(
    rotations_npz: Path,
    out_bvh: Path,
    *,
    rig_path: Path | None = None,
    scale: float = 0.01,
) -> Path:
    """Export a `.bvh` file directly from a saved `mixamo_rotations.npz` artifact."""
    from scipy.spatial.transform import Rotation

    data = np.load(rotations_npz, allow_pickle=True)
    bone_names = [str(b) for b in data["bone_names"]]
    q_rig = np.asarray(data["quat_wxyz_rig"], dtype=np.float64)
    hips_cm_rig = [np.asarray(h, dtype=np.float64) for h in data["hips_translation_cm_rig"]]
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    resolved_rig = Path(rig_path) if rig_path is not None else Path(str(data["rig"]))

    asset = load_character_asset(resolved_rig, textures=False)
    n_frames = len(q_rig)
    r_anim_rig: list[dict[str, np.ndarray]] = []
    for i in range(n_frames):
        frame_rots: dict[str, np.ndarray] = {}
        for b_idx, b_name in enumerate(bone_names):
            frame_rots[b_name] = Rotation.from_quat(
                q_rig[i, b_idx][[1, 2, 3, 0]]
            ).as_matrix()
        r_anim_rig.append(frame_rots)

    return write_bvh(out_bvh, asset, r_anim_rig, hips_cm_rig, fps, scale=scale)
