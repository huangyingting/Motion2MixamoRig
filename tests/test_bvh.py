import json
import tempfile
import unittest
import io
import sys
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from motion2mixamorig.mixamo.bvh import (
    bvh_joint_name, canonicalize_bvh, write_bvh, write_bvh_from_npz,
)
from motion2mixamorig.mixamo.kinematics import (
    MixamoBone,
    MixamoRig,
    forward_kinematics,
    mat4,
)


def read_bvh(path):
    nodes, stack = [], []
    lines = path.read_text().splitlines()
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] in ("ROOT", "JOINT"):
            nodes.append({"name": parts[1], "parent": stack[-1] if stack else None})
            pending = len(nodes) - 1
        elif parts[0] == "End":
            pending = None
        elif parts[0] == "{":
            stack.append(pending)
        elif parts[0] == "}":
            stack.pop()
        elif parts[0] == "OFFSET" and stack[-1] is not None:
            nodes[stack[-1]]["offset"] = np.array(parts[1:], dtype=float)
        elif parts[0] == "OFFSET":
            nodes[stack[-2]]["end_offset"] = np.array(parts[1:], dtype=float)
        elif parts[0] == "MOTION":
            break
    start = lines.index("MOTION")
    rows = np.loadtxt(lines[start + 3:], ndmin=2)
    return nodes, rows


def bvh_positions(nodes, row):
    world = []
    for i, node in enumerate(nodes):
        local_rot = Rotation.from_euler(
            "ZXY", row[3 + i * 3:6 + i * 3], degrees=True
        ).as_matrix()
        if node["parent"] is None:
            world.append((local_rot, row[:3]))
        else:
            parent_rot, parent_pos = world[node["parent"]]
            world.append((
                parent_rot @ local_rot,
                parent_pos + parent_rot @ node["offset"],
            ))
    return {node["name"]: transform[1] for node, transform in zip(nodes, world)}


class BvhTests(unittest.TestCase):
    def setUp(self):
        bones = {}
        specs = [
            ("Hips", None, [0, 100, 0], [15, 25, -10]),
            ("Arm", "Hips", [20, 30, 5], [0, 0, 120]),
            ("Hand", "Arm", [0, 25, 0], [45, -20, 10]),
            ("Leg", "Hips", [10, -40, 0], [180, 0, 0]),
            ("Foot", "Leg", [0, 40, 5], [-30, 0, 0]),
        ]
        for name, parent, translation, angles in specs:
            rotation = Rotation.from_euler("XYZ", angles, degrees=True).as_matrix()
            local = mat4(rotation, np.array(translation, dtype=float))
            world = local if parent is None else bones[parent].rest_world @ local
            bones[name] = MixamoBone(
                name, parent, [], rotation, local[:3, 3], world, local, np.zeros(3)
            )
            if parent is not None:
                bones[parent].children.append(name)
        self.rig = MixamoRig(bones, list(bones), 0.01, Path("test.fbx"))
        self.asset = SimpleNamespace(rig=self.rig)

    def assert_matches_fk(self, path, animations, translations):
        nodes, rows = read_bvh(path)
        self.assertEqual(len(rows), len(animations))
        for row, anim, translation in zip(rows, animations, translations):
            actual = bvh_positions(nodes, row)
            expected = {
                bvh_joint_name(name): matrix
                for name, matrix in forward_kinematics(self.rig, anim, translation).items()
            }
            for name, position in actual.items():
                np.testing.assert_allclose(
                    position, expected[name][:3, 3] * 0.01, atol=3e-6, rtol=0
                )

    def test_rest_pose_with_nonidentity_bind_rotations(self):
        translations = [self.rig.bones["Hips"].lcl_translation_cm]
        with tempfile.TemporaryDirectory() as tmp:
            path = write_bvh(Path(tmp) / "rest.bvh", self.asset, [{}], translations, 30)
            self.assert_matches_fk(path, [{}], translations)
            _, rows = read_bvh(path)
            np.testing.assert_allclose(rows[:, 3:], 0, atol=1e-6)

    def test_animated_pose_with_nonidentity_bind_rotations(self):
        animations = [
            {name: Rotation.from_euler("XYZ", angles, degrees=True).as_matrix()
             for name in self.rig.order}
            for angles in ([30, -45, 60], [-80, 20, 125])
        ]
        translations = [np.array([10, 110, -20]), np.array([-30, 90, 40])]
        with tempfile.TemporaryDirectory() as tmp:
            path = write_bvh(
                Path(tmp) / "animated.bvh", self.asset, animations, translations, 30
            )
            self.assert_matches_fk(path, animations, translations)

    def test_reexport_saved_wxyz_rotations(self):
        rotation = Rotation.from_euler("XYZ", [35, -20, 75], degrees=True)
        animations = [{name: rotation.as_matrix() for name in self.rig.order}]
        translations = [np.array([15, 95, -10])]
        with tempfile.TemporaryDirectory() as tmp:
            npz = Path(tmp) / "rotations.npz"
            np.savez(
                npz, bone_names=self.rig.order,
                quat_wxyz_rig=np.tile(
                    rotation.as_quat()[[3, 0, 1, 2]], (1, len(self.rig.order), 1)
                ),
                hips_translation_cm_rig=translations, fps=30, rig="test.fbx",
            )
            with patch(
                "motion2mixamorig.mixamo.bvh.load_character_asset",
                return_value=self.asset,
            ):
                path = write_bvh_from_npz(npz, Path(tmp) / "saved.bvh")
            self.assert_matches_fk(path, animations, translations)

    def test_vrm_joint_names_and_terminal_sites_preserve_animation(self):
        specs = [
            ("Hips", None, [0, 100, 0]), ("Spine", "Hips", [0, 10, 0]),
            ("Spine1", "Spine", [0, 10, 0]), ("Spine2", "Spine1", [0, 10, 0]),
            ("Neck", "Spine2", [0, 10, 0]), ("Head", "Neck", [0, 10, 0]),
            ("HeadTop_End", "Head", [0, 5, 0]),
        ]
        for side, sign in (("Left", 1), ("Right", -1)):
            specs.extend([
                (f"{side}Shoulder", "Spine2", [10 * sign, 0, 0]),
                (f"{side}Arm", f"{side}Shoulder", [10 * sign, 0, 0]),
                (f"{side}ForeArm", f"{side}Arm", [20 * sign, 0, 0]),
                (f"{side}Hand", f"{side}ForeArm", [20 * sign, 0, 0]),
                (f"{side}UpLeg", "Hips", [10 * sign, -5, 0]),
                (f"{side}Leg", f"{side}UpLeg", [0, -40, 0]),
                (f"{side}Foot", f"{side}Leg", [0, -40, 0]),
                (f"{side}ToeBase", f"{side}Foot", [0, 0, 10]),
                (f"{side}Toe_End", f"{side}ToeBase", [0, 0, 3]),
            ])
            for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
                for segment in range(1, 5):
                    parent = f"{side}Hand" if segment == 1 else f"{side}Hand{finger}{segment - 1}"
                    specs.append((f"{side}Hand{finger}{segment}", parent, [3 * sign, 0, 0]))
        bones = {}
        for suffix, parent_suffix, translation in specs:
            name = f"mixamorig:{suffix}"
            parent = f"mixamorig:{parent_suffix}" if parent_suffix else None
            local = mat4(np.eye(3), np.asarray(translation, dtype=float))
            world = local if parent is None else bones[parent].rest_world @ local
            bones[name] = MixamoBone(name, parent, [], np.eye(3), local[:3, 3], world, local, np.zeros(3))
            if parent:
                bones[parent].children.append(name)
        self.rig = MixamoRig(bones, list(bones), 0.01, Path("test.fbx"))
        self.asset = SimpleNamespace(rig=self.rig)
        animation = {
            "mixamorig:Hips": Rotation.from_euler("XYZ", [15, 25, 10], degrees=True).as_matrix(),
            "mixamorig:LeftForeArm": Rotation.from_euler("Z", 60, degrees=True).as_matrix(),
            "mixamorig:LeftHandPinky2": Rotation.from_euler("X", 20, degrees=True).as_matrix(),
        }
        translation = np.asarray([1, 110, 2])
        with tempfile.TemporaryDirectory() as tmp:
            path = write_bvh(Path(tmp) / "humanoid.bvh", self.asset, [animation], [translation], 30)
            self.assert_matches_fk(path, [animation], [translation])
            nodes, rows = read_bvh(path)
            self.assertEqual(len(nodes), 52)
            self.assertEqual(rows.shape, (1, 159))
            parents = {node["name"]: nodes[node["parent"]]["name"] if node["parent"] is not None else None for node in nodes}
            self.assertEqual(parents["hips"], None)
            self.assertEqual(parents["neck"], "upperChest")
            self.assertEqual(parents["leftUpperArm"], "leftShoulder")
            self.assertEqual(parents["leftThumbMetacarpal"], "leftHand")
            self.assertEqual(parents["leftThumbProximal"], "leftThumbMetacarpal")
            self.assertEqual(parents["leftLittleIntermediate"], "leftLittleProximal")
            self.assertEqual(parents["rightToes"], "rightFoot")
            self.assertNotIn("HeadTop_End", parents)
            rest = forward_kinematics(self.rig)
            by_name = {node["name"]: node for node in nodes}
            for suffix, parent, _ in specs:
                if suffix == "HeadTop_End" or suffix.endswith(("4", "Toe_End")):
                    expected = (rest[f"mixamorig:{suffix}"][:3, 3] - rest[f"mixamorig:{parent}"][:3, 3]) * 0.01
                    np.testing.assert_allclose(by_name[bvh_joint_name(parent)]["end_offset"], expected)
            with patch("motion2mixamorig.mixamo.bvh._terminal", return_value=False), patch(
                "motion2mixamorig.mixamo.bvh.bvh_joint_name",
                side_effect=lambda name: name.split(":")[-1],
            ):
                legacy = write_bvh(Path(tmp) / "old.bvh", self.asset, [animation], [translation], 30)
            original = legacy.read_bytes()
            canonicalize_bvh(legacy, Path(tmp) / "converted.bvh")
            self.assertEqual(legacy.read_bytes(), original)
            converted_nodes, converted_rows = read_bvh(Path(tmp) / "converted.bvh")
            self.assertEqual([node["name"] for node in converted_nodes], [node["name"] for node in nodes])
            np.testing.assert_array_equal(converted_rows, rows)
            self.assert_matches_fk(Path(tmp) / "converted.bvh", [animation], [translation])

    def test_reused_delivery_updates_bvh_without_changing_original(self):
        from motion2mixamorig.run_report import copy_run_deliveries

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "old"
            old.mkdir()
            source = root / "clip.mp4"
            outputs = {
                "skeleton_motion": "skeleton.npz", "mixamo_rotations": "rotations.npz",
                "mixamo_character_glb": "character.glb", "mixamo_motion_bvh": "clip.bvh",
            }
            (old / "skeleton.npz").touch()
            (old / "character.glb").touch()
            translation = self.rig.bones["Hips"].lcl_translation_cm
            np.savez(
                old / "rotations.npz", bone_names=self.rig.order,
                quat_wxyz_rig=np.tile([1, 0, 0, 0], (1, len(self.rig.order), 1)),
                hips_translation_cm_rig=[translation], fps=30, rig="test.fbx",
            )
            path = write_bvh(old / "clip.bvh", self.asset, [{}], [translation], 30)
            legacy = path.read_text().replace("ROOT hips", "ROOT Hips")
            path.write_text(legacy)
            (old / "run.json").write_text(json.dumps({
                "video": str(source), "rig": "test.fbx", "outputs": outputs,
            }))
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({"file": {"sha256": "abc"}}))
            with patch("motion2mixamorig.source_metadata.record_source_run"):
                report = copy_run_deliveries(old, root / "new", source, metadata)
            copied = root / "new" / report["outputs"]["mixamo_motion_bvh"]
            self.assertEqual(path.read_text(), legacy)
            self.assertIn("ROOT hips", copied.read_text())
            self.assert_matches_fk(copied, [{}], [translation])
            broken = root / "broken.bvh"
            truncated = "\n".join(legacy.splitlines()[:-1]) + "\n"
            broken.write_text(truncated)
            with self.assertRaises(ValueError):
                canonicalize_bvh(broken, broken)
            self.assertEqual(broken.read_text(), truncated)

    def test_duplicate_humanoid_names_are_rejected_before_writing(self):
        for name in ("LeftArm", "leftUpperArm"):
            self.rig.bones[name] = replace(self.rig.bones["Arm"], name=name, children=[])
            self.rig.order.append(name)
            self.rig.bones["Hips"].children.append(name)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ambiguous.bvh"
            with self.assertRaisesRegex(ValueError, "duplicate BVH humanoid name: leftUpperArm"):
                write_bvh(path, self.asset, [{}], [np.array([0, 100, 0])], 30)
            self.assertFalse(path.exists())


class YoutubeOutputTests(unittest.TestCase):
    def test_only_source_named_bvh_is_written(self):
        from motion2mixamorig.youtube import run_youtube_to_bvh

        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "clip.mp4"
            video.touch()
            run_dir = Path(tmp) / "output"
            calls = []

            def fake_pipeline(*args, output_dir, **kwargs):
                calls.append(args)
                output_dir.mkdir(exist_ok=True)
                (output_dir / f"{args[0].stem}.bvh").write_text("motion")
                return output_dir

            output = io.StringIO()
            with patch.dict(
                sys.modules,
                {"motion2mixamorig.pipeline": SimpleNamespace(run_pipeline=fake_pipeline)},
            ), redirect_stdout(output):
                result = run_youtube_to_bvh(
                    str(video), run_dir, device="cpu", no_preview=True
                )
            self.assertEqual(result, run_dir)
            self.assertEqual(
                [p.name for p in run_dir.glob("*.bvh")], ["clip.bvh"]
            )
            self.assertIn(f"BVH file:  {run_dir / 'clip.bvh'}", output.getvalue())
            self.assertNotIn("BVH (std)", output.getvalue())
            self.assertEqual(calls[0][1].name, "X_Bot.fbx")

    def test_explicit_rig_overrides_x_bot_default(self):
        from motion2mixamorig.youtube import run_youtube_to_bvh

        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "clip.mp4"
            video.touch()
            rig = Path(tmp) / "Y_Bot.fbx"
            runner = unittest.mock.Mock(return_value=Path(tmp))
            with patch.dict(
                sys.modules,
                {"motion2mixamorig.pipeline": SimpleNamespace(run_pipeline=runner)},
            ), redirect_stdout(io.StringIO()):
                run_youtube_to_bvh(str(video), tmp, rig=rig, device="cpu", no_preview=True)
            self.assertEqual(runner.call_args.args[1], rig.resolve())


if __name__ == "__main__":
    unittest.main()
