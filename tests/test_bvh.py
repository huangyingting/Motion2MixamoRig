import tempfile
import unittest
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from motion2mixamorig.mixamo.bvh import write_bvh, write_bvh_from_npz
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
            expected = forward_kinematics(self.rig, anim, translation)
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
