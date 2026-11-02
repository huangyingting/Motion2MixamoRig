import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from motion2mixamorig import checks, person_tracking
from motion2mixamorig.extractors import _external_worker as worker
from motion2mixamorig.extractors.external import ExternalExtractor
from motion2mixamorig.extractors.gvhmr import GvhmrExtractor
from motion2mixamorig.source_metadata import file_digest


class PersonTrackingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.video = self.root / "group.mp4"
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (32, 48))
        self.assertTrue(writer.isOpened())
        for _ in range(4):
            writer.write(np.zeros((48, 32, 3), dtype=np.uint8))
        writer.release()
        self.track = self.root / "person_1.npz"
        self.boxes = np.tile([2, 2, 30, 46], (4, 1)).astype(np.float32)
        self.save_track()

    def save_track(self, **changes):
        data = dict(
            schema=person_tracking.SCHEMA, source_sha256=file_digest(self.video),
            person_id=1, bbox=self.boxes, width=32, height=48, frame_count=4, fps=30,
        )
        data.update(changes)
        np.savez(self.track, **data)

    def test_explicit_selection_allows_group_without_disabling_default_guard(self):
        with patch.object(checks, "_count_people", return_value=2) as count:
            self.assertIn("shows about 2 people", checks.check_video(self.video))
            count.reset_mock()
            self.assertIsNone(checks.check_video(self.video, person_track=self.track))
            count.assert_not_called()

    def test_wrong_video_partial_track_and_identity_jump_are_rejected(self):
        for changes, expected in (
            ({"source_sha256": "wrong"}, "does not match"),
            ({"bbox": self.boxes[:3]}, "every frame"),
            ({"bbox": np.array([[0, 0, 8, 8], [24, 40, 32, 48]] * 2)}, "abrupt jump"),
        ):
            with self.subTest(changes=changes):
                self.save_track(**changes)
                self.assertIn(expected, checks.check_video(self.video, person_track=self.track))

    def test_gvhmr_receives_the_explicit_track(self):
        with patch("motion2mixamorig.extractors.gvhmr._extract_video") as extract:
            GvhmrExtractor().extract(
                self.video, self.root / "motion.npz", work_dir=self.root / "cache",
                person_track=self.track,
            )
        self.assertEqual(extract.call_args.kwargs["person_track"], self.track)

    def test_external_worker_receives_track_and_has_isolated_cache(self):
        extractor = ExternalExtractor("flowhmr")
        with patch.object(extractor, "check", return_value=[]), patch(
            "motion2mixamorig.extractors.external.subprocess.run"
        ) as run, patch("motion2mixamorig.extractors.native.convert_native_motion"):
            extractor.extract(
                self.video, self.root / "motion.npz", work_dir=self.root / "cache",
                person_track=self.track, device="cuda",
            )
        command = run.call_args.args[0]
        self.assertEqual(command[-1], str(self.track))
        self.assertIn(f"person_{file_digest(self.track)}", command[4])
        with self.assertRaisesRegex(ValueError, "does not yet support"):
            ExternalExtractor("prompthmr").extract(
                self.video, self.root / "out.npz", work_dir=self.root,
                person_track=self.track, device="cuda",
            )

    def test_flow_selected_boxes_override_automatic_gvhmr_lookup(self):
        from motion2mixamorig.extractors import tracking

        processor = Mock()
        processor.transcode.return_value = str(self.video)
        with patch.object(tracking.paths, "OUTPUTS", self.root / "missing-registry"):
            result = tracking.prepare_full_frame_tracking(
                self.video, self.root / "flow", processor, person_track=self.track,
            )
        self.assertEqual(result["person_id"], 1)
        self.assertEqual(result["tracking_source"], "selected_person_track")
        with np.load(self.root / "flow/group/group_bbox.npz") as boxes:
            np.testing.assert_array_equal(boxes["bbox"], self.boxes)
            np.testing.assert_array_equal(boxes["start_end"], [0, 3])

    def test_track_export_does_not_publish_fragmented_people(self):
        from motion2mixamorig import person_tracking as tracking

        results = []
        for frame in range(4):
            ids = [1, 2] if frame < 3 else [1]
            boxes = np.asarray([[2, 2, 30, 46], [3, 3, 25, 40]][:len(ids)], dtype=float)
            tensor = lambda value: types.SimpleNamespace(cpu=lambda: types.SimpleNamespace(numpy=lambda: value))
            results.append(types.SimpleNamespace(boxes=types.SimpleNamespace(
                id=tensor(np.asarray(ids)), xyxy=tensor(boxes),
            )))
        detector = Mock()
        detector.track.side_effect = [[result] for result in results]
        with patch("ultralytics.YOLO", return_value=detector), patch.object(
            tracking, "save_source_metadata", return_value=self.root / "metadata.json",
        ), patch.object(tracking, "record_source_run"), patch.object(
            tracking.paths, "source_output_dir", return_value=self.root / "source",
        ):
            summary = tracking.track_people(self.video, output_dir=self.root / "tracks")
        data = json.loads(summary.read_text())
        self.assertTrue(data["tracks"][0]["selectable"])
        self.assertFalse(data["tracks"][1]["selectable"])
        self.assertFalse((summary.parent / "group_person_2.npz").exists())

    def test_exmr_preprocessing_uses_selected_boxes_not_its_main_person(self):
        import torch

        class Tracker:
            def get_main_track(self, frames):
                raise AssertionError("default tracker must not select another person")

        def preprocess(video, frames, count, width, height, options, assets, cache):
            np.testing.assert_array_equal(Tracker().get_main_track(frames).numpy(), self.boxes)

        upstream = types.SimpleNamespace(_preprocess=preprocess, V=types.SimpleNamespace(video_fps=lambda video: 30))

        def run(video, work, options, assets):
            upstream._preprocess(video, self.root, 4, 32, 48, options, assets, work)
            return {
                "smplx_global": {"transl": np.zeros((4, 3))},
                "static_conf_logits": torch.zeros((4, 6)), "fps": 30,
            }

        modules = {
            "exmr": types.SimpleNamespace(pipeline=upstream),
            "exmr.pipeline": types.SimpleNamespace(Options=Mock(), run_video=run),
            "exmr.assets": types.SimpleNamespace(Assets=Mock()),
            "exmr.preprocessing.tracker": types.SimpleNamespace(Tracker=Tracker),
        }
        with patch.dict(sys.modules, modules):
            result = worker.recover_exmr(self.video, self.root, "cuda", person_track=self.track)
        self.assertEqual(str(result["person_track_sha256"]), file_digest(self.track))
        self.assertEqual(int(result["person_id"]), 1)

    def test_gvhmr_seeds_selected_boxes_and_records_selection(self):
        import torch
        import gvhmr
        import gvhmr.cli
        from motion2mixamorig.extractors import gvhmr as adapter

        cache = self.root / "gvhmr-cache"
        box_file = cache / "group/preprocess/bbx.pt"

        def preprocess(cfg, flip_test=False):
            np.testing.assert_array_equal(torch.load(box_file, weights_only=True)["bbx_xyxy"].numpy(), self.boxes)

        demo = types.SimpleNamespace(
            run_preprocess=preprocess, get_bbx_xys_from_xyxy=lambda boxes, **kwargs: boxes[:, :3],
        )

        def recover(*args, **kwargs):
            demo.run_preprocess(types.SimpleNamespace(
                output_dir=cache / "group", video_path=self.video,
                paths=types.SimpleNamespace(bbx=box_file),
            ))
            return types.SimpleNamespace(
                smpl_params_world={"transl": np.zeros((4, 3)), "global_orient": np.zeros((4, 3))},
                fps=30, camera="static", raw=None,
            )

        with patch.object(gvhmr, "recover", recover), patch.dict(sys.modules, {"gvhmr.cli.demo": demo}), patch.object(
            adapter, "joints_world_from_smplx", return_value=np.zeros((4, 24, 3)),
        ):
            output = adapter.GvhmrExtractor().extract(
                self.video, self.root / "motion.npz", work_dir=cache,
                person_track=self.track, verbose=False,
            )
        with np.load(output) as motion:
            self.assertEqual(str(motion["person_track_sha256"]), file_digest(self.track))
        self.save_track(person_id=2)
        with self.assertRaisesRegex(ValueError, "another person selection"):
            adapter._extract_video(
                self.video, self.root / "out.npz", work_dir=cache, person_track=self.track,
            )

    def test_reused_skeleton_and_comparison_cannot_mix_dancers(self):
        from motion2mixamorig.pipeline import run_pipeline
        from motion2mixamorig.comparison import COMPARISON_MODELS, render_model_comparison

        skeleton = self.root / "wrong.npz"
        np.savez(skeleton, person_track_sha256="another-track")
        with self.assertRaisesRegex(ValueError, "reused skeleton"):
            run_pipeline(
                self.video, self.root / "rig.fbx", skeleton_npz=skeleton, person_track=self.track,
            )
        np.savez(skeleton, person_track_sha256=file_digest(self.track))
        with self.assertRaisesRegex(ValueError, "--person-track"):
            run_pipeline(self.video, self.root / "rig.fbx", skeleton_npz=skeleton)
        with patch(
            "motion2mixamorig.comparison.load_comparison_run",
            side_effect=[types.SimpleNamespace(person_track_sha256=value) for value in ("a", "b", "a")],
        ), self.assertRaisesRegex(ValueError, "same person selection"):
            render_model_comparison(
                self.video, self.root / "rig.fbx",
                {name: self.root / name for name in COMPARISON_MODELS}, self.root / "compare.mp4",
            )

    def test_detection_gaps_are_opt_in_bounded_and_never_extend_endpoints(self):
        detections = {0: self.boxes[0], 3: self.boxes[3]}
        with self.assertRaisesRegex(ValueError, "detection gap"):
            person_tracking.complete_detection_boxes(detections, (32, 48, 4, 30))
        boxes, observed = person_tracking.complete_detection_boxes(detections, (32, 48, 4, 30), 2)
        np.testing.assert_array_equal(boxes, self.boxes)
        np.testing.assert_array_equal(observed, [True, False, False, True])
        with self.assertRaisesRegex(ValueError, "first and last"):
            person_tracking.complete_detection_boxes({0: self.boxes[0]}, (32, 48, 4, 30), 9)
        with self.assertRaisesRegex(ValueError, "identity jump"):
            person_tracking.complete_detection_boxes(
                {0: np.array([0, 0, 8, 8]), 3: np.array([24, 40, 32, 48])},
                (32, 48, 4, 30), 2,
            )
        self.save_track(detected=np.array([True, False, False, True]), max_gap_frames=1)
        self.assertIn("detection gap", checks.check_video(self.video, person_track=self.track))


if __name__ == "__main__":
    unittest.main()
