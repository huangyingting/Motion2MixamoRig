import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from motion2mixamorig import benchmark, paths
from motion2mixamorig import comparison, setup_models
from motion2mixamorig.extractors import extractor_names, get_extractor
from motion2mixamorig.extractors import _external_worker as worker
from motion2mixamorig.extractors import native
from motion2mixamorig.extractors.base import CheckResult
from motion2mixamorig.extractors.external import ExternalExtractor


def motion_params(n=4):
    return {
        "transl": np.zeros((n, 3), np.float32),
        "global_orient": np.zeros((n, 3), np.float32),
        "body_pose": np.zeros((n, 63), np.float32),
        "betas": np.zeros((1, 10), np.float32),
        "fps": np.asarray(30.0),
    }


class ExternalTests(unittest.TestCase):
    def test_missing_runtime_is_not_replaced_with_default_extractor(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"M2MR_EXMR_ROOT": tmp, "M2MR_EXMR_PYTHON": f"{tmp}/missing-python"}
        ):
            extractor = get_extractor("exmr")
            self.assertIsInstance(extractor, ExternalExtractor)
            self.assertTrue(any(not item.ok for item in extractor.check()))
            with self.assertRaises(FileNotFoundError), patch(
                "motion2mixamorig.extractors.external.subprocess.run"
            ) as run:
                extractor.extract(Path("clip.mp4"), Path("out.npz"), work_dir=Path(tmp), device="cuda")
            run.assert_not_called()

    def test_worker_failure_cannot_reuse_old_parameters(self):
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "exmr"
            work.mkdir()
            stale = work / "motion_params.npz"
            stale.write_bytes(b"stale")
            extractor = ExternalExtractor("exmr")
            with patch.object(extractor, "check", return_value=[]), patch(
                "motion2mixamorig.extractors.external.subprocess.run",
                side_effect=subprocess.CalledProcessError(1, ["worker"]),
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    extractor.extract(
                        Path("clip.mp4"), Path(tmp) / "out.npz", work_dir=Path(tmp), device="cuda"
                    )
            self.assertFalse(stale.exists())

    def test_unsupported_input_and_device_are_rejected(self):
        extractor = get_extractor("prompthmr")
        for source, device in (("pose.jpg", "cuda"), ("clip.mp4", "cpu")):
            with self.subTest(source=source), self.assertRaises(ValueError):
                extractor.extract(Path(source), Path("out.npz"), work_dir=Path("."), device=device)
        self.assertTrue(any(
            not item.ok and item.section == "hand" for item in extractor.check(detect_hands=True)
        ))


class WorkerTests(unittest.TestCase):
    def test_full_tracking_cache_verifies_source_and_invalidates_partial_features(self):
        import torch
        from motion2mixamorig.extractors import tracking
        from motion2mixamorig.source_metadata import file_digest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "clip.mp4"
            source.write_bytes(b"source")
            outputs = root / "outputs"
            run = outputs / "saved"
            run.mkdir(parents=True)
            report = {
                "status": "completed", "extractor": "gvhmr", "source_sha256": file_digest(source),
                "organized_from": str(run), "video": str(source),
            }
            (run / "run.json").write_text(json.dumps(report))
            (outputs / "organization.json").write_text(json.dumps({
                "runs": {"old": {"status": "completed", "run_dir": str(run)}},
            }))
            cache = run / "cache"
            (cache / "preprocess").mkdir(parents=True)
            torch.save({"bbx_xyxy": torch.tensor([[0, 10, 20, 30], [10, 20, 30, 40]])}, cache / "preprocess/bbx.pt")
            work = root / "work"
            destination = work / "clip"
            destination.mkdir(parents=True)
            np.savez(destination / "clip_bbox.npz", bbox=np.zeros((1, 4)), start_end=[0, 0])
            feature = destination / "clip_sam3d_feat_static.pt"
            feature.write_bytes(b"partial")
            processor = Mock()
            processor.transcode.return_value = str(root / "working.mp4")
            with patch.object(paths, "OUTPUTS", outputs), patch.object(
                tracking, "find_body_cache", return_value=cache,
            ), patch.object(tracking, "_video_meta", side_effect=[(100, 200, 2, 2), (200, 400, 4, 4)]):
                provenance = tracking.prepare_full_frame_tracking(source, work, processor)
            self.assertEqual(provenance["tracking_source"], "gvhmr_cached_boxes")
            self.assertFalse(feature.exists())
            with np.load(destination / "clip_bbox.npz") as data:
                np.testing.assert_array_equal(data["start_end"], [0, 3])
                self.assertEqual(data["bbox"].shape, (4, 4))
            source.write_bytes(b"changed")
            processor.reset_mock()
            with patch.object(paths, "OUTPUTS", outputs):
                self.assertIsNone(tracking.prepare_full_frame_tracking(source, work, processor))
            processor.transcode.assert_not_called()

    def test_full_video_tracking_boxes_resample_time_and_pixels_not_motion(self):
        from motion2mixamorig.extractors.tracking import resample_boxes

        boxes = np.asarray([[0, 10, 20, 30], [10, 20, 30, 40]], dtype=np.float32)
        actual = resample_boxes(boxes, (100, 200, 2, 2), (200, 400, 4, 4))
        np.testing.assert_allclose(actual, [[0, 20, 40, 60], [10, 30, 50, 70], [20, 40, 60, 80], [20, 40, 60, 80]])
        with self.assertRaisesRegex(ValueError, "full source duration"):
            resample_boxes(boxes, (100, 200, 2, 30), (100, 200, 240, 30))
        with self.assertRaisesRegex(ValueError, "crop/aspect"):
            resample_boxes(boxes, (100, 200, 2, 2), (200, 200, 2, 2))

    def test_worker_imports_upstream_hamer_not_local_extractor(self):
        import importlib

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider = root / "provider"
            provider.mkdir()
            (provider / "hamer.py").write_text("provider = True\n")

            def recover(*args):
                self.assertTrue(importlib.import_module("hamer").provider)
                return motion_params()

            with patch.object(sys, "path", [str(Path(worker.__file__).parent), str(provider)]), \
                    patch.object(sys, "argv", [
                        "worker", "exmr", "clip.mp4", "work", str(root / "params.npz"), "cuda",
                    ]), patch.dict(sys.modules), patch.object(worker, "recover_exmr", recover):
                sys.modules.pop("hamer", None)
                worker.main()
            self.assertTrue((root / "params.npz").is_file())

    def test_exmr_public_api_and_static_probabilities(self):
        params = motion_params()
        logits = np.tile([-100, 0, 100, 1, 2, 3], (4, 1))
        result = {"smplx_global": params, "static_conf_logits": logits, "fps": 30}
        run = Mock(return_value=result)
        options = Mock()
        modules = {
            "exmr": types.ModuleType("exmr"),
            "exmr.assets": types.SimpleNamespace(Assets=Mock()),
            "exmr.pipeline": types.SimpleNamespace(Options=options, run_video=run),
        }
        with patch.dict(sys.modules, modules):
            actual = worker.recover_exmr(Path("clip.mp4"), Path("work"), "cuda")
        options.assert_called_once_with(static_cam=True, render=False, fps30=True)
        self.assertEqual(actual["foot_contact"].shape, (4, 4))
        np.testing.assert_allclose(actual["foot_contact"][:, 1], 0.5)
        self.assertTrue(np.isfinite(actual["foot_contact"]).all())

    def test_flowhmr_joint_slices_and_track_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            poses = np.arange(4 * 52 * 3, dtype=np.float32).reshape(4, 52, 3) / 1000
            np.savez(
                work / "clip_seed0_origin.npz", poses=poses, trans=np.zeros((4, 3)),
                betas=np.zeros((1, 16)), mocap_framerate=30,
                foot_contacts=np.tile([-0.02, 0.25, 0.75, 1.06], (4, 1)),
            )
            engine = Mock()
            engine.process_video.return_value = str(work)
            engine_cls = Mock()
            engine_cls.build.return_value = engine
            module = types.SimpleNamespace(V2MInferenceEngine=engine_cls)
            with patch.dict(sys.modules, {"flowhmr.pipeline.v2m_inference": module}):
                np.savez(work / "clip_bbox.npz", start_end=[0, 3])
                actual = worker.recover_flowhmr(Path("clip.mp4"), work, "cuda")
                np.testing.assert_array_equal(
                    actual["left_hand_pose"], poses[:, 22:37].reshape(4, 45)
                )
                np.testing.assert_array_equal(
                    actual["right_hand_pose"], poses[:, 37:52].reshape(4, 45)
                )
                np.testing.assert_allclose(
                    actual["foot_contact"], np.tile([0, 0.25, 0.75, 1], (4, 1))
                )
                np.savez(work / "clip_bbox.npz", start_end=[2, 5])
                with self.assertRaisesRegex(ValueError, "after frame 0"):
                    worker.recover_flowhmr(Path("clip.mp4"), work, "cuda")

    def test_prompthmr_does_not_silently_drop_people_or_missing_frames(self):
        pipe = Mock()
        pipe.images = [None] * 4
        pipe.fps = 30
        person = {
            "frames": np.arange(4),
            "smplx_world": {
                "pose": np.zeros((4, 165)), "shape": np.zeros((4, 10)),
                "trans": np.zeros((4, 3)),
            },
        }
        cls = Mock(return_value=pipe)
        with patch.dict(sys.modules, {"pipeline": types.SimpleNamespace(Pipeline=cls)}):
            pipe.return_value = {"people": {1: person}}
            actual = worker.recover_prompthmr(Path("clip.mp4"), Path("work"), "cuda")
            self.assertEqual(actual["body_pose"].shape, (4, 63))
            self.assertNotIn("left_hand_pose", actual)
            pipe.return_value = {"people": {1: person, 2: person}}
            with self.assertRaisesRegex(ValueError, "2 people"):
                worker.recover_prompthmr(Path("clip.mp4"), Path("work"), "cuda")
            pipe.return_value = {"people": {1: {**person, "frames": np.arange(1, 5)}}}
            with self.assertRaisesRegex(ValueError, "complete video"):
                worker.recover_prompthmr(Path("clip.mp4"), Path("work"), "cuda")


class NativeTests(unittest.TestCase):
    def test_amass_smplh_keeps_sixteenth_beta_without_hand_pca(self):
        import torch

        params = motion_params(65)
        params["betas"] = np.zeros((1, 16), np.float32)
        params["betas"][0, 15] = 2
        params["transl"][:, 0] = np.arange(65)
        shapedirs = np.zeros((52, 3, 16), np.float32)
        shapedirs[:, 0, 15] = 0.2
        regressor = torch.eye(52)[:24]
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.npz"
            np.savez(
                model, v_template=np.zeros((52, 3), np.float32),
                shapedirs=shapedirs, posedirs=np.zeros((52, 3, 459), np.float32),
                J_regressor=np.eye(52, dtype=np.float32),
                weights=np.eye(52, dtype=np.float32),
                kintree_table=np.stack([np.zeros(52, np.int64), np.arange(52)]),
            )
            with patch("torch.load", return_value=regressor):
                actual = native.smpl24_joints(params, model, "smplh", "cpu")
        self.assertEqual(actual.shape, (65, 24, 3))
        np.testing.assert_allclose(actual[:, 0, 0], np.arange(65) + 0.4, atol=1e-5)

    def test_native_geometry_handles_short_final_batch_without_losing_frames(self):
        import torch

        params = motion_params(65)
        params["transl"][:, 0] = np.arange(65)
        batch_lengths = []

        class Model:
            def to(self, device):
                return self

            def __call__(self, **inputs):
                batch_lengths.append(len(inputs["transl"]))
                return types.SimpleNamespace(
                    vertices=inputs["transl"][:, None].expand(-1, 3, -1)
                )

        regressor = torch.zeros(24, 3)
        regressor[:, 0] = 1
        with patch("smplx.SMPLX", return_value=Model()), patch(
            "torch.load", side_effect=[regressor, torch.eye(3)]
        ):
            actual = native.smpl24_joints(params, Path("model.npz"), "smplx", "cpu")
        self.assertEqual(batch_lengths, [64, 64])
        self.assertEqual(actual.shape, (65, 24, 3))
        np.testing.assert_array_equal(actual[:, 0], params["transl"])

    def test_invalid_arrays_and_probabilities_are_rejected(self):
        invalid = {
            "body_pose": np.full((4, 63), np.nan),
            "global_orient": np.zeros((3, 3)),
            "betas": np.zeros((2, 10)),
            "fps": np.asarray(float("inf")),
            "foot_contact": np.full((4, 4), 2),
            "left_hand_pose": np.zeros((4, 15)),
        }
        for key, value in invalid.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                native.validate_params({**motion_params(), key: value})

    def test_partial_motion_is_not_stretched_and_valid_motion_preserves_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            params_path, output = Path(tmp) / "params.npz", Path(tmp) / "out.npz"
            params = {**motion_params(), "foot_contact": np.full((4, 4), 0.75)}
            np.savez(params_path, **params)
            args = {
                "source": Path("clip.mp4"), "extractor": "exmr",
                "body_model": Path("model.npz"), "model_type": "smplx",
            }
            with patch.object(native, "video_duration_seconds", return_value=10), patch.object(
                native, "smpl24_joints"
            ) as joints:
                with self.assertRaisesRegex(ValueError, "partial track"):
                    native.convert_native_motion(params_path, output, **args)
                joints.assert_not_called()
            with patch.object(native, "video_duration_seconds", return_value=4 / 30), patch.object(
                native, "smpl24_joints", return_value=np.ones((4, 24, 3), np.float32)
            ), patch.object(native, "playback_fps", return_value=30):
                native.convert_native_motion(params_path, output, **args)
            with np.load(output, allow_pickle=False) as data:
                self.assertEqual(str(data["extractor"]), "exmr")
                np.testing.assert_array_equal(data["root_translation"], np.tile([-2, 0, -2], (4, 1)))
                np.testing.assert_array_equal(data["foot_contact"], params["foot_contact"])
                self.assertNotIn("hand_pose", data)

    def test_world_axis_conversion_preserves_turns_and_shaped_pelvis(self):
        from scipy.spatial.transform import Rotation

        for model in ("exmr", "flowhmr"):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp:
                params = motion_params(4)
                params["transl"] = np.array([[0, 0, 0], [1, 0, 2], [2, 0, 4], [3, 0, 6]], np.float32)
                params["global_orient"] = np.zeros((4, 3))
                params["global_orient"][:, 1] = np.deg2rad([0, 30, 90, 180])
                rest = np.array([0.1, 0.8, 0.2], np.float32)
                joints = np.repeat((params["transl"] + rest)[:, None], 24, axis=1)
                source, output = Path(tmp) / "params.npz", Path(tmp) / "out.npz"
                np.savez(source, **params)
                with patch.object(native, "video_duration_seconds", return_value=4 / 30), \
                        patch.object(native, "smpl24_joints", return_value=joints), \
                        patch.object(native, "playback_fps", return_value=30):
                    native.convert_native_motion(
                        source, output, source=Path("clip.mp4"), extractor=model,
                        body_model=Path("model.npz"), model_type="smplh" if model=="flowhmr" else "smplx",
                    )
                with np.load(output, allow_pickle=False) as data:
                    rotation = np.diag([-1, 1, -1])
                    np.testing.assert_allclose(data["joints_3d"], joints @ rotation.T)
                    np.testing.assert_allclose(
                        data["root_translation"] + rest, data["joints_3d"][:, 0], atol=1e-6,
                    )
                    original = Rotation.from_rotvec(params["global_orient"]).as_matrix()
                    actual = Rotation.from_rotvec(data["root_orientation"]).as_matrix()
                    np.testing.assert_allclose(actual, rotation @ original, atol=1e-6)
                    np.testing.assert_allclose(
                        actual[0].T @ actual[-1], original[0].T @ original[-1], atol=1e-6,
                    )

    def test_native_fingers_preserve_left_right_and_flat_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            mano = Path(tmp) / "mano.npz"
            np.savez(mano, J=np.zeros((16, 3), np.float32))
            params = motion_params()
            params["left_hand_pose"] = np.zeros((4, 45))
            params["right_hand_pose"] = np.zeros((4, 45))
            params["right_hand_pose"][:, 0] = np.pi / 2
            with patch("motion2mixamorig.extractors.hamer.ensure_mano_npz", return_value=mano):
                hands = native.native_hands(params, "exmr")
            np.testing.assert_allclose(hands["hand_pose"][:, 0], np.broadcast_to(
                np.eye(3), (4, 15, 3, 3)
            ), atol=1e-6)
            np.testing.assert_allclose(
                hands["hand_pose"][0, 1, 0], [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
                atol=1e-6,
            )


class BenchmarkTests(unittest.TestCase):
    def test_default_comparison_uses_native_fingers_and_only_one_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video, rig = root / "clip.mp4", root / "rig.fbx"
            video.write_bytes(b"video")
            rig.write_bytes(b"rig")
            ready = Mock()
            ready.check.return_value = []
            with patch.multiple(
                benchmark, get_extractor=Mock(return_value=ready),
                summarize_run=Mock(return_value={}),
            ), patch.multiple(
                benchmark.checks, check_video=Mock(return_value=None), check_rig=Mock(return_value=None),
            ), patch(
                "motion2mixamorig.pipeline.run_pipeline",
                side_effect=lambda *a, **kw: kw["output_dir"],
            ) as run, patch.object(
                comparison, "render_model_comparison", return_value=root / "compare/clip_compare.mp4",
            ) as render, redirect_stdout(io.StringIO()):
                destination = benchmark.run_benchmark(video, rig, output_dir=root / "compare")
            report = json.loads(destination.read_text())
            self.assertEqual([item["model"] for item in report["runs"]], ["gvhmr", "exmr", "flowhmr"])
            self.assertTrue(all(call.kwargs["detect_hands"] for call in run.call_args_list))
            self.assertTrue(all(call.kwargs["no_preview"] for call in run.call_args_list))
            render.assert_called_once()
            self.assertEqual(report["preview"]["panels"], [
                "Original Video", "GVHMR + HaMeR", "ExMR", "FlowHMR",
            ])

    def test_reused_runs_skip_inference_copy_deliveries_and_preserve_originals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video, rig = root / "clip.mp4", root / "rig.fbx"
            video.write_bytes(b"video")
            rig.write_bytes(b"rig")
            reuse = {label: root / label for label in ("gvhmr_hands", "exmr", "flowhmr")}
            for label, folder in reuse.items():
                folder.mkdir()
                outputs = {
                    "skeleton_motion": "clip_skeleton.npz", "mixamo_rotations": "clip_rotations.npz",
                    "mixamo_character_glb": "clip_character.glb", "mixamo_motion_bvh": "clip.bvh",
                    "diagnostics": "clip_diagnostics.json",
                }
                for name in outputs.values():
                    (folder / name).write_text("{}" if name.endswith(".json") else label)
                (folder / "clip_run.json").write_text(json.dumps({
                    "video": str(video), "rig": str(rig), "outputs": outputs,
                }))
            with patch.multiple(
                benchmark, get_extractor=Mock(), summarize_run=Mock(return_value={}),
            ), patch.multiple(
                benchmark.checks, check_video=Mock(return_value=None), check_rig=Mock(return_value=None),
            ), patch.object(comparison, "load_comparison_run"), patch(
                "motion2mixamorig.pipeline.run_pipeline",
            ) as run, redirect_stdout(io.StringIO()):
                destination = benchmark.run_benchmark(
                    video, rig, output_dir=root / "compare", reuse_runs=reuse, no_preview=True,
                )
            run.assert_not_called()
            report = json.loads(destination.read_text())
            self.assertTrue(all(item["reused"] for item in report["runs"]))
            self.assertEqual({item["run_dir"] for item in report["runs"]}, {
                str(root / "compare" / name) for name in ("gvhmr", "exmr", "flowhmr")
            })
            self.assertTrue(all((folder / "clip_character.glb").read_text() == label for label, folder in reuse.items()))
            self.assertTrue(all((Path(item["run_dir"]) / "clip.bvh").is_file() for item in report["runs"]))
            self.assertTrue(Path(report["source_metadata"]).is_file())

    def test_benchmark_records_missing_and_failed_models_without_false_winner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video, rig = root / "clip.mp4", root / "rig.fbx"
            video.write_bytes(b"video")
            rig.write_bytes(b"rig")
            missing = Mock()
            missing.check.return_value = [CheckResult("gated checkpoint", False, "import it locally")]
            ready = Mock()
            ready.check.return_value = []
            extractors = {"exmr": missing, "gvhmr": ready}
            summary = {
                "quality_report": str(root / "diagnostics.json"),
                "intersection_frame_fraction": 0.25,
            }
            with patch.multiple(
                benchmark,
                get_extractor=Mock(side_effect=extractors.__getitem__),
                summarize_run=Mock(return_value=summary),
            ), patch.multiple(
                benchmark.checks, check_video=Mock(return_value=None), check_rig=Mock(return_value=None)
            ), patch(
                "motion2mixamorig.pipeline.run_pipeline", return_value=root / "compare/gvhmr"
            ) as run, redirect_stdout(io.StringIO()):
                destination = benchmark.run_benchmark(
                    video, rig, models=("exmr", "gvhmr"), output_dir=root / "compare"
                )
            report = json.loads(destination.read_text())
            self.assertEqual([r["status"] for r in report["runs"]], ["unavailable", "completed"])
            self.assertIsNone(report["winner"])
            run.assert_called_once()
            self.assertEqual(run.call_args.kwargs["extractor"], "gvhmr")
            rig.write_bytes(b"changed rig")
            with patch.multiple(
                benchmark.checks, check_video=Mock(return_value=None), check_rig=Mock(return_value=None)
            ), self.assertRaisesRegex(ValueError, "inputs differ"):
                benchmark.run_benchmark(video, rig, output_dir=root / "compare")

    def test_failure_does_not_prevent_next_model_and_is_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video, rig = root / "clip.mp4", root / "rig.fbx"
            video.write_bytes(b"video")
            rig.write_bytes(b"rig")
            ready = Mock()
            ready.check.return_value = []
            with patch.multiple(
                benchmark, get_extractor=Mock(return_value=ready),
                summarize_run=Mock(return_value={}),
            ), patch.multiple(
                benchmark.checks, check_video=Mock(return_value=None), check_rig=Mock(return_value=None)
            ), patch(
                "motion2mixamorig.pipeline.run_pipeline",
                side_effect=[RuntimeError("inference failed"), root / "compare/gvhmr"],
            ), redirect_stdout(io.StringIO()):
                destination = benchmark.run_benchmark(
                    video, rig, models=("exmr", "gvhmr"), output_dir=root / "compare"
                )
            report = json.loads(destination.read_text())
            self.assertEqual([r["status"] for r in report["runs"]], ["failed", "completed"])
            self.assertIn("inference failed", report["runs"][0]["error"])

    def test_blender_backend_keeps_model_names_without_loading_ml_packages(self):
        import importlib.util

        file = paths.PROJECT_ROOT / "blender_extension/motion2mixamorig_blender/services/backend.py"
        spec = importlib.util.spec_from_file_location("model_test_backend", file)
        backend = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(backend)
        self.assertEqual(backend.EXTRACTORS, extractor_names())
        for name in extractor_names():
            args = backend.build_run_argv(
                "/python", source_type="video", source="/clip.mp4", rig="/rig.fbx",
                device="cuda", output_dir="/out", progress_jsonl="/out/progress.jsonl",
                preview=False, extractor=name,
            )
            self.assertEqual(args[args.index("--set-extractor") + 1], name)
        with self.assertRaisesRegex(ValueError, "videos only"):
            backend.build_run_argv(
                "/python", source_type="image", source="/pose.jpg", rig="/rig.fbx",
                device="cuda", output_dir="/out", progress_jsonl="/out/progress.jsonl",
                preview=False, extractor="exmr",
            )


class ComparisonTests(unittest.TestCase):
    def test_repeated_animation_frames_reuse_panels_without_dropping_source_frames(self):
        from motion2mixamorig import presentation, videos
        from motion2mixamorig.mixamo import animation, render, skinned_mesh, tpose_calibration

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video, rig, destination = root / "clip.mp4", root / "rig.fbx", root / "comparison.mp4"
            anim = types.SimpleNamespace(n=3, fps=3, person_track_sha256=None)
            view = Mock()
            view.points.side_effect = lambda points: points
            frame = np.zeros((12, 12, 3), dtype=np.uint8)
            reader = Mock()
            reader.frame.side_effect = lambda index: frame.copy()
            writer = Mock()

            def make_writer(path, *args):
                path.touch()
                return writer

            with patch.object(comparison, "load_comparison_run", return_value=anim), patch(
                "cv2.VideoCapture",
            ) as capture, patch(
                "motion2mixamorig.extract.video_native_fps", return_value=6,
            ), patch.object(presentation, "animation_view", return_value=view), patch.object(
                skinned_mesh, "load_character_asset", return_value=types.SimpleNamespace(meshes=[]),
            ), patch.object(tpose_calibration, "load_calibration"), patch.object(
                animation, "skin_frames", return_value=[[np.zeros((3, 3))]],
            ) as skin, patch.multiple(
                render, ground_backdrop=Mock(side_effect=lambda *args: frame.copy()),
                draw_shadow=Mock(), draw_mesh=Mock(), label_panel=Mock(),
            ), patch.multiple(
                videos, panel_size=Mock(return_value=(12, 12)),
                _OriginalReader=Mock(return_value=reader), _StreamWriter=Mock(side_effect=make_writer),
            ), patch.object(comparison.subprocess, "run"):
                capture.return_value.get.return_value = 6
                comparison.render_model_comparison(video, rig, {model: root / model for model in comparison.COMPARISON_MODELS}, destination)
            self.assertEqual(skin.call_count, 9)
            self.assertEqual(writer.write.call_count, 6)
            self.assertEqual(reader.frame.call_count, 6)

    def test_comparison_rejects_old_native_world_axes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video, rig = root / "clip.mp4", root / "rig.fbx"
            (root / "clip_run.json").write_text(json.dumps({
                "status": "completed", "video": str(video), "rig": str(rig),
                "extractor": "flowhmr", "hands": "flowhmr",
                "outputs": {"skeleton_motion": "clip_skeleton.npz"},
            }))
            np.savez(root / "clip_skeleton.npz", joints_3d=np.zeros((1, 24, 3)))
            with self.assertRaisesRegex(ValueError, "old world axes"):
                comparison.load_comparison_run(root, video, rig, "flowhmr")

    def test_different_frame_rates_align_endpoints_without_dropping_source_frames(self):
        indices = comparison.comparison_indices(447, 449)
        self.assertEqual(len(indices), 447)
        self.assertEqual((indices[0], indices[-1]), (0, 448))
        self.assertTrue(np.all(np.diff(indices) >= 0))
        with self.assertRaises(ValueError):
            comparison.comparison_indices(0, 449)

    def test_native_panels_reject_hamer_hybrids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video, rig = root / "clip.mp4", root / "rig.fbx"
            report = {
                "status": "completed", "video": str(video), "rig": str(rig),
                "extractor": "exmr", "hands": "hamer",
            }
            (root / "clip_run.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "requires exmr fingers"):
                comparison.load_comparison_run(root, video, rig, "exmr")


class SetupTests(unittest.TestCase):
    def test_installer_preserves_existing_packages_and_skips_upstream_pins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            python = root / "python"
            python.touch()
            extractor = Mock(root=root, python=python)
            extractor.check.return_value = []
            from motion2mixamorig.models import get_model_spec

            with patch(
                "motion2mixamorig.extractors.get_extractor", return_value=extractor,
            ), patch.object(
                setup_models, "get_model_spec", return_value={
                    **get_model_spec("exmr"),
                    "packages": ("timm<1", "sapiens @ git+https://example.invalid/code", "missing-package"),
                },
            ), patch.object(setup_models.subprocess, "check_output", side_effect=[
                get_model_spec("exmr")["revision"], '["timm", "sapiens"]',
            ]), patch.object(setup_models.subprocess, "run") as run:
                setup_models.setup_model("exmr", install_deps=True)
            pip = [call.args[0] for call in run.call_args_list if "pip" in call.args[0]]
            self.assertEqual(pip, [[str(python), "-m", "pip", "install", "--no-deps", "missing-package"]])


if __name__ == "__main__":
    unittest.main()
