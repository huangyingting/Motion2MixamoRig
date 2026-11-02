import importlib
import io
import json
import re
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from motion2mixamorig import paths
from motion2mixamorig.run_report import write_failed_report, write_run_json


def completed_fixture(directory, source, rig, model, hands=None):
    from motion2mixamorig.source_metadata import file_digest

    directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "skeleton_motion": f"{source.stem}_skeleton.npz",
        "mixamo_rotations": f"{source.stem}_rotations.npz",
        "mixamo_character_glb": f"{source.stem}_character.glb",
        "mixamo_motion_bvh": f"{source.stem}.bvh",
        "diagnostics": f"{source.stem}_diagnostics.json",
    }
    for filename in outputs.values():
        (directory / filename).write_bytes(model.encode())
    (directory / outputs["diagnostics"]).write_text(json.dumps({
        "source": outputs["mixamo_character_glb"], "frame_count": 2,
        "summary": {"self_intersection": {"issue_count": 0, "frame_count": 0}},
    }))
    return write_run_json(directory, {
        "status": "completed", "video": str(source), "rig": str(rig),
        "extractor": model, "hands": hands, "outputs": outputs,
        "n_frames": 2, "fps": 30,
        "source_sha256": file_digest(source), "rig_sha256": file_digest(rig),
    })


class IdempotentOutputTests(unittest.TestCase):
    def test_removing_model_previews_preserves_comparison_and_animation(self):
        from motion2mixamorig.run_report import remove_video_previews

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, rig, output = root / "clip.mp4", root / "rig.fbx", root / "gvhmr"
            source.write_bytes(b"source")
            rig.write_bytes(b"rig")
            report_path = completed_fixture(output, source, rig, "gvhmr")
            report = json.loads(report_path.read_text())
            report["outputs"]["videos"] = {"compare": "clip_compare.mp4"}
            write_run_json(output, report)
            comparison = root / "clip_compare.mp4"
            comparison.write_bytes(b"three models")
            (output / "clip_compare.mp4").write_bytes(b"one model")
            remove_video_previews(output)
            self.assertFalse((output / "clip_compare.mp4").exists())
            self.assertEqual(comparison.read_bytes(), b"three models")
            updated = json.loads(report_path.read_text())
            self.assertNotIn("videos", updated["outputs"])
            self.assertTrue(all((output / name).is_file() for name in updated["outputs"].values()))
            updated["outputs"]["videos"] = {"compare": "../clip_compare.mp4"}
            write_run_json(output, updated)
            with self.assertRaisesRegex(ValueError, "invalid preview filename"):
                remove_video_previews(output)
            self.assertTrue(comparison.is_file())

    def test_old_previews_refresh_without_rebuilding_animation(self):
        from motion2mixamorig import pipeline, preview

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, rig, output = root / "clip.mp4", root / "rig.fbx", root / "result"
            source.write_bytes(b"source")
            rig.write_bytes(b"rig")
            report_path = completed_fixture(output, source, rig, "gvhmr")
            report = json.loads(report_path.read_text())
            report["foot_lock"] = None
            write_run_json(output, report)
            with patch.object(preview, "render_saved_preview") as render, patch.object(
                pipeline, "build_animation",
            ) as build, redirect_stdout(io.StringIO()):
                self.assertEqual(pipeline.run_pipeline(source, rig, output_dir=output), output)
            render.assert_called_once_with(output, verbose=True)
            build.assert_not_called()

    def test_foot_lock_reuse_checks_request_not_contact_outcome(self):
        from motion2mixamorig.run_report import completed_run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, rig, output = root / "clip.mp4", root / "rig.fbx", root / "result"
            source.write_bytes(b"source")
            rig.write_bytes(b"rig")
            report_path = completed_fixture(output, source, rig, "gvhmr")
            report = json.loads(report_path.read_text())
            report["foot_lock"] = {"enabled": False}
            write_run_json(output, report)
            self.assertIsNotNone(completed_run(output, source, rig, "gvhmr", foot_lock=True))
            report["foot_lock_requested"] = False
            write_run_json(output, report)
            with self.assertRaisesRegex(ValueError, "foot lock differs"):
                completed_run(output, source, rig, "gvhmr", foot_lock=True)

    def test_pipeline_reuses_complete_output_and_force_invokes_extraction(self):
        from motion2mixamorig import pipeline

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, rig, output = root / "clip.mp4", root / "rig.fbx", root / "result"
            source.write_bytes(b"source")
            rig.write_bytes(b"rig")
            completed_fixture(output, source, rig, "gvhmr", "hamer")
            extractor = Mock()
            extractor.extract.side_effect = RuntimeError("extraction invoked")
            with patch.object(pipeline, "get_extractor", return_value=extractor), redirect_stdout(io.StringIO()):
                self.assertEqual(pipeline.run_pipeline(source, rig, output_dir=output, no_preview=True), output)
                extractor.extract.assert_not_called()
                with self.assertRaisesRegex(RuntimeError, "extraction invoked"):
                    pipeline.run_pipeline(source, rig, output_dir=output, no_preview=True, force=True)
                extractor.extract.assert_called_once()

    def test_changed_rig_is_not_silently_reused(self):
        from motion2mixamorig import pipeline

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, rig, output = root / "clip.mp4", root / "rig.fbx", root / "result"
            source.write_bytes(b"source")
            rig.write_bytes(b"rig")
            completed_fixture(output, source, rig, "gvhmr")
            rig.write_bytes(b"changed rig")
            with self.assertRaisesRegex(ValueError, "--force"), redirect_stdout(io.StringIO()):
                pipeline.run_pipeline(source, rig, output_dir=output, no_preview=True)

    def test_missing_export_reuses_existing_skeleton_not_inference(self):
        from motion2mixamorig import pipeline

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, rig, output = root / "clip.mp4", root / "rig.fbx", root / "result"
            source.write_bytes(b"source")
            rig.write_bytes(b"rig")
            completed_fixture(output, source, rig, "gvhmr")
            (output / "clip_character.glb").unlink()
            extractor = Mock()
            with patch.object(pipeline, "get_extractor", return_value=extractor), patch.object(
                pipeline, "load_human_motion", side_effect=RuntimeError("loading saved skeleton"),
            ), self.assertRaisesRegex(RuntimeError, "loading saved skeleton"), redirect_stdout(io.StringIO()):
                pipeline.run_pipeline(source, rig, output_dir=output, no_preview=True)
            extractor.extract.assert_not_called()

    def test_benchmark_fills_only_missing_model_and_reuses_comparison(self):
        from motion2mixamorig import benchmark, comparison

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, rig, output = root / "clip.mp4", root / "rig.fbx", root / "outputs" / "clip"
            source.write_bytes(b"source")
            rig.write_bytes(b"rig")
            completed_fixture(output / "gvhmr", source, rig, "gvhmr", "hamer")
            saved_run = output / "gvhmr" / "clip_run.json"
            old = json.loads(saved_run.read_text())
            old["outputs"]["videos"] = {"compare": "clip_compare.mp4"}
            write_run_json(saved_run.parent, old)
            (saved_run.parent / "clip_compare.mp4").write_bytes(b"obsolete single preview")
            completed_fixture(output / "flowhmr", source, rig, "flowhmr", "flowhmr")
            ready = Mock()
            ready.check.return_value = []

            def generate(video, rig, **kwargs):
                model = kwargs["extractor"]
                directory = kwargs["output_dir"]
                completed_fixture(directory, video, rig, model, "hamer" if model == "gvhmr" else model)
                return directory

            def render(video, rig, runs, destination):
                destination.write_bytes(b"preview")
                return destination

            with patch.object(paths, "OUTPUTS", root / "outputs"), patch.multiple(
                benchmark, get_extractor=Mock(return_value=ready), summarize_run=Mock(return_value={}),
            ), patch.multiple(
                benchmark.checks, check_video=Mock(return_value=None), check_rig=Mock(return_value=None),
            ), patch("motion2mixamorig.pipeline.run_pipeline", side_effect=generate) as run, patch.object(
                comparison, "render_model_comparison", side_effect=render,
            ) as preview, redirect_stdout(io.StringIO()):
                report = benchmark.run_benchmark(source, rig)
                self.assertEqual(report.parent, output)
                self.assertEqual(run.call_count, 1)
                self.assertEqual(run.call_args.kwargs["extractor"], "exmr")
                self.assertFalse((saved_run.parent / "clip_compare.mp4").exists())
                benchmark.run_benchmark(source, rig)
                self.assertEqual(run.call_count, 1)
                self.assertEqual(preview.call_count, 1)
                saved = json.loads(report.read_text())
                saved["preview"].pop("presentation_version")
                report.write_text(json.dumps(saved))
                benchmark.run_benchmark(source, rig)
                self.assertEqual(run.call_count, 1)
                self.assertEqual(preview.call_count, 2)
                benchmark.run_benchmark(source, rig, force=True)
                self.assertEqual(run.call_count, 4)
                self.assertEqual(preview.call_count, 3)
            self.assertFalse(any(path.name.startswith("202") for path in output.iterdir()))


class RunFileTests(unittest.TestCase):
    def test_selected_person_track_is_copied_with_portable_report_reference(self):
        from motion2mixamorig.run_report import copy_run_deliveries
        from motion2mixamorig.source_metadata import save_source_metadata

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, rig = root / "clip.mp4", root / "rig.fbx"
            source.write_bytes(b"source")
            rig.write_bytes(b"rig")
            old, new = root / "old", root / "new"
            path = completed_fixture(old, source, rig, "gvhmr", "hamer")
            report = json.loads(path.read_text())
            report["outputs"]["person_track"] = "clip_person_track.npz"
            (old / "clip_person_track.npz").write_bytes(b"selected track")
            write_run_json(old, report)
            copied = copy_run_deliveries(old, new, source, save_source_metadata(source, root))
            self.assertEqual((new / copied["outputs"]["person_track"]).read_bytes(), b"selected track")
            self.assertEqual(copied["person_track"], str(new / "clip_person_track.npz"))

    def test_prefixed_report_and_failed_report_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            report = {"video": "/inputs/dance.v1.mp4", "outputs": {"test": "value"}}
            path = write_run_json(directory, report)
            self.assertEqual(path.name, "dance.v1_run.json")
            failed = write_failed_report(
                directory, stage="export", error_code="FAILED", message="failed"
            )
            self.assertEqual(failed, path)
            self.assertEqual(json.loads(path.read_text())["outputs"], report["outputs"])
            self.assertFalse((directory / "run.json").exists())

    def test_legacy_reports_and_artifacts_remain_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            old = directory / "run.json"
            old.write_text(json.dumps({"video": "/inputs/dance.mp4", "fps": 30}))
            self.assertEqual(paths.find_run_file(directory, "run.json"), old)
            write_failed_report(directory, stage="preview", error_code="FAILED", message="failed")
            self.assertFalse((directory / "dance_run.json").exists())
            legacy_glb = directory / "mixamo_character.glb"
            legacy_glb.touch()
            self.assertEqual(paths.find_run_file(directory, "mixamo_character.glb"), legacy_glb)
            prefixed = directory / "dance_character.glb"
            prefixed.touch()
            self.assertEqual(paths.find_run_file(directory, "mixamo_character.glb"), prefixed)

    def test_multiple_runs_are_not_silently_mixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "dance_run.json").touch()
            (directory / "pose_run.json").touch()
            with self.assertRaises(ValueError):
                paths.find_run_file(directory, "run.json")


class PipelineLayoutTests(unittest.TestCase):
    def test_video_and_photo_outputs_are_prefixed_and_flat(self):
        from motion2mixamorig import pipeline

        for image_input in (False, True):
            with self.subTest(image=image_input), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                source = directory / ("pose.v1.jpg" if image_input else "dance.v1.mp4")
                source.touch()
                (directory / "rig.fbx").touch()
                run_dir = directory / "result"
                motion = SimpleNamespace(n_frames=2, fps=30, foot_contact=None)
                anim = SimpleNamespace(
                    r_anim_rig=[{}, {}], hips_cm_rig=[np.zeros(3)] * 2, fps=30,
                    twist_events=[], sign_events=[], foot_lock={}, ground={},
                )
                extractor = SimpleNamespace(
                    extract=lambda src, out, **kw: out.write_bytes(b"skeleton")
                )

                def write_artifact(path, *args, **kwargs):
                    path.write_bytes(b"artifact")
                    return path

                def write_rotations(anim, mapping, rig, path, **kwargs):
                    return write_artifact(path)

                def diagnose(run, *, source, **kwargs):
                    paths.run_output_path(run, source, "diagnostics.json").write_text("{}")
                    return {"summary": {"self_intersection": {}}}

                def preview(anim, asset, calibration, source, out, **kwargs):
                    from motion2mixamorig.presentation import VIEW_VERSION
                    anim.presentation_view = SimpleNamespace(metadata=lambda: {"version": VIEW_VERSION})
                    suffix = ".png" if image_input else ".mp4"
                    path = out / f"{source.stem}_compare{suffix}"
                    path.write_bytes(b"preview")
                    return {"compare": path}

                patches = {
                    "get_extractor": Mock(return_value=extractor),
                    "load_human_motion": Mock(return_value=motion),
                    "load_calibration": Mock(return_value=SimpleNamespace(skeleton_rig=None)),
                    "verify_rest_fk": Mock(return_value=0),
                    "build_mapping": Mock(return_value=None),
                    "load_character_asset": Mock(return_value=None),
                    "rest_skin_error": Mock(return_value=0),
                    "load_hand_motion": Mock(return_value=None),
                    "build_animation": Mock(return_value=anim),
                    "save_mixamo_rotations": write_rotations,
                    "write_glb": write_artifact,
                    "write_bvh": write_artifact,
                    "diagnose_run": diagnose,
                    "summary_line": Mock(return_value="diagnostics"),
                    "render_run_videos": preview,
                    "render_run_images": preview,
                    "read_hands_name": Mock(return_value=None),
                }
                with patch.multiple(pipeline, **patches), redirect_stdout(io.StringIO()):
                    pipeline.run_pipeline(
                        None if image_input else source, directory / "rig.fbx",
                        image=source if image_input else None, output_dir=run_dir,
                        verbose=False,
                    )
                report = json.loads((run_dir / f"{source.stem}_run.json").read_text())
                expected = {
                    f"{source.stem}_skeleton.npz", f"{source.stem}_rotations.npz",
                    f"{source.stem}_character.glb", f"{source.stem}.bvh",
                    f"{source.stem}_diagnostics.json", f"{source.stem}_run.json",
                    f"{source.stem}_compare{'.png' if image_input else '.mp4'}",
                }
                self.assertEqual({p.name for p in run_dir.iterdir()}, expected)
                for value in report["outputs"].values():
                    for name in (value.values() if isinstance(value, dict) else [value]):
                        self.assertEqual(Path(name).name, name)
                        self.assertTrue((run_dir / name).is_file())


class PreviewLayoutTests(unittest.TestCase):
    def test_video_writes_only_comparison_and_closes_on_error(self):
        from motion2mixamorig import videos

        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                source = directory / "dance.mp4"
                anim = SimpleNamespace(n=2, fps=30, frames=[0, 1])
                panels = tuple(np.full((2, 2, 3), k, dtype=np.uint8) for k in range(4))
                reader = Mock()
                writer = Mock(out_path=directory / "dance_compare.mp4")
                if fail:
                    writer.write.side_effect = RuntimeError("encoding failed")
                with patch.multiple(
                    videos, panel_size=Mock(return_value=(2, 2)),
                    _setup_view=Mock(return_value=(None,) * 6),
                    _draw_panels=Mock(return_value=panels),
                    video_native_fps=Mock(return_value=60),
                    _OriginalReader=Mock(return_value=reader),
                    _StreamWriter=Mock(return_value=writer),
                ):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "encoding failed"):
                            videos.render_run_videos(anim, None, None, source, directory, verbose=False)
                    else:
                        result = videos.render_run_videos(
                            anim, None, None, source, directory, verbose=False
                        )
                        self.assertEqual(result, {"compare": writer.out_path})
                        self.assertEqual(writer.write.call_count, 2)
                        np.testing.assert_array_equal(
                            writer.write.call_args.args[0], videos._grid2x2(*panels)
                        )
                        self.assertEqual(reader.frame.call_args.args[0], 2)
                    videos._StreamWriter.assert_called_once_with(
                        directory / "dance_compare.mp4", 30, 4, 4,
                        audio_src=source, audio_start_s=0, audio_dur_s=2 / 30,
                    )
                reader.close.assert_called_once()
                writer.close.assert_called_once()

    def test_photo_previews_use_source_prefix_in_run_root(self):
        from motion2mixamorig import videos

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source = directory / "pose.jpg"
            panels = tuple(np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(4))
            with patch.multiple(
                videos, panel_size=Mock(return_value=(2, 2)),
                _setup_view=Mock(return_value=(None,) * 6),
                _letterbox_still=Mock(return_value=panels[0]),
                _draw_panels=Mock(return_value=panels),
                _write_png=Mock(side_effect=lambda path, pixels: path),
                render_orbit_compare=Mock(return_value=directory / "pose_before_after_360_compare.mp4"),
            ):
                out = videos.render_run_images(
                    SimpleNamespace(n=2), None, None, source, directory, verbose=False
                )
                self.assertEqual(
                    {p.name for p in out.values()},
                    {"pose_human_skeleton.png", "pose_mixamo_skeleton.png",
                     "pose_mixamo_character.png", "pose_compare.png",
                     "pose_before_after_360_compare.mp4"},
                )
                self.assertTrue(all(p.parent == directory for p in out.values()))


class DiagnosticsLayoutTests(unittest.TestCase):
    def test_diagnose_writes_prefixed_and_legacy_reports(self):
        from motion2mixamorig import diagnose, diagnose_report

        for filename, expected in (
            ("dance.v1_character.glb", "dance.v1_diagnostics.json"),
            ("mixamo_character.glb", "diagnostics.json"),
        ):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                glb = directory / filename
                glb.touch()
                report = {"source": filename, "issues": []}
                with patch.object(diagnose, "diagnose_glb", return_value=report), patch.object(
                    diagnose, "SkinnedClip", return_value=None
                ), patch.object(diagnose_report, "write_html_report") as html:
                    diagnose.diagnose_run(directory, visualize=True, verbose=False)
                self.assertEqual(json.loads((directory / expected).read_text()), report)
                self.assertEqual(html.call_args.args[0], (directory / expected).with_suffix(".html"))

    def test_html_review_lists_actual_prefixed_artifacts(self):
        from motion2mixamorig.diagnose_report import write_html_report

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source = directory / "dance.mp4"
            source.touch()
            write_run_json(directory, {"video": str(source), "rig": "/rig.fbx"})
            names = ["dance_character.glb", "dance_rotations.npz",
                     "dance_skeleton.npz", "dance_diagnostics.json"]
            for name in names:
                (directory / name).touch()
            with patch(
                "motion2mixamorig.diagnose_report._source_video",
                return_value=({"kind": "video", "mime": "video/mp4"}, ""),
            ):
                path = write_html_report(
                    directory / "dance_diagnostics.html", directory / names[0], {},
                    SimpleNamespace(primitives=[]),
                )
            payload = json.loads(re.search(
                r'<script type="application/json" id="data">(.*?)</script>',
                path.read_text(), re.S,
            ).group(1))
            self.assertEqual(payload["files"], names)
            self.assertEqual(payload["source"]["video"], str(source))


class BlenderLayoutTests(unittest.TestCase):
    def test_blender_reads_prefixed_and_legacy_results_without_backend_imports(self):
        service_dir = (
            Path(__file__).resolve().parents[1] / "blender_extension" /
            "motion2mixamorig_blender" / "services"
        )
        package = types.ModuleType("layout_test_services")
        package.__path__ = [str(service_dir)]
        with patch.dict(sys.modules, {"layout_test_services": package}):
            reader = importlib.import_module("layout_test_services.result_reader")
            diagnostics = importlib.import_module("layout_test_services.diagnostics_reader")
            for prefix in ("dance_", ""):
                with self.subTest(prefix=prefix), tempfile.TemporaryDirectory() as tmp:
                    directory = Path(tmp)
                    glb = directory / ("dance_character.glb" if prefix else "mixamo_character.glb")
                    glb.touch()
                    (directory / f"{prefix}run.json").write_text(json.dumps({
                        "status": "completed", "fps": 30, "n_frames": 2,
                        "outputs": {"mixamo_character_glb": glb.name},
                    }))
                    (directory / f"{prefix}diagnostics.json").write_text(json.dumps({
                        "issues": [{"ranges": [[0, 1]]}], "frame_count": 2,
                    }))
                    result = reader.read_run_json(directory)
                    self.assertEqual(result.status, "completed")
                    self.assertEqual(result.glb_path, glb)
                    self.assertEqual(diagnostics.load_diagnostics(directory)["_problem_frames"], [0, 1])


if __name__ == "__main__":
    unittest.main()
