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


class RunFileTests(unittest.TestCase):
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
