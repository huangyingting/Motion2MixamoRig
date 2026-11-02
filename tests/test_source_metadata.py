import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from motion2mixamorig import paths
from motion2mixamorig.source_metadata import file_digest, probe_media, record_source_run, safe_source_url, save_source_metadata


class SourceMetadataTests(unittest.TestCase):
    def test_stable_model_directories_and_different_source_contents(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first, second = root / "a" / "dance.mp4", root / "b" / "dance.mp4"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            with patch.object(paths, "OUTPUTS", root / "outputs"), patch(
                "motion2mixamorig.source_metadata.probe_media", return_value={"status": "unavailable"},
            ):
                when = datetime(2026, 1, 1)
                run = paths.new_run_dir(first, when, model="gvhmr_hamer")
                again = paths.new_run_dir(first, when, model="exmr")
                other = paths.new_run_dir(second, when, model="flowhmr")
                self.assertEqual(run.relative_to(root / "outputs"), Path("dance/gvhmr"))
                self.assertEqual(again.parent, run.parent)
                self.assertEqual(paths.new_run_dir(first, when, model="gvhmr_hamer"), run)
                self.assertNotEqual(other.parent, run.parent)
                metadata = json.loads((run.parent / "dance_source.json").read_text())
                self.assertEqual(metadata["file"]["sha256"], file_digest(first))
                self.assertEqual(list(run.parent.glob("*.json")), [run.parent / "dance_source.json"])

    def test_existing_empty_source_directory_is_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "clip.mp4"
            source.touch()
            (root / "outputs" / "clip").mkdir(parents=True)
            with patch.object(paths, "OUTPUTS", root / "outputs"):
                self.assertEqual(paths.source_output_dir(source), root / "outputs" / "clip")

    def test_registered_derived_source_reuses_its_existing_group(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "clip_visible.mp4"
            source.write_bytes(b"source")
            with patch.object(paths, "OUTPUTS", root / "outputs"):
                canonical = root / "outputs" / "clip"
                save_source_metadata(source, canonical)
                (root / "outputs" / "organization.json").write_text(json.dumps({
                    "schema": "m2mr-organization-v2",
                    "sources": {str(source.resolve()): str(canonical)},
                }))
                self.assertEqual(paths.source_output_dir(source), canonical)

    def test_media_allowlist_retains_timing_audio_rotation_not_secrets(self):
        raw = {
            "format": {"duration": "2.4", "format_name": "mov,mp4", "tags": {"token": "SECRET"}},
            "streams": [
                {"codec_type": "video", "width": 720, "height": 1280, "nb_frames": "72",
                 "avg_frame_rate": "30/1", "time_base": "1/15360",
                 "side_data_list": [{"rotation": 90, "secret": "SECRET"}],
                 "tags": {"comment": "SECRET"}},
                {"codec_type": "audio", "sample_rate": "48000", "channels": 2},
            ],
        }
        with patch("motion2mixamorig.source_metadata.shutil.which", return_value="ffprobe"), patch(
            "motion2mixamorig.source_metadata.subprocess.run",
        ) as run:
            run.return_value.returncode = 0
            run.return_value.stdout = json.dumps(raw)
            metadata = probe_media(Path("clip.mp4"))
        self.assertNotIn("SECRET", json.dumps(metadata))
        self.assertEqual(metadata["streams"][0]["rotation_degrees"], 90)
        self.assertEqual(metadata["streams"][1]["channels"], 2)
        self.assertEqual(metadata["streams"][0]["variable_frame_rate"], "unknown")

    def test_safe_provenance_and_explicit_directory_hash_conflicts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "clip.mp4"
            source.write_bytes(b"first")
            with patch("motion2mixamorig.source_metadata.probe_media", return_value={"status": "unavailable"}):
                record = save_source_metadata(source, root / "records", provenance={
                    "title": "Dance", "url": "https://user:SECRET@www.youtube.com/watch?v=123&token=SECRET#SECRET",
                    "cookies": "SECRET",
                })
                original = record.read_bytes()
                source.write_bytes(b"second")
                other = save_source_metadata(source, root / "records")
            self.assertNotEqual(record, other)
            self.assertEqual(record.read_bytes(), original)
            self.assertNotIn("SECRET", record.read_text())
            self.assertEqual(json.loads(other.read_text())["file"]["sha256"], file_digest(source))
            self.assertEqual(safe_source_url("https://example.com/video?signature=SECRET"), "https://example.com/video")

    def test_probe_failure_does_not_prevent_metadata(self):
        with patch("motion2mixamorig.source_metadata.shutil.which", return_value="ffprobe"), patch(
            "motion2mixamorig.source_metadata.subprocess.run", side_effect=OSError("missing"),
        ):
            self.assertEqual(probe_media(Path("clip.mp4"))["status"], "unavailable")

    def test_processing_history_is_persistent_and_not_duplicated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "clip.mp4"
            source.touch()
            with patch("motion2mixamorig.source_metadata.probe_media", return_value={"status": "unavailable"}):
                metadata = save_source_metadata(source, root / "outputs")
            report = root / "outputs" / "batch" / "exmr" / "clip_run.json"
            record_source_run(metadata, report, "exmr", "completed")
            record_source_run(metadata, report, "exmr", "completed")
            history = json.loads(metadata.read_text())["processing"]
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["report"], str(report))
            self.assertFalse(any(p.name.startswith("tmp") for p in metadata.parent.iterdir()))

    def test_organizing_legacy_run_is_resumable_and_removes_superseded_layout(self):
        import numpy as np
        from motion2mixamorig.organize_outputs import organize_outputs

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "clip.mp4"
            source.write_bytes(b"source")
            (root / "rig.fbx").write_bytes(b"rig")
            old = root / "outputs" / "old_run"
            old.mkdir(parents=True)
            outputs = {
                "skeleton_motion": "skeleton_motion.npz", "mixamo_rotations": "mixamo_rotations.npz",
                "mixamo_character_glb": "mixamo_character.glb", "mixamo_motion_bvh": "clip.bvh",
                "diagnostics": "diagnostics.json",
            }
            np.savez(old / "skeleton_motion.npz", joints_3d=np.zeros((2, 24, 3)))
            for name in ("mixamo_rotations.npz", "mixamo_character.glb", "clip.bvh"):
                (old / name).write_bytes(b"delivery")
            (old / "diagnostics.json").write_text(json.dumps({"source": "mixamo_character.glb"}))
            report = {
                "video": str(source), "rig": str(root / "rig.fbx"), "extractor": "gvhmr",
                "status": "completed", "started": "2026-01-01T00:00:00", "outputs": outputs,
            }
            (old / "run.json").write_text(json.dumps(report))
            original = {p.name: p.read_bytes() for p in old.iterdir()}
            with patch.multiple(paths, OUTPUTS=root / "outputs", list_videos=lambda: [source], list_images=lambda: []), patch(
                "motion2mixamorig.source_metadata.probe_media", return_value={"status": "unavailable"},
            ):
                receipt_path = organize_outputs()
                receipt = json.loads(receipt_path.read_text())
                folder = root / "outputs" / "clip" / "gvhmr"
                self.assertEqual(set(receipt["runs"]), {str(folder / "clip_run.json")})
                self.assertTrue((folder / "clip_character.glb").is_file())
                self.assertEqual(json.loads((folder / "clip_diagnostics.json").read_text())["source"], "clip_character.glb")
                organize_outputs()
                self.assertEqual(json.loads(receipt_path.read_text()), receipt)
            self.assertFalse(old.exists())
            self.assertEqual((folder / "clip_character.glb").read_bytes(), original["mixamo_character.glb"])

    def test_consolidation_keeps_latest_three_paired_models_and_removes_extra_versions(self):
        import numpy as np
        from motion2mixamorig.models import get_model_spec
        from motion2mixamorig.organize_outputs import organize_outputs
        from test_output_layout import completed_fixture

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, rig = root / "clip.mp4", root / "rig.fbx"
            source.write_bytes(b"source")
            rig.write_bytes(b"rig")
            outputs = root / "outputs"
            with patch.object(paths, "OUTPUTS", outputs):
                save_source_metadata(source, outputs / "clip")
                for stamp in ("20260101_000000", "20260102_000000"):
                    for model in ("gvhmr", "exmr", "flowhmr"):
                        hands = "hamer" if model == "gvhmr" else model
                        folder = outputs / "clip" / stamp / paths.model_variant(model, hands)
                        path = completed_fixture(folder, source, rig, model, hands)
                        report = json.loads(path.read_text())
                        report["started"] = stamp
                        path.write_text(json.dumps(report))
                        (folder / "clip_character.glb").write_bytes(f"{stamp}_{model}".encode())
                        np.savez(folder / "clip_skeleton.npz", joints_3d=np.zeros((2, 24, 3)),
                                 world_to_shared=get_model_spec(model).get("world_to_shared"))
                    completed_fixture(outputs / "clip" / stamp / "gvhmr", source, rig, "gvhmr")
                organize_outputs()
                for model in ("gvhmr", "exmr", "flowhmr"):
                    self.assertEqual((outputs / "clip" / model / "clip_character.glb").read_bytes(),
                                     f"20260102_000000_{model}".encode())
                self.assertFalse(any(path.name.startswith("202") for path in (outputs / "clip").iterdir()))
                self.assertFalse((outputs / "clip" / "gvhmr_hamer").exists())


if __name__ == "__main__":
    unittest.main()
