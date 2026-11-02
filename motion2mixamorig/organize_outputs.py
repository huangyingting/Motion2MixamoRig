"""Consolidate deliveries into one current directory per source and body model."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np

from . import paths
from .models import get_model_spec
from .run_report import copy_run_deliveries, remove_video_previews, repair_delivery_references, write_run_json
from .source_metadata import file_digest, record_source_run, save_source_metadata


MODELS = ("gvhmr", "exmr", "flowhmr")


def _order(item):
    path, report, _, _ = item
    stamps = [re.sub(r"\D", "", report.get("started", ""))[:14]]
    for directory in (path.parent.parent, Path(report.get("organized_from") or path.parent).parent):
        if re.fullmatch(r"\d{8}_\d{6}(?:_\d+)?", directory.name):
            stamps.append(re.sub(r"\D", "", directory.name)[:14])
    return max(stamps)


def _rewrite(value, replacements, key=None):
    if key in {"organized_from", "original_hashes", "removed"}:
        return value
    if isinstance(value, str):
        for old, new in replacements:
            if value == old or value.startswith(old + "/"):
                return new + value[len(old):]
        return value
    if isinstance(value, list):
        return [_rewrite(item, replacements) for item in value]
    if isinstance(value, dict):
        return {name: _rewrite(item, replacements, name) for name, item in value.items()}
    return value


def organize_outputs() -> Path:
    """Keep the newest compatible paired-finger models, then remove old batches."""
    root = paths.OUTPUTS.resolve()
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "organization.json"
    groups = {}
    for path in sorted(root.rglob("*run.json")):
        relative = path.relative_to(root)
        if "cache" in relative.parts or relative.parts[0] == "blender_jobs":
            continue
        try:
            report = json.loads(path.read_text())
            source = Path(report.get("image") or report.get("video") or "")
            model = report.get("extractor") or "gvhmr"
            if model not in MODELS or not source.is_file():
                continue
            first = root / relative.parts[0]
            group = first if first.is_dir() and any(first.glob("*_source.json")) else paths.source_output_dir(source)
            groups.setdefault(group, []).append((path, report, source, model))
        except (OSError, ValueError):
            continue
    replacements = {}
    removed = []
    current = {}
    sources = {}
    for group, items in groups.items():
        ready = [
            item for item in items
            if item[1].get("status") == "completed"
            and all((item[0].parent / item[1].get("outputs", {}).get(role, "_missing_")).is_file()
                    for role in ("skeleton_motion", "mixamo_rotations", "mixamo_character_glb",
                                 "mixamo_motion_bvh", "diagnostics"))
            and (item[3] == "gvhmr" or item[1].get("hands") == item[3])
        ]
        if any(item[3] == "gvhmr" and item[1].get("hands") == "hamer" for item in ready):
            ready = [item for item in ready if item[3] != "gvhmr" or item[1].get("hands") == "hamer"]
        if not ready:
            continue
        anchor = max(ready, key=_order)
        source, rig = anchor[2], Path(anchor[1]["rig"])
        identity = file_digest(source)
        selection = anchor[1].get("person_track_sha256")
        metadata = save_source_metadata(source, group)
        sources[str(source.resolve())] = str(group)
        chosen = {}
        for model in MODELS:
            candidates = [
                item for item in ready if item[3] == model
                and file_digest(item[2]) == identity and Path(item[1]["rig"]) == rig
                and item[1].get("person_track_sha256") == selection
            ]
            if not candidates:
                continue
            target = group / model
            path, report, _, _ = max(candidates, key=_order)
            old = path.parent
            if old != target:
                stage = group / "cache" / f".organize_{model}"
                if stage.exists():
                    shutil.rmtree(stage)
                generated = copy_run_deliveries(old, stage, source, metadata, previews=paths.is_image(source))
                if (target / "cache").is_dir():
                    shutil.copytree(target / "cache", stage / "cache", symlinks=True, dirs_exist_ok=True)
                if (old / "cache").is_dir():
                    for link in (stage / "cache").rglob("*"):
                        if link.is_symlink() and (old / "cache" / link.relative_to(stage / "cache")).exists():
                            link.unlink()
                    shutil.copytree(old / "cache", stage / "cache", symlinks=True, dirs_exist_ok=True)
                expected = get_model_spec(model).get("world_to_shared")
                if expected is not None:
                    skeleton = stage / generated["outputs"]["skeleton_motion"]
                    with np.load(skeleton, allow_pickle=False) as data:
                        payload = {name: data[name] for name in data.files}
                    if "world_to_shared" not in payload:
                        from .extractors.native import normalize_world_coordinates
                        from .pipeline import run_pipeline

                        joints, transl, orient, rotation = normalize_world_coordinates(
                            {"transl": payload["root_translation"], "global_orient": payload["root_orientation"]},
                            payload["joints_3d"], model,
                        )
                        payload.update(joints_3d=joints, root_translation=transl,
                                       root_orientation=orient, world_to_shared=rotation,
                                       coordinate_convention=np.asarray("Y-up metres, shared Motion2MixamoRig world frame"))
                        normalized = stage / "cache" / "normalized_skeleton.npz"
                        normalized.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(normalized, **payload)
                        run_pipeline(source, rig, extractor=model, device="cpu",
                                     skeleton_npz=normalized, output_dir=stage,
                                     source_metadata_path=metadata, detect_hands=True,
                                     no_preview=True, force=True, verbose=False,
                                     **({"person_track": stage / generated["outputs"]["person_track"]}
                                        if "person_track" in generated["outputs"] else {}))
                        generated = json.loads(paths.find_run_file(stage, "run.json").read_text())
                for extra in old.glob("*.html"):
                    shutil.copy2(extra, stage / extra.name)
                    generated["outputs"]["diagnostics_html"] = extra.name
                generated["rig_sha256"] = file_digest(rig)
                generated["original_cache"] = str(target / "cache") if (stage / "cache").is_dir() else None
                if generated.get("person_track"):
                    generated["person_track"] = str(target / generated["outputs"]["person_track"])
                write_run_json(stage, generated)
                for name in generated["outputs"].values():
                    for filename in name.values() if isinstance(name, dict) else [name]:
                        assert (stage / filename).is_file(), (stage, filename)
                if target.exists():
                    assert target.parent == group and target.name in MODELS and not target.is_symlink()
                    shutil.rmtree(target)
                stage.rename(target)
                replacements[str(stage)] = str(target)
                print(f"organized: {old} -> {target}")
            chosen[model] = target
            if not paths.is_image(source):
                remove_video_previews(target)
            repair_delivery_references(target)
            new_report = paths.find_run_file(target, "run.json")
            record_source_run(metadata, new_report, model, "completed", operation="delivery_organization")
            current[str(new_report)] = {"status": "completed", "run_dir": str(target)}
        signatures = {
            model: file_digest(paths.find_run_file(folder, "mixamo_character.glb"))
            for model, folder in chosen.items()
        }
        previews = []
        for benchmark in group.glob("*/*_benchmark.json"):
            data = json.loads(benchmark.read_text())
            if data.get("preview", {}).get("status") != "completed":
                continue
            if data.get("video_sha256") != identity or data.get("person_track_sha256") != selection:
                continue
            runs = data.get("runs", [])
            if len(runs) != 3 or set(signatures) != set(MODELS):
                continue
            try:
                actual = {
                    entry["model"]: file_digest(paths.find_run_file(Path(entry["run_dir"]), "mixamo_character.glb"))
                    for entry in runs
                }
                file = Path(data["preview"]["file"])
                if not file.is_absolute():
                    file = benchmark.parent / file
                if actual == signatures and file.is_file():
                    previews.append((benchmark, data, file))
            except (OSError, ValueError):
                continue
        existing_benchmark = group / f"{source.stem}_benchmark.json"
        if previews:
            _, comparison, file = max(previews, key=lambda item: str(item[0]))
            destination = group / f"{source.stem}_compare.mp4"
            shutil.copy2(file, destination)
            comparison["preview"].update(file=str(destination), runs={k: str(v) for k, v in chosen.items()},
                                         run_signatures=signatures)
        elif existing_benchmark.is_file():
            comparison = json.loads(existing_benchmark.read_text())
        else:
            comparison = {"schema": "m2mr-benchmark-v1", "preview": {"status": "unavailable"}}
        comparison.update(video=str(source), video_sha256=identity, rig=str(rig),
                          rig_sha256=file_digest(rig), source_metadata=str(metadata))
        if selection:
            comparison["person_track_sha256"] = selection
        comparison["runs"] = []
        for model, folder in chosen.items():
            report = json.loads(paths.find_run_file(folder, "run.json").read_text())
            comparison["runs"].append({
                "model": model, "variant": paths.model_variant(model, report.get("hands")),
                "detect_hands": bool(report.get("hands")), "status": "completed",
                "run_dir": str(folder), "outputs": report["outputs"],
                "quality_report": str(folder / report["outputs"]["diagnostics"]),
            })
        existing_benchmark.write_text(json.dumps(comparison, indent=2) + "\n")
        for path, report, item_source, model in items:
            if model not in chosen:
                continue
            old, target = path.parent, chosen[model]
            replacements[str(old)] = str(target)
            replacements[str(path)] = str(paths.find_run_file(target, "run.json"))
            if report.get("source_metadata"):
                replacements[report["source_metadata"]] = str(metadata)
            old_outputs = report.get("outputs", {})
            new_outputs = json.loads(paths.find_run_file(target, "run.json").read_text())["outputs"]
            for role, name in old_outputs.items():
                if isinstance(name, str) and isinstance(new_outputs.get(role), str):
                    replacements[str(old / name)] = str(target / new_outputs[role])
            if old.parent == root and old != group and old not in removed:
                removed.append(old)
        for batch in group.iterdir():
            if batch.is_dir() and re.fullmatch(r"\d{8}_\d{6}(?:_\d+)?", batch.name):
                replacements[str(batch)] = str(group)
                replacements[str(batch / f"{source.stem}_benchmark.json")] = str(existing_benchmark)
                replacements[str(batch / f"{source.stem}_source.json")] = str(metadata)
                replacements[str(batch / f"{source.stem}_compare.mp4")] = str(group / f"{source.stem}_compare.mp4")
                removed.append(batch)
    for group in root.iterdir():
        if not group.is_dir() or not any(group.glob("*_source.json")):
            continue
        for batch in group.iterdir():
            if batch.is_dir() and re.fullmatch(r"\d{8}_\d{6}(?:_\d+)?", batch.name) and batch not in removed:
                if not any(batch.rglob("*_skeleton.npz")):
                    removed.append(batch)
    ordered = sorted(replacements.items(), key=lambda item: -len(item[0]))
    for path in root.rglob("*.json"):
        if any(path.is_relative_to(folder) for folder in removed) or path == receipt_path:
            continue
        data = json.loads(path.read_text())
        changed = _rewrite(data, ordered)
        if changed != data:
            path.write_text(json.dumps(changed, indent=2, ensure_ascii=False) + "\n")
    for group in groups:
        for model in ("gvhmr_hamer", "exmr_hamer", "flowhmr_hamer", "gvhmr_hands"):
            extra = group / model
            if extra.is_dir() and (group / model.split("_", 1)[0]).is_dir():
                assert extra.parent == group and not extra.is_symlink()
                shutil.rmtree(extra)
    for folder in removed:
        assert folder.parent in groups or folder.parent == root or any(folder.parent.glob("*_source.json"))
        assert not folder.is_symlink()
        shutil.rmtree(folder)
    receipt = {"schema": "m2mr-organization-v2", "runs": current, "sources": sources,
               "removed": [str(folder) for folder in removed]}
    if receipt_path.is_file():
        previous = json.loads(receipt_path.read_text())
        receipt["removed"] = sorted(set(previous.get("removed", []) + receipt["removed"]))
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt_path
