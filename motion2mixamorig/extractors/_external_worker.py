"""File-based bridge to upstream APIs. Executed by the upstream runtime's Python."""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np


def _array(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def recover_exmr(source: Path, work: Path, device: str, *, person_track: Path | None = None) -> dict:
    from exmr.assets import Assets
    from exmr.pipeline import Options, run_video

    options = Options(static_cam=True, render=False, fps30=True)
    selection = None
    if person_track is None:
        result = run_video(source, work, options, Assets(Path.cwd() / "checkpoints"))
    else:
        import torch
        from unittest.mock import patch
        from exmr import pipeline as upstream
        from exmr.preprocessing.tracker import Tracker
        from motion2mixamorig.person_tracking import read_person_track
        from motion2mixamorig.extractors.tracking import resample_boxes

        selection = read_person_track(person_track, source)
        preprocess = upstream._preprocess

        def selected_preprocess(video, frames, count, width, height, opts, assets, cache):
            boxes = resample_boxes(
                selection["bbox"], selection["source_meta"],
                (width, height, count, upstream.V.video_fps(video)),
            )
            with patch.object(Tracker, "get_main_track", return_value=torch.from_numpy(boxes)):
                return preprocess(video, frames, count, width, height, opts, assets, cache)

        with patch.object(upstream, "_preprocess", selected_preprocess):
            result = run_video(source, work, options, Assets(Path.cwd() / "checkpoints"))
    params = {key: _array(value) for key, value in result["smplx_global"].items()}
    logits = _array(result["static_conf_logits"])[:, :4]
    params["foot_contact"] = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
    params["fps"] = np.asarray(result["fps"])
    if selection is not None:
        params.update(
            person_id=np.asarray(selection["person_id"]),
            person_track_sha256=np.asarray(selection["person_track_sha256"]),
            tracking_source=np.asarray("selected_person_track"),
        )
    return params


def recover_flowhmr(source: Path, work: Path, device: str, *, person_track: Path | None = None) -> dict:
    from flowhmr.pipeline.v2m_inference import V2MInferenceEngine

    root = Path.cwd()
    engine = V2MInferenceEngine.build(
        ckpt=str(root / "checkpoints/flowhmr_latest/flowhmr_latest.ckpt"),
        model_cfg_path=str(root / "checkpoints/flowhmr_latest/config.yml"),
        device=device, postprocess=False,
    )
    from motion2mixamorig.extractors.tracking import prepare_full_frame_tracking

    tracking = prepare_full_frame_tracking(
        source, work, engine.proc, **({"person_track": person_track} if person_track is not None else {}),
    )
    directory = Path(engine.process_video(
        str(source), str(work), seed=0, static_camera=True,
    ))
    with np.load(directory / f"{source.stem}_seed0_origin.npz", allow_pickle=False) as result:
        poses = result["poses"]
        if poses.ndim != 3 or poses.shape[1:] != (52, 3):
            raise ValueError(f"expected FlowHMR SMPL-H poses (frames, 52, 3), got {poses.shape}")
        params = {
            "global_orient": poses[:, 0],
            "body_pose": poses[:, 1:22].reshape(-1, 63),
            "left_hand_pose": poses[:, 22:37].reshape(-1, 45),
            "right_hand_pose": poses[:, 37:52].reshape(-1, 45),
            "transl": result["trans"],
            "betas": result["betas"],
            "fps": result["mocap_framerate"],
            "tracking_source": np.asarray(tracking["tracking_source"] if tracking else "flowhmr"),
        }
        if tracking:
            params["tracking_run"] = np.asarray(tracking["tracking_run"])
            for key in ("person_id", "person_track_sha256"):
                if key in tracking:
                    params[key] = np.asarray(tracking[key])
        if "foot_contacts" in result:
            # FlowHMR regresses binary contacts; match upstream's probability clipping.
            params["foot_contact"] = np.clip(result["foot_contacts"], 0, 1)
    with np.load(directory / f"{source.stem}_bbox.npz", allow_pickle=False) as boxes:
        if int(boxes["start_end"][0]) != 0:
            raise ValueError("FlowHMR selected a track starting after frame 0; use a full-visibility clip")
    return params


def recover_prompthmr(source: Path, work: Path, device: str) -> dict:
    from pipeline import Pipeline

    pipeline = Pipeline(static_cam=True)
    result = pipeline(str(source), str(work), save_only_essential=True)
    people = result["people"]
    if len(people) != 1:
        raise ValueError(
            f"PromptHMR recovered {len(people)} people; this exporter requires exactly one actor"
        )
    person = next(iter(people.values()))
    frames = _array(person["frames"])
    if not np.array_equal(frames, np.arange(len(pipeline.images))):
        raise ValueError("PromptHMR track does not cover the complete video from frame 0")
    world = person["smplx_world"]
    poses = _array(world["pose"]).reshape(-1, 55, 3)
    return {
        "global_orient": poses[:, 0],
        "body_pose": poses[:, 1:22].reshape(-1, 63),
        "transl": _array(world["trans"]),
        "betas": _array(world["shape"]),
        "fps": np.asarray(pipeline.fps),
    }


def main() -> None:
    name, source, work, destination, device = sys.argv[1:6]
    selection = Path(sys.argv[6]) if len(sys.argv) == 7 else None
    # Direct script execution otherwise shadows upstream HaMeR with our hamer.py.
    script_dir = Path(__file__).resolve().parent
    sys.path[:] = [entry for entry in sys.path if Path(entry).resolve() != script_dir]
    # Upstream projects use repository-relative imports and asset paths.
    sys.path.insert(0, str(Path.cwd()))
    catalog = json.loads(Path(__file__).parents[1].joinpath("model_catalog.json").read_text())
    recover = globals()[catalog["models"][name]["worker"]]
    params = recover(
        Path(source), Path(work), device,
        **({"person_track": selection} if selection is not None else {}),
    )
    np.savez_compressed(destination, **params)


if __name__ == "__main__":
    main()
