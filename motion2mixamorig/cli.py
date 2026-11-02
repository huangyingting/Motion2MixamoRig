"""The `m2mr` command-line entry point: doctor / run.

    m2mr doctor                 check assets, weights and dependencies
    m2mr run                    latest video in assets/video/ + default X Bot
    m2mr run --video V --rig R  pick the inputs explicitly (flags combine freely)
    m2mr run --image I --rig R  still photo -> 3D pose on the same Mixamo path
    m2mr run --set-extractor NAME   choose how the skeleton is recovered (default: gvhmr)
    m2mr run --detect-hands         also run that extractor's hand model (GVHMR: HaMeR)

`m2mr run` performs the same asset checks as `m2mr doctor` before doing any
work, and reports everything that is missing in one pass.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from . import checks, paths
from .extractors import DEFAULT_EXTRACTOR, extractor_names, get_extractor
from .progress import ProgressWriter
from .run_report import write_failed_report

OK = "[ok]     "
MISS = "[missing]"
BAD = "[invalid]"

RIG_HINT = (
    "download a character from https://www.mixamo.com/ (FBX Binary, T-pose) "
    "into assets/mixamo/ , e.g. X_Bot.fbx"
)
VIDEO_HINT = (
    "put a *single-person* motion video (.mp4/.mov/...) into assets/video/ "
    "— group shots need track-people + --person-track"
)
IMAGE_HINT = (
    "put a *single-person* still (.jpg/.png/...) into assets/image/ "
    "— group photos are rejected before extraction"
)
SOURCE_HINT = VIDEO_HINT + "; or " + IMAGE_HINT


def _check(label: str, present: bool, hint: str) -> bool:
    print(f"{OK if present else MISS} {label}")
    if not present:
        print(f"          -> {hint}")
    return present


def _check_content(label: str, path: Path, checker) -> bool:
    """Existence is assumed; `checker` inspects the file and returns a hint or None."""
    problem = checker(path)
    print(f"{OK if problem is None else BAD} {label}")
    if problem is not None:
        print(f"          -> {path.name} {problem}")
    return problem is None


def _report_requirement(item) -> bool:
    if item.ok:
        print(f"{OK} {item.label}")
        return True
    print(f"{BAD if item.invalid else MISS} {item.label}")
    if item.hint:
        print(f"          -> {item.hint}")
    return False


def _report_hand_model(extractor, name: str, requirements) -> bool:
    """The hand model paired with `extractor`. False when it cannot run."""
    hand_model = getattr(extractor, "hand_model", None)
    if not hand_model:
        return _check(
            "hand detection      --detect-hands",
            False,
            f"{name} has no paired hand model",
        )
    ok = _check(f"hand detection      {hand_model}", True, "")
    for item in requirements:
        if item.section == "hand":
            ok &= _report_requirement(item)
    return ok


def _module_available(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def _available_rigs() -> list[Path]:
    return sorted(paths.MIXAMO_DIR.glob("*.fbx")) if paths.MIXAMO_DIR.is_dir() else []


def doctor(extractor_name: str | None = None, *, detect_hands: bool = False) -> int:
    """Check every user-supplied asset and dependency; say where to get what's missing."""
    print(f"project root: {paths.PROJECT_ROOT}\n")

    name = extractor_name or DEFAULT_EXTRACTOR
    try:
        extractor = get_extractor(name)
    except ValueError as exc:
        print(f"{BAD} motion extractor   {name}")
        print(f"          -> {exc}")
        return 1
    requirements = extractor.check(detect_hands=detect_hands)
    ok = _check(f"motion extractor   {name}", True, "")
    for item in requirements:
        if item.section == "asset" and item.audience in ("doctor", "both"):
            ok &= _report_requirement(item)
    rigs = _available_rigs()
    ok &= _check(
        f"Mixamo rig .fbx    assets/mixamo/  ({len(rigs)} found"
        + (f": {', '.join(r.name for r in rigs[:5])})" if rigs else ")"),
        bool(rigs),
        RIG_HINT,
    )
    for rig in rigs:
        ok &= _check_content(f"  - {rig.name}", rig, checks.check_rig)
    videos = paths.list_videos()
    images = paths.list_images()
    ok &= _check(
        "input source       assets/video/ + assets/image/  "
        f"({len(videos)} videos"
        + (f", latest {videos[0].name}" if videos else "")
        + f"; {len(images)} images"
        + (f", latest {images[0].name}" if images else "")
        + ")",
        bool(videos) or bool(images),
        SOURCE_HINT,
    )
    for video in videos:
        ok &= _check_content(f"  - {video.name}", video, checks.check_video)
    for image in images:
        ok &= _check_content(f"  - {image.name}", image, checks.check_image)

    if detect_hands:
        print()
        ok &= _report_hand_model(extractor, name, requirements)

    print()
    deps_ok = True
    for module, hint in (
        ("numpy", "pip install -e ."),
        ("cv2", "pip install -e ."),
    ):
        deps_ok &= _check(f"python module      {module}", _module_available(module), hint)
    for item in requirements:
        if item.section == "module" and item.audience in ("doctor", "both"):
            deps_ok &= _report_requirement(item)
    import shutil as _shutil

    _check(
        "ffmpeg (optional)  H.264 output with the source video's audio",
        _shutil.which("ffmpeg") is not None,
        "brew install ffmpeg / apt install ffmpeg — runs fine without, but videos are silent mp4v",
    )

    if extractor.weights_note:
        print(f"\ninference weights: {extractor.weights_note}")
    if detect_hands:
        note = getattr(extractor, "hand_weights_note", "")
        if note:
            print(f"hand weights: {note}")

    if ok and deps_ok:
        suggestion = "m2mr run --detect-hands" if detect_hands else "m2mr run"
        print(f"\neverything in place — try: {suggestion}")
        return 0
    print("\nfix the [missing] items above, then re-run: m2mr doctor")
    return 1


def preflight(args: argparse.Namespace) -> tuple[Path | None, Path, Path | None, Path | None]:
    """Resolve run inputs, checking everything doctor checks for this run.

    All problems are collected and reported together (same format as doctor)
    instead of failing on the first one. Exits with code 1 if anything is
    missing. Returns (video, rig, skeleton, image).
    """
    problems = 0
    person_track = Path(args.person_track).resolve() if getattr(args, "person_track", None) else None

    def check_run_video(path):
        return checks.check_video(path, **({"person_track": person_track} if person_track else {}))

    def check(label: str, present: bool, hint: str) -> bool:
        nonlocal problems
        if not _check(label, present, hint):
            problems += 1
        return present

    def check_content(label: str, path: Path | None, checker) -> None:
        """Inspect the file's contents, but only when it exists at all."""
        nonlocal problems
        if path is not None and path.exists() and not _check_content(label, path, checker):
            problems += 1

    video: Path | None = None
    image: Path | None = None
    want_image = getattr(args, "image", None) is not None
    want_video = bool(getattr(args, "video", None))

    if want_image and want_video:
        check(
            "input source       --video / --image",
            False,
            "pass only one of --video or --image",
        )
    elif want_image:
        if args.image:
            image = Path(args.image)
            check(f"input image        {image}", image.exists(), "check the --image path")
        else:
            image = paths.latest_image()
            check(
                "input image        assets/image/"
                + (f"  (latest: {image.name})" if image else "  (0 found)"),
                image is not None,
                IMAGE_HINT + ", or pass --image PATH",
            )
        if image is not None and image.exists() and not paths.is_image(image):
            check(
                f"image suffix       {image.name}",
                False,
                "use a still (.jpg/.png/webp/bmp), or pass a clip with --video",
            )
        check_content(f"image is usable    {image.name if image else ''}", image, checks.check_image)
    elif want_video:
        from .youtube import download_youtube_video, is_url

        if is_url(args.video):
            dl_dir = (
                Path(args.output_dir).resolve()
                if getattr(args, "output_dir", None)
                else paths.VIDEO_DIR
            )
            video = download_youtube_video(args.video, dl_dir)
        else:
            video = Path(args.video)
        if video.exists() and paths.is_image(video):
            image = video
            video = None
            print(f"{OK} input image        {image}  (detected from --video path)")
            check_content(f"image is usable    {image.name}", image, checks.check_image)
        else:
            check(f"input video        {video}", video.exists(), "check the --video path")
            check_content(f"video is usable    {video.name}", video, check_run_video)
    else:
        video = paths.latest_video()
        if video is None:
            image = paths.latest_image()
            check(
                "input image        assets/image/"
                + (f"  (latest: {image.name})" if image else "  (0 found)"),
                image is not None,
                SOURCE_HINT + ", or pass --video / --image",
            )
            check_content(
                f"image is usable    {image.name if image else ''}", image, checks.check_image
            )
        else:
            check(
                "input video        assets/video/"
                + f"  (latest: {video.name})",
                True,
                VIDEO_HINT + ", or pass --video",
            )
            check_content(f"video is usable    {video.name}", video, check_run_video)

    # Rig: --rig wins, else X_Bot.fbx, else the first .fbx in assets/mixamo/.
    if args.rig:
        rig = Path(args.rig)
        check(f"Mixamo rig .fbx    {rig}", rig.exists(), "check the --rig path")
    else:
        rig = paths.DEFAULT_RIG
        if not rig.exists():
            rigs = _available_rigs()
            rig = rigs[0] if rigs else None
        check(
            "Mixamo rig .fbx    " + (str(rig) if rig else "assets/mixamo/  (0 found)"),
            rig is not None,
            RIG_HINT + ", or pass --rig",
        )
    check_content(f"rig is Mixamo      {rig.name if rig else ''}", rig, checks.check_rig)

    # Skeleton reuse skips the body-model check. Hand detection still needs
    # either fingers already stored in that file, or the body cache beside it.
    detect_hands = bool(getattr(args, "detect_hands", False))
    chosen_name = getattr(args, "set_extractor", None) or DEFAULT_EXTRACTOR
    if person_track is not None:
        from .person_tracking import SUPPORTED_EXTRACTORS

        check(
            "person selection", image is None and chosen_name in SUPPORTED_EXTRACTORS,
            "use a video and --set-extractor gvhmr, exmr or flowhmr",
        )
    try:
        chosen = get_extractor(chosen_name)
        extractor_error = ""
    except ValueError as exc:
        chosen = None
        extractor_error = str(exc)

    skeleton = Path(args.skeleton) if args.skeleton else None
    if skeleton is not None:
        check(
            f"skeleton npz       {skeleton}",
            skeleton.exists(),
            "check the --skeleton path (a previous run's skeleton_motion.npz)",
        )
        check_content(f"skeleton loads     {skeleton.name}", skeleton, checks.check_skeleton_npz)
        if detect_hands and skeleton.exists():
            from .extractors.hamer import find_body_cache, skeleton_has_hands

            if skeleton_has_hands(skeleton):
                check("hand motion         already in the skeleton file", True, "")
            else:
                if chosen is None:
                    check("motion extractor   --set-extractor", False, extractor_error)
                else:
                    problems += _hand_problem_count(chosen, chosen_name)
                cache = find_body_cache(skeleton.parent)
                check(
                    "body cache         next to the reused skeleton",
                    cache is not None,
                    "re-run without --skeleton; this skeleton has no saved wrist cache",
                )
    elif chosen is None:
        check("motion extractor   --set-extractor", False, extractor_error)
    else:
        check(f"motion extractor   {chosen_name}", True, "")
        if image is not None and not getattr(chosen, "supports_images", True):
            check(
                f"{chosen_name} input", False,
                f"{chosen_name} currently accepts videos only; use gvhmr for --image",
            )
        for item in chosen.check():
            if item.section == "asset" and item.audience in ("preflight", "both"):
                if not _report_requirement(item):
                    problems += 1
        if detect_hands:
            problems += _hand_problem_count(chosen, chosen_name)

    if problems:
        print(f"\n{problems} problem(s) — nothing was run. `m2mr doctor` checks everything.")
        sys.exit(1)
    return video, rig, skeleton, image


def _hand_problem_count(extractor, name: str) -> int:
    """Print the paired hand model's requirements and return how many failed."""
    hand_model = getattr(extractor, "hand_model", None)
    if not hand_model:
        if not _check(
            "hand detection      --detect-hands",
            False,
            f"{name} has no paired hand model",
        ):
            return 1
        return 0
    failed = 0 if _check(f"hand detection      {hand_model}", True, "") else 1
    for item in extractor.check(detect_hands=True):
        if item.section == "hand" and not _report_requirement(item):
            failed += 1
    return failed


def run(args: argparse.Namespace) -> int:
    progress = ProgressWriter(getattr(args, "progress_jsonl", None))
    output_dir = Path(args.output_dir).resolve() if getattr(args, "output_dir", None) else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    progress.emit("preflight", 0.05, "Checking inputs")
    try:
        video, rig, skeleton, image = preflight(args)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code in (0, None) else 1)
        if code != 0:
            progress.emit("failed", 0.05, "Input check failed")
            write_failed_report(
                output_dir,
                stage="preflight",
                error_code="PREFLIGHT_FAILED",
                message="Input check failed",
            )
        raise

    print()

    from .pipeline import run_pipeline

    try:
        run_dir = run_pipeline(
            video,
            rig,
            image=image,
            device=args.device,
            extractor=getattr(args, "set_extractor", None) or DEFAULT_EXTRACTOR,
            detect_hands=bool(getattr(args, "detect_hands", False)),
            hand_smooth_s=float(getattr(args, "hand_smooth", 0.08)),
            skeleton_npz=skeleton,
            output_dir=output_dir,
            progress=progress,
            no_preview=bool(getattr(args, "no_preview", False)),
            foot_lock=not bool(getattr(args, "no_foot_lock", False)),
            force=bool(getattr(args, "force", False)),
            **({"person_track": Path(args.person_track)} if getattr(args, "person_track", None) else {}),
        )
    except Exception:
        import traceback

        traceback.print_exc()
        return 1
    print(f"\nresults: {run_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="m2mr",
        description=(
            "Drive a Mixamo-rigged character with human motion from a video, "
            "or a static 3D pose from a still photo."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("organize-outputs", help="consolidate deliveries into source/model folders")

    p_doctor = sub.add_parser("doctor", help="check assets, weights and dependencies")
    _add_extractor_arg(p_doctor)
    _add_detect_hands_arg(p_doctor)

    p_run = sub.add_parser("run", help="video or still -> Mixamo rig, results in outputs/")
    p_run.add_argument(
        "--video",
        help="input video (default: the file most recently added to assets/video/)",
    )
    p_run.add_argument(
        "--image",
        nargs="?",
        const="",
        default=None,
        help=(
            "still photo of one person (default: the file most recently added "
            "to assets/image/). Recovers a static 3D pose; do not combine with --video"
        ),
    )
    p_run.add_argument(
        "--rig",
        help="Mixamo character .fbx (default: assets/mixamo/X_Bot.fbx)",
    )
    p_run.add_argument(
        "--skeleton",
        help="reuse a previous run's *_skeleton.npz (or legacy skeleton_motion.npz) and skip extraction",
    )
    _add_extractor_arg(p_run)
    _add_detect_hands_arg(p_run)
    p_run.add_argument(
        "--person-track", metavar="NPZ",
        help="select one full-clip person track from `m2mr track-people`; permits group videos",
    )
    p_run.add_argument(
        "--hand-smooth",
        type=float,
        default=0.08,
        metavar="SECONDS",
        help=(
            "smooth detected finger rotations over this many seconds "
            "(0 keeps the raw estimate; default: 0.08)"
        ),
    )
    p_run.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="inference device for the extraction step (default: cpu)",
    )
    p_run.add_argument(
        "--output-dir",
        help="write this run into PATH instead of outputs/<source>/<model>/",
    )
    p_run.add_argument("--force", action="store_true", help="regenerate an existing output")
    p_run.add_argument(
        "--progress-jsonl",
        help="append machine-readable JSONL progress events to PATH",
    )
    p_run.add_argument(
        "--no-preview",
        action="store_true",
        help="skip previews; still write NPZ, GLB, BVH, and the run report",
    )
    p_run.add_argument(
        "--no-foot-lock",
        action="store_true",
        help="do not pin planted feet (keeps the raw retarget, feet may slide)",
    )

    p_diag = sub.add_parser(
        "diagnose",
        help="check a run's character GLB for self-intersection -> *_diagnostics.json",
    )
    p_diag.add_argument("run_dir", help="a run folder from `m2mr run` (contains the character GLB)")
    p_diag.add_argument(
        "--visualize",
        action="store_true",
        help="also write *_diagnostics.html: 3D view, timeline, and clickable issue list",
    )
    p_diag.add_argument(
        "--threshold-cm",
        type=float,
        default=None,
        help="report penetrations deeper than this (default: 1.0)",
    )
    p_bench = sub.add_parser(
        "benchmark", help="compare available extractors on one video and rig",
    )
    p_bench.add_argument("--video", required=True, help="local, single-person video")
    p_bench.add_argument("--rig", default=str(paths.DEFAULT_RIG))
    p_bench.add_argument("--extractors", nargs="+", choices=extractor_names())
    p_bench.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda")
    p_bench.add_argument("--output-dir", help="comparison folder (default: outputs/<source>/)")
    p_bench.add_argument("--force", action="store_true", help="regenerate existing model outputs and comparison")
    p_bench.add_argument(
        "--include-hands", action="store_true", help="include paired fingers for explicitly requested extractors",
    )
    p_bench.add_argument("--no-preview", action="store_true")
    p_bench.add_argument("--person-track", metavar="NPZ", help="use the same selected dancer for every model")
    p_bench.add_argument(
        "--reuse-run", action="append", default=[], metavar="VARIANT=DIR",
        help="reuse a completed same-video/same-rig run (default variants: gvhmr_hamer, exmr, flowhmr)",
    )
    p_setup = sub.add_parser("setup-model", help="prepare a pinned model checkout using the current venv")
    p_setup.add_argument("model", choices=extractor_names())
    p_setup.add_argument(
        "--install-deps", action="store_true",
        help="add missing optional packages without replacing installed packages; native libraries and licensed assets are manual",
    )
    p_people = sub.add_parser("track-people", help="track people in a group video and write selectable NPZ tracks")
    p_people.add_argument("--video", required=True, help="local video")
    p_people.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p_people.add_argument("--output-dir", help="new, empty folder for track files and ID preview")
    p_people.add_argument(
        "--max-gap-frames", type=int, default=0,
        help="allow short internal box-detection gaps in the same ID (default: 0); never pads motion",
    )
    return parser


def diagnose(args: argparse.Namespace) -> int:
    from .diagnose import DEFAULT_THRESHOLD_CM, DIAGNOSTICS_HTML, DIAGNOSTICS_JSON, diagnose_run, summary_line

    run_dir = Path(args.run_dir)
    threshold = DEFAULT_THRESHOLD_CM if args.threshold_cm is None else args.threshold_cm
    try:
        report = diagnose_run(run_dir, threshold_cm=threshold, visualize=args.visualize)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(summary_line(report))
    print(f"report: {paths.find_run_file(run_dir, DIAGNOSTICS_JSON)}")
    if args.visualize:
        print(f"view:   {paths.find_run_file(run_dir, DIAGNOSTICS_HTML)}")
    return 0


def _add_detect_hands_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--detect-hands",
        action="store_true",
        help=(
            "turn on the hand detection bundled with the selected body extractor "
            "(GVHMR uses HaMeR). Needs assets/body_models/mano/MANO_RIGHT.pkl. Off by default"
        ),
    )


def _add_extractor_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--set-extractor",
        default=DEFAULT_EXTRACTOR,
        choices=list(extractor_names()),
        help=(
            "how to turn the video or still into skeleton_motion.npz "
            f"(default: {DEFAULT_EXTRACTOR})"
        ),
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "track-people":
        from .person_tracking import track_people

        try:
            summary = track_people(
                Path(args.video), device=args.device,
                output_dir=Path(args.output_dir) if args.output_dir else None,
                max_gap_frames=args.max_gap_frames,
            )
            import json
            data = json.loads(summary.read_text())
            print(f"people: {summary}")
            print(f"preview: {summary.parent / data['preview']}")
            for item in data["tracks"]:
                print(f"person {item['person_id']}: {item.get('file') or item['reason']}")
            sys.exit(0 if any(item["selectable"] for item in data["tracks"]) else 1)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)
    if args.command == "organize-outputs":
        from .organize_outputs import organize_outputs

        receipt = organize_outputs()
        print(f"organization report: {receipt}")
        import json

        data = json.loads(receipt.read_text())
        sys.exit(1 if any(item["status"] == "failed" for item in data["runs"].values()) else 0)
    if args.command == "doctor":
        sys.exit(
            doctor(
                getattr(args, "set_extractor", None),
                detect_hands=bool(getattr(args, "detect_hands", False)),
            )
        )
    if args.command == "diagnose":
        sys.exit(diagnose(args))
    if args.command == "setup-model":
        from .setup_models import setup_model

        try:
            print(f"setup receipt: {setup_model(args.model, install_deps=args.install_deps)}")
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)
        sys.exit(0)
    if args.command == "benchmark":
        from .benchmark import run_benchmark

        try:
            reuse = {}
            for item in args.reuse_run:
                variant, separator, directory = item.partition("=")
                if not separator or not variant or not directory or variant in reuse:
                    raise ValueError("--reuse-run must be unique VARIANT=DIR pairs")
                reuse[variant] = Path(directory)
            report_path = run_benchmark(
                Path(args.video), Path(args.rig),
                models=tuple(args.extractors) if args.extractors else None,
                device=args.device,
                output_dir=Path(args.output_dir) if args.output_dir else None,
                include_hands=args.include_hands, no_preview=args.no_preview, reuse_runs=reuse,
                force=args.force,
                **({"person_track": Path(args.person_track)} if args.person_track else {}),
            )
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)
        import json

        report = json.loads(report_path.read_text())
        if report.get("preview", {}).get("status") == "failed":
            print(f"comparison failed: {report['preview']['error']}", file=sys.stderr)
            sys.exit(1)
        sys.exit(0 if any(item["status"] == "completed" for item in report["runs"]) else 1)
    sys.exit(run(args))


if __name__ == "__main__":
    main()
