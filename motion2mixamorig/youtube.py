"""Download a YouTube (or web) video with yt-dlp and run the Motion2MixamoRig pipeline."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from . import paths

DEFAULT_TOR_PROXY = "socks5h://127.0.0.1:9050"


def is_url(value: str | None) -> bool:
    if not value:
        return False
    return bool(re.match(r"^https?://", value.strip(), re.IGNORECASE))


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _ensure_tor_proxy() -> str | None:
    """Return socks5h://127.0.0.1:9050 if local Tor is listening (or can be started)."""
    if _port_open("127.0.0.1", 9050):
        return DEFAULT_TOR_PROXY
    if shutil.which("systemctl"):
        subprocess.run(
            ["systemctl", "start", "tor"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if _port_open("127.0.0.1", 9050):
            return DEFAULT_TOR_PROXY
    return None


def _yt_dlp_binary() -> str:
    venv_bin = Path(sys.executable).resolve().parent / "yt-dlp"
    if venv_bin.is_file():
        return str(venv_bin)
    found = shutil.which("yt-dlp")
    if found:
        return found
    raise FileNotFoundError("yt-dlp not found; install with `uv pip install yt-dlp[default] pysocks`")


def download_youtube_video(
    url: str,
    dest_dir: Path,
    *,
    filename_stem: str | None = None,
    proxy: str | None = None,
) -> Path:
    """Download via Tor by default; an explicit proxy (empty for direct) overrides it."""
    dest_dir = Path(dest_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    yt_dlp = _yt_dlp_binary()

    selected_proxy = proxy if proxy is not None else os.environ.get("YT_DLP_PROXY")
    if selected_proxy is None:
        selected_proxy = _ensure_tor_proxy()
        if selected_proxy is None:
            raise RuntimeError(
                "Tor proxy is unavailable at 127.0.0.1:9050. Start Tor, supply --proxy URL, "
                'or explicitly use --proxy "" for a direct connection.'
            )

    out_tmpl = (
        str(dest_dir / f"{filename_stem}.%(ext)s")
        if filename_stem
        else str(dest_dir / "%(id)s.%(ext)s")
    )

    cmd = [
        yt_dlp,
        "--no-playlist",
        "--js-runtimes",
        "node",
        "-S",
        "vcodec:h264,res,acodec:m4a",
        "-f",
        "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
        "--merge-output-format",
        "mp4",
        "--print",
        "after_move:filepath",
        "-o",
        out_tmpl,
        "--proxy",
        selected_proxy,
        url,
    ]
    label = f"via proxy {selected_proxy}" if selected_proxy else "direct"
    print(f"downloading {url} ({label}) ...")
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode == 0:
        lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        for line in reversed(lines):
            p = Path(line)
            if p.is_file():
                print(f"downloaded  {p}")
                return p
        mp4s = sorted(dest_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if mp4s:
            print(f"downloaded  {mp4s[0]}")
            return mp4s[0]
    last_err = (proc.stderr or proc.stdout or "").strip()
    raise RuntimeError(f"Failed to download {url} with yt-dlp:\n{last_err}")


def _default_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def run_youtube_to_bvh(
    url_or_video: str,
    output_dir: str | Path | None = None,
    *,
    rig: str | Path | None = None,
    device: str | None = None,
    detect_hands: bool = True,
    hand_smooth_s: float = 0.08,
    no_preview: bool = False,
    foot_lock: bool = True,
    proxy: str | None = None,
) -> Path:
    """Download a YouTube URL (or use a local video) and write all outputs to `output_dir`."""
    from .pipeline import run_pipeline

    chosen_device = device or _default_device()
    rig_path = Path(rig).resolve() if rig else paths.DEFAULT_RIG

    if is_url(url_or_video):
        if output_dir is not None:
            run_dir = Path(output_dir).resolve()
            run_dir.mkdir(parents=True, exist_ok=True)
            video_path = download_youtube_video(url_or_video, run_dir, proxy=proxy)
        else:
            paths.VIDEO_DIR.mkdir(parents=True, exist_ok=True)
            video_path = download_youtube_video(url_or_video, paths.VIDEO_DIR, proxy=proxy)
            started = datetime.now()
            run_dir = paths.new_run_dir(video_path, started)
            run_dir.mkdir(parents=True, exist_ok=True)
            copied_video = run_dir / video_path.name
            if copied_video.resolve() != video_path.resolve():
                shutil.copy2(video_path, copied_video)
                video_path = copied_video
    else:
        video_path = Path(url_or_video).resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if output_dir is not None:
            run_dir = Path(output_dir).resolve()
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_dir = paths.new_run_dir(video_path, datetime.now())

    run_dir = run_pipeline(
        video_path,
        rig_path,
        device=chosen_device,
        detect_hands=detect_hands,
        hand_smooth_s=hand_smooth_s,
        output_dir=run_dir,
        no_preview=no_preview,
        foot_lock=foot_lock,
    )

    bvh_path = run_dir / f"{video_path.stem}.bvh"

    print("\n=== Output Summary ===")
    print(f"Folder:    {run_dir}")
    print(f"Video:     {video_path}")
    print(f"BVH file:  {bvh_path}")
    glb_path = run_dir / f"{video_path.stem}_character.glb"
    if not glb_path.is_file():
        glb_path = run_dir / "mixamo_character.glb"
    print(f"GLB file:  {glb_path}")
    preview = run_dir / f"{video_path.stem}_compare.mp4"
    if not preview.is_file():
        preview = run_dir / "videos" / "compare.mp4"
    if not no_preview and preview.is_file():
        print(f"Preview:   {preview}")
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="yt2bvh",
        description=(
            "Download a YouTube video (or local video) and generate Mixamo BVH + GLB + preview "
            "into a single output folder."
        ),
    )
    parser.add_argument(
        "url",
        help="YouTube URL (e.g. https://www.youtube.com/shorts/...) or local video path",
    )
    parser.add_argument(
        "output_dir_pos",
        nargs="?",
        default=None,
        metavar="output_dir",
        help="Output folder path (optional positional; can also use -o / --output-dir)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir_flag",
        default=None,
        help="Output folder path (default: outputs/<timestamp>_<video_id>/)",
    )
    parser.add_argument(
        "--rig",
        default=None,
        help="Mixamo character .fbx (default: assets/mixamo/X_Bot.fbx)",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Inference device (default: auto-detect cuda, else cpu)",
    )
    parser.add_argument(
        "--no-hands",
        action="store_true",
        help="Disable HaMeR hand/finger tracking (enabled by default)",
    )
    parser.add_argument(
        "--hand-smooth",
        type=float,
        default=0.08,
        help="Smooth finger rotations over this many seconds (default: 0.08)",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip rendering preview MP4 videos (faster if you only need .bvh / .glb)",
    )
    parser.add_argument(
        "--no-foot-lock",
        action="store_true",
        help="Disable foot contact locking",
    )
    parser.add_argument(
        "--proxy",
        default=None,
        help='Proxy override (default: local Tor or YT_DLP_PROXY); use --proxy "" for direct access',
    )
    args = parser.parse_args(argv)
    out_dir = args.output_dir_flag or args.output_dir_pos
    try:
        run_youtube_to_bvh(
            args.url,
            out_dir,
            rig=args.rig,
            device=args.device,
            detect_hands=not args.no_hands,
            hand_smooth_s=args.hand_smooth,
            no_preview=args.no_preview,
            foot_lock=not args.no_foot_lock,
            proxy=args.proxy,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
