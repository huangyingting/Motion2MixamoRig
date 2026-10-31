#!/usr/bin/env bash
# ==============================================================================
# One-Click Installer: YouTube -> Mixamo BVH / GLB Pipeline (yt2bvh + m2mr)
# Uses `uv` to manage Python 3.12 virtual environment and dependencies.
#
# Usage on any new machine:
#   bash install_yt2bvh.sh [INSTALL_DIR] [--skip-weights]
#
# Examples:
#   bash install_yt2bvh.sh
#   bash install_yt2bvh.sh /opt/Motion2MixamoRig
# ==============================================================================
set -euo pipefail

INSTALL_DIR=""
SKIP_WEIGHTS=0

for arg in "$@"; do
    case "$arg" in
        --skip-weights)
            SKIP_WEIGHTS=1
            ;;
        -h|--help)
            echo "Usage: bash $0 [INSTALL_DIR] [--skip-weights]"
            exit 0
            ;;
        *)
            if [[ -z "$INSTALL_DIR" ]]; then
                INSTALL_DIR="$arg"
            fi
            ;;
    esac
done

if [[ -z "$INSTALL_DIR" ]]; then
    if [[ -d "/data" && -w "/data" ]]; then
        INSTALL_DIR="/data/Motion2MixamoRig"
    else
        INSTALL_DIR="$HOME/Motion2MixamoRig"
    fi
fi

echo "============================================================"
echo " Installing Motion2MixamoRig + yt2bvh into: $INSTALL_DIR"
echo "============================================================"

# ------------------------------------------------------------------------------
# 1. System packages (git, curl, ffmpeg, nodejs, tor)
# ------------------------------------------------------------------------------
echo "[1/7] Checking system dependencies (git, curl, ffmpeg, node, tor)..."
if command -v apt-get >/dev/null 2>&1; then
    MISSING_PKGS=()
    command -v git >/dev/null 2>&1    || MISSING_PKGS+=(git)
    command -v curl >/dev/null 2>&1   || MISSING_PKGS+=(curl ca-certificates)
    command -v ffmpeg >/dev/null 2>&1 || MISSING_PKGS+=(ffmpeg)
    command -v node >/dev/null 2>&1   || MISSING_PKGS+=(nodejs)
    command -v tor >/dev/null 2>&1    || MISSING_PKGS+=(tor)
    if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
        echo "  -> Installing system packages via apt: ${MISSING_PKGS[*]}"
        sudo apt-get update -y -qq
        sudo apt-get install -y -qq "${MISSING_PKGS[@]}"
    fi
    if command -v systemctl >/dev/null 2>&1 && command -v tor >/dev/null 2>&1; then
        sudo systemctl enable --now tor >/dev/null 2>&1 || true
    fi
elif command -v brew >/dev/null 2>&1; then
    for pkg in git curl ffmpeg node tor; do
        command -v "$pkg" >/dev/null 2>&1 || brew install "$pkg"
    done
fi

# ------------------------------------------------------------------------------
# 2. Ensure `uv` is installed
# ------------------------------------------------------------------------------
echo "[2/7] Checking uv package manager..."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "  -> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
echo "  -> Using $(uv --version)"

# Put uv cache on the same volume as INSTALL_DIR if on /data to avoid filling /
INSTALL_PARENT="$(dirname "$INSTALL_DIR")"
mkdir -p "$INSTALL_PARENT"
if [[ "$INSTALL_DIR" == /data/* && -w /data ]]; then
    export UV_CACHE_DIR="/data/uv-cache"
    export TMPDIR="/data/tmp"
    mkdir -p "$UV_CACHE_DIR" "$TMPDIR"
fi

# ------------------------------------------------------------------------------
# 3. Clone or update Motion2MixamoRig repository
# ------------------------------------------------------------------------------
echo "[3/7] Preparing Motion2MixamoRig repository at $INSTALL_DIR..."
if [[ ! -d "$INSTALL_DIR/.git" ]]; then
    git clone https://github.com/huangyingting/Motion2MixamoRig.git "$INSTALL_DIR" \
        || git clone https://github.com/MaxLiu22/Motion2MixamoRig.git "$INSTALL_DIR"
else
    echo "  -> Repository already exists at $INSTALL_DIR"
fi
cd "$INSTALL_DIR"

# ------------------------------------------------------------------------------
# 4. Inject BVH exporter (pose-studio & Blender compatible) + yt2bvh CLI module
# ------------------------------------------------------------------------------
echo "[4/7] Installing BVH exporter & YouTube-to-BVH pipeline modules..."

cat <<'EOF_BVH' > "$INSTALL_DIR/motion2mixamorig/mixamo/bvh.py"
"""Write an animated Mixamo skeleton to a Biovision Hierarchy (.bvh) motion file.

Uses the exact same bind pose, per-frame rest-relative rotations (`r_anim_rig`),
and root translations (`hips_cm_rig`) as `write_glb` in `gltf.py`, so the
exported `.bvh` matches `mixamo_character.glb` frame-for-frame and imports
cleanly into Three.js BVHLoader (`pose-studio`) and Blender.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .fbx_skeleton import bone_suffix
from .kinematics import MixamoRig, forward_kinematics
from .skinned_mesh import YBotAsset, load_character_asset


def _rotmat_to_euler_zxy_deg(r: np.ndarray) -> tuple[float, float, float]:
    """Decompose 3x3 rotation matrix R = Rz(z) @ Rx(x) @ Ry(y) into degrees (z, x, y).

    For R = Rz(z) * Rx(x) * Ry(y):
        R = [
            [cz*cy - sz*sx*sy, -sz*cx, cz*sy + sz*sx*cy],
            [sz*cy + cz*sx*sy,  cz*cx, sz*sy - cz*sx*cy],
            [          -cx*sy,     sx,            cx*cy],
        ]
    """
    sx = float(np.clip(r[2, 1], -1.0, 1.0))
    x = float(np.arcsin(sx))
    cx = float(np.cos(x))
    if abs(cx) > 1e-6:
        z = float(np.arctan2(-r[0, 1], r[1, 1]))
        y = float(np.arctan2(-r[2, 0], r[2, 2]))
    else:
        z = float(np.arctan2(r[1, 0], r[0, 0]))
        y = 0.0
    rad2deg = 180.0 / np.pi
    return z * rad2deg, x * rad2deg, y * rad2deg


def _compute_world_bind_positions(rig: MixamoRig) -> dict[str, np.ndarray]:
    """Compute world-space bind positions (in rig units, cm) for every bone."""
    fk_rest = forward_kinematics(rig)
    return {name: np.asarray(mat[:3, 3], dtype=np.float64) for name, mat in fk_rest.items()}


def write_bvh(
    out_path: Path,
    asset: YBotAsset,
    r_anim_rig: list[dict[str, np.ndarray]],
    hips_cm_rig: list[np.ndarray],
    fps: float,
    *,
    scale: float = 0.01,
) -> Path:
    """Write a `.bvh` file with world-aligned zero-rotation rest pose in meters (`scale=0.01`).

    To ensure universal compatibility with BVH importers (Three.js `BVHLoader` in
    `pose-studio`, Blender, MotionBuilder, Unity):
    - Bone names strip any `mixamorig:` prefix (colons break Three.js PropertyBinding).
    - `ROOT Hips` uses `OFFSET 0 0 0` so `BVHLoader` does not double-add root offset.
    - All joint axes at rest are aligned with the global coordinate axes (identity rest
      orientation). At frame `t`, if a bone's world rotation is `W_t[i]` and its parent's
      world rotation is `W_t[p]`, the BVH joint rotation matrix is:
          `R_bvh[i] = W_t[p]^T @ W_t[i]` (and `W_t[root]` for the root joint).
    """
    rig = asset.rig
    root_name = rig.root
    rest_pos_cm = _compute_world_bind_positions(rig)
    rest_pos_m = {name: pos * scale for name, pos in rest_pos_cm.items()}

    joint_order: list[str] = []
    lines: list[str] = ["HIERARCHY"]

    def emit_node(name: str, depth: int, is_root: bool = False) -> None:
        joint_order.append(name)
        bone = rig.bones[name]
        bvh_name = bone_suffix(name)
        indent = "\t" * depth
        inner = "\t" * (depth + 1)
        if is_root:
            lines.append(f"{indent}ROOT {bvh_name}")
            offset = np.zeros(3, dtype=np.float64)
        else:
            lines.append(f"{indent}JOINT {bvh_name}")
            assert bone.parent is not None
            offset = rest_pos_m[name] - rest_pos_m[bone.parent]
        lines.append(f"{indent}{{")
        lines.append(f"{inner}OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}")
        if is_root:
            lines.append(
                f"{inner}CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation"
            )
        else:
            lines.append(f"{inner}CHANNELS 3 Zrotation Xrotation Yrotation")

        children = [c for c in bone.children if c in rig.bones]
        if children:
            for child in children:
                emit_node(child, depth + 1, is_root=False)
        else:
            if bone.parent is not None:
                parent_vec = rest_pos_m[name] - rest_pos_m[bone.parent]
                norm = float(np.linalg.norm(parent_vec))
                end_offset = parent_vec * 0.25 if norm > 1e-6 else np.array([0.0, 0.05, 0.0])
            else:
                end_offset = np.array([0.0, 0.05, 0.0])
            lines.append(f"{inner}End Site")
            lines.append(f"{inner}{{")
            lines.append(
                f"{inner}\tOFFSET {end_offset[0]:.6f} {end_offset[1]:.6f} {end_offset[2]:.6f}"
            )
            lines.append(f"{inner}}}")
        lines.append(f"{indent}}}")

    emit_node(root_name, 0, is_root=True)

    n_frames = len(r_anim_rig)
    frame_time = 1.0 / float(fps) if fps > 0 else 1.0 / 30.0
    lines.append("MOTION")
    lines.append(f"Frames: {n_frames}")
    lines.append(f"Frame Time: {frame_time:.6f}")

    for f_idx in range(n_frames):
        fk = forward_kinematics(rig, r_anim_rig[f_idx], hips_cm_rig[f_idx])
        root_pos = fk[root_name][:3, 3] * scale
        vals: list[str] = [
            f"{root_pos[0]:.6f}",
            f"{root_pos[1]:.6f}",
            f"{root_pos[2]:.6f}",
        ]
        for name in joint_order:
            bone = rig.bones[name]
            w_curr = fk[name][:3, :3]
            if bone.parent is None:
                r_bvh = w_curr
            else:
                w_parent = fk[bone.parent][:3, :3]
                r_bvh = w_parent.T @ w_curr
            z_deg, x_deg, y_deg = _rotmat_to_euler_zxy_deg(r_bvh)
            vals.append(f"{z_deg:.6f}")
            vals.append(f"{x_deg:.6f}")
            vals.append(f"{y_deg:.6f}")
        lines.append(" ".join(vals))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def write_bvh_from_npz(
    rotations_npz: Path,
    out_bvh: Path,
    *,
    rig_path: Path | None = None,
    scale: float = 0.01,
) -> Path:
    """Export a `.bvh` file directly from a saved `mixamo_rotations.npz` artifact."""
    from .kinematics import rotmat_from_quat_wxyz

    data = np.load(rotations_npz, allow_pickle=True)
    bone_names = [str(b) for b in data["bone_names"]]
    q_rig = np.asarray(data["quat_wxyz_rig"], dtype=np.float64)
    hips_cm_rig = [np.asarray(h, dtype=np.float64) for h in data["hips_translation_cm_rig"]]
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    resolved_rig = Path(rig_path) if rig_path is not None else Path(str(data["rig"]))

    asset = load_character_asset(resolved_rig, textures=False)
    n_frames = len(q_rig)
    r_anim_rig: list[dict[str, np.ndarray]] = []
    for i in range(n_frames):
        frame_rots: dict[str, np.ndarray] = {}
        for b_idx, b_name in enumerate(bone_names):
            frame_rots[b_name] = rotmat_from_quat_wxyz(q_rig[i, b_idx])
        r_anim_rig.append(frame_rots)

    return write_bvh(out_bvh, asset, r_anim_rig, hips_cm_rig, fps, scale=scale)
EOF_BVH

cat <<'EOF_YT' > "$INSTALL_DIR/motion2mixamorig/youtube.py"
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
    """Download `url` as an H.264/MP4 file into `dest_dir` and return the saved file path."""
    dest_dir = Path(dest_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    yt_dlp = _yt_dlp_binary()

    env_proxy = proxy or os.environ.get("YT_DLP_PROXY")
    tor_proxy = _ensure_tor_proxy()
    if env_proxy:
        proxy_candidates: list[str | None] = [env_proxy]
    elif tor_proxy:
        # Try direct first on residential machines, then Tor proxy if datacenter IP is blocked;
        # or if Tor is open, try Tor then direct.
        proxy_candidates = [None, tor_proxy]
    else:
        proxy_candidates = [None]

    out_tmpl = (
        str(dest_dir / f"{filename_stem}.%(ext)s")
        if filename_stem
        else str(dest_dir / "%(id)s.%(ext)s")
    )

    last_err = ""
    for candidate in proxy_candidates:
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
        ]
        if candidate:
            cmd.extend(["--proxy", candidate])
        cmd.append(url)

        label = f"via proxy {candidate}" if candidate else "direct"
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
        print(
            f"warning: yt-dlp ({label}) failed: "
            f"{last_err.splitlines()[-1] if last_err else 'unknown error'}"
        )

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

    bvh_path = run_dir / "mixamo_motion.bvh"
    named_bvh = run_dir / f"{video_path.stem}.bvh"
    if bvh_path.is_file() and named_bvh.resolve() != bvh_path.resolve():
        shutil.copy2(bvh_path, named_bvh)

    print("\n=== Output Summary ===")
    print(f"Folder:    {run_dir}")
    print(f"Video:     {video_path}")
    print(f"BVH file:  {named_bvh}")
    print(f"BVH (std): {bvh_path}")
    print(f"GLB file:  {run_dir / 'mixamo_character.glb'}")
    if not no_preview and (run_dir / "videos" / "compare.mp4").is_file():
        print(f"Preview:   {run_dir / 'videos' / 'compare.mp4'}")
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
        help="Mixamo character .fbx (default: assets/mixamo/Y_Bot.fbx)",
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
        help="Optional proxy URL for yt-dlp (e.g. socks5h://127.0.0.1:9050)",
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
EOF_YT

# Patch pipeline.py and pyproject.toml idempotently
python3 - "$INSTALL_DIR" <<'EOF_PATCH'
import sys
from pathlib import Path

root = Path(sys.argv[1])

# 1. Patch pipeline.py to call write_bvh alongside write_glb
pipeline_py = root / "motion2mixamorig" / "pipeline.py"
text = pipeline_py.read_text(encoding="utf-8")
if "from .mixamo.bvh import write_bvh" not in text:
    text = text.replace(
        "from .mixamo.gltf import write_glb\n",
        "from .mixamo.bvh import write_bvh\nfrom .mixamo.gltf import write_glb\n",
    )
if 'bvh_path = run_dir / "mixamo_motion.bvh"' not in text:
    text = text.replace(
        '        glb_path = run_dir / "mixamo_character.glb"\n'
        '        write_glb(glb_path, asset, anim.r_anim_rig, anim.hips_cm_rig, anim.fps)\n'
        '        print(f"saved {glb_path}")\n',
        '        glb_path = run_dir / "mixamo_character.glb"\n'
        '        write_glb(glb_path, asset, anim.r_anim_rig, anim.hips_cm_rig, anim.fps)\n'
        '        print(f"saved {glb_path}")\n\n'
        '        bvh_path = run_dir / "mixamo_motion.bvh"\n'
        '        write_bvh(bvh_path, asset, anim.r_anim_rig, anim.hips_cm_rig, anim.fps)\n'
        '        print(f"saved {bvh_path}")\n',
    )
if '"mixamo_motion_bvh": bvh_path.name' not in text:
    text = text.replace(
        '"mixamo_character_glb": glb_path.name,\n',
        '"mixamo_character_glb": glb_path.name,\n            "mixamo_motion_bvh": bvh_path.name,\n',
    )
pipeline_py.write_text(text, encoding="utf-8")

# 2. Patch cli.py so `m2mr run --video <youtube_url>` also auto-downloads
cli_py = root / "motion2mixamorig" / "cli.py"
ctext = cli_py.read_text(encoding="utf-8")
if "from .youtube import download_youtube_video, is_url" not in ctext:
    ctext = ctext.replace(
        "    elif want_video:\n        video = Path(args.video)\n",
        "    elif want_video:\n"
        "        from .youtube import download_youtube_video, is_url\n\n"
        "        if is_url(args.video):\n"
        "            dl_dir = (\n"
        "                Path(args.output_dir).resolve()\n"
        '                if getattr(args, "output_dir", None)\n'
        "                else paths.VIDEO_DIR\n"
        "            )\n"
        "            video = download_youtube_video(args.video, dl_dir)\n"
        "        else:\n"
        "            video = Path(args.video)\n",
    )
    cli_py.write_text(ctext, encoding="utf-8")

# 3. Patch pyproject.toml to expose yt2bvh entrypoint
pyproject = root / "pyproject.toml"
ptext = pyproject.read_text(encoding="utf-8")
if 'yt2bvh = "motion2mixamorig.youtube:main"' not in ptext:
    ptext = ptext.replace(
        'm2mr = "motion2mixamorig.cli:main"',
        'm2mr = "motion2mixamorig.cli:main"\nyt2bvh = "motion2mixamorig.youtube:main"',
    )
    pyproject.write_text(ptext, encoding="utf-8")
EOF_PATCH

# ------------------------------------------------------------------------------
# 5. Create Python 3.12 virtualenv via `uv` and install packages
# ------------------------------------------------------------------------------
echo "[5/7] Creating Python 3.12 virtualenv with uv and installing packages..."
if [[ ! -x "$INSTALL_DIR/.venv/bin/python" ]]; then
    uv venv "$INSTALL_DIR/.venv" --python 3.12
fi

VENV_PY="$INSTALL_DIR/.venv/bin/python"
uv pip install --python "$VENV_PY" "numpy>=1.26" setuptools wheel pip
uv pip install --python "$VENV_PY" --no-build-isolation "chumpy==0.70"
uv pip install --python "$VENV_PY" -e "$INSTALL_DIR" "yt-dlp[default]" yt-dlp-ejs pysocks huggingface_hub

# ------------------------------------------------------------------------------
# 6. Download required body/hand models (SMPL-X, MANO), Mixamo Y_Bot.fbx & weights
# ------------------------------------------------------------------------------
echo "[6/7] Checking / downloading SMPL-X, MANO, Mixamo Y_Bot.fbx, and model weights..."
"$VENV_PY" - "$INSTALL_DIR" "$SKIP_WEIGHTS" <<'EOF_ASSETS'
import sys
import urllib.request
from pathlib import Path
from huggingface_hub import hf_hub_download

root = Path(sys.argv[1])
skip_weights = sys.argv[2] == "1"

# 1. SMPLX_NEUTRAL.npz
smplx_path = root / "assets" / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"
if not smplx_path.is_file():
    print("  -> Downloading SMPLX_NEUTRAL.npz...")
    smplx_path.parent.mkdir(parents=True, exist_ok=True)
    dl = hf_hub_download(repo_id="artabatest/smplx_modelv2", filename="SMPLX_NEUTRAL.npz")
    Path(dl).replace(smplx_path) if False else None
    import shutil
    shutil.copy2(dl, smplx_path)

# 2. MANO_RIGHT.pkl + MANO_RIGHT.npz
mano_pkl = root / "assets" / "body_models" / "mano" / "MANO_RIGHT.pkl"
if not mano_pkl.is_file():
    print("  -> Downloading MANO_RIGHT.pkl...")
    mano_pkl.parent.mkdir(parents=True, exist_ok=True)
    dl = hf_hub_download(repo_id="warmshao/WiLoR-mini", filename="pretrained_models/MANO_RIGHT.pkl")
    import shutil
    shutil.copy2(dl, mano_pkl)

from motion2mixamorig.extractors.hamer import ensure_mano_npz
ensure_mano_npz()

# 3. Mixamo Y_Bot.fbx
ybot_path = root / "assets" / "mixamo" / "Y_Bot.fbx"
if not ybot_path.is_file():
    print("  -> Downloading Mixamo Y_Bot.fbx...")
    ybot_path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://raw.githubusercontent.com/Glitshy/Godot-Lag-Compensation/main/Models/Y_Bot.fbx"
    urllib.request.urlretrieve(url, ybot_path)

# 4. Pre-download GVHMR + HaMeR weights unless --skip-weights was passed
if not skip_weights:
    print("  -> Ensuring GVHMR & HaMeR checkpoints are downloaded...")
    from motion2mixamorig import paths
    paths.export_gvhmr_env()
    from gvhmr.utils.assets import fetch, select
    from motion2mixamorig.extractors.hamer import ensure_hamer_weights
    fetch(select("all"))
    ensure_hamer_weights(verbose=True)
EOF_ASSETS

# ------------------------------------------------------------------------------
# 7. Create launcher script & global symlinks (`yt2bvh` and `m2mr`)
# ------------------------------------------------------------------------------
echo "[7/7] Installing global CLI commands (yt2bvh, m2mr)..."
cat <<EOF > "$INSTALL_DIR/yt2bvh.sh"
#!/usr/bin/env bash
set -euo pipefail
exec "$INSTALL_DIR/.venv/bin/python" -m motion2mixamorig.youtube "\$@"
EOF
chmod +x "$INSTALL_DIR/yt2bvh.sh"

mkdir -p "$HOME/.local/bin"
ln -sf "$INSTALL_DIR/yt2bvh.sh" "$HOME/.local/bin/yt2bvh"
ln -sf "$INSTALL_DIR/.venv/bin/m2mr" "$HOME/.local/bin/m2mr"

if command -v sudo >/dev/null 2>&1; then
    sudo ln -sf "$INSTALL_DIR/yt2bvh.sh" /usr/local/bin/yt2bvh || true
    sudo ln -sf "$INSTALL_DIR/.venv/bin/m2mr" /usr/local/bin/m2mr || true
fi

echo ""
echo "Running self-check (m2mr doctor --detect-hands)..."
"$INSTALL_DIR/.venv/bin/m2mr" doctor --detect-hands

echo ""
echo "============================================================"
echo " Installation Complete!"
echo " Run from anywhere:"
echo "   yt2bvh \"https://www.youtube.com/shorts/nXs5tALPGWA\" /path/to/output_dir"
echo "============================================================"
