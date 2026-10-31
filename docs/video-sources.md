# Video sources: URLs, local files, and saved motion

Motion2MixamoRig can download a video from a supported website or process a
video already on your computer. Despite its name, `yt2bvh` is not limited to
YouTube: it passes HTTP/HTTPS URLs to **yt-dlp**, then runs the motion pipeline.

This guide covers YouTube, TikTok, Douyin, Instagram, X/Twitter, Vimeo, direct
media links, and local files. It also explains still photos and reusing a
previously extracted skeleton.

Examples use placeholder IDs. Replace them with your own URLs and paths.
Shell examples assume Bash on Linux or macOS. Only download and process
content you own or have permission to use; these commands do not bypass
authentication, DRM, or a site's access restrictions.

## 1. Choose the right command

| Input | Recommended command | Notes |
|---|---|---|
| Public video URL | `yt2bvh "URL"` | Downloads, extracts motion, and exports BVH/GLB |
| Local video | `yt2bvh "/path/to/clip.mp4"` | Skips the website downloader |
| Video needing login or extra downloader options | Download with yt-dlp separately, then run `yt2bvh` on the local file | Cookie options are not exposed by our CLI |
| Still photo | `m2mr run --image "/path/to/pose.jpg"` | Produces a static pose, not motion from a video |
| Saved human skeleton | `m2mr run --skeleton "/path/to/clip_skeleton.npz" --rig "/path/to/rig.fbx"` | Reuses extraction to retarget another character |

`m2mr run --video "URL"` also uses the same downloader. Prefer `yt2bvh` for
website videos because it exposes `--proxy` and automatically selects CUDA
when available.

Important default differences:

| Setting | `yt2bvh` | `m2mr run` |
|---|---|---|
| Body extractor | GVHMR | GVHMR; configurable with `--set-extractor` |
| Device | CUDA if available, otherwise CPU | CPU; override with `--device` |
| Hand tracking | On; disable with `--no-hands` | Off; enable with `--detect-hands` |
| Foot locking | On | On |
| Preview | On; one four-panel MP4 for video input | On; same video preview |
| Preferred character | `assets/mixamo/X_Bot.fbx` | X Bot, with an existing fallback to another local FBX if it is missing |

On Apple Silicon, choose `--device mps` explicitly; `yt2bvh` does not
automatically select MPS.

## 2. Prepare the runtime and models

Follow the [repository setup instructions](../README.md#prerequisites).
The one-click command-line installer is:

```bash
bash install_yt2bvh.sh
```

It prepares the Python environment, CLI launchers, dependencies, and inference
weights. It checks/installs system tools on supported Linux/macOS setups,
including FFmpeg, Node.js, and Tor. It does **not** install Blender.
Generating BVH and GLB files does not require Blender.

The default X Bot FBX is not provided by the public repository. Obtain it
under your own Adobe/Mixamo account and save the **FBX Binary, T-pose**
character as:

```text
assets/mixamo/X_Bot.fbx
```

You can instead supply another compatible character with `--rig`.
You also need the SMPL-X body model. Hand tracking additionally needs the
MANO model and HaMeR dependencies; use `--no-hands` if you do not want that
pass. Check the environment before a long run:

```bash
m2mr doctor
m2mr doctor --detect-hands
```

If you set up your own virtual environment rather than using the installer,
install the downloader dependencies in that **same environment**, after the
main project setup:

```bash
python -m pip install "yt-dlp[default]" yt-dlp-ejs pysocks
```

FFmpeg is needed to merge separate audio/video streams and produce
browser-friendly previews. Node.js is used by the downloader's JavaScript
runtime, particularly for YouTube.

## 3. Tor and proxy selection

Website downloads use local Tor at `socks5h://127.0.0.1:9050` by default.
The `socks5h` scheme sends hostname resolution through the proxy.

Selection order is:

1. An explicit `yt2bvh --proxy` value.
2. The `YT_DLP_PROXY` environment variable, if set.
3. The local Tor proxy.

The downloader checks whether port 9050 is listening and, on systems with
`systemctl`, attempts to start the Tor service. **It does not silently retry
directly** when Tor is unavailable or a download fails.

Start Tor yourself if necessary:

```bash
# Linux with a Tor system service
sudo systemctl start tor

# macOS with Homebrew
brew services start tor
```

Use a different proxy:

```bash
yt2bvh "URL" --proxy "socks5h://127.0.0.1:9150"
yt2bvh "URL" --proxy "http://127.0.0.1:8080"
```

Port 9150 is only an example: use it if you actually have a SOCKS proxy
listening there.

Explicitly choose a direct connection:

```bash
yt2bvh "URL" --proxy ""
```

For `m2mr run`, which has no `--proxy` argument, use the environment variable:

```bash
YT_DLP_PROXY="socks5h://127.0.0.1:9050" m2mr run --video "URL"
YT_DLP_PROXY="" m2mr run --video "URL"
```

Tor can be slower and some websites block its exit addresses. It does not
log you into a website or grant permission to view restricted content.
This setting controls **video downloads**, not every network request made
by the application; inference weights have their own download mechanisms.
Local video inputs do not invoke the website downloader.

## 4. Source-by-source examples

The installed yt-dlp includes extractors for these services. That is not a
guarantee that every URL works: websites change, and login requirements,
regional availability, rate limits, or proxy blocking can affect a download.

Always quote URLs, especially those containing `&` or other shell characters.
Use an individual video/post URL, not a profile, hashtag, live stream, or
playlist. Our downloader uses `--no-playlist`; batch crawling is not a
feature of this command.

### YouTube: regular videos and Shorts

```bash
yt2bvh "https://www.youtube.com/watch?v=VIDEO_ID"
yt2bvh "https://www.youtube.com/shorts/VIDEO_ID"
yt2bvh "https://youtu.be/VIDEO_ID" -o outputs/youtube_clip
```

Choose the actual video page, not a channel page. For a long video, download
and trim a suitable segment first rather than inferring motion over the
entire video. Age-restricted, private, or sign-in-required videos may need
the separate authenticated-download workflow below.

### TikTok

```bash
yt2bvh "https://www.tiktok.com/@USERNAME/video/VIDEO_ID"
```

Prefer the full video URL copied from the video's page. If a short share
link fails, open it in your browser and copy the resolved video URL.
TikTok may require a browser session or reject a Tor address. In that case,
download an authorized copy separately and pass the resulting file locally.

### Douyin

```bash
yt2bvh "https://www.douyin.com/video/VIDEO_ID"
```

Douyin has a separate yt-dlp extractor; it is not simply a TikTok URL alias.
Prefer the full video URL over copied share text. If the share message
includes prose and a URL, pass only the URL. Browser/session requirements
and regional access can prevent downloading.

### Instagram: Reels and video posts

```bash
yt2bvh "https://www.instagram.com/reel/SHORTCODE/"
yt2bvh "https://www.instagram.com/p/SHORTCODE/"
```

The post must actually contain a video. A photo post is not a video input.
For carousels, use a separately downloaded video file if the page yields
multiple items; the wrapper is designed for one source clip.
Private posts, Stories, and even some public posts may require login.
Do not assume profile or hashtag downloads will work.

### X / Twitter

```bash
yt2bvh "https://x.com/USERNAME/status/POST_ID"
yt2bvh "https://twitter.com/USERNAME/status/POST_ID"
```

Use a post containing an embedded video. A text-only post is not an input.
Multi-video posts can yield more than one downloaded item; use the desired
local file explicitly instead of relying on which item the wrapper selects.

### Vimeo

```bash
yt2bvh "https://vimeo.com/VIDEO_ID"
```

Public, accessible video pages are the simplest case. Password-protected or
account-restricted videos need options not exposed by our wrapper, so use
the standalone downloader or the service's own authorized download feature.

### Direct media URLs and other websites

```bash
yt2bvh "https://example.com/media/dance.mp4"
```

Other websites can work through a dedicated yt-dlp extractor or its generic
media extraction. A webpage URL is not necessarily a downloadable video URL.
Signed links may expire or require session headers; do not publish links
containing private access tokens.

The downloader prefers H.264 video, M4A audio, and MP4 output, but format
selection has fallbacks. Merging into MP4 is **not** the same as transcoding
the video codec. If the resulting file cannot be decoded, convert it as
described under local files.

## 5. Local videos and predictable filenames

Local files avoid website/proxy problems and give you control over the output
prefix:

```bash
yt2bvh assets/video/dance.mp4
yt2bvh "/home/user/Videos/dance clip.mp4" -o outputs/dance_test
yt2bvh assets/video/dance.mp4 --rig assets/mixamo/Y_Bot.fbx
```

Alternatively:

```bash
m2mr run --video assets/video/dance.mp4 --device cuda
```

MP4 is recommended. The project also recognizes MOV, AVI, MKV, and WebM as
video inputs, but successful decoding depends on the file's codec and the
runtime. If necessary, make a compatible MP4:

```bash
ffmpeg -i input.mov \
  -c:v libx264 -pix_fmt yuv420p \
  -c:a aac -movflags +faststart \
  assets/video/dance.mp4
```

This command does not force a new frame rate. Avoid changing FPS metadata
without resampling the frames, which can change playback duration.

To use only a segment, trim to a **different** file:

```bash
ffmpeg -ss 00:00:10 -i input.mp4 -t 8 \
  -c:v libx264 -pix_fmt yuv420p -c:a aac \
  assets/video/dance_segment.mp4
yt2bvh assets/video/dance_segment.mp4
```

Frame 0 in the outputs now refers to the first frame of the trimmed clip,
not frame 0 of the original full-length video.

## 6. Videos requiring login or extra downloader options

Our CLI does not expose `--cookies`, `--cookies-from-browser`, passwords,
custom HTTP headers, or arbitrary yt-dlp arguments. Passing those options
to `yt2bvh` will not work.

For content your own account is authorized to access, use the downloader
separately. This example uses a locally installed Firefox browser session:

```bash
yt-dlp \
  --proxy "socks5h://127.0.0.1:9050" \
  --cookies-from-browser firefox \
  --no-playlist \
  -S "vcodec:h264,res,acodec:m4a" \
  --merge-output-format mp4 \
  -o "assets/video/authorized_clip.%(ext)s" \
  "URL"
```

Then pass the actual downloaded file to the motion pipeline:

```bash
yt2bvh assets/video/authorized_clip.mp4
```

Use the filename printed by yt-dlp if the output extension differs.
The standalone yt-dlp command does **not** inherit our Tor default or
`YT_DLP_PROXY` handling; specify `--proxy` there yourself, including
`--proxy ""` if you intentionally want direct access.

Keep browser profiles and cookie files private. Do not commit, upload, or
share cookies; they can contain credentials for your account. Browser-cookie
extraction is dependent on the browser and operating system. See the
[yt-dlp FAQ](https://github.com/yt-dlp/yt-dlp/wiki/FAQ) for its supported
authentication workflows.

## 7. Photos and saved skeletons

Use `m2mr`, not the website-video wrapper, for a still photo:

```bash
m2mr run --image assets/image/pose.jpg --rig assets/mixamo/X_Bot.fbx
```

The photo should show one person from head to toe. The result is a static
pose. Photo previews remain PNG views plus an orbit-comparison MP4, all
source-prefixed and in the run directory.

To change the character without extracting the human motion again:

```bash
m2mr run \
  --skeleton outputs/previous_run/dance_skeleton.npz \
  --rig assets/mixamo/Y_Bot.fbx \
  --output-dir outputs/dance_ybot
```

`--skeleton` requires the human skeleton artifact, not the rotations NPZ,
BVH, or character GLB. Old `skeleton_motion.npz` files are still supported.
The source recorded in that skeleton must still be available for preflight
and any requested preview.

## 8. Outputs and useful options

For an input named `dance.mp4`, outputs look like:

```text
outputs/<timestamp>_dance/
  dance.bvh
  dance_character.glb
  dance_rotations.npz
  dance_skeleton.npz
  dance_run.json
  dance_diagnostics.json
  dance_compare.mp4
  cache/                         # extraction intermediates, when generated
```

Website downloads are normally named by the platform's video ID, so that ID
becomes the output prefix. `-o` changes the **run directory**, not the file
prefix. Download separately to a chosen local filename if you want a more
descriptive prefix.

All result files are flat; there is no separate `videos/` directory and no
duplicate BVH. The comparison MP4 includes the original, Mixamo skeleton,
human skeleton, and skinned character in a four-panel layout. It retains
source audio when FFmpeg is available.

For `yt2bvh`, a URL download is saved inside the run directory. With automatic
output naming it is initially downloaded to `assets/video/` and then copied
into that run. An existing local video is used in place rather than copied.
`m2mr run --video "URL"` without `--output-dir` keeps its downloaded source
in `assets/video/`.

Useful options:

```bash
# Skip rendered previews, but retain BVH, GLB, NPZs, and reports
yt2bvh "URL" --no-preview

# Skip the additional HaMeR finger pass
yt2bvh "URL" --no-hands

# Choose the output directory and character
yt2bvh "URL" -o outputs/my_run --rig assets/mixamo/X_Bot.fbx

# Select the device explicitly
yt2bvh "URL" --device cuda
yt2bvh "URL" --device cpu
yt2bvh "URL" --device mps
```

Choose a new output directory for a revision; reusing an explicit directory
can overwrite files for the same source prefix. Legacy fixed-name runs remain
readable, but do not combine several runs into one folder.

`dance_diagnostics.json` reports character self-intersections, not whether
the downloaded source is good or the motion looks natural. A failed quality
check does not fail generation. Re-check or build the interactive report:

```bash
m2mr diagnose outputs/previous_run
m2mr diagnose outputs/previous_run --visualize
```

The latter writes `<prefix>_diagnostics.html`. Its timeline can also collect
human review notes about problems such as sliding feet.

## 9. Troubleshooting

| Symptom | What to check |
|---|---|
| `Tor proxy is unavailable` | Start Tor on port 9050, provide a working proxy, or explicitly choose direct access |
| Download blocked, HTTP 403/429, or a sign-in message | Check that the video is accessible to you; avoid repeated retries; try an authorized local download |
| `Unsupported URL` | Use the actual individual video URL; check the installed yt-dlp extractor support |
| `yt-dlp not found` | Install it in the external Python environment used by the CLI |
| YouTube JavaScript-runtime errors | Check that Node.js and yt-dlp's JavaScript dependencies are installed |
| Audio/video merge fails | Check that FFmpeg is installed and on `PATH` |
| CLI rejects cookie options | Use standalone yt-dlp, then process the local file |
| Download succeeds but preflight rejects the clip | Check decoding and the single-person requirement; this is a source/extraction issue, not necessarily a downloader issue |
| X Bot file is missing | Supply `assets/mixamo/X_Bot.fbx`, or pass another authorized local FBX with `--rig` |
| MANO/HaMeR requirements are missing | Use `--no-hands` or complete the hand-model setup |
| Motion is slow or CUDA is unavailable | Verify the chosen device; CPU inference is much slower |
| Animation playback duration is wrong | Match the clip FPS; do not merely relabel the source FPS |

The source should contain one clearly visible person, with the full body
in frame. Prefer a stable camera, limited occlusion, and one continuous shot.
Group shots are rejected in preflight; fast edits, hidden limbs, and
low-resolution clips can still produce poor motion even after downloading
successfully.

If a site's support has recently broken, update the downloader in the
environment you actually use:

```bash
python -m pip install --upgrade "yt-dlp[default]" yt-dlp-ejs
yt-dlp --list-extractors
```

For a one-click installation, use its `.venv/bin/python` and
`.venv/bin/yt-dlp` rather than accidentally updating a different system copy.
Website extractor support is maintained upstream, not independently by
Motion2MixamoRig.

## References

- [yt-dlp supported sites](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md):
  inclusion does not guarantee that every URL currently works.
- [yt-dlp usage and options](https://github.com/yt-dlp/yt-dlp#usage-and-options).
- [yt-dlp FAQ](https://github.com/yt-dlp/yt-dlp/wiki/FAQ).
- [Motion2MixamoRig setup and outputs](../README.md).
- [Blender user guide](user-guide.md).
