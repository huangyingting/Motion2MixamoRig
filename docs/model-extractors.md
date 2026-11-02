# Optional model extractors and comparison

The model inventory and optional runtime isolation were informed by
[RGB MoCap](https://github.com/toyxyz/toyxyz_RGBMocap). These connectors call
upstream public APIs; they do not vendor RGB MoCap or model implementation code,
weights, or registered geometry.

| Model | Motion2MixamoRig integration | Additional requirements |
| --- | --- | --- |
| GVHMR | Existing default video/still extractor; optional HaMeR fingers | Standard SMPL-X; MANO for fingers |
| ExMR | Optional video extractor, native body/foot/finger estimates | CUDA; SAM 3D Body access; upstream ExMR/Sapiens2/HaMeR files |
| FlowHMR | Optional video extractor, native SMPL-H body/fingers | Upstream runtime; SAM 3D Body access; Extended SMPL-H neutral with 16 shape components |
| PromptHMR | Optional single-actor export from its video pipeline; no fingers | CUDA; upstream runtime, registered SMPL and SMPL-X, upstream checkpoints |
| MAMMA | **Not implemented**: a different, calibrated multi-camera capture workflow | Synchronized views, calibration, restricted assets/native builds |
| PEAR | **Not implemented**: RGB MoCap uses it for experimental live preview | SMPL-X 2020 and FLAME 2020; not interchangeable with standard SMPL-X |

Group videos support **explicit selection of one person per run** with GVHMR,
ExMR and FlowHMR. The default single-person preflight remains in place.
PromptHMR additionally requires one continuous actor; exporting everyone in a
shared multi-person scene is not implemented. MAMMA and PEAR are not fake choices
in the extractor menu. Supporting them requires new input/live workflows, not
just an adapter to this video-to-continuous-skeleton interface.

HandFlow, HTD-Refine, scene-camera estimators, and RGB MoCap contact/IK tools
are separate add-ons, not implemented by these body-extractor connectors.
Motion2MixamoRig keeps its existing HaMeR, foot-lock, ground and diagnostics
implementations.

When a completed same-source GVHMR run has a verified full-video tracking
cache, FlowHMR reuses its person boxes, resampling only box timing and pixel
coordinates for the 30 FPS input. This prevents fragmented ByteTrack IDs from
silently shortening a dance clip. Source SHA-256, frame coverage and aspect
ratio are checked; changed boxes invalidate dependent SAM/camera caches.
Body and finger motion are still estimated by FlowHMR, not copied from GVHMR.
Tracking provenance is saved beside the worker's bbox cache and in the skeleton
NPZ. Without a matching saved cache, FlowHMR uses its own tracker, and partial
motion exports remain rejected.

## Set up an upstream runtime

The standard-library-only `model_catalog.json` is the shared model inventory for
CLI adapters, setup, and Blender. Prepare a pinned checkout using the current
project venv:

```bash
m2mr setup-model exmr --install-deps
m2mr setup-model flowhmr --install-deps
```

Setup reuses an existing runtime Python, or links the checkout's `.venv` to the
current venv. It refuses to change an existing checkout's revision. Dependency
installation is opt-in: only missing distributions are added, with `--no-deps`,
so upstream torch/NumPy pins cannot replace the working stack. A setup receipt
lists asset checks; it is **not proof of inference compatibility**. Native
libraries and registered/gated weights still need provider-authorized setup.
For ExMR on Ubuntu, the tested JPEG binding uses `libturbojpeg`; HaMeR imports
also need `libglu1-mesa`. PromptHMR still requires its additional upstream
compiled dependencies; its minimal setup profile is not a complete installer.

Follow each model's upstream guide. Separate environments are recommended when
dependency versions conflict, but are **not required**:

- [ExMR installation](https://github.com/twehrbein/ExMR/blob/main/docs/INSTALL.md)
  and [output schema](https://github.com/twehrbein/ExMR/blob/main/docs/OUTPUTS.md).
  The bridge targets the public `Options` / `run_video` API from ExMR v0.1.5.
- [FlowHMR installation](https://github.com/flowhmr/flowhmr/blob/main/docs/install.md)
  and [inference](https://github.com/flowhmr/flowhmr/blob/main/docs/inference.md).
  The bridge uses `V2MInferenceEngine.build` / `process_video`, the recommended
  `flowhmr_latest` checkpoint, and raw SMPL-H exports.
- [PromptHMR installation](https://github.com/yufu-wang/PromptHMR#installation).
  The bridge uses `Pipeline(static_cam=True)` and the released video-head
  checkpoint `prhmr_release_002`.

By default each source checkout is expected under
`weights/runtimes/<extractor>/`, with Python at `.venv/bin/python` (Windows:
`.venv/Scripts/python.exe`). Override both with absolute paths, for example:

```bash
export M2MR_EXMR_ROOT=/path/to/ExMR
export M2MR_EXMR_PYTHON=/path/to/exmr-venv/bin/python
m2mr doctor --set-extractor exmr
m2mr run --video assets/video/GDeEngdz9ec.mp4 --set-extractor exmr --device cuda
m2mr run --video assets/video/GDeEngdz9ec.mp4 --set-extractor exmr --device cuda --detect-hands
```

The same `M2MR_FLOWHMR_ROOT/PYTHON` and `M2MR_PROMPTHMR_ROOT/PYTHON` variables
configure the other models. These are trusted local source/runtime paths.
Install the upstream dependencies and exact assets before running the checks.
Blender still launches the external Motion2MixamoRig Python.

To reuse the project's existing venv, set the Python overrides to its absolute
path instead of installing upstream pinned requirements wholesale:

```bash
export M2MR_EXMR_PYTHON="$PWD/.venv/bin/python"
export M2MR_FLOWHMR_PYTHON="$PWD/.venv/bin/python"
export M2MR_PROMPTHMR_PYTHON="$PWD/.venv/bin/python"
```

Exports apply only to that shell and its child processes. For persistent local
setup on Linux/macOS, each checkout's `.venv` can instead be a symlink to the
project venv; the default runtime lookup then works without exports, including
from Blender. Do not replace an existing checkout venv. Reusing Python and
downloading weights do not establish compatibility: check real upstream
imports and licensed assets before attempting inference. Add missing packages
selectively and preserve the working torch, torchvision and NumPy versions.

Keep assets in the upstream layouts reported by `doctor`. ExMR uses
`<root>/checkpoints/`; FlowHMR uses its `checkpoints/`, `ckpts/` and
`assets/body_models/`; PromptHMR uses `<root>/data/`. Existing files can be reused
only when their versions and provider terms match. No automatic registered
model downloads or account login are attempted.

These video connectors currently use **static-camera** inference, matching
Motion2MixamoRig's existing GVHMR setting. ExMR and FlowHMR use 30 fps working
video; PromptHMR retains its pipeline's processed frame rate. The skeleton FPS
is reconciled with source duration. Parameters are finite-checked, then
reconstructed with a flat-hand SMPL-X/SMPL-H model and the same SMPL-24 joint
regressor used for GVHMR. SMPL-X vertices use the existing SMPL-X-to-SMPL map.
ExMR's forward-normalized world and FlowHMR's Y-180 export are rotated into the
shared world using the catalog's `world_to_shared` matrix. The same fixed,
proper rotation transforms joints, root orientation and translation (including
the shaped SMPL pelvis offset); parent-relative fingers and turns are unchanged.
Older native-model runs lack this normalization and must be regenerated from
cached parameters for a correctly oriented shared-camera comparison.
Finger exports use the existing MANO-to-Mixamo retargeter, requiring this
project's MANO template even when the upstream model already predicts fingers.

The new optional workers need end-to-end validation with their licensed assets
before their results can be treated as production-ready. Unit tests of bridge
calls and conversion are not a substitute for real inference.

## Compare a downloaded clip

### Select a dancer in a group video

```bash
m2mr track-people --video assets/video/group.mp4 --device cuda
# Open the *_people.jpg preview, then choose an ID from *_people.json:
m2mr run --video assets/video/group.mp4 --set-extractor flowhmr \
  --person-track outputs/group/tracks/<tracking-run>/group_person_1.npz \
  --device cuda --detect-hands
m2mr benchmark --video assets/video/group.mp4 \
  --person-track outputs/group/tracks/<tracking-run>/group_person_1.npz
```

The track stores source SHA-256, dimensions, timing, person ID and full-frame
bounding boxes, not motion. All models estimate that selected dancer from the
original frames without moving crops. Track fingerprints isolate caches and
prevent skeleton reuse or comparisons from mixing people. The track is copied
into each delivery and listed in its run report.

By default every frame must have a detection of the same ID; identity jumps,
fragmented IDs and missing endpoints are rejected. `track-people
--max-gap-frames N` explicitly allows bounded internal detection gaps with
overlapping endpoints: only 2D boxes are interpolated, the detected-frame mask
is retained, and motion is never padded or stretched. It does not join IDs.
Review the selected person visually: tracker IDs are not an identity guarantee
through occlusion. Remove blank intro/outro frames into a separately named
input if no person is visible there; retain the original and document the trim.
Blender's panel still uses the default single-person workflow.

### Generate the comparison

```bash
m2mr benchmark --video assets/video/GDeEngdz9ec.mp4 \
  --rig assets/mixamo/X_Bot.fbx --device cuda --include-hands
```

The default benchmark runs **GVHMR + HaMeR, ExMR with native fingers, and FlowHMR
with native fingers**. Each writes an independent directory with source-prefixed
skeleton, rotations, GLB, BVH and diagnostics. There are no single-model preview
MP4s in the model directories, including when older runs are reused or consolidated.
Only `<video>_compare.mp4` is rendered, with Original Video / GVHMR + HaMeR on the top row and
ExMR / FlowHMR on the bottom. All characters share the same camera, scale and
floor. Initial pelvis heading is aligned by a constant display-only yaw;
orthographic framing uses the rig's rest mesh, not each model's motion bounds.
Horizontal root-following keeps the body framed, with the world floor grid
retaining movement cues. Real pose differences remain visible. Saved GLB,
BVH and NPZ animations are never rotated or rescaled for these previews.
Rendering-version changes invalidate only previews, not completed inference.
Complete clips with minor frame-count differences are endpoint-aligned
to the source timeline. Original audio is retained without truncating the last
video frame. If a required model is unavailable, no substitute is mislabeled as
that model; the report records why the comparison could not be produced.

With an explicit `--extractors` list, body-only variants remain available.
`--include-hands` adds supported finger variants; GVHMR's finger pass then reuses
the body-only skeleton and wrist cache. The comparison
writes `<video>_benchmark.json`, recording source/rig hashes, availability,
failures, timings, frame rates, intersection counts/fractions/depths and raw
root-relative joint jerk. Missing models are recorded as **unavailable**, not
replaced with GVHMR outputs. MAMMA/PEAR are listed as not applicable to this
benchmark; they are not executed. Existing runs are never overwritten.

Use `--extractors gvhmr exmr` to limit models, `--no-preview` to omit videos,
or `--output-dir <new-empty-folder>` to pick a destination.

To compare completed runs without rerunning extraction:

```bash
m2mr benchmark --video assets/video/dance.mp4 --rig assets/mixamo/X_Bot.fbx \
  --reuse-run gvhmr_hamer=outputs/<gvhmr-hamer-run> \
  --reuse-run exmr=outputs/<exmr-native-run> \
  --reuse-run flowhmr=outputs/<flowhmr-native-run>
```

Reused runs must be completed, use the same video and rig, and contain the
correct finger model. Native panels reject HaMeR-swapped hybrids. Original
run files are not modified; comparison provenance and quality-report paths
are recorded in the new benchmark JSON.

Defaults write `outputs/<video-stem>/`: shared
`<video-stem>_source.json` is at the source root, the four-panel MP4 and benchmark
JSON are at the source root, and model deliveries are in `gvhmr/`, `exmr/`,
and `flowhmr/`. Complete matching outputs and comparisons are reused; only
missing models run. Use `--force` to regenerate. Each body model has one current
finger configuration, not multiple version folders. The old `gvhmr_hands` reuse
label remains an alias for `gvhmr_hamer`; both target `gvhmr/`.
`--output-dir` overrides the comparison location; its metadata is shared there.

BVH exports use the 52 VRM 1.0 humanoid names documented in
[pose-studio's skeleton contract](https://github.com/huangyingting/pose-studio/issues/30#issuecomment-6555778217):
22 body joints and 30 finger joints. Mixamo terminal markers become End Sites;
unrecognized nonterminal helpers are retained. Reusing an older delivery
updates its BVH names in the new directory, leaving the original
untouched. Motion, metre/Y-up coordinates and ZXY Euler channels are preserved.
The rotations NPZ still uses quaternion **wxyz**, not the JSON clip's xyzw.

## Adding another model

Add its capabilities, source revision, packages, assets and native body format
to `motion2mixamorig/model_catalog.json`. Implement its official API bridge in
`extractors/_external_worker.py` and point the catalog's `worker` at that
function. The worker writes finite, flat-hand SMPL-X/SMPL-H parameters; other
body formats need their own adapter implementing `extractors/base.py`.
CLI choices, setup and Blender choices read the same catalog. Keep heavyweight
imports in the adapter/worker, never in Blender or the catalog. Add focused
conversion/bridge tests and perform licensed end-to-end inference before
claiming support. The default comparison's three panel roles are deliberately
explicit; adding a model does not silently replace an existing panel.

Without motion-capture ground truth, no objective "best model" is reported.
Intersection and jerk measurements are proxies: less jerk may mean
over-smoothing, and less intersection does not establish fidelity to the video.
The native models also use different upstream postprocessing. Compare the
previews visually, especially feet, knees, hands, timing and trajectory.

## Licensing

Review each provider agreement before installing or using the models.
ExMR/GVHMR-derived code and PromptHMR/MAMMA have noncommercial/research
restrictions. FlowHMR source uses Apache-2.0, PEAR source Apache-2.0, and
HaMeR/HandFlow source MIT; that does not relicense their weights, body data
or supporting components. SAM 3D Body, Sapiens2, SMPL-X, SMPL-H, MANO, FLAME
and other dependencies retain separate terms. External workers do not waive
license obligations. See the upstream licenses and
[RGB MoCap's third-party notices](https://github.com/toyxyz/toyxyz_RGBMocap/blob/main/THIRD_PARTY_NOTICES.md).
