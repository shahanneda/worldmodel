# WebGPU Inference

This folder is a separate browser-side inference path for the current coordinate-to-image checkpoints.

It does not touch the existing Flask server stack under `inference/`.

## What it does

- serves from a static `index.html`
- loads an exported ONNX model plus external weight data
- runs inference in the browser with `WebGPU` when available
- falls back to `wasm` if `WebGPU` is unavailable

The browser export path works for both:

- `coord_to_image_unet`
- `pointing_cvae`

For `pointing_cvae`, the browser graph uses deterministic prior-mean decoding.

## Current default deployment

The static frontend currently points at this exported checkpoint:

- `model/checkpoints/ckpt000045_finger_xy_cvae_march_BIG_v1_pointing_cvae_best_epoch0195_2026-03-16T02-29-47Z.pt`

As of March 16, 2026, this is the latest checkpoint available locally and in S3. There is not an `epoch0200` checkpoint present yet.

Current default manifest URL:

```text
https://zimpmodels.s3.us-east-2.amazonaws.com/worldmodel/webgpu-inference/browser-model/ckpt000045_finger_xy_cvae_march_BIG_v1_pointing_cvae_best_epoch0195_2026-03-16T02-29-47Z/manifest.json
```

Verified export details for that checkpoint:

- `fp16` total browser model size: `247,914,419` bytes
- ONNX Runtime parity check:
  - `max_abs_diff=0.001909256`
  - `mean_abs_diff=0.000010511`

The frontend also includes a built-in checkpoint picker with:

- `Pointing CVAE · March BIG v1 · best epoch 195`
- `Pointing CVAE · March BIG v1 · epoch 150`
- `Baseline U-Net · March 13 export`
- a persisted stage layout toggle that can overlay the generated frame directly under the input grid

## Browser caching

The static app now keeps a small best-effort local cache of downloaded ONNX assets.

- cache keying is based on the asset URLs, so unique checkpoint filenames produce separate cache entries
- only a small number of checkpoints are kept locally
- current cap: `2` checkpoints
- if a checkpoint URL changes, the browser treats it as a new asset and fetches it again

This cache is only for model graph/data downloads. It does not cache generated images.

## Export the model

Use a Python environment with:

- `torch`
- `onnx`
- `onnxruntime`
- `onnxscript`
- `onnxconverter-common`
- `numpy`
- `pillow`

## Environment notes

The checkpoint was trained in a different environment from the one used for browser export.

- Remote training environment:
  - Linux server
  - NVIDIA GPU / CUDA used for training
  - used to produce the original `.pt` checkpoint
- Local browser-export environment:
  - macOS on the laptop
  - CPU-only export and verification
  - temporary venv used here: `/tmp/worldmodel-webgpu-venv`
  - AWS CLI used locally to upload the exported browser assets to S3

The browser export does not require CUDA. As long as the checkpoint file is already available locally, the Mac can export and upload it. A local conda env with the same packages is also fine if you do not want to use a temporary venv.

From the repo root:

```bash
/tmp/worldmodel-webgpu-venv/bin/python scripts/export_webgpu_model.py \
  --checkpoint model/checkpoints/ckpt000045_finger_xy_cvae_march_BIG_v1_pointing_cvae_best_epoch0195_2026-03-16T02-29-47Z.pt \
  --output-dir webgpu-inference/model \
  --precision fp16
```

That writes:

- `webgpu-inference/model/manifest.json`
- `webgpu-inference/model/*.onnx`
- `webgpu-inference/model/*.onnx.data`

The app loads the selected checkpoint manifest by default, and still supports a custom manifest URL when you want to point at another host.

## Run locally

Serve the folder over HTTP:

```bash
python3 -m http.server 5173 --directory webgpu-inference
```

Then open:

```text
http://127.0.0.1:5173
```

Do not open the page with `file://`.

## Hosting notes

The export works, but the current deployed checkpoint is still large:

- current `fp16` export: about `248 MB`

An earlier deterministic baseline export was about `400 MB` in `fp16`, so export size depends heavily on the architecture and channel counts.

So the static frontend is ready, but GitHub Pages is only realistic if you:

- host the ONNX weight files on a different static host and point the manifest there, or
- train/export a substantially smaller browser model

The page already supports a manifest URL that can point at another host.

## S3 workflow

There are now two helper scripts for S3:

- `scripts/s3_configure_webgpu_public_access.py`
  - applies a public-read bucket policy scoped to the model prefix
  - applies bucket CORS rules for browser `GET` and `HEAD`
- `scripts/s3_upload_webgpu_assets.py`
  - uploads the ONNX files
  - uploads a rewritten manifest with public S3 URLs
  - writes `webgpu-inference/runtime-config.js` so the page loads that S3 manifest by default

Example:

```bash
export S3_BUCKET=your-bucket
export S3_PREFIX=worldmodel
export WEBGPU_S3_PREFIX='worldmodel/webgpu-inference/browser-model/your-checkpoint-name'
export WEBGPU_CORS_ALLOWED_ORIGINS='https://yourname.github.io,http://127.0.0.1:5174,http://localhost:5174'
python scripts/s3_configure_webgpu_public_access.py
python scripts/s3_upload_webgpu_assets.py
```

The upload script prints the public manifest URL after upload.
