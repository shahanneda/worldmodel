# WebGPU Inference

## Purpose

This repo now has a second inference path besides the existing Flask server app.

The original inference stack is:

1. browser sends `(x, y)` to Python
2. Python loads the PyTorch checkpoint
3. Python runs inference
4. Python sends the generated image back

The new browser-side path is:

1. export the PyTorch checkpoint to ONNX once
2. host a static HTML app
3. load the ONNX graph plus weight data in the browser
4. run inference locally with `onnxruntime-web`
5. prefer `WebGPU`, fall back to `wasm`

This new path lives in its own folder and does not modify the existing server-side inference code.

The export path now works for both:

- `coord_to_image_unet`
- `pointing_cvae`

For `pointing_cvae`, the exported browser graph uses deterministic prior-mean decoding. The browser input still stays in normalized `[0, 1]` space, and the export wrapper applies any required coordinate remapping internally.

## What was added

### New files

Main pieces added for the browser path:

- `scripts/export_webgpu_model.py`
  - exports the trained checkpoint to ONNX
  - optionally converts to `fp16`
  - writes a browser manifest
  - verifies ONNX Runtime output against PyTorch
- `scripts/s3_upload_webgpu_assets.py`
  - uploads browser model artifacts to S3
  - rewrites the manifest to public S3 URLs
  - writes `webgpu-inference/runtime-config.js`
- `scripts/s3_configure_webgpu_public_access.py`
  - sets a prefix-scoped public-read bucket policy
  - sets bucket CORS for browser fetches
- `webgpu-inference/index.html`
  - static entrypoint intended for GitHub Pages or another static host
- `webgpu-inference/app.js`
  - loads the manifest
  - downloads the ONNX graph and external weight file(s)
  - creates an ONNX Runtime Web session
  - runs inference from clicked `(x, y)` coordinates
- `webgpu-inference/styles.css`
  - standalone UI styling
- `webgpu-inference/model/`
  - target folder for generated ONNX artifacts
- `webgpu-inference/runtime-config.js`
  - frontend default manifest URL
  - can point at local artifacts or S3-hosted artifacts

### Runtime choice

The browser app uses `onnxruntime-web` instead of hand-writing custom WebGPU shaders.

That choice was deliberate:

- the model already exists in PyTorch
- ONNX export is much faster to iterate on than reimplementing the model by hand
- ONNX Runtime Web already supports `WebGPU` and `wasm`
- the same exported model can be validated against PyTorch output

## Current model/export reality

The current browser path works with the newer latent checkpoint too.

Most recently verified export:

- checkpoint:
  - `model/checkpoints/ckpt000045_finger_xy_cvae_march_BIG_v1_pointing_cvae_best_epoch0195_2026-03-16T02-29-47Z.pt`
- model kind:
  - `pointing_cvae`
- image size:
  - `128`
- coord space:
  - `minus_one_to_one`
- precision:
  - `fp16`
- total browser model size:
  - `247,914,419` bytes
- ONNX Runtime parity check:
  - `max_abs_diff=0.001909256`
  - `mean_abs_diff=0.000010511`

As of March 16, 2026, this `best_epoch0195` checkpoint is the latest one present both locally and in the S3 checkpoint bucket. There is no `epoch0200` checkpoint available yet.

Earlier exports from the deterministic baseline were significantly larger, so hosting constraints still matter, but the newer checkpoint is materially smaller than the first browser export.

## Generated artifact format

The export script writes:

- `manifest.json`
- `*.onnx`
- `*.onnx.data`

### Why there are two model files

The `.onnx` file is small because it mostly stores the graph structure.

The heavy weights are stored in the external data file:

- `.onnx.data`

This is expected and is the format the browser app loads.

### Manifest role

The manifest tells the browser app:

- which ONNX file to load
- which external data file(s) to mount
- input and output names
- image size
- preferred execution providers
- model size metadata

The app defaults to:

```text
./model/manifest.json
```

unless `webgpu-inference/runtime-config.js` points elsewhere.

The manifest URL can also be changed in the UI or via query param.

The frontend now also supports:

- a named checkpoint picker for switching between multiple known manifests
- a stage layout toggle that can merge the input grid and generated frame into one stacked stage
- a small best-effort local asset cache so previously downloaded checkpoints do not need to be fetched again every time

The cache is intentionally bounded. Right now it keeps at most `2` checkpoints locally.

## How to repeat the whole process

### 1. Prepare a Python environment

You need an environment that has:

- `torch`
- `onnx`
- `onnxruntime`
- `onnxscript`
- `onnxconverter-common`
- `numpy`
- `pillow`

If the usual project env is unavailable on the machine, a temporary venv works fine.

Example:

```bash
python3 -m venv /tmp/worldmodel-webgpu-venv
/tmp/worldmodel-webgpu-venv/bin/python -m pip install --upgrade pip setuptools wheel
/tmp/worldmodel-webgpu-venv/bin/pip install \
  torch \
  onnx \
  onnxruntime \
  onnxscript \
  onnxconverter-common \
  numpy \
  pillow
```

That is enough for the export path.

### Environment differences: remote training vs local browser export

The checkpoint may come from a different machine than the one doing the browser export.

In the current setup:

- remote training environment
  - Linux server
  - NVIDIA GPU / CUDA used for training
  - source of the `.pt` checkpoints
- local browser export environment
  - macOS laptop
  - CPU-only export and ONNX Runtime verification
  - AWS CLI used locally for S3 upload
  - temporary export env used here:
    - `/tmp/worldmodel-webgpu-venv`

That difference is fine because:

- training requires the server GPU
- ONNX export does not require CUDA
- the checkpoint format already stores the model config needed to reconstruct `pointing_cvae`

If you prefer conda on the laptop, that also works. The important part is just having the same export packages installed locally.

### 2. Export the checkpoint

From the repo root:

```bash
/tmp/worldmodel-webgpu-venv/bin/python scripts/export_webgpu_model.py \
  --checkpoint model/checkpoints/ckpt000045_finger_xy_cvae_march_BIG_v1_pointing_cvae_best_epoch0195_2026-03-16T02-29-47Z.pt \
  --output-dir webgpu-inference/model \
  --precision fp16
```

If you want the newest matching checkpoint automatically, omit `--checkpoint`:

```bash
python scripts/export_webgpu_model.py \
  --output-dir webgpu-inference/model \
  --precision fp16
```

### 3. What the export script does

The export script:

1. loads the checkpoint using the same model/inference code as the existing server path
2. exports a full ONNX graph
3. converts it to `fp16` if requested
4. writes `manifest.json`
5. runs an ONNX Runtime parity check unless `--skip-verify` is passed

Useful flags:

- `--precision fp32`
- `--precision fp16`
- `--output-dir webgpu-inference/model`
- `--basename my_model_name`
- `--manifest-name manifest.json`
- `--opset 18`
- `--skip-verify`

### 4. Inspect the generated output

After export, expect files like:

- `webgpu-inference/model/manifest.json`
- `webgpu-inference/model/<checkpoint-stem>_webgpu_fp16.onnx`
- `webgpu-inference/model/<checkpoint-stem>_webgpu_fp16.onnx.data`

The script also prints:

- chosen checkpoint
- generated graph path
- external data path
- image size
- total model bytes
- verification error summary

### 5. Serve the static app

Do not use `file://`.

Serve the folder over HTTP:

```bash
python3 -m http.server 5174 --directory webgpu-inference
```

Then open:

```text
http://127.0.0.1:5174
```

The app was smoke-tested this way on March 13, 2026.

### 6. Run the browser app

The page will:

1. fetch the manifest
2. fetch the ONNX graph
3. fetch and mount the external data file
4. try `WebGPU`
5. fall back to `wasm` if needed

Then you can:

- click in the input stage
- drag around to change the point
- rerun the current point manually

## S3 upload workflow

The browser app now supports a simple deployment split:

- host `webgpu-inference/` on GitHub Pages
- host the ONNX assets on S3
- point the frontend at the public S3 manifest URL

### 1. Configure S3 access for public browser downloads

The browser needs to fetch the model files directly from S3.

That means S3 needs:

- a bucket policy allowing public `s3:GetObject` on the model prefix
- bucket CORS allowing `GET` and `HEAD` from your site origin
- bucket/account Block Public Access settings that do not block that public bucket policy

The helper script for this is:

```bash
export S3_BUCKET=your-bucket
export S3_PREFIX=worldmodel
export WEBGPU_S3_PREFIX='worldmodel/webgpu-inference/browser-model/your-checkpoint-name'
export WEBGPU_CORS_ALLOWED_ORIGINS='https://yourname.github.io,http://127.0.0.1:5174,http://localhost:5174'
python scripts/s3_configure_webgpu_public_access.py
```

Optional:

```bash
export WEBGPU_DISABLE_BUCKET_PUBLIC_ACCESS_BLOCK=1
```

That only changes the bucket-level public access block settings.

It does not change account-level Block Public Access.

### 2. Upload the exported browser assets

After the ONNX export exists locally:

```bash
export S3_BUCKET=your-bucket
export S3_PREFIX=worldmodel
export WEBGPU_S3_PREFIX='worldmodel/webgpu-inference/browser-model/your-checkpoint-name'
python scripts/s3_upload_webgpu_assets.py
```

By default this uploads from:

```text
webgpu-inference/model/
```

and targets this S3 prefix by default:

```text
worldmodel/webgpu-inference/model/
```

For checkpoint-specific browser assets, override it with a separate subfolder such as:

```text
worldmodel/webgpu-inference/browser-model/<checkpoint-stem>/
```

unless overridden with:

- `WEBGPU_MODEL_DIR`
- `WEBGPU_MANIFEST_NAME`
- `WEBGPU_S3_PREFIX`
- `WEBGPU_PUBLIC_ROOT_URL`
- `WEBGPU_RUNTIME_CONFIG_PATH`

### 3. What the upload script does

The upload script:

1. reads the local manifest
2. uploads the ONNX graph and external data files to S3
3. rewrites the manifest so the graph/data URLs are public S3 URLs
4. uploads that rewritten manifest to S3
5. writes `webgpu-inference/runtime-config.js` so the frontend loads that S3 manifest by default
6. prints the public manifest and asset URLs

### 4. Frontend default behavior after upload

The page now checks manifest URLs in this order:

1. `?manifest=...` query param
2. last saved manifest URL in browser storage
3. `webgpu-inference/runtime-config.js`
4. local `./model/manifest.json`

So after running the upload script and deploying the updated frontend, the page will automatically load from S3 unless the user has already picked and saved a different checkpoint in the browser.

## Hosting options

### GitHub Pages

The static entrypoint itself is compatible with GitHub Pages.

The main issue is asset size, not HTML compatibility.

At the current model size, GitHub Pages is likely a bad fit for the weights unless you are comfortable with very large static downloads and any relevant Pages/LFS/release-hosting constraints in your workflow.

### Better current option

A more practical setup right now is:

- host `webgpu-inference/` as the static app
- host the ONNX weight assets on another static host
- point the manifest at those hosted weight URLs

The browser app supports this already because the manifest can reference external URLs.

### Query-param override

The page supports overriding the manifest URL with:

```text
?manifest=https://your-host/path/to/manifest.json
```

That is the easiest way to test alternate hosted assets.

## Accuracy check

The exported ONNX model was checked against the PyTorch model.

For the current verified `pointing_cvae` `fp16` export:

- max absolute difference: about `0.001909256`
- mean absolute difference: about `0.000010511`

That is small enough for this current demo path.

## Why this was done on the MacBook

The export step does not require an NVIDIA GPU.

It was run on the MacBook CPU successfully, even though the checkpoint itself came from a different remote training environment.

That means:

- you can repeat the export locally without the server GPU
- you only need the server if you want to train a new model or export from a different checkpoint living there

## Security notes

The upload path intentionally does not use object ACLs.

That is on purpose because many modern S3 buckets use Object Ownership settings that disable ACL-based public access.

The intended public-access path here is:

- bucket policy for public `GetObject`
- bucket CORS for browser fetches
- no object ACL requirement

If the files still return `403`, the most likely reason is that bucket-level or account-level Block Public Access is still blocking the public bucket policy.

## When you probably want a different checkpoint/model

If the goal is real GitHub Pages hosting with a lightweight first-load experience, the current checkpoint is too large.

The next likely improvement is one of:

- train a smaller architecture specifically for browser inference
- reduce channel sizes or the largest skip projection layers
- export a more compressed model family designed for web delivery
- host the big weights elsewhere and keep Pages only for the frontend shell

## Files to read next

- `webgpu-inference/README.md`
- `scripts/export_webgpu_model.py`
- `webgpu-inference/index.html`
- `webgpu-inference/app.js`
