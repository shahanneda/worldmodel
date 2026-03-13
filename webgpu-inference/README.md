# WebGPU Inference

This folder is a separate browser-side inference path for the current coordinate-to-image model.

It does not touch the existing Flask server stack under `inference/`.

## What it does

- serves from a static `index.html`
- loads an exported ONNX model plus external weight data
- runs inference in the browser with `WebGPU` when available
- falls back to `wasm` if `WebGPU` is unavailable

## Export the model

Use a Python environment with:

- `torch`
- `onnx`
- `onnxruntime`
- `onnxscript`
- `onnxconverter-common`

From the repo root:

```bash
python scripts/export_webgpu_model.py \
  --checkpoint model/checkpoints/coord_to_image_unet_2026-03-13T19-20-32Z.pt \
  --output-dir webgpu-inference/model \
  --precision fp16
```

That writes:

- `webgpu-inference/model/manifest.json`
- `webgpu-inference/model/*.onnx`
- `webgpu-inference/model/*.onnx.data`

The app loads `./model/manifest.json` by default.

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

The export works, but the current model is large:

- `fp32` export: about `763 MB`
- `fp16` export: about `400 MB`

The main reason is a very large tensor in `skip_projections.3.weight`:

- `512 MB` in `fp32`
- `256 MB` in `fp16`

So the static frontend is ready, but GitHub Pages is only realistic if you:

- host the ONNX weight files on a different static host and point the manifest there, or
- train/export a substantially smaller browser model

The page already supports a manifest URL that can point at another host.
