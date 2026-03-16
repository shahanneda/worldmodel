# Docs Index

This repo now has four main layers:

1. A data-generation pipeline that turns a raw finger video into segmented training data.
2. A training stack that learns a mapping from fingertip `(x, y)` coordinates to an output image.
   It now supports both the original deterministic U-Net baseline and a latent `pointing_cvae` variant with an image posterior plus coordinate prior.
3. An inference webapp that serves a trained checkpoint over a Python server.
4. A separate static browser inference app that loads an exported ONNX model and runs it with WebGPU in the browser.

Start here if you are not sure which doc to read.

| Document | Type | Purpose |
| --- | --- | --- |
| `docs/finger_workflow.md` | Workflow guide | The best high-level overview. Explains the overall goal, the two-stage workflow, the current training setup, and how the pieces fit together. |
| `docs/finger_pipeline_status.md` | Status / progress report | What has already been built, what is working now, measured results from the latest data run, and what still remains unfinished. |
| `docs/training_configs.md` | Design / migration guide | Source of truth for the YAML-based training config system, including `configs/`, `VERSION`, W&B logging, reproducibility rules, notebook behavior, and backward-compatibility strategy. |
| `docs/inference_webapp.md` | Serving guide | How to launch the server-side inference webapp, call the API, and port-forward it over SSH. |
| `docs/webgpu_inference.md` | Browser inference guide | What was added for static in-browser inference, how the ONNX export works, exact repeatable commands, and current hosting/model-size constraints. |
| `docs/setting_up_the_project.md` | Environment / setup guide | Which Python and Jupyter environment to use, whether conda is involved, and how to verify CUDA and packages. |
| `docs/helpful_commands.md` | Command reference | Common commands for data sync, segmentation/export, environment checks, loader checks, checkpoint management, and notebook-related training setup. |

## Suggested reading order

If you are new to the repo:

1. `docs/finger_workflow.md`
2. `docs/setting_up_the_project.md`
3. `docs/training_configs.md`
4. `docs/finger_pipeline_status.md`
5. `docs/inference_webapp.md`
6. `docs/webgpu_inference.md`
7. `docs/helpful_commands.md`

## Current code entry points

| File | Type | Purpose |
| --- | --- | --- |
| `data-processing/process_finger_video.py` | Pipeline script | Builds the processed dataset from the raw video plus landmarks JSON. |
| `model/load.py` | Data loader | Loads processed segmented frames plus fingertip coordinates for PyTorch. |
| `model/model.py` | Model definition | Contains the coordinate-conditioned U-Net generator. |
| `model/cvae.py` | Model definition | Contains the posterior encoder, coordinate prior, and latent-conditioned decoder wrapper. |
| `model/config.py` | Config loader | Validates YAML training configs, resolves paths, and exposes typed training settings. |
| `model/train.py` | Training code | Contains loader splitting, loss, epoch loops, and training helpers. |
| `model/checkpoints.py` | Checkpoint utility | Creates ID-prefixed timestamped checkpoint paths, lists local checkpoints, and finds the latest checkpoint by checkpoint ID. |
| `model/workspace.ipynb` | Notebook | Thin orchestration notebook that imports and runs the code in the files above. |
| `scripts/train_from_config.py` | Training CLI | Runs training directly from a YAML config in `configs/`. |
| `inference/server.py` | Inference web server | Starts the server-side coordinate-to-image demo app and serves the browser UI. |
| `scripts/export_webgpu_model.py` | Export utility | Converts a PyTorch checkpoint into ONNX plus a manifest for browser-side inference. |
| `scripts/s3_upload_webgpu_assets.py` | S3 deploy utility | Uploads the browser model artifacts to S3, rewrites the manifest to public S3 URLs, and updates the frontend runtime config. |
| `scripts/s3_configure_webgpu_public_access.py` | S3 access utility | Applies a prefix-scoped public-read bucket policy plus CORS settings for browser access to the uploaded model files. |
| `webgpu-inference/` | Static web app | Browser-side click-to-generate demo that runs ONNX Runtime Web with WebGPU or wasm. |
| `notebooks/finger_loader_example.py` | Example script | Minimal example for loading the processed dataset in a notebook-style workflow. |
| `scripts/manage_checkpoints.py` | CLI utility | Lists local checkpoints, prints the latest checkpoint, resolves a checkpoint ID to a path, or generates a new ID-prefixed checkpoint path. |
| `scripts/s3_upload_checkpoints.py` | S3 sync script | Uploads local checkpoints to S3. |
| `scripts/s3_download_checkpoints.py` | S3 sync script | Downloads checkpoints from S3. |
