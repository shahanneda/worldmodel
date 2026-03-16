# Finger Workflow

## Overall goal

The project goal is now:

1. Start from a raw finger video.
2. Segment the person cleanly away from the background.
3. Track the fingertip location for each frame.
4. Save the processed result as a reproducible dataset.
5. Train a model that maps fingertip coordinates to an image, either directly or through a latent posterior/prior.
6. Save checkpoints without overwriting older runs.
7. Log training runs, validation previews, configs, and checkpoints to Weights & Biases.
8. Sync checkpoints to and from S3.

At the moment, the dataset-building part is working, the first training stack is in place, and there is now a lightweight inference webapp for serving a trained checkpoint.

## Big picture

There are now three stages.

### Stage 1: Build the dataset

Input:
- raw video
- landmarks JSON

Output:
- segmented person-only frames with white background
- binary masks
- fingertip coordinate JSON
- debug frames and debug video

Main file:

```text
data-processing/process_finger_video.py
```

### Stage 2: Train a coordinate-to-image model

Input:
- fingertip `(x, y)` coordinates
- segmented frames

Output:
- a model that tries to generate the frame from just the coordinate

Main files:

```text
model/load.py
model/model.py
model/train.py
model/config.py
model/checkpoints.py
model/workspace.ipynb
configs/
scripts/train_from_config.py
```

### Stage 3: Serve the trained checkpoint

Input:
- normalized fingertip coordinate `(x, y)`
- trained checkpoint

Output:
- generated image returned by a Python web server

Main files:

```text
inference/engine.py
inference/server.py
inference/webapp/
```

## Stage 1 in more detail

### Inputs

Current example inputs:

- `data/finger-video-2026-01-30T22-41-47-949Z.webm`
- `data/finger-landmarks-2026-01-30T22-41-47-949Z.json`

The landmarks JSON is expected to contain per-frame `multiHandLandmarks`, where landmark index `8` is the index-finger tip.

### Segmentation stack

The current person-segmentation path is:

- `torchvision` Mask R-CNN for a coarse full-person mask
- Meta Segment Anything (`sam_vit_b_01ec64.pth`) for refinement
- fallback to the coarse mask when SAM drifts too far

This replaced the older OpenCV / MediaPipe path, which was not good enough.

### Output directory

Current processed output:

```text
data/processed-finger-sam-2026-01-30T22-41-47-949Z/
```

Contents:

- `segmented_frames/`
- `mask_frames/`
- `debug_frames/`
- `segmented_video.mp4`
- `debug_video.mp4`
- `index_finger_positions.json`

### JSON meaning

`index_finger_positions.json` contains one entry per frame with:

- `frame_index`
- `timestamp_ms`
- `x_px`
- `y_px`
- `x_norm`
- `y_norm`
- `coarse_mask_score`
- `sam_score`
- `sam_vs_coarse_iou`
- `fallback_used`

## Stage 2 in more detail

### Current training task

The current training task is:

- input `x`: fingertip coordinate `(x, y)`
- target `y`: segmented RGB frame

There are now two supported model families:

- `coord_to_image_unet`
  - deterministic coordinate-conditioned generator
- `pointing_cvae`
  - posterior encoder `q(z | image)`
  - coordinate prior `p(z | coord)`
  - reused coordinate-conditioned U-Net decoder fed with `[coord, z]`

The baseline U-Net is still useful as a simple first experiment. The newer `pointing_cvae` path is the latent-space version meant to support plausible multi-modal outputs without rewriting the decoder.

### Data loader

The PyTorch loader is in:

```text
model/load.py
```

It returns:

```python
(coords, frame)
```

Where:

- `coords` is a tensor shaped `(2,)`
- `frame` is a tensor shaped `(3, H, W)`

By default:

- coordinates are normalized to `[0, 1]`
- configs can optionally remap them to `[-1, 1]` before they hit the model
- missing coordinate frames are dropped
- images are downsampled to `128 x 128`

The downsampling was added on purpose so training is lighter and faster.

### Model

The current model is in:

```text
model/model.py
```

It contains:

```python
CoordinateToImageUNet
```

This is a small coordinate-conditioned U-Net style generator:

- the `(x, y)` coordinate goes through an MLP
- the embedding is projected into a bottleneck feature map
- the embedding also creates skip feature maps at multiple resolutions
- a decoder upsamples to RGB output

The latent variant lives in:

```text
model/cvae.py
```

It adds:

- `PosteriorEncoder`
- `PriorNet`
- `PointingCVAE`

`PointingCVAE` keeps the current decoder architecture and only changes the conditioning input from `coord_dim` to `coord_dim + latent_dim`.

### Training code

The training code is in:

```text
model/train.py
```

It now provides:

- train/validation split creation
- split artifact persistence
- reconstruction loss
- KL loss and beta warmup for the latent path
- one epoch loop
- full training loop
- config-driven training entrypoint
- W&B metrics, latent diagnostics, validation preview, config artifact, and checkpoint artifact logging
- prediction helper

### Notebook

The notebook is:

```text
model/workspace.ipynb
```

The notebook is intentionally thin. It should be used to:

- load one YAML config from `configs/`
- inspect the resolved config
- import the code from the Python files
- run training
- inspect outputs

The same shared training path is responsible for W&B logging, so notebook-triggered training and command-line training land in the same experiment tracking system.

The notebook should not become the main source of truth for the training logic.
The command-line entrypoint for real runs is:

```text
scripts/train_from_config.py
```

Recommended latent smoke test:

```text
configs/pointing_cvae_smoke.yaml
```

## Checkpoints

### Current checkpoint behavior

Checkpoint files should not overwrite each other.

The repo now has:

```text
model/checkpoints.py
```

This utility provides:

- `make_checkpoint_path(...)`
  - creates an ID-prefixed timestamped checkpoint filename
- `list_checkpoints(...)`
  - lists local checkpoints
- `latest_checkpoint(...)`
  - returns the latest local checkpoint path, preferring the highest checkpoint ID

Checkpoint filenames now follow an ID-first pattern such as:

```text
model/checkpoints/ckpt000006_finger_xy_baseline_v1_coord_to_image_unet_2026-03-13T21-33-38Z.pt
```

The checkpoint ID keeps incrementing across saves, which makes it easier to refer to checkpoints quickly and still guarantees that future checkpoints do not overwrite each other.

### Existing checkpoint safety

There is already an older checkpoint file in the checkpoints folder being used by something else.

The new system does not modify that old file automatically.
Future saves should use the new ID-prefixed checkpoint paths instead.

### Notebook checkpoint flow

The training notebook now:

1. generates a fresh ID-prefixed checkpoint path
2. saves the trained model to that new path
3. can list available checkpoints
4. can load the latest checkpoint back into the model

### Git behavior

Local checkpoints are now ignored by git via:

```text
model/checkpoints/
```

in `.gitignore`.

## Checkpoint S3 sync

There are now dedicated checkpoint sync scripts:

```text
scripts/s3_upload_checkpoints.py
scripts/s3_download_checkpoints.py
```

They sync the local checkpoint directory:

```text
model/checkpoints/
```

to and from:

```text
s3://<bucket>/<prefix>checkpoints/
```

These are separate from the data sync scripts on purpose.

## Current example notebook workflow

The notebook currently does this:

1. loads one training config
2. prints a summary of the resolved config
3. visualizes a target example
4. calls the config-driven training path
5. plots train/validation/test loss
6. reloads the saved checkpoint and compares target images vs generated images

## Stage 3 in more detail

### Current inference task

The inference stack now lets you:

1. launch a Python web server on the remote machine
2. open a browser UI over SSH port forwarding
3. click a point in the UI
4. run checkpoint inference on the server GPU
5. display the generated image in the browser

### Inference implementation

The inference implementation is intentionally separate from the training package:

- `inference/engine.py`
  - checkpoint loading
  - CUDA/CPU selection
  - normalized `(x, y)` inference
- `inference/server.py`
  - Flask server
  - web UI and JSON API
- `docs/inference_webapp.md`
  - launch and SSH forwarding instructions

## Repro commands

### Build the processed dataset

```bash
python3 data-processing/process_finger_video.py \
  --video data/finger-video-2026-01-30T22-41-47-949Z.webm \
  --landmarks data/finger-landmarks-2026-01-30T22-41-47-949Z.json \
  --out data/processed-finger-sam-2026-01-30T22-41-47-949Z
```

### Loader sanity check

```bash
python3 - <<'PY'
from model.load import build_finger_dataloader

coords, frames = next(iter(build_finger_dataloader(
    "data/processed-finger-sam-2026-01-30T22-41-47-949Z",
    batch_size=4,
    shuffle=False,
)))
print(coords.shape)
print(frames.shape)
PY
```

### Launch the inference webapp

```bash
/venv/main/bin/python3 inference/server.py \
  --checkpoint model/checkpoints/coord_to_image_unet.pt \
  --host 0.0.0.0 \
  --port 8000
```

### Training smoke test

```bash
python3 scripts/train_from_config.py configs/coord_to_image_unet_smoke.yaml
```

## What has been done so far

Already completed:

- built a reproducible segmentation/export pipeline
- replaced poor OpenCV / MediaPipe segmentation with Mask R-CNN + SAM refinement
- exported segmented frames, masks, debug frames, debug video, and fingertip JSON
- set up the Jupyter kernel and CUDA-capable environment
- added a PyTorch loader for the processed data
- added image downsampling in the loader for easier training
- added the first coordinate-to-image U-Net model
- added a separate training module, YAML config loader, CLI training entrypoint, and a notebook that calls into the code
- added monotonic checkpoint IDs plus timestamped checkpoint utilities
- added checkpoint upload/download scripts for S3
- added a separate server-side inference webapp

## Current limitations

- fingertip coordinates still come from the source landmarks JSON rather than image-only fingertip detection
- the current model is only conditioned on a single `(x, y)` point, so it should be treated as a baseline experiment
- the generator is likely under-conditioned for reconstructing the full frame well
- generated sample saving during training has not been added yet

## Likely next steps

Reasonable next upgrades:

- save generated sample grids during training
- add train/val/test split persistence
- condition the generator on more than one point or on temporal context
- replace landmark-JSON fingertip labels with image-only fingertip detection
- improve segmentation further with SAM2-style video propagation if needed
