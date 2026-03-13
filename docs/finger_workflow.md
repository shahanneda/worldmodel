# Finger Workflow

## Overall goal

The project goal is now:

1. Start from a raw finger video.
2. Segment the person cleanly away from the background.
3. Track the fingertip location for each frame.
4. Save the processed result as a reproducible dataset.
5. Train a model that maps fingertip `(x, y)` coordinates to an image.
6. Save checkpoints without overwriting older runs.
7. Sync checkpoints to and from S3.

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
model/checkpoints.py
model/workspace.ipynb
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

So we are currently learning a coordinate-conditioned image generator.

This is a deliberately simple first experiment. It is not expected to solve the full problem perfectly, but it gives a clean testbed for the processed data and model wiring.

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

### Training code

The training code is in:

```text
model/train.py
```

It currently provides:

- train/validation split creation
- reconstruction loss
- one epoch loop
- full training loop
- prediction helper

### Notebook

The notebook is:

```text
model/workspace.ipynb
```

The notebook is intentionally thin. It should be used to:

- import the code from the Python files
- configure hyperparameters
- run training
- inspect outputs

The notebook should not become the main source of truth for the training logic.

## Checkpoints

### Current checkpoint behavior

Checkpoint files should not overwrite each other.

The repo now has:

```text
model/checkpoints.py
```

This utility provides:

- `make_checkpoint_path(...)`
  - creates a timestamped checkpoint filename
- `list_checkpoints(...)`
  - lists local checkpoints
- `latest_checkpoint(...)`
  - returns the newest local checkpoint path

Checkpoint filenames now follow a timestamp-based pattern such as:

```text
model/checkpoints/coord_to_image_unet_2026-03-13T19-18-51Z.pt
```

This was added specifically so future checkpoints do not overwrite each other.

### Existing checkpoint safety

There is already an older checkpoint file in the checkpoints folder being used by something else.

The new system does not modify that old file automatically.
Future saves should use timestamped checkpoint paths instead.

### Notebook checkpoint flow

The training notebook now:

1. generates a fresh timestamped checkpoint path
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

1. imports the loader, model, and training functions
2. builds train/validation loaders
3. visualizes a target example
4. trains the model
5. plots train/validation loss
6. compares target images vs generated images

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
python3 - <<'PY'
import torch
from model.model import CoordinateToImageUNet
from model.train import make_train_val_loaders, train_model

train_loader, val_loader = make_train_val_loaders(
    "data/processed-finger-sam-2026-01-30T22-41-47-949Z",
    image_size=(128, 128),
    batch_size=8,
)
model = CoordinateToImageUNet(image_size=128)
train_model(
    model,
    train_loader,
    val_loader,
    device="cuda" if torch.cuda.is_available() else "cpu",
    epochs=1,
)
PY
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
- added a separate training module and a notebook that calls into the code
- added timestamped checkpoint utilities
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
