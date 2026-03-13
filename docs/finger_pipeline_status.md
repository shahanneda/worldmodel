# Finger Pipeline Status

## Project status summary

The repo is now in this state:

### Data-generation pipeline

Working:
- raw video -> segmented frames
- raw video -> binary masks
- raw video -> debug frames and debug video
- raw video + landmarks JSON -> fingertip coordinate JSON

### Training stack

Working:
- processed dataset -> PyTorch loader
- downsampled training frames -> coordinate-conditioned U-Net
- train/validation split
- basic training loop
- checkpoint save/load support
- local checkpoint management utility
- S3 upload/download scripts for checkpoints
- notebook orchestration
- server-side inference webapp

Not yet done:
- generated sample saving during training
- better conditioning than a single `(x, y)` point
- image-only fingertip label generation

## Overall goal

The overall project goal is:

1. Take a raw finger video.
2. Turn it into a clean supervised dataset.
3. Train models on top of that dataset.

The current supervised pair is:

- input: fingertip coordinate `(x, y)`
- target: segmented frame

## What has been done so far

### Earlier attempts

The original segmentation attempts used:

- OpenCV background subtraction
- MediaPipe Selfie Segmentation

That path produced broken masks and was not acceptable.

### Current segmentation implementation

The current script is:

```text
data-processing/process_finger_video.py
```

The current stack is:

- `torchvision` Mask R-CNN for coarse person masks
- Meta Segment Anything for refinement
- coarse-mask fallback when SAM drifts

### Current processed output

Current output directory:

```text
data/processed-finger-sam-2026-01-30T22-41-47-949Z/
```

Main artifacts:

- `segmented_frames/`
- `mask_frames/`
- `debug_frames/`
- `segmented_video.mp4`
- `debug_video.mp4`
- `index_finger_positions.json`

### Loader and model work added

Added after the data pipeline:

- `model/load.py`
  - dataset and dataloader
  - now downsamples images to `128 x 128` by default
- `model/model.py`
  - `CoordinateToImageUNet`
- `model/train.py`
  - training split, loss, train loop, predict helper
- `model/checkpoints.py`
  - timestamped checkpoint path generation
  - checkpoint listing
  - latest-checkpoint lookup
- `model/workspace.ipynb`
  - notebook that drives the training code
- `scripts/manage_checkpoints.py`
  - local checkpoint CLI utility
- `scripts/s3_upload_checkpoints.py`
  - upload checkpoints to S3
- `scripts/s3_download_checkpoints.py`
  - download checkpoints from S3
- `inference/engine.py`
  - checkpoint loader and GPU inference wrapper
- `inference/server.py`
  - Flask app for click-to-generate inference
- `inference/webapp/`
  - separate browser UI for server-side model execution

## Measured results from the latest processed data run

For the current example clip:

- frame count processed: `300`
- missing fingertip labels in the source landmarks JSON: `8`
- frames where coarse fallback was used instead of the SAM output: `26`
- median SAM-vs-coarse IoU: about `0.909`
- mean SAM-vs-coarse IoU: about `0.871`

## Training smoke test status

The current coordinate-to-image model has already passed a basic smoke test:

- loader shapes are correct
- training runs on the current environment
- a 1-epoch train/validation pass completed successfully
- checkpoint save/load round-trip works
- server-side inference requests run against the saved checkpoint

This confirms the wiring is correct, but it does not mean the model is already good.

## Current limitations

### Data limitations

- fingertip labels still come from the landmarks JSON
- some segmentation frames still need coarse fallback
- segmentation can still have minor artifacts on fine details

### Modeling limitations

- the generator only sees `(x, y)`
- that is probably too little information to reconstruct the whole frame well
- checkpointing exists now, but best-checkpoint selection and richer experiment logging do not

## Current recommendation

Treat the current model as:

- a baseline experiment
- a test that the dataset, loader, model, notebook, and CUDA setup are all working together

## Next likely upgrades

- add best-checkpoint selection and periodic autosaving
- add generated sample saving
- add a stronger experiment structure in the notebook or scripts
- add more conditioning than a single `(x, y)` point
- move from landmark-derived fingertip labels to image-only fingertip supervision
