# Helpful Commands

## Repo

Clone:

```bash
git@github.com:shahanneda/worldmodel.git
```

## Data sync

Upload data:

```bash
set -a
source .env
set +a
./scripts/s3_upload_data.py
```

Download data:

```bash
set -a
source .env
set +a
./scripts/s3_download_data.py
```

## Build the processed finger dataset

```bash
python3 data-processing/process_finger_video.py \
  --video data/finger-video-2026-01-30T22-41-47-949Z.webm \
  --landmarks data/finger-landmarks-2026-01-30T22-41-47-949Z.json \
  --out data/processed-finger-sam-2026-01-30T22-41-47-949Z
```

By default this now runs a post-processing quality filter and marks suspicious segmentation frames in `index_finger_positions.json`.

Skip the quality filter during a fresh run:

```bash
python3 data-processing/process_finger_video.py \
  --video data/finger-video-2026-01-30T22-41-47-949Z.webm \
  --landmarks data/finger-landmarks-2026-01-30T22-41-47-949Z.json \
  --out data/processed-finger-sam-2026-01-30T22-41-47-949Z \
  --skip-quality-filter
```

Run the quality filter later on an existing processed dataset without re-segmenting:

```bash
python3 scripts/filter_processed_finger_data.py \
  data/processed-finger-sam-2026-01-30T22-41-47-949Z
```

## Derived features

Fresh segmentation runs now also compute derived per-frame features such as shirt color by default.

Run derived feature extraction later on an existing processed dataset without rerunning segmentation:

```bash
python3 scripts/extract_processed_features.py \
  data/processed-finger-sam-2026-01-30T22-41-47-949Z
```

Run it across every processed SAM dataset under `data/`:

```bash
python3 scripts/extract_processed_features.py data
```

## Environment checks

Check Python:

```bash
python3 --version
python3 -c "import sys; print(sys.executable)"
```

Check CUDA:

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Check key packages:

```bash
python3 - <<'PY'
import torch
import torchvision
import cv2
import matplotlib
from segment_anything import SamPredictor

print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("cv2", cv2.__version__)
print("matplotlib", matplotlib.__version__)
print("cuda", torch.cuda.is_available())
PY
```

Check W&B:

```bash
python3 - <<'PY'
import wandb

print("wandb", wandb.__version__)
PY
```

## Notebook kernel

Install the repo notebook kernel:

```bash
python3 -m ipykernel install --user --name worldmodel-gpu --display-name "Worldmodel GPU (venv main)"
```

List kernels:

```bash
jupyter kernelspec list
```

## Loader sanity check

```bash
python3 - <<'PY'
from model.load import build_finger_dataloader

coords, frames = next(iter(build_finger_dataloader(
    "data/processed-finger-sam-2026-01-30T22-41-47-949Z",
    batch_size=4,
    shuffle=False,
    require_finger=True,
    require_shirt=True,
    min_shirt_sample_count=1500,
)))
print(coords.shape)
print(frames.shape)
PY
```

When loading from YAML training configs, the same filters now live under `data:`:

```yaml
data:
  coord_space: minus_one_to_one
  drop_quality_flagged: true
  require_finger: true
  require_shirt: true
  min_shirt_confidence: 0.0
  min_shirt_sample_count: 1500
```

## Model sanity check

```bash
python3 - <<'PY'
from model.model import CoordinateToImageUNet
import torch

model = CoordinateToImageUNet(image_size=128)
coords = torch.rand(4, 2)
images = model(coords)
print(images.shape)
PY
```

Latent model sanity check:

```bash
python3 - <<'PY'
from model.cvae import PointingCVAE
import torch

model = PointingCVAE(image_size=64, latent_dim=16, base_channels=8, latent_channels=32)
coords = torch.rand(4, 2) * 2.0 - 1.0
images = torch.rand(4, 3, 64, 64)
out = model.forward_train(coords, images)
print(out["img_hat"].shape)
print(out["mu_post"].shape)
print(out["mu_prior"].shape)
PY
```

## Check inference server dependencies

```bash
/venv/main/bin/python3 - <<'PY'
import flask
import torch

print("flask", flask.__version__)
print("cuda", torch.cuda.is_available())
PY
```

## Training smoke test

```bash
/venv/main/bin/python3 scripts/train_from_config.py configs/pointing_cvae_smoke.yaml
```

This command auto-loads `WANDB_API_KEY` from the repo-root `.env` and logs the run to W&B using the YAML config.

## Training config dry run

```bash
/venv/main/bin/python3 scripts/train_from_config.py configs/pointing_cvae_v1.yaml --dry-run
```

## Training with config overrides

```bash
/venv/main/bin/python3 scripts/train_from_config.py configs/pointing_cvae_v1.yaml \
  --set VERSION=finger_xy_cvae_debug_v1 \
  --set model.image_size=32 \
  --set model.base_channels=8 \
  --set model.latent_dim=16 \
  --set training.epochs=1
```

## Inference smoke test

```bash
/venv/main/bin/python3 - <<'PY'
from inference.engine import CoordinateToImageInference
from model.checkpoints import latest_checkpoint

checkpoint_path = latest_checkpoint(root="model/checkpoints")
engine = CoordinateToImageInference(checkpoint_path or "model/checkpoints/coord_to_image_unet.pt")
result = engine.generate(0.5, 0.5)
print(engine.metadata.device)
print(engine.metadata.image_size)
print(engine.metadata.model_kind)
print(round(result.latency_ms, 2))
print(len(result.image_base64) > 0)
PY
```

## Launch the inference webapp

```bash
/venv/main/bin/python3 inference/server.py \
  --host 0.0.0.0 \
  --port 8000
```

## Port forward the webapp over SSH

```bash
ssh -L 8000:127.0.0.1:8000 your-user@your-server
```

## Checkpoint utilities

Generate a new ID-prefixed checkpoint path:

```bash
python3 scripts/manage_checkpoints.py --action new-path --run-name coord_to_image_unet --version finger_xy_baseline_v1
```

List local checkpoints:

```bash
python3 scripts/manage_checkpoints.py --action list
```

Print the latest local checkpoint:

```bash
python3 scripts/manage_checkpoints.py --action latest
```

Print the latest checkpoint for a specific run/version:

```bash
python3 scripts/manage_checkpoints.py --action latest --run-name coord_to_image_unet --version finger_xy_baseline_v1
```

Resolve a specific checkpoint ID to its file path:

```bash
python3 scripts/manage_checkpoints.py --action resolve-id --checkpoint-id 123
```

Checkpoint filenames now look like:

```text
model/checkpoints/ckpt000123_finger_xy_baseline_v1_coord_to_image_unet_2026-03-13T21-33-38Z.pt
```

## Checkpoint S3 sync

Upload checkpoints:

```bash
set -a
source .env
set +a
./scripts/s3_upload_checkpoints.py
```

Download checkpoints:

```bash
set -a
source .env
set +a
./scripts/s3_download_checkpoints.py
```
