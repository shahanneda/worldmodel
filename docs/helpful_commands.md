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
)))
print(coords.shape)
print(frames.shape)
PY
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

## Inference smoke test

```bash
/venv/main/bin/python3 - <<'PY'
from inference.engine import CoordinateToImageInference

engine = CoordinateToImageInference("model/checkpoints/coord_to_image_unet.pt")
result = engine.generate(0.5, 0.5)
print(engine.metadata.device)
print(engine.metadata.image_size)
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

Generate a new timestamped checkpoint path:

```bash
python3 scripts/manage_checkpoints.py --action new-path --run-name coord_to_image_unet
```

List local checkpoints:

```bash
python3 scripts/manage_checkpoints.py --action list
```

Print the latest local checkpoint:

```bash
python3 scripts/manage_checkpoints.py --action latest
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
