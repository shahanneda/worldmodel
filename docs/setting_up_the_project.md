# Setting Up The Project

## Active environment

The active Python environment for this repo is:

```text
/venv/main/bin/python3
```

This is the environment currently used for:

- segmentation pipeline runs
- PyTorch training
- Jupyter notebooks

## Are we using conda?

There is a conda installation on the machine, but the current project workflow is not using conda as the active environment.

Current answer:

- conda exists on the machine
- the repo is running from `/venv/main`
- the notebook kernel should point to `/venv/main`

## Packages currently expected

The main packages needed by the current workflow are:

- `torch`
- `torchvision`
- `cv2`
- `segment_anything`
- `ipykernel`
- `matplotlib`
- `flask`
- `wandb`

## CUDA

The current workflow expects CUDA-enabled PyTorch. The environment has already been verified with:

- `torch.cuda.is_available() == True`
- GPU visible from PyTorch

## Notebook kernel

The notebook kernel created for this repo is:

```text
Worldmodel GPU (venv main)
```

Kernel spec path:

```text
/root/.local/share/jupyter/kernels/worldmodel-gpu/kernel.json
```

If VS Code asks for an interpreter path instead of a kernel, use:

```text
/venv/main/bin/python3
```

## VS Code workspace setting

The repo includes:

```text
.vscode/settings.json
```

That points VS Code to:

```text
/venv/main/bin/python3
```

## Sanity checks

### Check Python

```bash
python3 --version
python3 -c "import sys; print(sys.executable)"
```

### Check CUDA

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Check core libraries

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

### Check W&B

```bash
python3 - <<'PY'
import wandb

print("wandb", wandb.__version__)
PY
```

## Environment variables

Training now auto-loads the repo-root `.env` file.

The relevant training variable is:

- `WANDB_API_KEY`

That is used by `scripts/train_from_config.py` and notebook-driven `run_training_from_config(...)` calls so training runs show up in Weights & Biases without manually sourcing `.env` first.

## Current project layout

### Data pipeline

- `data-processing/process_finger_video.py`
  - builds the processed finger dataset

### Model/training stack

- `model/load.py`
  - processed-data dataset and dataloader
- `model/model.py`
  - coordinate-conditioned U-Net generator
- `model/cvae.py`
  - latent posterior/prior model that reuses the U-Net decoder
- `model/train.py`
  - loss, split creation, training loops, and W&B logging hooks
- `model/checkpoints.py`
  - checkpoint utility with monotonic ID-prefixed filenames
- `model/workspace.ipynb`
  - notebook orchestration

### Inference stack

- `inference/engine.py`
  - checkpoint loading and prediction
- `inference/server.py`
  - Flask server for the click-to-generate demo
- `inference/webapp/`
  - HTML, CSS, and JS for the browser UI

### Checkpoint and sync scripts

- `scripts/manage_checkpoints.py`
  - local checkpoint listing / latest / new-path helper with checkpoint IDs
- `scripts/s3_upload_checkpoints.py`
  - upload local checkpoints to S3
- `scripts/s3_download_checkpoints.py`
  - download checkpoints from S3

### Docs

- `docs/finger_workflow.md`
- `docs/finger_pipeline_status.md`
- `docs/inference_webapp.md`
- `docs/helpful_commands.md`
- `docs/README.md`

## Practical recommendation

When working in the notebook:

1. choose `Worldmodel GPU (venv main)`
2. keep real logic in the Python files
3. use the notebook only to run and inspect experiments
4. save checkpoints into `model/checkpoints/`, which is ignored by git
