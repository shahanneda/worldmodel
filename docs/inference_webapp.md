# Inference Webapp

## Purpose

This repo now includes a separate inference stack for the trained coordinate-to-image models.

The goal of the webapp is:

1. click a point in the browser
2. send that normalized `(x, y)` coordinate to the Python server
3. run model inference on the server GPU
4. return the generated image to the browser
5. for `pointing_cvae` checkpoints, inspect the coordinate-conditioned prior, latent samples, and uncertainty heatmap

This keeps all model execution on the remote machine. The browser only sends coordinates and receives PNG results.

## Code layout

The inference code lives outside `model/` on purpose:

```text
inference/
  engine.py
  server.py
  webapp/
    templates/index.html
    static/styles.css
    static/app.js
```

### File roles

- `inference/engine.py`
  - loads the checkpoint
  - chooses CPU or CUDA
  - runs inference from normalized coordinates
  - automatically applies the checkpoint's configured coordinate space
  - supports both deterministic baseline checkpoints and `pointing_cvae` checkpoints
  - converts predictions to base64 PNG responses
- `inference/server.py`
  - starts the Flask app
  - serves the browser UI
  - exposes `/api/health`
  - exposes `/api/infer`
- `inference/webapp/...`
  - browser UI for clicking and viewing generated output
  - when the checkpoint is `pointing_cvae`, shows:
    - a spotlight image
    - prior mean plus multiple sampled decodes
    - a latent chart with prior mean, per-dimension sigma, and sampled latent points
    - a decoded-image uncertainty heatmap

## Launch the server

From the repo root:

```bash
/venv/main/bin/python3 inference/server.py \
  --host 0.0.0.0 \
  --port 8000
```

By default, the server now picks the latest `*.pt` file in `model/checkpoints/`, preferring the highest checkpoint ID when the filename starts with `ckpt000123_...`.

When the server is running in this default mode, each page refresh and inference request checks again for a newer checkpoint. That means you do not need to manually restart the webapp after every new training save.

## Get the latest checkpoint path

Print the latest local checkpoint with the repo helper:

```bash
python3 scripts/manage_checkpoints.py --action latest
```

Resolve it from Python:

```bash
/venv/main/bin/python3 - <<'PY'
from model.checkpoints import latest_checkpoint

print(latest_checkpoint(root="model/checkpoints"))
PY
```

### Optional flags

- `--device cuda`
- `--device cuda:0`
- `--device cpu`
- `--checkpoint /absolute/path/to/another_checkpoint.pt`

If CUDA is available, the server defaults to CUDA automatically.
If you pass `--checkpoint`, auto-reloading is disabled and that exact checkpoint stays pinned until the server is restarted.

## API

### Health check

```bash
curl http://127.0.0.1:8000/api/health
```

### Run inference

```bash
curl \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"x_norm": 0.5, "y_norm": 0.5, "sample_count": 4, "temperature": 0.7}' \
  http://127.0.0.1:8000/api/infer
```

The response includes:

- `x_norm`
- `y_norm`
- `model_x`
- `model_y`
- `latency_ms`
- `image_base64`
- `gallery`
- `default_gallery_key`
- `supports_latent_exploration`
- `prior_distribution`
- `uncertainty_heatmap_base64`

The health payload also reports:

- `model_kind`
- `coord_space`
- `latent_dim`
- `prior_sample_temperature`

## SSH port forwarding

Because the server runs on the remote machine, forward the port from your local machine:

```bash
ssh -L 8000:127.0.0.1:8000 your-user@your-server
```

Then open this locally:

```text
http://127.0.0.1:8000
```

If port `8000` is busy locally, choose another local port:

```bash
ssh -L 9000:127.0.0.1:8000 your-user@your-server
```

Then open:

```text
http://127.0.0.1:9000
```

## Notes

- the browser does not run the PyTorch model
- the server clamps inputs into `[0, 1]`
- if the checkpoint was trained with `data.coord_space: minus_one_to_one`, the server remaps the clamped input into `[-1, 1]` before inference
- the live webapp can only inspect the prior; the posterior still requires an observed image input
- the current checkpoint produces square outputs based on the training image size
- the checkpoint loader supports both the current training checkpoint format and a plain `state_dict` save
