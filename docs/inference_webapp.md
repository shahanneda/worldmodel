# Inference Webapp

## Purpose

This repo now includes a separate inference stack for the trained coordinate-to-image model.

The goal of the webapp is:

1. click a point in the browser
2. send that normalized `(x, y)` coordinate to the Python server
3. run model inference on the server GPU
4. return the generated image to the browser

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
  - converts predictions to base64 PNG responses
- `inference/server.py`
  - starts the Flask app
  - serves the browser UI
  - exposes `/api/health`
  - exposes `/api/infer`
- `inference/webapp/...`
  - browser UI for clicking and viewing generated output

## Launch the server

From the repo root:

```bash
/venv/main/bin/python3 inference/server.py \
  --checkpoint model/checkpoints/coord_to_image_unet.pt \
  --host 0.0.0.0 \
  --port 8000
```

### Optional flags

- `--device cuda`
- `--device cuda:0`
- `--device cpu`
- `--checkpoint /absolute/path/to/another_checkpoint.pt`

If CUDA is available, the server defaults to CUDA automatically.

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
  -d '{"x_norm": 0.5, "y_norm": 0.5}' \
  http://127.0.0.1:8000/api/infer
```

The response includes:

- `x_norm`
- `y_norm`
- `latency_ms`
- `image_base64`

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
- the current checkpoint produces square outputs based on the training image size
- the checkpoint loader supports both the current training checkpoint format and a plain `state_dict` save
