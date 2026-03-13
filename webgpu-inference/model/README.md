Exported ONNX artifacts go in this directory.

Expected files after running `scripts/export_webgpu_model.py`:

- `manifest.json`
- `*.onnx`
- `*.onnx.data`

The generated `.onnx` and `.onnx.data` files are ignored by git because they are too large for normal source control in this repo.
